import clip
import os
import statistics as stats
import pandas as pd
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
import skimage.io as io
import PIL.Image
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
from enum import Enum
from nltk.translate.bleu_score import sentence_bleu as bleu_score
from nltk.translate.meteor_score import single_meteor_score
from nltk.translate import meteor_score
from nltk import word_tokenize
import nltk
from rouge_score import rouge_scorer
from train import ClipCaptionPrefix
nltk.download('punkt')
nltk.download('wordnet')

def generate_beam(model, tokenizer, beam_size: int = 20, prompt=None, embed=None,
				  entry_length=67, temperature=1., stop_token: str = '.'):

	model.eval()
	stop_token_index = tokenizer.encode(stop_token)[0]
	tokens = None
	scores = None
	device = next(model.parameters()).device
	seq_lengths = torch.ones(beam_size, device=device)
	is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
	with torch.no_grad():
		if embed is not None:
			generated = embed
		else:
			if tokens is None:
				tokens = torch.tensor(tokenizer.encode(prompt))
				tokens = tokens.unsqueeze(0).to(device)
				generated = model.gpt.transformer.wte(tokens)
		for i in range(entry_length):
			outputs = model.gpt(inputs_embeds=generated)
			logits = outputs.logits
			logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
			logits = logits.softmax(-1).log()
			if scores is None:
				scores, next_tokens = logits.topk(beam_size, -1)
				generated = generated.expand(beam_size, *generated.shape[1:])
				next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
				if tokens is None:
					tokens = next_tokens
				else:
					tokens = tokens.expand(beam_size, *tokens.shape[1:])
					tokens = torch.cat((tokens, next_tokens), dim=1)
			else:
				logits[is_stopped] = -float(np.inf)
				logits[is_stopped, 0] = 0
				scores_sum = scores[:, None] + logits
				seq_lengths[~is_stopped] += 1
				scores_sum_average = scores_sum / seq_lengths[:, None]
				scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
				next_tokens_source = next_tokens // scores_sum.shape[1]
				seq_lengths = seq_lengths[next_tokens_source]
				next_tokens = next_tokens % scores_sum.shape[1]
				next_tokens = next_tokens.unsqueeze(1)
				tokens = tokens[next_tokens_source]
				tokens = torch.cat((tokens, next_tokens), dim=1)
				generated = generated[next_tokens_source]
				scores = scores_sum_average * seq_lengths
				is_stopped = is_stopped[next_tokens_source]
			next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
			generated = torch.cat((generated, next_token_embed), dim=1)
			is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
			if is_stopped.all():
				break
	scores = scores / seq_lengths
	output_list = tokens.cpu().numpy()
	output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
	order = scores.argsort(descending=True)
	output_texts = [output_texts[i] for i in order]
	return output_texts

def generate_based_on_clipscore(
		model,
		tokenizer,
		clip_image, 
		clip_model,
		gamma = 0.9,
		beta = 0.3,
		tokens=None,
		prompt=None,
		embed=None,
		entry_length=67,  # maximum number of words
		temperature=1.,
		stop_token: str = '.',
):
	model.eval()
	stop_token_index = tokenizer.encode(stop_token)[0]
	device = next(model.parameters()).device
	generated = embed
	with torch.no_grad():

		for i in range(entry_length):

			outputs = model.gpt(inputs_embeds=generated)
			logits = outputs.logits
			logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
			logits = logits.softmax(-1)
			sorted_logits, sorted_indices = torch.sort(logits, descending=True)
			if tokens is not None:
				Z = torch.zeros(*logits.shape).to(device)
				for j in range(10):
					aux_next_token = sorted_indices[:,j].unsqueeze(0)
					aux = torch.cat((tokens, aux_next_token), dim = 1)
					aux_list = list(aux.squeeze().cpu().numpy())
					aux_text = tokenizer.decode(aux_list)

					# compute clipscore
					tokens_clip = clip.tokenize([aux_text]).to(device).long()
					clip_text = clip_model.encode_text(tokens_clip).detach()
					clipscore = 2.5*np.clip( torch.cosine_similarity(clip_text, clip_image).cpu().numpy()[0], 0, None)
					Z[ :, aux_next_token] = clipscore
					#debug
					sum_probs = gamma*Z[ :, aux_next_token] + beta*logits[ :, aux_next_token]
					print(aux_text, ' ---> beta*logits + gamma*Z: ', sum_probs[0][0][0].unsqueeze(0))
					#print(aux_text, 'CLIPScore: ', clipscore)
				print('----------------')
				logits = beta*logits + gamma*Z
			next_token = torch.argmax(logits, -1).unsqueeze(0)
			next_token_embed = model.gpt.transformer.wte(next_token)
			if tokens is None:
				tokens = next_token
			else:
				tokens = torch.cat((tokens, next_token), dim=1)
			generated = torch.cat((generated, next_token_embed), dim=1)
			if stop_token_index == next_token.item():
				break

		output_list = list(tokens.squeeze().cpu().numpy())
		output_text = tokenizer.decode(output_list)
		# compute clipscore
		tokens_clip = clip.tokenize([output_text]).to(device).long()
		clip_text = clip_model.encode_text(tokens_clip).detach()
		clipscore_text = 2.5*np.clip( torch.cosine_similarity(clip_text, clip_image).cpu().numpy()[0], 0, None)

	return output_text, clipscore_text

def generate2(
		model,
		tokenizer,
		tokens=None,
		prompt=None,
		embed=None,
		entry_count=1,
		entry_length=67,  # maximum number of words
		top_p=0.8,
		temperature=1.,
		stop_token: str = '.',
):
	model.eval()
	generated_num = 0
	generated_list = []
	stop_token_index = tokenizer.encode(stop_token)[0]
	filter_value = -float("Inf")
	device = next(model.parameters()).device

	with torch.no_grad():

		for entry_idx in trange(entry_count):
			if embed is not None:
				generated = embed
			else:
				if tokens is None:
					tokens = torch.tensor(tokenizer.encode(prompt))
					tokens = tokens.unsqueeze(0).to(device)

				generated = model.gpt.transformer.wte(tokens)

			for i in range(entry_length):

				outputs = model.gpt(inputs_embeds=generated)
				logits = outputs.logits
				logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
				sorted_logits, sorted_indices = torch.sort(logits, descending=True)
				cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
				sorted_indices_to_remove = cumulative_probs > top_p
				sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
													..., :-1
													].clone()
				sorted_indices_to_remove[..., 0] = 0

				indices_to_remove = sorted_indices[sorted_indices_to_remove]
				logits[:, indices_to_remove] = filter_value
				next_token = torch.argmax(logits, -1).unsqueeze(0)
				next_token_embed = model.gpt.transformer.wte(next_token)
				if tokens is None:
					tokens = next_token
				else:
					tokens = torch.cat((tokens, next_token), dim=1)
				generated = torch.cat((generated, next_token_embed), dim=1)
				if stop_token_index == next_token.item():
					break

			output_list = list(tokens.squeeze().cpu().numpy())
			output_text = tokenizer.decode(output_list)
			generated_list.append(output_text)

	return generated_list[0]

def clipscore_karpathy_directories(dir_images, df_results, device, clip_model, preprocess):
	'''
	Code for CLIPScore (https://arxiv.org/abs/2104.08718)
	@inproceedings{hessel2021clipscore,
	  title={{CLIPScore:} A Reference-free Evaluation Metric for Image Captioning},
	  author={Hessel, Jack and Holtzman, Ari and Forbes, Maxwell and Bras, Ronan Le and Choi, Yejin},
	  booktitle={EMNLP},
	  year={2021}
	}
	'''
	N = 0
	CLIP_SCORE = 0
	REFCLIP_SCORE = 0
	file = open(dir_images,'r')
	for ind_image, test_img in enumerate(file.readlines()):
		file_path, number_instance = test_img.split()
		_, name_img = file_path.split('/')
		name_img = 'data/coco/val2014/'+ name_img
		image = io.imread(name_img)
		pil_image = PIL.Image.fromarray(image)
		image = preprocess(pil_image).unsqueeze(0).to(device)
		df_row = df_results.iloc[ind_image]
		caption1, caption2, caption3, caption4, caption5, prediction = df_row['caption 1'], df_row['caption 2'], df_row['caption 3'], df_row['caption 4'], df_row['caption 5'], df_row['prediction']  

		# CLIP-S
		print('[INFO] Computing ClipScore for image {}'.format(name_img))
		with torch.no_grad():
			tokens = clip.tokenize([prediction]).to(device).long()
			text_features = clip_model.encode_text(tokens).detach()
			image_features = clip_model.encode_image(image).to(device, dtype=torch.float32)
		clip_score = 2.5*np.clip( torch.cosine_similarity(text_features, image_features).cpu().numpy()[0], 0, None)
		
		# RefCLIPScore
		clip_score_references = []
		for ref in [caption1, caption2, caption3, caption4, caption5]:
			tokens_ref = clip.tokenize([ref]).to(device).long()
			ref_text_feature = clip_model.encode_text(tokens_ref).detach()
			ref_score = np.clip( torch.cosine_similarity(text_features, ref_text_feature).cpu().numpy()[0], 0, None)
			clip_score_references.append(ref_score)
		max_clip_score_references = max(clip_score_references)
		refclip_score = stats.harmonic_mean([clip_score, max_clip_score_references])

		N += 1
		CLIP_SCORE += clip_score
		REFCLIP_SCORE += refclip_score

	CLIP_SCORE = CLIP_SCORE / N
	REFCLIP_SCORE = REFCLIP_SCORE / N
	return CLIP_SCORE, REFCLIP_SCORE

def compute_metrics(df_results):
	N = 0
	BLEU_1 = 0
	BLEU_2 = 0
	BLEU_3 = 0
	BLEU_4 = 0
	BLEU_comb = 0
	METEOR = 0
	ROUGE_L = 0

	for index, row in df_results.iterrows():
		caption1, caption2, caption3, caption4, caption5, prediction = row['caption 1'], row['caption 2'], row['caption 3'], row['caption 4'], row['caption 5'], row['prediction']
		references = [word_tokenize(caption1), word_tokenize(caption2), word_tokenize(caption3), 
					  word_tokenize(caption4), word_tokenize(caption5) ]

		candidate = word_tokenize(prediction)

		# BLEU
		bleu_1 = bleu_score(references, candidate, weights=(1, 0, 0, 0))
		bleu_2 = bleu_score(references, candidate, weights=(0, 1, 0, 0))
		bleu_3 = bleu_score(references, candidate, weights=(0, 0, 1, 0))
		bleu_4 = bleu_score(references, candidate, weights=(0, 0, 0, 1))
		bleu = bleu_score(references, candidate, weights=(1/4, 1/4, 1/4, 1/4))

		N += 1
		BLEU_1 += bleu_1
		BLEU_2 += bleu_2
		BLEU_3 += bleu_3
		BLEU_4 += bleu_4
		BLEU_comb += bleu

		# METEOR
		#meteor = 0
		#for h, r in zip([candidate]*5, references):
		#    meteor += single_meteor_score(r, h)
		#meteor = meteor/5
		meteor = meteor_score.meteor_score(references, candidate)
		METEOR += meteor

		# ROUGE-L
		rouge_l = 0
		for h, r in zip([prediction]*5, [caption1, caption2, caption3, caption4, caption5]):
			scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
			score = scorer.score(r, h)['rougeL']
			score = score.fmeasure
			rouge_l += score
		rouge_l = rouge_l/5
		ROUGE_L += rouge_l

	BLEU_1 = BLEU_1/N
	BLEU_2 = BLEU_2/N
	BLEU_3 = BLEU_3/N
	BLEU_4 = BLEU_4/N
	BLEU_comb = BLEU_comb/N
	METEOR = METEOR/N
	ROUGE_L = ROUGE_L/N
	return BLEU_1, BLEU_2, BLEU_3, BLEU_4, BLEU_comb, METEOR, ROUGE_L

def best_n_sim_clip(text_captions, image_features, clip_model, device, similarity = 'cos'):
	best = text_captions[0]
	hypothesis = 0
	best_sim = -1000000
	for i, caption in enumerate(text_captions):
		try:
			tokens = clip.tokenize([caption]).to(device).long()
			text_features = clip_model.encode_text(tokens).detach()
			if similarity == 'cos':
				sim = torch.cosine_similarity(text_features, image_features).cpu().numpy()[0]
			else:
				sim = torch.cosine_similarity(text_features, image_features).cpu().numpy()[0] # todo: use a different similarity metric
			if sim > best_sim:
				best = caption
				best_sim = sim
				hypothesis = i
		except: # if context length is too long it will raise an error
			pass

	return best, best_sim, hypothesis