import clip
import os
import json
import pandas as pd
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import skimage.io as io
import PIL.Image
from enum import Enum
from nltk.translate.bleu_score import sentence_bleu as bleu_score
from nltk.translate.meteor_score import single_meteor_score
from nltk import word_tokenize
import nltk
from rouge_score import rouge_scorer
from train import ClipCaptionPrefix
# saving names
system_caption_file = 'system_caption_file_max_sim_clip.json'
system_predictions_df = 'karpathy_test_predictions_max_sim_clip.csv'
system_scores_df = 'scores_karpathy_test_predictions_max_sim_clip.csv'
max_sim_clip = True
## saving names when no max_sim_clip is done
#system_caption_file = 'system_caption_file.json'
#system_predictions_df = 'karpathy_test_predictions.csv'
#system_scores_df = 'scores_karpathy_test_predictions.csv'
#max_sim_clip = False

nltk.download('punkt')
nltk.download('wordnet')

N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]


D = torch.device
CPU = torch.device('cpu')


def get_device(device_id: int) -> D:
	if not torch.cuda.is_available():
		return CPU
	device_id = min(torch.cuda.device_count() - 1, device_id)
	return torch.device(f'cuda:{device_id}')


CUDA = get_device

current_directory = os.getcwd()
save_path = os.path.join(os.path.dirname(current_directory), "pretrained_models")
os.makedirs(save_path, exist_ok=True)
model_path = '/home/jagargi2/CLIP_prefix_caption/coco_train/coco_prefix_latest.pt'


def generate_beam(model, tokenizer, beam_size: int = 5, prompt=None, embed=None,
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
		meteor = 0
		for h, r in zip([candidate]*5, references):
			meteor += single_meteor_score(r, h)
		meteor = meteor/5
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

def best_n_sim_clip(text_captions, image_features, clip_model):
	best = None
	hypothesis = None
	best_sim = -1000000 
	for i, caption in enumerate(text_captions):
		tokens = clip.tokenize([caption]).to(device).long()
		text_features = clip_model.encode_text(tokens).detach()
		sim = torch.cosine_similarity(text_features, image_features).numpy()[0]
		print(sim)
		if sim > best_sim:
			best = caption
			best_sim = sim
			hypothesis = i
	return best, best_sim, hypothesis

is_gpu = True #@param {type:"boolean"}  
use_beam_search = True

device = CUDA(0) if is_gpu else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

#  Load model weights

prefix_length = 40
model = ClipCaptionPrefix(prefix_length, clip_length=40, prefix_size=512,
								  num_layers=8, mapping_type='transformer')
model.load_state_dict(torch.load(model_path, map_location=CPU)) 

model = model.eval() 
device = CUDA(0) if is_gpu else "cpu"
model = model.to(device)

#Get ground truth captions validation
with open('data/coco/karpathy_validation_captions.json') as json_file:
	captions_valid_test = json.load(json_file)

if not os.path.isfile(system_caption_file):
	captions = []
	predictions = []
	file = open('data/coco/karpathy_valid_images.txt','r')
	for test_img in file.readlines():
		file_path, number_instance = test_img.split()
		_, name_img = file_path.split('/')
		name_img = 'data/coco/val2014/'+ name_img
		caption_img = captions_valid_test[number_instance][:5]

		image = io.imread(name_img)
		pil_image = PIL.Image.fromarray(image)
		image = preprocess(pil_image).unsqueeze(0).to(device)
		with torch.no_grad():
			prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
			#prefix = prefix / prefix.norm(2, -1).item()
			prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
		if use_beam_search:
			if max_sim_clip:
				text_captions = generate_beam(model, tokenizer, embed=prefix_embed)
				text_caption, clip_sim, hypothesis = best_n_sim_clip(text_captions, prefix, clip_model)
				print("PREDICT CAPTION: %s COSINE SIMILARITY: %s HYPOTHESIS: %s " %(text_caption, clip_sim, hypothesis))
			else:
				text_caption = generate_beam(model, tokenizer, embed=prefix_embed)[0]
				print("PREDICT CAPTION: %s" %(text_caption))

		else:
			text_caption = generate2(model, tokenizer, embed=prefix_embed)
			print("PREDICT CAPTION: %s" %(text_caption))

		caption_img.append(text_caption)

		captions.append(caption_img)
		predictions.append([number_instance, caption_img])

	df = pd.DataFrame(captions, columns = ['caption 1', 'caption 2', 'caption 3', 'caption 4', 'caption 5', 'prediction'])
	df.to_csv(system_predictions_df)

	coco_res_df = pd.DataFrame(predictions, columns = ['image_id', 'caption'])
	print('\nWriting predictions to file "{}".'.format(system_caption_file))
	coco_res_df.to_json(system_caption_file, orient='records')

df_results = pd.read_csv('karpathy_test_predictions.csv')
BLEU_1, BLEU_2, BLEU_3, BLEU_4, BLEU_comb, METEOR, ROUGE_L = compute_metrics(df_results)

df_scores = pd.DataFrame({'bleu_1': [BLEU_1], 'bleu_2': [BLEU_2], 
						  'bleu_3': [BLEU_3], 'bleu_4': [BLEU_4],
						  'BLEU_comb' : [BLEU_comb], 'METEOR' : [METEOR],
						  'ROUGE_L' : [ROUGE_L] })

df_scores.to_csv(system_scores_df)

print('[INFO] Scores. Bleu 1 = {:.4} Bleu 2 = {:.4} Bleu 3 = {:.4} Bleu 4 = {:.4} Bleu_comb = {:.4} METEOR = {:.4} ROUGE_L = {:.4}'.format(BLEU_1, BLEU_2, BLEU_3, BLEU_4, BLEU_comb, METEOR, ROUGE_L))