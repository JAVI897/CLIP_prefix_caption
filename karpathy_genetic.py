from utils import generate_beam, compute_metrics, best_n_sim_clip, clipscore_karpathy_directories
import pandas as pd
import torch
import os
import numpy as np
import argparse
from typing import Tuple, List, Union, Optional
from train import ClipCaptionPrefix
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
import clip
import skimage.io as io
import PIL.Image
import json
from pymoo.optimize import minimize
from pymoo.factory import get_algorithm, get_decision_making, get_decomposition
from pymoo.factory import get_sampling, get_crossover, get_mutation

parser = argparse.ArgumentParser()

parser.add_argument("--beam_size", type=int, default=5)
parser.add_argument("--generations", type=int, default=50)
parser.add_argument("--similarity_clip", type=str, default='cos')
con = parser.parse_args()

def configuration():
	output_predictions = 'genetic_alg_karpathy_test_predictions_generations{}_beam_size_{}_similarity_clip_{}.csv'.format( con.generations, con.beam_size, con.similarity_clip)
	config ={'beam_size': con.beam_size,
			 'generations': con.generations,
			 'similarity_clip': con.similarity_clip,
			 'output_predictions': output_predictions
			  }
	return config

def genetic_alg(prefix_embed, config):

	# Generate initial population
	initial_solutions = None

	algorithm = get_algorithm(
			    'ga',
			    pop_size=100,
			    sampling= initial_solutions,
			    crossover=get_crossover("int_sbx", prob=1.0, eta=3.0),
			    mutation=get_mutation("int_pm", prob=0.5, eta=3.0),
		
			)

	res = minimize(
			    problem,
			    algorithm,
			    ("n_gen", config['generations']),
			    save_history=False,
			    verbose=True,
			    seed = 344
			)
	print(res.X)
	return prefix_embed

def main():
	config = configuration()
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

	def get_device(device_id: int) -> D:
		if not torch.cuda.is_available():
			return CPU
		device_id = min(torch.cuda.device_count() - 1, device_id)
		return torch.device(f'cuda:{device_id}')

	CPU = torch.device('cpu')
	CUDA = get_device
	current_directory = os.getcwd()
	save_path = os.path.join(os.path.dirname(current_directory), "pretrained_models")
	os.makedirs(save_path, exist_ok=True)
	model_path = '/home/jagargi2/CLIP_prefix_caption/coco_train/coco_prefix_latest.pt'
	is_gpu = True #@param {type:"boolean"}  
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
	config['model'] = model
	config['tokenizer'] = tokenizer
	config['clip_model'] = clip_model
	config['device'] = device 
	config['prefix_length'] = prefix_length
	#Get ground truth captions validation
	with open('data/coco/karpathy_validation_captions.json') as json_file:
		captions_valid_test = json.load(json_file)

	# generate predictions if they do not exist
	if not os.path.isfile(config['output_predictions']):
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
				config['prefix'] = prefix
				#prefix = prefix / prefix.norm(2, -1).item()
				prefix_embed = model.clip_project(prefix)

			print(prefix_embed)
			print(prefix_embed.shape)
			prefix_embed_flattened = torch.flatten(prefix_embed)
			print(prefix_embed_flattened)
			print(prefix_embed_flattened.shape)

			prefix_embed_unflattened = prefix_embed_flattened.reshape(1, prefix_length, -1)
			print(prefix_embed_unflattened)
			print(prefix_embed_unflattened.shape)
			
			prefix_embed = genetic_alg(prefix_embed, config)
			prefix_embed = prefix_embed.reshape(1, prefix_length, -1)
			break
			text_captions = generate_beam(model, tokenizer, beam_size=config['beam_size'], embed=prefix_embed)
			text_caption, clip_sim, hypothesis = best_n_sim_clip(text_captions, prefix, clip_model, device, similarity = config['similarity_clip'])
			print("PREDICT CAPTION: {} COSINE SIMILARITY: {:.3} HYPOTHESIS: {} BEAM SIZE: {} ".format(text_caption, clip_sim, hypothesis, config['beam_size']))

			caption_img.append(text_caption)
			captions.append(caption_img)

		#df = pd.DataFrame(captions, columns = ['caption 1', 'caption 2', 'caption 3', 'caption 4', 'caption 5', 'prediction'])
		#print('\nWriting predictions to file "{}".'.format(config['output_predictions']))
		#df.to_csv(config['output_predictions'])

if __name__ == '__main__':
	main()