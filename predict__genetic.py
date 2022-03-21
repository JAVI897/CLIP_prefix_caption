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
from problem import ClipGAProblem
from pymoo.interface import crossover
from PIL import Image
import requests
from io import BytesIO

parser = argparse.ArgumentParser()

parser.add_argument("--beam_size", type=int, default=5)
parser.add_argument("--generations", type=int, default=50)
parser.add_argument("--similarity_clip", type=str, default='cos')
parser.add_argument("--url", type=str)
con = parser.parse_args()

def configuration():
	config ={'beam_size': con.beam_size,
			 'generations': con.generations,
			 'similarity_clip': con.similarity_clip,
			 'url' : con.url
			  }
	return config

def genetic_alg(prefix_embed, config):
	population_size = 20
	# Generate initial population
	initial_solutions = [ prefix_embed + torch.randn(prefix_embed.shape[0]).to(config['device'], dtype=torch.float32) for i in range(population_size)]
	initial_solutions + [ prefix_embed ]
	initial_solutions = torch.stack(initial_solutions, 0).cpu().numpy()
	
	algorithm = get_algorithm(
			    'ga',
			    pop_size=population_size,
			    sampling= initial_solutions,
			    crossover= get_crossover("real_k_point", n_points=5),
			    mutation=get_mutation("real_pm", prob=0.99, eta=0.5),
		
			)

	res = minimize(
			    ClipGAProblem(config),
			    algorithm,
			    ("n_gen", config['generations']),
			    save_history=False,
			    verbose=True,
			    seed = 344
			)
	
	return res.X

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

	# load image from url

	response = requests.get(config['url'])
	pil_image = Image.open(BytesIO(response.content))
	image = preprocess(pil_image).unsqueeze(0).to(device)
	with torch.no_grad():
		prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
		config['prefix'] = prefix
		#prefix = prefix / prefix.norm(2, -1).item()
		prefix_embed = model.clip_project(prefix)

	prefix_embed_flattened = torch.flatten(prefix_embed)
	prefix_embed = genetic_alg(prefix_embed_flattened, config)

	prefix_embed = torch.from_numpy(prefix_embed).to(device, dtype=torch.float32)
	prefix_embed = prefix_embed.reshape(1, prefix_length, -1)
	text_captions = generate_beam(model, tokenizer, beam_size=config['beam_size'], embed=prefix_embed)
	text_caption, clip_sim, hypothesis = best_n_sim_clip(text_captions, prefix, clip_model, device, similarity = config['similarity_clip'])
	print("PREDICT CAPTION: {} COSINE SIMILARITY: {:.3} HYPOTHESIS: {} BEAM SIZE: {} ".format(text_caption, clip_sim, hypothesis, config['beam_size']))
			

if __name__ == '__main__':
	main()