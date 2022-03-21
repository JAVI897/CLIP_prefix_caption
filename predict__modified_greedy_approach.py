from utils import generate_based_on_clipscore, compute_metrics, best_n_sim_clip, clipscore_karpathy_directories
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
from PIL import Image
import requests
from io import BytesIO

parser = argparse.ArgumentParser()

parser.add_argument("--gamma", type=int, default=10)
parser.add_argument("--beta", type=float, default=0.3)
parser.add_argument("--url", type=str)
con = parser.parse_args()

def configuration():
	config ={
			 'gamma': con.gamma,
			 'beta':  con.beta
			 'url':   con.url
			 }
	return config

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
		#prefix = prefix / prefix.norm(2, -1).item()
		prefix_embed = model.clip_project(prefix)

	text_caption = generate_based_on_clipscore(model, tokenizer, prefix, clip_model,
													   gamma = config['gamma'], beta =config['beta'], embed=prefix_embed) # change greedy approach
	print("PREDICT CAPTION: {}".format(text_caption))

if __name__ == '__main__':
	main()