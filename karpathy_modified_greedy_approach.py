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

parser = argparse.ArgumentParser()

parser.add_argument("--gamma", type=float, default=0.9)
parser.add_argument("--beta", type=float, default=0.3)
con = parser.parse_args()

def configuration():
	output_predictions = 'modified_greedy_approach_karpathy_test_predictions_gamma_{}_beta_{}.csv'.format(con.gamma, con.beta)
	output_scores = 'scores_modified_greedy_approach_karpathy_test_predictions_gamma_{}_beta_{}.csv'.format(con.gamma, con.beta)
	config ={
			 'gamma': con.gamma,
			 'beta': con.beta,
			 'output_predictions': output_predictions,
			 'output_scores' : output_scores 
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
	#Get ground truth captions validation
	with open('data/coco/karpathy_validation_captions.json') as json_file:
		captions_valid_test = json.load(json_file)

	# generate predictions if they do not exist
	if not os.path.isfile(config['output_predictions']):
		captions = []
		predictions = []
		file = open('data/coco/karpathy_valid_images.txt','r')

		to_remove_c = 0

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
				prefix_embed = model.clip_project(prefix)

			text_caption, clipscore_text = generate_based_on_clipscore(model, tokenizer, prefix, clip_model,
													   gamma = config['gamma'], beta =config['beta'], embed=prefix_embed) # change greedy approach
			print("PREDICT CAPTION: {} CLIPScore {}".format(text_caption, clipscore_text))
			to_remove_c += 1
			if to_remove_c == 10:
				break
			caption_img.append(text_caption)
			captions.append(caption_img)

		#df = pd.DataFrame(captions, columns = ['caption 1', 'caption 2', 'caption 3', 'caption 4', 'caption 5', 'prediction'])
		#print('\nWriting predictions to file "{}".'.format(config['output_predictions']))
		#df.to_csv(config['output_predictions'])

if __name__ == '__main__':
	main()