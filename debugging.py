from torchinfo import summary
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

print(model.clip_project)

summary_transformer = summary(model.clip_project, (1, 512))
print(summary_transformer)