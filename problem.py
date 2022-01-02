import numpy as np
import torch
from pymoo.core.problem import Problem
from utils import generate_beam, compute_metrics, best_n_sim_clip

class ClipGAProblem(Problem):

	def __init__(self, config):
		super().__init__(n_var=512, n_obj=1, n_constr=1, xl=0.0, xu=1.0)
		self.config = config

	def _evaluate(self, x, out, *args, **kwargs):
		print(x)
		x = torch.tensor(x.astype(int)).long().to(self.config['device'])
		x_reshaped = x.reshape(1, self.config['prefix_length'], -1)
		text_captions = generate_beam(self.config['model'], self.config['tokenizer'], beam_size=self.config['beam_size'], embed=x_reshaped)
		text_caption, clip_sim, hypothesis = best_n_sim_clip(text_captions, self.config['prefix'], self.config['clip_model'], self.config['device'], 
															 similarity = self.config['similarity_clip'])
		out["F"] = -clip_sim
		out["G"] = np.zeros((x.shape[0]))
