import numpy as np
import torch
from pymoo.core.problem import ElementwiseProblem
from utils import generate_beam, compute_metrics, best_n_sim_clip

class ClipGAProblem(ElementwiseProblem):

	def __init__(self, config):
		xl = -10*np.ones(768*40)
		xu =  10*np.ones(768*40)
		super().__init__(n_var=768*40, n_obj=1, n_constr=1, xl=xl, xu=xu)
		self.config = config

	def _evaluate(self, x, out, *args, **kwargs):
		x_reshaped = x.reshape(1, self.config['prefix_length'], -1)
		text_captions = generate_beam(self.config['model'], self.config['tokenizer'], beam_size=self.config['beam_size'], embed=x_reshaped)
		text_caption, clip_sim, hypothesis = best_n_sim_clip(text_captions, self.config['prefix'], self.config['clip_model'], self.config['device'], 
															 similarity = self.config['similarity_clip'])
		out["F"] = -clip_sim
		out["G"] = np.column_stack([10 - out["F"], out["F"] - 10])
