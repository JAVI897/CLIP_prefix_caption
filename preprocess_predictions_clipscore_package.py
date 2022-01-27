import json
import pandas as pd
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--beam_size", type=int, default=5)
parser.add_argument("--maximize_clip", type=str, default='yes')
parser.add_argument("--genetic", type=str, default='no')
parser.add_argument("--generations", type=int, default=2)
parser.add_argument("--similarity_clip", type=str, default='cos')
parser.add_argument("--greedy", type=str, default = 'no')
parser.add_argument("--gamma", type=int, default = 20)
parser.add_argument("--beta", type=float, default = 0.1)

con = parser.parse_args()

def configuration():
	maximize_clip = True if con.maximize_clip == 'yes' else False
	greedy = True if con.greedy == 'yes' else False
	genetic = True if con.genetic == 'yes' else False
	if genetic:
		output_predictions = 'genetic_alg_karpathy_test_predictions_generations{}_beam_size_{}_similarity_clip_{}.csv'.format(con.generations, con.beam_size, con.similarity_clip)
	elif greedy:
		output_predictions = 'modified_greedy_approach_karpathy_test_predictions_gamma_{}_beta_{}.csv'.format(con.gamma, con.beta)
	else:
		output_predictions = 'karpathy_test_predictions_max_sim_clip_{}_beam_size_{}_similarity_clip_{}.csv'.format(maximize_clip, con.beam_size, con.similarity_clip)
	
	output_scores = 'scores_karpathy_test_predictions_max_sim_clip_{}_beam_size_{}_similarity_clip_{}.csv'.format(maximize_clip, con.beam_size, con.similarity_clip)
	
	if genetic:
		output_candidates = 'candidates_genetic_alg_karpathy_test_predictions_generations{}_beam_size_{}_similarity_clip_{}.json'.format(con.generations, con.beam_size, con.similarity_clip)
	elif greedy:
		output_candidates = 'candidates_modified_greedy_approach_karpathy_test_predictions_gamma_{}_beta_{}.csv'.format(con.gamma, con.beta)
	else:
		output_candidates = 'candidates_karpathy_test_predictions_max_sim_clip_{}_beam_size_{}_similarity_clip_{}.json'.format(maximize_clip, con.beam_size, con.similarity_clip)
	config ={'beam_size': con.beam_size,
			 'maximize_clip': maximize_clip,
			 'similarity_clip': con.similarity_clip,
			 'output_predictions': output_predictions,
			 'output_candidates':output_candidates,
			 'output_scores' : output_scores }
	return config


def main():
	config = configuration()
	df_results = pd.read_csv(config['output_predictions'])
	with open('refs.json') as json_file:
		refs = json.load(json_file)
	candidates = [[k, None] for k, _ in refs.items()]
	cont = 0
	for index, row in df_results.iterrows():
		candidates[cont][1] = row['prediction']
		candidates[cont] = tuple(candidates[cont])
		cont += 1
	candidates = dict(candidates)

	with open(config['output_candidates'], 'w') as fp:
		json.dump(candidates, fp)

if __name__ == '__main__':
	main()