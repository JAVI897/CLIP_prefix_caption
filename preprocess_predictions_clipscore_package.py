import json
import pandas as pd
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--beam_size", type=int, default=5)
parser.add_argument("--maximize_clip", type=str, default='yes')
parser.add_argument("--similarity_clip", type=str, default='cos')
con = parser.parse_args()

def configuration():
	maximize_clip = True if con.maximize_clip == 'yes' else False
	output_predictions = 'karpathy_test_predictions_max_sim_clip_{}_beam_size_{}_similarity_clip_{}.csv'.format(maximize_clip, con.beam_size, con.similarity_clip)
	output_scores = 'scores_karpathy_test_predictions_max_sim_clip_{}_beam_size_{}_similarity_clip_{}.csv'.format(maximize_clip, con.beam_size, con.similarity_clip)
	output_candidates = 'candidates_karpathy_test_predictions_max_sim_clip_{}_beam_size_{}_similarity_clip_{}.csv'.format(maximize_clip, con.beam_size, con.similarity_clip)
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
	candidates = [(k, None) for k, _ in refs.items()]
	cont = 0
	for index, row in df_results.iterrows():
		candidates[cont][1] = row['prediction']
		cont += 1
	candidates = dict(candidates)

	with open(config['output_candidates'], 'w') as fp:
		json.dump(candidates, fp)

if __name__ == '__main__':
	main()