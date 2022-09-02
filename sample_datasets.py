import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import resample

from datasets.mobi_act_data import (MOBI_ACT_COLUMNS_TO_IGNORE,
                                    SCENARIOS_TO_IGNORE, MobiActDataset,
                                    MobiActInstance)
from datasets.motion_sense_data import (MOTION_SENSE_COLUMNS_TO_IGNORE,
                                        MotionSenseDataset,
                                        MotionSenseInstance)
from datasets.pamap_data import PamapDataset, PamapInstance
from datasets.ucihar_data import (UCI_ACTIVITIES_TO_IGNORE,
                                  SmartphoneRawDataset, SmartphoneRawInstance)
from datasets.uschad_data import USCDataset, USCInstance

DATASETS_DICT = {
	'uci_smartphones': {
		'dataset_class': SmartphoneRawDataset, 
		'instance_class': SmartphoneRawInstance,
		'type': 'raw',
		'frequency': 50},
	'pamap2': {
		'dataset_class': PamapDataset, 
		'instance_class': PamapInstance,
		'type': 'raw',
		'frequency': 100,
		'optional_labels': [0, 9, 10, 11, 18, 19, 20]},
	'usc_had': {
		'dataset_class': USCDataset, 
		'instance_class': USCInstance,
		'type': 'predefined',
		'frequency': 100},
	'motion_sense': {
		'dataset_class': MotionSenseDataset, 
		'instance_class': MotionSenseInstance,
		'type': 'predefined',
		'frequency': 50},
	'mobi_act': {
		'dataset_class': MobiActDataset, 
		'instance_class': MobiActInstance,
		'type': 'raw',
		'frequency': 200}
}

def downsample_dataframe(df, downsample_freq, sample_dur):
	downsampling_factor = int(downsample_freq * sample_dur)
	arr = df.values
	new_values = resample(arr, downsampling_factor)
	return pd.DataFrame(new_values, columns=df.columns)

def sample_raw_dataframe(data_instance, sample_dur, overlap_pct, frequency, ignore_labels=[], downsample_freq=None, num_frames=None):
	""" Function for sampling raw signals
	
		Parameters
		----------
		data_instance : DatasetInstance 
			instance of the dataset to be sampled
		sample_dur : int
			duration of the output frame in seconds
		overlap_pct : float
			overlap percentage between frames
		frequency : int
			sensors' sampling frequency in Hz
	"""
	df_list = []
	df = data_instance.data
	df['timestep'] = df['timestep'].apply(lambda x: round(x, 4))
	if not num_frames:
		num_frames = int(frequency * sample_dur)
	for _, row in data_instance.labels_summary.iterrows():
		tmp_label = row['label']
		if int(tmp_label) not in ignore_labels:
			try:
				tmp_start = max(int(df.loc[df['timestep'] == round(row['start_timestep'], 4)].index.to_list()[0] - num_frames / 2), 0)
			except:
				print(row['start_timestep'])
				print('start_passed')
			try:
				end = min(int(df.loc[df['timestep'] == round(row['end_timestep'], 4)].index.to_list()[0] + num_frames / 2), max(df.index))
			except:
				print('end passed')
				print(row['end_timestep'])
			while end - tmp_start >= num_frames:
				tmp_end = int(tmp_start + num_frames)
				tmp_df = df.iloc[tmp_start: tmp_end]
				if downsample_freq:
					tmp_df = downsample_dataframe(tmp_df, downsample_freq, sample_dur)
				tmp_df = tmp_df.fillna(0.0)
				df_list.append((tmp_df.iloc[:, 1:], int(tmp_label)))
				tmp_start += int((1 - overlap_pct) * num_frames)
	return df_list

def sample_predefined_dataframe(data_instance, sample_dur, overlap_pct, frequency, downsample_freq=None, num_frames=None):
	""" Function for sampling samples with a single activity
	
		Parameters
		----------
		data_instance : DatasetInstance 
			instance of the dataset to be sampled
		sample_dur : int
			duration of the output frame in seconds
		overlap_pct : float
			overlap percentage between frames
		frequency : float
			initial frequency
		downsample_freq : float
			desired frequency after downsampling
	"""
	df = data_instance.data
	label = data_instance.label
	tmp_start = df.index[0]
	end = df.index[-1]
	
	df_list = []
	if not num_frames:
		num_frames = int(sample_dur * frequency)
	
	while end - tmp_start >= num_frames:
		tmp_end = tmp_start + num_frames
		tmp_df = df.iloc[tmp_start: tmp_end]
		if downsample_freq:
			tmp_df = downsample_dataframe(tmp_df, downsample_freq, sample_dur)
		tmp_df = tmp_df.fillna(0.0)
		df_list.append((tmp_df, label))
		tmp_start += int((1 - overlap_pct) * num_frames)
	
	return df_list

def parse_arguments():
	parser = argparse.ArgumentParser()

	parser.add_argument("--datasets", help="specify datasets to be sampled", required=True, nargs="+")
	parser.add_argument("--paths", help="path to the datasets", required=True, nargs="+")
	parser.add_argument("--destination", help="path to store sampled datasets", default='./sampled_data/')
	# parser.add_argument("--opportunity_locomotion", help='specify opportunity mode', action="store_true", default=False)
	parser.add_argument("--num_frames", help='number of frames in the time-window', type=int, default=None)
	parser.add_argument("--sample_dur", help='sample duration in seconds', type=float, default=1)
	parser.add_argument("--overlap_pct", help='overlap percentage between samples', type=float, default=0.5)
	parser.add_argument("--pamap_include_optional", help='include optional labels', action='store_true')
	parser.add_argument("--downsample_freq", help='frequency for downsampling', type=float, default=None)
	parser.add_argument("--pamap_columns_ignore", help='substrings of columns to ignore pamap', default=[], nargs='+')
	
	args = parser.parse_args()

	return args

def main():
	args = parse_arguments()

	if not args.pamap_include_optional:
		ignore_labels = DATASETS_DICT['pamap2']['optional_labels']
	else:
		ignore_labels = []
	
	for i, dataset in enumerate(args.datasets):
		print('Sampling {} dataset..'.format(dataset))
		tmp_dataset = DATASETS_DICT[dataset]['dataset_class'](args.paths[i])
		# if dataset == 'opportunity':
		#     if not args.opportunity_locomotion:
		#         dataset_destination_path = os.path.join(args.destination, dataset + '_gesture')
		#         mode = 'gesture'
		#     else:
		#         dataset_destination_path = os.path.join(args.destination, dataset + '_locomotion')
		#         mode = 'locomotion'
		# else:
		dataset_destination_path = os.path.join(args.destination, dataset)
		if not os.path.exists(dataset_destination_path):
			os.makedirs(dataset_destination_path)

		if DATASETS_DICT[dataset]['type'] == 'raw':
			users = set(tmp_dataset.datafiles_dict.keys())
			exps = []
			for user in users:
				exps.extend(tmp_dataset.datafiles_dict[user].keys())
			exps = set(exps)
			for user in users:
				for exp in exps:
					try:
						if dataset == 'mobi_act':
							scenarios = [scenario for scenario in os.listdir(args.paths[i]) if scenario not in SCENARIOS_TO_IGNORE]
							for scenario in scenarios:
								# print(user, exp)
								files = tmp_dataset.get_files(user, exp, scenario)
								if not files:
									continue
								tmp_instance = DATASETS_DICT[dataset]['instance_class'](files, columns_to_ignore=MOBI_ACT_COLUMNS_TO_IGNORE)
								tmp_sample_list = sample_raw_dataframe(tmp_instance, args.sample_dur, args.overlap_pct, DATASETS_DICT[dataset]['frequency'], num_frames=args.num_frames, ignore_labels=[-1], downsample_freq=args.downsample_freq)
								user_name = 'subject' + ''.join(str(user))
								for j, sample in enumerate(tmp_sample_list):
									sample[0].to_csv(os.path.join(dataset_destination_path, '{}_{}_a{}_{}.csv'.format(user_name, exp, sample[1], j)), index=None) 
						else:
							files = tmp_dataset.get_files(user, exp)
							# if dataset == 'opportunity':
							#     tmp_instance = DATASETS_DICT[dataset]['instance_class'](files, mode=mode, include_columns=OPP_COLUMNS)
							if dataset == 'pamap2':
							    tmp_instance = DATASETS_DICT[dataset]['instance_class'](files, substring_eliminate=args.pamap_columns_ignore)
							else:  
								tmp_instance = DATASETS_DICT[dataset]['instance_class'](files)
							if dataset == 'uci_smartphones':
							    ignore_labels = UCI_ACTIVITIES_TO_IGNORE
							elif dataset not in ['pamap2', 'daphnet']:
								ignore_labels = []
							tmp_sample_list = sample_raw_dataframe(tmp_instance, args.sample_dur, args.overlap_pct, DATASETS_DICT[dataset]['frequency'], ignore_labels=ignore_labels, num_frames=args.num_frames, downsample_freq=args.downsample_freq)
							user_name = 'subject' + ''.join(list(filter(str.isdigit, user)))
							for j, sample in enumerate(tmp_sample_list):
								sample[0].to_csv(os.path.join(dataset_destination_path, '{}_{}_a{}_{}.csv'.format(user_name, exp, sample[1], j)), index=None) 
					except KeyError:
						pass
		elif DATASETS_DICT[dataset]['type'] == 'predefined':
			users = set(tmp_dataset.datafiles_dict.keys())
			actions = []
			for user in users:
				actions.extend(tmp_dataset.datafiles_dict[user].keys())
			actions = set(actions)
			exps = []
			for user in users:
				for action in actions:
					exps.extend(tmp_dataset.datafiles_dict[user][action].keys())
			exps = set(exps)
			for user in users:
				for action in actions: 
					for exp in exps:
						try:
							files = tmp_dataset.get_file(user, action, exp)
							if dataset == 'motion_sense':
								tmp_instance = DATASETS_DICT[dataset]['instance_class'](files, columns_to_ignore=MOTION_SENSE_COLUMNS_TO_IGNORE)
							else:
								tmp_instance = DATASETS_DICT[dataset]['instance_class'](files)
							tmp_sample_list = sample_predefined_dataframe(tmp_instance, args.sample_dur, args.overlap_pct, DATASETS_DICT[dataset]['frequency'], num_frames=args.num_frames, downsample_freq=args.downsample_freq)
							user_name = 'subject' + ''.join(list(filter(str.isdigit, user)))
							for i, sample in enumerate(tmp_sample_list):
								if str(action)[0] != 'a':
									action = 'a' + str(action)
								sample[0].to_csv(os.path.join(dataset_destination_path, '{}_{}_{}_{}.csv'.format(user_name, exp, action, i)), index=None) 
						except KeyError:
							pass
						

if __name__ == '__main__':
	main()
