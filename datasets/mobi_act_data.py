import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

SCENARIOS_TO_IGNORE = {
	'FOL',
	'FKL',
	'SDL',
	'LYI',
	'SLH',
	'SBW',
	'SLW',
	'SBE',
	'SRH',
	'BSC'
}

MOBI_ACT_LABELS_DICT = {
	'STD': 0,
	'WAL': 1,
	'JOG': 2,
	'JUM': 3,
	'STU': 4,
	'STN': 5,
	'SCH': 6,
	'SIT': 7,
	'CHU': 8,
	'CSI': 9,
	'CSO': 10,
	'LYI': -1,
	'FOL': -1,
	'FKL': -1,
	'BSC': -1,
	'SDL': -1
}

MOBI_ACT_COLUMNS_TO_IGNORE = [
	'rel_time',
	'azimuth',
	'pitch',
	'roll'
]

class MobiActDataset():
	""" A class for MobiAct dataset structure inculding paths to each subject and experiment file

		Attributes:
		-----------
		root_dir : str
			Path to the root directory of the dataset (data/mobi_act/MobiAct_Dataset_v2.0/Annotated Data/)
		datafiles_dict : dict
			Dictionary for storing paths to each user and experiment
	"""
	def __init__(self, root_dir):
		self.root_dir = root_dir
		self.datafiles_dict = self.get_datafiles_dict()
	
	def get_datafiles_dict(self):
		""" Get dictionary with all subjects with corresponding raw datasets
		"""
		scenarios = os.listdir(self.root_dir)
		datafiles_dict = {}
		for scenario in scenarios:
			tmp_path = os.path.join(self.root_dir, scenario)
			files = os.listdir(tmp_path)
			for file_ in files:
				tmp_subj = int(file_.split('_')[1])
				tmp_exp = int(file_.split('_')[2])
				if tmp_subj not in datafiles_dict:
					datafiles_dict[tmp_subj] = {}
				if tmp_exp not in datafiles_dict[tmp_subj]:
					datafiles_dict[tmp_subj][tmp_exp] = []
				datafiles_dict[tmp_subj][tmp_exp].append(os.path.join(tmp_path, file_))
		return datafiles_dict

	def get_files(self, subject, exp, scenario):
		""" Get file for a subject
		"""
		file_ = [file_ for file_ in self.datafiles_dict[subject][exp] if scenario in file_]
		if len(file_) == 1:
			file_ = file_[0]
		return file_

class MobiActInstance():
	def __init__(self, data_path, columns_to_ignore=None):
		self.data_path = data_path
		self.user_id, self.exp_id = self.parse_user_exp()
		self.data, self.labels_col = self.read_data(columns_to_ignore)
		self.labels_summary = self.form_labels_df()

	def read_data(self, columns_to_ignore=None):
		data = pd.read_csv(self.data_path)
		data = data.rename(columns={'timestamp': 'timestep'})
		labels = pd.DataFrame(data['label'])
		if columns_to_ignore:
			data = data.drop(columns_to_ignore, axis=1)
		data = data.drop(['label'], axis=1)
		labels = labels.replace({'label': MOBI_ACT_LABELS_DICT})
		return data, labels
	
	def parse_user_exp(self):
		""" Get user ID for current file from its path
		"""
		filename = os.path.split(self.data_path)[-1]
		subj = int(filename.split('_')[1])
		exp = int(filename.split('_')[2])
		return subj, exp

	def form_labels_df(self):
		""" Function for forming labels summary dataframe from the columns
		"""
		timestep_label = pd.concat([self.data, self.labels_col], axis=1).drop([col for col in self.data.columns if col not in ['timestep', 'label']], axis=1)
		
		min_timestep_label = timestep_label.groupby((timestep_label.label != timestep_label.label.shift()).cumsum()).min()
		max_timestep_label = timestep_label.groupby((timestep_label.label != timestep_label.label.shift()).cumsum()).max()

		conc = pd.concat([min_timestep_label, max_timestep_label], axis=1)
		conc.columns = ['start_timestep', 'label_to_drop', 'end_timestep', 'label']
		conc = conc.drop('label_to_drop', axis=1).reset_index(drop=True)

		return conc


def test():
	data = "data/mobi_act/MobiAct_Dataset_v2.0/Annotated Data/"
	test_dataset = MobiActDataset(data)
	print(test_dataset.datafiles_dict)
	print('-----------------------------')

	exp_id = 1
	user_name = 1
	scenario = 'SLW'
	file_ = test_dataset.get_files(user_name, exp_id, scenario)
	print(file_)
	print('-----------------------------')

	instance = MobiActInstance(file_)
	print(instance.data)
	print(instance.labels_col)
	print('-----------------------------')

	print(instance.labels_summary)


if __name__ == "__main__":
	test()