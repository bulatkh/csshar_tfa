import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

ACTIVITIES_DICT = {
	'dws': 0, 
	'jog': 1, 
	'sit': 2, 
	'std': 3, 
	'ups': 4, 
	'wlk': 5
	}

MOTION_SENSE_COLUMNS_TO_IGNORE = [
	'attitude.roll',
	'attitude.pitch',
	'attitude.yaw',
	'gravity.x',
	'gravity.y',
	'gravity.z'
]

class MotionSenseDataset():
	""" A class for Opportunity dataset structure inculding paths to each subject and experiment file

		Attributes:
		-----------
		root_dir : str
			Path to the root directory of the dataset (data/motion_sense/A_DeviceMotion_data/A_DeviceMotion_data/)
		datafiles_dict : dict
			Dictionary for storing paths to each user and experiment
	"""
	def __init__(self, root_dir):
		self.root_dir = root_dir
		self.datafiles_dict = self.get_datafiles_dict()

	def get_datafiles_dict(self):
		""" Get dictionary with all subjects with corresponding raw datasets
		"""
		folders = os.listdir(self.root_dir)
		all_files_paths = []
		datafiles_dict = {}
		for folder in folders:
			act, exp_id = folder.split('_')
			exp_id = int(exp_id)
			act_id = ACTIVITIES_DICT[act]
			tmp_path = os.path.join(self.root_dir, folder)
			files = os.listdir(tmp_path)
			for file_ in files:
				subj = file_.split('.')[0]
				all_files_paths.append((os.path.join(tmp_path, file_), subj, act_id, exp_id))
		for file_ in all_files_paths:
			if file_[1] not in datafiles_dict:
				datafiles_dict[file_[1]] = {}
			if file_[2] not in datafiles_dict[file_[1]]:
				datafiles_dict[file_[1]][file_[2]] = {}
			datafiles_dict[file_[1]][file_[2]][file_[3]] = file_[0]
		return datafiles_dict
	
	def get_file(self, user_name, activity, experiment):
		""" Get file for a subject
		"""
		return self.datafiles_dict[user_name][activity][experiment]

class MotionSenseInstance():
	def __init__(self, data_path, columns_to_ignore=None):
		self.data_path = data_path
		self.user_id, self.exp_id, self.label = self.parse_userexplabel()
		self.data, self.labels_col = self.read_data(columns_to_ignore)

	def parse_userexplabel(self):
		subject = self.data_path.split('/')[-1][:-4]
		act_exp = self.data_path.split('/')[-2]
		act, exp_id = act_exp.split('_')
		act_id = ACTIVITIES_DICT[act]
		exp_id = int(exp_id)
		return subject, exp_id, act_id

	def read_data(self, columns_to_ignore=None):
		data = pd.read_csv(self.data_path)
		data = data.iloc[:, 1:]
		if columns_to_ignore:
			data = data.drop(columns_to_ignore, axis=1)
		labels = pd.DataFrame([self.label] * data.shape[0])
		labels.columns = ['label']
		return data, labels

def test():
	data = "data/motion_sense/A_DeviceMotion_data/A_DeviceMotion_data/"
	test_dataset = MotionSenseDataset(data)
	print(test_dataset.datafiles_dict)
	print('-----------------------------')

	act_id = 0
	exp_id = 2
	user_name = 'sub_1'
	file_ = test_dataset.get_file(user_name, act_id, exp_id)
	print(file_)
	print('-----------------------------')

	instance = MotionSenseInstance(file_)
	print(instance.data)
	print(instance.labels_col)



if __name__ == "__main__":
	test()