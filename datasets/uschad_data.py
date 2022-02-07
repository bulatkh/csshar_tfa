import os

import numpy as np
import pandas as pd
from scipy.io import loadmat

FREQUENCY = 100

COLUMNS = [
    'acc_x, w/ unit g (gravity)',
    'acc_y, w/ unit g',
    'acc_z, w/ unit g',
    'gyro_x, w/ unit dps (degrees per second)',
    'gyro_y, w/ unit dps',
    'gyro_z, w/ unit dps'
]

class USCDataset():
    """ A class for Opportunity dataset structure inculding paths to each subject and experiment file

        Attributes:
        -----------
        root_dir : str
            Path to the root directory of the dataset
        datafiles_dict : dict
            Dictionary for storing paths to each user and experiment
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.datafiles_dict = self.get_datafiles_dict()

    def get_datafiles_dict(self):
        """ Get dictionary with all subjects with corresponding raw datasets
        """
        filenames = os.listdir(self.root_dir)
        subject_folders =  [file_ for file_ in filenames if 'Subject' in file_]
        subject_folders_paths = [os.path.join(self.root_dir, file_) for file_ in subject_folders]
        
        datafiles_dict = {}
        for i, folder in enumerate(subject_folders_paths):
            tmp_subject_files = sorted(os.listdir(folder))
            tmp_activities = set([file_.split('t')[0] for file_ in tmp_subject_files]) 
            datafiles_dict[subject_folders[i]] = {}
            for act in tmp_activities:
                tmp_exps = set(['t' + file_.split('.')[0].split('t')[1] for file_ in tmp_subject_files if act in file_]) 
                datafiles_dict[subject_folders[i]][act] = {}
                for exp in tmp_exps:
                    datafiles_dict[subject_folders[i]][act][exp] = [os.path.join(folder, file_) for file_ in tmp_subject_files if act in file_ and exp in file_ and len(file_.split('t')[0]) == len(act)][0]
        return datafiles_dict
    
    def get_file(self, user_name, activity, experiment):
        """ Get file for a subject
        """
        return self.datafiles_dict[user_name][activity][experiment]

class USCInstance():
    def __init__(self, data_path):
        self.data_path = data_path
        self.user_id, self.exp_id, self.label = self.parse_userexplabel()
        self.data, self.labels_col = self.read_data()

    def parse_userexplabel(self):
        subject = self.data_path.split('/')[-2]
        filename = os.path.basename(self.data_path)
        activity = int(filename.split('t')[0][1:]) - 1
        exp = 't' + filename.split('t')[1]
        return subject, exp, activity

    def read_data(self):
        mat = loadmat(self.data_path)
        data = pd.DataFrame(mat['sensor_readings'])
        data.columns = COLUMNS
        label = int(mat['activity_number'][0]) - 1 
        labels = pd.DataFrame([label] * data.shape[0])
        labels.columns = ['label']
        return data, labels

def nested_dict_len(nested_dict):
    c = 0
    for key in nested_dict:
        if type(nested_dict[key]) is dict:
            c += nested_dict_len(nested_dict[key])
        else: 
            c += 1
    return c

def test():
    data = "data/usc_had/USC-HAD"
    test_dataset = USCDataset(data)
    print(test_dataset.datafiles_dict)
    print(nested_dict_len(test_dataset.datafiles_dict))

    data_path = test_dataset.datafiles_dict['Subject1']['a3']['t3']
    print(data_path)
    test_instance = USCInstance(data_path)
    print(test_instance.data)
    print(test_instance.labels_col)
    print(test_instance.data.index[-1])

if __name__ == '__main__':
    test()
