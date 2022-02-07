import numpy as np
import os
import pandas as pd

UCI_ACTIVITIES_TO_IGNORE = [7, 8, 9, 10, 11, 12]

class SmartphoneRawDataset():
    """ A class for uci smartphones dataset structure inculding paths to each subject and experiment file

        Attributes:
        -----------
        root_dir : str
            Path to the root directory of the dataset
        datafiles_dict : dict
            Dictionary for storing paths to each user and experiment
    """
    def __init__(self, root_dir, devices=['acc', 'gyro']):
        self.root_dir = root_dir
        self.devices = devices
        self.datafiles_dict = self.get_datafiles_dict()
    
    def get_datafiles_dict(self):
        """ Group file paths by user and experiment numbers
        """
        files = [file for file in os.listdir(self.root_dir) if 'labels' not in file]
        users = set([file.split('_')[-1][:-4] for file in files])
        user_exp_dict = {}
        for user in users:
            tmp_user_files = [file for file in files if user in file]
            exps = set([file.split('_')[-2] for file in tmp_user_files])
            user_exp_dict[user] = {}
            for exp in exps:
                tmp_user_exp_files = [file for file in tmp_user_files if exp in file]
                user_exp_dict[user][exp] = [os.path.join(self.root_dir, filename) for filename in tmp_user_exp_files]
        return user_exp_dict

    def get_files(self, user_name, experiment):
        """ Query data sample by username, experiment and device
        """
        return self.datafiles_dict[user_name][experiment]

class SmartphoneRawInstance():
    def __init__(self, data_path_list):
        self.data_path_list = data_path_list
        self.labels_path = os.path.join(data_path_list[0].replace(os.path.basename(self.data_path_list[0]), ''), 'labels.txt')
        self.user_id, self.exp_id = self.parse_userexpdev()
        self.data = self.read_sensor_data()
        self.labels_summary = self.read_labels()
        self.labels_col = self.get_labels_col()
        
    def parse_userexpdev(self):
        """ Parse user ID and experiment ID from data_path
        """
        filename = os.path.basename(self.data_path_list[0])
        
        user_id = filename.split('_')[2].split('.')[0]
        exp_id = filename.split('_')[1]
        return user_id, exp_id
        
    def read_sensor_data(self):
        """ Read files with accelerometer and gyroscope data
        """
        df = pd.DataFrame()
        for path in self.data_path_list:
            device = os.path.basename(path).split('_')[0]
            tmp_file = pd.read_csv(path, delimiter=' ', header=None, names=[device + '_X', device + '_Y', device + '_Z'])
            df = pd.concat([df, tmp_file], axis=1)
        df = self.calculate_timesteps(df)
        return df

    def read_labels(self):
        """ Read labels file for the user and experiment
        """
        labels = pd.read_csv(self.labels_path, delimiter=' ', header=None, 
                           names=['Experiment', 'User', 'Label', 'Start', 'End'])
        labels = labels[(labels['User'] == int(self.user_id[-2:])) & (labels['Experiment'] == int(self.exp_id[-2:]))]
        only_labels = labels['Label']
        labels_time = self.calculate_labels_timesteps(labels).drop(['Experiment', 'User', 'Start', 'End', 'Label'], axis=1)
        labels_time['label'] = only_labels
        return labels_time.reset_index(drop=True)
    
    def get_labels_col(self):
        """ Generate label column from label summary dataframe
        """
        res = pd.DataFrame(np.zeros(self.data.shape[0]), columns=['label'])
        for i, row in self.labels_summary.iterrows():
            res.loc[(self.data.index >= row[0]) & (self.data.index <= row[1]), 'label'] = self.labels_summary.label[i]
        return res

    @staticmethod
    def calculate_timesteps(data, frequency=50):
        """ Calculate the time steps
        """
        res = pd.DataFrame()
        num_frames = data.shape[0]
        step = 1 / frequency
        res['timestep'] = np.arange(0, num_frames * step, step)
        res = pd.concat([res, data], axis=1)
        return res
    
    @staticmethod
    def calculate_labels_timesteps(labels, frequency=50):
        res = labels.copy()
        res['start_timestep'] = res['Start'] / frequency
        res['end_timestep'] = res['End'] / frequency
        return res
       