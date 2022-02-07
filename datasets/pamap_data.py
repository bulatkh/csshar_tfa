import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class PamapDataset():
    """ A class for Pamap2 dataset structure inculding paths to each subject and experiment file

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
        filenames_protocol = os.listdir(self.root_dir + '/Protocol')
        filenames_optional = os.listdir(self.root_dir + '/Optional')
        file_paths_protocol = [os.path.join(self.root_dir + '/Protocol', file) for file in filenames_protocol]
        file_paths_optional = [os.path.join(self.root_dir + '/Optional', file) for file in filenames_optional]
        subjects = sorted(set([file.split('.')[0] for file in filenames_protocol]))
        res_dict = {}
        for subject in subjects:
            res_dict[subject] = {}
            res_dict[subject]['exp1'] = [file_path for file_path in file_paths_protocol if subject in file_path][0]
            for file_path in file_paths_optional:
                if subject in file_path:
                    res_dict[subject]['exp2'] = file_path
        return res_dict

    def get_files(self, subject, exp):
        """ Get file for a subject
        """
        return self.datafiles_dict[subject][exp]

class PamapInstance():
    def __init__(self, data_path, substring_eliminate=['orient', 'temperature', 'acc2']):
        self.data_path = data_path
        self.user_id, self.exp_id = self.parse_user_exp()
        self.data, self.labels_col = self.read_data(substring_eliminate)
        self.labels_summary = self.form_labels_df()

    def read_data(self, substring_eliminate):
        """ Read a single PAMAP2 instance data and eliminate redundnant features
            Returns two dataframes: features and labels

            Attributes:
                substring_eliminate (list): a list of substrings of columns names to be eliminated
        """
        # Define all feature names
        imu_cols = ['temperature', 
                    'acc1_X', 'acc1_Y', 'acc1_Z', 
                    'acc2_X', 'acc2_Y', 'acc2_Z',
                    'gyro_X', 'gyro_Y', 'gyro_Z',
                    'mg_X', 'mg_Y', 'mg_Z',
                    'orient1', 'orient2', 'orient3', 'orient4']
        locations = ['hand', 'body', 'ankle']
        df_cols = ['timestep', 'label', 'heartrate']
        imu_cols_full = [location + '_' + imu_col for location in locations for imu_col in imu_cols]
        df_cols.extend(imu_cols_full)
    	
        # Read datasets
        df = pd.read_csv(self.data_path, sep=' ', header=None)
        df.columns = df_cols

        # Eliminate redundant columns
        eliminate_cols = []
        for substring in substring_eliminate:
            eliminate_cols.extend([col for col in df_cols if substring in col])
        data = df.drop(eliminate_cols, axis=1).drop('label', axis=1)
        labels = pd.DataFrame(df['label'])

        # Split to data and labels
        return data, labels
    
    def parse_user_exp(self):
        """ Get user ID for current file from its path
        """
        if 'Optional' in self.data_path:
            exp = 'exp2'
        else:
            exp = 'exp1'
        return os.path.basename(self.data_path).split('.')[0], exp

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
