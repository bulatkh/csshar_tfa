import os

import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset
from tqdm import tqdm


class SensorTorchDataset(Dataset):
    def __init__(self, data_path, get_subjects=False, subj_act=False, ignore_subject=None, ssl=False, transforms=None, limited=False, limited_k=1, instance_data=False, cae=False, store_in_ram=True, use_devices=None, randomly_masked_channels=0):
        super().__init__()
        self.data_path = data_path
        self.limited = limited
        self.limited_k = limited_k
        self.ignore_subject = ignore_subject
        self.use_devices = use_devices
        self.randomly_masked_channels = randomly_masked_channels
    
        # if true the whole dataset is processed to RAM for faster training
        self.store_in_ram = store_in_ram
        if store_in_ram:
            print("Reading CSV files of {}...".format(data_path))
            self.dataframes = self._read_data_to_ram()
            print("Done")

        self.len = len(self.data_files)
        self.subj_act = subj_act
        self.subj_act_to_id, self.id_to_subj_act = get_subject_activity_dictionaries(self.data_path)
        self.act_to_id, self.id_to_act = get_activity_dictionaries(self.data_path)
        self.subj_to_id, self.id_to_subj = get_subject_dictionaries(self.data_path)
        self.subjects = [self.subj_to_id[int(''.join(list(filter(str.isdigit, os.path.basename(file_).split('_')[0]))))] for file_ in self.data_files]
        self.subejcts_str = [os.path.basename(file_).split('_')[0] for file_ in self.data_files]
        self.activities = [self.act_to_id[int(os.path.basename(file_).split('_')[2][1:])] for file_ in self.data_files]
        self.get_subjects = get_subjects
        self.ssl = ssl
        self.cae = cae
        self.instance_data = instance_data
        if self.ssl or self.cae:
            if transforms:
                self.transforms = transforms
            else:
                raise AttributeError('Provide tranforms in order to use ssl approach')


    def _read_data_to_ram(self):
        dfs = []
        for i, file_ in enumerate(tqdm(self.data_files)):
            df = self._read_and_mask_df(file_)
            dfs.append(df)
        return dfs


    def _read_and_mask_df(self, file_,):
        signals = pd.read_csv(file_).fillna(0)
        for col in signals.columns:
            if col not in self.use_devices:
                signals[col] = np.random.normal(0, 1, len(signals[col]))
        if self.randomly_masked_channels > 0:
            num_channels = len(self.use_devices)
            channels_to_mask = np.random.choice(np.array(signals.columns), size=self.randomly_masked_channels)
            for col in channels_to_mask:
                signals[col] = np.random.normal(0, 1, len(signals[col]))
        return signals


    @property
    def data_files(self):
        if self.ignore_subject:
            return [os.path.join(self.data_path, filename) for filename in os.listdir(self.data_path) if self.ignore_subject not in filename]
        elif self.limited:
            files_per_activity = {}
            for filename in os.listdir(self.data_path):
                tmp_act = filename.split('_')[2]
                tmp_path = os.path.join(self.data_path, filename)
                if tmp_act in files_per_activity:
                    files_per_activity[tmp_act].append(tmp_path)
                else:
                    files_per_activity[tmp_act] = [tmp_path]
            data_files = []
            for activity in files_per_activity:
                data_files.extend(random.sample(files_per_activity[activity], self.limited_k))
            return data_files
        else:
            return [os.path.join(self.data_path, filename) for filename in os.listdir(self.data_path)]
    @property
    def labels(self):
        if self.subj_act:
            return [self.subj_act_to_id['subject' + str(self.id_to_subj[self.subjects[i]])][self.id_to_act[self.activities[i]]] for i in range(self.len)]
        else:
            return self.activities

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        # read file by index
        if self.store_in_ram:
            signals = self.dataframes[idx]
        else:
            signals = self._read_and_mask_df(self.data_files[idx])

        # get its label (activity or subject-activity)
        if self.subj_act:
            subject = self.subjects[idx]
            act = self.activities[idx]
            label = self.subj_act_to_id['subject' + str(self.id_to_subj[subject])][self.id_to_act[act]]
        else:
            label = self.labels[idx]
        
        # framework checks
        if self.ssl:
            x1 = self.transforms(np.array(signals))
            x2 = self.transforms(np.array(signals))
            if self.instance_data:
                return np.array(signals), x1, x2
            else:
                return x1, x2	
        elif self.cae:
            return self.transforms(np.array(signals)), np.array(signals)
        else:
            if self.get_subjects:
                subject = self.subejcts_str[idx]
                return np.array(signals), label, subject
            else:
                return np.array(signals), label


def get_activity_dictionaries(path):
    """ Creates dictionaries mapping label from dataset to numbered list and vice versa
    
        Parameters
        ----------
        path : str
            path to the folder with sampled files
    """
    activities = sorted(set([int(file.split('_')[2][1:]) for file in os.listdir(path)]))
    act_to_id = dict()
    id_to_act = dict()
    for i in range(len(activities)):
        act_to_id[activities[i]] = i
        id_to_act[i] = activities[i]
    return act_to_id, id_to_act   

def get_subject_dictionaries(path):
    """ Creates dictionaries mapping subjects from dataset to numbered list and vice versa
    
        Parameters
        ----------
        path : str
            path to the folder with sampled files
    """
    subjects = sorted(set([int(''.join(list(filter(str.isdigit, file.split('_')[0])))) for file in os.listdir(path)]))
    subj_to_id = dict()
    id_to_subj = dict()
    for i in range(len(subjects)):
        subj_to_id[subjects[i]] = i
        id_to_subj[i] = subjects[i]
    return subj_to_id, id_to_subj   

def get_subject_activity_dictionaries(path, exclude_subject=''):
    activities = sorted(set([int(file.split('_')[2][1:]) for file in os.listdir(path)]))
    subjects = sorted(set([file.split('_')[0] for file in os.listdir(path)]))
    if exclude_subject in subjects:
        subjects.remove(exclude_subject)
    idx = 0
    subj_act_to_id = dict()
    id_to_subj_act = dict()
    for subject in subjects:
        subj_act_to_id[subject] = dict()
        for activity in activities:
            subj_act_files = [file_ for file_ in os.listdir(path) if subject in file_ and file_.split('_')[2] == 'a' + str(activity)]
            if len(subj_act_files) > 0:
                subj_act_to_id[subject][activity] = idx
                id_to_subj_act[idx] = subject + '_a' + str(activity)
                idx += 1            
    return subj_act_to_id, id_to_subj_act

def test_limited(data_path, iterations=30):
    for _ in range(iterations):
        k = random.randint(1, 100)
        limited_train_dataset = SensorTorchDataset(data_path, limited=True, limited_k=k)
        activities = set(map(lambda x: int(x.split('_')[2][1:]) - 1, os.listdir(data_path)))
        assert sorted(list(set(limited_train_dataset.activities))) == sorted(list(activities)), 'Not all activities present'
        for activity in activities:
            assert limited_train_dataset.activities.count(activity) == k, 'Activities present in different proportions'
        print('All activities are present equally')

def main():
    data_path_uci = "./sampled_data/uci_har/uci_smartphones/train"
    test_limited(data_path_uci)
    
if __name__ == '__main__':
    main()