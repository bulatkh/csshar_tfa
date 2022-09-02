import argparse
import itertools
import os
import numpy as np

from models.mlp import LinearClassifierProbing
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import LightningDataModule

from utils.experiment_utils import load_yaml_to_dict, generate_experiment_id
from utils.training_utils import setup_loggers, setup_confusion_matrix_logger, setup_classifier_metrics_logger, setup_model_checkpoint_callback_last


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    # configs paths
    parser.add_argument('--data_path', required=True, help='Path to data: features, labels and subjects')
    parser.add_argument('--dataset_config_path', default='configs/dataset_configs.yaml', help='Path to datasets yaml file')

    # data and models
    parser.add_argument('--dataset', required=True, choices=['uci_har', 'mobi_act', 'usc_had'], help='Dataset name')
    parser.add_argument('--framework', default='simclr' , help='SSL framework')
    parser.add_argument('--model', required=True, choices=['cnn1d', 'transformer'], help='Encoder model')

    parser.add_argument('--probing_tasks', required=True,  nargs='+')
    parser.add_argument('--seed', default=28)

    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    
    return parser.parse_args()


class ProbingDataset(Dataset):
    def __init__(self, features, probing_labels):
        super().__init__()
        self.features = features
        self.probing_labels = probing_labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.probing_labels[idx]


class ProbingDatamodule(LightningDataModule):
    def __init__(self, train_features, test_features, train_labels, test_labels, batch_size=128):
        super().__init__()
        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.batch_size = batch_size
        self._init_dataloaders()

    def _init_dataloaders(self):
        train_dataset = self._create_train_dataset()
        test_dataset = self._create_test_dataset()
        # val_dataset = self._create_val_dataset()

        self.train = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False, pin_memory=True)
        self.test = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, pin_memory=True)
        # self.val = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers, pin_memory=True)

    def _create_train_dataset(self):
        return ProbingDataset(self.train_features, self.train_labels)

    def _create_test_dataset(self):
        return ProbingDataset(self.test_features, self.test_labels)

    def train_dataloader(self):
        return self.train

    # def val_dataloader(self):
    #     return self.val

    def test_dataloader(self):
        return self.test


def subject_het_split(features, subjects, test_size=0.2, rs=28):
    return train_test_split(features, subjects, test_size=test_size, random_state=rs)


def activity_type_split(dataset_cfg, features, labels):
    act_to_type = dataset_cfg['act_to_type']
    type_encoding = dataset_cfg['type_encoding']
    activities = dataset_cfg['class_names']
    train = dataset_cfg['probing_train_type']
    test = dataset_cfg['probing_test_type']

    train_features = []
    test_features = []
    train_types = []
    test_types = []
    for feat, l in zip(features, labels):
        activity = activities[l]
        if activity in train:
            train_features.append(feat)
            train_types.append(type_encoding[act_to_type[activity]])
        elif activity in test:
            test_features.append(feat)
            test_types.append(type_encoding[act_to_type[activity]])
    train_features = np.array(train_features)
    test_features = np.array(test_features)
    train_types = np.array(train_types)
    test_types = np.array(test_types)
    return train_features, test_features, train_types, test_types

    
def init_probing_datamodule(features, labels, subjects, dataset_cfg, probing_type):
    if probing_type == 'subject_heterogeneity':
        unique_subjects = np.unique(subjects)
        subject_encoding = dict(zip(unique_subjects, np.arange(len(unique_subjects))))
        encoded_subjects = np.array([subject_encoding[s] for s in subjects])
        # encoded_subjects = np.random.randint(0, len(unique_subjects), len(subjects))
        train_feat, test_features, train_labels, test_labels = subject_het_split(features, encoded_subjects)
    elif probing_type == 'activity_type':
        train_feat, test_features, train_labels, test_labels = activity_type_split(dataset_cfg, features, labels)
    probing_dm = ProbingDatamodule(train_feat, test_features, train_labels, test_labels)
    return probing_dm


def init_probing_loggers(args, experiment_id, approach, probing_task):
    experiment_info = { 
        "dataset": args.dataset,
        "model": args.model,
        "probing_type": probing_task,
        "seed": args.seed,
        "ssl_framework": args.framework,
        "num_epochs_ssl": args.num_epochs,
        "model_name": args.model,
        "model_experiment": os.path.dirname(args.data_path)
    }

    loggers_list, loggers_dict = setup_loggers(tb_dir="tb_logs", experiment_info=experiment_info, dataset=args.dataset, 
        experiment_id=experiment_id, experiment_config_path=None, approach=approach)
    return loggers_list, loggers_dict


def load_and_merge(train, val, test):
    train_embeddings = np.load(train['embeddings'])
    train_embeddings = train_embeddings.reshape((train_embeddings.shape[0], -1))
    val_embeddings = np.load(val['embeddings'])
    val_embeddings = val_embeddings.reshape((val_embeddings.shape[0], -1))
    test_embeddings = np.load(test['embeddings'])
    test_embeddings = test_embeddings.reshape((test_embeddings.shape[0], -1))
    
    train_labels = np.load(train['labels'])
    val_labels = np.load(val['labels'])
    test_labels = np.load(test['labels'])
    
    train_subjects = np.load(train['subjects'])
    val_subjects = np.load(val['subjects'])
    test_subjects = np.load(test['subjects'])
    
    merged_embeddings = np.concatenate([train_embeddings, val_embeddings, test_embeddings])
    merged_labels = np.concatenate([train_labels, val_labels, test_labels])
    merged_subjects = np.concatenate([train_subjects, val_subjects, test_subjects])
    return merged_embeddings, merged_labels, merged_subjects


def get_data(args):
    train = {
        'embeddings': os.path.join(args.data_path, 'train_embeddings.npy'),
        'labels': os.path.join(args.data_path, 'train_labels.npy'),
        'subjects': os.path.join(args.data_path,'train_subjects.npy')
    }
    val = {
        'embeddings': os.path.join(args.data_path,'val_embeddings.npy'),
        'labels': os.path.join(args.data_path,'val_labels.npy'),
        'subjects': os.path.join(args.data_path,'val_subjects.npy')
    }
    test = {
        'embeddings': os.path.join(args.data_path,'test_embeddings.npy'),
        'labels': os.path.join(args.data_path,'test_labels.npy'),
        'subjects': os.path.join(args.data_path,'test_subjects.npy')
    }

    features, labels, subjects = load_and_merge(train, val, test)

    return features, labels, subjects


def run_probing_task(args, dataset_cfg, probing_task, features, labels, subjects, datamodule=None):
    experiment_id = generate_experiment_id()
    seed_everything(args.seed)

    approach = 'probing'

    loggers_list, loggers_dict = init_probing_loggers(args, experiment_id, approach, probing_task)

    if probing_task == 'subject_heterogeneity':
        num_classes = len(np.unique(subjects))
        class_names = np.unique(subjects)
    elif probing_task == 'activity_type':
        num_classes = len(np.unique(labels))
        class_names = list(dataset_cfg['type_encoding'].keys())

    if datamodule is None:
        datamodule = init_probing_datamodule(features, labels, subjects, dataset_cfg, probing_task)
    
    probing_model = LinearClassifierProbing(features[0].shape[0], num_classes, args.lr)

    callbacks = [
        # setup_confusion_matrix_logger(class_names), 
        setup_classifier_metrics_logger(num_classes),
        setup_model_checkpoint_callback_last('./probing_model_weights', args.dataset, args.model, experiment_id)
        ]
    
    trainer = Trainer.from_argparse_args(args=args, logger=loggers_list, gpus=1, deterministic=True, max_epochs=args.num_epochs, callbacks=callbacks, default_root_dir='logs')
    # train the model
    trainer.fit(probing_model, datamodule)
    trainer.test(probing_model, datamodule)
    
    if 'wandb' in loggers_dict:
        loggers_dict['wandb'].experiment.finish()


def subject_heterogeneity_probing(args, dataset_cfg, features, labels, subjects, num_trials=5):
    unique_subjects = np.unique(subjects)
    subject_encoding = dict(zip(unique_subjects, np.arange(len(unique_subjects))))
    encoded_subjects = np.array([subject_encoding[s] for s in subjects])
    kf = KFold(n_splits=num_trials, shuffle=True)
    for train, test in kf.split(features):
        probing_dm = ProbingDatamodule(features[train], features[test], encoded_subjects[train], encoded_subjects[test])
        run_probing_task(args, dataset_cfg, 'subject_heterogeneity', features, labels, subjects, probing_dm)


def extract_test_activities(act_to_type):
    type_to_act = {}
    for act, type_ in act_to_type.items():
        if type_ not in type_to_act:
            type_to_act[type_] = [act]
        else:
            type_to_act[type_].append(act)
    grouped_activities = list(type_to_act.values())
    return list(itertools.product(*grouped_activities))


def train_test_split_activity_type(dataset_cfg, features, labels):
    train_test_idx_splits = []
    activity_dict = dict((j, i) for i, j in enumerate(dataset_cfg['class_names']))
    test_activity_splits = extract_test_activities(dataset_cfg['act_to_type'])
    for split in test_activity_splits:
        test_label_split = [activity_dict[act] for act in split]
        test_idx = np.where(np.isin(labels, test_label_split))[0]
        train_idx = np.where(~np.isin(labels, test_label_split))[0]
        train_test_idx_splits.append((train_idx, test_idx))
        assert(len(labels == len(train_idx) + len(test_idx)))
    return train_test_idx_splits


def activity_type_probing(args, dataset_cfg, features, labels, subjects):
    train_test_idx_splits = train_test_split_activity_type(dataset_cfg, features, labels)
    activities = [dataset_cfg['class_names'][l] for l in labels]
    type_labels = np.array([dataset_cfg['type_encoding'][dataset_cfg['act_to_type'][a]] for a in activities])
    for train, test in train_test_idx_splits:
        probing_dm = ProbingDatamodule(features[train], features[test], type_labels[train], type_labels[test])
        run_probing_task(args, dataset_cfg, 'activity_type', features, type_labels, subjects, probing_dm)


if __name__ == '__main__':
    args = parse_arguments()

    dataset_cfg = load_yaml_to_dict(args.dataset_config_path)[args.dataset]

    features, labels, subjects = get_data(args)

    for probing_task in args.probing_tasks:
        if probing_task == 'subject_heterogeneity':
            subject_heterogeneity_probing(args, dataset_cfg, features, labels, subjects)
            # run_probing_task(args, dataset_cfg, probing_task, features, labels, subjects)
        elif probing_task == 'activity_type':
            activity_type_probing(args,dataset_cfg, features, labels, subjects)
            # run_probing_task(args, dataset_cfg, probing_task, features, labels, subjects)