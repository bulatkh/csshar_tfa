import datetime
import json
import os
import random

import numpy as np
import torch
import yaml


def generate_experiment_id():
    """ A function for generating unique experiment id based on the current time"""
    return str(datetime.datetime.now()).replace(' ', '_').replace(':', '_').replace('.', '_')


def generate_weights_file(directory, name, experiment_id):
    """ A function returning unique tensorboard filename for an experiment
    """
    return os.path.join(directory, name + '_' + experiment_id + '.pth')


def get_device(device_name):
    """ A function for setting available device """
    return torch.device(device_name) if torch.cuda.is_available() else torch.device('cpu')


def read_json_to_dict(path):
    with open(path) as json_file: 
        res_dict = json.load(json_file)
    return res_dict


def load_yaml_to_dict(path):
    with open(path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            exit(1)


def create_results_csv(csv_path, df):
    if not os.path.exists(os.path.split(csv_path)[0]):
        os.makedirs(os.path.split(csv_path)[0])
    df.to_csv(csv_path, index=None)


def dict_to_json(dict_, json_path):
    if not os.path.exists(os.path.split(json_path)[0]):
        os.makedirs(os.path.split(json_path)[0])
    with open(json_path, 'w') as fp:
            json.dump(dict_, fp)
