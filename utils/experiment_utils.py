import datetime
import json
import os
import random
import wandb

import numpy as np
import pandas as pd
import torch
import yaml

from pandas.io.json import json_normalize


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


def seed_all(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.cuda.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def make_dirs(dir_):
	if not os.path.exists(dir_):
		os.makedirs(dir_)


def save_wandb_logs_from_api(project_name, csv_path):
	api = wandb.Api()

	# Project is specified by <entity/project-name>
	runs = api.runs(project_name)

	summary_list, config_list, name_list = [], [], []
	for run in runs: 
		# .summary contains the output keys/values for metrics like accuracy.
		#  We call ._json_dict to omit large files 
		summary_list.append(run.summary._json_dict)

		# .config contains the hyperparameters.
		#  We remove special values that start with _.
		config_list.append(
			{k: v for k,v in run.config.items()
			if not k.startswith('_')})

		# .name is the human-readable name of the run.
		name_list.append(run.name)
		
	summary_df = json_normalize(summary_list)
	config_df = json_normalize(config_list)
	name_df = pd.DataFrame(name_list)

	runs_df = pd.concat([name_df, config_df, summary_df], axis=1)

	runs_df.to_csv(csv_path, index=False)
