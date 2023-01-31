import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

import torch
from tqdm import tqdm

from utils.training_utils import init_encoder, init_finetuned_ckpt
from utils.experiment_utils import load_yaml_to_dict 

from datasets.sensor_torch_datamodule import SensorDataModule


def parse_arguments():
    parser = argparse.ArgumentParser()

    # configs paths
    parser.add_argument('--model_config_path', required=True, help='Path to experiment yaml file')
    parser.add_argument('--model_ckpt_path', required=True, help='Path to the pre-trained encoder')
    parser.add_argument('--dataset_config_path', default='configs/dataset_configs.yaml', help='Path to datasets yaml file')
    parser.add_argument('--save_path', default='./xai_results', help='Path to datasets yaml file')

    # data and models
    parser.add_argument('--dataset', required=True, choices=['uci_har', 'mobi_act', 'pamap2', 'usc_had'], help='Dataset name')
    parser.add_argument('--model', required=True, choices=['cnn1d', 'transformer'], help='Encoder model')
    
    parser.add_argument('--batch_size', default=128)

    parser.add_argument('--no_ram', action='store_true', default=False, help='If true, dataset is not first read into RAM')
    parser.add_argument('--num-workers', default=1, type=int, help='Num workers in dataloaders')

    parser.add_argument('--partition', default='test', help='Partition used to create embeddings')

    parser.add_argument('--linear_eval', default=False, action='store_true', help='Flag for using linear evaluation protocol')

    parser.add_argument('--noise_devices', nargs='+', help='Devices that should be masked out with Gaussian nouse in occlusion experiments')

    return parser.parse_args()


def produce_embeddings_and_labels(datamodule, encoder, batch_size, partition='test'):
    encoder.eval()

    embeddings = []
    labels = []
    subjects = []

    if partition == 'test':
        dataloader = datamodule.test
    elif partition == 'train':
        dataloader = datamodule.train
    elif partition == 'val':
        dataloader = datamodule.val

    # iterate over test dataloader
    for i, data in tqdm(enumerate(dataloader)):
        # read data
        example_batch = data[0].permute(0, 2, 1).float()
        label_batch = data[1]
        subject_batch = data[2]
        # generate embeddings
        embedding_batch = encoder(example_batch)
        
        embeddings.append(embedding_batch.detach().numpy())
        labels.append(label_batch.detach().numpy())
        subjects.extend(subject_batch)

    embeddings = np.vstack(embeddings)
    labels = np.hstack(labels)
    subjects = np.array(subjects)
    return embeddings, labels, subjects


def generate_path(save_path, model_ckpt_path):
    model_name = os.path.basename(os.path.dirname(model_ckpt_path))
    res_path = os.path.join(save_path, model_name)
    os.makedirs(res_path, exist_ok=True)
    return res_path


def save_array(arr, save_path):
    np.save(save_path, arr)


def generate_and_save(args):
    # pre-process configurations
    cfg = load_yaml_to_dict(args.model_config_path)
    dataset_cfg = load_yaml_to_dict(args.dataset_config_path)[args.dataset]

    model_cfg = cfg['model'][args.model]
    model_cfg['kwargs'] = {**dataset_cfg, **model_cfg['kwargs']}

    # init encoder
    n_classes = dataset_cfg['n_classes']
    model = init_finetuned_ckpt(model_cfg, n_classes, ckpt_path=args.model_ckpt_path, le=args.linear_eval, mlp_do=~args.linear_eval)
    encoder = getattr(model, 'encoder')

    # init datamodule
    datamodule = SensorDataModule(
        dataset_cfg['train'],
        dataset_cfg['val'],
        dataset_cfg['test'],
        batch_size = args.batch_size,
        store_in_ram = ~args.no_ram,
        devices = dataset_cfg['devices'],
        get_subjects=True
    )

    # produce embeddings and get labels and subjects
    embeddings, labels, subjects = produce_embeddings_and_labels(datamodule, encoder, args.batch_size, args.partition)

    # save all input and outputs
    results_save_path = generate_path(args.save_path, args.model_ckpt_path)

    save_array(embeddings, save_path=os.path.join(results_save_path, args.partition + '_embeddings.npy'))
    save_array(labels, save_path=os.path.join(results_save_path, args.partition + '_labels.npy'))
    save_array(subjects, save_path=os.path.join(results_save_path, args.partition + '_subjects.npy'))

    return embeddings, labels, subjects


if __name__ == '__main__':
    args = parse_arguments()
    generate_and_save(args)