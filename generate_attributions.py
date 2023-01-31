import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from scipy.stats import entropy
from scipy.spatial import distance

import torch

from captum.attr import GuidedBackprop, GuidedGradCam
from utils.training_utils import init_finetuned_ckpt
from utils.experiment_utils import load_yaml_to_dict 

from utils.xai_utils import *
from datasets.sensor_torch_datamodule import SensorDataModule


def parse_arguments():
    parser = argparse.ArgumentParser()

    # configs paths
    parser.add_argument('--model_config_path', help='Path to experiment yaml file')
    parser.add_argument('--model_ckpt_path', help='Path to the pre-trained encoder')
    parser.add_argument('--dataset_config_path', default='configs/dataset_configs.yaml', help='Path to datasets yaml file')
    parser.add_argument('--save_path', default='./xai_results', help='Path to folder for data/predictions/attributions to be saved')

    parser.add_argument('--draw_global', action='store_true', default=False, help='Flag for drawing global attribution scores')
    parser.add_argument('--use_saved_attributions', action='store_true', default=False, help='Flag for using saved attributions')

    # data and models
    parser.add_argument('--dataset', required=True, choices=['uci_har', 'mobi_act', 'usc_had', 'pamap2'], help='Dataset name')
    parser.add_argument('--model', required=True, choices=['cnn1d', 'transformer'], help='Encoder model')
    parser.add_argument('--xai_model', required=True, choices=['guided_gradcam', 'guided_backprop'], help='Encoder model')
    
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--linear_eval', default=False, action='store_true', help='Flag for using linear evaluation protocol')

    parser.add_argument('--no_ram', action='store_true', default=False, help='If true, dataset is not first read into RAM')
    parser.add_argument('--num-workers', default=1, type=int, help='Num workers in dataloaders')

    parser.add_argument('--noise_devices', nargs='+', help='Devices that should be masked out with Gaussian nouse in occlusion experiments')

    return parser.parse_args()


def initialize_xai_model(xai_model_name, model, layer):
    if xai_model_name == 'guided_gradcam':
        xai_model = GuidedGradCam(model, layer)
    elif xai_model_name == 'guided_backprop':
        xai_model = GuidedBackprop(model)
    return xai_model


def pick_last_cnn_layer(model, model_name):
    if model_name == 'transformer':
        grad_cnn_layer = model.encoder.cnn.conv3
    elif model_name == 'cnn1d':
        grad_cnn_layer = model.encoder.conv3
    return grad_cnn_layer


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

    n_classes = dataset_cfg['n_classes']
    devices = dataset_cfg['devices']
    class_names = dataset_cfg['class_names']

    model_cfg = cfg['model'][args.model]
    model_cfg['kwargs'] = {**dataset_cfg, **model_cfg['kwargs']}

    # init finetuned model
    model = init_finetuned_ckpt(model_cfg, n_classes, ckpt_path=args.model_ckpt_path, le=args.linear_eval, mlp_do=~args.linear_eval)

    # init datamodule
    datamodule = SensorDataModule(
        dataset_cfg['train'],
        dataset_cfg['val'],
        dataset_cfg['test'],
        batch_size = args.batch_size,
        store_in_ram = ~args.no_ram,
        devices = dataset_cfg['devices']
    )

    grad_cnn_layer = pick_last_cnn_layer(model, args.model)
    xai_model = initialize_xai_model(args.xai_model, model, grad_cnn_layer)

    # produce predictions and attributions, and get examples and labels
    examples, labels, preds, attributions = produce_all_preds_and_attrs(
        datamodule, model, xai_model, args.batch_size
    )

    # save all input and outputs
    results_save_path = generate_path(args.save_path, args.model_ckpt_path)

    save_array(examples, save_path=os.path.join(results_save_path, 'data.npy'))
    save_array(labels, save_path=os.path.join(results_save_path, 'labels.npy'))
    save_array(preds, save_path=os.path.join(results_save_path, 'predictions.npy'))
    save_array(attributions, save_path=os.path.join(results_save_path, 'attributions.npy'))

    return examples, labels, preds, attributions


def load_logs(save_path):
    examples = np.load(os.path.join(save_path, 'data.npy'))
    attributions = np.load(os.path.join(save_path, 'attributions.npy'))
    labels = np.load(os.path.join(save_path, 'labels.npy'))
    preds = np.load(os.path.join(save_path, 'predictions.npy'))
    return examples, labels, preds, attributions


def draw_global_attribution_scores(labels, preds, attributions, save_path, dataset_cfg):
    # get true positives indices
    print((labels == preds).sum() / len(labels))
    tp_idx = np.where(labels == preds)
    # select true positive attributions and labels
    tp_attributions = attributions[tp_idx]
    tp_labels = labels[tp_idx]
    tp_preds = preds[tp_idx]

    attr_sum_over_ts = np.abs(tp_attributions).sum(axis=2)
    sum_per_example = attr_sum_over_ts.sum(axis=1).reshape(len(attr_sum_over_ts), 1)
    normalized_attr_sum_over_ts = (attr_sum_over_ts / sum_per_example)
    
    tp_labels = tp_labels[~np.isnan(normalized_attr_sum_over_ts.sum(axis=1))]

    normalized_attr_sum_over_ts = normalized_attr_sum_over_ts[~np.isnan(normalized_attr_sum_over_ts.sum(axis=1))]
    
    kl_div_summary(tp_labels, normalized_attr_sum_over_ts, save_path, dataset_cfg)

    draw_global_heatmap(tp_labels, normalized_attr_sum_over_ts, save_path, dataset_cfg)
    draw_global_barplot(tp_labels, normalized_attr_sum_over_ts, save_path, dataset_cfg)

def kl_div_summary(tp_labels, normalized_attr_sum_over_ts, save_path, dataset_cfg):
    unique_labels = np.unique(tp_labels)
    uniform = np.zeros(len(dataset_cfg['devices'])) + 1 / len(dataset_cfg['devices'])

    kl_list = []
    kl_lists_by_labels = [[] for _ in range(len(unique_labels))]
    kl_summary = []
    for i in range(len(normalized_attr_sum_over_ts)):
        # tmp_kl = entropy(normalized_attr_sum_over_ts[i], base=2)
        tmp_kl = entropy(normalized_attr_sum_over_ts[i], base=2)
        kl_list.append(tmp_kl)
        kl_lists_by_labels[tp_labels[i]].append(tmp_kl)
    
    for i in range(len(unique_labels)):
        tmp_mean_kl = np.array(kl_lists_by_labels[i]).mean()
        kl_summary.append((dataset_cfg['class_names'][i].capitalize(), round(tmp_mean_kl,2)))

    average_kl = np.array(kl_list).mean()
    kl_summary.append(('Average', round(average_kl, 2)))
    kl_summary_df = pd.DataFrame(kl_summary)
    print(kl_summary_df)
    kl_summary_df.to_csv(os.path.join(save_path, 'kl_div_summary.csv') , index=None, header=None)

def draw_global_heatmap(tp_labels, normalized_attr_sum_over_ts, save_path, dataset_cfg):
    unique_labels = np.unique(tp_labels)
    contributions = np.zeros((len(dataset_cfg['class_names']), len(dataset_cfg['devices'])))
    for l in unique_labels:
        ids = np.where(tp_labels == l) 
        avg_channel_importance = normalized_attr_sum_over_ts[ids].mean(axis=0)
        contributions[l] = avg_channel_importance

    contributions_df = pd.DataFrame(contributions)
    contributions_df.columns = dataset_cfg['devices']
    contributions_df.index = dataset_cfg['class_names']

    if len(dataset_cfg['devices']) > 10:
        plt.figure(figsize=(15, 10))
    sns.heatmap(contributions_df, xticklabels=True, yticklabels=True, annot=True, vmin=0, vmax=0.3)
    plt.savefig(os.path.join(save_path, 'global_heatmap.png'), bbox_inches="tight")
    plt.show()   


def draw_global_barplot(tp_labels, normalized_attr_sum_over_ts, save_path, dataset_cfg):
    res_list = []
    for i in range(len(normalized_attr_sum_over_ts)):
        for j in range(len(normalized_attr_sum_over_ts[0])):
            res_list.append((dataset_cfg['devices'][j], tp_labels[i], dataset_cfg['class_names'][tp_labels[i]], normalized_attr_sum_over_ts[i, j]))
    
    attr_df = pd.DataFrame(res_list)
    attr_df.columns = ['Device', 'Activity ID', 'Activity', 'Attribution score']
    attr_df = attr_df.sort_values(by=['Activity ID', 'Device'])

    plt.figure(figsize=(20, 10))
    sns.set_style("whitegrid")
    sns.barplot(x='Activity', y='Attribution score', hue='Device', data=attr_df)
    plt.savefig(os.path.join(save_path, 'global_barplot.png'), bbox_inches="tight")
    plt.show()   


if __name__ == '__main__':
    args = parse_arguments()
    if args.use_saved_attributions:
        _, labels, preds, attributions = load_logs(args.save_path)
        res_save_path = args.save_path
    else:
        examples, labels, preds, attributions = generate_and_save(args)
        res_save_path = generate_path(args.save_path, args.model_ckpt_path)

    if args.draw_global:
        dataset_cfg = load_yaml_to_dict(args.dataset_config_path)[args.dataset]
        draw_global_attribution_scores(labels, preds, attributions, res_save_path, dataset_cfg)
