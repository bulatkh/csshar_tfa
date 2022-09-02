import argparse
import os

import numpy as np
import pandas as pd 
import scipy.stats

from utils.experiment_utils import make_dirs, save_wandb_logs_from_api


def parse_arguments():
    parser = argparse.ArgumentParser()

    # paths
    parser.add_argument('--wandb_project_name', required=True, help='Wandb project name')
    parser.add_argument('--logs_path', required=True, help='Path for wandb logs to save')
    parser.add_argument('--summary_path', required=True, help='Destination path for generated summaries')
    
    # parameters for summaries
    parser.add_argument('--probing_tasks', required=True, nargs='+', help='Probing tasks from experiments for summaries')
    parser.add_argument('--frameworks', required=True, nargs='+', help='Frameworks from experiments for summaries')
    parser.add_argument('--metric', default='test_f1-score', help='Metric for summaries')

    return parser.parse_args()


def calculate_mean_and_conf_intervals(array, conf=0.95):
    n = len(array)
    m = np.mean(array)
    se = scipy.stats.sem(array)
    margin = se * scipy.stats.t.ppf((1 + conf) / 2, n-1)
    return m, margin, m - margin, m + margin


def summarize_probing_experiment(logs_path, destination_path, probing_tasks, frameworks, metric='test_f1-score'):
    probing_results = pd.read_csv(logs_path)
    summary = []
    for probing_task in probing_tasks:
        probing_task_results = probing_results[probing_results['probing_type'] == probing_task]
        for framework in frameworks:
            probing_task_framework_results = probing_task_results[probing_task_results['ssl_framework'] == framework]
            mean, margin, lower, upper = calculate_mean_and_conf_intervals(probing_task_framework_results[metric])
            summary.append((probing_task, framework, mean, margin, lower, upper))
    summary_df = pd.DataFrame(summary)
    summary_df.columns = ['probing_task', 'framework', 'mean', 'margin', 'lower', 'upper']
    summary_df.to_csv(destination_path, index=None)


def run_summary_generation(args):
    make_dirs(os.path.dirname(args.logs_path))
    make_dirs(os.path.dirname(args.summary_path))
    save_wandb_logs_from_api(args.wandb_project_name, args.logs_path)
    summarize_probing_experiment(args.logs_path, args.summary_path, args.probing_tasks, args.frameworks, args.metric)


if __name__ == '__main__':
    args = parse_arguments()
    run_summary_generation(args)
    