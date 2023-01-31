import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns
import scipy.stats

from utils.experiment_utils import make_dirs, save_wandb_logs_from_api, read_json_to_dict


def parse_arguments():
    parser = argparse.ArgumentParser()

    # paths
    parser.add_argument('--wandb_project_name', required=True, help='Wandb project name')
    parser.add_argument('--logs_path', required=True, help='Path for wandb logs to save')
    parser.add_argument('--summary_path', required=True, help='Destination path for generated summaries')
    

    return parser.parse_args()


def generate_summary(logs_path, summary_path):
    logs = pd.read_csv(logs_path)

    draw_test_occlusion(logs, summary_path)

    names = logs.iloc[:,0]
    wandb_local_runs = os.listdir('./wandb')
    test_occlusion_jsons = {}
    for name in names:
        for run in wandb_local_runs:
            if name in run:
                confusion_matrix_path = os.path.join('./wandb', run, 'files/media/table')
                test_occlusion_jsons[name] = os.path.join(confusion_matrix_path, os.listdir(confusion_matrix_path)[0])
    dfs = []
    for name, file_ in test_occlusion_jsons.items():
        table_dict = read_json_to_dict(file_)
        table_df = pd.DataFrame(table_dict['data'])
        table_df.columns = table_dict['columns']
        num_act = table_df.groupby(['Actual']).sum()
        table_df['Norm'] = table_df.nPredictions.div(table_df.Actual.map(num_act.nPredictions))
        table_df = table_df[table_df['Actual'] == table_df['Predicted']]
        table_df = table_df.drop(['Predicted', 'nPredictions'], axis=1)
        transposed = table_df.set_index('Actual').transpose().reset_index(drop=True)
        transposed['name'] = name
        dfs.append(transposed)
    
    performance_by_activity = pd.concat(dfs, axis=0).reset_index()
    merged = logs[['0', 'randomly_masked_channels_test', 'ssl_framework']].merge(performance_by_activity, left_on='0', right_on='name')
    averaged_groups = merged.groupby(['randomly_masked_channels_test', 'ssl_framework']).mean().reset_index()
    for framework in ['simclr', 'vicreg', 'supervised']:
        framework_avg = averaged_groups[averaged_groups['ssl_framework'] == framework].reset_index(drop=True)
        diff = framework_avg.rolling(2).apply(lambda x: x[1] - x[0])
        drop_per_activity = (diff * 100).iloc[1:, 2:].abs().round(1).transpose().iloc[:,:2]
        drop_per_activity.to_csv(os.path.join(os.path.dirname(summary_path), f'{framework}_test_occlusion_drop.csv'), header=None)


def draw_test_occlusion(logs, summary_path):
    logs_main = logs[['ssl_framework', 'randomly_masked_channels_test', 'test_f1-score']]
    logs_main['test_f1-score'] = logs_main['test_f1-score'] * 100
    logs_main.columns = ['Framework', 'Masked Channels', 'Test F1-score (%)']
    sns.set_style("whitegrid")
    palette = sns.color_palette("Set2")
    sns.lineplot(data=logs_main, x='Masked Channels', y='Test F1-score (%)', hue='Framework', markers=True, style='Framework', errorbar=("ci", 95))
    plt.xticks(np.arange(0, 6, 1))
    plt.yticks(np.arange(0, 101, 10))
    plt.savefig(os.path.join(os.path.dirname(summary_path), 'random_test_noise.pdf'), bbox_inches='tight')
    

def run_summary_generation(args):
    make_dirs(os.path.dirname(args.logs_path))
    make_dirs(os.path.dirname(args.summary_path))
    save_wandb_logs_from_api(args.wandb_project_name, args.logs_path)
    generate_summary(args.logs_path, args.summary_path)


if __name__ == '__main__':
    args = parse_arguments()
    run_summary_generation(args)