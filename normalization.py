import argparse
import os

import numpy as np
import pandas as pd


def get_means(file_paths):
    """ Function for calculating means for each column accross the whole training set consisting of multiple files
    
        Parameters
        ----------
        file_paths : array-like
            a list of paths to training set files
    """
    cols = pd.read_csv(file_paths[0]).shape[1]
    means = np.zeros(cols)
    for path in file_paths:
        tmp_df = pd.read_csv(path)
        means += tmp_df.mean()
    means /= len(file_paths)
    return means

def get_stds_given_means(file_paths, means):
    """ Function for calculating stds for each column accross the whole training set consisting of multiple files
    
        Parameters
        ----------
        file_paths : array-like
            a list of paths to training set files
        means : array-like
            mean values for each column
    """
    shape = pd.read_csv(file_paths[0]).shape
    stds = np.zeros(shape[1])
    for path in file_paths:
        tmp_df = pd.read_csv(path)
        stds += ((tmp_df - means) * (tmp_df - means)).sum()
    stds /= (len(file_paths) * shape[0] - 1)
    stds = np.sqrt(stds)
    return stds

def normalize_given_mean_std(file_paths, means, stds):
    """ Function for calculating means for each column accross the whole training set consisting of multiple files
    
        Parameters
        ----------
        file_paths : array-like
            a list of files to be normalized
        means : array-like
            mean values for each column
        stds : array-like
            std values for each column
    """
    for path in file_paths:
        tmp_df = pd.read_csv(path)
        tmp_df = (tmp_df - means) / stds
        tmp_df.to_csv(path, index=None)

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', help='path to the training folder', required=True)
    parser.add_argument('--normalization_paths', help='path to the folder for normalization', required=True, nargs='+')
    
    args = parser.parse_args()

    return args

def main():
    args = parse_arguments()

    train_paths = [os.path.join(args.train_path, file) for file in os.listdir(args.train_path)]

    print('Calculating means..')
    means = get_means(train_paths)
    print('Calculating stds..')
    stds = get_stds_given_means(train_paths, means)

    for path in args.normalization_paths:
        files = [os.path.join(path, file_) for file_ in os.listdir(path)]
        print('Normalizing {}'.format(path))
        normalize_given_mean_std(files, means, stds)

if __name__ == "__main__":
    main()
