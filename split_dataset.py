import argparse
import math
import os
import numpy as np
import random
import shutil

from utils.utils import seed_all


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', help='seed_value', default=28)
    parser.add_argument("--dataset", help="dataset to split", default='pamap2')
    parser.add_argument("--path", help="path to the sampled dataset folder", required=True)
    parser.add_argument("--random_subject_split", help="flag for selecting subjects randomly", action='store_true', default=False)
    parser.add_argument("--test_proportion", type=float, default=0.2)
    parser.add_argument("--val_proportion_remaining", type=float, default=0.2)
    parser.add_argument("--val_users", help="user IDs for validation split", nargs='+')
    parser.add_argument("--test_users", help="user IDs for test split", nargs='+')
    parser.add_argument("--opportunity_val_runs", help='runs for validation', nargs='+', default=[])
    parser.add_argument("--opportunity_test_runs", help='runs for validation', nargs='+', default=[])
    parser.add_argument("--cross_subject_splits", action='store_true', default=False)
    parser.add_argument("--num_folds", type=int, default=5)
    args = parser.parse_args()

    return args


def copy_files_array(file_paths, destination_path):
    """ Copies array of files to a destination folder

        Parameters
        ----------
        file_paths : list
            list of paths to files
        destination_path : str
            path to the destination folder
    """
    for file in file_paths:
        shutil.copy(file, destination_path)


def random_subject_splits(subjects_list, test_proportion, val_proportion_remaining):
    num_subjects = len(subjects_list)
    num_subjects_test = math.ceil(num_subjects * test_proportion)
    num_subjects_val = math.ceil((num_subjects - num_subjects_test) * val_proportion_remaining)

    test_subjects = random.sample(subjects_list, num_subjects_test)
    remaining_subjects = [subject for subject in subjects_list if subject not in test_subjects]
    val_subjects = random.sample(remaining_subjects, num_subjects_val)
    train_subjects = [subject for subject in remaining_subjects if subject not in val_subjects]

    print("Random train subjects", train_subjects)
    print("Random val subjects", val_subjects)
    print("Random test subjects", test_subjects)
    return train_subjects, val_subjects, test_subjects


def generate_subject_folds(subjects_list, num_folds):
    np.random.shuffle(subjects_list)
    folds_subjects = np.array_split(subjects_list, num_folds)
    return folds_subjects


def split_data(data_path, destination_path, files, test_users, val_users):
    train_path = os.path.join(destination_path, 'train')
    val_path = os.path.join(destination_path, 'val')
    test_path = os.path.join(destination_path, 'test')

    if not os.path.exists(train_path):
        os.makedirs(train_path)

    if not os.path.exists(val_path):
        os.makedirs(val_path)

    if not os.path.exists(test_path):
        os.makedirs(test_path)

    if val_users:
        val_users_files = [file for file in files if file.split('_')[0] in val_users]
        val_users_full = [os.path.join(data_path, file) for file in val_users_files]
        if len(val_users_files) == 0:
            raise ValueError('inappropriate val_user ID for specified data folder')

    test_users_files = [file for file in files if file.split('_')[0] in test_users]
    test_users_full = [os.path.join(data_path, file) for file in test_users_files]
    if len(val_users_files) == 0:
        raise ValueError('inappropriate test_user ID for specified data folder')

    train_users_full = [os.path.join(data_path, file) for file in files if file not in val_users_files and file not in test_users_files and file.endswith('.csv')]

    
    print('Copying train split..')
    copy_files_array(train_users_full, train_path)
    print('Copying val split..')
    copy_files_array(val_users_full, val_path)
    print('Copying test split..')
    copy_files_array(test_users_full, test_path)


def main():
    args = parse_arguments()
    seed_all(args.seed)
    files = os.listdir(args.path)

    subjects = set([file_.split('_')[0] for file_ in files if file_ not in ['train', 'val', 'test']])
    if args.random_subject_split:
        _, val_users, test_users = random_subject_splits(subjects, args.test_proportion, args.val_proportion_remaining)
    elif args.cross_subject_splits:
        folds_subjects = generate_subject_folds(sorted(list(subjects)), args.num_folds)
    else:
        val_users = args.val_users
        test_users = args.test_users

    if args.cross_subject_splits:
        print(folds_subjects)
        for i in range(0, args.num_folds):
            test_users = folds_subjects[i]
            train_and_val_users = [subject for subject in subjects if subject not in test_users]
            val_users = random.sample(train_and_val_users, int(args.val_proportion_remaining * len(train_and_val_users)))
            print('val:', sorted(val_users))
            print('test:', sorted(test_users))
            print('----------------------------------------')
            destination_path = os.path.join(args.path, 'fold{}'.format(i + 1))
            split_data(args.path, destination_path, files, test_users, val_users)
    else:
        split_data(args.path, args.path, files, test_users, val_users)
        
    

if __name__ == '__main__':
    main()
