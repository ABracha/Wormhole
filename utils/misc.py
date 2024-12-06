import os
import os.path as osp
import random
import time
import numpy as np
import torch

from .dist_util import master_only
from itertools import combinations



def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


@master_only
def mkdir_and_rename(path):
    """
    Make directory. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    if osp.exists(path):
        new_name = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)


@master_only
def make_exp_dirs(opt):
    """Make dirs for experiments."""
    path_opt = opt['path'].copy()
    if opt['is_train']:
        mkdir_and_rename(path_opt['experiments_root'])
        os.makedirs(path_opt['models'], exist_ok=True)
        os.makedirs(path_opt['log'], exist_ok=True)
    else:
        mkdir_and_rename(path_opt['results_root'])
        os.makedirs(path_opt['visualization'], exist_ok=True)
        os.makedirs(path_opt['log'], exist_ok=True)


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative pathes.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def sizeof_fmt(size, suffix='B'):
    """Get human readable file size.

    Args:
        size (int): File size.
        suffix (str): Suffix. Default: 'B'.

    Return:
        str: Formatted file siz.
    """
    for unit in ['B', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(size) < 1024.0:
            return f'{size:3.1f} {unit}{suffix}'
        size /= 1024.0
    return f'{size:3.1f} Y{suffix}'

def read_words_from_file(file_path):
    words_list = []
    with open(file_path, 'r') as file:
        for line in file:
            # Strip away newline and any leading/trailing whitespaces
            word = line.strip()
            words_list.append(word)
    return words_list



def find_existing_pairs(folder_name, words_list):
    # Generate all unique pairs of words
    words_list = [w[:-4] for w in words_list]
    paired_list = list(combinations(words_list, 2))
    # Initialize a list to store tuples that have a corresponding file
    existing_files = []
    filenames = []

    # Iterate through each pair to check if the corresponding file exists
    for pair in paired_list:
        # Format the filename as 'word1-word2.map'
        filename = f"{pair[0]}_{pair[1]}.map"
        # Construct the full path to check
        file_path = os.path.join(folder_name, filename)
        # Check if the file exists
        if os.path.isfile(file_path):
            existing_files.append(pair)
            filenames.append(file_path)

    return existing_files, filenames