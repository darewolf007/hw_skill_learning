import yaml
import pickle
import torch
import numpy as np
import csv
import os
from utils.general_utils import map_dict, listdict2dictlist

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def read_pickle(filename):
    with open(filename, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def euclidean_distance(tensor1, tensor2):
    """
    calculate the euclidean distance between two tensors
    """
    squared_difference = (tensor1 - tensor2).pow(2).sum(dim=-1)
    euclidean_dist = squared_difference.sqrt()
    return euclidean_dist

def read_dict_with_twodim_numpy_to_csv(filename):
    csv.field_size_limit(1000000)
    with open(filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)
        read_data_dict = {key: [] for key in header}
        for row in csv_reader:
            for key, value_str in zip(header, row):
                value = np.array(eval(value_str))
                read_data_dict[key].append(value)
    return read_data_dict

def check_and_create_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

def args_overwrite_config(base_config):
    task_name = base_config['task_name']
    task_config = base_config[task_name]
    for key, value in task_config.items():
        if value is not None:
            base_config[key] = value
    return base_config

def list_files_in_directory(directory):
    path_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            path_list.append(os.path.join(root, file))
        for subdir in dirs:
            path_list.append(os.path.join(root, subdir))
    return path_list

def make_recursive_list(fn):
    """ Takes a fn and returns a function that can apply fn across tuples of tensor structures,
     each of which can be a single tensor, tuple or a list. """

    def recursive_map(tensors):
        if tensors is None:
            return tensors
        elif isinstance(tensors[0], list) or isinstance(tensors[0], tuple):
            return type(tensors[0])(map(recursive_map, zip(*tensors)))
        elif isinstance(tensors[0], dict):
            return map_dict(recursive_map, listdict2dictlist(tensors))
        elif isinstance(tensors[0], torch.Tensor):
            return fn(*tensors)
        else:
            try:
                return fn(*tensors)
            except Exception as e:
                print("The following error was raised when recursively applying a function:")
                print(e)
                raise ValueError("Type {} not supported for recursive map".format(type(tensors)))

    return recursive_map

def map_recursive_list(fn, tensors):
    return make_recursive_list(fn)(tensors)

def make_recursive(fn, *argv, **kwargs):
    """ Takes a fn and returns a function that can apply fn on tensor structure
     which can be a single tensor, tuple or a list. """
    
    def recursive_map(tensors):
        if tensors is None:
            return tensors
        elif isinstance(tensors, list) or isinstance(tensors, tuple):
            return type(tensors)(map(recursive_map, tensors))
        elif isinstance(tensors, dict):
            return type(tensors)(map_dict(recursive_map, tensors))
        elif isinstance(tensors, torch.Tensor) or isinstance(tensors, np.ndarray):
            return fn(tensors, *argv, **kwargs)
        else:
            try:
                return fn(tensors, *argv, **kwargs)
            except Exception as e:
                print("The following error was raised when recursively applying a function:")
                print(e)
                raise ValueError("Type {} not supported for recursive map".format(type(tensors)))
    
    return recursive_map

def map_recursive(fn, tensors):
    return make_recursive(fn)(tensors)
