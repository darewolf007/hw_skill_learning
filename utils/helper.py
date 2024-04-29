import yaml
import pickle
import torch
import numpy as np
import collections

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

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def prefix_dict(d, prefix):
    """Adds the prefix to all keys of dict d."""
    return type(d)({prefix+k: v for k, v in d.items()})

def euclidean_distance(tensor1, tensor2):
    """
    calculate the euclidean distance between two tensors
    """
    squared_difference = (tensor1 - tensor2).pow(2).sum(dim=-1)
    euclidean_dist = squared_difference.sqrt()
    return euclidean_dist