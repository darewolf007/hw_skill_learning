import yaml
import pickle
import torch
import numpy as np
import collections
import csv
import os

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

def quaternion_to_rotation_matrix(quaternion):
    """
    将四元数表示的旋转角度转换为旋转矩阵
    """
    w, x, y, z = quaternion
    rotation_matrix = torch.tensor([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ], device=quaternion.device)  # 将旋转矩阵分配到与四元数相同的设备上
    return rotation_matrix

def angle_loss(quaternion1, quaternion2):
    """
    计算两个四元数之间的角度损失
    """
    rotation_matrix1 = quaternion_to_rotation_matrix(quaternion1)
    rotation_matrix2 = quaternion_to_rotation_matrix(quaternion2)
    dot_product = torch.trace(torch.matmul(rotation_matrix1, torch.transpose(rotation_matrix2, 0, 1)))
    cosine_similarity = dot_product / 3.0
    angle = torch.acos(torch.clamp(cosine_similarity, -1.0 + 1e-7, 1.0 - 1e-7))
    return angle

def position_loss(position1, position2):
    """
    计算两个位置向量之间的位置损失
    """
    return torch.norm(position1 - position2)

def combined_loss(quaternion1, quaternion2, position1, position2, angle_weight=0.5):
    """
    计算结合了角度和位置的损失函数
    """
    combined_loss = 0.0
    for bs in range(quaternion1.shape[0]):
        angle_loss_val = angle_loss(quaternion1[bs], quaternion2[bs])
        position_loss_val = position_loss(position1[bs], position2[bs])
        combined_loss += (1.0 - angle_weight) * angle_loss_val + angle_weight * position_loss_val
    combined_loss /= quaternion1.shape[0]
    return combined_loss