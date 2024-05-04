from functools import partial
import math
import numpy as np
from contextlib import contextmanager
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torch.nn.parallel._functions import Gather
from torch.optim.optimizer import Optimizer
from torch.nn.modules import BatchNorm1d, BatchNorm2d, BatchNorm3d
from torch.nn.functional import interpolate

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, digits=None):
        """
        :param digits: number of digits returned for average value
        """
        self._digits = digits
        self.reset()

    def reset(self):
        self.val = 0
        self._avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self._avg = self.sum / self.count

    @property
    def avg(self):
        if self._digits is not None:
            return np.round(self._avg, self._digits)
        else:
            return self._avg

def map_dict(fn, d):
    """takes a dictionary and applies the function to every element"""
    return type(d)(map(lambda kv: (kv[0], fn(kv[1])), d.items()))

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

def ten2ar(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    elif torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    elif np.isscalar(tensor):
        return tensor
    elif hasattr(tensor, 'to_numpy'):
        return tensor.to_numpy()
    else:
        import pdb; pdb.set_trace()
        raise ValueError('input to ten2ar cannot be converted to numpy array')
    
def avg_grad_norm(model):
    """Computes average gradient norm for the given model."""
    grad_norm = AverageMeter()
    for p in model.parameters():
        if p.grad is not None:
            grad_norm.update(torch.norm(p.grad.data, p=2))
    return grad_norm.avg

class TensorModule(nn.Module):
    """A dummy module that wraps a single tensor and allows it to be handled like a network (for optimizer etc)."""
    def __init__(self, t):
        super().__init__()
        self.t = nn.Parameter(t)

    def forward(self, *args, **kwargs):
        return self.t
    
def check_shape(t, target_shape):
    if not list(t.shape) == target_shape:
        raise ValueError(f"Temsor should have shape {target_shape} but has shape {list(t.shape)}!")
    
def ar2ten(array, device, dtype=None):
    if isinstance(array, list) or isinstance(array, dict):
        return array

    if isinstance(array, np.ndarray):
        tensor = torch.from_numpy(array).to(device)
    else:
        tensor = torch.tensor(array).to(device)
    if dtype is not None:
        tensor = tensor.to(dtype)
    return tensor

def map2torch(struct, device):
    """Recursively maps all elements in struct to torch tensors on the specified device."""
    return map_recursive(partial(ar2ten, device=device, dtype=torch.float32), struct)


def map2np(struct):
    """Recursively maps all elements in struct to numpy ndarrays."""
    return map_recursive(ten2ar, struct)