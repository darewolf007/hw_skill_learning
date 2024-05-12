import cv2
import numpy as np
import torch
from mpi4py import MPI
from functools import partial, reduce

class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, attr):
        # Take care that getattr() raises AttributeError, not KeyError.
        # Required e.g. for hasattr(), deepcopy and OrderedDict.
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" % attr)

    def __getstate__(self):
        return self

    def __setstate__(self, d):
        self = d

class ParamDict(AttrDict):
    def overwrite(self, new_params):
        for param in new_params:
            # print('overriding param {} to value {}'.format(param, new_params[param]))
            self.__setattr__(param, new_params[param])
        return self

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


def map_recursive_list(fn, tensors):
    return make_recursive_list(fn)(tensors)

class RecursiveAverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0

    def update(self, val):
        self.val = val
        if self.sum is None:
            self.sum = val
        else:
            self.sum = map_recursive_list(lambda x, y: x + y, [self.sum, val])
        self.count += 1
        self.avg = map_recursive(lambda x: x / self.count, self.sum)

def map_dict(fn, d):
    """takes a dictionary and applies the function to every element"""
    return type(d)(map(lambda kv: (kv[0], fn(kv[1])), d.items()))

def str2int(str):
    try:
        return int(str)
    except ValueError:
        return None

def np2obj(np_array):
    if isinstance(np_array, list) or np_array.size > 1:
        return [e[0] for e in np_array]
    else:
        return np_array[0]


def add_caption_to_img(img, info, name=None, flip_rgb=False):
    """ Adds caption to an image. info is dict with keys and text/array.
        :arg name: if given this will be printed as heading in the first line
        :arg flip_rgb: set to True for inputs with BGR color channels
    """
    offset = 12

    frame = img * 255.0 if img.max() <= 1.0 else img
    if flip_rgb:
        frame = frame[:, :, ::-1]

    # make frame larger if needed
    if frame.shape[0] < 300:
        frame = cv2.resize(frame, (400, 400), interpolation=cv2.INTER_CUBIC)

    fheight, fwidth = frame.shape[:2]
    frame = np.concatenate([frame, np.zeros((offset * (len(info.keys()) + 2), fwidth, 3))], 0)

    font_size = 0.4
    thickness = 1
    x, y = 5, fheight + 10
    if name is not None:
        cv2.putText(frame, '[{}]'.format(name),
                    (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_size, (100, 100, 0), thickness, cv2.LINE_AA)
    for i, k in enumerate(info.keys()):
        v = info[k]
        key_text = '{}: '.format(k)
        (key_width, _), _ = cv2.getTextSize(key_text, cv2.FONT_HERSHEY_SIMPLEX,
                                            font_size, thickness)

        cv2.putText(frame, key_text,
                    (x, y + offset * (i + 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_size, (66, 133, 244), thickness, cv2.LINE_AA)

        cv2.putText(frame, str(v),
                    (x + key_width, y + offset * (i + 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_size, (100, 100, 100), thickness, cv2.LINE_AA)

    if flip_rgb:
        frame = frame[:, :, ::-1]

    return frame

def add_captions_to_seq(img_seq, info_seq, **kwargs):
    """Adds caption to sequence of image. info_seq is list of dicts with keys and text/array."""
    return [add_caption_to_img(img, info, name='Timestep {:03d}'.format(i), **kwargs) for i, (img, info) in enumerate(zip(img_seq, info_seq))]

def _get_flat_params(network):
    param_shape = {}
    flat_params = None
    for key_name, value in network.named_parameters():
        param_shape[key_name] = value.cpu().detach().numpy().shape
        if flat_params is None:
            flat_params = value.cpu().detach().numpy().flatten()
        else:
            flat_params = np.append(flat_params, value.cpu().detach().numpy().flatten())
    return flat_params, param_shape

def _set_flat_params(network, params_shape, params):
    pointer = 0
    if hasattr(network, '_config'):
        device = network._config.device
    else:
        device = torch.device("cpu")

    for key_name, values in network.named_parameters():
        # get the length of the parameters
        len_param = int(np.prod(params_shape[key_name]))
        copy_params = params[pointer:pointer + len_param].reshape(params_shape[key_name])
        copy_params = torch.tensor(copy_params).to(device)
        # copy the params
        values.data.copy_(copy_params.data)
        # update the pointer
        pointer += len_param

def sync_networks(network):
    """
    netowrk is the network you want to sync
    """
    comm = MPI.COMM_WORLD
    flat_params, params_shape = _get_flat_params(network)
    comm.Bcast(flat_params, root=0)
    # set the flat params back to the network
    _set_flat_params(network, params_shape, flat_params)

class Schedule:
    """Container for parameter schedules."""
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)

    def _default_hparams(self):
        return ParamDict({})

    def __call__(self, t):
        raise NotImplementedError()

class ConstantSchedule(Schedule):
    def __init__(self, config):
        super().__init__(config)
        self._p = self._hp.p

    def _default_hparams(self):
        return super()._default_hparams().overwrite(AttrDict(
            p=None
        ))

    def __call__(self, t):
        return self._p

def listdict2dictlist(LD):
    """ Converts a list of dicts to a dict of lists """
    
    # Take intersection of keys
    keys = reduce(lambda x,y: x & y, (map(lambda d: d.keys(), LD)))
    return AttrDict({k: [dic[k] for dic in LD] for k in keys})

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