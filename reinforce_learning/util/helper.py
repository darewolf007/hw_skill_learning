import cv2
import numpy as np

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
    
def prefix_dict(d, prefix):
    """Adds the prefix to all keys of dict d."""
    return type(d)({prefix+k: v for k, v in d.items()})

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

