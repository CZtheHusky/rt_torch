from einops import pack, unpack
import cv2
import numpy as np
import torch
# helpers

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)


def pack_one(x, pattern):
    return pack([x], pattern)


def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]

def nlp_inst_decoder(bytes_list):
    non_zero = bytes_list[np.where(bytes_list != 0)]
    if non_zero.shape[0] == 0:
        return ''
    else:
        bytes_list = bytes(non_zero.tolist())
    return bytes_list.decode('utf-8')

def lt_env_rgb_preprocess(rgb):
    # pad rgb (180, 320, 3) tp (320, 320, 3)
    rgb = np.pad(rgb, ((70, 70), (0, 0), (0, 0)), 'constant', constant_values=0)
    # resize rgb (320, 320, 3) to (3, 300, 300)
    rgb = cv2.resize(rgb, (300, 300))
    rgb = rgb.transpose(2, 0, 1)
    rgb = torch.tensor(rgb).float() / 255
    return rgb


