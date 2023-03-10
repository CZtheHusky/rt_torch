from einops import pack, unpack
import cv2
import numpy as np
import torch
import random
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


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

def print_with_rank(message):
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        print(f"rank: {rank}", message, flush=True)
    else:
        print(message, flush=True)

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # tf.random.set_seed(seed)

def getModelSize(model, logger):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    logger.info('Model size: {:.3f}MB'.format(all_size))
    logger.info('Parameter size: {:.3f}MB'.format(param_size / 1024 / 1024))
    logger.info('Parameter num: {:.3f} m'.format(param_sum / 1024 / 1024))
    logger.info('Buffer size: {:.3f}MB'.format(buffer_size / 1024 /1024))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)