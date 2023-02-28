import argparse
import logging
import os
import random
import re
import shutil
import time
import json
import numpy as np
import torch
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import EfficientNet_B3_Weights
from tqdm import tqdm
# import tensorflow as tf
from robotic_transformer import RT1, film_efficientnet_b3
from robotic_transformer.dataset import language_table_dataset
from robotic_transformer.dataset_subproc import SubProcLTDataset
from robotic_transformer.dataset_torch_wrapper import language_table_dataset_py
from robotic_transformer.dataset_npz import language_table_dataset_npz
from robotic_transformer.helper_classes import ActionTokenizer
from collections import defaultdict
from torch.utils.data import DataLoader
from robotic_transformer.helper_classes import InstEmbeddingBuffer
from operator import itemgetter 
import tensorflow as tf
import time
from collections import deque
import deepspeed
import mpu
from mpu import print_rank_0, print_with_rank

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # tf.random.set_seed(seed)


def log_init(args=None):
    history = os.path.expanduser('~/history')
    # log_name = time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '/'
    if args.load_path is not None:
        log_name = os.path.basename(args.load_path) + "_" + time.strftime("%m%d-%H%M",time.localtime())
    else:
        log_name = time.strftime("%m%d-%H%M",time.localtime()) + "_" + args.text_encoder + "_" + str(args.lr) + "_" + args.alias
    log_path = os.path.join(history, log_name)
    models_path = os.path.join(log_path, 'models')
    print(history)
    print(log_path)
    print(models_path)
    os.makedirs(history, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    logfile = os.path.join(log_path, log_name + '.log')
    handler = logging.FileHandler(logfile, mode='a+')
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(handler)
    logger.info("Start print log")
    setup_seed(args.seed)
    logger.info("seed: {}".format(args.seed))
    writer = SummaryWriter(log_path)
    # filename_list = os.listdir('.')
    # expr = '\.py'
    # for filename in filename_list:
    #     if re.search(expr, filename) is not None:
    #         print()
    #         shutil.copyfile('./' + filename, os.path.join(log_path, filename))
    with open(os.path.join(log_path, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f)
    return logger, writer, models_path, log_path, handler

parser = argparse.ArgumentParser()
parser.add_argument('--device', default="cuda", type=str)
parser.add_argument('--device_idx', default="7", type=str)
parser.add_argument('--text_encoder', default="t5", type=str)
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--num_epoch', default=1000, type=int, help='num_epoch')
parser.add_argument('--norm_clip', default=40, type=int, help='clip norm')
parser.add_argument('--test_interval', default=1000, type=int, help='test_interval')
parser.add_argument('--seed', default=100, type=int, help='seed of numpy, tensorflow and pytorch')
parser.add_argument('--save_interval', default=10000, type=int)
parser.add_argument('--alias', default="", type=str, help="alias of the experiment")
parser.add_argument('--sub_data', default="mix", type=str, help="data for training")
parser.add_argument('--max_save_num', default=20, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--load_path', default=None, type=str, help="checkpoint path to load")
parser.add_argument('--load_args', action='store_true', help="load the args")


def main(args):
    effnet = film_efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    model = RT1(efficient=effnet, num_actions=2, depth=8, heads=16, dim_head=128, cond_drop_prob=0.2, text_encoder=text_encoder)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
    )
    print("deep args", args.deepspeed_port)

    deepspeed.init_distributed(distributed_port=args.deepspeed_port)

    mpu.initialize_model_parallel()

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        print(
            " > number of parameters on (tensor, pipeline) "
            "model parallel rank ({}, {}): {}".format(
                mpu.get_tensor_model_parallel_rank(),
                mpu.get_pipeline_model_parallel_rank(),
                sum([p.nelement() for p in model.parameters()]),
            ),
            flush=True,
        )
    print_rank_0(" ============= MPU_INIT ==============")
    model_engine, _, _, _ = deepspeed.initialize(
        args,
        model,
        model_parameters=model.parameters(),
        mpu=mpu,
        optimizer=optimizer,
        lr_scheduler=optimizer,
    )
    print_rank_0(" ============= DS_INIT ==============")
    args.device = model_engine.device
    from torch.distributed import get_rank

    (
        train_data_iterator,
        valid_data_iterator,
        eval_blendable_dataset,
    ) = get_data_iterators(args)

    if get_rank() == 0:
        logger, writer, save_path, log_path, handler = log_init(args)
    else:
        writer = None
    train(
        args,
        model_engine,
        train_data_iterator,
        valid_data_iterator,
        writer,
        logger,
    )
    writer.close()
    logger.removeHandler(handler)
    logging.shutdown()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)