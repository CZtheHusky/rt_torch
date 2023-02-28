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
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import EfficientNet_B3_Weights
from tqdm import tqdm
from robotic_transformer import RT1, film_efficientnet_b3
from robotic_transformer.helper_classes import ActionTokenizer
import time
from torch.optim import Adam, SGD, AdamW


"""Example for running the Language-Table environment."""

from collections.abc import Sequence

from absl import app

from language_table.environments import blocks
from language_table.environments import language_table
from language_table.environments.rewards import block2block

from matplotlib import pyplot as plt
from robotic_transformer.helpers import nlp_inst_decoder, lt_env_rgb_preprocess
from collections import deque
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1."

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
parser.add_argument('--depth', default=8, type=int)
parser.add_argument('--heads', default=8, type=int)
parser.add_argument('--layersize', default=512, type=int)
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--load_path', default=None, type=str, help="checkpoint path to load")
parser.add_argument('--load_args', action='store_true', help="load the args")
parser.add_argument('--optimizer', default="adam", type=str, help="type of the optimizer")
parser.add_argument('--scheduler', default=None, type=str, help="type of the optimizer")
parser.add_argument('--warmup', action='store_true', help="using warmup scheduler")
parser.add_argument('--adam_beta1', default=0.9, type=float, help="")
parser.add_argument('--adam_beta2', default=0.99, type=float, help="")
parser.add_argument('--adam_eps', default=1e-8, type=float, help="")
parser.add_argument('--weight_decay', default=0, type=float, help="")
parser.add_argument('--sgd_momentum', default=0, type=float, help="")





# parser.add_argument('--check_point', action='store_true')
# parser.add_argument('--reward_shaping', action='store_true')
# parser.add_argument('--mix_maps', action='store_true')


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # tf.random.set_seed(seed)


def log_init(args=None):
    history = os.path.expanduser('~/history')
    # history = "/raid/history"
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

# # Load the arguments from a file
# with open('args.json', 'r') as f:
#     args_dict = json.load(f)

# # Create a new Namespace object from the saved arguments
# args = argparse.Namespace(**args_dict)



def main(args):
    print(args)
    device = args.device
    text_encoder = args.text_encoder
    load_path = args.load_path
    depth = args.depth
    heads = args.heads
    layersize = args.layersize
    print('device: ', device)
    effnet = film_efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    model = RT1(efficient=effnet, num_actions=2, depth=depth, heads=heads, layer_size=layersize, cond_drop_prob=0.1, text_encoder=text_encoder)
    model.to(device)
    model_path = os.path.join(load_path, 'models')
    file_models = sorted(os.listdir(model_path))
    newest_model = os.path.join(model_path, file_models[-1])
    state_dict = torch.load(newest_model)
    model.load_state_dict(state_dict['model_state_dict'])

    action_tokenizer = ActionTokenizer(num_action_bin=256, action_max=0.1, action_min=-0.1)
    env = language_table.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
        reward_factory=block2block.BlockToBlockReward,
        control_frequency=10.0,
    )
    with torch.no_grad():
        for ep in range(args.eval_ep):
            model.eval()
            env_obs = env.reset()
            tmp = [torch.zeros((3, 300, 300))] * 5
            rgb_buffer = deque(tmp, maxlen=6)
            instruction = [nlp_inst_decoder(env_obs['instruction'])]
            rgb = lt_env_rgb_preprocess(env_obs['rgb'])
            rgb_buffer.append(rgb)
            # Take a few random actions.
            for _ in range(5):
                video = torch.cat(rgb_buffer).permute((0, 2, 1, 3, 4))
                predicts = model(video=video, texts=instruction, texts_embeddings=None)
                action = action_tokenizer.discrete2Scalar(predicts)
                env_obs = env.step(action)
                print(env_obs)
            
            



if __name__ == "__main__":
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_idx
    if args.load_path is not None:
        load_path = args.load_path
        if args.load_args:
            # Load the arguments from a file
            with open(os.path.join(args.load_path, 'args.json'), 'r') as f:
                args_dict = json.load(f)
            args_dict["load_path"] = load_path
            args_dict["load_args"] = args.load_args
            # Create a new Namespace object from the saved arguments
            print("loading args")
            args = argparse.Namespace(**args_dict)
    
    main(args)
