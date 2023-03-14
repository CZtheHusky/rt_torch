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
from rt_torch.rt1.rt_vanilla import RT1_transformer
# from rt_torch.rt1.rt_xl import RT1_transformerxl
from rt_torch.dataset.dataset_npz import build_language_table_ds
from rt_torch.tokenizers.action_tokenizer import ActionTokenizer
from collections import defaultdict
from torch.utils.data import DataLoader
import time
from collections import deque
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler_pytorch import WarmUpScheduler
import tensorflow as tf
from itertools import islice
from rt_torch.utilizes.optimizer_param_scheduler import OptimizerParamScheduler
from rt_torch.utilizes.eval_env import eval_in_env
from rt_torch.utilizes.train_configs import *
from rt_torch.utilizes.training_functions import get_batch
# from rt_torch.utilizes.loggings import log_init

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1."

parser = argparse.ArgumentParser()
parser.add_argument('--device', default="cuda", type=str)
parser.add_argument('--device_idx', default="0", type=str)
parser.add_argument('--text_encoder', default="t5", type=str)
parser.add_argument('--batch_size', default=96, type=int, help='batch size')
parser.add_argument('--loader_bs', default=1, type=int, help='')
parser.add_argument('--loader_shuffle', default=True, type=bool, help="")
parser.add_argument('--loader_worker', default=16, type=int, help='')
parser.add_argument('--train-iters', default=500000, type=int, help='train_iters')
parser.add_argument('--test-iters', default=500, type=int, help='test_iter')
parser.add_argument('--iteration', default=0, type=int, help='iteration')
parser.add_argument('--norm_clip', default=40, type=int, help='clip norm')
parser.add_argument('--test-interval', default=10000, type=int, help='test_interval')
parser.add_argument('--seed', default=100, type=int, help='')
parser.add_argument('--save-interval', default=10000, type=int)
parser.add_argument('--alias', default="", type=str, help="alias of the experiment")
parser.add_argument('--sub_data', default="mix", type=str, help="data for training")
parser.add_argument('--max_save_num', default=5, type=int)
parser.add_argument('--depth', default=2, type=int)
parser.add_argument('--heads', default=8, type=int)
parser.add_argument('--key_dim', default=512, type=int)
parser.add_argument('--model_dim', default=512, type=int)
parser.add_argument('--vocab_size', default=256, type=int)
parser.add_argument('--num_actions', default=2, type=int)
parser.add_argument('--token_learner_num', default=8, type=int)
parser.add_argument('--seq_len', default=6, type=int)
parser.add_argument('--scheduler_t', default=5, type=int)
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--lr_t', default=1, type=float)
parser.add_argument('--lr_eff', default=1, type=float)
parser.add_argument('--min_lr', default=1e-5, type=float)
parser.add_argument('--load_path', default=None, type=str, help="checkpoint path to load")
parser.add_argument('--load_args', action='store_true', help="load the args")
parser.add_argument('--fp16', action='store_true', help="")
parser.add_argument('--eval-eps', default=10, type=int)
parser.add_argument('--eval-timeout', default=100, type=int)
parser.add_argument('--optimizer', default="adam", type=str, help="type of the optimizer")
parser.add_argument('--scheduler', default=None, type=str, help="")
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
    # history = os.path.expanduser('~/history')
    history = "/home/cz/bs/rt_torch/history"
    # log_name = time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '/'
    if args.load_path is not None:
        log_name = os.path.basename(args.load_path) + "-" + time.strftime("%m%d-%H%M%S",time.localtime())
    else:
        time_str = time.strftime("%m%d-%H%M%S",time.localtime()) + "-"
        log_name = f"{args.text_encoder}-{args.lr}-{args.lr_t}-{args.lr_eff}-{args.depth}-{args.key_dim}-{args.alias}"
        log_name = time_str + log_name
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

    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.DEBUG)
    # console_handler.setFormatter(formatter)
    # logger.addHandler(console_handler)

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info("Start print log")
    setup_seed(args.seed)
    logger.info("seed: {}".format(args.seed))
    writer = SummaryWriter(log_path, max_queue=100)
    # filename_list = os.listdir('.')
    # expr = '\.py'
    # for filename in filename_list:
    #     if re.search(expr, filename) is not None:
    #         shutil.copyfile('./' + filename, os.path.join(log_path, filename))
    with open(os.path.join(log_path, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f)
    return logger, writer, models_path, log_path, handler

# # Load the arguments from a file
# with open('args.json', 'r') as f:
#     args_dict = json.load(f)

# # Create a new Namespace object from the saved arguments
# args = argparse.Namespace(**args_dict)

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

def main(args):
    print(args)
    device = args.device
    batch_size = args.batch_size
    train_iters = args.train_iters
    norm_clip = args.norm_clip
    test_interval = args.test_interval
    seed = args.seed
    lr_min = args.min_lr
    save_interval = args.save_interval
    max_save_num = args.max_save_num
    sub_data = args.sub_data
    text_encoder = args.text_encoder
    load_path = args.load_path
    warmup = args.warmup
    logger, writer, save_path, log_path, handler = log_init(args)
    scheduler = args.scheduler
    scheduler_t = args.scheduler_t
    depth = args.depth
    heads = args.heads
    key_dim = args.key_dim
    vocab_size = args.vocab_size
    num_actions = args.num_actions
    model_dim = args.model_dim
    seq_len = args.seq_len
    token_learner_num = args.token_learner_num
    loader_shuffle = True if args.loader_shuffle else False
    loader_bs = args.loader_bs
    loader_worker = args.loader_worker

    print('device: ', device)
    
    train_set, test_set = build_language_table_ds(split=0.9, batch_size=batch_size, seq_len=seq_len, seed=seed, dumb=False, sub_data="language_table_sim")
    train_loader = DataLoader(dataset=train_set, batch_size=loader_bs, num_workers=loader_worker, shuffle=loader_shuffle)
    test_loader = DataLoader(dataset=test_set, batch_size=loader_bs, num_workers=loader_worker, shuffle=loader_shuffle)
    model = RT1_transformer(
            num_actions=num_actions,
            vocab_size=vocab_size,
            num_layers=depth,
            heads=heads,
            key_dim=key_dim,
            feed_forward_size=model_dim,
            text_encoder=text_encoder,
            seq_len=seq_len,
            text_model_device='cpu',
            token_learner=True,
            learned_token_num=token_learner_num,
            token_learner_dropout=0.1,
            transformer_dropout=0.1,
            return_last=True,
    )
    model.to(device)
    optimizer = get_optimizer(args, model)
    if scheduler == "cosine":
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=train_iters, eta_min=lr_min)

    if warmup:
        warmup_scheduler = WarmUpScheduler(optimizer, lr_scheduler,
                                        len_loader=len(train_loader),
                                        warmup_steps=1000,
                                        warmup_start_lr=0,
                                        warmup_mode='linear',
                                        )
    if load_path is not None:
        model_path = os.path.join(load_path, 'models')
        file_models = sorted(os.listdir(model_path))
        newest_model = os.path.join(model_path, file_models[-1])
        logger.info(f"loading ckpt: {newest_model}")
        state_dict = torch.load(newest_model)
        model.load_state_dict(state_dict['model_state_dict'])
        if state_dict.get("iteration") is None:
            iteration = state_dict.get("loss_step")
        else:
            iteration = state_dict.get('iteration', 0)
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        if scheduler is not None:
            lr_scheduler.load_state_dict(state_dict['scheduler'])
    else:
        iteration = 0
    logger.info(f"\nTotal:")
    getModelSize(model, logger)
    logger.info(f"\nfilm_efficientnet_b3:")
    getModelSize(model.image_tokenizer.film_efficient_net, logger)
    logger.info(f"\nTokenLearner:")
    getModelSize(model.image_tokenizer.token_learner, logger)
    logger.info(f"\ntext embedding:")
    getModelSize(model.text_tokenizer.text_model.t5, logger)
    logger.info(f"\nTransformer:")
    getModelSize(model.transformer, logger)


    print(f"split: train-{len(train_loader)}, test-{len(test_loader)}")
    # train_loss = deque(maxlen=100)

    # avg_time = {"data_prep": 0, "inference": 0, "gradient_step": 0}

    def cyclic_iter(iter):
        while True:
            for x in iter:
                yield x
    train_data_iterator = iter(cyclic_iter(train_loader))
    test_data_iterator = iter(cyclic_iter(test_loader))
    iteration = args.iteration
    pbar = tqdm(range(args.train_iters))
    while iteration < train_iters:
        args.iteration = iteration
        if test_interval != 0 and iteration % test_interval == 0:
            model.eval()
            eval_rewards = eval_in_env(args, model, log_path, 0, iteration)
            writer.add_scalar('Train/Samples/eval_reward', float(eval_rewards), iteration * batch_size)
            writer.add_scalar('Train/Samples/eval_reward-iter', float(eval_rewards), iteration)
            logger.info(f"Iteration: {iteration}, eval_reward: {eval_rewards:.5f}")
            test(args, model, test_data_iterator, logger, writer, iteration)
            model.train()
        model.train()
        # time0 = time.time()
        data = get_batch(args, train_data_iterator)
        loss = model.forward(data)
        # time2 = time.time()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), norm_clip)
        optimizer.step()
        if warmup:
            warmup_scheduler.step()
        elif scheduler is not None:
            lr_scheduler.step()
        pbar.update(1)
        # time0 = time.time()
        # avg_time["inference"] += (time2 - time1)
        # avg_time["gradient_step"] += (time0 - time2)
        iteration += 1
        train_loss = loss.detach().cpu().item()
        # print(loss_step)
        # mean_loss = np.mean(list(train_loss))
        if scheduler is not None:
            writer.add_scalar('Train/Samples/lr', float(lr_scheduler.get_last_lr()[0]), iteration * batch_size)
            writer.add_scalar('Train/Samples/lr-iter', float(lr_scheduler.get_last_lr()[0]), iteration)
        writer.add_scalar('Train/Samples/train_loss', float(train_loss), iteration * batch_size)
        writer.add_scalar('Train/Samples/train_loss-iter', float(train_loss), iteration)
        logger.info(f"Iteration: {iteration}, Loss: {train_loss:.5f}")

            # for key, value in avg_time.items():
            #     logger.info(f"{key}: {(value / loss_step):.2f}")

        if save_interval != 0 and (iteration) % save_interval == 0:
            if scheduler is not None:
                model.save_check_point(iteration, optimizer, save_path, logger, max_save_num, lr_scheduler)
            else:
                model.save_check_point(iteration, optimizer, save_path, logger, max_save_num)

            # pbar.update(1)
    writer.close()
    logger.removeHandler(handler)
    logging.shutdown()

def test(args, model, test_data_iterator, logger, writer, iteration):
    test_loss = 0
    num_step = 0
    with torch.no_grad():
        while num_step < args.test_iters:
            data = get_batch(args, test_data_iterator)
            loss = model.forward(data)
            num_step += 1
            test_loss += loss
    test_loss /= args.test_iters
    logger.info(f"Iteration: {iteration * args.batch_size}, Test_Loss: {test_loss:.5f}")
    writer.add_scalar('Train/Samples/test_loss', float(test_loss), iteration * args.batch_size)
    writer.add_scalar('Train/Samples/test_loss-iter', float(test_loss), iteration)

if __name__ == "__main__":
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_idx
    # if args.load_path is not None:
    #     load_path = args.load_path
    #     if args.load_args:
    #         # Load the arguments from a file
    #         with open(os.path.join(args.load_path, 'args.json'), 'r') as f:
    #             args_dict = json.load(f)
    #         args_dict["load_path"] = load_path
    #         args_dict["load_args"] = args.load_args
    #         # Create a new Namespace object from the saved arguments
    #         print("loading args")
    #         args = argparse.Namespace(**args_dict)
    main(args)
