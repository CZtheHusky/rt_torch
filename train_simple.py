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
from robotic_transformer import simple_rt, film_efficientnet_b3
from robotic_transformer.dataset import language_table_dataset
from robotic_transformer.dataset_subproc import SubProcLTDataset
from robotic_transformer.dataset_torch_wrapper import language_table_dataset_py
from robotic_transformer.dataset_npz import language_table_dataset_npz
from robotic_transformer.helper_classes import ActionTokenizer
from collections import defaultdict
from torch.utils.data import DataLoader
from robotic_transformer.helper_classes import InstEmbeddingBuffer
from operator import itemgetter 
# import tensorflow as tf
import time
from collections import deque
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler_pytorch import WarmUpScheduler


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
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--load_path', default=None, type=str, help="checkpoint path to load")
parser.add_argument('--load_args', action='store_true', help="load the args")
parser.add_argument('--optimizer', default="adam", type=str, help="type of the optimizer")
parser.add_argument('--warmup', action='store_true', help="using warmup scheduler")
parser.add_argument('--adam_beta1', default=0.9, type=float, help="")
parser.add_argument('--adam_beta2', default=0.99, type=float, help="")
parser.add_argument('--adam_eps', default=1e-8, type=float, help="")
parser.add_argument('--weight_decay', default=1e-2, type=float, help="")
parser.add_argument('--sgd_momentum', default=0, type=float, help="")




# python /home/cz/bs/robotic-transformer-pytorch/robotic_transformer/train.py --device_idx 7 --lr 0.0001
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
    num_epoch = args.num_epoch
    norm_clip = args.norm_clip
    test_interval = args.test_interval
    seed = args.seed
    alias = args.alias
    save_interval = args.save_interval
    max_save_num = args.max_save_num
    sub_data = args.sub_data
    text_encoder = args.text_encoder
    load_path = args.load_path
    warmup = args.warmup
    logger, writer, save_path, log_path, handler = log_init(args)
    print('device: ', device)
    
    # train_loader = language_table_dataset(dataset_type="train", sub_data=sub_data, batch_size=batch_size, weight=[1, 1, 0, 0, 0, 0, 0, 0, 0])
    # test_loader = language_table_dataset(dataset_type="test", sub_data=sub_data, batch_size=batch_size, weight=[1, 1, 0, 0, 0, 0, 0, 0, 0])
    train_loader = language_table_dataset_npz(
        mode="train", 
        ds_type=sub_data, 
        batch_size=batch_size, 
        weight=[1, 1, 0, 0, 0, 0, 0, 0, 0]
        )
    test_loader = language_table_dataset_npz(
        mode="test", 
        ds_type=sub_data, 
        batch_size=batch_size, 
        weight=[1, 1, 0, 0, 0, 0, 0, 0, 0]
        )
    # train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True)
    # test_loader = DataLoader(dataset=test_set, batch_size=1)
    # train_loader._create_process()
    # test_loader._create_process()
    effnet = film_efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    model = simple_rt(efficient=effnet, num_actions=2)
    model.to(device)

    optimizer = get_optimizer(args, model)
    if load_path is not None:
        model_path = os.path.join(load_path, 'models')
        file_models = sorted(os.listdir(model_path))
        newest_model = os.path.join(model_path, file_models[-1])
        logger.info(f"loading ckpt: {newest_model}")
        state_dict = torch.load(newest_model)
        model.load_state_dict(state_dict['model_state_dict'])
        epoch_s = state_dict['epoch']
        loss_step = state_dict['loss_step']
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    else:
        loss_step = 0
        epoch_s = 0
    action_tokenizer = ActionTokenizer(num_action_bin=256, action_max=0.1, action_min=-0.1)
    criterion = nn.CrossEntropyLoss()
    # embedding_buffer = InstEmbeddingBuffer()
    print(f"split: train-{len(train_loader)}, test-{len(test_loader)}")
    train_loss = deque(maxlen=100)
    if warmup:
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6)
        warmup_scheduler = WarmUpScheduler(optimizer, lr_scheduler,
                                        len_loader=len(train_loader),
                                        warmup_steps=1000,
                                        warmup_start_lr=1e-3,
                                        warmup_mode='linear',
                                        )

    # avg_time = {"data_prep": 0, "move_data": 0, "inference": 0, "gradient_step": 0}

    for epoch in range(epoch_s, num_epoch):
        model.train()

        # times = deque(maxlen=4)
        # time0 = time.time()

        for ep_idx in tqdm(range(len(train_loader))):
            rgbs, instructions, actions = train_loader.__get_unstacked_item__(ep_idx)
            # time1 = time.time()
            rgbs = rgbs.to(device)
            actions = actions.to(device)
            actions_discretes = action_tokenizer.discretize(actions)
            # time2 = time.time()

            predicts = model(rgbs, instructions)
            # import pdb
            # pdb.set_trace()
            # time3 = time.time()
            # avg_time["data_prep"] += (time1 - time0)

            # import pdb
            # pdb.set_trace()
            # predicts_prob = nn.functional.softmax(predicts, dim=-1)
            # test_loss = criterion(predicts_prob, actions_discretes.float())
            loss = criterion(predicts, actions_discretes.float())
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), norm_clip)
            optimizer.step()
            if warmup:
                warmup_scheduler.step()

            # time0 = time.time()
            # avg_time["move_data"] += (time2 - time1)
            # avg_time["inference"] += (time3 - time2)
            # avg_time["gradient_step"] += (time0 - time3)

            loss_step += 1
            train_loss.append(loss.detach().cpu().numpy())
            # print(loss_step)
            if loss_step % 100 == 0:
                mean_loss = np.mean(list(train_loss))
                writer.add_scalar('Train_Loss', float(mean_loss), loss_step)
                logger.info(f"EP: {epoch}, Loss step: {loss_step}, Loss: {mean_loss:.5f}")

                # for key, value in avg_time.items():
                #     logger.info(f"{key}: {(value / loss_step):.2f}")

            if save_interval != 0 and (loss_step) % save_interval == 0:
                epoch_str = str(epoch)
                epoch_str = epoch_str.zfill(4)
                file_name = str(loss_step)
                file_name = epoch_str + "_" + file_name.zfill(10)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "loss_step": loss_step,
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                    os.path.join(save_path, file_name + '.pt')
                )
                logger.info(f"check point saved")
                saved_list = os.listdir(save_path)
                if len(saved_list) > max_save_num:
                    sorted(saved_list)
                    oldest = os.path.join(save_path, saved_list[0])
                    logger.info(f"oldest check point removed, path: {oldest}")
                    os.remove(oldest)

                test_loss = 0
                num_step = 0
                model.eval()
                with torch.no_grad():
                    indexes = np.random.randint(0, len(test_loader), size=1000)
                    for idx in indexes:
                        rgbs, instructions, actions = test_loader.__get_unstacked_item__(idx)
                        rgbs = rgbs.to(device)
                        actions = actions.to(device)
                        actions_discretes = action_tokenizer.discretize(actions)
                        predicts = model(rgbs, instructions)
                        loss = criterion(predicts, actions_discretes.float())
                        num_step += 1
                        test_loss += loss
                        if num_step % 100 == 0:
                            print(f"Test step: {num_step}, Test_Loss: {test_loss / num_step:.5f}")
                test_loss /= num_step
                logger.info(f"EP: {epoch}, Loss step: {loss_step}, Test_Loss: {test_loss:.5f}")
                writer.add_scalar('Test_Loss', float(test_loss), loss_step)
                model.train()
        writer.flush()
    writer.close()
    logger.removeHandler(handler)
    logging.shutdown()

def get_optimizer(args, model):
    # Base optimizer.
    if args.optimizer == "adam":
        optimizer = Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
        )
    elif args.optimizer == "adamw":
        optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
        )
    elif args.optimizer == "sgd":
        optimizer = SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.sgd_momentum,
        )
    else:
        raise Exception("{} optimizer is not supported.".format(args.optimizer))

    return optimizer


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
