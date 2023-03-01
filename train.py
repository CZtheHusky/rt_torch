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
from rt_torch.dataset.dataset_npz import language_table_dataset_npz
from rt_torch.tokenizers.action_tokenizer import ActionTokenizer
from collections import defaultdict
from torch.utils.data import DataLoader
import time
from collections import deque
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler_pytorch import WarmUpScheduler
import tensorflow as tf

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1."

parser = argparse.ArgumentParser()
parser.add_argument('--device', default="cuda", type=str)
parser.add_argument('--device_idx', default="7", type=str)
parser.add_argument('--text_encoder', default="t5", type=str)
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--num_epoch', default=20, type=int, help='num_epoch')
parser.add_argument('--norm_clip', default=40, type=int, help='clip norm')
parser.add_argument('--test_interval', default=1000, type=int, help='test_interval')
parser.add_argument('--seed', default=100, type=int, help='seed of numpy, tensorflow and pytorch')
parser.add_argument('--save_interval', default=10000, type=int)
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
parser.add_argument('--lr_min', default=1e-5, type=float)
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
    # history = os.path.expanduser('~/history')
    history = "/home/cz/bs/rt_torch/history"
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
    
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.DEBUG)
    # console_handler.setFormatter(formatter)
    # logger.addHandler(console_handler)
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info("Start print log")
    setup_seed(args.seed)
    logger.info("seed: {}".format(args.seed))
    writer = SummaryWriter(log_path)
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
    num_epoch = args.num_epoch
    norm_clip = args.norm_clip
    test_interval = args.test_interval
    seed = args.seed
    alias = args.alias
    lr_min = args.lr_min
    save_interval = args.save_interval
    max_save_num = args.max_save_num
    sub_data = args.sub_data
    text_encoder = args.text_encoder
    load_path = args.load_path
    warmup = args.warmup
    logger, writer, save_path, log_path, handler = log_init(args)
    scheduler = args.scheduler
    depth = args.depth
    heads = args.heads
    key_dim = args.key_dim
    vocab_size = args.vocab_size
    num_actions = args.num_actions
    model_dim = args.model_dim
    seq_len = args.seq_len
    token_learner_num = args.token_learner_num
    print('device: ', device)
    
    # train_loader = language_table_dataset(dataset_type="train", sub_data=sub_data, batch_size=batch_size, weight=[1, 1, 0, 0, 0, 0, 0, 0, 0])
    # test_loader = language_table_dataset(dataset_type="test", sub_data=sub_data, batch_size=batch_size, weight=[1, 1, 0, 0, 0, 0, 0, 0, 0])
    train_set = language_table_dataset_npz(
        mode="train", 
        ds_type=sub_data, 
        batch_size=batch_size, 
        stack=False,
        rgb_list=True,
        seq_len=seq_len,
        )
    test_set = language_table_dataset_npz(
        mode="test", 
        ds_type=sub_data, 
        batch_size=batch_size, 
        stack=False,
        rgb_list=True,
        seq_len=seq_len,
        )
    train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True, num_workers=8)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=True, num_workers=8)
    # train_loader._create_process()
    # test_loader._create_process()
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
    )
    model.to(device)
    optimizer = get_optimizer(args, model)
    if scheduler == "cosine":
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=lr_min)

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
        epoch_s = state_dict['epoch']
        loss_step = state_dict['loss_step']
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        if scheduler is not None:
            lr_scheduler.load_state_dict(state_dict['scheduler'])
        loss_step %= len(train_loader)
        for v, weight in zip(train_loader.ds_stats.values(), train_loader.weight):
            v["current_idx"] = int(loss_step * weight)
        test_step = state_dict['test_step']
    else:
        loss_step = 0
        epoch_s = 0
        test_step = 0
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

    # embedding_buffer = InstEmbeddingBuffer()
    print(f"split: train-{len(train_loader)}, test-{len(test_loader)}")
    train_loss = deque(maxlen=100)

    # avg_time = {"data_prep": 0, "inference": 0, "gradient_step": 0}

    for epoch in range(epoch_s, num_epoch):
        model.train()
        
        # time0 = time.time()

        for data in tqdm(train_loader):

            # time1 = time.time()
            # avg_time["data_prep"] += (time1 - time0)

            loss = model.cal_loss(data, device)

            # time2 = time.time()
            # avg_time["inference"] += (time2 - time1)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), norm_clip)
            optimizer.step()
            if warmup:
                warmup_scheduler.step()
            elif scheduler is not None:
                lr_scheduler.step()

            # time0 = time.time()
            # avg_time["gradient_step"] += (time0 - time2)
            
            loss_step += 1
            train_loss.append(loss.detach().cpu().numpy())
            # print(loss_step)
            if loss_step % 100 == 0:
                mean_loss = np.mean(list(train_loss))
                writer.add_scalar('Learning Rate', float(optimizer.param_groups[0]['lr']), loss_step)
                writer.add_scalar('Train_Loss', float(mean_loss), loss_step)
                logger.info(f"EP: {epoch}, Loss step: {loss_step}, Loss: {mean_loss:.5f}")

                # for key, value in avg_time.items():
                #     logger.info(f"{key}: {(value / loss_step):.2f}")

            if save_interval != 0 and (loss_step) % save_interval == 0:
                epoch_str = str(epoch)
                epoch_str = epoch_str.zfill(4)
                file_name = str(loss_step)
                file_name = epoch_str + "_" + file_name.zfill(10)
                dict2save = {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "loss_step": loss_step,
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                if scheduler is not None:
                    dict2save["scheduler"] = lr_scheduler.state_dict()
                torch.save(dict2save,
                    os.path.join(save_path, file_name + '.pt')
                )
                logger.info(f"check point saved")
                saved_list = os.listdir(save_path)
                if len(saved_list) > max_save_num:
                    saved_list = sorted(saved_list)
                    oldest = os.path.join(save_path, saved_list[0])
                    logger.info(f"oldest check point removed, path: {oldest}")
                    os.remove(oldest)

                test_loss = 0
                num_step = 0
                model.eval()
                with torch.no_grad():
                    indexes = np.random.randint(0, len(test_loader), size=1000)
                    for idx in indexes:
                        data = test_loader.__getitem__(idx)
                        loss = model.cal_loss(data, device)
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
