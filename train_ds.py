from rt_torch.utilizes.train_args import parse_args 
from argparse import Namespace
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
from rt_torch.rt1.rt_vanilla import RT1_transformer
# from rt_torch.rt1.rt_xl import RT1_transformerxl
from rt_torch.dataset.dataset_npz import build_language_table_ds
from torch.utils.data import DataLoader
import time
from torch.optim import Adam, SGD, AdamW
from itertools import islice
from rt_torch.utilizes.optimizer_param_scheduler import OptimizerParamScheduler
import deepspeed
from rt_torch import mpu
from rt_torch.utilizes.utilize import print_rank_0, print_with_rank
from tqdm import tqdm
import torch.distributed as dist


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # tf.random.set_seed(seed)


def log_init(args=None, rank=0):
    history = "/home/cz/bs/rt_torch/history"
    # history = os.path.expanduser('~/history')
    # log_name = time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '/'
    if args.load_dir is not None:
        log_name = os.path.basename(args.load_dir) + "_" + args.exp_name
    else:
        log_name = args.exp_name + "_" + args.text_encoder + "_" + str(args.lr_t) + "_" + str(args.lr_eff) + "_" + args.alias
    log_path = os.path.join(history, log_name)
    models_path = os.path.join(log_path, 'models')
    # print(history)
    # print(log_path)
    # print(models_path)
    os.makedirs(models_path, exist_ok=True)
    if rank == args.master_rank:
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
        # with open(os.path.join(log_path, 'args.json'), 'w') as f:
        #     json.dump(args.__dict__, f)
        return logger, writer, models_path, log_path, handler
    else:
        return None, None, models_path, log_path, None

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
    batch_size = args.batch_size
    text_encoder = args.text_encoder
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
    master_rank = args.master_rank
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
    optimizer = get_optimizer(args, model)
    opt_param_scheduler = get_optimizer_param_scheduler(args, optimizer)
    deepspeed.init_distributed(distributed_port=args.deepspeed_port)
    # mpu.initialize_model_parallel()
    # if mpu.get_data_parallel_rank() == 0:
    #     print(
    #         " > number of parameters on (tensor, pipeline) "
    #         "model parallel rank ({}, {}): {}".format(
    #             mpu.get_tensor_model_parallel_rank(),
    #             mpu.get_pipeline_model_parallel_rank(),
    #             sum([p.nelement() for p in model.parameters()]),
    #         ),
    #         flush=True,
    #     )
    print_rank_0(" ============= MPU_INIT ==============")
    model_engine, _, _, _ = deepspeed.initialize(
        args,
        model,
        model_parameters=model.parameters(),
        # mpu=mpu,
        optimizer=optimizer,
        lr_scheduler=opt_param_scheduler,
    )
    print_rank_0(" ============= DS_INIT ==============")
    assert args.fp16 == model_engine.fp16_enabled()
    if args.load_dir:
        load_dir, client_state = model_engine.load_checkpoint(
            args.load_dir,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
        )
        args.iteration = client_state["iteration"]
        print_with_rank(
            "load at iter: {}, lr: {}".format(
                client_state["iteration"], model_engine.client_lr_scheduler.get_lr()
            )
        )
    args.device = model_engine.device
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    print_with_rank(f"building dataset with seed: {rank}")
    train_set, test_set = build_language_table_ds(split=0.9, batch_size=batch_size // loader_bs, rgb_list=True, seq_len=seq_len, seed=rank)
    train_loader = DataLoader(dataset=train_set, batch_size=loader_bs, num_workers=loader_worker, shuffle=loader_shuffle)
    test_loader = DataLoader(dataset=test_set, batch_size=loader_bs, num_workers=loader_worker, shuffle=loader_shuffle)
    if rank == master_rank:
        print(args)
        print(f"split: train-{len(train_loader)}, test-{len(test_loader)}")
        print(train_set.mode)
        print(f"ds_list: {train_set.ds_list}")
        print(f"length: {train_set.len}")
        print(f"sub_ds_len: {train_set.sub_ds_len}")
        print(f"bucket: {train_set.bucket}")
        print(test_set.mode)
        print(f"ds_list: {test_set.ds_list}")
        print(f"length: {test_set.len}")
        print(f"sub_ds_len: {test_set.sub_ds_len}")
        print(f"bucket: {test_set.bucket}")
    logger, writer, save_path, log_path, handler = log_init(args, rank)

    def cyclic_iter(iter):
        while True:
            for x in iter:
                yield x

    train_data_iterator = iter(cyclic_iter(train_loader))
    test_data_iterator = iter(cyclic_iter(test_loader))
    iteration = args.iteration
    if rank == master_rank:
        pbar = tqdm(range(args.train_iters))
    while iteration < args.train_iters:
        losses = train_step(args, model_engine, train_data_iterator)
        args.iteration = iteration
        iteration += 1
        if rank == master_rank:
            loss = sum(losses) / len(losses)
            writer.add_scalar('Train Loss', float(loss), iteration)
            logger.info(f"Iteration: {iteration}, Loss: {loss:.5f}")
            current_lr = opt_param_scheduler.get_lr()
            writer.add_scalar('Learning Rate', float(current_lr), iteration)
            writer.flush()
            pbar.update(1)
        if args.test_interval and iteration % args.test_interval == 0 or iteration == args.train_iters:
            local_test_loss = cal_test_loss(
                args,
                model_engine,
                test_data_iterator,
            )
            # print_with_rank(f"local test loss: {local_test_loss}")
            total_test_loss = [None for _ in range(world_size)]
            dist.gather_object(
                obj=local_test_loss, 
                object_gather_list=total_test_loss if rank == master_rank else None, 
                dst=master_rank,
            )
            if rank == master_rank:
                total_loss = 0
                for idx, _res in enumerate(total_test_loss):
                    if _res is not None:
                        total_loss += _res
                        # print(f"collecting loss from rank {idx}")
                total_loss /= world_size
                logger.info(f"Iteration: {iteration}, Test Loss: {total_loss:.5f}")
                writer.add_scalar('Test Loss', float(total_loss), iteration)
        if args.save_interval and iteration % args.save_interval == 0 or iteration == args.train_iters:
            # print_with_rank(f"saving ckpt: {save_path}, iteration: {iteration}")
            save_checkpoint(args, save_path, iteration, model_engine)
            # print_with_rank(f"saving ckpt done, path: {save_path}, iteration: {iteration}")
    if rank == master_rank:
        writer.close()
        logger.removeHandler(handler)
        logging.shutdown()

def save_checkpoint(args, save_path, iteration, model_engine: deepspeed.DeepSpeedEngine):
    client_state = {}
    client_state["args"] = args
    client_state["iteration"] = iteration
    model_engine.save_checkpoint(save_path, client_state=client_state, tag="latest_model")

def train_step(args, model, data_iterator):
    model.train()
    losses = forward_and_backward_step(
        args, model, data_iterator, do_backward=True
    )
    return losses

def get_batch(args, data_iterator):
    data = next(data_iterator)
    device = args.device
    rgbs, instructions, actions, split_idx = data
    if args.fp16:
        rgbs = rgbs.to(dtype=torch.half)
        instructions = instructions.to(dtype=torch.half)
    if len(instructions.shape) == 3:
        # print_rank_0(rgbs.shape)
        # print_rank_0(instructions.shape)
        # print_rank_0(actions.shape)
        # print_rank_0(split_idx.shape)
        rgbs = rgbs.view(-1, *rgbs.shape[2:])
        instructions = instructions.view(-1, *instructions.shape[2:])
        actions = actions.view(-1, *actions.shape[2:])
        split_idx = split_idx.view(-1, *split_idx.shape[2:])
        # print_rank_0(rgbs.shape)
        # print_rank_0(instructions.shape)
        # print_rank_0(actions.shape)
        # print_rank_0(split_idx.shape)
    rgbs = rgbs.to(device)
    actions = actions.to(device)
    instructions = instructions.to(device)
    return [rgbs, instructions, actions, split_idx]
    
    

def forward_and_backward_step(
    args, model, data_iterator, do_backward=True
):
    loss_list = []
    ga_steps = model.gradient_accumulation_steps()
    for i in range(ga_steps):
        data = get_batch(args, data_iterator)
        loss = model(data)

        if do_backward:
            model.backward(loss)
            model.step()
        loss_list.append(loss)
    return loss_list

def cal_loss(args, model, data):
    device = args.device
    rgbs, instructions, actions, split_idx = data
    # print_rank_0(rgbs.shape)
    # print_rank_0(instructions.shape)
    # print_rank_0(actions.shape)
    # print_rank_0(split_idx.shape)
    if args.fp16:
        rgbs = rgbs.to(dtype=torch.half)
        instructions = instructions.to(dtype=torch.half)
    if len(instructions.shape) == 3:
        rgbs = rgbs.squeeze(0)
        instructions = instructions.squeeze(0)
        actions = actions.squeeze(0)
        split_idx = split_idx.squeeze(0)
    rgbs = rgbs.to(device)
    actions = actions.to(device)
    actions_discretes = model.action_tokenizer.discretize(actions)
    if not isinstance(instructions, list):
        instructions = instructions.to(device)
    predicts = model.forward(video=[rgbs, split_idx], texts_embeddings=instructions)
    predicts = predicts.permute(0, 2, 1)
    loss = model.criterion(predicts, actions_discretes)
    return loss

def cal_test_loss(
    args: Namespace,
    model: torch.nn.Module,
    test_data_iterator,
):
    model.eval()

    with torch.no_grad():
        iteration = 0
        total_loss = 0

        while iteration < args.test_iters:
            iteration += 1
            # get total valid loss
            loss_list = forward_and_backward_step(
                args,
                model,
                test_data_iterator,
                do_backward=False,
            )
            loss_list = [l.cpu().item() for l in loss_list]
            total_loss = total_loss + np.mean(loss_list)
        # XXX: be careful with this when use parallel
        total_loss /= args.test_iters
    return total_loss


def get_optimizer_param_scheduler(args, optimizer):
    """Build the learning rate scheduler."""

    # Iteration-based training.
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters
    lr_decay_steps = args.lr_decay_iters * args.global_batch_size
    wd_incr_steps = args.train_iters * args.global_batch_size
    # if args.lr_warmup_fraction is not None:
    #     lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
    # else:
    #     lr_warmup_steps = args.lr_warmup_iters * args.global_batch_size

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        max_lr=args.lr,
        min_lr=args.min_lr,
        lr_warmup_steps=0,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=args.lr_decay_style,
        start_wd=args.start_weight_decay,
        end_wd=args.end_weight_decay,
        wd_incr_steps=wd_incr_steps,
        wd_incr_style=args.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=args.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=args.override_opt_param_scheduler,
    )

    return opt_param_scheduler


def get_paramaters(args, model: nn.Module):
    lr_t = args.lr_t
    lr_efficientnet = args.lr_eff
    pretrained = set()
    unpretrained = set()
    for module_name, module in model.named_modules():
        for parameter_name, paramater in module.named_parameters():
            full_parameter_name = "%s.%s" % (module_name, parameter_name) if module_name else parameter_name  # full param name
            if full_parameter_name.endswith("bias") or full_parameter_name.endswith("weight"):
                if "film_efficient_net" in full_parameter_name:
                    if "conditioning_layers" in full_parameter_name:
                        unpretrained.add(full_parameter_name)
                    else:
                        pretrained.add(full_parameter_name)
                else:
                    unpretrained.add(full_parameter_name)
            else:
                import pdb; pdb.set_trace()
    param_dict = {parameter_name: paramater for parameter_name, paramater in model.named_parameters()}
    pretrained = param_dict.keys() & pretrained
    unpretrained = param_dict.keys() & unpretrained
    inter_params = pretrained & unpretrained
    union_params = pretrained | unpretrained
    assert (
        len(inter_params) == 0
    ), "parameters %s made it into both pretrained/unpretrained sets!" % (str(inter_params),)
    assert len(param_dict.keys() ^ union_params) == 0, (
        "parameters %s were not separated into either pretrained/unpretrained set!"
        % (str(param_dict.keys() - union_params),)
    )
    optim_groups = [
        {
            "params": [
                param_dict[pn] for pn in sorted(list(pretrained)) if pn in param_dict
            ],
            "lr": lr_efficientnet,
        },
        {"params": [param_dict[pn] for pn in sorted(list(unpretrained))], "lr": lr_t,},
    ]
    return optim_groups
        

def get_optimizer(args, model):
    # Base optimizer.
    paramaters = get_paramaters(args, model)
    if args.optimizer == "adam":
        optimizer = Adam(
            paramaters,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
        )
    elif args.optimizer == "adamw":
        optimizer = AdamW(
            paramaters,
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
        )
    elif args.optimizer == "sgd":
        optimizer = SGD(
            paramaters,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.sgd_momentum,
        )
    else:
        raise Exception("{} optimizer is not supported.".format(args.optimizer))

    return optimizer


if __name__ == "__main__":
    args = parse_args()
    # if args.load_dir is not None:
    #     load_dir = args.load_dir
    #     if args.load_args:
    #         # Load the arguments from a file
    #         with open(os.path.join(args.load_dir, 'args.json'), 'r') as f:
    #             args_dict = json.load(f)
    #         args_dict["load_dir"] = load_dir
    #         args_dict["load_args"] = args.load_args
    #         # Create a new Namespace object from the saved arguments
    #         print("loading args")
    #         args = Namespace(**args_dict)
    main(args)
