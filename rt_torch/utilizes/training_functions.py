from argparse import Namespace
import numpy as np
import torch
import deepspeed
from torch.utils.data import DataLoader



def build_distributed_dataloader(args, train_set, test_set, world_size, rank):
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, world_size, rank, seed=args.seed)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set, world_size, rank, seed=args.seed)
    train_loader = DataLoader(dataset=train_set, batch_size=args.loader_bs, num_workers=args.loader_worker, shuffle=False, sampler=train_sampler)
    test_loader = DataLoader(dataset=test_set, batch_size=args.loader_bs, num_workers=args.loader_worker, shuffle=False, sampler=test_sampler)
    return train_loader, test_loader

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