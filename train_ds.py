from rt_torch.utilizes.train_args import parse_args 
import os
import shutil
from rt_torch.rt1.rt_vanilla import RT1_transformer
# from rt_torch.rt1.rt_xl import RT1_transformerxl
from rt_torch.dataset.dataset_npz import build_language_table_ds
import logging
import deepspeed
from rt_torch import mpu
from rt_torch.utilizes.utilize import print_rank_0, print_with_rank, getModelSize
from tqdm import tqdm
import torch.distributed as dist
from rt_torch.utilizes.eval_env import eval_in_env
from rt_torch.utilizes.train_configs import *
from rt_torch.utilizes.training_functions import *
from rt_torch.utilizes.loggings import log_init

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
    args.loader_shuffle = True if args.loader_shuffle else False
    loader_bs = args.loader_bs
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
    log_path = args.log_path
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
    rank = model_engine.global_rank
    # logger, handler = log_init(args, rank)
    # print_with_rank(f"building dataset with seed: {rank}")
    train_set, test_set = build_language_table_ds(split=0.9, batch_size=batch_size // loader_bs, seq_len=seq_len, seed=args.seed)
    train_loader, test_loader = build_distributed_dataloader(args, train_set, test_set, world_size, rank)
    if rank == 0:
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

    def cyclic_iter(iter):
        while True:
            for x in iter:
                yield x

    train_data_iterator = iter(cyclic_iter(train_loader))
    test_data_iterator = iter(cyclic_iter(test_loader))
    iteration = args.iteration
    if rank == 0:
        pbar = tqdm(range(args.train_iters))
    while iteration < args.train_iters:
        losses = train_step(args, model_engine, train_data_iterator)
        args.iteration = iteration
        if args.test_interval and iteration % args.test_interval == 0 or iteration == args.train_iters:
            local_eval_rewards = eval_in_env(args, model_engine, log_path, rank, iteration)
            local_test_loss = cal_test_loss(
                args,
                model_engine,
                test_data_iterator,
            )
            # print_with_rank(f"local test loss: {local_test_loss}")
            total_test_loss = [None for _ in range(world_size)]
            total_eval_reward = [None for _ in range(world_size)]
            dist.gather_object(
                obj=local_test_loss, 
                object_gather_list=total_test_loss if rank == 0 else None, 
                dst=0,
            )
            dist.gather_object(
                obj=local_eval_rewards, 
                object_gather_list=total_eval_reward if rank == 0 else None, 
                dst=0,
            )
            if rank == 0:
                total_loss = 0
                total_reward = 0
                for _res, _rew in zip(total_test_loss, total_eval_reward):
                    if _res is not None:
                        total_loss += _res
                    if _rew is not None:
                        total_reward += _rew
                        # print(f"collecting loss from rank {idx}")
                total_loss /= world_size
                total_reward /= world_size
                # logger.info(f"Iteration: {model_engine.global_samples}, Eval Reward: {total_reward:.5f}")
                if rank == 0 and model_engine.monitor.enabled:
                    summary_events = [
                        (f'Train/Samples/eval_reward', float(total_reward), model_engine.global_samples), 
                        (f'Train/Samples/test_loss', float(total_loss), model_engine.global_samples), 
                        ]
                    model_engine.monitor.write_events(summary_events)
                # logger.info(f"Iteration: {model_engine.global_samples}, Test Loss: {total_loss:.5f}")
                video_path = os.path.join(log_path, "videos")
                video_dirs = os.listdir(video_path)
                if len(video_dirs) > 10:
                    video_dirs = sorted(video_dirs)
                    num = len(video_dirs) - 10
                    for i in range(num):
                        shutil.rmtree(os.path.join(video_path, video_dirs[i]))
        iteration += 1
        if rank == 0:
            # loss = sum(losses) / len(losses)
            # writer.add_scalar('Train Loss', float(loss), iteration)
            # logger.info(f"Iteration: {model_engine.global_samples}, Train Loss: {loss:.5f}")
            # current_lr = opt_param_scheduler.get_lr()
            # writer.add_scalar('Learning Rate', float(current_lr), iteration)
            # writer.flush()
            pbar.update(1)
        if args.save_interval and iteration % args.save_interval == 0 or iteration == args.train_iters:
            # print_with_rank(f"saving ckpt: {log_path}, iteration: {iteration}")
            save_checkpoint(args, log_path, iteration, model_engine)
            # print_with_rank(f"saving ckpt done, path: {log_path}, iteration: {iteration}")
    # if rank == 0:
        # writer.close()
        # logger.removeHandler(handler)
        # logging.shutdown()


if __name__ == "__main__":
    args = parse_args()
    main(args)
