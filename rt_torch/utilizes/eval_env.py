# coding=utf-8
# Copyright 2022 The Language Tale Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example for running the Language-Table environment."""

from collections.abc import Sequence
from collections import deque
from language_table.environments import blocks
from language_table.environments import language_table
from language_table.environments.rewards import block2block
from matplotlib import pyplot as plt
from rt_torch.utilizes.utilize import nlp_inst_decoder
import os
import cv2 as cv
import torch
import numpy as np



def save_video(rgbs, path, rank, iteration, eval_eps):
    eval_eps = str(eval_eps)
    file_name = eval_eps.zfill(3) + ".mp4"
    file_name = str(rank) + "_" + file_name
    output_file = os.path.join(path, file_name)
    frame_rate = 10
    height, width, channels = rgbs[0].shape
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video_writer = cv.VideoWriter(output_file, fourcc, frame_rate, (width, height))
    for frame in rgbs:
        video_writer.write(frame)


def frame_resize(frame, fp16=False, device=None):
    resized_rgb = cv.resize(frame, (300, 168))
    # print(resized_rgb.shape)
    resized_rgb = np.pad(resized_rgb, ((66, 66), (0, 0), (0, 0))).transpose(2, 0, 1) / 255
    if fp16:
        return torch.tensor(resized_rgb, dtype=torch.half, device=device)
    else:
        return torch.tensor(resized_rgb, dtype=torch.float, device=device)



def eval_in_env(args, model, video_path, rank, iteration):
    env = language_table.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
        reward_factory=block2block.BlockToBlockReward,
        control_frequency=10.0,
    )
    with torch.no_grad():
        model.eval()
        video_path = os.path.join(video_path, "videos")
        iteration = str(iteration).zfill(6)
        video_path = os.path.join(video_path, iteration)
        os.makedirs(video_path, exist_ok=True)
        fp16 = args.fp16
        device = args.device
        videos = []
        env_obs = env.reset()
        # import pdb; pdb.set_trace()
        instruction = nlp_inst_decoder(env_obs['instruction'])
        rgb = frame_resize(env_obs['rgb'], fp16, device)
        # Take a few random actions.
        eval_eps = 0
        rewards = []
        ep_reward = 0
        ep_steps = 0
        # print(f"t5 poos: {model.text_tokenizer.text_model.t5.device}, move to {device}")
        model.text_tokenizer.text_model.t5 = model.text_tokenizer.text_model.t5.to(device)
        # model.text_tokenizer.text_model.device = device
        # print(f"t5 done: {model.text_tokenizer.text_model.t5.device}")
        if args is None:
            rgbs = deque([torch.zeros([300, 300, 3])] * 5, maxlen=6)
            inst_buffer = deque([torch.zeros(768)] * 5, maxlen=5)
            while eval_eps < 10:
                rgbs.append(rgb)
                # import pdb; pdb.set_trace()
                env_obs, reward, terminal, info = env.step(env.action_space.sample())
                ep_steps += 1
                instruction = nlp_inst_decoder(env_obs['instruction'])
                rgb = frame_resize(env_obs['rgb'])
                ep_reward += reward
                videos.append(env.render())
                if terminal or ep_steps >= 100:
                    rewards.append(ep_reward)
                    ep_reward = 0
                    rgbs = deque([torch.ones(300, 300, 3)] * 5, maxlen=6)
                    inst_buffer = deque([""] * 5, maxlen=6)
                    env_obs = env.reset()
                    instruction = nlp_inst_decoder(env_obs['instruction'])
                    rgb = frame_resize(env_obs['rgb'])
                    save_video(videos, video_path, rank, iteration, eval_eps)
                    videos = []
                    if ep_steps == 100:
                        print(f"ep {eval_eps} timeout")
                    else:
                        print(f"ep {eval_eps} done")
                    ep_steps = 0
                    eval_eps += 1
        else:
            if args.fp16:
                rgbs_tmp = [torch.zeros([3, 300, 300], dtype=torch.half, device=device)] * 5
                inst_tmp = [torch.zeros(768, dtype=torch.half, device=device)] * 5
            else:
                rgbs_tmp = [torch.zeros([3, 300, 300], dtype=torch.float, device=device)] * 5
                inst_tmp = [torch.zeros(768, dtype=torch.float, device=device)] * 5
            rgbs = deque(rgbs_tmp, maxlen=6)
            inst_buffer = deque(inst_tmp, maxlen=5)
            while eval_eps < args.eval_eps:
                # print('env step')
                rgbs.append(rgb)
                action, texts_embeddings = model.inference(torch.stack(list(rgbs)), [instruction], torch.stack(list(inst_buffer)), device)
                inst_buffer.append(texts_embeddings.squeeze(0))
                env_obs, reward, terminal, info = env.step(action)
                ep_steps += 1
                instruction = nlp_inst_decoder(env_obs['instruction'])
                rgb = frame_resize(env_obs['rgb'], fp16, device)
                ep_reward += reward
                videos.append(env.render())
                if terminal or ep_steps >= args.eval_timeout:
                    # if ep_steps >= args.eval_timeout:
                    #     rewards.append(0)
                    # else:
                    rewards.append(ep_reward)
                    ep_steps = 0
                    ep_reward = 0
                    rgbs = deque(rgbs_tmp, maxlen=6)
                    inst_buffer = deque(inst_tmp, maxlen=5)
                    env_obs = env.reset()
                    instruction = nlp_inst_decoder(env_obs['instruction'])
                    rgb = frame_resize(env_obs['rgb'], fp16, device)
                    save_video(videos, video_path, rank, iteration, eval_eps)
                    videos = []
                    eval_eps += 1

        model.text_tokenizer.text_model.t5 = model.text_tokenizer.text_model.t5.to("cpu")
        
        return np.array(rewards).mean()



if __name__ == "__main__":
    eval_in_env(None, None, video_path="/home/cz/bs/rt_torch/video", rank=0, iteration=0)

