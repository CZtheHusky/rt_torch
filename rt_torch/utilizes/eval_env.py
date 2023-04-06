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
import shutil
from torchvision import transforms
from PIL import Image
# helpers



def frame_preprocess(transorm, img, fp16, device):
    img = transorm(Image.fromarray(img)).detach()
    if fp16:
        img = torch.tensor(img, dtype=torch.half)
    img = img.to(device)
    return img

    


def save_video(rgbs, path, rank, ep_reward, eval_eps):
    eval_eps = str(eval_eps)
    file_name = f"{rank}_{eval_eps.zfill(3)}_Rew_{ep_reward}.mp4"
    output_file = os.path.join(path, file_name)
    frame_rate = 5
    height, width, channels = rgbs[0].shape
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video_writer = cv.VideoWriter(output_file, fourcc, frame_rate, (width, height))
    for frame in rgbs:
        video_writer.write(frame)





def eval_in_env(args, model, video_path, rank, iteration, text_embed_dim, action_tokenizer):
    env = language_table.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
        reward_factory=block2block.BlockToBlockReward,
        control_frequency=5,
        # seed=0,
    )
    # import pdb; pdb.set_trace()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if args.text_encoder == "t5":
        text_embed_dim = 768
    else:
        text_embed_dim = 512
    with torch.no_grad():
        root_path = os.path.join(video_path, "videos")
        iteration = str(iteration).zfill(8)
        video_path = os.path.join(root_path, iteration)
        os.makedirs(video_path, exist_ok=True)
        videos = []
        rewards = []
        ep_length = []
        env_obs = env.reset()
        # import pdb; pdb.set_trace()
        instruction = nlp_inst_decoder(env_obs['instruction'])
        # Take a few random actions.
        eval_eps = 0
        ep_reward = 0
        ep_steps = 0
        if args is None:
            rgb = transform(env_obs['rgb'])
            rgb_video = []
            rgb_video.append(env_obs['rgb'])
            videos.append(env.render())
            rgbs = deque([torch.zeros([300, 300, 3])] * 5, maxlen=args.seq_len)
            inst_buffer = deque([torch.zeros(text_embed_dim)] * 5, maxlen=args.seq_len - 1)
            while eval_eps < 10:
                rgbs.append(rgb)
                # import pdb; pdb.set_trace()
                env_obs, reward, terminal, info = env.step(env.action_space.sample())
                ep_steps += 1
                instruction = nlp_inst_decoder(env_obs['instruction'])
                rgb_video.append(env_obs['rgb'])
                rgb = transform(env_obs['rgb'])
                ep_reward += reward
                videos.append(env.render())
                if terminal or ep_steps >= 100:
                    rewards.append(ep_reward)
                    ep_length.append(ep_steps)
                    save_video(videos, video_path, rank, ep_reward, eval_eps)
                    save_video(rgb_video, video_path, rank, ep_reward, eval_eps + 10)
                    ep_reward = 0
                    rgbs = deque([torch.ones(300, 300, 3)] * 5, maxlen=args.seq_len)
                    inst_buffer = deque([""] * 5, maxlen=args.seq_len - 1)
                    env_obs = env.reset()
                    instruction = nlp_inst_decoder(env_obs['instruction'])
                    rgb = transform(env_obs['rgb'])
                    videos = []
                    # if ep_steps == 100:
                    #     print(f"ep {eval_eps} timeout")
                    # else:
                    #     print(f"ep {eval_eps} done")
                    ep_steps = 0
                    eval_eps += 1
        else:
            fp16 = args.fp16
            device = args.device
            # import pdb; pdb.set_trace()
            rgb = frame_preprocess(transform, env_obs['rgb'], fp16, device)
            videos.append(env.render()[:,:,::-1])
            model.eval()
            if args.fp16:
                rgbs_tmp = [torch.zeros(rgb.shape, dtype=torch.half, device=device)] * 5
                inst_tmp = [torch.zeros(text_embed_dim, dtype=torch.half, device=device)] * 5
            else:
                rgbs_tmp = [torch.zeros(rgb.shape, dtype=torch.float, device=device)] * 5
                inst_tmp = [torch.zeros(text_embed_dim, dtype=torch.float, device=device)] * 5
            rgbs = deque(rgbs_tmp, maxlen=args.seq_len)
            inst_buffer = deque(inst_tmp, maxlen=args.seq_len - 1)
            while eval_eps < args.eval_eps:
                # print('env step')
                rgbs.append(rgb)
                out, texts_embeddings = model.inference(torch.stack(list(rgbs)), [instruction], torch.stack(list(inst_buffer)), device)
                # import pdb; pdb.set_trace()
                action = action_tokenizer.discrete2Scalar(out)
                inst_buffer.append(texts_embeddings.squeeze(0))
                env_obs, reward, terminal, info = env.step(action)
                ep_steps += 1
                instruction = nlp_inst_decoder(env_obs['instruction'])
                rgb = frame_preprocess(transform, env_obs['rgb'], fp16, device)
                ep_reward += reward
                videos.append(env.render()[:,:,::-1])
                if terminal or ep_steps >= args.eval_timeout:
                    # if ep_steps >= args.eval_timeout:
                    #     rewards.append(0)
                    # else:
                    # print(f"==========rank {rank} episode done==========")
                    ep_length.append(ep_steps)
                    rewards.append(ep_reward)
                    save_video(videos, video_path, rank, ep_reward, eval_eps)
                    ep_steps = 0
                    ep_reward = 0
                    rgbs = deque(rgbs_tmp, maxlen=args.seq_len)
                    inst_buffer = deque(inst_tmp, maxlen=args.seq_len - 1)
                    env_obs = env.reset()
                    instruction = nlp_inst_decoder(env_obs['instruction'])
                    rgb = frame_preprocess(transform, env_obs['rgb'], fp16, device)
                    videos = []
                    videos.append(env.render()[:,:,::-1])
                    eval_eps += 1
        if rank == 0:
            video_dirs = os.listdir(root_path)
            if len(video_dirs) > 10:
                video_dirs = sorted(video_dirs)
                num = len(video_dirs) - 10
                for i in range(num):
                    shutil.rmtree(os.path.join(root_path, video_dirs[i]))
        return np.array([np.array(rewards).mean(), np.array(ep_length).mean()])



if __name__ == "__main__":
    eval_in_env(None, None, video_path="/home/cz/bs/rt_torch", rank=0, iteration=0)

