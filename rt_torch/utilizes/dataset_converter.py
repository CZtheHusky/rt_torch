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

"""Example for loading the Language-Table dataset.

Language-Table data is in the [RLDS](https://github.com/google-research/rlds)
format.
See the [RLDS Tutorial](https://colab.research.google.com/github/
google-research/rlds/blob/main/rlds/examples/rlds_tutorial.ipynb)
for more details on how to use RLDS datasets.
"""

import numpy as np
import tensorflow_datasets as tfds
import os
from tqdm import tqdm
import argparse
import tensorflow as tf
import cv2 as cv
import os
import torch
from rt_torch.tokenizers.text_tokenizer import UniversalSentenceEncoder
from torchvision import transforms
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = ''

parser = argparse.ArgumentParser()
parser.add_argument('--ds_name', default="", type=str)

dataset_paths = {
    'language_table': '/raid/robotics_data/language_table',
    'language_table_sim': '/raid/robotics_data/language_table_sim',
    'language_table_blocktoblock_sim': '/raid/robotics_data/language_table_blocktoblock_sim',
    'language_table_blocktoblock_4block_sim': '/raid/robotics_data/language_table_blocktoblock_4block_sim',
    'language_table_blocktoblock_oracle_sim': '/raid/robotics_data/language_table_blocktoblock_oracle_sim',
    'language_table_blocktoblockrelative_oracle_sim': '/raid/robotics_data/language_table_blocktoblockrelative_oracle_sim',
    'language_table_blocktoabsolute_oracle_sim': '/raid/robotics_data/language_table_blocktoabsolute_oracle_sim',
    'language_table_blocktorelative_oracle_sim': '/raid/robotics_data/language_table_blocktorelative_oracle_sim',
    'language_table_separate_oracle_sim': '/raid/robotics_data/language_table_separate_oracle_sim',
}

def nlp_inst_decoder(bytes_list):
    non_zero = bytes_list[np.where(bytes_list != 0)]
    if non_zero.shape[0] == 0:
        return ''
    else:
        bytes_list = bytes(non_zero.tolist())
    return bytes_list.decode('utf-8')


def quantileBinning(all_actions=None,
                    path=None,
                    num_bins=256):
    quantiles = torch.linspace(0, 1, num_bins + 1)
    # import pdb; pdb.set_trace()
    all_actions = np.concatenate(all_actions)
    boundaries = torch.quantile(torch.tensor(all_actions), quantiles, dim=0)
    np.save(os.path.join(path, "action_quantiles.npy"), boundaries.numpy())

def main(dataset_name):
    transform = transforms.Compose([
        transforms.Resize((180, 320), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    text_encoder = UniversalSentenceEncoder(device="cuda")
    dataset_path = dataset_paths[dataset_name]
    np_path = dataset_path + "_npz"
    obs_path = np_path + "/observations"
    print("np_path: ", np_path)
    print("obs_path: ", obs_path)
    # paths = {    
    #     "action": [np_path + "/actions", []],
    #     "is_terminal": [np_path + "/is_terminals", []],
    #     "instruction": [obs_path + "/instructions", []],
    #     "is_first": [np_path + "/is_firsts", []],
    #     "is_last": [np_path + "/is_lasts", []],
    #     "effector_target_translation": [obs_path + "/effector_target_translations", []],
    #     "effector_translation": [obs_path + "/effector_translations", []],
    #     "rgb": [obs_path + "/rgbs", []],
    #     "reward": [np_path + "/rewards", []],
    # }
    paths = {    
        "action": [np_path + "/actions", []],
        "instruction": [obs_path + "/instructions", []],
        "rgb": [obs_path + "/rgbs", []],
        "reward": [np_path + "/rewards", []],
        "inst_embed": [obs_path + "/inst_embedding_use", None],
    }
    traj_len = [np_path + "/traj_len.npz", []]
    for values in paths.values():
        os.makedirs(values[0], exist_ok=True)
    builder = tfds.builder_from_directory(dataset_path)
    print(builder.info.features)
    dataset = tfds.as_numpy(builder.as_dataset(split='train'))
    print("episode num: ", len(dataset))
    actions = []
    for ep_idx, ep in tqdm(enumerate(dataset)):
        episode = ep['steps']
        for key in paths.keys():
            paths[key][1] = []
        ep_step_num = 0
        for step in episode:
            ep_step_num += 1
            paths["action"][1].append(step["action"])
            # paths["is_first"][1].append(step["is_first"])
            # paths["is_last"][1].append(step["is_last"])
            # paths["is_terminal"][1].append(step["is_terminal"])
            # print(step["observation"]["rgb"].shape)
            # resized_rgb = cv.resize(step["observation"]["rgb"], (300, 168))
            # print(resized_rgb.shape)
            # resized_rgb = np.pad(resized_rgb, ((66, 66), (0, 0), (0, 0))).transpose((2, 0, 1))
            # print(resized_rgb.shape)
            # print(resized_rgb.shape)
            # import pdb; pdb.set_trace()
            frame = transform(Image.fromarray(step["observation"]["rgb"]))
            paths["rgb"][1].append(frame.numpy())
            paths["instruction"][1].append(nlp_inst_decoder(step["observation"]["instruction"]))
            # paths["effector_target_translation"][1].append(step["observation"]["effector_target_translation"])
            # paths["effector_translation"][1].append(step["observation"]["effector_translation"])
            paths["reward"][1].append(step["reward"])
        # import pdb; pdb.set_trace()
        traj_len[1].append(ep_step_num)
        inst_embed = text_encoder.embed_text(paths["instruction"][1]).detach().cpu().numpy()
        paths["inst_embed"][1] = inst_embed
        ep_act_array = np.array(paths["action"][1])
        actions.append(ep_act_array)
        for value in paths.values():
            np.savez_compressed(os.path.join(value[0], str(ep_idx) + '.npz'), value[1])
            # np.save(os.path.join(value[0], str(ep_idx) + '.npy'), value[1])
    np.savez_compressed(traj_len[0], np.array(traj_len[1]))
    quantileBinning(actions, np_path)

    


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    args = parser.parse_args()
    main(args.ds_name)
