import numpy as np
import tensorflow_datasets as tfds
import torch
from torch.utils.data import Dataset, DataLoader
from collections import deque
import tensorflow as tf
import cv2 as cv
import time
import torch.nn.functional as F
from multiprocessing import Process, Pipe, Queue
from multiprocessing.connection import Connection
import os


INPUT_SIZE = 300

def rgb_preprocess(images,
                   crop_size: int = INPUT_SIZE
                   ):
    """
    EP_LEN * 360 * 640 * 3 to EP_LEN * 3 * 300 * 300
    uint8 to float
    """
    # import pdb 
    # pdb.set_trace()
    images = torch.permute(images, (0, 3, 1, 2)).to(torch.float32)    # EP_LEN * 3 * 360 * 640
    images = torch.nn.functional.interpolate(images, scale_factor=0.46875, mode="bilinear")  # EP_LEN * 3 * 168 * 300
    images = torch.nn.functional.pad(images, (0, 0, 66, 66))    # EP_LEN * 3 * 300 * 300
    images = images.float()
    # images = torch.tensor(images).permute((2, 0, 1))
    # images = images.unsqueeze(0)
    # images = torch.nn.functional.interpolate(images, size=(crop_size, crop_size))
    return images / 255

"""
x = B* ep_len * C * H * W
x -> B x 6 x C x H x W



x[:, 0:-6]
x[:, 1:-5]

(B x T-6) x 6 ....
"""

def frame_stack(images, stack_num):
    tmp = [torch.zeros((3, 300, 300))] * (stack_num - 1)
    frames = deque(tmp, maxlen=stack_num)
    rgbs = []
    for image in images:
        frames.append(image)
        rgbs.append(torch.stack(list(frames)))
    return torch.stack(rgbs).permute(0, 2, 1, 3, 4) # EP_LEN 6 3 300 300 -> EP_LEN 3 6 300 300 

dataset_paths = {
    'language_table': '/raid/robotics_data/language_table_npz',
    'language_table_sim': '/raid/robotics_data/language_table_sim_npz',
    'language_table_blocktoblock_sim': '/raid/robotics_data/language_table_blocktoblock_sim_npz',
    'language_table_blocktoblock_4block_sim': '/raid/robotics_data/language_table_blocktoblock_4block_sim_npz',
    'language_table_blocktoblock_oracle_sim': '/raid/robotics_data/language_table_blocktoblock_oracle_sim_npz',
    'language_table_blocktoblockrelative_oracle_sim': '/raid/robotics_data/language_table_blocktoblockrelative_oracle_sim_npz',
    'language_table_blocktoabsolute_oracle_sim': '/raid/robotics_data/language_table_blocktoabsolute_oracle_sim_npz',
    'language_table_blocktorelative_oracle_sim': '/raid/robotics_data/language_table_blocktorelative_oracle_sim_npz',
    'language_table_separate_oracle_sim': '/raid/robotics_data/language_table_separate_oracle_sim_npz',
}

sub_datas = ["mix"] + list(dataset_paths.keys())


class language_table_dataset_npz(Dataset):
    def __init__(self, mode: str, ds_type: str, split=0.9, weight=None, batch_size=16, stack=True, rgb_list=False, stack_num=6) -> None:
        super().__init__()
        assert ds_type in sub_datas
        assert weight is None or len(weight) == 9
        assert 0 <= split <= 1
        if mode == "train":
            self.mode = 0
        elif mode == "test":
            self.mode = 1
        self.ds_stats = {}
        self.split = split
        self.step_size = 0
        self.ds_type = ds_type
        self.index_range = {}
        self.ds_list = []
        self.weight = []
        self.sub_size = {}
        self.sub_len = {}
        self.batch_size = batch_size
        self.stack = stack
        self.rgb_list = rgb_list
        self.stack_num = stack_num
        for idx, (k, v) in enumerate(dataset_paths.items()):
            obs_path = v + "/observations"
            act_path = v + "/actions"
            inst_path = obs_path + "/instructions"
            inst_embed_t5_path = obs_path + "/inst_embedding_t5"
            rgb_path = obs_path + "/rgbs"
            rew_path = v + "/rewards"
            traj_len_path = v + "/traj_len.npz"
            try:
                traj_len = np.load(traj_len_path)['arr_0']
                if os.path.exists(inst_embed_t5_path):
                    assert 1 == len(set([len(os.listdir(act_path)), len(os.listdir(inst_path)), len(os.listdir(rgb_path)), len(os.listdir(rew_path)), len(os.listdir(inst_embed_t5_path)), len(traj_len)]))
                    inst_embed_status = True
                else:
                    assert 1 == len(set([len(os.listdir(act_path)), len(os.listdir(inst_path)), len(os.listdir(rgb_path)), len(os.listdir(rew_path)), len(traj_len)]))
                    inst_embed_status = False
                size = len(traj_len)
                train_size = int(size * self.split)
                test_size = size - int(size * self.split)
                self.ds_stats[k] = {
                    "path": {    
                        "action": act_path,
                        "instruction": inst_path,
                        "rgb": rgb_path,
                        "reward": rew_path,
                        "inst_embed": inst_embed_t5_path if inst_embed_status else None,
                        "traj_len": traj_len_path,
                    },
                    "size": size,
                    "train_size": train_size,
                    "test_size": test_size,
                    "current_idx": 0,
                }
                if self.mode == 0:
                    ds_traj_len = traj_len[:train_size].sum()
                    self.index_range[k] = [0, train_size]
                    self.step_size += ds_traj_len
                    self.sub_len[k] = ds_traj_len
                    self.sub_size[k] = train_size
                else:
                    ds_traj_len = traj_len[train_size:].sum()
                    self.index_range[k] = [train_size, size]
                    self.step_size += ds_traj_len
                    self.sub_len[k] = ds_traj_len
                    self.sub_size[k] = test_size

                if weight is None:
                    self.weight.append(size)
                    self.ds_list.append(k)
                elif weight[idx] != 0:
                    self.ds_list.append(k)
                    self.weight.append(weight[idx])
            except Exception as e:
                # self.ds_stats[k] = None
                # self.index_range[k] = None
                print(f"dataset {k} not prepared")
        embed_ready = True
        for k, v in self.ds_stats.items():
            if v['path']['inst_embed'] is None:
                embed_ready = False
                break
        self.embed_ready = embed_ready
        self.len = self.step_size // self.batch_size
        self.weight = np.array(self.weight) / np.sum(self.weight)
        self.buffer = {"rgb": [], "inst": [], "act": []}
        self.buffer_len = 0
        self.new_ep = False
        self.next_new_ep = True
        assert len(self.weight) > 0
        print(self.weight)
        print(self.ds_list)
                
    def __loaditem__(self, ds=None):
        while self.buffer_len < self.batch_size:
            if ds is None:
                choice = np.random.choice(len(self.weight), p=self.weight)
                ds_name = self.ds_list[choice]
            else:
                ds_name = ds
            npz_name = str(self.ds_stats[ds_name]["current_idx"] % self.sub_size[ds_name] + self.index_range[ds_name][0]) + ".npz"
            if self.embed_ready:
                inst = np.load(os.path.join(self.ds_stats[ds_name]["path"]["inst_embed"], npz_name))['arr_0']
            else:
                inst = np.load(os.path.join(self.ds_stats[ds_name]["path"]["instruction"], npz_name))['arr_0']
            rgb = np.load(os.path.join(self.ds_stats[ds_name]["path"]["rgb"], npz_name))['arr_0']
            act = np.load(os.path.join(self.ds_stats[ds_name]["path"]["action"], npz_name))['arr_0']
            self.buffer_len += len(rgb)
            # print(f"add {len(rgb)} into buffer: {self.buffer_len}")
            rgb, inst, act = self.return_preprocess(rgb, inst, act)
            self.buffer["rgb"].append(rgb)
            self.buffer["inst"].append(inst)
            self.buffer['act'].append(act)
            self.ds_stats[ds_name]["current_idx"] += 1
            self.ds_stats[ds_name]["current_idx"] %= self.sub_size[ds_name]
            

    def return_preprocess(self, rgb, inst, act):
        if self.stack:
            # import pdb; pdb.set_trace()
            return frame_stack(torch.tensor(rgb).float() / 255), torch.tensor(inst) if self.embed_ready else list(inst), torch.tensor(act)
        else:
            # return torch.cat([torch.zeros((self.stack_num - 1, 3, 300, 300)), torch.tensor(rgb).float() / 255]), torch.cat([torch.zeros((self.stack_num - 1, 768)), torch.tensor(inst)]) if self.embed_ready  else list(inst), torch.tensor(act)
            return torch.tensor(rgb).float() / 255, torch.tensor(inst) if self.embed_ready else list(inst), torch.tensor(act)

    def __getbuffer__(self):
        self.buffer_len -= self.batch_size
        # print("remove from buffer:", self.buffer_len)
        tmp_buffer = {"rgb": [], "inst": [], "act": []}
        self.new_ep = self.next_new_ep
        if self.buffer_len > 0:
            self.next_new_ep = False
            for k, v in self.buffer.items():
                if k != "act":
                    reserve_len = self.buffer_len + self.stack_num - 1
                    if len(v[-1]) < reserve_len:
                        # import pdb; pdb.set_trace()
                        if k == "rgb":
                            tmp_buffer[k] = [torch.cat([torch.zeros([reserve_len - len(v[-1]), 3, 300, 300]), v[-1]])]
                        else:
                            tmp_buffer[k] = [torch.cat([torch.zeros([reserve_len - len(v[-1]), 768]), v[-1]])]
                    else:
                        tmp_buffer[k] = [v[-1][-reserve_len:]]
                else:
                    tmp_buffer[k] = [v[-1][-(self.buffer_len):]]
                v[-1] = v[-1][:-self.buffer_len]
        else:
            self.next_new_ep = True
        if self.rgb_list:
            rgb = self.buffer["rgb"]
        else:
            rgb = torch.cat(self.buffer["rgb"], dim=0)
        if isinstance(self.buffer["inst"][0], list):
            inst = []
            [inst.extend(sub_list) for sub_list in self.buffer["inst"]]
        else:
            inst = torch.cat(self.buffer["inst"], dim=0)
        act = torch.cat(self.buffer["act"], dim=0)
        self.buffer = tmp_buffer
        return rgb, inst, act
    

    def __getitem__(self, index):
        if self.ds_type == 'mix':
            self.__loaditem__()
        else:
            ds_name = self.ds_type
            self.__loaditem__(ds_name)
        rgb, inst, act = self.__getbuffer__()
        if self.embed_ready:
            return rgb, None, act, inst, self.new_ep
        else:
            return rgb, inst, act, None, self.new_ep
    
    
            
    def __len__(self):
        return self.len


if __name__ == "__main__":
    from tqdm import tqdm
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    train_loader = language_table_dataset_npz(mode="train", ds_type='mix', batch_size=16, weight=[0, 1, 0, 0, 0, 0, 0, 0, 0])
    test_loader = language_table_dataset_npz(mode="test", ds_type='mix', batch_size=16, weight=[0, 1, 0, 0, 0, 0, 0, 0, 0])
    print("get pbar")
    print(len(train_loader))
    print(len(test_loader))
    pbar = tqdm(range(len(train_loader)))
    print("start iteration")
    for epoch in range(2):
        for idx, item in enumerate(train_loader):
            pbar.update(1)

    # train_loader = language_table_dataset(dataset_type="train", sub_data='mix', batch_size=1, weight=[1, 1, ] + [0] * 7)
    # test_loader = language_table_dataset(dataset_type="test", sub_data='mix', batch_size=1, weight=[1, 1, ] + [0] * 7)

    # pbar = tqdm(range(len(train_loader)))
    # for idx, item in enumerate(train_loader):
    #     pbar.update(1)
