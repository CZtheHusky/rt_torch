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
from math import ceil


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
    def __init__(self, mode: str, ds_type: str, split=0.9, weight=None, batch_size=16, stack=True, rgb_list=False, seq_len=6, token_num=8, seed=100, shuffle=True) -> None:
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
        self.chunk_size = 0
        self.ds_type = ds_type
        self.index_range = {}
        self.ds_list = []
        self.weight = []
        self.sub_size = {}
        self.sub_len = {}
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.stack = stack
        self.rgb_list = rgb_list
        self.action_idx = np.arange(token_num - 1, seq_len * token_num, token_num)
        self.rgb_padding = torch.zeros(seq_len - 1, 3, 300, 300)
        self.act_padding = torch.ones(seq_len - 1, 2)
        self.inst_embed_padding = torch.zeros(seq_len - 1, 768)
        self.inst_padding = [""] * (seq_len - 1)
        self.seed = seed
        if shuffle:
            np.random.seed(self.seed)
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
                chunk_len = np.ceil(traj_len / seq_len).astype(np.int32)
                if os.path.exists(inst_embed_t5_path):
                    assert 1 == len(set([len(os.listdir(act_path)), len(os.listdir(inst_path)), len(os.listdir(rgb_path)), len(os.listdir(rew_path)), len(os.listdir(inst_embed_t5_path)), len(traj_len)]))
                    inst_embed_status = True
                else:
                    assert 1 == len(set([len(os.listdir(act_path)), len(os.listdir(inst_path)), len(os.listdir(rgb_path)), len(os.listdir(rew_path)), len(traj_len)]))
                    inst_embed_status = False
                size = len(traj_len)
                train_size = int(size * self.split)
                test_size = size - int(size * self.split)
                file_idx = np.arange(size)
                if shuffle:
                    np.random.shuffle(file_idx)
                # import pdb; pdb.set_trace()
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
                    "file_idx": file_idx,
                }
                if self.mode == 0:
                    ds_chunk_len = chunk_len[:train_size].sum()
                    self.index_range[k] = [0, train_size]
                    self.chunk_size += ds_chunk_len
                    self.sub_len[k] = ds_chunk_len
                    self.sub_size[k] = train_size
                else:
                    ds_chunk_len = chunk_len[train_size:].sum()
                    self.index_range[k] = [train_size, size]
                    self.chunk_size += ds_chunk_len
                    self.sub_len[k] = ds_chunk_len
                    self.sub_size[k] = test_size

                if weight is None:
                    self.weight.append(self.sub_len[k])
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
        self.len = self.chunk_size // self.batch_size
        self.weight = np.array(self.weight) / np.sum(self.weight)
        self.buffer = {"rgb": [], "inst": [], "act": [], "act_mask": []}
        self.buffer_len = 0
        self.steps_per_batch = seq_len * batch_size
        assert len(self.weight) > 0
        print(self.ds_list)
                
    def __loaditem__(self, ds=None):
        while self.buffer_len < self.batch_size:
            if ds is None:
                choice = np.random.choice(len(self.weight), p=self.weight)
                ds_name = self.ds_list[choice]
            else:
                ds_name = ds
            npz_name = str(self.ds_stats[ds_name]["file_idx"][self.ds_stats[ds_name]["current_idx"] % self.sub_size[ds_name]] + self.index_range[ds_name][0]) + ".npz"
            # print(f"accessing {ds_name}, idx: {npz_name}")
            if self.embed_ready:
                inst = np.load(os.path.join(self.ds_stats[ds_name]["path"]["inst_embed"], npz_name))['arr_0']
            else:
                inst = np.load(os.path.join(self.ds_stats[ds_name]["path"]["instruction"], npz_name))['arr_0']
            rgb = np.load(os.path.join(self.ds_stats[ds_name]["path"]["rgb"], npz_name))['arr_0']
            act = np.load(os.path.join(self.ds_stats[ds_name]["path"]["action"], npz_name))['arr_0']
            split_num = int(ceil(len(rgb) / self.seq_len))
            self.buffer_len += split_num
            # print(f"add {len(rgb)} into buffer: {self.buffer_len}")
            rgbs, insts, acts = self.return_preprocess(rgb, inst, act)
            rgb = list(torch.split(rgbs, self.seq_len))
            if self.embed_ready:
                inst = list(torch.split(insts, self.seq_len))
            else:
                inst = np.array_split(insts, self.seq_len)
                inst = [list(sub_inst) for sub_inst in insts]
            act = list(torch.split(acts, self.seq_len))
            act_mask = [torch.zeros(self.seq_len) for i in range(split_num)]
            if len(rgb[-1]) != self.seq_len:
                num = self.seq_len - len(rgb[-1])
                rgb[-1] = torch.cat([rgb[-1], self.rgb_padding[:num]])
                # import pdb; pdb.set_trace()
                act_mask[-1][-num:] = 1
                if self.embed_ready:
                    inst[-1] = torch.cat([inst[-1], self.inst_embed_padding[:num]])
                else:
                    inst[-1] = inst[-1].extend(self.inst_padding[:num])
                act[-1] = torch.cat([act[-1], self.act_padding[:num]])
            self.buffer["rgb"].extend(rgb)
            self.buffer["inst"].extend(inst)
            self.buffer['act'].extend(act)
            self.buffer["act_mask"].extend(act_mask)
            self.ds_stats[ds_name]["current_idx"] += 1
            self.ds_stats[ds_name]["current_idx"] %= self.sub_size[ds_name]

    # def __loaditem__(self, ds=None):
    #     while self.buffer_len < self.batch_size:
    #         if ds is None:
    #             choice = np.random.choice(len(self.weight), p=self.weight)
    #             ds_name = self.ds_list[choice]
    #         else:
    #             ds_name = ds
    #         npz_name = str(self.ds_stats[ds_name]["file_idx"][self.ds_stats[ds_name]["current_idx"] % self.sub_size[ds_name]] + self.index_range[ds_name][0]) + ".npz"
    #         # print(f"accessing {ds_name}, idx: {npz_name}")
    #         if self.embed_ready:
    #             inst = np.load(os.path.join(self.ds_stats[ds_name]["path"]["inst_embed"], npz_name))['arr_0']
    #         else:
    #             inst = np.load(os.path.join(self.ds_stats[ds_name]["path"]["instruction"], npz_name))['arr_0']
    #         rgb = np.load(os.path.join(self.ds_stats[ds_name]["path"]["rgb"], npz_name))['arr_0']
    #         act = np.load(os.path.join(self.ds_stats[ds_name]["path"]["action"], npz_name))['arr_0']
    #         split_num = int(ceil(len(rgb) / self.seq_len))
    #         self.buffer_len += split_num
    #         # print(f"add {len(rgb)} into buffer: {self.buffer_len}")
    #         rgb, inst, act = self.return_preprocess(rgb, inst, act)
    #         n = len(rgb) % self.seq_len
    #         action_mask = torch.zeros(split_num, self.seq_len)
    #         if n:
    #             padding_num = self.seq_len - n
    #             # import pdb; pdb.set_trace()
    #             rgb = torch.cat([rgb, self.rgb_padding[:padding_num]])
    #             action_mask[-1, -padding_num:] = 1
    #             if self.embed_ready:
    #                 inst = torch.cat([inst, self.inst_embed_padding[:padding_num]])
    #             else:
    #                 inst = inst.extend(self.inst_padding[:padding_num])
    #             act = torch.cat([act, self.act_padding[:padding_num]])
    #         # if len(rgb) % self.seq_len != 0 or len(inst) % self.seq_len != 0 or len(act) % self.seq_len != 0:
    #         #     import pdb; pdb.set_trace()
    #         self.buffer["rgb"].append(rgb)
    #         self.buffer["inst"].append(inst)
    #         self.buffer['act'].append(act)
    #         self.buffer["act_mask"].append(action_mask)
    #         self.ds_stats[ds_name]["current_idx"] += 1
    #         self.ds_stats[ds_name]["current_idx"] %= self.sub_size[ds_name]
        

    def return_preprocess(self, rgb, inst, act):
        if self.stack:
            return frame_stack(torch.tensor(rgb).float() / 255), torch.tensor(inst) if self.embed_ready else list(inst), torch.tensor(act)
        else:
            return torch.tensor(rgb).float() / 255, torch.tensor(inst) if self.embed_ready else list(inst), torch.tensor(act)

    # def __getbuffer__(self):
    #     self.buffer_len -= self.batch_size
    #     tmp_buffer = {"rgb": [], "inst": [], "act": [], "act_mask": []}
    #     # import pdb; pdb.set_trace()
    #     # for k, v in self.buffer.items():
    #     #     print(k)
    #     #     for item in v:
    #     #         print(item.shape)
    #     if self.buffer_len != 0:
    #         for k, v in self.buffer.items():
    #             split_idx = self.buffer_len * self.seq_len
    #             if k != "act_mask":
    #                 tmp_buffer[k] = [v[-1][-split_idx:]]
    #                 self.buffer[k][-1] = v[-1][:-split_idx]
    #             else:
    #                 tmp_buffer[k] = [v[-1][-self.buffer_len:]]
    #                 self.buffer[k][-1] = v[-1][:-self.buffer_len]
    #     # import pdb; pdb.set_trace()
    #     # for k, v in self.buffer.items():
    #     #     print(k)
    #     #     for item in v:
    #     #         print(item.shape)
    #     # for k, v in tmp_buffer.items():
    #     #     print(k)
    #     #     for item in v:
    #     #         print(item.shape)
    #     act = torch.cat(self.buffer["act"])
    #     rgb = torch.cat(self.buffer["rgb"])
    #     if self.embed_ready:
    #         inst = torch.cat(self.buffer["inst"])
    #     else:
    #         inst = self.buffer["inst"]
    #     act_mask = torch.cat(self.buffer["act_mask"]).long()
    #     # mask_idx = torch.where(act_mask != 0)
    #     # import pdb; pdb.set_trace()
    #     # act_mask = act_mask.numpy()
    #     # mask_idx = mask_idx[0].numpy()
    #     # bs_idx = torch.cat([torch.ones(int(act_mask[idx])) * idx for idx in mask_idx]).long()
    #     # row_idx = torch.cat([torch.arange(self.seq_len - int(act_mask[idx]), self.seq_len) for idx in mask_idx]).long()
    #     # # import pdb; pdb.set_trace()
    #     # act_mask = tuple([bs_idx, row_idx])
    #     self.buffer = tmp_buffer
    #     # if len(rgb) != self.steps_per_batch or len(inst) != self.steps_per_batch or len(act) != self.steps_per_batch or len(rgb) != self.steps_per_batch:
    #     #     import pdb; pdb.set_trace()
    #     return rgb, inst, act, act_mask

    def __getbuffer__(self):
        self.buffer_len -= self.batch_size
        tmp_buffer = {"rgb": [], "inst": [], "act": [], "act_mask": []}
        if self.buffer_len != 0:
            for k, v in self.buffer.items():
                tmp_buffer[k] = v[-self.buffer_len:]
                self.buffer[k] = v[:-self.buffer_len]
        # import pdb; pdb.set_trace()
        act = torch.stack(self.buffer["act"])
        rgb = torch.stack(self.buffer["rgb"])
        act_mask = torch.stack(self.buffer["act_mask"])
        if self.embed_ready:
            inst = torch.stack(self.buffer["inst"])
        else:
            inst = self.buffer["inst"]
        self.buffer = tmp_buffer
        return rgb, inst, act, act_mask
    

    def __getitem__(self, index):
        if self.ds_type == 'mix':
            self.__loaditem__()
        else:
            ds_name = self.ds_type
            self.__loaditem__(ds_name)
        rgb, inst, act, act_mask = self.__getbuffer__()
        if self.embed_ready:
            return rgb, None, act, inst, act_mask
        else:
            return rgb, inst, act, None, act_mask
    
    
            
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
