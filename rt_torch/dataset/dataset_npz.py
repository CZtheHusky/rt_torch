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
from torch.utils.data import DataLoader

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

def frame_stack(images, seq_len):
    tmp = [torch.zeros((3, 300, 300))] * (seq_len - 1)
    frames = deque(tmp, maxlen=seq_len)
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


def build_language_table_ds(split=0.9, batch_size=16, rgb_list=True, seq_len=6, seed=100):
    np.random.seed(seed)
    ds_stats = {}
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
            ds_stats[k] = {
                "path": {    
                    "action": act_path,
                    "instruction": inst_path,
                    "rgb": rgb_path,
                    "reward": rew_path,
                    "inst_embed": inst_embed_t5_path if inst_embed_status else None,
                    "traj_len": traj_len_path,
                },
                "num_ep": size,
                "current_idx": 0,
            }
        except Exception as e:
            pass

    indices = {}
    indices_index = {}
    for k, v in ds_stats.items():
        indices[k] = []
        traj_len_path = v["path"]["traj_len"]
        traj_len = np.load(traj_len_path)['arr_0']
        # traj_len = traj_len[:200]
        current_slice = [0]
        current_len = 0
        # import pdb; pdb.set_trace()
        done = True
        for idx, item in enumerate(traj_len):
            remainder = item % seq_len
            if remainder != 0:
                pad_num = seq_len - remainder
                item += pad_num
            current_len += item
            current_slice.append(idx)
            while current_len >= batch_size:
                if current_len == batch_size:
                    current_slice.append(None)
                    indices[k].append(current_slice)
                    current_len = 0
                    current_slice = [0]
                    done = True
                else:
                    extra = current_len - batch_size
                    current_slice.append(-extra)
                    indices[k].append(current_slice)
                    current_slice = [item - extra]
                    current_slice.append(idx)
                    current_len = extra
                    done = False
        if not done:
            current_slice.append(item)
            indices[k].append(current_slice)
        # import pdb; pdb.set_trace()
        total_indice_num = len(indices[k])
        train_indice_num = int(total_indice_num * split)
        indices_index[k] = {"train": None, "test": None}
        total_indexes = np.arange(total_indice_num)
        np.random.shuffle(total_indexes)
        indices_index[k]["train"] = total_indexes[:train_indice_num]
        indices_index[k]["test"] = total_indexes[train_indice_num:]

    train_set = language_table_dataset_npz(mode="train", indices_index=indices_index, indices=indices, ds_stats=ds_stats, rgb_list=rgb_list,seq_len=seq_len)
    test_set = language_table_dataset_npz(mode="test", indices_index=indices_index, indices=indices, ds_stats=ds_stats, rgb_list=rgb_list, seq_len=seq_len)
    return train_set, test_set

class language_table_dataset_npz(Dataset):
    def __init__(self, mode, indices_index, indices, ds_stats, rgb_list=True, seq_len=6) -> None:
        super().__init__()
        self.mode = mode
        self.ds_stats = ds_stats
        self.indices_index = indices_index
        self.indices = indices
        self.rgb_list = rgb_list
        self.seq_len = seq_len
        self.sub_ds_len = []
        self.ds_list = []
        self.len =0
        self.bucket = [0]
        for k, v in self.indices_index.items():
            self.len += len(v[mode])
            self.bucket.append(self.len)
            self.ds_list.append(k)
            self.sub_ds_len.append(len(v[mode]))
        print(mode)
        print(f"ds_list: {self.ds_list}")
        print(f"length: {self.len}")
        print(f"sub_ds_len: {self.sub_ds_len}")
        print(f"bucket: {self.bucket}")
            

    def return_preprocess(self, rgb, inst, act):
        return torch.tensor(rgb).float() / 255, torch.tensor(inst), torch.tensor(act)


    def __getindice__(self, index):
        assert index >= 0 and index < self.len
        bucket_len = len(self.bucket)
        sub_ds_index = index
        # import pdb; pdb.set_trace()
        for idx in range(0, bucket_len-1):
            if index < self.bucket[idx + 1] and index >= self.bucket[idx]:
                break
            else:
                sub_ds_index -= self.bucket[idx + 1]
        ds_name = self.ds_list[idx]
        # print(f"ds_type: {self.mode}, ds_name: {ds_name}, sub_ds_index: {sub_ds_index}")
        indice = self.indices[ds_name][sub_ds_index]
        # print("indice:", indice)
        ep_start_idx = indice[0]
        ep_end_idx = indice[-1]
        rgbs = []
        acts = []
        insts = []
        act_mask= []
        for id_idx in range(1, len(indice) - 1):
            # import pdb; pdb.set_trace()
            npz_name = str(indice[id_idx]) + '.npz'
            rgb = np.load(os.path.join(self.ds_stats[ds_name]["path"]["rgb"], npz_name))['arr_0']
            
            act = np.load(os.path.join(self.ds_stats[ds_name]["path"]["action"], npz_name))['arr_0']
            
            inst = np.load(os.path.join(self.ds_stats[ds_name]["path"]["inst_embed"], npz_name))['arr_0']
            
            rgb, inst, act = self.return_preprocess(rgb, inst, act)
            remainder = len(rgb) % self.seq_len
            pad_num = self.seq_len - remainder
            if remainder != 0:
                rgb = torch.cat([rgb, torch.zeros(pad_num, 3, 300, 300)])
                act = torch.cat([act, torch.zeros(pad_num, 2)])
                inst = torch.cat([inst, torch.zeros(pad_num, 768)])
                
            action_mask = torch.zeros(len(rgb))
            action_mask[-pad_num:] = 1
            # print(f"loading rgb {npz_name}, length: {len(rgb)}")
            # print(f"loading act {npz_name}, length: {len(act)}")
            # print(f"loading inst {npz_name}, length: {len(inst)}")
            # print(f"loading action_mask {npz_name}, length: {len(action_mask)}")
            rgbs.append(rgb)
            acts.append(act)
            insts.append(inst)
            act_mask.append(action_mask)
        acts = torch.cat(acts)
        rgbs = torch.cat(rgbs)
        act_mask = torch.cat(act_mask)
        if isinstance(insts[0], list):
            tmp_insts = []
            [tmp_insts.extend(sub_list) for sub_list in insts]
            insts = tmp_insts
        else:
            insts = torch.cat(insts)
        # import pdb; pdb.set_trace()
        # print(f"rgbs: {len(rgbs)}")
        # print(f"acts: {len(acts)}")
        # print(f"insts: {len(insts)}")
        # print(f"act_mask: {len(act_mask)}")
        insts = insts[ep_start_idx:]
        rgbs = rgbs[ep_start_idx:]
        acts = acts[ep_start_idx:]
        act_mask = act_mask[ep_start_idx:]
        if ep_end_idx is not None:
            rgbs = rgbs[:ep_end_idx]
            acts = acts[:ep_end_idx]
            insts = insts[:ep_end_idx]
            act_mask = act_mask[:ep_end_idx]

        # if len(insts) != self.batch_size or len(rgbs) != self.batch_size or len(acts) != self.batch_size or len(act_mask) != self.batch_size:
        #     [print(len(rgb)) for rgb in rgbs] 
        # import pdb; pdb.set_trace()
        return rgbs, insts, acts, act_mask
            
        
    def __getitem__(self, index):
        return self.__getindice__(index)
    
    def __len__(self):
        return self.len


# class language_table_dataset_npz(Dataset):
#     def __init__(self, mode: str, ds_type: str, split=0.9, weight=None, batch_size=96, stack=False, rgb_list=False, seq_len=6) -> None:
#         super().__init__()
#         assert ds_type in sub_datas
#         assert weight is None or len(weight) == 9
#         assert 0 <= split <= 1
#         if mode == "train":
#             self.mode = 0
#         elif mode == "test":
#             self.mode = 1
#         self.ds_stats = {}
#         self.split = split
#         self.indices_num = 0
#         self.ds_type = ds_type
#         self.index_range = {}
#         self.ds_list = []
#         self.weight = []
#         self.sub_len = {}
#         self.batch_size = batch_size
#         self.stack = stack
#         self.rgb_list = rgb_list
#         self.seq_len = seq_len
#         for idx, (k, v) in enumerate(dataset_paths.items()):
#             obs_path = v + "/observations"
#             act_path = v + "/actions"
#             inst_path = obs_path + "/instructions"
#             inst_embed_t5_path = obs_path + "/inst_embedding_t5"
#             rgb_path = obs_path + "/rgbs"
#             rew_path = v + "/rewards"
#             traj_len_path = v + "/traj_len.npz"
#             try:
#                 traj_len = np.load(traj_len_path)['arr_0']
#                 if os.path.exists(inst_embed_t5_path):
#                     assert 1 == len(set([len(os.listdir(act_path)), len(os.listdir(inst_path)), len(os.listdir(rgb_path)), len(os.listdir(rew_path)), len(os.listdir(inst_embed_t5_path)), len(traj_len)]))
#                     inst_embed_status = True
#                 else:
#                     assert 1 == len(set([len(os.listdir(act_path)), len(os.listdir(inst_path)), len(os.listdir(rgb_path)), len(os.listdir(rew_path)), len(traj_len)]))
#                     inst_embed_status = False
#                 size = len(traj_len)
#                 self.ds_stats[k] = {
#                     "path": {    
#                         "action": act_path,
#                         "instruction": inst_path,
#                         "rgb": rgb_path,
#                         "reward": rew_path,
#                         "inst_embed": inst_embed_t5_path if inst_embed_status else None,
#                         "traj_len": traj_len_path,
#                     },
#                     "num_ep": size,
#                     "current_idx": 0,
#                 }
#             except Exception as e:
#                 pass
#         self.indices = {}
#         self.bucket = [0]
#         for k, v in self.ds_stats.items():
#             self.indices[k] = []
#             traj_len_path = v["path"]["traj_len"]
#             traj_len = np.load(traj_len_path)['arr_0']
#             # traj_len = traj_len[:200]

#             current_slice = [0]
#             current_len = 0

#             # import pdb; pdb.set_trace()
#             done = True


#             # for idx, item in enumerate(traj_len):
#             #     remainder = item % seq_len
#             #     if remainder != 0:
#             #         pad_num = seq_len - remainder
#             #         item += pad_num
#             #     tmp_item = item
#             #     split_num = tmp_item / seq_len
#             #     while split_num > 0:
#             #         current_slice.append(idx)
#             #         end_idx = current_slice[0] + self.seq_len
#             #         neg_end_idx = end_idx - tmp_item
#             #         if neg_end_idx != 0:
#             #             current_slice.append(neg_end_idx)
#             #             self.indices[k].append(current_slice)
#             #             current_slice = [end_idx]
#             #         else:
#             #             current_slice.append(None)
#             #             self.indices[k].append(current_slice)
#             #             current_slice = [0]
#             #         split_num -= 1

#             for idx, item in enumerate(traj_len):
#                 remainder = item % seq_len
#                 if remainder != 0:
#                     pad_num = seq_len - remainder
#                     item += pad_num
#                 current_len += item
#                 current_slice.append(idx)
#                 if current_len > self.batch_size:
#                     extra = current_len - self.batch_size
#                     current_slice.append(-extra)
#                     self.indices[k].append(current_slice)
#                     current_slice = [item - extra]
#                     current_slice.append(idx)
#                     current_len = extra
#                     done = False
#                 elif current_len == self.batch_size:
#                     current_slice.append(None)
#                     self.indices[k].append(current_slice)
#                     current_len = 0
#                     current_slice = [0]
#                     done = True
#             import pdb; pdb.set_trace()
#             if not done:
#                 if item % seq_len != 0:
#                     pad_num = seq_len - item % seq_len
#                     item += pad_num
#                 current_slice.append(item)
#                 self.indices[k].append(np.array(current_slice))

#             total_indice_num = len(self.indices[k])
#             train_indice_num = int(total_indice_num * split)
#             test_indice_num = total_indice_num - train_indice_num
#             if self.mode == 0:
#                 self.index_range[k] = [0, train_indice_num]
#                 self.indices_num += train_indice_num
#                 self.sub_len[k] = train_indice_num
#                 self.bucket.append(train_indice_num + self.bucket[-1])
#             else:
#                 self.index_range[k] = [train_indice_num, total_indice_num]
#                 self.indices_num += test_indice_num
#                 self.sub_len[k] = test_indice_num
#                 self.bucket.append(test_indice_num + self.bucket[-1])

#             if weight is None:
#                 self.weight.append(self.sub_len[k])
#                 self.ds_list.append(k)
#             elif weight[idx] != 0:
#                 self.ds_list.append(k)
#                 self.weight.append(weight[idx])
#         # import pdb; pdb.set_trace()

#         embed_ready = True
#         for k, v in self.ds_stats.items():
#             if v['path']['inst_embed'] is None:
#                 embed_ready = False
#                 break
#         self.embed_ready = embed_ready
#         self.len = self.indices_num
#         self.weight = np.array(self.weight) / np.sum(self.weight)
#         self.buffer = {"rgb": [], "inst": [], "act": []}
#         self.buffer_len = 0
#         self.new_ep = False
#         self.next_new_ep = True
#         assert len(self.weight) > 0
#         print(mode)
#         print(f"ds_list: {self.ds_list}")
#         print(f"weight: {self.weight}")
#         print(f"bucket: {self.bucket}")
#         print(f"indices_num: {self.indices_num}")
            

#     def return_preprocess(self, rgb, inst, act):
#         if self.stack:
#             # import pdb; pdb.set_trace()
#             return frame_stack(torch.tensor(rgb).float() / 255), torch.tensor(inst) if self.embed_ready else list(inst), torch.tensor(act)
#         else:
#             # return torch.cat([torch.zeros((self.seq_len - 1, 3, 300, 300)), torch.tensor(rgb).float() / 255]), torch.cat([torch.zeros((self.seq_len - 1, 768)), torch.tensor(inst)]) if self.embed_ready  else list(inst), torch.tensor(act)
#             return torch.tensor(rgb).float() / 255, torch.tensor(inst) if self.embed_ready else list(inst), torch.tensor(act)


#     def __getindice__(self, index):
#         assert index >= 0 and index < self.len
#         bucket_len = len(self.bucket)
#         sub_ds_index = index
#         # import pdb; pdb.set_trace()
#         for idx in range(0, bucket_len-1):
#             if index < self.bucket[idx + 1] and index >= self.bucket[idx]:
#                 break
#             else:
#                 sub_ds_index -= self.bucket[idx + 1]
#         ds_name = self.ds_list[idx]
#         # print(f"ds_type: {self.mode}, ds_name: {ds_name}, sub_ds_index: {sub_ds_index}")
#         indice = self.indices[ds_name][sub_ds_index]
#         # print("indice:", indice)
#         ep_start_idx = indice[0]
#         ep_end_idx = indice[-1]
#         rgbs = []
#         acts = []
#         insts = []
#         act_mask= []
#         for id_idx in range(1, len(indice) - 1):
#             # import pdb; pdb.set_trace()
#             npz_name = str(indice[id_idx]) + '.npz'
#             rgb = np.load(os.path.join(self.ds_stats[ds_name]["path"]["rgb"], npz_name))['arr_0']
            
#             act = np.load(os.path.join(self.ds_stats[ds_name]["path"]["action"], npz_name))['arr_0']
            
#             if self.embed_ready:
#                 inst = np.load(os.path.join(self.ds_stats[ds_name]["path"]["inst_embed"], npz_name))['arr_0']
#             else:
#                 inst = np.load(os.path.join(self.ds_stats[ds_name]["path"]["instruction"], npz_name))['arr_0']
            
#             rgb, inst, act = self.return_preprocess(rgb, inst, act)
#             remainder = len(rgb) % self.seq_len
#             pad_num = self.seq_len - remainder
#             if remainder != 0:
#                 rgb = torch.cat([rgb, torch.zeros(pad_num, 3, 300, 300)])
#                 act = torch.cat([act, torch.zeros(pad_num, 2)])
#                 inst = torch.cat([inst, torch.zeros(pad_num, 768)])
                
#             action_mask = torch.zeros(len(rgb))
#             action_mask[-pad_num:] = 1
#             # print(f"loading rgb {npz_name}, length: {len(rgb)}")
#             # print(f"loading act {npz_name}, length: {len(act)}")
#             # print(f"loading inst {npz_name}, length: {len(inst)}")
#             # print(f"loading action_mask {npz_name}, length: {len(action_mask)}")
#             rgbs.append(rgb)
#             acts.append(act)
#             insts.append(inst)
#             act_mask.append(action_mask)
#         acts = torch.cat(acts)
#         rgbs = torch.cat(rgbs)
#         act_mask = torch.cat(act_mask)
#         if isinstance(insts[0], list):
#             tmp_insts = []
#             [tmp_insts.extend(sub_list) for sub_list in insts]
#             insts = tmp_insts
#         else:
#             insts = torch.cat(insts)
#         # import pdb; pdb.set_trace()
#         # print(f"rgbs: {len(rgbs)}")
#         # print(f"acts: {len(acts)}")
#         # print(f"insts: {len(insts)}")
#         # print(f"act_mask: {len(act_mask)}")
#         insts = insts[ep_start_idx:]
#         rgbs = rgbs[ep_start_idx:]
#         acts = acts[ep_start_idx:]
#         act_mask = act_mask[ep_start_idx:]
#         if ep_end_idx is not None:
#             rgbs = rgbs[:ep_end_idx]
#             acts = acts[:ep_end_idx]
#             insts = insts[:ep_end_idx]
#             act_mask = act_mask[:ep_end_idx]

#         # if len(insts) != self.batch_size or len(rgbs) != self.batch_size or len(acts) != self.batch_size or len(act_mask) != self.batch_size:
#         #     [print(len(rgb)) for rgb in rgbs] 
#         # import pdb; pdb.set_trace()
#         return rgbs, insts, acts, act_mask
            
        
#     def __getitem__(self, index):
#         return self.__getindice__(index)
    
#     def __len__(self):
#         return self.len


if __name__ == "__main__":
    from tqdm import tqdm
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    train_set = language_table_dataset_npz(mode="train", ds_type='mix')
    test_set = language_table_dataset_npz(mode="test", ds_type='mix')
    train_loader = DataLoader(dataset=train_set, batch_size=16, num_workers=16, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=16, num_workers=16, shuffle=True)
    for data in tqdm(train_loader):
        rgbs, inst_embeddings, actions, act_mask = data
    # for epoch in range(2):
    #     for idx, item in enumerate(train_loader):
    #         pbar.update(1)

    # train_loader = language_table_dataset(dataset_type="train", sub_data='mix', batch_size=1, weight=[1, 1, ] + [0] * 7)
    # test_loader = language_table_dataset(dataset_type="test", sub_data='mix', batch_size=1, weight=[1, 1, ] + [0] * 7)

    # pbar = tqdm(range(len(train_loader)))
    # for idx, item in enumerate(train_loader):
    #     pbar.update(1)
