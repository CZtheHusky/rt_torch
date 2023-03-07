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


def build_language_table_ds(split=0.9, batch_size=16, rgb_list=True, seq_len=6, seed=100):
    if seed:
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
            current_slice.append(None)
            indices[k].append(current_slice)
        # import pdb; pdb.set_trace()
        total_indice_num = len(indices[k])
        train_indice_num = int(total_indice_num * split)
        indices_index[k] = {"train": None, "test": None}
        total_indexes = np.arange(total_indice_num)
        if seed:
            np.random.shuffle(total_indexes)
        indices_index[k]["train"] = total_indexes[:train_indice_num]
        indices_index[k]["test"] = total_indexes[train_indice_num:]

    train_set = language_table_dataset_npz(mode="train", indices_index=indices_index, indices=indices, ds_stats=ds_stats, rgb_list=rgb_list,seq_len=seq_len, batch_size=batch_size)
    test_set = language_table_dataset_npz(mode="test", indices_index=indices_index, indices=indices, ds_stats=ds_stats, rgb_list=rgb_list, seq_len=seq_len, batch_size=batch_size)
    return train_set, test_set

class language_table_dataset_npz(Dataset):
    def __init__(self, mode, indices_index, indices, ds_stats, rgb_list=True, seq_len=6, batch_size=96) -> None:
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
        self.batch_size= batch_size
        self.bucket = [0]
        for k, v in self.indices_index.items():
            self.len += len(v[mode])
            self.bucket.append(self.len)
            self.ds_list.append(k)
            self.sub_ds_len.append(len(v[mode]))
            

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
        assert sub_ds_index >=0 and sub_ds_index <= self.sub_ds_len[idx]
        indice_idx = self.indices_index[ds_name][self.mode][sub_ds_index]
        indice = self.indices[ds_name][indice_idx]
        # indice_detail = f"ds_type: {self.mode}, ds_name: {ds_name}, sub_ds_index: {sub_ds_index}, indice:{indice}\n"
        ep_start_idx = indice[0]
        ep_end_idx = indice[-1]
        rgbs = []
        acts = []
        insts = []
        # loading_details = ""
        for id_idx in range(1, len(indice) - 1):
            # import pdb; pdb.set_trace()
            npz_name = str(indice[id_idx]) + '.npz'
            rgb = np.load(os.path.join(self.ds_stats[ds_name]["path"]["rgb"], npz_name))['arr_0']
            act = np.load(os.path.join(self.ds_stats[ds_name]["path"]["action"], npz_name))['arr_0']
            inst = np.load(os.path.join(self.ds_stats[ds_name]["path"]["inst_embed"], npz_name))['arr_0']
            # loading_details += f"loading {npz_name}, rgb: {len(rgb)}, inst: {len(inst)}, act: {len(act)}\n"
            rgb, inst, act = self.return_preprocess(rgb, inst, act)   
            rgbs.append(rgb)
            acts.append(act)
            insts.append(inst)
        acts = torch.cat(acts)
        if isinstance(insts[0], list):
            tmp_insts = []
            [tmp_insts.extend(sub_list) for sub_list in insts]
            insts = tmp_insts
        else:
            insts = torch.cat(insts)
        # import pdb; pdb.set_trace()
        # padding_detail = "padding: "
        if ep_start_idx < self.seq_len - 1:
            rgbs[0] = torch.cat([torch.zeros(self.seq_len - ep_start_idx - 1, 3, 300, 300), rgbs[0]])
            insts = torch.cat([torch.zeros(self.seq_len - ep_start_idx - 1, 768), insts])
            # padding_detail += f"length {self.seq_len - ep_start_idx}\n"
        else:
            # padding_detail += f"None\n"
            rgbs[0] = rgbs[0][ep_start_idx - self.seq_len + 1:]
            insts = insts[ep_start_idx - self.seq_len + 1:]
        acts = acts[ep_start_idx:]
        if ep_end_idx is not None:
            rgbs[-1] = rgbs[-1][:ep_end_idx]
            acts = acts[:ep_end_idx]
            insts = insts[:ep_end_idx]
        # total = 0
        # for rgb in rgbs:
        #     total += len(rgb)
        # post_process_detail = f"post processed, rgb: {total}, inst: {len(inst)}, act: {len(act)}\n"
        # # print(indice_detail + loading_details + padding_detail + post_process_detail)
        # if total != len(insts) or total != len(acts) + 5:
        #     print(indice_detail + loading_details + padding_detail + post_process_detail)
            # import pdb; pdb.set_trace()
        split_idx = torch.ones(self.batch_size) * -1
        for idx, vid in enumerate(rgbs):
            split_idx[idx] = len(vid)
        rgbs = torch.cat(rgbs)
        return rgbs, insts, acts, split_idx
            
        
    def __getitem__(self, index):
        return self.__getindice__(index)
     
    def __len__(self):
        return self.len


# class language_table_dataset_npz(Dataset):
#     def __init__(self, mode: str, ds_type: str, split=0.9, weight=None, rgb_list=True, stack_num=6) -> None:
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
#         self.rgb_list = rgb_list
#         self.stack_num = stack_num



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



#     # def __loaditem__(self, index, ds=None):
#     #     while self.buffer_len < self.batch_size:
#     #         if ds is None:
#     #             choice = np.random.choice(len(self.weight), p=self.weight)
#     #             ds_name = self.ds_list[choice]
#     #         else:
#     #             ds_name = ds
#     #         npz_name = str(self.ds_stats[ds_name]["current_idx"] % self.sub_size[ds_name] + self.index_range[ds_name][0]) + ".npz"
#     #         if self.embed_ready:
#     #             inst = np.load(os.path.join(self.ds_stats[ds_name]["path"]["inst_embed"], npz_name))['arr_0']
#     #         else:
#     #             inst = np.load(os.path.join(self.ds_stats[ds_name]["path"]["instruction"], npz_name))['arr_0']
#     #         rgb = np.load(os.path.join(self.ds_stats[ds_name]["path"]["rgb"], npz_name))['arr_0']
#     #         act = np.load(os.path.join(self.ds_stats[ds_name]["path"]["action"], npz_name))['arr_0']
#     #         self.buffer_len += len(rgb)
#     #         # print(f"add {len(rgb)} into buffer: {self.buffer_len}")
#     #         rgb, inst, act = self.return_preprocess(rgb, inst, act)
#     #         self.buffer["rgb"].append(rgb)
#     #         self.buffer["inst"].append(inst)
#     #         self.buffer['act'].append(act)
#     #         self.ds_stats[ds_name]["current_idx"] += 1
#     #         self.ds_stats[ds_name]["current_idx"] %= self.sub_size[ds_name]
            

#     def return_preprocess(self, rgb, inst, act):
#         return torch.tensor(rgb).float() / 255, torch.tensor(inst) if self.embed_ready else list(inst), torch.tensor(act)

#     # def __getbuffer__(self):
#     #     self.buffer_len -= self.batch_size
#     #     # print("remove from buffer:", self.buffer_len)
#     #     tmp_buffer = {"rgb": [], "inst": [], "act": []}
#     #     self.new_ep = self.next_new_ep
#     #     if self.buffer_len > 0:
#     #         self.next_new_ep = False
#     #         for k, v in self.buffer.items():
#     #             if k != "act":
#     #                 reserve_len = self.buffer_len + self.stack_num - 1
#     #                 if len(v[-1]) < reserve_len:
#     #                     # import pdb; pdb.set_trace()
#     #                     if k == "rgb":
#     #                         tmp_buffer[k] = [torch.cat([torch.zeros([reserve_len - len(v[-1]), 3, 300, 300]), v[-1]])]
#     #                     else:
#     #                         tmp_buffer[k] = [torch.cat([torch.zeros([reserve_len - len(v[-1]), 768]), v[-1]])]
#     #                 else:
#     #                     tmp_buffer[k] = [v[-1][-reserve_len:]]
#     #             else:
#     #                 tmp_buffer[k] = [v[-1][-(self.buffer_len):]]
#     #             v[-1] = v[-1][:-self.buffer_len]
#     #     else:
#     #         self.next_new_ep = True
#     #     if self.rgb_list:
#     #         rgb = self.buffer["rgb"]
#     #     else:
#     #         rgb = torch.cat(self.buffer["rgb"], dim=0)
#     #     if isinstance(self.buffer["inst"][0], list):
#     #         inst = []
#     #         [inst.extend(sub_list) for sub_list in self.buffer["inst"]]
#     #     else:
#     #         inst = torch.cat(self.buffer["inst"], dim=0)
#     #     act = torch.cat(self.buffer["act"], dim=0)
#     #     self.buffer = tmp_buffer
#     #     return rgb, inst, act
    


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
#         for id_idx in range(1, len(indice) - 1):
#             # import pdb; pdb.set_trace()
#             npz_name = str(indice[id_idx]) + '.npz'
#             rgb = np.load(os.path.join(self.ds_stats[ds_name]["path"]["rgb"], npz_name))['arr_0']
#             # print(f"loading rgb {npz_name}, length: {len(rgb)}")
#             act = np.load(os.path.join(self.ds_stats[ds_name]["path"]["action"], npz_name))['arr_0']
#             # print(f"loading act {npz_name}, length: {len(act)}")
#             if self.embed_ready:
#                 inst = np.load(os.path.join(self.ds_stats[ds_name]["path"]["inst_embed"], npz_name))['arr_0']
#             else:
#                 inst = np.load(os.path.join(self.ds_stats[ds_name]["path"]["instruction"], npz_name))['arr_0']
#             # print(f"loading inst {npz_name}, length: {len(inst)}")
#             rgb, inst, act = self.return_preprocess(rgb, inst, act)   
#             rgbs.append(rgb)
#             acts.append(act)
#             insts.append(inst)
#         acts = torch.cat(acts)
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
#         if ep_start_idx < self.stack_num - 1:
#             rgbs[0] = torch.cat([torch.zeros(self.stack_num - ep_start_idx - 1, 3, 300, 300), rgbs[0]])
#             insts = torch.cat([torch.zeros(self.stack_num - ep_start_idx - 1, 768), insts])
#         else:
#             rgbs[0] = rgbs[0][ep_start_idx - self.stack_num + 1:]
#             insts = insts[ep_start_idx - self.stack_num + 1:]
#         acts = acts[ep_start_idx:]
#         if ep_end_idx is not None:
#             rgbs[-1] = rgbs[-1][:ep_end_idx]
#             acts = acts[:ep_end_idx]
#             insts = insts[:ep_end_idx]
#         # total = 0
#         # for rgb in rgbs:
#         #     total += len(rgb)

#         # if total != len(insts) + 5:
#         #     [print(len(rgb)) for rgb in rgbs] 
#         #     import pdb; pdb.set_trace()
#         return rgbs, insts, acts
            
        
#     def __getitem__(self, index):
#         return self.__getindice__(index)
    

#     # def __getitem__(self, index):
#     #     if self.ds_type == 'mix':
#     #         self.__loaditem__(index=index)
#     #     else:
#     #         ds_name = self.ds_type
#     #         self.__loaditem__(index, ds_name)
#     #     rgb, inst, act = self.__getbuffer__()
#     #     if self.embed_ready:
#     #         return rgb, inst, act, self.new_ep
#     #     else:
#     #         return rgb, inst, act, self.new_ep
    
    
            
#     def __len__(self):
#         return self.len


if __name__ == "__main__":
    from tqdm import tqdm
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    train_set, test_set = build_language_table_ds(split=0.9, batch_size=96, rgb_list=True, seq_len=6, seed=None)
    train_loader = DataLoader(dataset=train_set, batch_size=1, num_workers=32, shuffle=False)
    test_loader = DataLoader(dataset=test_set, batch_size=1, num_workers=32, shuffle=False)
    for item in tqdm(train_loader):
        pass
    for item in tqdm(test_loader):
        pass

    # train_loader = language_table_dataset(dataset_type="train", sub_data='mix', batch_size=1, weight=[1, 1, ] + [0] * 7)
    # test_loader = language_table_dataset(dataset_type="test", sub_data='mix', batch_size=1, weight=[1, 1, ] + [0] * 7)

    # pbar = tqdm(range(len(train_loader)))
    # for idx, item in enumerate(train_loader):
    #     pbar.update(1)
