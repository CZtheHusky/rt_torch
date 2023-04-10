import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import deque
import cv2 as cv
import time
import torch.nn.functional as F
from multiprocessing import Process, Pipe, Queue
from multiprocessing.connection import Connection
import os
from rt_torch.tokenizers.action_tokenizer import ActionTokenizer

dataset_paths = {
    'language_table': '/raid/robotics_data/language_table_npy',
    'language_table_sim': '/raid/robotics_data/language_table_sim_npy',
    'language_table_blocktoblock_sim': '/raid/robotics_data/language_table_blocktoblock_sim_npy',
    'language_table_blocktoblock_4block_sim': '/raid/robotics_data/language_table_blocktoblock_4block_sim_npy',
    'language_table_blocktoblock_oracle_sim': '/raid/robotics_data/language_table_blocktoblock_oracle_sim_npy',
    'language_table_blocktoblockrelative_oracle_sim': '/raid/robotics_data/language_table_blocktoblockrelative_oracle_sim_npy',
    'language_table_blocktoabsolute_oracle_sim': '/raid/robotics_data/language_table_blocktoabsolute_oracle_sim_npy',
    'language_table_blocktorelative_oracle_sim': '/raid/robotics_data/language_table_blocktorelative_oracle_sim_npy',
    'language_table_separate_oracle_sim': '/raid/robotics_data/language_table_separate_oracle_sim_npy',
}

sub_datas = ["mix"] + list(dataset_paths.keys())

text_encoder_path = {"use":"/inst_embedding_use",
                 "t5":"/inst_embedding_t5",
                  "use_tf": "/inst_embedding_use_tf"}

def build_language_table_ds(args, split=0.9, dumb=False):
    seq_len = args.seq_len
    sub_data = args.sub_data
    text_encoder = args.text_encoder
    seed = args.seed
    if seed:
        np.random.seed(seed)
    ds_stats = {}
    for idx, (k, v) in enumerate(dataset_paths.items()):
        if sub_data != "mix":
            if k != sub_data:
                continue
        obs_path = v + "/observations"
        act_path = v + "/actions"
        inst_path = obs_path + "/instructions"
        inst_embed_path = obs_path + text_encoder_path[text_encoder]
        rgb_path = obs_path + "/rgbs"
        rew_path = v + "/rewards"
        traj_len_path = v + "/traj_len.npy"
        quantile_path = v + "/all_actions.npy"
        try:
            traj_len = np.load(traj_len_path)
            if os.path.exists(inst_embed_path):
                assert 1 == len(set([len(os.listdir(act_path)), len(os.listdir(inst_path)), len(os.listdir(rgb_path)), len(os.listdir(rew_path)), len(os.listdir(inst_embed_path)), len(traj_len)]))
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
                    "inst_embed": inst_embed_path if inst_embed_status else None,
                    "traj_len": traj_len_path,
                },
                "num_ep": size,
                "current_idx": 0,
                "action_path": quantile_path,
            }
        except Exception as e:
            # import pdb; pdb.set_trace()
            pass

    indices = {}
    indices_index = {}
    for k, v in ds_stats.items():
        indices[k] = []
        traj_len_path = v["path"]["traj_len"]
        traj_len = np.load(traj_len_path)
        # traj_len = traj_len[:200]
        # import pdb; pdb.set_trace()
        for traj_idx, traj_l in enumerate(traj_len):
            for step_idx in range(traj_l):
                indices[k].append([traj_idx, step_idx])
        # import pdb; pdb.set_trace()
        total_indice_num = len(indices[k])
        train_indice_num = int(total_indice_num * split)
        indices_index[k] = {"train": None, "test": None}
        total_indexes = np.arange(total_indice_num)
        if seed:
            np.random.shuffle(total_indexes)
        indices_index[k]["train"] = total_indexes[:train_indice_num]
        indices_index[k]["test"] = total_indexes[train_indice_num:]
    # import pdb; pdb.set_trace()
    train_set = language_table_dataset_npy(args, mode="train", indices_index=indices_index, indices=indices, ds_stats=ds_stats, seq_len=seq_len, dumb=dumb, text_encoder=text_encoder)
    test_set = language_table_dataset_npy(args, mode="test", indices_index=indices_index, indices=indices, ds_stats=ds_stats, seq_len=seq_len, dumb=dumb, text_encoder=text_encoder)
    return train_set, test_set

class language_table_dataset_npy(Dataset):
    def __init__(self, args, mode, indices_index, indices, ds_stats, seq_len=6, dumb=False, text_encoder="t5") -> None:
        super().__init__()
        self.mode = mode
        self.ds_stats = ds_stats
        self.indices_index = indices_index
        self.indices = indices
        self.seq_len = seq_len
        self.sub_ds_len = []
        self.ds_list = []
        self.len =0
        self.bucket = [0]
        self.dumb = dumb
        self.dumb_tmp = None
        self.text_embed_dim = 768 if text_encoder == "t5" else 512
        self.action_tokenizer = ActionTokenizer(num_action_bin=args.vocab_size, action_path=self.ds_stats[args.sub_data]["action_path"], quantile=args.quantile)
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
        traj_idx = indice[0]
        step_idx = indice[1]
        npy_name = str(traj_idx) + '.npy'
        rgb = np.load(os.path.join(self.ds_stats[ds_name]["path"]["rgb"], npy_name), mmap_mode='r')
        act = np.load(os.path.join(self.ds_stats[ds_name]["path"]["action"], npy_name), mmap_mode='r')
        inst = np.load(os.path.join(self.ds_stats[ds_name]["path"]["inst_embed"], npy_name), mmap_mode='r')
        rgb = torch.tensor(rgb)
        act = torch.tensor(act)
        inst = torch.tensor(inst)
        act_mask = torch.ones(self.seq_len, 2)
        if step_idx < self.seq_len - 1:
            rgb_seq = torch.cat([rgb[0:step_idx + 1], torch.zeros(self.seq_len - step_idx - 1, *rgb.shape[1:])])
            inst_seq = torch.cat([inst[0:step_idx + 1], torch.zeros(self.seq_len - step_idx - 1, self.text_embed_dim)])
            act = torch.cat([act[0:step_idx + 1], torch.zeros(self.seq_len - step_idx - 1, act.shape[1])])
            act_mask[step_idx + 1:] = 0
            # padding_detail += f"length {self.seq_len - ep_start_idx}\n"
        else:
            # padding_detail += f"None\n"
            rgb_seq = rgb[step_idx - self.seq_len + 1:step_idx + 1]
            inst_seq = inst[step_idx - self.seq_len + 1:step_idx + 1]
            act = act[step_idx - self.seq_len + 1:step_idx + 1]
        # import pdb; pdb.set_trace()
        act = self.action_tokenizer.discretize(act)
        return rgb_seq, inst_seq, act, act_mask
            
        
    def __getitem__(self, index):
        if self.dumb:
            if not self.dumb_tmp:
                self.dumb_tmp = self.__getindice__(0)
            return self.dumb_tmp
        else:
            return self.__getindice__(index)
     
    def __len__(self):
        return self.len


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
