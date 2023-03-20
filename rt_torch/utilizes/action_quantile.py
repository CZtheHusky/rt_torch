import torch
import numpy as np
import os
from tqdm import tqdm

def quantileBinning(path=None,
                    num_bins=256):
    files = os.listdir(path)
    all_actions = []
    for file in tqdm(files):
        file_path = os.path.join(path, file)
        arr = np.load(file_path)['arr_0']
        all_actions.append(arr)
    # import pdb; pdb.set_trace()
    all_actions = np.concatenate(all_actions)
    np.save(os.path.join(os.path.dirname(path), "all_actions.npy"), all_actions)

if __name__ == "__main__":
    quantileBinning("/raid/robotics_data/language_table_sim_npz/actions")