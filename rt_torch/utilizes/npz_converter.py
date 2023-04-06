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
import os
from tqdm import tqdm
import argparse
import shutil


os.environ["CUDA_VISIBLE_DEVICES"] = ''

parser = argparse.ArgumentParser()
parser.add_argument('--ds_name', default="language_table_sim", type=str)

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


def main(dataset_name):
    dataset_path = dataset_paths[dataset_name]
    npz_path = dataset_path + "_npz"
    npy_path = dataset_path + "_npy"
    iterative_walk(npz_path, npy_path)
    
    
    
def iterative_walk(source_path, dest_path):
    for root, dirs, files in os.walk(source_path):
        # get the path of the file relative to the source path
        rel_path = os.path.relpath(root, source_path)
        # create the corresponding path in the destination directory
        dest_dir = os.path.join(dest_path, rel_path)
        os.makedirs(dest_dir, exist_ok=True)
        print(f"root: {root}, dirs: {dirs}")
        for file in files:
            file_name = os.path.splitext(file)[0]
            src_file = os.path.join(root, file)
            if file.endswith(".npz"):
                # replace the file extension and copy the file
                npy_name = file_name + ".npy"
                dest_file = os.path.join(dest_dir, npy_name)
                src_data = np.load(src_file)['arr_0']
                np.save(dest_file, src_data)
            else:
                dest_file = os.path.join(dest_dir, file)
                shutil.copy(src_file, dest_file)
            print(f"Converting {src_file} to {dest_file}")
        # for dir in dirs:
        #     # get the path of the file relative to the source path
        #     tmp_src = os.path.join(source_path, dir)
        #     # create the corresponding path in the destination directory
        #     tmp_dest = os.path.join(dest_path, dir)
        #     iterative_walk(tmp_src, tmp_dest)

    

    


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.ds_name)
