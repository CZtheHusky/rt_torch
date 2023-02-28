import numpy as np
import tensorflow_datasets as tfds
import torch
from collections import deque
import tensorflow as tf
import cv2 as cv
import time
import torch.nn.functional as F
from multiprocessing import Process, Pipe, Queue
from multiprocessing.connection import Connection
import os
import tree
from utilize import nlp_inst_decoder



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

def frame_stack(images):
    tmp = [torch.zeros((3, 300, 300))] * 5
    frames = deque(tmp, maxlen=6)
    rgbs = []
    for image in images:
        frames.append(image)
        rgbs.append(torch.stack(list(frames)))
    return torch.stack(rgbs) # EP_LEN 6 3 300 300 

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

sub_datas = ["mix"] + list(dataset_paths.keys())

class language_table_dataset():
    def __init__(self, dataset_type: str, sub_data: str, batch_size: int, weight=None, seed=0):
        assert sub_data in sub_datas
        assert dataset_type in ["train", "test", "all"]
        assert batch_size > 0
        assert weight is None or len(weight) == 9
        self.size = 0
        self.batch_size = batch_size
        self.frame_stack = 6
        self.seed = seed
        np.random.seed(seed)
        # self.inst_time = 0
        # self.rgb_time = 0
        if sub_data == "mix":
            builders = {key: tfds.builder_from_directory(value) for key, value in dataset_paths.items()}
            if dataset_type == "train":
                self.datasets = {key: builder.as_dataset(split='train[:95%]').prefetch(32) for key, builder in
                                builders.items()}
            elif dataset_type == "test":
                self.datasets = {key: builder.as_dataset(split='train[95%:]').prefetch(32) for key, builder in
                                builders.items()}
            else:
                self.datasets = {key: builder.as_dataset(split='train').prefetch(32) for key, builder in
                                builders.items()}
            dataset_length = []
            self.dataset_list = []
            for key, dataset in self.datasets.items():
                dataset_length.append(len(dataset))
                self.dataset_list.append(key)
            self.dataset_length = np.array(dataset_length)
            if weight is None:
                self.weights = self.dataset_length / np.sum(self.dataset_length)
                self.size = np.sum(dataset_length)
                assert np.sum(self.weights) == 1
            else:
                self.weights = np.array(weight) / np.sum(weight)
                for l, w in zip(dataset_length, self.weights):
                    if w != 0:
                        self.size += l
            self.sub_data = 0
            self.dataset_iters = {key: iter(dataset) for key, dataset in self.datasets.items()}
        else:
            builder = tfds.builder_from_directory(dataset_paths[sub_data])
            if dataset_type == "train":
                self.datasets = builder.as_dataset(split='train[:95%]').prefetch(32)
            elif dataset_type == "test":
                self.datasets = builder.as_dataset(split='train[95%:]').prefetch(32)
            else:
                self.datasets = builder.as_dataset(split='train').prefetch(32)
            self.dataset_iters = iter(self.datasets)
            self.size += len(self.datasets)
            self.sub_data = sub_data
        self.rgbBuffer = []
        self.instBuffer = []
        self.actBuffer = []
        self.EpRgbBuffer = []
        self.EpInstBuffer = []
        self.EpActBuffer = []
        self.bufferSize = 0
        self.epSize = 0

    def decode_episode(self, episode):
        """
        FeaturesDict({
            'episode_id': tf.string,
            'steps': Dataset({
                'action': Tensor(shape=(2,), dtype=tf.float32),
                'is_first': tf.bool,
                'is_last': tf.bool,
                'is_terminal': tf.bool,
                'observation': FeaturesDict({
                    'effector_target_translation': Tensor(shape=(2,), dtype=tf.float32),
                    'effector_translation': Tensor(shape=(2,), dtype=tf.float32),
                    'instruction': Tensor(shape=(512,), dtype=tf.int32),
                    'rgb': Image(shape=(360, 640, 3), dtype=tf.uint8),
                }),
                'reward': Scalar(shape=(), dtype=tf.float32),
            }),
        })
        """
        ep = tree.map_structure(lambda *xs: np.stack(xs), *episode['steps'])
        ep_inst = [nlp_inst_decoder(x) for x in ep['observation']['instruction']]
        self.EpActBuffer.append(ep['action'])
        self.EpRgbBuffer.append(ep['observation']['rgb'])
        self.EpInstBuffer.append(ep_inst)
        self.epSize += len(ep['action'])

    def safe_get(self):
        if self.sub_data == 0:
            while True:
                try:
                    choice = np.random.choice(9, p=self.weights)
                    dataset_name = self.dataset_list[choice]
                    episode = self.dataset_iters[dataset_name].get_next()
                    break
                except Exception as e:
                    self.weights[choice] = 0
                    if np.sum(self.weights) == 0:
                        raise StopIteration
                    else:
                        self.weights = self.weights / np.sum(self.weights)
        else:
            try:
                episode = self.dataset_iters.get_next()
            except Exception as e:
                raise StopIteration
        self.decode_episode(episode)
        # return episode

    def extractBuffer(self, batch_size):
        extract_size = batch_size
        actions = self.actBuffer[0][:extract_size]
        self.actBuffer[0] = self.actBuffer[0][extract_size:]

        instructions = self.instBuffer[:extract_size]
        self.instBuffer = self.instBuffer[extract_size:]

        rgbs = self.rgbBuffer[0][:extract_size]
        self.rgbBuffer[0] = self.rgbBuffer[0][extract_size:]
        self.bufferSize -= len(rgbs)
        return rgbs, instructions, actions

    def extractEp(self):
        self.bufferSize += self.epSize
        # import pdb
        # pdb.set_trace()
        for actions, rgbs, instructions in zip(self.EpActBuffer, self.EpRgbBuffer, self.EpInstBuffer):
            # import pdb
            # pdb.set_trace()
            actions = torch.tensor(actions)
            rgbs = rgb_preprocess(torch.tensor(rgbs)) # EP_LEN * 3 * 300 * 300
            rgbs = frame_stack(rgbs)
            rgbs = torch.permute(rgbs, (0, 2, 1, 3, 4))  # EP_LEN 6 3 300 300 
            self.actBuffer.append(actions)
            self.rgbBuffer.append(rgbs)
            self.instBuffer.extend(instructions)
        self.EpRgbBuffer = []
        self.EpInstBuffer = []
        self.EpActBuffer = []
        self.epSize = 0
        if len(self.actBuffer) > 1:
            try:
                self.actBuffer = [torch.cat(self.actBuffer, dim=0)]
                self.rgbBuffer = [torch.cat(self.rgbBuffer, dim=0)]
            except Exception as e:
                import pdb
                pdb.set_trace()


    def get_item(self):
        while True:
            try:
                while (self.epSize + self.bufferSize) < self.batch_size:
                    self.safe_get()
                if len(self.EpRgbBuffer) > 0:
                    self.extractEp()
                rgbs, instructions, actions = self.extractBuffer(self.batch_size)
                return [rgbs, instructions, actions]
                # yield self.safe_get()
            except StopIteration:
                self.dataset_iters = {key: iter(dataset) for key, dataset in self.datasets.items()}
                return False

    def __len__(self):
        return self.size

    def __iter__(self):
        while True:
            try:
                while (self.epSize + self.bufferSize) < self.batch_size:
                    self.safe_get()
                if len(self.EpRgbBuffer) > 0:
                    self.extractEp()
                rgbs, instructions, actions = self.extractBuffer(self.batch_size)
                yield [rgbs, instructions, actions]
                # yield self.safe_get()
            except StopIteration:
                self.dataset_iters = {key: iter(dataset) for key, dataset in self.datasets.items()}
                return


    def __getitem__(self, idx):
        raise NotImplementedError

if __name__ == "__main__":
    from tqdm import tqdm
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    train_loader = language_table_dataset(dataset_type="train", sub_data='mix', batch_size=16, weight=[0, 1, 0, 0, 0, 0, 0, 0, 0])
    test_loader = language_table_dataset(dataset_type="test", sub_data='mix', batch_size=16, weight=[0, 1, 0, 0, 0, 0, 0, 0, 0])
    print("get pbar")
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
