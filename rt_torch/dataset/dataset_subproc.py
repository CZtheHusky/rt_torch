import numpy as np
import tensorflow_datasets as tfds
import torch
from collections import deque
import tensorflow as tf
import cv2 as cv
import time
import torch.nn.functional as F
from multiprocessing import Process, Pipe, Queue, Manager, Lock
from multiprocessing.connection import Connection
import os
import tree
from dataset import language_table_dataset


def task_running(q: Queue, terminal, len_cal, lock, dataset_type: str, sub_data: str, batch_size: int, weight=None, seed=0):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    dataset = language_table_dataset(dataset_type=dataset_type, sub_data=sub_data, batch_size=batch_size, weight=weight, seed=seed)
    while True:
        if terminal.get("close", None) is not None:
            del dataset
            terminal[seed] = 1
            break
        elif seed == 0 and len(len_cal) == 1:
            len_cal.append(len(dataset))
        else:
            ret = dataset.get_item()
            lock.acquire()
            if isinstance(ret, bool):
                q.put(seed)
                lock.release()
                break
            else:
                if q.full():
                    print(f"process: {seed}, queue is full")
                q.put(ret)
                lock.release()


class SubProcLTDataset():
    def __init__(self, dataset_type: str, sub_data: str, batch_size: int, weight=None, seed=0, num_threads=8):
        self.dataset_type = dataset_type
        self.sub_data = sub_data
        self.batch_size = batch_size
        self.weight = weight
        self.seed = seed
        self.num_threads = num_threads
        self.num = 10000
        self.process_status = {}
        

    def _create_process(self):
        self.manager = Manager()
        terminal = self.manager.dict()
        len_cal = Manager().list()
        lock = Lock()
        q = Queue(self.num_threads * 4)
        producer_processes = []
        for i in range(self.num_threads):
            p = Process(target=task_running, args=(q, terminal, len_cal, lock, self.dataset_type, self.sub_data, self.batch_size, self.weight, self.seed + i))
            producer_processes.append(p)
            p.start()
        self.producer_processes = producer_processes
        self.q = q
        self.terminal = terminal
        self.len_cal = len_cal
        self.lock = lock

    def __len__(self):
        if self.num is None:
            self.len_cal.append(0)
            while len(self.len_cal) != 2:
                pass
            num = self.len_cal[1]
            self.num = num
            while len(self.len_cal) != 0:
                self.len_cal.pop()
            return num * self.num_threads
        else:
            return self.num

    def get_one(self):
        while True:
            # print("trying to get data")
            ret = self.q.get()
            if isinstance(ret, int):
                self.process_status[ret] = False
            else:
                rgbs, instructions, actions = ret
                return rgbs, instructions, actions

    def __iter__(self):
        while True:
            # print(self.q.qsize())
            ret = self.q.get()
            if isinstance(ret, int):
                self.process_status[ret] = False
                done = True
                for i in range(self.num_threads):
                    if not self.process_status.get(i, None):
                        done = False
                        break
                if done:
                    return
            else:
                rgbs, instructions, actions = ret
                yield rgbs, instructions, actions

    def __getitem__(self, idx):
        raise NotImplementedError

    def __del__(self):
        self.terminal['close'] = True
        while True:
            kill_flag = True
            for i in range(self.num_threads):
                if not self.terminal.get(i, None):
                    kill_flag = False
                    break
            if kill_flag:
                break
        for p in self.producer_processes:
            p.join()
        del self.producer_processes
        del self.q
        del self.lock
        del self.terminal
        del self.len_cal
        del self.manager

if __name__ == "__main__":
    from tqdm import tqdm
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    train_loader = SubProcLTDataset(dataset_type="train", sub_data='mix', batch_size=16, weight=[0, 0, 0, 0, 0, 0, 1, 0, 0])
    test_loader = SubProcLTDataset(dataset_type="test", sub_data='mix', batch_size=16, weight=[0, 0, 0, 0, 0, 0, 1, 0, 0])
    train_loader._create_process()
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
