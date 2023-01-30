import os
import re
import torch
import json
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from itertools import cycle, islice
import random
from datasets import Dataset
from datasets import load_dataset, load_from_disk
from comm.comm_utils import *


from itertools import islice
from random import randint

SHOW_DATA = int(os.environ.get('SHOW_DATA', 1))


class StreamDatasetList(IterableDataset):
    def __init__(self, task_names, datasets, sample_probs, tokenizer, seq_length=1024, print_sample_every_n=64,):
        
        self.task_names = task_names
        self.datasets = datasets
        self.sample_probs = sample_probs
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.print_sample_every_n = print_sample_every_n
        
        self.it = None
        
    def state_dict(self):
        return {}
    
    def load_state_dict(self, state_dict):
        pass
        
    def get_sequence(self):
        
        iterators = [cycle(d.get_sequence()) for d in self.datasets]
        prob_ths = np.cumsum([p / sum(self.sample_probs) for p in self.sample_probs])
        
        global_i = 0
        
        while True:
            
            p = random.random()
            
            for task_name, it, th in zip(self.task_names, iterators, prob_ths):
                if p < th:
                    
                    inputs = next(it)
                    
                    if SHOW_DATA:
                        if global_i % self.print_sample_every_n == 0:
                            print(p, th)
                            print(f"**{task_name}**:", self.tokenizer.decode(inputs['input_ids']))
                        
                    yield inputs
                    global_i += 1
                    break
                
    def get_stream(self):
        return cycle(self.get_sequence())
    
    def __iter__(self):
        if self.it is None:
            self.it = self.get_stream()
        return self.it
    
    
def name_to_dataset(task, tokenizer, args):
    
    if task != '':
        if task == 'natural_instructions' or task == 'ni':
            from .natural_instructions import StreamDataset
            dataset = StreamDataset('./natural-instructions/', tokenizer, args.seq_length)
        elif task == 'p3':
            from .p3 import StreamDataset
            data = load_dataset("Muennighoff/P3", split="train").shuffle(seed=args.seed)
            dataset = StreamDataset(data, tokenizer, args.seq_length)
        elif task == 'pile':
            from .pile import StreamDataset
            data = load_dataset('the_pile', split="train", streaming=True).shuffle(buffer_size=1000_000, seed=args.seed).with_format("torch")
            # data = load_dataset('the_pile', split="train").shuffle(seed=args.seed)
            dataset = StreamDataset(data, tokenizer, args.seq_length)
        elif task == 'cot':
            from .cot import StreamDataset
            dataset = StreamDataset('./data/mmlu-cot.json', tokenizer, args.seq_length)
        else:
            if 'p3' in task:
                from .p3 import StreamDataset
            else:
                from .pile import StreamDataset
            data = load_dataset("json", data_files=task, split="train", streaming=True).shuffle(buffer_size=10_000, seed=args.seed)
            dataset = StreamDataset(data, tokenizer, args.seq_length)
        
    return dataset

    
def get_train_data_loader(args, tokenizer, num_workers=1, state_dict=None):
    
    task_list = args.task_name.split(',')
    task_names = []
    datasets = []
    probs = []
    
    print('data_utils: parse task_list')
    
    for task in task_list:
        if ':' in task:
            task, prob = task.strip().split(':')
            prob = float(prob)
        else:
            task = task.strip()
            prob = 1.0
            
        dataset = name_to_dataset(task, tokenizer, args)
            
        print('data_utils:', task, prob)
    
        task_names.append(task)
        datasets.append(dataset)
        probs.append(prob)
    
    stream_dataset = StreamDatasetList(
        task_names, datasets, probs,
        tokenizer=tokenizer, seq_length=args.seq_length)
    
    if state_dict is not None:
        stream_dataset.load_state_dict(state_dict)
    
    train_data_loader = torch.utils.data.DataLoader(stream_dataset,
                                                    batch_size=args.batch_size * args.data_group_size,
                                                    shuffle=False,
                                                    num_workers=num_workers,
                                                    pin_memory=True,
                                                    collate_fn=None)
    
    print('data_utils: get train_data_loader')
    
    return train_data_loader

