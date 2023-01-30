import os
import re
import torch
import json
from torch.utils.data import IterableDataset, DataLoader
from itertools import cycle, islice
import random
from datasets import Dataset
from datasets import load_dataset, load_from_disk
from comm.comm_utils import *



class StreamDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, seq_length=1024):
        
        self.dataset = dataset
        
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        self.it = None
        self.iter_count = 0
        
        self.input_prefixs = ['']
        self.output_prefixs = ['Output: ', 'Output: ', 'Ans: ', 'A: ', 'Answer: ', 'Label: ', 'Label: ']
        self.sample_splitters = ['\n', '\n\n', '\n\n', '\n\n\n', '\n###\n', '\n---\n']
        self.answer_splitters = ['\n', '\n', '\n\n', ' ']
        
    def state_dict(self):
        return {
            'iter_count': self.iter_count,
        }
    
    def load_state_dict(self, state_dict):
        self.iter_count = state_dict['iter_count']
        self.dataset = self.dataset.skip(self.iter_count)
        
    def get_sequence(self):
        
        it = iter(self.dataset)
        
        while True:
            
            sample_splitter = random.choice(self.sample_splitters)
            answer_splitter = random.choice(self.answer_splitters)

            text_input = random.choice(self.input_prefixs)
            text_output = random.choice(self.output_prefixs)

            text_context = ""

            while True:
                
                instance = next(it)
                
                prompt = instance['inputs'].rstrip()
                target = instance['targets']
                
                if prompt[-1] == ':':
                    # prompt includes prefix
                    text_context += sample_splitter + text_input + prompt + " " + target
                else:
                    text_context += sample_splitter + text_input + prompt + answer_splitter + text_output + target
                    
                input_ids = self.tokenizer(text_context.strip())['input_ids']
                if len(input_ids) > self.seq_length:
                    break
                
            input_ids = input_ids[:self.seq_length]
            input_ids = torch.tensor(input_ids).long()

            yield {
                'input_ids': input_ids,
            }
            
                
    def get_stream(self):
        return cycle(self.get_sequence())
    
    def __iter__(self):
        if self.it is None:
            self.it = self.get_stream()
        return self.it
    
    
    
def get_p3_train_data_loader(args, tokenizer, num_workers=0, state_dict=None):
    
    dataset = load_dataset("Muennighoff/P3", split="train").shuffle(seed=args.seed)
    stream_dataset = StreamDataset(dataset, tokenizer, args.seq_length)
    
    if state_dict is not None:
        stream_dataset.load_state_dict(state_dict)
    
    train_data_loader = torch.utils.data.DataLoader(stream_dataset,
                                                    batch_size=args.batch_size * args.data_group_size,
                                                    shuffle=False,
                                                    num_workers=num_workers,
                                                    pin_memory=True,
                                                    collate_fn=None)
    return train_data_loader