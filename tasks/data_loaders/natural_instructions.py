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
import numpy as np
import pickle 
from collections import OrderedDict

class StreamEvalDataset(IterableDataset):
    default_doc_separator = ''
    def __init__(self, data_path, tokenizer, dev_split_path, seq_length=1024, doc_separator=None):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.doc_separator = doc_separator or StreamEvalDataset.default_doc_separator
        self.it = None
        self.iter_count = 0
        self.buffer_tokens = []
        
        
        # self.input_prefixs = ['Input: ', 'Given: ', 'Context: ', 'Example: ', 'Question: ', '', '', '', '', '',]
        # self.output_prefixs = ['Output: ', 'Output: ', 'Ans: ', 'A: ', 'Answer: ', 'Label: ', 'Label: ']
        # self.sample_splitters = ['\n', '\n\n', '\n\n', '\n\n\n', '\n###\n', '\n---\n']
        # self.answer_splitters = ['\n', '\n', '\n\n']
        
        self.input_prefixs = ['Input: ']
        self.output_prefixs = ['Output: ']
        self.sample_splitters = ['\n\n']
        self.answer_splitters = ['\n']
        
        
            
        with open(dev_split_path, "rb") as f:
            self.dev_split = pickle.load(f)
        
        self.train_splits = []
        with open(os.path.join(data_path, 'splits/default/train_tasks.txt')) as f:
            for line in f:
                if line.strip() == '':
                    continue
                self.train_splits.append(line.strip() + '.json')
        
        self.task_paths = [
            os.path.join(data_path, 'tasks', p) for p in os.listdir(os.path.join(data_path, 'tasks')) if p.endswith('.json') and p in self.train_splits
        ]
        self.tasks = OrderedDict()
        self.classification_tasks = OrderedDict()
        for task_path in self.task_paths:
            with open(task_path) as f:
                task = json.load(f)
                
            output_space = set()
            is_classification = True
            
            # remove small tasks 
            if len(task['Instances']) < 100:
                continue 
                        
            # only keep dev data for eval
            #task['Instances'] = [obj for i, obj in enumerate(task['Instances']) if i in self.dev_split[task_name]]
            
            for instance in task['Instances']:
                output_space.add(instance['output'][0])
                if len(output_space) > 10:
                    is_classification = False
                    break
            task['IsClassification'] = is_classification
            task['OutputSpace'] = sorted(list(output_space)) if is_classification else None
            if is_classification:
                self.classification_tasks[task_path] = task
            self.tasks[task_path] = task
             
        # random.shuffle(self.data)
            
    def state_dict(self):
        return {
            'iter_count': self.iter_count,
            'buffer_tokens': self.buffer_tokens,
        }
    
    def load_state_dict(self, state_dict):
        self.iter_count = state_dict['iter_count']
        self.buffer_tokens = state_dict['buffer_tokens']
        self.data = self.data.skip(self.iter_count)
      

    def format_instance(self, x, task):
        is_classification = task['IsClassification']
        output_space = task['OutputSpace']
        
        sample_splitter = random.choice(self.sample_splitters)
        answer_splitter = random.choice(self.answer_splitters)
        text_def = random.choice(task['Definition'] + task['Definition'] + [""]).strip()
        if is_classification and random.random() < 0.5:
            text_def += '\nPossible labels:'
            for i, possible_output in enumerate(output_space):
                text_def += f'\n{i+1}. {possible_output}'
            text_def += '\n'
        
        text_input = random.choice(self.input_prefixs)
        text_output = random.choice(self.output_prefixs)
        
        return sample_splitter + text_input + x['input'] + answer_splitter + text_output + random.choice(x['output'])

    def get_task_def(self, task):
        is_classification = task['IsClassification']
        output_space = task['OutputSpace']
        
        text_def = random.choice(task['Definition'] + task['Definition'] + [""]).strip()
        if is_classification and random.random() < 0.5:
            text_def += '\nPossible labels:'
            for i, possible_output in enumerate(output_space):
                text_def += f'\n{i+1}. {possible_output}'
            text_def += '\n'
        
        return text_def
        
    def get_sequence(self):
        buffer_tokens = self.buffer_tokens
        
        task = self.tasks["./natural-instructions/tasks/task512_twitter_emotion_classification.json"]
        
        text_context = self.get_task_def(task) 
        
        for x in task['Instances']:
            text_context = self.format_instance(x, task)
            self.iter_count += 1      
            curr_tokens = self.tokenizer(self.doc_separator + text_context)['input_ids']
            buffer_tokens += curr_tokens
            while len(buffer_tokens) >= self.seq_length:
                tokens = buffer_tokens[:self.seq_length]
                buffer_tokens = buffer_tokens[self.seq_length:]
                input_ids = torch.tensor(tokens)
                self.buffer_tokens = buffer_tokens # update for restore
                yield {
                    'input_ids': input_ids,
                }
                
    def get_stream(self):
        return self.get_sequence()
    
    def __iter__(self):
        if self.it is None:
            self.it = self.get_stream()
            
        return self.it



class StreamDataset(IterableDataset):
    def __init__(self, data_path, tokenizer, dev_split_path, seq_length=1024, mixture_weights_path=None, cycling=True, eval=False):
        self.data_path = data_path
        self.cycling=cycling
        self.eval = eval
        
        if mixture_weights_path is not None:
            with open(mixture_weights_path, 'rb') as f:
                mixture_weights = pickle.load(f)
            
        with open(dev_split_path, "rb") as f:
            self.dev_split = pickle.load(f)
        
        self.train_splits = []
        with open(os.path.join(data_path, 'splits/default/train_tasks.txt')) as f:
            for line in f:
                if line.strip() == '':
                    continue
                self.train_splits.append(line.strip() + '.json')
        
        self.task_paths = [
            os.path.join(data_path, 'tasks', p) for p in os.listdir(os.path.join(data_path, 'tasks')) if p.endswith('.json') and p in self.train_splits
        ]
        self.tasks = OrderedDict()
        self.classification_tasks = OrderedDict()
        self.mixture_weights = []
        for task_path in self.task_paths:
            with open(task_path) as f:
                task = json.load(f)
                
            output_space = set()
            is_classification = True
            
            # remove small tasks 
            if len(task['Instances']) < 100:
                continue 
                        
            # remove dev data from the instances
            
            task_name = ".".join(task_path.split("/")[-1].split(".")[:-1])
            if not self.eval:
                # exclude dev data from training dataset
                task['Instances'] = [obj for i, obj in enumerate(task['Instances']) if i not in self.dev_split[task_name]]
            else:
                # only dev data comprises evaluation dataset 
                # JUST EVALUATE ON TRAINING DATA - sanity check that validation loss should then be low
                task['Instances'] = [obj for i, obj in enumerate(task['Instances']) if i in self.dev_split[task_name]]
                # assert len(task['Instances']) == 50
            
            for instance in task['Instances']:
                output_space.add(instance['output'][0])
                if len(output_space) > 10:
                    is_classification = False
                    break
            task['IsClassification'] = is_classification
            task['OutputSpace'] = sorted(list(output_space)) if is_classification else None
            if is_classification:
                self.classification_tasks[task_path] = task
            self.tasks[task_path] = task
            
            if mixture_weights_path is not None:
                self.mixture_weights.append(mixture_weights[task_name]) # we construct this here to make sure the correspondence to tasks is correct
        
        if mixture_weights_path is not None:
            self.mixture_weights = np.array(self.mixture_weights)
            print(f"Mixture weights loaded: {self.mixture_weights}")
        else:
            print("No mixture weights. Sampling uniformly. ")
        
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        self.it = None
        
        self.input_prefixs = ['Input: ', 'Given: ', 'Context: ', 'Example: ', 'Question: ', '', '', '', '', '',]
        self.output_prefixs = ['Output: ', 'Output: ', 'Ans: ', 'A: ', 'Answer: ', 'Label: ', 'Label: ']
        self.sample_splitters = ['\n', '\n\n', '\n\n', '\n\n\n', '\n###\n', '\n---\n']
        self.answer_splitters = ['\n', '\n', '\n\n']
        
        self.iter_count = 0  
        
    def state_dict(self):
        return {
            'iter_count': self.iter_count,
        }
    
    def load_state_dict(self, state_dict):
        try:
            self.iter_count = state_dict['iter_count']
        except:
            print('cannot load ni states.')
    
    def sample_text_from_task(self, task):
        
        '''
        Task Definition(*33%)
        + Output Space(*50%)
        [
            + sample splitter
            + input prefix
            + input
            + answer splitter
            + output prefix
            + output
        ]
        '''
        
        is_classification = task['IsClassification']
        output_space = task['OutputSpace']
        
        sample_splitter = random.choice(self.sample_splitters)
        answer_splitter = random.choice(self.answer_splitters)
        text_def = random.choice(task['Definition'] + task['Definition'] + [""]).strip()
        if is_classification and random.random() < 0.5:
            text_def += '\nPossible labels:'
            for i, possible_output in enumerate(output_space):
                text_def += f'\n{i+1}. {possible_output}'
            text_def += '\n'
        
        text_input = random.choice(self.input_prefixs)
        text_output = random.choice(self.output_prefixs)

        text_context = text_def
        
        while True:
            # pick a random instance and do the formatting to add prompt stuff before and after , then tokenize
            instance = random.choice(task['Instances'])
            text_context += sample_splitter + text_input + instance['input'] + answer_splitter + text_output + random.choice(instance['output'])
            input_ids = self.tokenizer(text_context.strip())['input_ids']
            if len(input_ids) > self.seq_length:
                break
                
        input_ids = input_ids[:self.seq_length]
        input_ids = torch.tensor(input_ids).long()
        
        return input_ids
        
    def get_sequence(self):
        
        # hardcode a task
        task = self.tasks["./natural-instructions/tasks/task512_twitter_emotion_classification.json"]

        while True:
            # ensure at least 30% classification
            #if random.random() < 0.3:
            #    task_idx = np.random.choice(np.arange(len(self.classification_tasks)))
            #    task = list(self.classification_tasks.items())[task_idx][1]
            #else:
            # select a task with probability proportional to the weights 
            #    if len(self.mixture_weights) > 0:
            #        task_idx = np.random.choice(np.arange(len(self.tasks)), p=self.mixture_weights)
            #        task = list(self.tasks.items())[task_idx][1]
            #    else:
            #        task_idx = np.random.choice(np.arange(len(self.tasks)))
            #        task = list(self.tasks.items())[task_idx][1]
                # we either already have skimmed down the dataset, or we can perform the sampling here...
                # random.choice(self.tasks, p = the thing we learn!) 
                
                

            # then pick a random sample from that task. Note that we have already removed the dev data from the task.
            input_ids = self.sample_text_from_task(task)

            self.iter_count += 1
            
            yield {
                'input_ids': input_ids,
            }
            
                
    def get_stream(self):
        if self.cycling:
            return cycle(self.get_sequence())
        else:
            return self.get_sequence()
    
    def __iter__(self):
        if self.it is None:
            self.it = self.get_stream()
            
        for i in range(self.iter_count):
            next(self.it)
            
        return self.it
    
    
    
def get_natural_instructions_train_data_loader(args, tokenizer, num_workers=0, state_dict=None):
    
    stream_dataset = StreamDataset('/root/natural-instructions/', tokenizer, args.seq_length)
    
    if state_dict is not None:
        stream_dataset.load_state_dict(state_dict)
    
    train_data_loader = torch.utils.data.DataLoader(stream_dataset,
                                                    batch_size=args.batch_size * args.data_group_size,
                                                    shuffle=False,
                                                    num_workers=num_workers,
                                                    pin_memory=True,
                                                    collate_fn=None)
    return train_data_loader