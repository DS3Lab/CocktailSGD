import argparse
import time
import random
import numpy as np
import torch
import torch.autograd.profiler as profiler
from tasks.data_loaders.data_utils import get_train_data_loader, get_eval_data_loader
from modules.utils import gpt_loss_func
from modules.tokenizer import build_tokenizer
from pipeline_parallel.dist_pp_utils import get_pp_module

from transformers import AutoConfig, PretrainedConfig
import datasets

import wandb
from utils.dist_args_utils import *
from utils.dist_checkpoint_utils import *
from comm.comm_utils import *
import compress.flag

from collections import defaultdict

from itertools import chain
import sys

import pickle 


def test_loop(args, pipe, device, test_data_loader):
    
    if test_data_loader is None:
        return
    
    print('testing starts.....')
    
    pipe.model.eval()
    
    if get_pipeline_parallel_rank()  == args.pipeline_group_size - 1:
        def _lm_pred_func(x, y):
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            logits = x[:, :-1, :].contiguous().float()
            labels = y[:, 1:].contiguous()
            loss = loss_fct(logits.transpose(-1, -2), labels).mean(1).detach().cpu()
            return loss
        
        loss_dict = defaultdict(list)
        
        unique_tasks = set()
        for i, data in enumerate(test_data_loader):
            if args.evaluation_num_batch is not None and i >= args.evaluation_num_batch:
                break
                            
            input_ids = data['input_ids'].to(device)
            if 'task_path' in data:
                task_paths = data['task_path']
            else:
                task_paths = ""
                print("no task path found in data")
            # task_paths = data['task_path'] if 'task_path' in data else ""
            labels = input_ids.clone()
            
            # print(f"in eval_loop {i}: {task_paths}")
            unique_tasks = unique_tasks.union(set(task_paths))
                        
            pipe.infer_iter(input_ids, labels, tasks=task_paths, output_=loss_dict, pred_func=_lm_pred_func)
            print(f"Done with batch {i}.") #Batch size is {input_ids.shape} \n input_ids first row is {input_ids[0]}, first task is {task_paths[0]}, first end text context is {last_text_context if i ==8 else None} \n")
        
        #if len(loss_list) == 0:
        #    raise ValueError("Length of evaluation dataset is too small.")
        losses = list(loss_dict.values()) # list of lists 
        losses = list(chain(*losses))
        
        print(len(unique_tasks))
        print(len(loss_dict))
        
        loss = torch.tensor(losses).mean()
        
        # save all the losses
        method_name = args.checkpoint_path.split("/")[-1]
        log_dir = "/root/mayee/"
        with open(log_dir + method_name + "_" + str(pipe.global_step) + ".pkl", "wb") as f:
            pickle.dump(loss_dict, f)
        
        avg_loss_per_task = {k: torch.tensor(v).mean() for (k, v) in loss_dict.items()}
        avg_ppl_per_task = {k: torch.exp(torch.tensor(v).mean()) for (k, v) in loss_dict.items()}
        ppls = torch.exp(loss)
        metric = {"valid.perplexity": ppls.item(), "valid.loss": loss.item()}
        
        print(metric)
        wandb.log(
            metric, 
            step=pipe.global_step,
        )
        
        validation_loss = wandb.Artifact("val_loss_" + str(wandb.run.id), type="losses")
        columns = ["task_path", "average_loss"]
        val_loss_table = wandb.Table(columns=columns)
        for task in avg_loss_per_task:
            val_loss_table.add_data(task, avg_loss_per_task[task])
            
            
        validation_loss.add(val_loss_table, "losses")
        wandb.run.log_artifact(validation_loss)
        
        # validation_loss.download("./artifacts")
        
    else:
        for i, data in enumerate(test_data_loader):
            
            if args.evaluation_num_batch is not None and i >= args.evaluation_num_batch:
                break
            
            input_ids = data['input_ids'].to(device)
            labels = input_ids.clone()
            current_iter_time = pipe.infer_iter(input_ids, labels)
    
    pipe.model.train()
    


def train_loop(args, pipe, device, train_data_loader, test_data_loader):
    
    print('training starts......')

    pipe.model.train() # Flag .training to True to enable Dropout
    
    use_dp = (args.world_size != args.pipeline_group_size)
    if use_dp:
        # dp_comm = get_data_parallel_comm()
        dp_rank = get_data_parallel_rank()
        dp_size = get_data_parallel_world_size()
    else:
        dp_rank = 0
        dp_size = 1
    pp_comm = get_pipeline_parallel_comm()
    
    stop_flag = torch.zeros(1, dtype=torch.int64).to(device)
    
    input_ids = torch.zeros(
        [args.batch_size, args.seq_length], 
        dtype=torch.int64
    ).to(device)
    
    do_sync_before_save = (args.dp_mode in ['local'] and use_dp)
    
    if get_pipeline_parallel_rank() == 0 and dp_rank == 0:

        for i, data in enumerate(train_data_loader):
            if i < pipe.global_step:
                continue
                
            if use_dp:
                get_data_parallel_comm().broadcast(stop_flag, 0)
            pp_comm.broadcast(stop_flag, 0)
            
            if stop_flag.item() == 1:
                break
            
            input_ids_global = data['input_ids'].to(torch.int64).to(device)
            
            input_ids_list = input_ids_global.chunk(dp_size)
            
            if use_dp:
                for j in range(1, dp_size):
                    get_data_parallel_comm().send(
                        input_ids_list[j], j,
                    )
                
            input_ids = input_ids_list[0]
            
            pp_comm.broadcast(input_ids, 0)
            
            compress.flag.FLAG_DISABLE_COMPRESSION = (pipe.global_step < args.train_warmup_steps)
            labels = input_ids.clone()
            current_iter_time = pipe.sgd_iter(input_ids, labels, loss_func=gpt_loss_func)
            
            if pipe.global_step == 1 or (args.evaluation_steps > 0 and pipe.global_step % args.evaluation_steps == 0):
                test_loop(args, pipe, device, test_data_loader)
            
            if pipe.global_step % args.checkpoint_steps == 0:
                if do_sync_before_save:
                    pipe.dp_optim.allreduce_parameters()
                if dp_rank == 0:
                    save_checkpoint(pipe, args)
                if do_sync_before_save:
                    pipe.dp_optim.rollback_parameters()
            
            if pipe.global_step >= args.total_steps:
                stop_flag.data[:] = 1
            
    elif get_pipeline_parallel_rank() == 0:
        
        while True:
            get_data_parallel_comm().broadcast(stop_flag, 0)
            pp_comm.broadcast(stop_flag, 0)
            if stop_flag.item() == 1:
                break
                
            get_data_parallel_comm().recv(
                input_ids, 0,
            )
            pp_comm.broadcast(input_ids, 0)
            
            compress.flag.FLAG_DISABLE_COMPRESSION = (pipe.global_step < args.train_warmup_steps)
            labels = input_ids.clone()
            current_iter_time = pipe.sgd_iter(input_ids, labels, loss_func=gpt_loss_func)
            
            if pipe.global_step == 1 or (args.evaluation_steps > 0 and pipe.global_step % args.evaluation_steps == 0):
                test_loop(args, pipe, device, test_data_loader)
                
            if pipe.global_step % args.checkpoint_steps == 0:
                if do_sync_before_save:
                    pipe.dp_optim.allreduce_parameters()
                if dp_rank == 0:
                    save_checkpoint(pipe, args)
                if do_sync_before_save:
                    pipe.dp_optim.rollback_parameters()
            
            
    elif get_pipeline_parallel_rank()  == args.pipeline_group_size - 1:
        while True:
            
            pp_comm.broadcast(stop_flag, 0)
            if stop_flag.item() == 1:
                break
                
            pp_comm.broadcast(input_ids, 0)
            labels = input_ids.clone()
            compress.flag.FLAG_DISABLE_COMPRESSION = (pipe.global_step < args.train_warmup_steps)
            current_iter_time = pipe.sgd_iter(input_ids, labels, loss_func=gpt_loss_func) # lm loss func
            
            if pipe.global_step == 1 or (args.evaluation_steps > 0 and pipe.global_step % args.evaluation_steps == 0):
                test_loop(args, pipe, device, test_data_loader)
                
            if pipe.global_step % args.checkpoint_steps == 0:
                if do_sync_before_save:
                    pipe.dp_optim.allreduce_parameters()
                if dp_rank == 0:
                    save_checkpoint(pipe, args)
                    pipe.save_on_disk(args.checkpoint_path)
                if do_sync_before_save:
                    pipe.dp_optim.rollback_parameters()
    else:
        while True:
            pp_comm.broadcast(stop_flag, 0)
            if stop_flag.item() == 1:
                break
            pp_comm.broadcast(input_ids, 0)
            compress.flag.FLAG_DISABLE_COMPRESSION = (pipe.global_step < args.train_warmup_steps)
            current_iter_time = pipe.sgd_iter(None, None)
            
            if pipe.global_step == 1 or (args.evaluation_steps > 0 and pipe.global_step % args.evaluation_steps == 0):
                test_loop(args, pipe, device, test_data_loader)
                
            if pipe.global_step % args.checkpoint_steps == 0:
                if do_sync_before_save:
                    pipe.dp_optim.allreduce_parameters()
                if dp_rank == 0:
                    save_checkpoint(pipe, args)
                if do_sync_before_save:
                    pipe.dp_optim.rollback_parameters()
        

def main():
    parser = argparse.ArgumentParser(description='Gpipe-GPT')
    add_device_arguments(parser)
    add_torch_distributed_arguments(parser)
    add_model_arguments(parser)
    add_task_arguments(parser)
    add_training_hyper_parameter_arguments(parser)
    add_mixed_precision_arguments(parser)
    add_parallel_schema_arguments(parser)
    add_acitvation_compression_arguments(parser)
    parser.add_argument('--model-name', type=str, default='gpt2', metavar='S',
                        help='model name or path')
    parser.add_argument('--tokenizer-name', type=str, default='gpt2', metavar='S',
                        help='tokenizer name or path')
    parser.add_argument('--model-type', type=str, default='gpt2', metavar='S',
                        help='model name or path')
    parser.add_argument('--checkpoint-path', type=str, default='model_checkpoints/gpt2')
    parser.add_argument('--load-checkpoint-path', type=str, default=None, help='Path to load checkpoint from, if different from checkpoint-path')
    parser.add_argument('--task-name', type=str, default='cot', metavar='S',
                        help='task name')
    parser.add_argument('--mixture_weights_path', type=str, default=None, help=".pkl file of dictionary mapping from task name to sampling weight.")
    parser.add_argument('--dev_split_path', type=str, default="/root/mayee/dev_split_map.pkl", help=".pkl file of dictionary mapping from task name to sampling weight.")
    parser.add_argument('--train_data_no_replace', action="store_true", help="Sample without replacement for training dataset")
    parser.add_argument('--warmup-steps', type=int, default=0, help='-')
    parser.add_argument('--train-warmup-steps', type=int, default=0, help='-')
    parser.add_argument('--total-steps', type=int, default=None, help='-')
    parser.add_argument('--total-scheduler-steps', type=int, default=None, help='-')
    parser.add_argument('--scheduler', type=str, default='linear')
    parser.add_argument('--load-pretrained-model', 
                        type=lambda x: x.lower()=='true', default=True, metavar='S',
                        help='load pretrained model or not.')
    parser.add_argument('--load-checkpoint', 
                        type=lambda x: x.lower()=='true', default=True, metavar='S',
                        help='load pretrained model or not.')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--profiling', type=str, default='no-profiling', metavar='S',
                        help='enable which profiling? default: tidy mode')
    parser.add_argument('--trace-postfix', type=str, default='default', metavar='S',
                        help='postfix of the tracing file name.')
    parser.add_argument('--evaluation-steps', 
                        type=int, default=0, metavar='S',
                        help='every x steps, do evaluation. (0 means do not do evaluation)')
    parser.add_argument('--evaluation-data',
                        type=str, default=None, help="path of eval data in jsonl")
    parser.add_argument('--evaluation-num-batch',
                        type=int, default=None, help="for debug purpose, only eval the first several batch.")
    parser.add_argument('--checkpoint-steps', 
                        type=int, default=0, metavar='S',
                        help='every x steps, save checkpoint. (0 means do not save checkpoint)')
    parser.add_argument('--net-interface', 
                        type=str, default='lo', metavar='S',
                        help='net_interface')
    parser.add_argument('--job-id', 
                        type=str, default="0", metavar='S',
                        help='an uuid')
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    if args.use_cuda:
        assert (torch.cuda.is_available())
        device = torch.device('cuda', args.cuda_id)
    else:
        device = torch.device('cpu')
        
    init_communicators(args)
    
    use_dp = (args.world_size != args.pipeline_group_size)
    if use_dp:
        dp_comm = get_data_parallel_comm()
        dp_rank = get_data_parallel_rank()
        dp_size = get_data_parallel_world_size()
    else:
        dp_rank = 0
        dp_size = 1

    if args.model_type != 'h3': 
        config = AutoConfig.from_pretrained(args.model_name)
    else:
        # H3 does not have AutoConfig
        config = PretrainedConfig.from_dict({
            'n_layer': args.num_layers,
            'd_model': args.embedding_dim,
            'd_inner': args.embedding_dim * 4,
            'vocab_size': 50257,
            'attn_cfg': dict(num_heads = 12), # HARD CODED FOR 125M
            'attn_layer_idx': [1, 8], # HARD CODED FOR 125M
            'ssm_cfg': dict(mode='diag', measure='diag-lin'),
            'pad_vocab_size_multiple': 8,
            'max_position_embeddings': 0,
            'resid_dropout': 0.0,
            'embed_dropout': 0.1,
            'layer_norm_epsilon': 1e-5,
            'fused_mlp': True,
            'fused_dropout_add_ln': True,
            'residual_in_fp32': True
        })
    
    # num layer globally
    if hasattr(config, 'num_hidden_layers'):
        args.max_layers = config.num_hidden_layers
    elif hasattr(config, 'num_layers'):
        args.max_layers = config.num_layers 
    else:
        args.max_layers = config.n_layer
    
    tokenizer = build_tokenizer(args)
    tokenizer.model_max_length = args.seq_length
    # config.vocab_size = tokenizer.vocab_size
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id
    print("token vocab size:", config.vocab_size)
    
    if get_pipeline_parallel_rank() == 0 and dp_rank == 0:
        train_data_loader = get_train_data_loader(args, tokenizer)
    else:
        train_data_loader = None
        
    if args.evaluation_data is not None and dp_rank == 0:
        test_data_loader = get_eval_data_loader(args, tokenizer)
    else:
        test_data_loader = None
        
    if args.total_steps is None:
        args.total_steps = len(train_data_loader)
    
    use_dp = (args.world_size != args.pipeline_group_size)
    if use_dp:
        print("Running ", args.pp_mode, " with data parallel.")
    else:
        print("Running ", args.pp_mode, " without data parallel.")
    
    pipe = get_pp_module(args, config, device, use_dp)
    
    if args.load_checkpoint:
        load_checkpoint(pipe, args)

    if args.fp16:
        pipe.optimizer.reload_model_params()

    if args.profiling == 'no-profiling':
        train_loop(args, pipe, device, train_data_loader, test_data_loader)
    else:
        prefix = './trace_json/gpt3_' + args.pp_mode
        if use_dp:
            prefix = prefix + '_' + args.dp_mode
        trace_file = prefix + get_learning_arguments_str(args) + get_model_arguments_str(args) + \
                     get_dist_arguments_str(args) + get_mixed_precision_arguments_str(args) + '_' + \
                     args.profiling + '_' + args.trace_postfix + '.json'
        if args.profiling == 'tidy_profiling':
            try:
                train_loop(args, pipe, device, train_data_loader, test_data_loader)
            except Exception as e:
                raise e
                print(get_pipeline_parallel_rank(), e)
            pipe.export_profiling_result(filename=trace_file)
        elif args.profiling == 'pytorch_profiling':
            with profiler.profile(profile_memory=True, use_cuda=args.use_cuda) as prof:
                train_loop(args, pipe, device, train_data_loader, test_data_loader)
            print(prof.key_averages().table())
            prof.export_chrome_trace(trace_file)
        else:
            print("No recognized profiler?")
            assert False
    print(get_pipeline_parallel_rank(), 'finished.')

if __name__ == '__main__':
    main()
