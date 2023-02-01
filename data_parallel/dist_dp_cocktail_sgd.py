import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda
import math
from comm.comm_utils import *
from .flatten_utils import flatten_params, flatten_tensors
from compress.fixpoint import *
from compress.sparsification import *
from compress import flag
import cupy

import os

quantization_bits = int(os.environ.get('QUANT_BITS', 8))
quantization_bucket_size = int(os.environ.get('QUANT_BUCKET_SIZE', 128))
quantization_stochastic = int(os.environ.get('QUANT_STOCHASTIC', 0))
quantization_minimum_stochastic_distance = float(os.environ.get('QUANT_MIN_STOCHASTIC_DISTANCE', 0.2))
top_k_ratio = float(os.environ.get('TOPK_RATIO', 0.5))
random_p_ratio = float(os.environ.get('RANDOMP_RATIO', 0.5))
random_method = os.environ.get('RANDOM_METHOD', 'random_rolling')

import threading


class CocktailSGDDP:
    def __init__(self, args, device, module: torch.nn.Module, optimizer: torch.optim.Optimizer = None, flatten=False):
        # assert not flatten
        self.dp_bits = args.dp_bits
        self.flatten = flatten
        self.global_rank = args.rank
        self.dp_group_size = args.data_group_size
        self.enable_tidy_profiling = (args.profiling == 'tidy_profiling')
        self.dp_comm = get_data_parallel_comm()
        self.dp_rank = get_data_parallel_rank()
        self.pp_comm = get_pipeline_parallel_comm()
        self.pp_rank = get_pipeline_parallel_rank()
        self.dp_comm_stream = torch.cuda.Stream(device=device, priority=-1)
        self.torch_optim_comp_stream = torch.cuda.default_stream(device=device)
        self.backward_ready_event = torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
        self.sync_gradients_start_event = torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
        self.sync_gradients_ready_event = torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
        self.optimizer_step_ready_event = torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)

        self.module = module
        assert optimizer is not None
        self.optimizer = optimizer
        
        if self.flatten:
            _params = []
            for i_group, group in enumerate(self.optimizer.optimizer.param_groups):
                for i_para, para in enumerate(group["params"]):
                    _params.append(para)
            self.flatten_para = flatten_tensors(_params)
            print("Flattened parameter number: {}, element size: {}."
                  .format(self.flatten_para.data.numel(), self.flatten_para.data.element_size()))
        
        
        num_paras, element_size = self._compute_total_para_num()
        print("Total number of parameters: {}, element size: {}, total size {} MB."
              .format(num_paras, element_size, num_paras * element_size // 1024 // 1024))
        
        if self.enable_tidy_profiling:
            self.global_rank = args.rank
            self.init_event = None
            self.init_time_stamp = None

            # assert self.flatten
            self.sync_gradients_start_event = torch.cuda.Event(enable_timing=True, blocking=False)
            self.optimizer_step_start_event = torch.cuda.Event(enable_timing=True, blocking=False)
            
            self.gather_start_event = torch.cuda.Event(enable_timing=True, blocking=False)
            self.sync_start_event = torch.cuda.Event(enable_timing=True, blocking=False)
            self.gather_end_event = torch.cuda.Event(enable_timing=True, blocking=False)
            self.sync_end_event = torch.cuda.Event(enable_timing=True, blocking=False)
            
        self.dp_state_dict = {}

    def _compute_total_para_num(self):
        total_count = 0
        element_size = 0
        for para in self.module.parameters():
            # print("Parameter: ", para.data.shape)
            total_count += torch.numel(para.data)
            element_size = para.element_size()
        return total_count, element_size

    def profile_mark_sync_grad_start(self):
        if self.enable_tidy_profiling:
            self.dp_comm_stream.record_event(self.sync_gradients_start_event)

    def profile_mark_allreduce_end(self):
        pass

    def profile_mark_optimizer_step_start(self):
        if self.enable_tidy_profiling:
            self.torch_optim_comp_stream.record_event(self.optimizer_step_start_event)
            
    def _allreduce_gradients(self):
        with torch.cuda.stream(self.dp_comm_stream):
            cupy_dp_stream = cupy.cuda.ExternalStream(self.dp_comm_stream.cuda_stream)
            self.dp_comm_stream.wait_event(self.backward_ready_event)
            for name, para in self.module.named_parameters():
                if para.grad is None:
                    continue
                self.dp_comm.all_reduce(para.grad, stream=cupy_dp_stream)
            self.dp_comm_stream.record_event(self.sync_gradients_ready_event)
            
    def _compress(self, x):
        # return x
        dtype = x.dtype
        shape = x.shape
        with torch.cuda.stream(self.dp_comm_stream):
            cupy_dp_stream = cupy.cuda.ExternalStream(self.dp_comm_stream.cuda_stream)
            with cupy_dp_stream:
                
                k = max(int(top_k_ratio * x.numel()), 1)
                if k >= quantization_bucket_size:
                    # ensure dividable
                    k = k // quantization_bucket_size * quantization_bucket_size
                else:
                    # bucket_size will be set to k internally
                    pass
                    
                values, masks, indices = compress_topk(x, k, return_indices=True)

                values_q, scales_q = compress_flexible_nbits_by_bucket(
                    values, bits=quantization_bits, scale_method='max', bucket_size=quantization_bucket_size, 
                    stochastic=quantization_stochastic, minimum_stochastic_distance=quantization_minimum_stochastic_distance)
                
                return (values_q, scales_q, masks), (dtype, shape, values.shape)
    
    def _decompress(self, x_hat, meta_data):
        
        values_q, scales_q, masks = x_hat
        x_dtype, x_shape, values_shape = meta_data
        
        values = decompress_flexible_nbits_by_bucket(values_q, scales_q, bits=quantization_bits, original_shape=values_shape, bucket_size=quantization_bucket_size)
                    
        x = decompress_topk(values, masks, x_shape)
        x = x.view(x_shape).to(x_dtype)
        
        return x
    
    def _update_comm_mask(self, para, comm_mask=None):
        
        if random_method == 'random_rolling':
            
            sync_every_n_elements = int(1 / random_p_ratio)
            
            if comm_mask is None:
                para_shape = list(para.shape)
                assert para_shape[0] == para_shape[0] // self.dp_group_size * self.dp_group_size
                para_shape[0] = para_shape[0] // self.dp_group_size
                comm_mask = torch.zeros(para_shape, dtype=torch.bool, device=para.device)
                comm_mask.view(-1)[::sync_every_n_elements] = True
                n_potisive = comm_mask.sum().item() // quantization_bucket_size * quantization_bucket_size
                if n_potisive != 0:
                    comm_mask.view(-1)[comm_mask.view(-1).cumsum(-1) > n_potisive] = False
                    assert comm_mask.sum().item() == n_potisive
                else:
                    comm_mask[:] = True
                print('comm_mask:', comm_mask.sum().item(), comm_mask.shape)
            else:
                comm_mask = comm_mask.roll(1)
            
        elif random_method == 'random_w_replacement':
            
            seed = torch.randint(10000, [1])
            self.dp_comm.broadcast(seed, 0)
            torch.manual_seed(seed.item())
            
            para_shape = list(para.shape)
            assert len(para_shape) == 1
            assert para_shape[0] == para_shape[0] // self.dp_group_size * self.dp_group_size
            para_shape[0] = para_shape[0] // self.dp_group_size
            
            n_sample = int(random_p_ratio * para_shape[0])
            n_sample = n_sample // 8 * 8
            n_sample = n_sample // quantization_bucket_size * quantization_bucket_size
            comm_mask = torch.randint(para_shape[0], (n_sample,), device=para.device)
            
        elif random_method == 'random_wo_replacement':
            
            if comm_mask is None or ((self._cursor+1) * self._n_sample >= len(self._comm_indices)):
                
                seed = torch.randint(10000, [1])
                self.dp_comm.broadcast(seed, 0)
                torch.manual_seed(seed.item())
                
                para_shape = list(para.shape)
                assert len(para_shape) == 1
                assert para_shape[0] == para_shape[0] // self.dp_group_size * self.dp_group_size
                para_shape[0] = para_shape[0] // self.dp_group_size
                
                n_sample = int(random_p_ratio * para_shape[0])
                n_sample = n_sample // 8 * 8
                n_sample = n_sample // quantization_bucket_size * quantization_bucket_size
                self._n_sample = n_sample
                self._cursor = 0
                self._comm_indices = torch.randperm(para_shape[0], device=para.device)
                
            comm_mask = self._comm_indices[self._cursor * self._n_sample: (self._cursor+1) * self._n_sample]
            self._cursor += 1
        
        else:
            
            raise Exception(f"""Unknown random method '{random_method}'""")
                
                
        return comm_mask
        
        
        
            
    def _partial_sync(self):
        
        if self.flatten:
 
            cupy_dp_stream = cupy.cuda.ExternalStream(self.dp_comm_stream.cuda_stream)
            with torch.cuda.stream(self.dp_comm_stream), cupy_dp_stream:
            
                self.dp_comm_stream.record_event(self.sync_gradients_start_event)
                
                self.dp_comm.barrier()
                
                name = 'model'
                para = self.flatten_para
                
                dp_state_dict = self.dp_state_dict

                if name not in dp_state_dict:

                    # comm mask
                    comm_mask_list = []
                    comm_data_list = []
                    for i in range(self.dp_group_size):
                        comm_mask = self._update_comm_mask(para)
                        comm_mask_list.append(comm_mask)

                    # global para
                    global_para = para.data.half()

                    # server error
                    server_error = torch.zeros(
                        para.size(0) // self.dp_group_size, dtype=torch.float16, device=para.device,
                    )

                    dp_state_dict[name] = {
                        "comm_mask_list": comm_mask_list,
                        "global_para": global_para,
                        "server_error": server_error,
                    }
                else:
                    for i in range(self.dp_group_size):
                        dp_state_dict[name]['comm_mask_list'][i] = self._update_comm_mask(para, dp_state_dict[name]['comm_mask_list'][i])

                comm_mask_list = dp_state_dict[name]["comm_mask_list"]
                comm_data_list = comm_data_list = [None for _ in comm_mask_list]
                global_para = dp_state_dict[name]["global_para"]
                chunk_size = global_para.size(0) // self.dp_group_size
                server_error = dp_state_dict[name]["server_error"]
                server_mask = comm_mask_list[self.dp_rank].to(server_error.device)

                for i in range(self.dp_group_size):
                    comm_mask = comm_mask_list[i]
                    comm_data_list[i] = (para[i*chunk_size:(i+1)*chunk_size][comm_mask] - global_para[i*chunk_size:(i+1)*chunk_size][comm_mask]).half()
                    
                comm_data_compressed_list = []
                comm_data_meta_list = []
                for x in comm_data_list:
                    data, meta_data = self._compress(x)
                    comm_data_compressed_list.append(data)
                    comm_data_meta_list.append(meta_data)
                    del x
                # del comm_data_list
                comm_buffer_list = [[torch.zeros_like(x, device='cpu') for x in x_tuple] for x_tuple in comm_data_compressed_list]
                
                # revert
                for i in range(self.dp_group_size):
                    _data_compressed = self._decompress(comm_data_compressed_list[i], comm_data_meta_list[i])
                    para.data[i*chunk_size:(i+1)*chunk_size][comm_mask_list[i]] -= _data_compressed
                    del _data_compressed
                    
                _group_calls = []
                for i in range(self.dp_group_size):
                    for j, to_send in enumerate(comm_data_compressed_list[i]):
                        if i != self.dp_rank:
                            call = self.dp_comm.isend(
                                to_send, dst=i, stream=cupy_dp_stream)
                            _group_calls.append(call)
                        else:
                            comm_buffer_list[i][j][:] = to_send.cpu()
                    for to_recv in comm_buffer_list[i]:
                        if i != self.dp_rank:
                            call = self.dp_comm.irecv(
                                to_recv, src=i, stream=cupy_dp_stream)
                            _group_calls.append(call)
                for call in _group_calls:
                    call.wait()  

                server_data = self._decompress([z.to(para.device) for z in comm_buffer_list[0]], comm_data_meta_list[0]) / len(comm_buffer_list)
                for i in range(1, self.dp_group_size):
                    server_data.data += self._decompress([z.to(para.device) for z in comm_buffer_list[i]], comm_data_meta_list[i]) / len(comm_buffer_list)
                server_data.add_(server_error[server_mask].to(server_data.device))
                server_data_compressed, server_data_meta = self._compress(server_data)
                server_error.data[server_mask] = (server_data - self._decompress(server_data_compressed, server_data_meta)).to(server_error.device)
                
                _group_calls = []
                for i in range(self.dp_group_size):
                    for j, to_send in enumerate(server_data_compressed):
                        if i != self.dp_rank:
                            call = self.dp_comm.isend(
                                to_send, dst=i, stream=cupy_dp_stream)
                            _group_calls.append(call)
                        else:
                            comm_buffer_list[i][j][:] = to_send.cpu()
                    for to_recv in comm_buffer_list[i]:
                        if i != self.dp_rank:
                            call = self.dp_comm.irecv(
                                to_recv, src=i, stream=cupy_dp_stream)
                            _group_calls.append(call)
                for call in _group_calls:
                    call.wait()
                
                for i in range(self.dp_group_size):
                    
                    _data = self._decompress([z.to(para.device) for z in comm_buffer_list[i]], comm_data_meta_list[i])
                    para.data[i*chunk_size:(i+1)*chunk_size][comm_mask_list[i]] += _data
                    global_para.data[i*chunk_size:(i+1)*chunk_size][comm_mask_list[i]] += _data
                    
                    del _data
                    
                self.dp_comm_stream.record_event(self.sync_gradients_ready_event)
            
        else:
            
            cupy_dp_stream = cupy.cuda.ExternalStream(self.dp_comm_stream.cuda_stream)
            with torch.cuda.stream(self.dp_comm_stream), cupy_dp_stream:
                
                for i_group, group in enumerate(self.optimizer.optimizer.param_groups):
                    for i_para, para in enumerate(group["params"]):
                        
                        
                        para = para.view(-1)
                        
                        name = f"{i_group}-{i_para}"
                    
                        dp_state_dict = self.dp_state_dict

                        if name not in dp_state_dict:

                            # comm mask
                            comm_mask_list = []
                            comm_data_list = []
                            for i in range(self.dp_group_size):
                                comm_mask = self._update_comm_mask(para)
                                comm_mask_list.append(comm_mask)

                            # global para
                            global_para = para.data.half()

                            # server error
                            # server_error = torch.zeros_like(global_para.chunk(self.dp_group_size, 0)[self.dp_rank])
                            server_error = torch.zeros(
                                para.size(0) // self.dp_group_size, dtype=torch.float16, device='cpu',
                            )

                            # print('server error shape:', server_error.shape)
                            dp_state_dict[name] = {
                                "comm_mask_list": comm_mask_list,
                                "global_para": global_para,
                                "server_error": server_error,
                            }
                        else:
                            for i in range(self.dp_group_size):
                                dp_state_dict[name]['comm_mask_list'][i] = self._update_comm_mask(para, dp_state_dict[name]['comm_mask_list'][i])

                        comm_mask_list = dp_state_dict[name]["comm_mask_list"]
                        comm_data_list = [None for _ in comm_mask_list]
                        global_para = dp_state_dict[name]["global_para"]
                        chunk_size = global_para.size(0) // self.dp_group_size
                        server_error = dp_state_dict[name]["server_error"]
                        server_mask = comm_mask_list[self.dp_rank]

                        for i in range(self.dp_group_size):
                            comm_mask = comm_mask_list[i]
                            comm_data_list[i] = (para[i*chunk_size:(i+1)*chunk_size][comm_mask] - global_para[i*chunk_size:(i+1)*chunk_size][comm_mask]).half()
                            
                        comm_data_compressed_list = []
                        comm_data_meta_list = []
                        for x in comm_data_list:
                            data, meta_data = self._compress(x)
                            comm_data_compressed_list.append(data)
                            comm_data_meta_list.append(meta_data)
                        comm_buffer_list = [[torch.zeros_like(x, device='cpu') for x in x_tuple] for x_tuple in comm_data_compressed_list]
                        
                        # revert
                        for i in range(self.dp_group_size):
                            _data_compressed = self._decompress(comm_data_compressed_list[i], comm_data_meta_list[i])
                            print(comm_data_list[i].shape, _data_compressed.shape, comm_mask_list[i].shape)
                            para.data[i*chunk_size:(i+1)*chunk_size][comm_mask_list[i]] -= _data_compressed
                            del _data_compressed

                        _group_calls = []
                        for i in range(self.dp_group_size):
                            for j, to_send in enumerate(comm_data_compressed_list[i]):
                                # print(f"send from {self.dp_rank} to {i}")
                                if i != self.dp_rank:
                                    call = self.dp_comm.isend(
                                        to_send, dst=i, stream=cupy_dp_stream)
                                    _group_calls.append(call)
                                else:
                                    comm_buffer_list[i][j][:] = to_send.cpu()
                            for to_recv in comm_buffer_list[i]:
                                # print(f"recv from {i} to {self.dp_rank}")
                                if i != self.dp_rank:
                                    call = self.dp_comm.irecv(
                                        to_recv, src=i, stream=cupy_dp_stream)
                                    _group_calls.append(call)
                        for call in _group_calls:
                            call.wait()
                            
                        server_data = self._decompress([z.to(para.device) for z in comm_buffer_list[0]], comm_data_meta_list[0]) / len(comm_buffer_list)
                        for i in range(1, self.dp_group_size):
                            server_data.data += self._decompress([z.to(para.device) for z in comm_buffer_list[i]], comm_data_meta_list[i]) / len(comm_buffer_list)
                        server_data.add_(server_error[server_mask].to(server_data.device))
                        server_data_compressed, server_data_meta = self._compress(server_data)
                        server_error.data[server_mask] = (server_data - self._decompress(server_data_compressed, server_data_meta)).cpu()
                        

                        _group_calls = []
                        for i in range(self.dp_group_size):
                            for j, to_send in enumerate(server_data_compressed):
                                if i != self.dp_rank:
                                    call = self.dp_comm.isend(
                                        to_send, dst=i, stream=cupy_dp_stream)
                                    _group_calls.append(call)
                                else:
                                    comm_buffer_list[i][j][:] = to_send.cpu()
                            for to_recv in comm_buffer_list[i]:
                                if i != self.dp_rank:
                                    call = self.dp_comm.irecv(
                                        to_recv, src=i, stream=cupy_dp_stream)
                                    _group_calls.append(call)
                        for call in _group_calls:
                            call.wait()
                        

                        for i in range(self.dp_group_size):
                            
                            _data = self._decompress([z.to(para.device) for z in comm_buffer_list[i]], comm_data_meta_list[i])
                            para.data[i*chunk_size:(i+1)*chunk_size][comm_mask_list[i]] += _data
                            global_para.data[i*chunk_size:(i+1)*chunk_size][comm_mask_list[i]] += _data
                            
                            del _data
                            
                self.dp_comm_stream.record_event(self.sync_gradients_ready_event)
                

    def pre_optimizer_step(self):
        if not flag.FLAG_DISABLE_COMPRESSION:
            self.t = threading.Thread(target=self._partial_sync)
            self.t.start()
            
    def optimizer_step(self):
        
        if flag.FLAG_DISABLE_COMPRESSION:
            self._allreduce_gradients()
        else:
            self.t.join()
            
        with torch.cuda.stream(self.torch_optim_comp_stream):
            self.torch_optim_comp_stream.wait_event(self.sync_gradients_ready_event)
            self.torch_optim_comp_stream.wait_event(self.backward_ready_event)
            self.profile_mark_optimizer_step_start()
            self.optimizer.step()
            print('done optim')
            self.torch_optim_comp_stream.record_event(self.optimizer_step_ready_event)

    def set_time_stamp(self, init_time_stamp, init_event):
        self.init_event = init_event
        self.init_time_stamp = init_time_stamp

    def get_ts(self, event):
        return self.init_time_stamp + self.init_event.elapsed_time(event) * 1e+3

    def profiling_data_parallel(self, init_time_stamp, init_event):
        self.set_time_stamp(init_time_stamp, init_event)
        profiling_log = []

        # assert self.flatten
        allreduce_slot = self.sync_gradients_start_event.elapsed_time(self.sync_gradients_ready_event)*1e+3
        allreduce_log = {"name": "opt_shardedPS_sync", "ph": "X", "pid": self.global_rank, "tid": "7. optimizer-comm",
                         "ts": self.get_ts(self.sync_gradients_start_event),
                         "dur": allreduce_slot, "cname": "cq_build_passed",
                         "args": {'para': 'flattened_grad', 'size': self.flatten_para.numel()}}
        # print(allreduce_log)
        profiling_log.append(allreduce_log)

        optimizer_slot = self.optimizer_step_start_event.elapsed_time(self.optimizer_step_ready_event) * 1e+3
        optimizer_log = {"name": "opt_comp", "ph": "X", "pid": self.global_rank, "tid": "8. optimizer-comp",
                         "ts": self.get_ts(self.optimizer_step_start_event), "dur": optimizer_slot, "cname": "bad"}
        # print(optimizer_log)
        profiling_log.append(optimizer_log)
        
        return profiling_log
