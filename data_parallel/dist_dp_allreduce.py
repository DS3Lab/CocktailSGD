import torch.cuda
from comm.comm_utils import *
from .flatten_utils import flatten_params

count = 0

class AllReduceDP:
    def __init__(self, args, device, module: torch.nn.Module, optimizer: torch.optim.Optimizer = None, flatten=True):
        self.args = args
        self.flatten = flatten
        self.global_rank = args.rank
        self.dp_group_size = args.data_group_size
        self.enable_tidy_profiling = (args.profiling == 'tidy_profiling')
        # self.dp_comm = get_data_parallel_comm()
        self.dp_rank = get_data_parallel_rank()
        self.pp_comm = get_pipeline_parallel_comm()
        self.pp_rank = get_pipeline_parallel_rank()
        self.pp_group_size = get_pipeline_parallel_world_size()
        self.device = device
        self.dp_comm_stream = torch.cuda.Stream(device=device, priority=-1)
        self.torch_optim_comp_stream = torch.cuda.default_stream(device=device)
        self.backward_ready_event = torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
        self.allreduce_grad_ready_event = torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
        self.optimizer_step_ready_event = torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)

        self.flag_dp_exception = 0
        self.module = module
        num_paras, element_size = self._compute_total_para_num()
        print("Total number of parameters: {}, element size: {}, total size {} MB."
              .format(num_paras, element_size, num_paras * element_size // 1024 // 1024))

        if self.flatten:
            self.flatten_para = flatten_params(self.module.parameters())
            print("Flattened parameter number: {}, element size: {}."
                  .format(self.flatten_para.data.numel(), self.flatten_para.data.element_size()))
            print("Flattened parameter grad number: {}, element size: {}."
                  .format(self.flatten_para.grad.numel(), self.flatten_para.grad.element_size()))

        assert optimizer is not None
        self.optimizer = optimizer

        if self.enable_tidy_profiling:
            self.global_rank = args.rank
            self.init_event = None
            self.init_time_stamp = None
            if self.flatten:
                self.allreduce_gradients_start_event = torch.cuda.Event(enable_timing=True, blocking=False)
            else:
                self.allreduce_gradients_start_events = dict()
                self.allreduce_gradients_end_events = dict()
                for name, _ in self.module.named_parameters():
                    self.allreduce_gradients_start_events[name] = torch.cuda.Event(enable_timing=True, blocking=False)
                    self.allreduce_gradients_end_events[name] = torch.cuda.Event(enable_timing=True, blocking=False)

            self.optimizer_step_start_event = torch.cuda.Event(enable_timing=self.enable_tidy_profiling,
                                                               blocking=False)

    @property
    def dp_comm(self):
        return get_data_parallel_comm()

    def _compute_total_para_num(self):
        total_count = 0
        element_size = 0
        for para in self.module.parameters():
            # print("Parameter: ", para.data.shape)
            total_count += torch.numel(para.data)
            element_size = para.element_size()
        return total_count, element_size

    def profile_mark_allreduce_start(self, name=None):
        if self.enable_tidy_profiling:
            if name is None:
                self.dp_comm_stream.record_event(self.allreduce_gradients_start_event)
            else:
                self.dp_comm_stream.record_event(self.allreduce_gradients_start_events[name])

    def profile_mark_allreduce_end(self, name=None):
        if self.enable_tidy_profiling:
            if name:
                self.dp_comm_stream.record_event(self.allreduce_gradients_end_events[name])

    def profile_mark_optimizer_step_start(self):
        if self.enable_tidy_profiling:
            self.torch_optim_comp_stream.record_event(self.optimizer_step_start_event)

    def _allreduce_gradients(self):
        with torch.cuda.stream(self.dp_comm_stream):
            cupy_dp_stream = cupy.cuda.ExternalStream(self.dp_comm_stream.cuda_stream)
            self.dp_comm_stream.wait_event(self.backward_ready_event)
            if self.flatten:
                self.profile_mark_allreduce_start()
                self.dp_comm.all_reduce(self.flatten_para.grad, stream=cupy_dp_stream)
                self.profile_mark_allreduce_end()
            else:
                for name, para in self.module.named_parameters():
                    if para.grad is None:
                        continue
                    self.profile_mark_allreduce_start(name)
                    self.dp_comm.all_reduce(para.grad, stream=cupy_dp_stream)
                    self.profile_mark_allreduce_end(name)
            self.dp_comm_stream.record_event(self.allreduce_grad_ready_event)

    def reinit_dp_comm_if_wrong(self):
        buffers = [torch.zeros(1).long().to(self.device) for _ in range(self.pp_group_size)]
        self.pp_comm.all_gather(torch.tensor(self.flag_dp_exception).long().to(self.device), buffers)
        self.flag_dp_exception = max([s.item() for s in buffers])

        if self.flag_dp_exception:
            reinit_dp_communicator(self.args)
            self.flag_dp_exception = 0

    def optimizer_step(self):
        try:
            self._allreduce_gradients()
        except Exception as e:
            print("_allreduce_gradients except exception: {}.".format(str(e)))
            self.flag_dp_exception = 1
            self.reinit_dp_comm_if_wrong()

        with torch.cuda.stream(self.torch_optim_comp_stream):
            self.torch_optim_comp_stream.wait_event(self.allreduce_grad_ready_event)
            self.profile_mark_optimizer_step_start()
            
            global count
            
            if count > 50:
                grads = {}
                for name, para in self.module.named_parameters():
                    if para.grad is None:
                        continue
                    grads[name] = para.grad.detach().cpu().numpy()
                torch.save(grads, f'grad_{self.global_rank}.pt')
            count += 1
            
            self.optimizer.step()
            self.torch_optim_comp_stream.record_event(self.optimizer_step_ready_event)

    def set_time_stamp(self, init_time_stamp, init_event):
        self.init_event = init_event
        self.init_time_stamp = init_time_stamp

    def get_ts(self, event):
        return self.init_time_stamp + self.init_event.elapsed_time(event) * 1e+3

    def profiling_data_parallel(self, init_time_stamp, init_event):
        self.set_time_stamp(init_time_stamp, init_event)
        profiling_log = []

        if self.flatten:
            allreduce_slot = self.allreduce_gradients_start_event.elapsed_time(self.allreduce_grad_ready_event)*1e+3
            allreduce_log = {"name": "opt_allreduce", "ph": "X", "pid": self.global_rank, "tid": "7. optimizer-comm",
                             "ts": self.get_ts(self.allreduce_gradients_start_event),
                             "dur": allreduce_slot, "cname": "cq_build_passed",
                             "args": {'para': 'flattened_grad', 'size': self.flatten_para.grad.numel()}}
            # print(allreduce_log)
            profiling_log.append(allreduce_log)
        else:
            for name, para in self.module.named_parameters():
                allreduce_slot = self.allreduce_gradients_start_events[name].elapsed_time(
                    self.allreduce_gradients_end_events[name]) * 1e+3
                allreduce_log = {"name": "opt_allreduce", "ph": "X", "pid": self.global_rank, "tid": "7. optimizer-comm",
                                 "ts": self.get_ts(self.allreduce_gradients_start_events[name]), "dur": allreduce_slot,
                                 "cname": "cq_build_passed", "args": {'para': name, 'size': torch.numel(para.data)}}
                # print(allreduce_log)
                profiling_log.append(allreduce_log)

        optimizer_slot = self.optimizer_step_start_event.elapsed_time(self.optimizer_step_ready_event) * 1e+3
        optimizer_log = {"name": "opt_comp", "ph": "X", "pid": self.global_rank, "tid": "8. optimizer-comp",
                         "ts": self.get_ts(self.optimizer_step_start_event), "dur": optimizer_slot, "cname": "bad"}
        # print(optimizer_log)
        profiling_log.append(optimizer_log)
        return profiling_log
