# CocktailSGD

## Quick Start

### (1) Setup

- Install PyTorch env: 

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge cupy nccl cudatoolkit=11.8

pip install transformers
pip install datasets
pip install netifaces
pip install zstandard
pip install wandb
```

As we use wandb to manage experiments, one should also configure `wandb` before running the code
```shell
wandb login
```

### (2) Download Pretrained Models

We provide pretrained model checkpoints that are sharded by layers:
- [OPT-1.3B](https://drive.google.com/file/d/12KrRkuxSREnvl9XGWCD72PHOIuVAod_V/view?usp=share_link)
- [GPT-J-6B](https://drive.google.com/file/d/1xy2EzUvUhelNyeDZ43iyIok2O7PUqwWB/view?usp=share_link)
- [GPT-NeoX-20B](https://drive.google.com/file/d/1Yj-_r-0kBSasfJpVhcKH2SdU36LjtJJc/view?usp=share_link)

Please download and unzip the above ckpts to fine-tune them. 
The path of unzipped model should be passed to `--model-name` and `--tokenizer-name` for fine-tuning.

### (3) Run Fine-Tuning

#### An Example of OPT-1.3B

Please refer to `example_scripts/finetune_opt1.3b.sh`, which shows an example to fine-tune OPT-1.3B on mmlu-cot data.
The script will launch 8 processes with a data parallel degree of 4 and a pipeline parallel degree of 2.

In case of geo-distributed training, please first make sure the network interface is correctly set and the master (rank 0 worker) IP and port are accesible by all the workers.
After that, run the corresponding process on each GPU node.

```shell
# set enviroment vars
...

# run on each GPU node
python dist_lm_train.py ... --cuda-id 0 --rank ${GLOBAL_RANK}
```

## Arguments

Enviroment vars that should be set:
```bash
export GLOO_SOCKET_IFNAME=lo # the correct interface
export NCCL_SOCKET_IFNAME=lo # the correct interface
export WANDB_NAME=opt-test # wandb run name

export RANDOMP_RATIO=0.1   # CocktailSGD: Random sparsity ratio
export TOPK_RATIO=0.2      # CocktailSGD: TopK sparsity ratio
export QUANT_BITS=4        # CocktailSGD: Quantization bits
```

The following arguments should be carefully set:
- `--model-name`: The path of model ckpt sharded by layers.
- `--tokenizer-name`: Usually the same to `--model-name`. You can also use HF's model name.
- `--model-type`: Indicate the model type. {opt, flash_opt, gptj, gptneox}. The 'flash_' prefix uses flash attention to accelerate training.
- `--num-layers`: Number of Transformer layers **for each GPU**. E.g. OPT-1.3B has 24 layers, if we use two GPUs to form a pipeline, `--num-layers` should be 12.
- `--embedding-dim`: The hidden size of the model. OPT-1.3B is 2048, GPT-J-6B is 4096, GPT-NeoX-20B is 6144. This is used to create buffers.
- `--dist-url`: URL of rank 0 worker (master). It is the same to all workers. And this URL should be accessible by all workers. For local training (single machine multiple GPUs), this can be like `--dist-url tcp://127.0.0.1:7033`
- `--world-size`: The total number of workers. `world-size == pipeline-group-size * data-group-size`
- `--pipeline-group-size`: Number of GPU workers for each pipeline
- `--data-group-size`: Number of data parallel workers. Also the number of pipelines.
- `--net-interface`: Network interface. Should be consistent with `GLOO_SOCKET_IFNAME` and `NCCL_SOCKET_IFNAME`.

The following arguments can be tuned / changed:
- `--optimizer`: Optimizer type. {adam, 8bit-adam} (8bit-adam requires `pip install bitsandbytes`)
- `--load-pretrained-model`: Whether to load model weights. Usually `true`.
- `--task-name`: The task name or the path of a `jsonl` file. For multi-task training separate task names by `,`. 
   There is an optional sampling weight after each task name, separated by `:` (default is 1.0). Sampling weights will be normalized. 
   E.g. it should be like `--task-name cot:0.1,/path_task0.jsonl:1.0,/path_task0.jsonl:1.0,/path_task0.jsonl:1.0`.
- `--checkpoint-path`: Path to save fine-tuned checkpoints.
- `--checkpoint-steps`: Save ckpt every `checkpoint-steps`.
- `--total-steps`: Total number of steps for training. (This counts all `gradient-accumulate-step`s.)
- `--warmup-steps`: LR warmup steps.
- `--lr`: learning rate
- `--seq-length`: sequence length
- `--batch-size`: batch size for each GPU device (of each gradient accumulation step).
- `--micro-batch-size`: micro batch size for pipeline parallelism. 1 works fine.
- `--gradient-accumulate-step`: Accumulate gradients for several steps before updating parameters. This is another way to achieve large batch sizes when GPU memory is not enough.
- `--dp-backend`: {gloo, nccl}
- `--dp-mode`: {allreduce, cocktail_sgd}. `cocktail_sgd` should always set `--dp-backend gloo`, `allreduce` performs better at `nccl`.

The following arguments usually do not change:
- `--fp16`: Flag to enable FP16 mixed precision training. Should always adding it for the current impl.
- `--pp-mode`: always `gpipe`
- `--profiling`: {no-profiling, tidy_profiling}. `tidy_profiling` will generate profile jsons.



## Optional Package

### Install bitsandbytes for 8bit Adam

```shell
pip install bitsandbytes # optional, to use 8bit-adam
```

### Install FlashAttention

https://github.com/HazyResearch/flash-attention

Install FlashAttention
```shell
export CUDA_HOME=/usr/local/cuda-11.8
git clone https://github.com/HazyResearch/flash-attention.git
cd flash-attention
git checkout tags/v1.0.4
pip install .
```

Install other optimized kernels:
```shell
cd flash-attention/csrc/rotary
pip install .
```

### Install Xformers

```shell
export CUDA_HOME=/usr/local/cuda-11.8
git clone https://github.com/facebookresearch/xformers.git
cd xformers
git submodule update --init --recursive
pip install .
```

### Install Nvidia Apex

```shell
export CUDA_HOME=/usr/local/cuda-11.8
git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
```
