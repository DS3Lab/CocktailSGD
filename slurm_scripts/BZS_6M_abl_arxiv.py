
import os
import uuid
import time

template = '''#!/bin/bash
#SBATCH --job-name=arxiv
#SBATCH --time=999:59:00
#SBATCH --output=/var/cr01_data/logs/slurm_%j.log
#SBATCH --exclusive

nvidia-smi

docker rm -f $(docker ps -a | grep -w lorrin | cut -d " " -f 1)

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/root/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/root/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/root/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

if [ -f "/root/miniconda3/etc/profile.d/mamba.sh" ]; then
    . "/root/miniconda3/etc/profile.d/mamba.sh"
fi
# <<< conda initialize <<<

cd /work/data/
rm -rf CocktailSGD
git clone https://github.com/DS3Lab/CocktailSGD.git
cd CocktailSGD
git checkout rp

source /root/.bashrc
conda activate cocktail

netif=enp12s0
master_ip=172.27.6.25
export GLOO_SOCKET_IFNAME=${netif}
export NCCL_SOCKET_IFNAME=${netif}
export WANDB_NAME=RP-1TT-arxiv-boost
export WANDB_ENTITY=asdfffjj
export WANDB_DISABLED=1

export QUANT_BITS=4
export TOPK_RATIO=0.2
export RANDOMP_RATIO=0.1

export SHOW_DATA=0

ARGS="--model-name /var/cr01_data/_root_fm_models_rp_1t_real_fp16 \
--tokenizer-name /var/cr01_data/_root_fm_models_rp_1t_real_fp16 \
--load-pretrained-model true \
--project-name redpajama \
--model-type flash_gptneox \
--optimizer fusedadam \
--seed 42 \
--task-name /var/cr01_data/tokenized_data/to_target/arxiv_en_to_target_128_text_document \
--checkpoint-path /var/cr01_data/model_ckpts/$WANDB_NAME \
--num-layers {{N_LAYER_PER_DEVICE}} --embedding-dim 4096 \
--initial-loss-scale 4096 \
--total-steps 159000 --warmup-steps 10 --train-warmup-steps 0 \
--stop-steps 2001 \
--checkpoint-steps 200 \
--lr 1e-5 --seq-length 2048 --batch-size 64 --micro-batch-size 2 --gradient-accumulate-step 6 \
--dist-url tcp://${master_ip}:8951 \
--world-size $(({{PP_DEGREE}}*{{DP_DEGREE}})) --pipeline-group-size {{PP_DEGREE}} --data-group-size {{DP_DEGREE}} \
--job-id {{JOB_ID}} --net-interface ${netif} \
--fp16 \
--dp-backend nccl \
--dp-mode allreduce \
--pp-mode gpipe --profiling no-profiling"

(trap 'kill 0' SIGINT; \
python -u dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 0 --rank 0 \
    & \
python -u dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 1 --rank 0 \
    & \
python -u dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 2 --rank 0 \
    & \
python -u dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 3 --rank 0 \
    & sleep 2 ; \
python -u dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 4 --rank 0 \
    & \
python -u dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 5 --rank 0 \
    & \
python -u dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 6 --rank 0 \
    & \
python -u dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 7 --rank 0 \
    & sleep 2 ; \
wait)
'''

if __name__ == '__main__':

    job_id = str(uuid.uuid4())
    pp_degree=4
    dp_degree=8
    n_layer_per_device=8
    node_size=4

    template = template.replace('{{JOB_ID}}', job_id)
    template = template.replace('{{PP_DEGREE}}', str(pp_degree))
    template = template.replace('{{DP_DEGREE}}', str(dp_degree))
    template = template.replace('{{N_LAYER_PER_DEVICE}}', str(n_layer_per_device))

    with open('slurm_scripts/train_to_submit.slurm.sh', 'w') as f:
        f.write(template)
        
    for i in range(node_size):
        os.system('sbatch slurm_scripts/train_to_submit.slurm.sh')
        if i == 0:
            time.sleep(6)
