
import os
import uuid
import time

template = '''#!/bin/bash
#SBATCH --job-name=abl_slice
#SBATCH --time=999:59:00
#SBATCH --output=/work/logs/slurm_%j.log
#SBATCH --exclusive

nvidia-smi

cd /root/
rm -rf CocktailSGD
git clone https://github.com/DS3Lab/CocktailSGD.git
cd CocktailSGD
git checkout rp

source /root/.bashrc
conda activate base

netif=enp12s0
master_ip=172.27.6.25
export GLOO_SOCKET_IFNAME=${netif}
export NCCL_SOCKET_IFNAME=${netif}
export WANDB_NAME=RP-7B-700BT-subtitle-fix
export WANDB_ENTITY=asdfffjj
export WANDB_DISABLED=1

export SHOW_DATA=0

ARGS="--model-name /work/data/_root_fm_models_rp_700b_real_fp16 \
--tokenizer-name /work/data/_root_fm_models_rp_700b_real_fp16 \
--load-pretrained-model true \
--project-name redpajama \
--model-type flash_gptneox \
--optimizer adam \
--seed 42 \
--task-name rp_subtitle \
--checkpoint-path /work/data/model_ckpts/$WANDB_NAME \
--num-layers {{N_LAYER_PER_DEVICE}} --embedding-dim 4096 \
--initial-loss-scale 512 \
--total-steps 238418 --warmup-steps 10 --train-warmup-steps 0 \
--stop-steps 238419 \
--checkpoint-steps 100 \
--lr 4e-5 --seq-length 2048 --batch-size 16 --micro-batch-size 1 --gradient-accumulate-step 8 \
--dist-url tcp://${master_ip}:7026 \
--world-size $(({{PP_DEGREE}}*{{DP_DEGREE}})) --pipeline-group-size {{PP_DEGREE}} --data-group-size {{DP_DEGREE}} \
--job-id {{JOB_ID}} --net-interface ${netif} \
--fp16 \
--dp-backend nccl \
--dp-mode allreduce \
--pp-mode gpipe --profiling no-profiling"

(trap 'kill 0' SIGINT; \
RP_PREFIX=/work/data/data_0 python -u dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 0 --rank 0 \
    & \
RP_PREFIX=/work/data/data_0 python -u dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 1 --rank 0 \
    & sleep 2 ; \
RP_PREFIX=/work/data/data_1 python -u dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 2 --rank 0 \
    & \
RP_PREFIX=/work/data/data_1 python -u dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 3 --rank 0 \
    & sleep 2 ; \
RP_PREFIX=/work/data/data_2 python -u dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 4 --rank 0 \
    & \
RP_PREFIX=/work/data/data_2 python -u dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 5 --rank 0 \
    & sleep 2 ; \
RP_PREFIX=/work/data/data_3 python -u dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 6 --rank 0 \
    & \
RP_PREFIX=/work/data/data_3 python -u dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 7 --rank 0 \
    & sleep 2 ; \
wait)


'''

if __name__ == '__main__':

    job_id = str(uuid.uuid4())
    pp_degree=2
    dp_degree=16
    n_layer_per_device=16
    node_size=4

    template = template.replace('{{JOB_ID}}', job_id)
    template = template.replace('{{PP_DEGREE}}', str(pp_degree))
    template = template.replace('{{DP_DEGREE}}', str(dp_degree))
    template = template.replace('{{N_LAYER_PER_DEVICE}}', str(n_layer_per_device))

    with open('slurm_scripts/train_to_submit.slurm.sh', 'w') as f:
        f.write(template)
        
    for i in range(node_size):
        os.system('sbatch slurm_scripts/train_to_submit.slurm.sh')
        time.sleep(10)