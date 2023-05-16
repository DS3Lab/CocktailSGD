
import os
import uuid

template = '''#!/bin/bash
#SBATCH --job-name=dummy
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

netif=enp12s0
master_ip=172.27.6.25
export GLOO_SOCKET_IFNAME=${netif}
export NCCL_SOCKET_IFNAME=${netif}
export WANDB_NAME=RP-7B-700BT-debug
export WANDB_ENTITY=asdfffjj
export WANDB_API_KEY=6fae2eb8adcb7b687f143acdf784e301ad45d82a

# INODE
# export RP_PREFIX=/var/data/data_$INODE
export SHOW_DATA=0

# NOTE: for debug purpose, the evaluation-steps is set to be 100 and the evaluation-num-batch is set to 10
# feel free to increase evaluation-steps and remove evaluation-num-batch (for full eval).

ARGS="--model-name /var/data/data/_root_fm_models_rp_700b_real_fp16 \
--tokenizer-name /var/data/data/_root_fm_models_rp_700b_real_fp16 \
--load-pretrained-model true \
--project-name redpajama \
--model-type flash_gptneox \
--optimizer adam \
--seed 42 \
--task-name \
rp_arxiv:0.052,\
rp_book:0.03,\
rp_c4:0.460,\
rp_common_crawl:0.26,\
rp_github_no_markdown:0.1,\
rp_github_md:0.035,\
rp_stackexchange:0.014,\
rp_wikipedia:0.04 \
--checkpoint-path /var/data/model_ckpts/$WANDB_NAME \
--num-layers {{N_LAYER_PER_DEVICE}} --embedding-dim 4096 \
--initial-loss-scale 512 \
--total-steps 238418 --warmup-steps 100 --train-warmup-steps 0 \
--stop-steps 238419 \
--checkpoint-steps 500 \
--lr 6e-5 --seq-length 2048 --batch-size 16 --micro-batch-size 1 --gradient-accumulate-step 8 \
--dist-url tcp://${master_ip}:7026 \
--world-size $(({{PP_DEGREE}}*{{DP_DEGREE}})) --pipeline-group-size {{PP_DEGREE}} --data-group-size {{DP_DEGREE}} \
--job-id {{JOB_ID}} --net-interface ${netif} \
--fp16 \
--dp-backend nccl \
--dp-mode allreduce \
--pp-mode gpipe --profiling no-profiling"

(trap 'kill 0' SIGINT; \
WANDB_DISABLED=1 RP_PREFIX=/var/data/data_0 /home/jue@together.xyz/miniconda3/bin/python -u dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 0 --rank 0 \
    & \
WANDB_DISABLED=1 RP_PREFIX=/var/data/data_0 /home/jue@together.xyz/miniconda3/bin/python -u dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 1 --rank 0 \
    & \
WANDB_DISABLED=1 RP_PREFIX=/var/data/data_1 /home/jue@together.xyz/miniconda3/bin/python -u dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 2 --rank 0 \
    & \
WANDB_DISABLED=1 RP_PREFIX=/var/data/data_1 /home/jue@together.xyz/miniconda3/bin/python -u dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 3 --rank 0 \
    & \
WANDB_DISABLED=1 RP_PREFIX=/var/data/data_2 /home/jue@together.xyz/miniconda3/bin/python -u dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 4 --rank 0 \
    & \
WANDB_DISABLED=1 RP_PREFIX=/var/data/data_2 /home/jue@together.xyz/miniconda3/bin/python -u dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 5 --rank 0 \
    & \
WANDB_DISABLED=1 RP_PREFIX=/var/data/data_3 /home/jue@together.xyz/miniconda3/bin/python -u dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 6 --rank 0 \
    & \
WANDB_DISABLED=1 RP_PREFIX=/var/data/data_3 /home/jue@together.xyz/miniconda3/bin/python -u dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 7 --rank 0 \
    & \
wait)


'''

if __name__ == '__main__':

    # with open('slurms_scrips/train_template.lsf.sh') as f:
    #     template = f.read()

    job_id = str(uuid.uuid4())
    pp_degree=2
    dp_degree=128
    n_layer_per_device=16
    node_size = 32

    template = template.replace('{{JOB_ID}}', job_id)
    template = template.replace('{{PP_DEGREE}}', str(pp_degree))
    template = template.replace('{{DP_DEGREE}}', str(dp_degree))
    template = template.replace('{{N_LAYER_PER_DEVICE}}', str(n_layer_per_device))

    with open('slurm_scripts/train_to_submit.slurm.sh', 'w') as f:
        f.write(template)
        
    for i in range(node_size):
        os.system('sbatch slurm_scripts/train_to_submit.slurm.sh')