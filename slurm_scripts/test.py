
#!/bin/bash
#SBATCH --job-name=bzs_4m_lr_4e5
#SBATCH --time=999:59:00
#SBATCH --output=/work/logs/slurm_%j.log
#SBATCH --nodes=32

nvidia-smi

netif=enp12s0
export GLOO_SOCKET_IFNAME=${netif}
export NCCL_SOCKET_IFNAME=${netif}
export WANDB_ENTITY=asdfffjj


source /root/.bashrc

mamba activate neox
cd /root/gpt-neox
git pull

python ./deepy.py train.py  configs/rp_7b_512_nodes_continue.yml configs/rp_data_setup.yml
