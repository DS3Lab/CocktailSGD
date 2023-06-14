
import os
import uuid
import time

template = '''#!/bin/bash
#SBATCH --job-name=bzs_6m_lr_1e5
#SBATCH --time=999:59:00
#SBATCH --output=/var/cr01_data/logs/slurm_%j.log
#SBATCH --exclusive

nvidia-smi

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
export WANDB_NAME=RP-7B-1T-further
export WANDB_ENTITY=asdfffjj
export WANDB_DISABLED=1

export QUANT_BITS=4
export TOPK_RATIO=0.2
export RANDOMP_RATIO=0.1

export SHOW_DATA=0

ARGS="--model-name /var/cr01_data/RedPajama-Code-1B \
--tokenizer-name /var/cr01_data/RedPajama-Code-1B \
--load-pretrained-model false \
--project-name redpajama \
--model-type flash_gptneox \
--optimizer fusedadam \
--seed 42 \
--task-name \
/var/cr05_data/tokenized-starcoderdata/ada:0.0002558905195007171,\
/var/cr05_data/tokenized-starcoderdata/agda:9.616486176014486e-05,\
/var/cr05_data/tokenized-starcoderdata/alloy:8.86925475737666e-06,\
/var/cr05_data/tokenized-starcoderdata/antlr:6.220339146332516e-05,\
/var/cr05_data/tokenized-starcoderdata/applescript:1.2519750400457396e-05,\
/var/cr05_data/tokenized-starcoderdata/assembly:0.0017223091158793437,\
/var/cr05_data/tokenized-starcoderdata/augeas:7.380063393953833e-07,\
/var/cr05_data/tokenized-starcoderdata/awk:3.2274598663915956e-05,\
/var/cr05_data/tokenized-starcoderdata/batchfile:0.0003556795195346786,\
/var/cr05_data/tokenized-starcoderdata/bluespec:3.513437322907307e-05,\
/var/cr05_data/tokenized-starcoderdata/c:0.06144428604983385,\
/var/cr05_data/tokenized-starcoderdata/c-sharp:0.044411165630011566,\
/var/cr05_data/tokenized-starcoderdata/clojure:0.0005846723437013604,\
/var/cr05_data/tokenized-starcoderdata/cmake:0.0005100941673685305,\
/var/cr05_data/tokenized-starcoderdata/coffeescript:0.0008532934725566479,\
/var/cr05_data/tokenized-starcoderdata/common-lisp:0.0012157072999383093,\
/var/cr05_data/tokenized-starcoderdata/cpp:0.05284159654650988,\
/var/cr05_data/tokenized-starcoderdata/css:0.011904358542878702,\
/var/cr05_data/tokenized-starcoderdata/cuda:0.0005825769328448628,\
/var/cr05_data/tokenized-starcoderdata/dart:0.003910590162979079,\
/var/cr05_data/tokenized-starcoderdata/dockerfile:0.000744042176956867,\
/var/cr05_data/tokenized-starcoderdata/elixir:0.0008588944135252735,\
/var/cr05_data/tokenized-starcoderdata/elm:0.00033057412531060346,\
/var/cr05_data/tokenized-starcoderdata/emacs-lisp:0.0005344088405146819,\
/var/cr05_data/tokenized-starcoderdata/erlang:0.000749933048987398,\
/var/cr05_data/tokenized-starcoderdata/f-sharp:0.0007027270006353575,\
/var/cr05_data/tokenized-starcoderdata/fortran:0.0019428412245122426,\
/var/cr05_data/tokenized-starcoderdata/git-commits-cleaned:0.05132823547547185,\
/var/cr05_data/tokenized-starcoderdata/github-issues-filtered-structured:0.09840178533222153,\
/var/cr05_data/tokenized-starcoderdata/glsl:0.0005728378849017701,\
/var/cr05_data/tokenized-starcoderdata/go:0.02602941769705201,\
/var/cr05_data/tokenized-starcoderdata/groovy:0.0010506969896602809,\
/var/cr05_data/tokenized-starcoderdata/haskell:0.0029100380683437887,\
/var/cr05_data/tokenized-starcoderdata/html:0.03247247661366635,\
/var/cr05_data/tokenized-starcoderdata/idris:3.720342671630655e-05,\
/var/cr05_data/tokenized-starcoderdata/isabelle:7.638365612742217e-05,\
/var/cr05_data/tokenized-starcoderdata/java:0.09651516437634966,\
/var/cr05_data/tokenized-starcoderdata/java-server-pages:0.001149734804670215,\
/var/cr05_data/tokenized-starcoderdata/javascript:0.08014794971230074,\
/var/cr05_data/tokenized-starcoderdata/json:0.010275196369978761,\
/var/cr05_data/tokenized-starcoderdata/julia:0.0017026728757775736,\
/var/cr05_data/tokenized-starcoderdata/jupyter-scripts-dedup-filtered:0.01056839574567284,\
/var/cr05_data/tokenized-starcoderdata/jupyter-structured-clean-dedup:0.00910175311165547,\
/var/cr05_data/tokenized-starcoderdata/kotlin:0.007217807428763905,\
/var/cr05_data/tokenized-starcoderdata/lean:0.00013129923498921792,\
/var/cr05_data/tokenized-starcoderdata/literate-agda:7.28781260152941e-06,\
/var/cr05_data/tokenized-starcoderdata/literate-coffeescript:6.734307846982872e-06,\
/var/cr05_data/tokenized-starcoderdata/literate-haskell:7.865038988413656e-05,\
/var/cr05_data/tokenized-starcoderdata/lua:0.0034078497016352387,\
/var/cr05_data/tokenized-starcoderdata/makefile:0.0015894416174189105,\
/var/cr05_data/tokenized-starcoderdata/maple:5.535047545465375e-06,\
/var/cr05_data/tokenized-starcoderdata/markdown:0.12944803833763702,\
/var/cr05_data/tokenized-starcoderdata/mathematica:0.0012941995456068611,\
/var/cr05_data/tokenized-starcoderdata/matlab:3.1628843116945e-07,\
/var/cr05_data/tokenized-starcoderdata/ocaml:0.0011874126640332755,\
/var/cr05_data/tokenized-starcoderdata/pascal:0.0017656933456880867,\
/var/cr05_data/tokenized-starcoderdata/perl:0.0027983091800331803,\
/var/cr05_data/tokenized-starcoderdata/php:0.06589414798795684,\
/var/cr05_data/tokenized-starcoderdata/powershell:0.0012876233819754629,\
/var/cr05_data/tokenized-starcoderdata/prolog:1.0503411651752151e-05,\
/var/cr05_data/tokenized-starcoderdata/protocol-buffer:0.00035884240384637307,\
/var/cr05_data/tokenized-starcoderdata/python:0.07451307363074752,\
/var/cr05_data/tokenized-starcoderdata/r:0.0003090533333064488,\
/var/cr05_data/tokenized-starcoderdata/racket:3.691349565440123e-05,\
/var/cr05_data/tokenized-starcoderdata/restructuredtext:0.005316637205058238,\
/var/cr05_data/tokenized-starcoderdata/rmarkdown:8.379007689064013e-05,\
/var/cr05_data/tokenized-starcoderdata/ruby:0.009110424686143365,\
/var/cr05_data/tokenized-starcoderdata/rust:0.009277187761477458,\
/var/cr05_data/tokenized-starcoderdata/sas:9.948589028742408e-05,\
/var/cr05_data/tokenized-starcoderdata/scala:0.005685100048686015,\
/var/cr05_data/tokenized-starcoderdata/scheme:0.0002353844862132311,\
/var/cr05_data/tokenized-starcoderdata/shell:0.004890688939065413,\
/var/cr05_data/tokenized-starcoderdata/smalltalk:0.0008474684939492771,\
/var/cr05_data/tokenized-starcoderdata/solidity:0.0009814957166573315,\
/var/cr05_data/tokenized-starcoderdata/sparql:2.7714773781223055e-05,\
/var/cr05_data/tokenized-starcoderdata/sql:0.010892301456559622,\
/var/cr05_data/tokenized-starcoderdata/stan:1.5300452857822144e-05,\
/var/cr05_data/tokenized-starcoderdata/standard-ml:0.0002058115178988875,\
/var/cr05_data/tokenized-starcoderdata/stata:0.0003175667635787598,\
/var/cr05_data/tokenized-starcoderdata/systemverilog:0.00041178118001335975,\
/var/cr05_data/tokenized-starcoderdata/tcl:0.0003961117239858399,\
/var/cr05_data/tokenized-starcoderdata/tcsh:1.996570721757153e-05,\
/var/cr05_data/tokenized-starcoderdata/tex:0.007326597470401564,\
/var/cr05_data/tokenized-starcoderdata/thrift:1.5722170766048078e-05,\
/var/cr05_data/tokenized-starcoderdata/typescript:0.033771565451271704,\
/var/cr05_data/tokenized-starcoderdata/verilog:4.217179082259333e-07,\
/var/cr05_data/tokenized-starcoderdata/vhdl:0.000988559491620116,\
/var/cr05_data/tokenized-starcoderdata/visual-basic:0.0012748927726208925,\
/var/cr05_data/tokenized-starcoderdata/xslt:0.00038911384244621583,\
/var/cr05_data/tokenized-starcoderdata/yacc:0.00011627553450866905,\
/var/cr05_data/tokenized-starcoderdata/yaml:0.006749858694848704,\
/var/cr05_data/tokenized-starcoderdata/zig:0.00016055591487239206 \
--checkpoint-path /var/cr05_data/model_ckpts/$WANDB_NAME \
--num-layers {{N_LAYER_PER_DEVICE}} --embedding-dim 2048 \
--initial-loss-scale 4096 \
--total-steps 262144 --warmup-steps 1000 --train-warmup-steps 0 \
--stop-steps 262145 \
--checkpoint-steps 1000 \
--lr 2e-4 --seq-length 8192 --batch-size 16 --micro-batch-size 16 --gradient-accumulate-step 4 \
--dist-url tcp://${master_ip}:8956 \
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
    & \
python -u dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 4 --rank 0 \
    & \
python -u dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 5 --rank 0 \
    & \
python -u dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 6 --rank 0 \
    & \
python -u dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 7 --rank 0 \
    & \
wait)


'''

if __name__ == '__main__':

    job_id = str(uuid.uuid4())
    pp_degree=1
    dp_degree=8
    n_layer_per_device=16
    node_size=1

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
