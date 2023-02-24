netif=lo
export GLOO_SOCKET_IFNAME=${netif}
export NCCL_SOCKET_IFNAME=${netif}
export WANDB_NAME=h3-125m-pretrain-pile-cocktail-5btok-linear
# export WANDB_NAME=test

export QUANT_BITS=4
export TOPK_RATIO=0.5
export RANDOMP_RATIO=0.4

export SHOW_DATA=0

# the model name argument is IGNORED
ARGS="--model-name ./empty_model_configs/h3 \
--tokenizer-name gpt2 \
--load-pretrained-model false \
--project-name cocktail-sgd \
--model-type h3 \
--optimizer adam \
--seed 42 \
--task-name pile \
--checkpoint-path ./model_ckpts/$WANDB_NAME \
--num-layers 12 --embedding-dim 768 \
--total-steps 20000 --warmup-steps 200 --train-warmup-steps 1000 \
--checkpoint-steps 500 \
--evaluation-steps 4000 \
--evaluation-num-batch 256 \
--evaluation-data pile \
--lr 6e-4 --seq-length 2048 --batch-size 32 --micro-batch-size 1 --gradient-accumulate-step 1 \
--dist-url tcp://127.0.0.1:7033 \
--world-size 8 --pipeline-group-size 1 --data-group-size 8 \
--job-id 0 --net-interface ${netif} \
--dp-backend gloo \
--dp-mode cocktail_sgd \
--pp-mode gpipe --profiling no-profiling"

(trap 'kill 0' SIGINT; \
python dist_lm_train.py $(echo ${ARGS}) --cuda-id 0 --rank 0 \
    & \
python dist_lm_train.py $(echo ${ARGS}) --cuda-id 1 --rank 1 \
    & \
python dist_lm_train.py $(echo ${ARGS}) --cuda-id 2 --rank 2 \
    & \
python dist_lm_train.py $(echo ${ARGS}) --cuda-id 3 --rank 3 \
    & \
python dist_lm_train.py $(echo ${ARGS}) --cuda-id 4 --rank 4 \
    & \
python dist_lm_train.py $(echo ${ARGS}) --cuda-id 5 --rank 5 \
    & \
python dist_lm_train.py $(echo ${ARGS}) --cuda-id 6 --rank 6 \
    & \
python dist_lm_train.py $(echo ${ARGS}) --cuda-id 7 --rank 7 \
    & \
wait)

