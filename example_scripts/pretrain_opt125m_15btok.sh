netif=lo
export GLOO_SOCKET_IFNAME=${netif}
export NCCL_SOCKET_IFNAME=${netif}
export WANDB_NAME=opt-125m-pretrain-pile-allreduce-nccl-15b-tok

export QUANT_BITS=4
export TOPK_RATIO=0.2
export RANDOMP_RATIO=0.1

export SHOW_DATA=0

ARGS="--model-name ./empty_model_configs/opt-125m \
--tokenizer-name ./empty_model_configs/opt-125m \
--load-pretrained-model false \
--project-name cocktail-sgd \
--model-type flash_opt \
--optimizer adam \
--seed 42 \
--task-name pile \
--checkpoint-path ./model_ckpts/$WANDB_NAME \
--num-layers 12 --embedding-dim 768 \
--total-steps 30000 --warmup-steps 300 --train-warmup-steps 0 \
--checkpoint-steps 500 \
--evaluation-steps 1000 \
--evaluation-data pile \
--lr 6e-4 --seq-length 2048 --batch-size 32 --micro-batch-size 1 --gradient-accumulate-step 1 \
--dist-url tcp://127.0.0.1:7033 \
--world-size 8 --pipeline-group-size 1 --data-group-size 8 \
--job-id 0 --net-interface ${netif} \
--fp16 \
--dp-backend nccl \
--dp-mode allreduce \
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

