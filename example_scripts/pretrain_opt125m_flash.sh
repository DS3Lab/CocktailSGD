netif=lo
export GLOO_SOCKET_IFNAME=${netif}
export NCCL_SOCKET_IFNAME=${netif}
export WANDB_NAME=opt-125m-flash

export QUANT_BITS=4
export TOPK_RATIO=0.2
export RANDOMP_RATIO=0.1

export SHOW_DATA=0

# NOTE: for debug purpose, the evaluation-steps is set to be 100 and the evaluation-num-batch is set to 10
# feel free to increase evaluation-steps and remove evaluation-num-batch (for full eval).

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
--total-steps 200 --warmup-steps 10 --train-warmup-steps 0 \
--checkpoint-steps 100 \
--evaluation-steps 10 \
--evaluation-data pile \
--evaluation-num-batch 10 \
--lr 1e-3 --seq-length 2048 --batch-size 32 --micro-batch-size 8 --gradient-accumulate-step 1 \
--dist-url tcp://127.0.0.1:7033 \
--world-size 4 --pipeline-group-size 1 --data-group-size 4 \
--job-id 0 --net-interface ${netif} \
--fp16 \
--dp-backend gloo \
--dp-mode cocktail_sgd \
--pp-mode gpipe --profiling no-profiling"

(trap 'kill 0' SIGINT; \
python dist_lm_train.py $(echo ${ARGS}) --cuda-id 4 --rank 0 \
    & \
python dist_lm_train.py $(echo ${ARGS}) --cuda-id 5 --rank 1 \
    & \
python dist_lm_train.py $(echo ${ARGS}) --cuda-id 6 --rank 2 \
    & \
python dist_lm_train.py $(echo ${ARGS}) --cuda-id 7 --rank 3 \
    & \
wait)

