
export WANDB_NAME=llama-7b-instruct-two-nodes

netif=enp19s0
master_ip=172.27.6.23
export GLOO_SOCKET_IFNAME=${netif}
export NCCL_SOCKET_IFNAME=${netif}
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export QUANT_BITS=4
export TOPK_RATIO=0.2
export RANDOMP_RATIO=0.1
export SHOW_DATA=0

ARGS="--model-name ./llama-7b-shard \
--tokenizer-name ./llama-7b-shard \
--project-name demo \
--model-type flash_llama \
--optimizer adam \
--seed 42 \
--load-pretrained-model true \
--task-name ni_dehelm:0.2,p3_dehelm:0.2,pile:0.6 \
--checkpoint-path ./model_ckpts/$WANDB_NAME \
--num-layers 8 --embedding-dim 4096 \
--total-steps 10000 --warmup-steps 10 --train-warmup-steps 0 \
--checkpoint-steps 1000 \
--lr 2e-5 --seq-length 2048 --batch-size 32 --micro-batch-size 2 --gradient-accumulate-step 1 \
--dist-url tcp://${master_ip}:7033 \
--world-size 16 --pipeline-group-size 4 --data-group-size 4 \
--job-id 0 --net-interface ${netif} \
--fp16 \
--dp-backend nccl \
--dp-mode allreduce \
--pp-mode gpipe --profiling no-profiling"

(trap 'kill 0' SIGINT; \
python dist_lm_train.py $(echo ${ARGS}) --cuda-id 0 --rank 8 \
    & \
python dist_lm_train.py $(echo ${ARGS}) --cuda-id 1 --rank 9 \
    & \
python dist_lm_train.py $(echo ${ARGS}) --cuda-id 2 --rank 10 \
    & \
python dist_lm_train.py $(echo ${ARGS}) --cuda-id 3 --rank 11 \
    & \
python dist_lm_train.py $(echo ${ARGS}) --cuda-id 4 --rank 12 \
    & \
python dist_lm_train.py $(echo ${ARGS}) --cuda-id 5 --rank 13 \
    & \
python dist_lm_train.py $(echo ${ARGS}) --cuda-id 6 --rank 14 \
    & \
python dist_lm_train.py $(echo ${ARGS}) --cuda-id 7 --rank 15 \
    & \
wait)
