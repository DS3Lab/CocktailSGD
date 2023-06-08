
netif=lo
export GLOO_SOCKET_IFNAME=${netif}
export NCCL_SOCKET_IFNAME=${netif}
export WANDB_NAME=rp-allreduce

export SHOW_DATA=0

ARGS="--model-name /root/fm/models/_root_fm_models_redpajama_7b \
--tokenizer-name /root/fm/models/_root_fm_models_redpajama_7b \
--project-name demo \
--model-type flash_gptneox \
--optimizer adam \
--seed 42 \
--load-pretrained-model true \
--task-name /root/ft_data/dolly.jsonl \
--checkpoint-path ./model_ckpts/$WANDB_NAME \
--num-layers 8 --embedding-dim 4096 \
--total-steps 2000 --warmup-steps 10 --train-warmup-steps 0 \
--checkpoint-steps 200 \
--lr 1e-5 --seq-length 2048 --batch-size 32 --micro-batch-size 2 --gradient-accumulate-step 1 \
--dist-url tcp://127.0.0.1:7033 \
--world-size 8 --pipeline-group-size 4 --data-group-size 2 \
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
