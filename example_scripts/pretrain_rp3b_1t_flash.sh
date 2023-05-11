netif=lo
export GLOO_SOCKET_IFNAME=${netif}
export NCCL_SOCKET_IFNAME=${netif}
export WANDB_NAME=RP-3B-1TT-new
export WANDB_ENTITY=asdfffjj

export QUANT_BITS=4
export TOPK_RATIO=0.2
export RANDOMP_RATIO=0.1

export SHOW_DATA=0

# NOTE: for debug purpose, the evaluation-steps is set to be 100 and the evaluation-num-batch is set to 10
# feel free to increase evaluation-steps and remove evaluation-num-batch (for full eval).

ARGS="--model-name /root/fm/models/_root_fm_models_rp_3b_1t_real_fp16 \
--tokenizer-name /root/fm/models/_root_fm_models_rp_3b_1t_real_fp16 \
--load-pretrained-model true \
--project-name redpajama \
--model-type flash_gptneox \
--optimizer adam \
--seed 42 \
--task-name rp_arxiv:0.1045484965,\
rp_book:0.01603303122,\
rp_c4:0.5082132003,\
rp_common_crawl:0.3385429763,\
rp_github:0.02514479823,\
rp_stackexchange:0.0008339357193,\
rp_wikipedia:0.006683561774 \
--checkpoint-path ./model_ckpts/$WANDB_NAME \
--num-layers 32 --embedding-dim 1 \
--total-steps 200000 --warmup-steps 100 --train-warmup-steps 0 \
--stop-steps 8001 \
--checkpoint-steps 1000 \
--lr 1e-5 --seq-length 2048 --batch-size 16 --micro-batch-size 16 --gradient-accumulate-step 16 \
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

