netif=lo
export GLOO_SOCKET_IFNAME=${netif}
export NCCL_SOCKET_IFNAME=${netif}
export WANDB_NAME=RP-3B-1TT-test2
export WANDB_ENTITY=asdfffjj
export WANDB_DISABLED=1

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
--task-name rp_arxiv:0.1045484965 \
--checkpoint-path ./model_ckpts/$WANDB_NAME \
--num-layers 1 --embedding-dim 1 \
--total-steps 200000 --warmup-steps 100 --train-warmup-steps 0 \
--stop-steps 800001 \
--checkpoint-steps 10 \
--lr 1e-5 --seq-length 2048 --batch-size 16 --micro-batch-size 16 --gradient-accumulate-step 1 \
--dist-url tcp://127.0.0.1:7033 \
--world-size 4 --pipeline-group-size 1 --data-group-size 4 \
--job-id 0 --net-interface ${netif} \
--fp16 \
--dp-backend nccl \
--dp-mode allreduce \
--pp-mode gpipe --profiling no-profiling"

(trap 'kill 0' SIGINT; \
python dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 4 --rank 0 \
    & \
python dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 5 --rank 1 \
    & \
python dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 6 --rank 2 \
    & \
python dist_lm_sharded_train.py $(echo ${ARGS}) --cuda-id 7 --rank 3 \
    & \
wait)

