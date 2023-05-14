
netif=lo
master_ip=127.0.0.1

export GLOO_SOCKET_IFNAME=${netif}
export NCCL_SOCKET_IFNAME=${netif}
export WANDB_NAME=RP-7B-700BT-mosaic

export SHOW_DATA=0

# NOTE: for debug purpose, the evaluation-steps is set to be 100 and the evaluation-num-batch is set to 10
# feel free to increase evaluation-steps and remove evaluation-num-batch (for full eval).

ARGS="--model-name /root/data/_root_fm_models_rp_700b_real_fp16 \
--tokenizer-name /root/data/_root_fm_models_rp_700b_real_fp16 \
--load-pretrained-model true \
--project-name redpajama \
--model-type flash_gptneox \
--optimizer adam \
--seed 42 \
--task-name \
rp_arxiv:0.052,\
rp_book:0.03,\
rp_mc4:0.33,\
rp_c4:0.299,\
rp_common_crawl:0.1,\
rp_github:0.1,\
rp_stackexchange:0.014,\
rp_wikipedia:0.04 \
--checkpoint-path ./model_ckpts/$WANDB_NAME \
--num-layers 16 --embedding-dim 4096 \
--total-steps 200000 --warmup-steps 100 --train-warmup-steps 0 \
--stop-steps 8000001 \
--checkpoint-steps 1000 \
--lr 8e-5 --seq-length 2048 --batch-size 16 --micro-batch-size 1 --gradient-accumulate-step 8 \
--dist-url tcp://${master_ip}:7033 \
--world-size 256 --pipeline-group-size 2 --data-group-size 128 \
--job-id 0 --net-interface ${netif} \
--fp16 \
--dp-backend nccl \
--dp-mode allreduce \
--pp-mode gpipe --profiling no-profiling"

echo $((0+8*$INODE))
echo $((1+8*$INODE))
echo $((2+8*$INODE))
echo $((3+8*$INODE))
echo $((4+8*$INODE))
echo $((5+8*$INODE))
echo $((6+8*$INODE))
echo $((7+8*$INODE)) 

(trap 'kill 0' SIGINT; \
python dist_lm_train.py $(echo ${ARGS}) --cuda-id 0 --rank $((0+8*$INODE)) \
    & \
python dist_lm_train.py $(echo ${ARGS}) --cuda-id 1 --rank $((1+8*$INODE)) \
    & \
python dist_lm_train.py $(echo ${ARGS}) --cuda-id 2 --rank $((2+8*$INODE)) \
    & \
python dist_lm_train.py $(echo ${ARGS}) --cuda-id 3 --rank $((3+8*$INODE)) \
    & \
python dist_lm_train.py $(echo ${ARGS}) --cuda-id 4 --rank $((4+8*$INODE)) \
    & \
python dist_lm_train.py $(echo ${ARGS}) --cuda-id 5 --rank $((5+8*$INODE)) \
    & \
python dist_lm_train.py $(echo ${ARGS}) --cuda-id 6 --rank $((6+8*$INODE)) \
    & \
python dist_lm_train.py $(echo ${ARGS}) --cuda-id 7 --rank $((7+8*$INODE)) \
    & \
wait)

