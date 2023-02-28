#!/bin/bash

# first, create the destination folder 
#RUN_NAME=gpt-neo-1_3b-ni-1-log-perp-temp-2
# RUN_NAME=gpt-j-ni-1-uniform
RUN_NAME=gpt-neo-1_3b-ni-1-uniform
mkdir -p ./hf/$RUN_NAME

if grep -q "neo" <<< $RUN_NAME; then
    CONVERT_SCRIPT=convert_gptneo_to_hf.py
else 
    CONVERT_SCRIPT=convert_gptj_to_hf.py
fi

echo $CONVERT_SCRIPT

# open the model checkpoints folder and create subfolders for each checkpoint of the run
cd ./model_ckpts/$RUN_NAME
for i in `ls -d */`
do 
    cd /root/CocktailSGD-main/
    mkdir -p ./hf/$RUN_NAME/$i 
    echo ./hf/$RUN_NAME/$i
    python $CONVERT_SCRIPT --ckpt-path ./model_ckpts/$RUN_NAME/$i --save-path ./hf/$RUN_NAME/$i
done 



