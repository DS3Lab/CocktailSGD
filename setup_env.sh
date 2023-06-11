#!/bin/bash

# This script fails when any of its commands fail.
set -e

wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run  --silent --toolkit

export CUDA_HOME=/usr/local/cuda-11.8

mamba create -n cocktail python=3.10
mamba activate cocktail

mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
mamba install -c conda-forge cupy nccl cudatoolkit=11.8 -y

pip install --upgrade pip
pip install --no-input transformers
pip install --no-input sentencepiece
pip install --no-input datasets
pip install --no-input netifaces
pip install --no-input zstandard
pip install --no-input wandb

rm -rf flash-attention
git clone https://github.com/HazyResearch/flash-attention.git
cd flash-attention
git checkout tags/v1.0.4
pip install .
cd ..

cd flash-attention/csrc/rotary && pip install . && ../..
cd flash-attention/csrc/xentropy && pip install . && ../..

git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--fast_layer_norm" ./
cd ..

git clone https://github.com/facebookresearch/xformers.git
cd xformers
git submodule update --init --recursive
pip install .
cd ..

