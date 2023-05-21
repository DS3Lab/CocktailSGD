FROM nvidia/cuda:11.6.0-cudnn8-devel-ubuntu20.04

# init workdir
RUN mkdir -p /build
WORKDIR /build

# install common tool & conda
RUN apt update && \
    apt install wget -y && \
    apt install git -y && \
    apt install vim -y && \
    wget --quiet https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    mkdir -p /opt/conda/envs/alpa && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
# echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
# echo "conda activate base" >> ~/.bashrc

# install conda alpa env
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda create --name alpa python=3.9 -y && \
    conda activate alpa && \
    pip3 install --upgrade pip && \
    pip3 install cupy-cuda116 && \
    pip3 install torch==1.13 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116  && \
    pip3 install transformers && \
    pip3 install datasets && \
    pip3 install netifaces && \
    pip3 install zstandard && \
    pip3 install wandb && \
    pip3 install sentencepiece && \
    pip3 install accelerate && \
    pip3 install flash-attn && \
    echo "conda activate alpa" >> ~/.bashrc

RUN mkdir -p /app
COPY . /app
WORKDIR /app