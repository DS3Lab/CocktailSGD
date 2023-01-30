# CocktailSGD

## Setup:


- Install PyTorch env: 

      conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
      conda install -c conda-forge cupy nccl cudatoolkit=11.6
      
      pip install transformers==4.21.1
      pip install datasets
      pip install netifaces
      pip install zstandard

## Run Example

Please refer to `run_example.sh`, which shows an example to fine-tune OPT-1.3B on mmlu-cot data.
The script will launch 8 processes with a data parallel degree of 4 and a pipeline parallel degree of 2.

In case of geo-distributed training, please first make sure the network interface is correctly set and the master (rank 0 worker) IP and port are accesible by all the workers.
After that, run the corresponding process on each GPU node.



