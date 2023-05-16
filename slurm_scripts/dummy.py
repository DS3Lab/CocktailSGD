
import os
import uuid

template = '''#!/bin/bash
#SBATCH --job-name=dummy
#SBATCH --time=999:59:00
#SBATCH --output=/work/logs/slurm_%j.log
#SBATCH --exclusive

cd /root/
rm -rf CocktailSGD
git clone https://github.com/DS3Lab/CocktailSGD.git
cd CocktailSGD
git checkout rp

ls -l
sleep 60

nvidia-smi

'''

if __name__ == '__main__':

    # with open('slurms_scrips/train_template.lsf.sh') as f:
    #     template = f.read()

    job_id = str(uuid.uuid4())
    pp_degree=2
    dp_degree=128
    n_layer_per_device=16
    node_size = 32

    template = template.replace('{{JOB_ID}}', job_id)
    template = template.replace('{{PP_DEGREE}}', str(pp_degree))
    template = template.replace('{{DP_DEGREE}}', str(dp_degree))
    template = template.replace('{{N_LAYER_PER_DEVICE}}', str(n_layer_per_device))

    with open('slurm_scripts/train_to_submit.slurm.sh', 'w') as f:
        f.write(template)
        
    for i in range(node_size):
        os.system('sbatch slurm_scripts/train_to_submit.slurm.sh')