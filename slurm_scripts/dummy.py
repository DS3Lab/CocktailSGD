
import os
import uuid

template = '''#!/bin/bash
#SBATCH --job-name=opt_1.3b
#SBATCH --gpus=1 
#SBATCH --gres=gpumem:20g
#SBATCH --time=9:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/work/logs/slurm_%j.log

cd $HOME     # Change directory

nvidia-smi

'''

if __name__ == '__main__':

    # with open('slurms_scrips/train_template.lsf.sh') as f:
    #     template = f.read()

    job_id = str(uuid.uuid4())
    pp_degree=2
    dp_degree=128
    n_layer_per_device=16
    world_size = pp_degree * dp_degree

    template = template.replace('{{JOB_ID}}', job_id)
    template = template.replace('{{PP_DEGREE}}', str(pp_degree))
    template = template.replace('{{DP_DEGREE}}', str(dp_degree))
    template = template.replace('{{N_LAYER_PER_DEVICE}}', str(n_layer_per_device))

    with open('slurm_scripts/train_to_submit.slurm.sh', 'w') as f:
        f.write(template)
        
    for i in range(world_size):
        os.system('sbatch slurm_scripts/train_to_submit.slurm.sh')