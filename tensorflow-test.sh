#!/bin/bash
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=0-00:05      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID

module load cuda cudnn 
source activate


python --version
python ./tensorflow-test.py
