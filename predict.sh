#!/bin/bash
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16000M        # memory per node
#SBATCH --time=0-00:05      # time (DD-HH:MM)
#SBATCH --output=logs/%x-%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=def-amartel


module load cuda cudnn 
source activate
python --version


input="./RawData/testDir"
output="./modelOutput/Xception/Xception_2019-06-09_22_38_"
model="./modelOutput/Xception/Xception_2019-06-09_22_38_/Xception.hdf5"
#"/scratch/amourav/Python/OCT-Image-Classification/modelOutput/Xception/Xception_2019-06-09_22_38_/Xception.hdf5"
#RX=224
#RY=224
python ./predict.py -i "$input" -o "$output" -m "$model" #-Rx "$RX" -Ry "$RY"