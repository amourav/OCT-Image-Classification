#!/bin/bash
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16000M        # memory per node
#SBATCH --time=0-00:10      # time (DD-HH:MM)
#SBATCH --output=logs/%x-%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=def-amartel


module load cuda cudnn 
source activate
python --version


input="./RawData/test/CNV"
output="./modelOutput/compareModels3/VGG16_2019-06-09_18_40_default_224_noprep"
model="./modelOutput/compareModels3/VGG16_2019-06-09_18_40_default_224_noprep/VGG16.hdf5"
RX=224
RY=224
python ./predict.py -i "$input" -o "$output" -m "$model" -Rx "$RX" -Ry "$RY"