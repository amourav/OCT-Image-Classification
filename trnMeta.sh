#!/bin/bash
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16000M        # memory per node
#SBATCH --time=0-06:00      # time (DD-HH:MM)
#SBATCH --output=logs/%x-%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=def-amartel

module load cuda cudnn 
source activate
python --version


trnPath="./RawData/OCT2017/train"
valPath="./RawData/OCT2017/val"
debug=0

python ./trnMeta.py -iTrn "$trnPath" -iVal "$valPath" -d "$debug"