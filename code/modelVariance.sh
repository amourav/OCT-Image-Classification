#!/bin/bash
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=2   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16000M        # memory per node
#SBATCH --time=0-00:30      # time (DD-HH:MM)
#SBATCH --output=../Evaluation/%x-%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=def-amartel

module load cuda cudnn 
source activate
python --version



m="../modelOutput/subsample_Xception"
d="../PreprocessedData/preprocessedForCNN"
out="../Evaluation/model variance/Xception"
modelName="Xception"

python ./modelVariance.py -model "$m" -data "$d" -o "$out" -m "$modelName"