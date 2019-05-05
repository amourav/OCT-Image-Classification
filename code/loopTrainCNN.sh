#!/bin/bash
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=2   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16000M        # memory per node
#SBATCH --time=0-5:00      # time (DD-HH:MM)
#SBATCH --output=../modelOutput/logs/%x-%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=def-amartel
#SBATCH --array=1-10

module load cuda cudnn 
source activate
python --version


basePath="../PreprocessedData/preprocessedForCNN/subsample_$SLURM_ARRAY_TASK_ID"
xtrnPath="$basePath/imgData_train_1000.npy"
xvalPath="$basePath/imgData_val.npy"
ytrnPath="$basePath/targetData_train.npy"
yvalPath="$basePath/targetData_val.npy"
model="Xception"
output="../modelOutput/subsample/$SLURM_ARRAY_TASK_ID"


python ./trainCNN.py -xtrn "$xtrnPath" -xval "$xvalPath" -ytrn "$ytrnPath" -yval "$yvalPath" -o "$output" -m "$model"
