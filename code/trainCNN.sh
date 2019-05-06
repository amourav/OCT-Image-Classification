#!/bin/bash
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=3   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16000M        # memory per node
#SBATCH --time=0-02:00      # time (DD-HH:MM)
#SBATCH --output=../modelOutput/%x-%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=def-amartel

module load cuda cudnn 
source activate
python --version



xtrnPath="../PreprocessedData/preprocessedForCNN/imgData_train_1000.npy"
xvalPath="../PreprocessedData/preprocessedForCNN/imgData_val.npy"
ytrnPath="../PreprocessedData/preprocessedForCNN/targetData_train.npy"
yvalPath="../PreprocessedData/preprocessedForCNN/targetData_val.npy"
model="VGG16"
output="../modelOutput"
aug=1

python ./trainCNN.py -xtrn "$xtrnPath" -xval "$xvalPath" -ytrn "$ytrnPath" -yval "$yvalPath" -o "$output" -m "$model" -aug "$aug"
