#!/bin/bash
#SBATCH --gres=gpu:0        # request GPU "generic resource"
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16000M        # memory per node
#SBATCH --time=0-00:05      # time (DD-HH:MM)
#SBATCH --output=logs/%x-%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=def-amartel


module load cuda cudnn 
source activate
python --version

yTruePath="./PreprocessedData/preprocessedForCNN/299 x 299/targetData_(299, 299, 3)_test.csv"
yPredPath="./modelOutput/VGG16/VGG16_2019-06-11_22_47_/VGG16_predictions.csv"

python ./evaluate.py -yTrue "$yTruePath" -yPred "$yPredPath"