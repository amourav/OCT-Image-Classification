#!/bin/bash
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16000M        # memory per node
#SBATCH --time=0-00:05      # time (DD-HH:MM)
#SBATCH --output=../logs/%x-%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=def-amartel

module load cuda cudnn 
source activate
python --version


xPath="../PreprocessedData/preprocessedForCNN/224 x 224 n/imgData_(224, 224, 3)_test.npy"
yPath="../PreprocessedData/preprocessedForCNN/224 x 224 n/targetData_(224, 224, 3)_test.csv"
mPath="../modelOutput/compareModels4/VGG16_2019-06-13_21_37_default_224_norm/VGG16.hdf5"
note="_testSet_"

echo "$xPath"
echo "$yPath"
echo "$mPath"
echo "$note"

python ./modelPred.py -x "$xPath" -y "$yPath" -m "$mPath" -n "$note"
