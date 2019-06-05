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


xPath="../PreprocessedData/preprocessedForCNN/299x299/imgData_(299, 299, 3)_test.npy"
yPath="../PreprocessedData/preprocessedForCNN/299x299/targetData_(299, 299, 3)_test.npy"
mPath="../modelOutput/compareModels2/VGG16_dataAug_False2019-06-04_23_47_default/VGG16.hdf5"
note="_testSet_"

echo "$xPath"
echo "$yPath"
echo "$mPath"
echo "$note"

python ./modelPred.py -x "$xPath" -y "$yPath" -m "$mPath" -n "$note"
