#!/bin/bash
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16000M        # memory per node
#SBATCH --time=0-00:10      # time (DD-HH:MM)
#SBATCH --output=../logs/%x-%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=def-amartel

module load cuda cudnn 
source activate
python --version


xtrnPath="../PreprocessedData/preprocessedForCNN/224x224/imgData_(224, 224, 3)_train_n1000.npy"
xvalPath="../PreprocessedData/preprocessedForCNN/224x224/imgData_(224, 224, 3)_val.npy"
xtestPath="../PreprocessedData/preprocessedForCNN/224x224/imgData_(224, 224, 3)_test.npy"
model="VGG16"
#weights="../pretrainedModelWeights/ResNet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
output="../modelOutput/featureExtraction/"

python ./featureExtraction.py -x "$xtrnPath" -o "$output" -m "$model" -n "trn" #-w "$weights"
python ./featureExtraction.py -x "$xvalPath" -o "$output" -m "$model" -n "val" #-w "$weights"
python ./featureExtraction.py -x "$xtestPath" -o "$output" -m "$model" -n "test" #-w "$weights"
