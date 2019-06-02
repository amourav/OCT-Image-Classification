#!/bin/bash
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16000M        # memory per node
#SBATCH --time=0-01:30      # time (DD-HH:MM)
#SBATCH --output=../logs/%x-%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=def-amartel

module load cuda cudnn 
source activate
python --version


xtrnPath="../PreprocessedData/preprocessedForCNN/299x299/imgData_(299, 299, 3)_train_n1000.npy"
xvalPath="../PreprocessedData/preprocessedForCNN/299x299/imgData_(299, 299, 3)_val.npy"
ytrnPath="../PreprocessedData/preprocessedForCNN/299x299/targetData_(299, 299, 3)_train.npy"
yvalPath="../PreprocessedData/preprocessedForCNN/299x299/targetData_(299, 299, 3)_val.npy"
xtestPath="../PreprocessedData/preprocessedForCNN/299x299/imgData_(299, 299, 3)_test.npy"
model="Xception"
note="default"
weights="../pretrainedModelWeights/Xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5"
output="../modelOutput/compareModels2"
aug=0
debug=0
python ./trainCNN.py -xtrn "$xtrnPath" -xval "$xvalPath" -ytrn "$ytrnPath" -yval "$yvalPath" -o "$output" -m "$model" -aug "$aug" -xtest "$xtestPath" -d "$debug" -n "$note" -xtest "$xtestPath" -w "$weights"
