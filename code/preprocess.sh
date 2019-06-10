#!/usr/bin/env bash
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16000M        # memory per node
#SBATCH --time=0-00:20      # time (DD-HH:MM)
#SBATCH --output=../logs/%x-%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=def-amartel

module load cuda cudnn 
source activate
python --version

inPath="../RawData/"
dir='all'
nTrn=1000
debug=0



outPath="../PreprocessedData/preprocessedForCNN/224 x 224 n"
RX=224
RY=224

python ./preprocess.py -i "$inPath" -o "$outPath" -subdir "$dir" -n "$nTrn" -Rx "$RX" -Ry "$RY" -d "$debug"



outPath="../PreprocessedData/preprocessedForCNN/299 x 299 n"
RX=299
RY=299

python ./preprocess.py -i "$inPath" -o "$outPath" -subdir "$dir" -n "$nTrn" -Rx "$RX" -Ry "$RY" -d "$debug"

