#!/bin/bash
input="..\RawData\OCT2017"
subDirs='all'
outputBase="..\PreprocessedData\preprocessedForCNN\resample_"
nsamples=1000
xRes=224
yRes=224
outputBase2="$outputBase($xRes, $yRes)\subsample_"

source "C:\Anaconda\Scripts\activate" py36
python --version

for ((i=1; i<=10; i++))
do
    echo  "$outputBase2$i"
    python preprocess.py -i "$input" -o "$outputBase2$i" -subdir "all" -n "$nsamples" -Rx $xRes -Ry $yRes
    sleep 5
done