#!/bin/bash
input="D:\Projects\OCT-Image-Classification\PreprocessedData\individual images"
subDirs='all'
outputBase="D:\Projects\OCT-Image-Classification\PreprocessedData\preprocessedForCNN\subsample_"
nsamples=1000
source "C:\Anaconda\Scripts\activate" py36
python --version

for ((i=1; i<=10; i++))
do
    python preprocessForCNN.py -i "$input" -o "$outputBase$i" -subdir "all" -n "$nsamples"
    sleep 5
done