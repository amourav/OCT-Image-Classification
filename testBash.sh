#!/bin/bash


source "C:\Anaconda\Scripts\activate" py36
python --version

base='loop_'
for ((i=1; i<=5; i++))
do
    python test.py "$base$i" $i
    sleep 5
done