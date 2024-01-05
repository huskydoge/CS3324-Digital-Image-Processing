#!/bin/bash

source /home/husky/anaconda3/bin/activate ImgReward

cd "$(dirname "$0")"/../validation

filename_prefix=$(date '+%Y%m%d-%H%M')
filename="./results/${filename_prefix}_valid_alignment.txt"

python ./valid_alignment.py >> $filename

echo "End Time: $(date)" >> $filename


filename_prefix=$(date '+%Y%m%d-%H%M')
filename="./results/${filename_prefix}_valid_quality.txt"

python ./valid_quality.py >> $filename

echo "End Time: $(date)" >> $filename