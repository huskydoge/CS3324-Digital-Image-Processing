#!/bin/bash

source /home/husky/anaconda3/bin/activate ImgReward

cd /data/husky/ImageReward/train_on_AGCIQA2023/validation

filename_prefix=$(date '+%Y%m%d-%H%M')
filename="/data/husky/ImageReward/train_on_AGCIQA2023/validation/results/${filename_prefix}_valid_quality.txt"

python /data/husky/ImageReward/train_on_AGCIQA2023/validation/valid_quality.py >> $filename

echo "End Time: $(date)" >> $filename
