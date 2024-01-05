#!/bin/bash

source /home/husky/anaconda3/bin/activate ImgReward

cd "$(dirname "$0")"/../validation

filename_prefix=$(date '+%Y%m%d-%H%M')
filename="./results/${filename_prefix}_valid_authenticity.txt"

python ./valid_authenticity.py >> $filename

echo "End Time: $(date)" >> $filename
