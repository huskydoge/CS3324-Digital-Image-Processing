#!/bin/bash

source /home/husky/anaconda3/bin/activate ImgReward

cd /data/husky/ImageReward/train_on_AGCIQA2023/validation

python valid_disentangled.py

