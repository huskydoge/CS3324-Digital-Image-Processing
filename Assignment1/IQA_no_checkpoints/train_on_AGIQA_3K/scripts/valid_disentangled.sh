#!/bin/bash

source /home/husky/anaconda3/bin/activate ImgReward

cd "$(dirname "$0")"/../validation

python ./valid_disentangled.py

