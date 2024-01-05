#!/bin/bash
# 第一步：激活环境
source /home/husky/anaconda3/bin/activate ImgReward

# 第二步：改变目录
cd "$(dirname "$0")"/..

# 第三步：运行Python脚本
python validate_model.py