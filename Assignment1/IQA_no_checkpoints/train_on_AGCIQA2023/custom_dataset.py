from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import torch
import random

import sys
# add this file's parent directory to path
# sys.path.append("/data/husky/ImageReward/train_on_AGCIQA2023")
from config.options import *
from utils import *

# init_seed(42)

dir = os.path.dirname(os.path.abspath(__file__))

class ImageRewardDataSet(Dataset):
    def __init__(self, img_table, img_file_prefix="/data/husky/ImageReward/assets/AIGCIQA2023/Images", transforms=None):
        self.img_table = img_table
        self.transforms = transforms
        self.prefix = img_file_prefix

    def __getitem__(self, idx):
        row = self.img_table.iloc[idx]
        prompt = row["prompt"]
        align_score = row["mos_align"]
        quality_score = row["mos_quality"]
        authenticity_score = row["mos_authenticity"]
        name = row["name"]
        img_path = os.path.join(self.prefix, name)

        return idx, prompt, img_path, align_score, quality_score, authenticity_score

        # return prompt, img_path, align_score, quality_score

    def __len__(self):
        return len(self.img_table)


# prepare dataset
prefix = os.path.join(dir,"../assets/AIGCIQA2023/Images")
data_df = pd.read_csv(os.path.join(dir, "../assets/AIGCIQA2023/AIGCIQA2023.csv"))
data_set = data_df
# for those columns with mos value, we should normalize them to [0,5], in order to align with AGIQA_3K
mos_cols = ["mos_align", "mos_quality", "mos_authenticity"]
for col in mos_cols:
    data_df[col] = data_df[col].apply(lambda x: (x - data_df[col].min()) / (data_df[col].max() - data_df[col].min()) * 5)

# split data_df to 80% train, 20% test, however, we should make sure those with same prompt should be in one group
# so we first get all prompts
prompts = data_df["prompt"].unique()
# then shuffle prompts
np.random.shuffle(prompts)
# then split prompts
train_prompts = prompts[:int(len(prompts) * 0.8)]
test_prompts = prompts[int(len(prompts) * 0.8):]

# then we get train_df and test_df
train_df = data_df[data_df["prompt"].isin(train_prompts)]
test_df = data_df[data_df["prompt"].isin(test_prompts)]


# build dataset, use train_df by torch.data.Dataset
train_dataset = ImageRewardDataSet(train_df, img_file_prefix=prefix)
test_dataset = ImageRewardDataSet(test_df, img_file_prefix=prefix)

# build dataloader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=True)
