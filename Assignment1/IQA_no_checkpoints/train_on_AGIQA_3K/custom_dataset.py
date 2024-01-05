from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import torch
from utils import *
from config.options import *
dir = os.path.dirname(os.path.abspath(__file__))
data_set = pd.read_csv(os.path.join(dir, "../assets/AGIQA-3K.csv"))
# init_seed(100)
# init_seed(200)
# init_seed(42)
class ImageRewardDataSet(Dataset):
    def __init__(self, img_table, img_file_prefix=None, transforms=None):
        self.img_table = img_table
        self.transforms = transforms
        self.prefix = img_file_prefix

    def __getitem__(self, idx):
        row = self.img_table.iloc[idx]
        prompt = row["prompt"]
        align_score = row["mos_align"]
        quality_score = row["mos_quality"]
        name = row["name"]
        img_path = os.path.join(self.prefix, name)

        return idx, prompt, img_path, align_score, quality_score

        # return prompt, img_path, align_score, quality_score

    def __len__(self):
        return len(self.img_table)


# prepare dataset
prefix = os.path.join(dir,"../assets/AGIQA_3K")
data_df = pd.read_csv(os.path.join(dir, "../assets/AGIQA-3K.csv"))


# split data_df to 80% train, 20% test, however, we should make sure those with same prompt should be in one group
# so we first get all prompts
prompts = data_df["prompt"].unique()
# then shuffle prompts
np.random.shuffle(prompts)
# then split prompts
train_prompts = prompts[:int(len(prompts) * 0.8)]
test_prompts = prompts[int(len(prompts) * 0.8):]

# get train_df and test_df
train_df = data_df[data_df["prompt"].isin(train_prompts)]
test_df = data_df[data_df["prompt"].isin(test_prompts)]

# build dataset, use train_df by torch.data.Dataset
train_dataset = ImageRewardDataSet(train_df, img_file_prefix=prefix)
test_dataset = ImageRewardDataSet(test_df, img_file_prefix=prefix)

# build dataloader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=True)
