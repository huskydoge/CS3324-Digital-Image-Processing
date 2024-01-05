import os
import numpy as np
import pandas as pd
import torch
import sys
from tqdm import tqdm


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.options import *

from utils import *
init_seed(42)

from custom_dataset import *

# get json file

def valid_alignment(data_path):
    # test model
    df_img_reward_trained = pd.read_csv(data_path)

    # get rid of those in train_loader

    id_list = []
    for batch in test_loader:
        ids = batch[0]
        for id in ids:
            id_list.append(id.item())


    y = []
    y_hat = []
    for idx, item in tqdm(data_set.iterrows(), total=len(data_set)):
        if idx not in id_list:
            continue
        y.append(item['mos_align'])

    for idx, item in tqdm(df_img_reward_trained.iterrows(), total=len(df_img_reward_trained)):
        if idx not in id_list:
            continue
        y_hat.append(item['score'])

    # print(y)
    # print(y_hat)
    print("len of y and y_hat = ", len(y))
    print("PLCC = ", get_PLLC(y, y_hat))
    print("SRCC = ", get_SRCC(y, y_hat))


if __name__ == "__main__":
    table_list = ["/data/husky/ImageReward/assets/results/AIGCIQA2023/ImgRward_trained_for_authenticity.csv",
                  "/data/husky/ImageReward/assets/results/AIGCIQA2023/ImgRward_raw.csv",
                  "/data/husky/ImageReward/assets/results/AIGCIQA2023/ImgRward_trained_for_all_epoch10_iqa.csv",
                  "/data/husky/ImageReward/assets/results/AIGCIQA2023/ImgRward_trained_for_quality_epoch10.csv",
                  "/data/husky/ImageReward/assets/results/AIGCIQA2023/ImgRward_trained_for_quality_epoch50.csv",
                  "/data/husky/ImageReward/assets/results/AIGCIQA2023/ImgRward_trained_for_authenticity_epoch50.csv",
                  "/data/husky/ImageReward/assets/results/AIGCIQA2023/ImgRward_trained_for_alll_epoch50.csv",
                  "/data/husky/ImageReward/assets/results/AIGCIQA2023/ImgRward_trained_for_align_epoch50.csv",
                  "/data/husky/ImageReward/assets/results/AIGCIQA2023/ImgRward_trained_for_alignment.csv",
                  "/data/husky/ImageReward/assets/results/AIGCIQA2023/ImgRward_trained_for_all_epoch50_iqa.csv",
                  "/data/husky/ImageReward/assets/results/AIGCIQA2023/ImgRward_trained_for_align_from_trained_quality_epoch50.csv",
                  "/data/husky/ImageReward/assets/results/AIGCIQA2023/ImgRward_trained_for_quality_from_trained_align_epoch50.csv"
                  ]

    for table in table_list:
        print("=====================================")
        print(table)
        valid_alignment(table)
