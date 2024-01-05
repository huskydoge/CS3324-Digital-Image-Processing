import os
import pandas as pd
import sys

# add parent path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm

from utils import *
# init_seed(100)
init_seed(200)
# init_seed(42)
from config.options import *
from custom_dataset import *




def valid_alignment(data_path):
    # test model
    df_img_reward_trained = pd.read_csv(data_path)

    # get rid of those in train_loader

    id_list = []
    for batch in test_loader:
        ids = batch[0]
        for id in ids:
            id_list.append(id.item())

    # id_list = [i for i in range(10, 15)]

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
    table_list = ["/data/husky/ImageReward/assets/results/AGIQA_3K/AGIQA_3K_ImgRward_align_seed=100.csv",
                  "/data/husky/ImageReward/assets/results/AGIQA_3K/AGIQA_3K_ImgRward_align_seed=200.csv",
                  "/data/husky/ImageReward/assets/results/AGIQA_3K/AGIQA_3K_ImgRward_quality_seed=100.csv",
                  "/data/husky/ImageReward/assets/results/AGIQA_3K/AGIQA_3K_ImgRward_quality_seed=200.csv"]

    for table in table_list:
        print("=====================================")
        print(table)
        valid_alignment(table)
