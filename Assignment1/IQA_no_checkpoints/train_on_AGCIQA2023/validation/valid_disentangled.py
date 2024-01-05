import os
import numpy as np
import pandas as pd
import torch
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
from config.options import *

from utils import *

init_seed(42)
from custom_dataset import *

# get json file

def valid_quality(data_path, key):
    # test model
    df_img_reward_trained = pd.read_csv(data_path)

    # get rid of those in train_loader

    id_list = []
    for batch in test_loader:
        ids = batch[0]
        for id in ids:
            id_list.append(id.item())

    # id_list = [i for i in range(10,15)]

    y = []
    y_hat = []
    for idx, item in tqdm(data_set.iterrows(), total=len(data_set)):
        if idx not in id_list:
            continue
        y.append(item['mos_quality'])

    for idx, item in tqdm(df_img_reward_trained.iterrows(), total=len(df_img_reward_trained)):
        if idx not in id_list:
            continue
        y_hat.append(item[key])
    print(key)
    # print(y)
    # print(y_hat)
    PLCC = get_PLLC(y, y_hat)
    SRCC = get_SRCC(y, y_hat)
    print("len of y and y_hat = ", len(y))
    return PLCC, SRCC

# 画一个扇形图
def get_sector_diagram(values=None, features=None, name="sector_diagram"):
    if values is None or features is None:
        print("Values and features cannot be None.")
        return

    # Ensure the number of colors match the number of features
    colors = plt.cm.viridis(np.linspace(0, 1, len(values)))

    # Set the style for the plot
    plt.style.use('seaborn-colorblind')

    # Set explode values to slightly separate the slices if there are more than 1 values
    explode = [0.05] * len(values) if len(values) > 1 else [0]

    # Create the pie chart
    plt.figure(figsize=(8, 6))
    wedges, texts, autotexts = plt.pie(values, explode=explode, labels=None, colors=colors,
                            startangle=140, pctdistance=0.85, autopct='%3.1f%%', labeldistance=1.15)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_weight('bold')
    # Draw a circle at the center of pie to make it look like a donut
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')

    # Set title and legend
    plt.legend(wedges, features, title="Features", loc='upper right', bbox_to_anchor=(1.3, 1.05))

    # Save the figure
    plt.savefig(f"{name}.pdf", bbox_inches='tight')

if __name__ == "__main__":
    # table_list = ["/data/husky/ImageReward/assets/results/AGIQA_3K/AGIQA_3K_ImgRward_disentangled.csv"]
    table_list = ["/data/husky/ImageReward/assets/results/AIGCIQA2023/ImgRward_disentangled.csv"]
    keys = ["quality_score","resolution_score","detail_score", "No_blur_score", "No_noise_score", "color_accuracy_score","contrast_score"]
    dict = dict()
    for table in table_list:
        for key in keys:
            PLCC, SRCC = valid_quality(table, key)
            dict[key] = {"PLCC": PLCC, "SRCC": SRCC}

    # save dict as json file
    import json
    with open("/data/husky/ImageReward/train_on_AGIQA_3K/AGIQA_3K_ImgRward_disentangled.json", "w") as f:
        json.dump(dict, f)

    # PLCC
    PLCC_base_score = dict["quality_score"]["PLCC"]
    PLCC_dict = {}
    for key in keys[1:]:
        PLCC = dict[key]["PLCC"]
        ratio = 1 / (PLCC_base_score - PLCC)
        PLCC_dict[key] = ratio
    # normalize
    sum = 0
    for key in PLCC_dict.keys():
        sum += PLCC_dict[key]
    for key in PLCC_dict.keys():
        PLCC_dict[key] /= sum
    PLCC_list = [PLCC_dict[key] for key in keys[1:]]
    # SRCC
    SRCC_base_score = dict["quality_score"]["SRCC"]
    SRCC_dict = {}
    for key in keys[1:]:
        SRCC = dict[key]["SRCC"]
        ratio = 1 / (SRCC_base_score - SRCC)
        SRCC_dict[key] = ratio
    # normalize
    sum = 0
    for key in SRCC_dict.keys():
        sum += SRCC_dict[key]
    for key in SRCC_dict.keys():
        SRCC_dict[key] /= sum
    SRCC_list = [SRCC_dict[key] for key in keys[1:]]

    # remove underscore in keys[1:]
    feature = [key.replace("_", " ") for key in keys[1:]]


    print(PLCC_list)
    print(SRCC_list)
    get_sector_diagram(PLCC_list, feature, name="PLCC")
    get_sector_diagram(SRCC_list, feature, name="SRCC")