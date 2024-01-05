import os
import torch
import sys
sys.path.append('../')
import ImageReward as RM
from utils import *
import os.path
import requests
from PIL import Image
import pandas as pd
from tqdm import tqdm
from custom_dataset import *

if __name__ == "__main__":
    model = RM.load(name="../checkpoint/ImageReward/ImageReward.pt")
    # model = load_model(model,
    #                    "/data/husky/ImageReward/train_on_AGCIQA2023/checkpoint/11100607_bsNone_fix=32_lr=0_for_quality_epoch50/best_lr=5e-06.pt")
    # model = load_model(model, "/data/husky/ImageReward/train_on_AGCIQA2023/checkpoint/11100604_bsNone_fix=32_lr=0_for_authenticity_epoch50/best_lr=5e-06.pt")
    # model = load_model(model, "/data/husky/ImageReward/train_on_AGCIQA2023/checkpoint/11100603_bsNone_fix=32_lr=0_for_align_epoch50/best_lr=5e-06.pt")
    model = load_model(model, "./checkpoint/11110735_bsNone_fix=32_lr=0_for_align_from_trained_quality_epoch50/best_lr=5e-06.pt")
    prefix = "../assets/AIGCIQA2023/Images"

    results = pd.DataFrame(columns=["name", "prompt1", "score", "prompt2", "quality_score","prompt3", "authenticity_score"])
    # forward pass
    cnt = 0
    with torch.no_grad():
        for idx, item in tqdm(data_set.iterrows(), total=len(data_set)):

            path = item["name"]
            prompt1 = item["prompt"]
            prompt2 = "extremely high quality image, with vivid details"
            # prompt2 = "high quality image"
            # prompt2 = "extremely high quality image, with high resolution"
            prompt3 = "very authentic image"
            image = Image.open(os.path.join(prefix, path))
            score = model.score(prompt1, image)
            quality_score = model.score(prompt2, image)
            authenticity_score = model.score(prompt3, image)
            results.loc[idx] = [path, prompt1, score, prompt2, quality_score, prompt3, authenticity_score]
            save_path = "../assets/results/AIGCIQA2023/ImgRward_trained_for_align_from_trained_quality_epoch500.csv"
            # print(save_path)
            results.to_csv(save_path)
