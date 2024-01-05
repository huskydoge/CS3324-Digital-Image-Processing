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
    # model = load_model(model, "/data/husky/ImageReward/train_on_AGCIQA2023/checkpoint/11100543_bsNone_fix=32_lr=0_for_all_epoch50/best_lr=5e-06.pt")
    model = load_model(model, "./checkpoint/11110048_bsNone_fix=32_lr=0_for_all_epoch50_iqa/best_lr=5e-06.pt")
    prefix = "../assets/AIGCIQA2023/Images"

    results = pd.DataFrame(columns=["name", "prompt", "score", "quality_score", "authenticity_score"])
    # forward pass
    cnt = 0
    with torch.no_grad():
        for idx, item in tqdm(data_set.iterrows(), total=len(data_set)):

            path = item["name"]
            prompt = item["prompt"]
            # prompt2 = "extremely high quality image, with vivid details"
            # prompt3 = "very authentic image"
            image = Image.open(os.path.join(prefix, path))
            scores = model.score_forall(prompt, image)
            align_score = scores[0].item()
            quality_score = scores[1].item()
            authenticity_score = scores[2].item()
            results.loc[idx] = [path, prompt, align_score,  quality_score,  authenticity_score]
            results.to_csv("../assets/results/AIGCIQA2023/ImgRward_trained_for_all_epoch50_iqa.csv")
