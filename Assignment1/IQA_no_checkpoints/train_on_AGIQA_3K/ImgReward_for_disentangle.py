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

if __name__ == "__main__":

    model = RM.load(name="../checkpoint/ImageReward/ImageReward.pt")
    model = load_model(model,
                       "./checkpoint/11101836_bsNone_fix=32_lr=0for_quality/best_lr=5e-06.pt")

    # model = load_model(model, "/data/husky/ImageReward/train_on_AGIQA_3K/checkpoint/11101836_bsNone_fix=32_lr=0for_quality/best_lr=5e-06.pt")
    prefix = "../assets/AGIQA_3K"
    data_set = pd.read_csv("../assets/AGIQA-3K.csv")

    results = pd.DataFrame(columns=["name",
                                    "prompt1", "align_score",
                                    "prompt2", "quality_score",
                                    "prompt3", "resolution_score",
                                    "prompt4", "detail_score",
                                    "prompt5", "No_blur_score",
                                    "prompt6", "No_noise_score",
                                    "prompt7", "color_accuracy_score",
                                    "prompt8", "contrast_score",
                                    "prompt9", "bad_score",
                                    "prompt10", "improper_score",
                                    "prompt11", "invalid_score"])
    # forward pass
    cnt = 0

    for idx, item in tqdm(data_set.iterrows(), total=len(data_set)):
        # if idx < 2683:
        #     continue
        path = item["name"]
        prompt1 = item["prompt"]
        prompt2 = "extremely high quality image, with vivid details"
        prompt3 = "high resolution"
        prompt4 = "with vivid details"
        prompt5 = "without blur"
        prompt6 = "without noise"
        prompt7 = "with accurate color"
        prompt8 = "with high contrast"
        prompt9 = "very bad quality"
        prompt10 = "improper content"
        prompt11 = "invalid bad numb noisy blur"

        image = Image.open(os.path.join(prefix, path))
        with torch.no_grad():
            score = model.score(prompt1, image)
            quality_score = model.score(prompt2, image)
            resolution_score = model.score(prompt3, image)
            detail_score = model.score(prompt4, image)
            No_blur_score = model.score(prompt5, image)
            No_noise_score = model.score(prompt6, image)
            color_accuracy_score = model.score(prompt7, image)
            contrast_score = model.score(prompt8, image)
            bad_score = model.score(prompt9, image)
            improper_score = model.score(prompt10, image)
            invalid_score = model.score(prompt11, image)
            results.loc[idx] = [path,
                                prompt1, score,
                                prompt2, quality_score,
                                prompt3, resolution_score,
                                prompt4, detail_score,
                                prompt5, No_blur_score,
                                prompt6, No_noise_score,
                                prompt7, color_accuracy_score,
                                prompt8, contrast_score,
                                prompt9, bad_score,
                                prompt10, improper_score,
                                prompt11, invalid_score
                                ]
            results.to_csv("../assets/results/AGIQA_3K/AGIQA_3K_ImgRward_disentangled_contrast3.csv")
