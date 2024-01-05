import torch
import sys
sys.path.append('../')
import ImageReward as RM
from utils import *
import os.path
from PIL import Image
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    model = RM.load(name="../checkpoint/ImageReward/ImageReward.pt")
    # model = load_model(model,
    #                    "/data/husky/ImageReward/train_on_AGIQA_3K/checkpoint/11092209_bsNone_fix=32_lr=0for_align/best_lr=5e-06.pt")

    # model = load_model(model, "./checkpoint/11121320_bsNone_fix=32_lr=0for_quality_seed=100/best_lr=5e-06.pt")
    prefix = "../assets/AGIQA_3K"
    data_set = pd.read_csv("../assets/AGIQA-3K.csv")

    # modify filename for saving here
    filename = "AGIQA_3K_ImgRward_quality_seed=300"

    results = pd.DataFrame(columns=["name", "prompt1", "score", "prompt2", "quality_score"])
    cnt = 0
    for idx, item in tqdm(data_set.iterrows(), total=len(data_set)):
        path = item["name"]
        prompt1 = item["prompt"]
        prompt2 = "extremely high quality image, with vivid details"
        image = Image.open(os.path.join(prefix, path))
        with torch.no_grad():
            score = model.score(prompt1, image)
            quality_score = model.score(prompt2, image)
        results.loc[idx] = [path, prompt1, score, prompt2, quality_score]
        results.to_csv("../assets/results/AGIQA_3K/{}.csv".format(filename))
