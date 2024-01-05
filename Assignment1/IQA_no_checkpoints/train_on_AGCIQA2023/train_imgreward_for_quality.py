import pandas as pd
import os
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import random
import numpy as np
import pandas as pd
import torch
import sys
from tqdm import tqdm
import pickle
import json

sys.path.append("../")
import ImageReward as RM

from config.options import *
from utils import *

# set random seed for torch, numpy, pandas and python.random
init_seed(42)
from custom_dataset import *


if __name__ == "__main__":


    model = RM.load_to_train("../checkpoint/ImageReward/ImageReward.pt",
                             download_root="../checkpoint/ImageReward",
                             device=opts.device,
                             med_config="./config/med_config.json")
    # model = load_model(model, "/data/husky/ImageReward/train_on_AGCIQA2023/checkpoint/11110048_bsNone_fix=32_lr=0_for_all_epoch50_iqa/best_lr=5e-06.pt")
    # train model
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, betas=(opts.adam_beta1, opts.adam_beta2),
                                 eps=opts.adam_eps)
    epoch_loss = dict()
    print("total epochs = ", opts.epochs)
    for epoch in range(opts.epochs):
        epoch_loss[epoch] = []
        i = 0
        loss = torch.zeros(1).requires_grad_(True).float().to(opts.device)
        pbar = tqdm(train_loader, total=len(train_loader),
                    desc="epoch = {} | training on image {} of this batch | batch loss = {}".format(epoch, i,
                                                                                                    loss.item()))
        # prompt = "high quality image"  # prompt 1
        # prompt = "extremely high quality image, with vivid details" # prompt 2
        prompt = "extremely high quality image, with high resolution" # prompt 3
        print("prompt: ", prompt)
        for batch in pbar:
            loss = torch.zeros(1).requires_grad_(True).float().to(opts.device)
            ids, prompts, img_paths, align_scores, quality_scores, authenticity_scores = batch
            i = 0
            for i in range(len(prompts)):

                img_path = img_paths[i]
                quality_score = quality_scores[i].float().to(opts.device)
                img = Image.open(img_path)
                reward = model(prompt, img).float().to(opts.device)
                loss += loss_func(reward, quality_score)
                pbar.set_description(
                    "epoch = {} | training on image {} of this batch | batch loss = {}".format(epoch, i, loss.item()))
            epoch_loss[epoch].append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("epoch loss = ", epoch_loss[epoch])
    save_model(model, subfix="_for_quality_prompt3_epoch50")
    pickle.dump(epoch_loss, open("./checkpoint/epoch_loss_quality_epoch50_prompt3.pkl", "wb"))

    # test model
