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


    # model = load_model(model, "/data/husky/ImageReward/train_on_AGIQA_3K/checkpoint/1108_model_for_quality2/best_lr=5e-06.pt")
    # train model
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, betas=(opts.adam_beta1, opts.adam_beta2),
                                 eps=opts.adam_eps)
    epoch_loss = dict()
    print(opts.epochs)
    for epoch in range(opts.epochs):
    # for epoch in range(50):
        epoch_loss[epoch] = []
        i = 0
        loss = torch.zeros(1).requires_grad_(True).float().to(opts.device)
        pbar = tqdm(train_loader, total=len(train_loader),
                    desc="epoch = {} | training on image {} of this batch | batch loss = {}".format(epoch, i,
                                                                                                    loss.item()))
        for batch in pbar:
            loss = torch.zeros(1).requires_grad_(True).float().to(opts.device)
            ids, prompts, img_paths, align_scores, quality_scores, authenticity_scores = batch
            i = 0
            for i in range(len(prompts)):
                promt = prompts[i]
                img_path = img_paths[i]
                img = Image.open(img_path)

                align_score = align_scores[i].float().to(opts.device)
                quality_score = quality_scores[i].float().to(opts.device)
                authenticity_score = authenticity_scores[i].float().to(opts.device)

                # reward = model(promt, img, use_transformer = True).float().to(opts.device)
                reward = model(promt, img, use_transformer=True)
                # rewards shape (1,3) , align, quality, authenticity
                loss = loss + loss_func(reward[0], align_score) + loss_func(reward[1], quality_score) + loss_func(reward[2], authenticity_score)
                pbar.set_description(
                    "epoch = {} | training on image {} of this batch | batch loss = {}".format(epoch, i, loss.item()))
            epoch_loss[epoch].append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("epoch loss = ", epoch_loss[epoch])
    save_model(model,subfix = "_for_all_epoch500_iqa")
    pickle.dump(epoch_loss, open("./checkpoint/epoch_loss_for_all_iqa_epoch500.pkl", "wb"))

    # test model
