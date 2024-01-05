from PIL import Image
import torch
import sys
from tqdm import tqdm
import pickle
sys.path.append("../")
import ImageReward as RM
from utils import *
# set random seed for torch, numpy, pandas and python.random
# init_seed(100)
init_seed(200)
# init_seed(42)

from custom_dataset import *


if __name__ == "__main__":
    model = RM.load_to_train('../checkpoint/ImageReward/ImageReward.pt',
                             download_root="../checkpoint/ImageReward/ImageReward.pt ",
                             device=opts.device,
                             med_config="./config/med_config.json")
    # model = load_model(model, "/data/husky/ImageReward/train_on_AGIQA_3K/checkpoint/1108_model_for_alignment1/best_lr=5e-06.pt")
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
        for batch in pbar:
            loss = torch.zeros(1).requires_grad_(True).float().to(opts.device)
            ids, prompts, img_paths, align_scores, quality_scores = batch
            i = 0
            for i in range(len(prompts)):
                promt = "extremely high quality image, with vivid details"
                img_path = img_paths[i]
                quality_score = quality_scores[i].float().to(opts.device)
                img = Image.open(img_path)
                reward = model(promt, img).float().to(opts.device)
                loss += loss_func(reward, quality_score)
                pbar.set_description(
                    "epoch = {} | training on image {} of this batch | batch loss = {}".format(epoch, i, loss.item()))
            epoch_loss[epoch].append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("epoch loss = ", epoch_loss[epoch])
    save_model(model,"for_quality_seed=100")
    # pickle.dump(epoch_loss, open("/data/husky/ImageReward/train_on_AGIQA_3K/checkpoint/epoch_loss_for_quality_epoch50.pkl", "wb"))

    # test model
