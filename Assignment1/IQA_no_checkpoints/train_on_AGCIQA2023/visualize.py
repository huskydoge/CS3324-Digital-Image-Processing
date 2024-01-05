import pickle
from matplotlib import pyplot as plt
import numpy as np
import torch.nn as nn


epoch_loss = pickle.load(open("/data/husky/ImageReward/checkpoint/ImageReward/epoch_loss.pkl", "rb"))
epoch_loss = pickle.load(open("/data/husky/ImageReward/train_on_AGCIQA2023/checkpoint/epoch_loss_all_epoch50.pkl", "rb"))

epoch_list = [np.array(epoch_loss[key]).mean() for key in epoch_loss.keys()]

iqa_epoch_loss = pickle.load(open("/data/husky/ImageReward/train_on_AGCIQA2023/checkpoint/epoch_loss_for_all_iqa_epoch50.pkl", "rb"))
iqa_epoch_list = [np.array(iqa_epoch_loss[key]).mean() for key in iqa_epoch_loss.keys()]

print(epoch_list[:10])
print(iqa_epoch_list[:10])


