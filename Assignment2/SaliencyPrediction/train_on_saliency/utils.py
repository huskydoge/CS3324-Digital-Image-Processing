'''
@File       :   utils.py
@Time       :   2023/01/14 22:49:00
@Auther     :   Jiazheng Xu
@Contact    :   xjz22@mails.tsinghua.edu.cn
@Description:   Some settings and tools.
'''

# encoding:utf-8
import os, shutil
import torch
from tensorboardX import SummaryWriter
from config.options import *
import torch.distributed as dist
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_curve, auc
from PIL import Image
import torch
import random
from torchvision.transforms import transforms

transform = transforms.ToTensor()
import numpy as np
import time

""" ==================== Data ======================== """
date = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


def collate_fn(batch):
    return batch


def make_path():
    return (
        f"{date}_epochs={opts.epochs}_bs{opts.batch_size}_loss={opts.loss}"
        f"_fixrate={opts.fix_rate}_reshape={opts.reshape}_lr={opts.lr}"
        f"_{opts.lr_decay_style}_{opts.task}"
        f"_ccw={opts.cc_w}_simw={opts.sim_w}_kldivw={opts.kldiv_w}_nssw={opts.nss_w}_msew={opts.mse_w}_modelname={opts.model_name}"
        f"_usecross={opts.use_cross}"
        f"_fse={opts.fse}_fsd={opts.fsd}"
    )

""" ==================== Models ======================== """


def save_model(model, subfix="subfix"):
    save_path = make_path()
    if not os.path.isdir(os.path.join(config['checkpoint_base'], save_path)):
        os.makedirs(os.path.join(config['checkpoint_base'], save_path), exist_ok=True)
    model_name = os.path.join(config['checkpoint_base'], save_path, 'best_lr={}_loss={}.pt'.format(opts.lr, subfix))
    torch.save(model.state_dict(), model_name)


def save_map_to_png(map, name):
    # Convert the tensor to a NumPy array if it's not already
    save_path = make_path()
    if not os.path.isdir(os.path.join(config['pics_base'], save_path)):
        os.makedirs(os.path.join(config['pics_base'], save_path), exist_ok=True)
    if not isinstance(map, np.ndarray):
        map = map.detach().cpu().numpy()

    path = os.path.join()

    map = np.squeeze(map)

    # Normalize the array to be in the range [0, 255] if it's not already
    # This step is optional and depends on the data range in your tensor
    map = (255 * (map - np.min(map)) / np.ptp(map)).astype(np.uint8)

    # Create an image from the array
    image = Image.fromarray(map)

    # Save the image
    image.save(name, format='PNG')

def load_model(model, ckpt_path=None):
    if ckpt_path is not None:
        model_name = ckpt_path
    else:
        load_path = make_path()
        if not os.path.isdir(os.path.join(config['checkpoint_base'], load_path)):
            os.makedirs(os.path.join(config['checkpoint_base'], load_path), exist_ok=True)
        model_name = os.path.join(config['checkpoint_base'], load_path, 'best_lr={}.pt'.format(opts.lr))

    print('load checkpoint from %s' % model_name)
    checkpoint = torch.load(model_name, map_location='cpu')
    state_dict = checkpoint
    msg = model.load_state_dict(state_dict, strict=False)
    print("missing keys:", msg.missing_keys)

    return model


def preload_model(model):
    state_dict = torch.load(opts.preload_path, map_location=model.device)
    msg = model.load_state_dict(state_dict, strict=False)
    print("missing keys:", msg.missing_keys)

    return model


""" ==================== Tools ======================== """


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path, 0o777)


def visualizer():
    if get_rank() == 0:
        # filewriter_path = config['visual_base']+opts.savepath+'/'
        save_path = make_path()
        filewriter_path = os.path.join(config['visual_base'], save_path)
        if opts.clear_visualizer and os.path.exists(filewriter_path):  # 删掉以前的summary，以免重合
            shutil.rmtree(filewriter_path)
        makedir(filewriter_path)
        writer = SummaryWriter(filewriter_path, comment='visualizer')
        return writer


def loss_func(reward, label):
    mse_loss = nn.MSELoss()
    loss = mse_loss(reward, label)
    return loss


## ==================== Others ======================== ##

import torch
import torchmetrics


def cross_entropy_loss(d, map):
    # flatten all
    d = d.view(-1)
    map = map.view(-1)

    # incase d is all 0, log will produce nan
    d = torch.clamp(d, min=1e-8, max=1 - 1e-8)

    loss = torch.mean(-map * torch.log(d) - (1 - map) * torch.log(1 - d))
    return 100 * loss

from sklearn.metrics import roc_auc_score

def AUC(saliency_mask, gt):
    saliency_mask = saliency_mask.view(-1).detach().cpu()
    gt = gt.view(-1).detach().cpu()
    thres_gt = gt.sort()[0][int(0.8*len(gt))]
    thres_sal = saliency_mask.sort()[0][int(0.8*len(saliency_mask))]
    gt = (gt > thres_gt)
    # print("auc")
    # print(gt.sum())
    saliency_mask = (saliency_mask > thres_sal).float()
    return roc_auc_score(gt.numpy(), saliency_mask.numpy())


def sAUC(saliency_mask, gt):
    saliency_mask = saliency_mask.view(-1).detach().cpu()
    gt = gt.view(-1).detach().cpu()
    thres_gt = gt.sort()[0][int(0.8 * len(gt))]
    thres_sal = saliency_mask.sort()[0][int(0.8 * len(saliency_mask))]
    gt = (gt > thres_gt).numpy()
    saliency_mask = (saliency_mask > thres_sal).float().numpy()
    # print("sauc")
    # print(gt.sum())
    # print(saliency_mask.sum())
    # 获取正样本和负样本的索引
    positive_indices = np.where(gt > 0)[0]
    negative_indices = np.where(gt == 0)[0]

    # 打乱负样本索引
    np.random.shuffle(negative_indices)

    # 取与正样本相同数量的负样本
    shuffled_indices = np.concatenate([positive_indices, negative_indices[:len(positive_indices)]])

    # 创建新的打乱后的标签
    shuffled_gt = np.zeros_like(gt)
    shuffled_gt[shuffled_indices] = 1

    return roc_auc_score(shuffled_gt, saliency_mask)
def CC(saliency_mask, gt):
    saliency_mask = saliency_mask.view(-1)
    gt = gt.view(-1)
    combined = torch.stack((saliency_mask, gt))
    # print("CC: ", torch.corrcoef(combined)[0, 1])
    return torch.corrcoef(combined)[0, 1]


import torch

def NSS(saliency_mask, gt):
    """
    计算 Normalized Scanpath Saliency (NSS)
    """
    # 将saliency_mask和gt展平
    saliency_mask = saliency_mask.view(-1)
    gt = gt.view(-1)

    # 标准化saliency_mask
    saliency_mask = (saliency_mask - saliency_mask.mean()) / saliency_mask.std()

    # 将gt转换为二进制掩码
    gt = (gt > 0)

    # 计算NSS得分
    nss_score = torch.sum(saliency_mask * gt) / torch.sum(gt)
    # print("NSS: ", nss_score)

    return nss_score



bce_loss = nn.BCELoss(size_average=True)

def mse_loss(d, map):
    # flatten all
    d = d.view(-1)
    print(d.max())
    print(d.min())
    map = map.view(-1)
    print(map.max())
    print(map.min())
    loss = torch.mean((d - map) ** 2)
    print(loss.requires_grad)
    return loss

def mse_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    # flatten all

    loss0 = mse_loss(d0, labels_v)
    loss1 = mse_loss(d1, labels_v)
    loss2 = mse_loss(d2, labels_v)
    loss3 = mse_loss(d3, labels_v)
    loss4 = mse_loss(d4, labels_v)
    loss5 = mse_loss(d5, labels_v)
    loss6 = mse_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
    #     loss0.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(), loss4.data.item(),
    #     loss5.data.item(),
    #     loss6.data.item()))

    return loss0, loss

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    # flatten all

    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
    #     loss0.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(), loss4.data.item(),
    #     loss5.data.item(),
    #     loss6.data.item()))

    return loss0, loss


def multi_metric_loss(d, map, fix_map):
    w_auc = 1.0
    w_sauc = 1.0
    w_cc = 10
    w_nss = 10
    loss = w_auc * AUC(d, map) + w_sauc * sAUC(d, map) + w_cc * CC(d, map) + w_nss * NSS(d, fix_map)
    return - loss


def multi_metric_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    # flatten all

    loss0 = multi_metric_loss(d0, labels_v)
    loss1 = multi_metric_loss(d1, labels_v)
    loss2 = multi_metric_loss(d2, labels_v)
    loss3 = multi_metric_loss(d3, labels_v)
    loss4 = multi_metric_loss(d4, labels_v)
    loss5 = multi_metric_loss(d5, labels_v)
    loss6 = multi_metric_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
    #     loss0.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(), loss4.data.item(),
    #     loss5.data.item(),
    #     loss6.data.item()))

    return loss0, loss

def total_variation_loss(img):
    """
    计算图像的总变差损失以鼓励图像平滑。
    """
    batch_size, _, height, width = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return (tv_h + tv_w) / (batch_size * height * width)

def transfer_map_to_tensor(map_img):
    tensor_image = transform(map_img)
    tensor_image = tensor_image.unsqueeze(0)
    return tensor_image


from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def transform_for_sal(image, device):
    # 定义转换流程
    transform_pipeline = transforms.Compose([
        transforms.Resize((288, 384)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 应用转换
    image = transform_pipeline(image).to(device)
    return image

def transform_ori(image):
    transform_pipeline = transforms.Compose([
        transforms.Resize((288, 384)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform_pipeline(image)
    return image

# def transform_map(n_px):
#     return Compose([
#         Resize(n_px, interpolation=BICUBIC),
#         CenterCrop(n_px),
#         ToTensor(),
#         Normalize(0, 1),
#     ])

def transform_map():
    return Compose([
        ToTensor(),
        Normalize(0, 1),
    ])

def save_map_to_png(map, path):
    # Convert the tensor to a NumPy array if it's not already
    if not isinstance(map, np.ndarray):
        map = map.detach().cpu().numpy()

    map = np.squeeze(map)

    # Normalize the array to be in the range [0, 255] if it's not already
    # This step is optional and depends on the data range in your tensor
    map = (255 * (map - np.min(map)) / np.ptp(map)).astype(np.uint8)

    # Create an image from the array
    image = Image.fromarray(map)

    # Save the image
    image.save(path, format='PNG')
