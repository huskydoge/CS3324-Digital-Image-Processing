'''
@File       :   utils.py
@Time       :   2023/01/14 22:49:00
@Auther     :   Jiazheng Xu
@Contact    :   xjz22@mails.tsinghua.edu.cn
@Description:   Some settings and tools.
'''

#encoding:utf-8
import os, shutil
import torch
from tensorboardX import SummaryWriter
from config.options import *
import torch.distributed as dist
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_curve, auc

""" ==================== Data ======================== """

def collate_fn(batch):
    return batch

def make_path():
    return "{}_bs{}_fix={}_lr={}{}".format(opts.savepath, opts.BatchSize, opts.fix_rate, opts.lr, opts.lr_decay_style)

""" ==================== Models ======================== """

def save_model(model):
    save_path = make_path()
    if not os.path.isdir(os.path.join(config['checkpoint_base'], save_path)):
        os.makedirs(os.path.join(config['checkpoint_base'], save_path), exist_ok=True)
    model_name = os.path.join(config['checkpoint_base'], save_path, 'best_lr={}.pt'.format(opts.lr))
    torch.save(model.state_dict(), model_name)


def load_model(model, ckpt_path = None):
    if ckpt_path is not None:
        model_name = ckpt_path
    else:
        load_path = make_path()
        if not os.path.isdir(os.path.join(config['checkpoint_base'], load_path)):
            os.makedirs(os.path.join(config['checkpoint_base'], load_path), exist_ok=True)
        model_name = os.path.join(config['checkpoint_base'], load_path, 'best_lr={}.pt'.format(opts.lr))
        
    print('load checkpoint from %s'%model_name)
    checkpoint = torch.load(model_name, map_location='cpu') 
    state_dict = checkpoint
    msg = model.load_state_dict(state_dict,strict=False)
    print("missing keys:", msg.missing_keys)

    return model 


def preload_model(model):

    state_dict = torch.load(opts.preload_path, map_location=model.device) 
    msg = model.load_state_dict(state_dict,strict=False)
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
        if opts.clear_visualizer and os.path.exists(filewriter_path):   # 删掉以前的summary，以免重合
            shutil.rmtree(filewriter_path)
        makedir(filewriter_path)
        writer = SummaryWriter(filewriter_path, comment='visualizer')
        return writer


def loss_func(reward, label):
    mse_loss = nn.MSELoss()
    loss = mse_loss(reward, label)
    return loss

## ==================== Others ======================== ##

def roc_auc_score(y_true, y_score):
    """
    y_true: ground truth from human
    y_score: output from model
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return auc(fpr, tpr)


## calculate AUC of model results
def AUC(saliency_mask, gt, thres = 100):
    """
    而如何评价显著性模型预测的好坏，一个主要的Metric就是AUC，
    首先，将Ground truth也就是人标记的显著图的像素二值化，标记为255或0，
    这里面阈值通常是任意选择的（以覆盖图片的 20% 左右）。
    saliency_mask: output from model
    gt: ground truth from human
    """
    saliency_mask = saliency_mask.squeeze()
    gt = gt.squeeze()
    gt[gt > thres] = 255
    gt[gt <= 0] = 0
    gt = gt.astype(np.uint8)
    saliency_mask[saliency_mask > thres] = 255
    saliency_mask[saliency_mask <= 0] = 0
    saliency_mask = saliency_mask.astype(np.uint8)
    return roc_auc_score(gt, saliency_mask)

def sAUC(saliency_mask, gt, thres = 100):
    """
    计算 shuffled AUC (sAUC)
    """
    # 把 saliency_mask 和 gt 向量化并转为浮点型
    saliency_mask = saliency_mask.ravel().astype(np.float32)
    gt = gt.ravel().astype(np.uint8)

    # 二值化
    gt[gt > thres] = 255
    gt[gt <= thres] = 0
    saliency_mask[saliency_mask > thres] = 255
    saliency_mask[saliency_mask <= thres] = 0

    # 获得显著区域 (salient) 和非显著区域 (non-salient/background) 的 indices
    salient_indices = np.where(gt == 255)[0]
    nonsalient_indices = np.where(gt == 0)[0]

    # 随机打乱非显著区域 indices
    np.random.shuffle(nonsalient_indices)

    # 创建一个新的 gt，其中非显著区域的 indices 被打乱
    shuffled_gt = np.zeros_like(gt)
    shuffled_gt[salient_indices] = 255
    shuffled_gt[nonsalient_indices] = 0

    # 计算并返回 sAUC
    return roc_auc_score(shuffled_gt, saliency_mask)

def CC(saliency_mask, gt):
    """
    CC是另一个常用的评价指标，它的全称是Correlation Coefficient，
    """
    saliency_mask = saliency_mask.squeeze()
    gt = gt.squeeze()
    return np.corrcoef(saliency_mask, gt)[0, 1]


def NSS(saliency_mask, gt):
    """
    计算 Normalized Scanpath Saliency (NSS)
    """
    # 把 saliency_mask 和 gt 向量化
    saliency_mask = saliency_mask.ravel()
    gt = gt.ravel()

    # 标准化 saliency_mask 使得均值为0，标准差为1
    saliency_mask = (saliency_mask - np.mean(saliency_mask)) / np.std(saliency_mask)

    # 二值化 gt
    gt[gt > 0] = 1
    gt[gt <= 0] = 0

    # 计算 NSS 分数
    nss_score = np.sum(saliency_mask * gt)

    return nss_score / (saliency_mask.shape[0] * saliency_mask.shape[1])

bce_loss = nn.BCELoss(size_average=True)
def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

	loss0 = bce_loss(d0,labels_v)
	loss1 = bce_loss(d1,labels_v)
	loss2 = bce_loss(d2,labels_v)
	loss3 = bce_loss(d3,labels_v)
	loss4 = bce_loss(d4,labels_v)
	loss5 = bce_loss(d5,labels_v)
	loss6 = bce_loss(d6,labels_v)

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
	print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

	return loss0, loss

