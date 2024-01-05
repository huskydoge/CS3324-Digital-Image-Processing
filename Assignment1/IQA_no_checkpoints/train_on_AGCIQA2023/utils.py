import torch
import numpy as np
import random
import torch.nn as nn
import os
from config.options import *
import time
from scipy import stats
def init_seed(seed = 42):
    # 设置random库的种子
    random_seed = seed
    random.seed(random_seed)

    # 设置numpy库的种子
    np.random.seed(random_seed)

    # 对于pandas，其底层使用的是numpy的随机数生成，因此已经通过numpy设置

    # 设置torch库的种子
    torch.manual_seed(random_seed)

    # 如果使用的是CUDA，则还应该设置CUDA的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)  # 如果使用多个CUDA设备

    # 对于torch的CuDNN后端，如果需要确保结果的确定性，可以设置以下两项
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_PLLC(y, y_hat):
    """
    Calculates the PLCC between two data.
    """
    return stats.pearsonr(y, y_hat)[0]


def get_SRCC(y, y_hat):
    """
    Calculates the SRCC between two data.
    """
    return stats.spearmanr(y, y_hat)[0]


def loss_func(reward, label):
    mse_loss = nn.MSELoss()
    loss = mse_loss(reward, label)
    return loss

def make_path():
    time_stramp = time.strftime("%m%d%H%M", time.localtime())
    return "{}_bs{}_fix={}_lr={}".format(time_stramp, opts.savepath, opts.batch_size, opts.fix_rate, opts.lr)


def save_model(model, subfix=None):
    save_path = make_path() + subfix
    if not os.path.isdir(os.path.join(config['checkpoint_base'], save_path)):
        os.makedirs(os.path.join(config['checkpoint_base'], save_path), exist_ok=True)
    model_name = os.path.join(config['checkpoint_base'], save_path, 'best_lr={}.pt'.format(opts.lr))
    torch.save(model.state_dict(), model_name)

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