from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import torch
from utils import *
from config.options import *
import random
dir = os.path.dirname(os.path.abspath(__file__))
RANDOM_STATE = 42  # 定义一个随机状态
# 设置全局种子
seed = RANDOM_STATE
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

# 如果使用GPU，还需要设置GPU种子
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class PureDataSet(Dataset):
    def __init__(self, img_table, img_file_prefix=None, map_prefix=None, fix_map_prefix = None, seed=None, transforms=None):
        self.img_table = img_table
        self.transforms = transforms
        self.prefix = img_file_prefix
        self.map_prefix = map_prefix
        self.fix_map_prefix = fix_map_prefix
        self.seed = seed

    def __getitem__(self, idx):
        row = self.img_table.iloc[idx]
        name = str(row["image_number"]) + "_0" + ".png"
        text = ""
        img_path = os.path.join(self.prefix, name)
        map_path = os.path.join(self.map_prefix, name)
        fix_map_path = os.path.join(self.fix_map_prefix, name)
        return img_path, map_path, fix_map_path, text
        # return prompt, img_path, align_score, quality_score

    def __len__(self):
        return len(self.img_table)


class Img_Text_DataSet(Dataset):
    def __init__(self, img_table, img_file_prefix=None, map_prefix=None, fix_map_prefix = None, data_type="all"):
        self.img_table = img_table
        self.prefix = img_file_prefix
        self.map_prefix = map_prefix
        self.fix_map_prefix = fix_map_prefix
        self.data_type = data_type
        self.type = {
            "whole": "",
            "all": "_1",
            "non_salient": "_2",
            "salient": "_3"

        }

    def __getitem__(self, idx):
        row = self.img_table.iloc[idx]
        name = (row["image_number"]) + self.type[self.data_type] + ".png"
        text = row["text"]
        img_path = os.path.join(self.prefix, name)
        map_path = os.path.join(self.map_prefix, name)
        fix_map_path = os.path.join(self.fix_map_prefix, name)

        return img_path, map_path, fix_map_path, text
        # return prompt, img_path, align_score, quality_score

    def __len__(self):
        return len(self.img_table)



img_prefix = os.path.join(dir, "../assets/saliency/image")
map_prefix = os.path.join(dir, "../assets/saliency/map")
fix_map_prefix = os.path.join(dir, "../assets/saliency/fixation")

pure_df_path = os.path.join(dir, "../assets/saliency/pure.csv")
all_df_path = os.path.join(dir, "../assets/saliency/all.csv")
non_salient_df_path = os.path.join(dir, "../assets/saliency/non_salient.csv")
salient_df_path = os.path.join(dir, "../assets/saliency/salient.csv")
whole_df_path = os.path.join(dir, "../assets/saliency/whole.csv")


def split_train_test(df, train_frac=0.8, seed=RANDOM_STATE):
    np.random.seed(seed)  # 设置随机种子以确保结果可复现
    shuffled_indices = np.random.permutation(len(df))  # 随机打乱索引
    train_set_size = int(len(df) * train_frac)  # 计算训练集大小
    train_indices = shuffled_indices[:train_set_size]
    test_indices = shuffled_indices[train_set_size:]
    return df.iloc[train_indices], df.iloc[test_indices]

# 加载数据
pure_df = pd.read_csv(pure_df_path, dtype=object)
whole_df = pd.read_csv(whole_df_path, dtype=object)
all_df = pd.read_csv(all_df_path, dtype=object)
non_salient_df = pd.read_csv(non_salient_df_path, dtype=object)
salient_df = pd.read_csv(salient_df_path, dtype=object)

# 分割数据集
pure_train_df, pure_test_df = split_train_test(pure_df)
whole_train_df, whole_test_df = split_train_test(whole_df)
all_train_df, all_test_df = split_train_test(all_df)
non_salient_train_df, non_salient_test_df = split_train_test(non_salient_df)
salient_train_df, salient_test_df = split_train_test(salient_df)

class SaliencyDataSet:
    def __init__(self):
        self.pure_type_train_dataset = PureDataSet(pure_train_df, img_prefix, map_prefix, fix_map_prefix)
        self.pure_type_test_dataset = PureDataSet(pure_test_df, img_prefix, map_prefix, fix_map_prefix)
        self.all_type_train_dataset = Img_Text_DataSet(all_train_df, img_prefix, map_prefix, fix_map_prefix, data_type="all")
        self.all_type_test_dataset = Img_Text_DataSet(all_test_df, img_prefix, map_prefix,fix_map_prefix, data_type="all")
        self.non_salient_type_train_dataset = Img_Text_DataSet(non_salient_train_df, img_prefix, map_prefix,fix_map_prefix, data_type="non_salient")
        self.non_salient_type_test_dataset = Img_Text_DataSet(non_salient_test_df, img_prefix, map_prefix, fix_map_prefix, data_type="non_salient")
        self.salient_type_train_dataset = Img_Text_DataSet(salient_train_df, img_prefix, map_prefix, fix_map_prefix, data_type="salient")
        self.salient_type_test_dataset = Img_Text_DataSet(salient_test_df, img_prefix, map_prefix, fix_map_prefix, data_type="salient")
        self.whole_type_train_dataset = Img_Text_DataSet(whole_train_df, img_prefix, map_prefix, fix_map_prefix, data_type="whole")
        self.whole_type_test_dataset = Img_Text_DataSet(whole_test_df, img_prefix, map_prefix, fix_map_prefix, data_type="whole")

    def get_oneforall_dataloader_for_training_and_testing(self):
        # 合并所有数据集
        all_datasets_train = torch.utils.data.ConcatDataset([
            self.pure_type_train_dataset,
            self.all_type_train_dataset,
            self.non_salient_type_train_dataset,
            self.salient_type_train_dataset,
            self.whole_type_train_dataset
        ])

        all_datasets_test = torch.utils.data.ConcatDataset([
            self.pure_type_test_dataset,
            self.all_type_test_dataset,
            self.non_salient_type_test_dataset,
            self.salient_type_test_dataset,
            self.whole_type_test_dataset
        ])

        # 创建训练和测试数据加载器
        dataloader_train = torch.utils.data.DataLoader(all_datasets_train, batch_size=opts.batch_size, shuffle=True)
        dataloader_test = torch.utils.data.DataLoader(all_datasets_test, batch_size=opts.batch_size, shuffle=True)

        return dataloader_train, dataloader_test

    def get_loader(self, type="pure"):
        if type == "pure":
            return torch.utils.data.DataLoader(self.pure_type_train_dataset, batch_size=opts.batch_size,
                                               shuffle=True), torch.utils.data.DataLoader(self.pure_type_test_dataset,
                                                                                          batch_size=opts.batch_size,
                                                                                          shuffle=True)
        elif type == "all":
            return torch.utils.data.DataLoader(self.all_type_train_dataset, batch_size=opts.batch_size,
                                               shuffle=True), torch.utils.data.DataLoader(self.all_type_test_dataset,
                                                                                          batch_size=opts.batch_size,
                                                                                          shuffle=True)
        elif type == "non_salient":
            return torch.utils.data.DataLoader(self.non_salient_type_train_dataset, batch_size=opts.batch_size,
                                               shuffle=True), torch.utils.data.DataLoader(
                self.non_salient_type_test_dataset, batch_size=opts.batch_size, shuffle=True)
        elif type == "salient":
            return torch.utils.data.DataLoader(self.salient_type_train_dataset, batch_size=opts.batch_size,
                                               shuffle=True), torch.utils.data.DataLoader(
                self.salient_type_test_dataset, batch_size=opts.batch_size, shuffle=True)
        elif type == "whole":
            return torch.utils.data.DataLoader(self.whole_type_train_dataset, batch_size=opts.batch_size,
                                               shuffle=True), torch.utils.data.DataLoader(
                self.whole_type_test_dataset, batch_size=opts.batch_size, shuffle=True)
        elif type == "oneforall":
            return self.get_oneforall_dataloader_for_training_and_testing()
        else:
            raise Exception("type not found")


if __name__ == "__main__":
    print(pure_df)
