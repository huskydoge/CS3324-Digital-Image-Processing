import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

filename = ["fse=1.0_fsd=1.0.csv", "fse=0.5_fsd=0.5.csv", "fse=0.0_fsd=1.0.csv","fse=0.0_fsd=0.0.csv"]

dirname = "/data/husky/ImageReward/train_on_saliency/visualization_for_paper/assets"

dfs = [pd.read_csv(os.path.join(dirname, file)) for file in filename]

metrics = ['auc', 'sauc', 'cc', 'nss']

# 创建一个 2x2 的图表布局
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# 数据集名称，用于图例
dataset_names = ["fse=1.0_fsd=1.0", "fse=0.5_fsd=0.5", "fse=0.0_fsd=1.0","fse=0.0_fsd=0.0"]

# 绘制每个指标的图表

for i, metric in enumerate(metrics):
    ax = axs[i // 2, i % 2]
    for df, label in zip(dfs, dataset_names):
        # 用颜色区分明显的线
        if label == "fse=1.0_fsd=1.0":
            ax.plot(df[metric], label=label, color='red')
        elif label == "fse=0.5_fsd=0.5":
            ax.plot(df[metric], label=label, color='blue')
        elif label == "fse=0.0_fsd=1.0":
            ax.plot(df[metric], label=label, color = 'grey')
        elif label == "fse=0.0_fsd=0.0":
            ax.plot(df[metric], label=label, color='green')
        else:
            print(label)
            raise ValueError("Unknown label")

        # 找出每条线的最大值及其索引
        max_value = df[metric].max()
        max_index = df[metric].idxmax()
        # 在图表上标注最大值，去掉箭头
        ax.text(max_index, max_value, f'{max_value:.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_title(metric.upper())
    ax.set_xlabel('Epochs')
    ax.set_ylabel(metric)

# 添加图例
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=len(dataset_names))

# 调整布局
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# 显示图表
plt.show()

# 保存为PDF文件
fig.savefig(os.path.join(dirname, '..', 'fixrate_ablation', 'results.pdf'))
