import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt

dirname = os.path.dirname(os.path.abspath(__file__))

# 假设路径和文件已正确设置和存在
# 示例路径和文件名
file_names = ["transalNet_all.csv", "transalNet_non_salient.csv", "transalNet_pure.csv", "transalNet_salient.csv", "transalNet_whole.csv"]
dfs = [pd.read_csv(os.path.join(dirname, "assets", file_name)) for file_name in file_names]

# 指标列表
metrics = ['auc', 'sauc', 'cc', 'nss']

# 创建一个 2x2 的图表布局
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# 数据集名称，用于图例
dataset_names = ["All", "Non-Salient", "Pure", "Salient", "Whole"]

# 绘制每个指标的图表
for i, metric in enumerate(metrics):
    ax = axs[i // 2, i % 2]
    for df, label in zip(dfs, dataset_names):
        ax.plot(df[metric], label=label)
        # 找出每条线的最大值及其索引
        max_value = df[metric].max()
        max_index = df[metric].idxmax()
        # 在图表上标注最大值，去掉箭头
        # ax.text(max_index, max_value, f'{max_value:.2f}', ha='center', va='bottom', fontsize=8)
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
fig.savefig(os.path.join(dirname, 'transalNet_results.pdf'))
