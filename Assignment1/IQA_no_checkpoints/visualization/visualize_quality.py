import matplotlib.pyplot as plt
import numpy as np

# 数据字典
data_dict = {
    "raw-3K": [0.685224719,0.685228547],
    "trained-3K": [0.969412492, 0.932892069],
    "best-3K": [0.8903, 0.8426],
    "raw-2023": [0.217584971281003, 0.215309636760576],
    "trained-2023": [0.88552886869668, 0.897613921935424],
    "best-2023": [0.8402, 0.7961]
}


def plot_bar(data_dict):
    # 设置bar的位置
    bar_width = 0.35
    index = np.arange(len(data_dict))
    plt.figure(figsize=(10, 6))

    # 创建两个列表，一个用于 PLCC，一个用于 SRCC
    plcc_values = [data_dict[model][0] for model in data_dict]
    srcc_values = [data_dict[model][1] for model in data_dict]

    # 使用暖色系和冷色系的配色方案绘制 PLCC 和 SRCC 的 bar
    plcc_bars = plt.bar(index, plcc_values, bar_width, label='PLCC', color='#FF7F0E')  # 橙色
    srcc_bars = plt.bar(index + bar_width, srcc_values, bar_width, label='SRCC', color='#1F77B4')  # 蓝绿色

    # 在bar的顶部添加值
    for bar in plcc_bars + srcc_bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom')

    # 添加标签、标题和图例
    plt.xlabel('Models')
    plt.ylabel('Scores')
    plt.title('PLCC and SRCC Scores by Model')
    plt.xticks(index + bar_width / 2, data_dict.keys(), rotation=45)
    plt.legend()

    # 显示图表
    plt.tight_layout()
    plt.savefig("quality.pdf")
    plt.show()


plot_bar(data_dict)
