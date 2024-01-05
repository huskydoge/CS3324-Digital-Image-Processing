import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
from bench import get_PLLC, get_SRCC


df_align = pd.read_csv("/data/husky/ImageReward/assets/AGIQA-3K.csv")['mos_align'].values
df_quality = pd.read_csv("/data/husky/ImageReward/assets/AGIQA-3K.csv")['mos_quality'].values
df_bridge = pd.read_csv("/data/husky/ImageReward/AGIQA_3K_text_align_bridge.csv")["score"].values
df_blip = pd.read_csv("/data/husky/ImageReward/AGIQA_3K_text_align_blip.csv")
df_img_reward = pd.read_csv("/data/husky/ImageReward/AGIQA_3K_text_align_ImgRward.csv")['score'].values
df_img_trained = pd.read_csv("/data/husky/ImageReward/AGIQA_3K_text_align_ImgRward_trained.csv")['score'].values


# 拟合的函数形式
def func(X, alpha_1, alpha_2, alpha_3, alpha_4, alpha_5):
  return alpha_1 * (0.5 - 1 / (1 + np.exp(alpha_2 * (X - alpha_3)))) + alpha_4 * X + alpha_5

# 假设你有x_data和y_data两个numpy数组存放你的数据
y_data = df_align
# x_data = df_img_reward
x_data = df_img_trained

# 初始参数猜测，可以根据你的实际数据进行更好的猜测
initial_guess = [1, 1, 1, 1, 1]

# 使用curve_fit进行拟合
popt, pcov = curve_fit(func, x_data, y_data, p0=initial_guess,maxfev=50000)

print(f"拟合参数为：{popt}")

# 计算拟合后的y值
y_fitted = func(x_data, *popt)

plt.figure(figsize=(8, 6))
plt.scatter(x_data, y_data, label='Data')
plt.scatter(x_data, y_fitted, label='Fitted curve')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


# Calculate the PLCC and SRCC
plcc_before = get_PLLC(df_align, df_img_trained)
srcc_before = get_SRCC(df_align, df_img_trained)
print(f"PLCC before = {plcc_before:.4f}")
print(f"SRCC before = {srcc_before:.4f}")

plcc_after = get_PLLC(df_align, y_fitted)
srcc_after = get_SRCC(df_align, y_fitted)
print(f"PLCC after = {plcc_after:.4f}")
print(f"SRCC after = {srcc_after:.4f}")

