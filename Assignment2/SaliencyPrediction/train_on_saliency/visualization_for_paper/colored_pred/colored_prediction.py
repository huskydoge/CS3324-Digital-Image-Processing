import pandas as pd
import numpy as np
from PIL import Image, ImageFilter
# 从 pred_map 中提取像素值
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

path = os.path.join(dirname, '..')

# pred_map = Image.open("/data/husky/ImageReward/train_on_saliency/non_salient/2023-12-04_11-38-20_epochs=15_bs32_loss=saloss_fixrate=0.9_reshape=False_lr=5e-05_cosine_non_salient_ccw=2_simw=1_kldivw=10_nssw=1_msew=4_modelname=blend_usecross=0_fse=0.0_fsd=0.5/test/000000011264_2.png/14/pred.jpg")
pred_map = Image.open(os.path.join(path,
                                   "salient/2023-12-04_10-08-38_epochs=15_bs32_loss=saloss_fixrate=0.9_reshape=False_lr=5e-05_cosine_salient_ccw=2_simw=1_kldivw=10_nssw=1_msew=4_modelname=blend_usecross=0_fse=0.0_fsd=0.0/test/000000011264_3.png/14/pred.jpg"))
org_img = Image.open(os.path.join(path,
                                  "non_salient/2023-12-04_11-38-20_epochs=15_bs32_loss=saloss_fixrate=0.9_reshape=False_lr=5e-05_cosine_non_salient_ccw=2_simw=1_kldivw=10_nssw=1_msew=4_modelname=blend_usecross=0_fse=0.0_fsd=0.5/test/000000011264_2.png/14/origin.jpg"))

# 加载图像

# 将 org_img 调整大小以匹配 pred_map
org_img_resized = org_img.resize(pred_map.size)

# 从 pred_map 中提取像素值，并归一化
pred_map_arr = np.array(pred_map).astype(float) / 255

# 应用颜色映射以创建热力图
# 使用 matplotlib 的 colormap
colormap = cm.get_cmap('inferno')  # 选择一个热力图颜色映射，例如 inferno, jet,
pred_heatmap = colormap(pred_map_arr, bytes=True)  # 这将返回一个 RGBA 图像

# 将 RGBA 转换为 PIL 图像
pred_heatmap_img = Image.fromarray(pred_heatmap, mode="RGBA")

# 将热力图叠加到原始图像上
blended_img = Image.alpha_composite(org_img_resized.convert('RGBA'), pred_heatmap_img)

# 转换回 RGB
blended_img = blended_img.convert("RGB")

# 显示或保存结果
# blended_img.save("blendedmap.png", "PNG")  # 保存图像


# 创建一个与原图大小相同的透明图层
transparent_layer = Image.new("RGBA", org_img_resized.size, (0, 0, 0, 0))

# 将 blended_img 转换为 RGBA 模式
blended_rgba = blended_img.convert("RGBA")

# 去除纯黑像素（将其设置为透明）
data = np.array(blended_rgba)
red, green, blue, alpha = data.T
black_areas = (red == 0) & (blue == 0) & (green == 0)
data[..., :-1][black_areas.T] = (255, 255, 255)  # 将黑色区域转换为白色
data[..., -1][black_areas.T] = 0  # 将黑色区域的透明度设置为0

# 创建新的 PIL 图像
blended_no_black = Image.fromarray(data)

# 使用 blend 方法将处理过的 blended_img 与原图 org_img_resized 结合
final_img = Image.blend(org_img_resized.convert("RGBA"), blended_no_black, alpha=0.5)

# 转换回 RGB
final_img = final_img.convert("RGB")

# 显示或保存结果
final_img.save("blended_with_org_img.pdf", "PDF")
