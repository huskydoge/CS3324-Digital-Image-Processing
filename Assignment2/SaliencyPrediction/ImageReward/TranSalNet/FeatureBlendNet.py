import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义交叉注意力层
class CrossAttentionLayer(nn.Module):
    def __init__(self, channel):
        super(CrossAttentionLayer, self).__init__()
        self.query_conv = nn.Conv2d(channel, channel // 8, kernel_size=1)
        self.bn_query = nn.BatchNorm2d(channel // 8)
        self.key_conv = nn.Conv2d(channel, channel // 8, kernel_size=1)
        self.bn_key = nn.BatchNorm2d(channel // 8)
        self.value_conv = nn.Conv2d(channel, channel, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(channel)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        m_batchsize, C, width, height = x1.size()
        proj_query = self.bn_query(self.query_conv(x1)).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.bn_key(self.key_conv(x2)).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key) / (C // 8) ** 0.5  # 缩放因子
        attention = self.softmax(energy)
        proj_value = self.bn_value(self.value_conv(x2)).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        return out


# 定义 MiniUpsamplingNetwork
class MiniUpsamplingNetwork(nn.Module):
    def __init__(self):
        super(MiniUpsamplingNetwork, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(18, 24), stride=(18, 24),
                                           padding=0)

    def forward(self, x):
        x = x.view(-1, 1, 1, 768)  # 形状变为[batch_size, channels, height, width]
        x = self.upsample(x)  # 应用卷积转置层进行上采样
        return x.view(1, 768, 18, 24)


# 定义 FeatureBlendingNetwork
class FeatureBlendingNetwork(nn.Module):
    def __init__(self, use_cross=1):
        super(FeatureBlendingNetwork, self).__init__()
        self.upsample = MiniUpsamplingNetwork()
        self.cross_attention = CrossAttentionLayer(768)
        self.conv1 = nn.Conv2d(768 * 2, 768, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(768)
        self.conv2 = nn.Conv2d(768, 768, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(768)
        self.use_cross = use_cross
        exit(0)

    def forward(self, x1, x2):
        x1_upsampled = self.upsample(x1)
        print("use cross is:", self.use_cross)
        if self.use_cross == 0:
            x1_att = x1_upsampled
            print("first")
            exit(0)
        elif self.use_cross == 1:
            print("use cross")
            exit(0)
            x1_att = self.cross_attention(x1_upsampled, x2)
            return x1_att
        else:
            x_combined = torch.cat((self.x1_att, x2), dim=1)
            x_combined = self.conv1(x_combined)
            x_combined = self.bn1(x_combined)
            x_combined = F.relu(x_combined, inplace=False)
            x_combined = self.conv2(x_combined)
            x_combined = self.bn2(x_combined)
            x_combined = F.relu(x_combined, inplace=False)
            print("third")
            exit(0)
        return x_combined


if __name__ == "__main__":
    # 创建模型实例并尝试运行
    model = FeatureBlendingNetwork(1)

    # 创建两个模拟输入张量
    x1 = torch.randn(1, 768)
    x2 = torch.randn(1, 768, 18, 24)

    # 通过模型运行输入
    output = model(x1, x2)
    print(output.shape)
