import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualConv, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return x + self.conv_block(x)

class TransformerBlock(nn.Module):
    def __init__(self, channel_size, num_heads, dim_feedforward=2048):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(channel_size)
        self.attn = nn.MultiheadAttention(channel_size, num_heads)
        self.norm2 = nn.LayerNorm(channel_size)
        self.ff = nn.Sequential(
            nn.Linear(channel_size, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, channel_size)
        )

    def forward(self, x):
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = self.norm2(x)
        x = self.ff(x)
        return x

class ImprovedUNet(nn.Module):
    def __init__(self):
        super(ImprovedUNet, self).__init__()
        self.initial_reshape = nn.Linear(768, 12 * 64)

        # 使用残差连接的卷积块
        self.bottleneck = ResidualConv(1, 32)

        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv1 = ResidualConv(16, 16)

        self.upconv2 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2 = ResidualConv(8, 8)

        # Transformer层
        self.transformer_block = TransformerBlock(8, 2)

        self.final_upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        self.final_conv = nn.Conv2d(8, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.initial_reshape(x)
        x = x.view(-1, 1, 12, 64)  # 初始转换

        x = self.bottleneck(x)

        x = self.upconv1(x)
        x = self.conv1(x)

        x = self.upconv2(x)
        x = self.conv2(x)

        # 保存空间尺寸
        spatial_size = x.size()[2:]

        # 应用Transformer层
        x = x.flatten(2).permute(2, 0, 1)
        x = self.transformer_block(x)

        # 重新调整尺寸
        x = x.permute(1, 2, 0).view(-1, 8, *spatial_size)
        x = self.final_upsample(x)
        x = self.final_conv(x)

        return x

if __name__ == "__main__":

    model = ImprovedUNet()
    input_tensor = torch.randn(1, 768)
    output = model(input_tensor)
    dummy_input = input_tensor
    print(output.shape)

    import torch.onnx
    torch.onnx.export(model, dummy_input, "upsample_model.onnx", verbose=True)