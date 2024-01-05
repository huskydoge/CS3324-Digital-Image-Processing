import os
import torch
import torch.nn as nn
from .utils import densenet, Encoder
import torch.nn.functional as F



# get current dir



cfg1 = {
    "hidden_size": 768,
    "mlp_dim": 768 * 4,
    "num_heads": 12,
    "num_layers": 2,
    "attention_dropout_rate": 0,
    "dropout_rate": 0.0,
}

cfg2 = {
    "hidden_size": 768,
    "mlp_dim": 768 * 4,
    "num_heads": 12,
    "num_layers": 2,
    "attention_dropout_rate": 0,
    "dropout_rate": 0.0,
}

cfg3 = {
    "hidden_size": 512,
    "mlp_dim": 512 * 4,
    "num_heads": 8,
    "num_layers": 2,
    "attention_dropout_rate": 0,
    "dropout_rate": 0.0,
}


class TranSalNet(nn.Module):

    def __init__(self, freeze_rate_encoder=1.0, freeze_rate_decoder=1.0, use_cross=1):
        super(TranSalNet, self).__init__()
        self.use_cross = use_cross
        self.encoder = _Encoder()
        self.decoder = _Decoder(self.use_cross)

        self.freeze_layers(self.encoder, freeze_rate_encoder)
        self.freeze_layers(self.decoder, freeze_rate_decoder, exclude_blendnet=True)

    def forward(self, x, txt_features=None, type='blend'):
        x = self.encoder(x)
        x = self.decoder(x, txt_features, type)
        return x

    def freeze_layers(self, module, freeze_rate, exclude_blendnet=False):

        print("fixing layers ============================")
        # 获取所有层的数量
        children = list(module.children())
        # print(children)
        total_layers = len(children)
        # 计算要冻结的层数
        freeze_layers = int(total_layers * freeze_rate)

        print("freezed layers: {}".format(freeze_layers))

        # 冻结前 freeze_layers 层
        for i, child in enumerate(children):
            if exclude_blendnet and isinstance(child, FeatureBlendingNetwork):
                continue
            if i < freeze_layers:
                for param in child.parameters():
                    param.requires_grad = False


class _Encoder(nn.Module):
    def __init__(self):
        super(_Encoder, self).__init__()
        base_model = densenet.densenet161(pretrained=True)
        base_layers = list(base_model.children())[0][:-1]
        self.encoder = nn.ModuleList(base_layers).eval()

    def forward(self, x):
        outputs = []
        for ii, layer in enumerate(self.encoder):
            x = layer(x)
            if ii in {6, 8, 10}:
                outputs.append(x)
        return outputs


class _Decoder(nn.Module):
    def __init__(self, use_cross=1):
        super(_Decoder, self).__init__()
        self.use_cross = use_cross
        self.blendnet = FeatureBlendingNetwork(self.use_cross)
        self.conv1 = nn.Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(768, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv6 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv7 = nn.Conv2d(32, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.batchnorm1 = nn.BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batchnorm2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batchnorm3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batchnorm4 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batchnorm5 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batchnorm6 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.TransEncoder1 = TransEncoder(in_channels=2208, spatial_size=9 * 12, cfg=cfg1)
        self.TransEncoder2 = TransEncoder(in_channels=2112, spatial_size=18 * 24, cfg=cfg2)
        self.TransEncoder3 = TransEncoder(in_channels=768, spatial_size=36 * 48, cfg=cfg3)

        self.add = torch.add
        self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')


    def forward(self, x, txt_features=None, type="blend"):
        # decoder has three input, for transe1 ~ transe3
        x3, x4, x5 = x
        # print(x5.shape)

        # block 1
        x5 = self.TransEncoder1(x5)
        x5 = self.conv1(x5)
        x5 = self.batchnorm1(x5)
        x5 = self.relu(x5)
        x5 = self.upsample(x5)

        x4_a = self.TransEncoder2(x4)  # elementwise product
        # print(x4_a.shape)  # [1, 768, 18, 24]

        x4 = x5 * x4_a

        # x4 = x4 * txt_features # used for simple upsample in Imgreward, where txt_features already be upsampled to [1, 768, 18, 24]
        if type == "blend":
            x4_ = self.blendnet(txt_features, x4)  # txt_features still [1, 768]
            x4 = x4_
        elif type == "minisample":
            x4 = x4 * txt_features  # used for simple upsample in Imgreward, where txt_features already be upsampled to [1, 768, 18, 24]

        x4 = self.relu(x4)
        x4 = self.conv2(x4)
        x4 = self.batchnorm2(x4)
        x4 = self.relu(x4)
        x4 = self.upsample(x4)

        x3_a = self.TransEncoder3(x3)
        # print(x3.shape)
        x3 = x4 * x3_a
        x3 = self.relu(x3)
        x3 = self.conv3(x3)
        x3 = self.batchnorm3(x3)
        x3 = self.relu(x3)
        x3 = self.upsample(x3)

        x2 = self.conv4(x3)
        x2 = self.batchnorm4(x2)
        x2 = self.relu(x2)
        x2 = self.upsample(x2)
        x2 = self.conv5(x2)
        x2 = self.batchnorm5(x2)
        x2 = self.relu(x2)

        x1 = self.upsample(x2)
        x1 = self.conv6(x1)
        x1 = self.batchnorm6(x1)
        x1 = self.relu(x1)
        x1 = self.conv7(x1)
        x = self.sigmoid(x1)

        return x


class TransEncoder(nn.Module):

    def __init__(self, in_channels, spatial_size, cfg):
        super(TransEncoder, self).__init__()

        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                          out_channels=cfg['hidden_size'],
                                          kernel_size=1,
                                          stride=1)
        self.position_embeddings = nn.Parameter(torch.zeros(1, spatial_size, cfg['hidden_size']))

        self.transformer_encoder = Encoder(cfg)

    def forward(self, x):
        a, b = x.shape[2], x.shape[3]
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)

        embeddings = x + self.position_embeddings
        x = self.transformer_encoder(embeddings)
        B, n_patch, hidden = x.shape
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, a, b)

        return x


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
        if self.use_cross == 1:
            print("FeatureBlendingNetwork is using cross attention!")
        else:
            print("FeatureBlendingNetwork is using upsampling and concatenate![if using transalnet, then there is no upsampling and concatenate]")

    def forward(self, x1, x2):
        x1_upsampled = self.upsample(x1)
        # print("use cross is:", self.use_cross)
        if self.use_cross == 0:
            x1_att = x1_upsampled
            # print("first")
            # exit(0)
        elif self.use_cross == 1:
            # print("use cross")
            # exit(0)
            x1_att = self.cross_attention(x1_upsampled, x2)
            return x1_att

        x_combined = torch.cat((x1_att, x2), dim=1)
        x_combined = self.conv1(x_combined)
        x_combined = self.bn1(x_combined)
        x_combined = F.relu(x_combined, inplace=False)
        x_combined = self.conv2(x_combined)
        x_combined = self.bn2(x_combined)
        x_combined = F.relu(x_combined, inplace=False)
        # print("third")
        # exit(0)
        return x_combined



if __name__ == "__main__":
    pass
    # dummy_input = torch.randn(1, 3, 288, 384)
    # device = torch.device('cuda:4')
    # te = TransEncoder(in_channels=3, spatial_size=288*384, cfg=cfg1).to(device)
    # # print(te)
    # output = te(dummy_input)
    # print(output)
    #
    # net = MiniUpsamplingNetwork()
    #
    # # Create a dummy input tensor
    # input_tensor = torch.randn(1, 768)
    #
    # # Pass the input tensor through the network
    # output_tensor = net(input_tensor)
    #
    # print("Output tensor shape:", output_tensor.shape)
