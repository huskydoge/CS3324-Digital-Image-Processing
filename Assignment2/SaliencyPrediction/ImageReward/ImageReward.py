'''
@File       :   ImageReward.py
@Time       :   2023/01/28 19:53:00
@Auther     :   Jiazheng Xu
@Contact    :   xjz22@mails.tsinghua.edu.cn
@Description:   ImageReward Reward model.
* Based on CLIP code base and improved-aesthetic-predictor code base
* https://github.com/openai/CLIP
* https://github.com/christophschuhmann/improved-aesthetic-predictor
'''

import os
import sys

# absolute path
sys.path.append((os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from PIL import Image
from .models.BLIP.blip_pretrain import BLIP_Pretrain
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision import transforms, utils, models
from config.options import *
import torch.nn.functional as F
from .myTransformer.transformer_encoder import TransformerEncoder
from .myTransformer.iqa_transformer import ImageAssessmentModel
from .myUnet.upSample import ImprovedUNet
from .TranSalNet import TranSalNet
from u2net import U2NET
from copy import deepcopy
import numpy as np
dirname = os.path.dirname(os.path.abspath(__file__))
dirname = os.path.join(dirname,'TranSalNet')
try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


# def _transform(n_px):
#     return Compose([
#         Resize(n_px, interpolation=BICUBIC),
#         CenterCrop(n_px),
#         _convert_image_to_rgb,
#         ToTensor(),
#         Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
#     ])

def _transform(n_px=224):
    return Compose([
        transforms.Resize((n_px, n_px)),
        _convert_image_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def _transform_2d(h, w, input):
    # 首先将 tensor reshape 到一个近似的正方形尺寸
    tensor = input.reshape(1, 1, 16, 48)  # 16*48=768

    # 使用双三次插值将 tensor resize 到目标尺寸
    tensor = F.interpolate(tensor, size=(h, w), mode='bicubic', align_corners=False)

    # 对 tensor 进行归一化
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    return tensor


class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            # nn.ReLU(),
            nn.Linear(16, 1)
        )

        # initial MLP param
        for name, param in self.layers.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0 / (self.input_size + 1))
            if 'bias' in name:
                nn.init.constant_(param, val=0)

    def forward(self, input):
        return self.layers(input)


class MLP_new(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 2048),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            # nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 224),
            # nn.ReLU(),
            nn.Linear(16, 1)
        )

        # initial MLP param
        for name, param in self.layers.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0 / (self.input_size + 1))
            if 'bias' in name:
                nn.init.constant_(param, val=0)

    def forward(self, input):
        return self.layers(input)


class MiniUpsamplingNetwork(nn.Module):
    def __init__(self):
        super(MiniUpsamplingNetwork, self).__init__()
        # 使用卷积转置层进行上采样
        self.upsample = nn.ConvTranspose2d(
            in_channels=1,  # 假设输入特征作为单通道处理
            out_channels=1,  # 输出同样是单通道
            kernel_size=(18, 24),  # 核心大小与上采样的维度相匹配
            stride=(18, 24),  # 步幅与核心大小相同
            padding=0,  # 无填充
        )

    def forward(self, x):
        # 首先将输入特征调整为合适的维度，以符合卷积转置层的输入要求
        x = x.view(-1, 1, 1, 768)  # 形状变为[batch_size, channels, height, width]
        x = self.upsample(x)  # 应用卷积转置层进行上采样
        x = x.view(1, 768, 18, 24)
        return x


class ImageReward(nn.Module):
    def __init__(self, med_config, device='cpu'):
        super().__init__()
        self.device = device

        self.blip = BLIP_Pretrain(image_size=224, vit='large', med_config=med_config)
        if opts.fix_base:
            self.blip.requires_grad_(False)

        for name, parms in self.blip.named_parameters():
            if '_proj' in name:
                parms.requires_grad_(False)

        # fix certain ratio of layers
        self.image_layer_num = 24 if config['BLIP']['vit'] == 'large' else 12
        if opts.fix_rate > 0:
            text_fix_num = "layer.{}".format(int(12 * opts.fix_rate))
            image_fix_num = "blocks.{}".format(int(self.image_layer_num * opts.fix_rate))
            for name, parms in self.blip.text_encoder.named_parameters():
                parms.requires_grad_(False)
                if text_fix_num in name:
                    break
            for name, parms in self.blip.visual_encoder.named_parameters():
                parms.requires_grad_(False)
                if image_fix_num in name:
                    break
        self.preprocess = _transform(224)
        self.mlp = MLP(768)
        self.mean = 0.16717362830052426
        self.std = 1.0333394966054072
        # self.transformer_encoder = TransformerEncoder(device = self.device).to(self.device)
        self.ImageAssessmentModel = ImageAssessmentModel(device=self.device).to(self.device)
        # self.mean = 2.5
        # self.std = 0.73

    def forward(self, prompt, image, use_transformer=False):
        """
        prompt: str, the prompt for the image
        image: PIL.Image or str, the image to be scored
        """

        # text encode
        text_input = self.blip.tokenizer(prompt, padding='max_length', truncation=True, max_length=35,
                                         return_tensors="pt").to(self.device)
        # image encode
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, str):
            if os.path.isfile(image):
                pil_image = Image.open(image)
        else:
            raise TypeError(
                r'This image parameter type has not been supportted yet. Please pass PIL.Image or file path str.')

        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        image_embeds = self.blip.visual_encoder(image)
        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
        text_output = self.blip.text_encoder(text_input.input_ids,
                                             attention_mask=text_input.attention_mask,
                                             encoder_hidden_states=image_embeds,
                                             encoder_attention_mask=image_atts,
                                             return_dict=True,
                                             )

        txt_features = text_output.last_hidden_state[:, 0, :].float()  # (feature_dim)
        # print(txt_features.shape)
        # exit(0)
        if use_transformer:
            # duplicate txt_features 3 times to get an input with shape (1,3,768)
            # print(txt_features.shape)
            # txt_features is (1, 768)
            # get (1 ,3, 768)
            # txt_features = txt_features.unsqueeze(0)
            # txt_features = txt_features.repeat(1, 3, 1)
            # # print(txt_features.shape)
            # rewards = self.transformer_encoder(txt_features)
            # rewards = rewards.squeeze(0)

            rewards = self.ImageAssessmentModel(txt_features)
        else:
            rewards = self.mlp(txt_features)
        # rewards = (rewards - self.mean) / self.std

        return rewards

    def score_gard(self, prompt_ids, prompt_attention_mask, image):

        image_embeds = self.blip.visual_encoder(image)
        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
        text_output = self.blip.text_encoder(prompt_ids,
                                             attention_mask=prompt_attention_mask,
                                             encoder_hidden_states=image_embeds,
                                             encoder_attention_mask=image_atts,
                                             return_dict=True,
                                             )

        txt_features = text_output.last_hidden_state[:, 0, :]  # (feature_dim)
        rewards = self.mlp(txt_features)
        rewards = (rewards - self.mean) / self.std

        return rewards

    def score(self, prompt, image):

        if (type(image).__name__ == 'list'):
            _, rewards = self.inference_rank(prompt, image)
            return rewards

        # text encode
        text_input = self.blip.tokenizer(prompt, padding='max_length', truncation=True, max_length=35,
                                         return_tensors="pt").to(self.device)

        # image encode
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, str):
            if os.path.isfile(image):
                pil_image = Image.open(image)
        else:
            raise TypeError(
                r'This image parameter type has not been supportted yet. Please pass PIL.Image or file path str.')

        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        image_embeds = self.blip.visual_encoder(image)

        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
        text_output = self.blip.text_encoder(text_input.input_ids,
                                             attention_mask=text_input.attention_mask,
                                             encoder_hidden_states=image_embeds,
                                             encoder_attention_mask=image_atts,
                                             return_dict=True,
                                             )

        txt_features = text_output.last_hidden_state[:, 0, :].float()  # (feature_dim)
        rewards = self.mlp(txt_features)
        rewards = (rewards - self.mean) / self.std

        return rewards.detach().cpu().numpy().item()

    def score_forall(self, prompt, image):

        if (type(image).__name__ == 'list'):
            _, rewards = self.inference_rank(prompt, image)
            return rewards

        # text encode
        text_input = self.blip.tokenizer(prompt, padding='max_length', truncation=True, max_length=35,
                                         return_tensors="pt").to(self.device)

        # image encode
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, str):
            if os.path.isfile(image):
                pil_image = Image.open(image)
        else:
            raise TypeError(
                r'This image parameter type has not been supportted yet. Please pass PIL.Image or file path str.')

        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        image_embeds = self.blip.visual_encoder(image)

        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
        text_output = self.blip.text_encoder(text_input.input_ids,
                                             attention_mask=text_input.attention_mask,
                                             encoder_hidden_states=image_embeds,
                                             encoder_attention_mask=image_atts,
                                             return_dict=True,
                                             )

        txt_features = text_output.last_hidden_state[:, 0, :].float()  # (feature_dim)

        # txt_features = txt_features.unsqueeze(0)
        # txt_features = txt_features.repeat(1, 3, 1)
        # print(txt_features.shape)
        # rewards = self.transformer_encoder(txt_features)
        # rewards = rewards.squeeze(0)
        # return rewards.detach().cpu()

        rewards = [score for score in self.ImageAssessmentModel(txt_features)]
        return rewards

    def inference_rank(self, prompt, generations_list):

        text_input = self.blip.tokenizer(prompt, padding='max_length', truncation=True, max_length=35,
                                         return_tensors="pt").to(self.device)

        txt_set = []
        for generation in generations_list:
            # image encode
            if isinstance(generation, Image.Image):
                pil_image = generation
            elif isinstance(generation, str):
                if os.path.isfile(generation):
                    pil_image = Image.open(generation)
            else:
                raise TypeError(
                    r'This image parameter type has not been supportted yet. Please pass PIL.Image or file path str.')
            image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            image_embeds = self.blip.visual_encoder(image)

            # text encode cross attention with image
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
            text_output = self.blip.text_encoder(text_input.input_ids,
                                                 attention_mask=text_input.attention_mask,
                                                 encoder_hidden_states=image_embeds,
                                                 encoder_attention_mask=image_atts,
                                                 return_dict=True,
                                                 )
            txt_set.append(text_output.last_hidden_state[:, 0, :])

        txt_features = torch.cat(txt_set, 0).float()  # [image_num, feature_dim]
        rewards = self.mlp(txt_features)  # [image_num, 1]
        rewards = (rewards - self.mean) / self.std
        rewards = torch.squeeze(rewards)
        _, rank = torch.sort(rewards, dim=0, descending=True)
        _, indices = torch.sort(rank, dim=0)
        indices = indices + 1

        return indices.detach().cpu().numpy().tolist(), rewards.detach().cpu().numpy().tolist()


class SaliencyImageReward(nn.Module):
    def __init__(self, med_config, device='cpu', reshape=False):
        super().__init__()
        self.device = device
        self.reshape = reshape

        self.blip = BLIP_Pretrain(image_size=224, vit='large', med_config=med_config)
        if opts.fix_base:
            self.blip.requires_grad_(False)

        for name, parms in self.blip.named_parameters():
            if '_proj' in name:
                parms.requires_grad_(False)

        # fix certain ratio of layers
        self.image_layer_num = 24 if config['BLIP']['vit'] == 'large' else 12
        if opts.fix_rate > 0:
            text_fix_num = "layer.{}".format(int(12 * opts.fix_rate))
            image_fix_num = "blocks.{}".format(int(self.image_layer_num * opts.fix_rate))
            for name, parms in self.blip.text_encoder.named_parameters():
                parms.requires_grad_(False)
                if text_fix_num in name:
                    break
            for name, parms in self.blip.visual_encoder.named_parameters():
                parms.requires_grad_(False)
                if image_fix_num in name:
                    break
        self.preprocess = _transform(224)
        self.mlp = MLP(768)
        self.mean = 0.16717362830052426
        self.std = 1.0333394966054072
        # self.transformer_encoder = TransformerEncoder(device = self.device).to(self.device)
        self.net = U2NET(1, 1).to(self.device)
        self.improvedUnet = ImprovedUNet().to(self.device)
        if opts.model_name == "transalnet":
            print("using raw transalnet, should not fix the net!!")
            self.salNet = TranSalNet(freeze_rate_encoder=0, freeze_rate_decoder=0,use_cross=opts.use_cross)
        else:
            print("not using raw transalnet, fix the net!!")
            self.salNet = TranSalNet(freeze_rate_encoder=opts.fse, freeze_rate_decoder=opts.fsd,use_cross=opts.use_cross)

        # print(self.salNet)
        self.salNet.load_state_dict(torch.load(os.path.join(os.path.join(dirname, 'pretrained_models/TranSalNet_Dense.pth'))),
                             strict=False)
        print("TransalNet Loaded!")
        self.miniUpsamplingNetwork = MiniUpsamplingNetwork()
        # self.mean = 2.5
        # self.std = 0.73

    def forward(self, prompt, image, use_transformer=False):
        """
        prompt: str, the prompt for the image
        image: PIL.Image or str, the image to be scored
        """

        # text encode
        text_input = self.blip.tokenizer(prompt, padding='max_length', truncation=True, max_length=35,
                                         return_tensors="pt").to(self.device)
        # image encode
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, str):
            if os.path.isfile(image):
                pil_image = Image.open(image)
        else:
            raise TypeError(
                r'This image parameter type has not been supportted yet. Please pass PIL.Image or file path str.')

        # process image for TranSalNet
        image_width, image_height = pil_image.size
        image_for_salnet = deepcopy(image).convert('RGB')
        image_for_salnet = transforms.Resize((288, 384))(image_for_salnet)
        image_for_salnet = transforms.ToTensor()(image_for_salnet)
        # normalize the image
        image_for_salnet = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image_for_salnet)
        image_for_salnet = image_for_salnet.unsqueeze(0).to(self.device)



        #
        # image_for_salnet = np.expand_dims(np.transpose(image_for_salnet, (2, 0, 1)),
        #                                   axis=0)  # reshape the image to 1x3xHxW
        # image_for_salnet = torch.from_numpy(image_for_salnet)
        # image_for_salnet = image_for_salnet.type(torch.cuda.FloatTensor).to(self.device)
        # print("image salient info: ", image_for_1salnet.shape)

        # print("before: ", image_width, image_height )
        with open("process.txt", 'w') as f:
            image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            image_embeds = self.blip.visual_encoder(image)
            # f.write("image_embeds.shape = " + str(image_embeds.shape) + "\n")
            # text encode cross attention with image
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
            # f.write("img_atts" + str(image_atts.shape) + str(image_atts))
            # f.write("text_input" + str(text_input))
            text_output = self.blip.text_encoder(text_input.input_ids,
                                                 attention_mask=text_input.attention_mask,
                                                 encoder_hidden_states=image_embeds,
                                                 encoder_attention_mask=image_atts,
                                                 return_dict=True,
                                                 )
            # print(text_output.last_hidden_state.shape)
            # f.write("textoutput" + str(text_output))
            txt_features = text_output.last_hidden_state[:, 0, :].float()  # (feature_dim) (1,768)
            # make txt_features (1,1,1,768)
            txt_features = txt_features.unsqueeze(0).unsqueeze(0)
            # print("txt_features.shape = ", txt_features.shape)

            model_name = opts.model_name
            if model_name == "unet":
                d1, d2, d3, d4, d5, d6, d7 = self.net(txt_features)
                # upsampling the saliency to the original image size, to get saliency map (grey img)
                # upsampling
                # saliency = torch.nn.functional.interpolate(saliency.unsqueeze(0), size=(image_height, image_width),
                #                                             mode='bilinear', align_corners=False)
                #
                # normalize
                if self.reshape:
                    d1 = _transform_2d(image_height, image_width, d1)
                    d2 = _transform_2d(image_height, image_width, d2)
                    d3 = _transform_2d(image_height, image_width, d3)
                    d4 = _transform_2d(image_height, image_width, d4)
                    d5 = _transform_2d(image_height, image_width, d5)
                    d6 = _transform_2d(image_height, image_width, d6)
                    d7 = _transform_2d(image_height, image_width, d7)

                return d1, d2, d3, d4, d5, d6, d7
            elif model_name == "toy":
                output = self.improvedUnet(txt_features)
                output.squeeze(0)
                # use tools from torch to normalize the ouput, retain grad

                output = F.normalize(output, p=2, dim=0)
                # print(output, output.shape)
                return output
            elif model_name == "blend":
                output = self.salNet(image_for_salnet, txt_features, model_name)
                return output
            elif model_name == "minsample":
                txt_features = self.miniUpsamplingNetwork(txt_features)
                output = self.salNet(image_for_salnet, txt_features, model_name)
                return output
            elif model_name == "transalnet":
                output = self.salNet(image_for_salnet, None, model_name)
                return output


if __name__ == "__main__":
    model = SaliencyImageReward(med_config="ImageReward/config/med_config.json", device='cuda:4')
    prompt = ""
    input = model.blip.tokenizer(prompt, padding='max_length', truncation=True, max_length=35,
                                 return_tensors="pt")
    print(input)
