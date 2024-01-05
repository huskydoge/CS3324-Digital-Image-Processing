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
import torch
import torch.nn as nn
from PIL import Image
from .models.BLIP.blip_pretrain import BLIP_Pretrain
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from config.options import *
from .myTransformer.transformer_encoder import TransformerEncoder
from .myTransformer.iqa_transformer import ImageAssessmentModel

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


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
        self.ImageAssessmentModel = ImageAssessmentModel(device = self.device).to(self.device)
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
        # print("image_embeds.shape = ", image_embeds.shape)
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
