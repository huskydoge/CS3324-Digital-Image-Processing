import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, src):
        src2 = self.norm1(src)
        src = src + self.dropout1(self.self_attn(src2, src2, src2)[0])
        src2 = self.norm2(src)
        src = src + self.dropout2(self.linear2(self.dropout(F.relu(self.linear1(src2)))))
        return src


class ImageAssessmentModel(nn.Module):
    def __init__(self, d_model=768, nhead=3, device=None):
        super(ImageAssessmentModel, self).__init__()
        self.device = device
        self.encoder = TransformerEncoderLayer(d_model, nhead)
        self.wq_quality = nn.Parameter(torch.randn(d_model, d_model))
        self.wk_quality = nn.Parameter(torch.randn(d_model, d_model))
        self.wv_quality = nn.Parameter(torch.randn(d_model, d_model))
        # Similar parameters for text_image_correspondence and authenticity
        self.wq_text_image_correspondence = nn.Parameter(torch.randn(d_model, d_model))
        self.wk_text_image_correspondence = nn.Parameter(torch.randn(d_model, d_model))
        self.wv_text_image_correspondence = nn.Parameter(torch.randn(d_model, d_model))
        self.wq_authenticity = nn.Parameter(torch.randn(d_model, d_model))
        self.wk_authenticity = nn.Parameter(torch.randn(d_model, d_model))
        self.wv_authenticity = nn.Parameter(torch.randn(d_model, d_model))

    def forward(self, features):
        encoded_features = self.encoder(features).to(self.device)
        quality_score = self.calculate_score(encoded_features, self.wq_quality, self.wk_quality, self.wv_quality)
        # Similar calculations for text_image_correspondence and authenticity
        text_image_correspondence_score = self.calculate_score(encoded_features, self.wq_text_image_correspondence,
                                                               self.wk_text_image_correspondence,
                                                               self.wv_text_image_correspondence)
        authenticity_score = self.calculate_score(encoded_features, self.wq_authenticity, self.wk_authenticity,
                                                  self.wv_authenticity)
        # print("quality_score = ", quality_score)
        # print("text_image_correspondence_score = ", text_image_correspondence_score)
        # print("authenticity_score = ", authenticity_score)
        return quality_score, text_image_correspondence_score, authenticity_score

    def calculate_score(self, features, wq, wk, wv):
        q = torch.matmul(features, wq).to(self.device)
        k = torch.matmul(features, wk).to(self.device)
        v = torch.matmul(features, wv).to(self.device)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / q.size(-1) ** 0.5
        attention = F.softmax(attention_scores, dim=-1) * v
        score = torch.mean(attention)
        return score


if __name__ == "__main__":
    # Example usage
    model = ImageAssessmentModel()
    features = torch.randn(1, 768)  # Example feature vector
    scores = model(features)
    print(scores)
