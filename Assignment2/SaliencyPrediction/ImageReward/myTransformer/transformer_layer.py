import torch
from torch import nn
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    def __init__(self, d_model=768, nhead=1, dim_feedforward=2048):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(dim_feedforward, 3)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout(src2)
        src = self.linear2(self.dropout(F.relu(self.linear1(src))))
        return src

# 初始化模型
model = TransformerEncoder()

# 假设我们有一个768维的特征向量F
input = torch.rand(1, 1, 768)

# 通过模型得到一个三维向量
output = model(input)

print(output)
print(model.parameters)