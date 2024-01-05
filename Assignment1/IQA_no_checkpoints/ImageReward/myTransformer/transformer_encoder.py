import torch
import torch.nn as nn
import math
import numpy as np


class Config(object):
    def __init__(self):
        self.vocab_size = 6

        self.d_model = 768
        self.n_heads = 1
        self.dim_k = 768
        self.dim_v = 768

        assert self.d_model % self.n_heads == 0
        # dim_k = d_model % n_heads
        # dim_v = d_model % n_heads

        self.padding_size = 30
        self.UNK = 5
        self.PAD = 4

        self.N = 6
        self.p = 0.1


config = Config()


class Mutihead_Attention(nn.Module):
    def __init__(self, d_model, dim_k, dim_v, n_heads):
        super(Mutihead_Attention, self).__init__()
        self.dim_v = dim_v
        self.dim_k = dim_k
        self.n_heads = n_heads

        self.q = nn.Linear(d_model, dim_k)
        self.k = nn.Linear(d_model, dim_k)
        self.v = nn.Linear(d_model, dim_v)

        self.o = nn.Linear(dim_v, d_model)
        self.norm_fact = 1 / math.sqrt(d_model)

    def forward(self, x, y, requires_mask=False):
        assert self.dim_k % self.n_heads == 0 and self.dim_v % self.n_heads == 0
        # size of x : [batch_size * seq_len * batch_size]
        # 对 x 进行自注意力, 只用encoder的话, y一般等于x
        # print(self.q(x))
        Q = self.q(x).reshape(-1, x.shape[0], x.shape[1],
                              self.dim_k // self.n_heads)  # n_heads * batch_size * seq_len * dim_k
        # print(Q.shape)
        K = self.k(x).reshape(-1, x.shape[0], x.shape[1],
                              self.dim_k // self.n_heads)  # n_heads * batch_size * seq_len * dim_k
        # print(K.shape)
        # print(K)
        V = self.v(y).reshape(-1, y.shape[0], y.shape[1],
                              self.dim_v // self.n_heads)  # n_heads * batch_size * seq_len * dim_v
        # print("Attention V shape : {}".format(V.shape))
        # print(K.permute(0, 1, 3, 2))
        attention_score = torch.matmul(Q, K.permute(0, 1, 3, 2)) * self.norm_fact
        # print(attention_score)
        output = torch.matmul(attention_score, V).reshape(y.shape[0], y.shape[1], -1)
        # print("Attention output shape : {}".format(output.shape))

        output = self.o(output)
        return output


class Feed_Forward(nn.Module):
    def __init__(self, input_dim, hidden_dim=2048):
        super(Feed_Forward, self).__init__()
        self.L1 = nn.Linear(input_dim, hidden_dim)
        self.L2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        output = nn.ReLU()(self.L1(x))
        output = self.L2(output)
        return output


class Add_Norm(nn.Module):
    def __init__(self, device = None):
        super(Add_Norm, self).__init__()
        self.dropout = nn.Dropout(config.p)
        self.device = device

    def forward(self, x, sub_layer, **kwargs):
        sub_output = sub_layer(x, **kwargs)
        # print("{} output : {}".format(sub_layer,sub_output.size()))
        x = self.dropout(x + sub_output)

        layer_norm = nn.LayerNorm(x.size()[1:]).to(self.device)
        out = layer_norm(x)
        # print(out.is_cuda)
        # out.to('cuda:1')
        return out


class TransformerEncoder(nn.Module):
    def __init__(self,device):
        super(TransformerEncoder, self).__init__()
        self.device = device
        # self.positional_encoding = Positional_Encoding(config.d_model)
        self.muti_atten = Mutihead_Attention(config.d_model, config.dim_k, config.dim_v, config.n_heads).to(self.device)
        self.feed_forward = Feed_Forward(config.d_model).to(self.device)

        self.add_norm = Add_Norm(device = self.device)

        self.output_layer = nn.Linear(config.d_model, 1).to(self.device)

    def forward(self, x):  # batch_size * seq_len 并且 x 的类型不是tensor，是普通list

        # x += self.positional_encoding(x.shape[1], config.d_model)
        # print("After positional_encoding: {}".format(x.size()))
        output = self.add_norm(x, self.muti_atten, y=x).to(self.device)
        output = self.add_norm(output, self.feed_forward).to(self.device)
        output = self.output_layer(output).squeeze(-1).to(self.device)
        return output


if __name__ == "__main__":
    model = TransformerEncoder()

    # 假设我们有一个768维的特征向量input, batchsize = 1, seq_len = 3, d_model = 768
    input = torch.rand(1, 3, 768)
    output = model(input)
    print(output.shape)
    print(output)