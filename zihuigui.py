import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder as Encoder
from torch.nn import TransformerDecoder as Decoder
from torch.nn import TransformerEncoderLayer as EncoderLayer
from torch.nn import TransformerDecoderLayer as DecoderLayer
from torch.nn import LayerNorm

import math
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.in_feat_size = cfg.in_feat_size
        # self.linear0 = nn.Linear(2, in_feature)
        # self.relu0 = nn.ReLU()
        self.lstm = nn.LSTM(cfg.in_feat_size, cfg.hidden_feat_size, 2)
        self.linear1 = nn.Linear(cfg.hidden_feat_size, cfg.out_seq_len)

    def forward(self, x: Tensor) -> Tensor:
        bs = x.size(0)
        x = x.view(bs, -1, self.in_feat_size)
        x = x.permute(1,0,2)
        x, (h, c) = self.lstm(x)
        x = x.permute(1,0,2)
        x = self.linear1(x[:,-1])
        
        return x

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# 位置编码
def _rope_cos_sin(seq_len: int, dim: int, device, dtype, base: float = 10000.0):
    """
    计算RoPE位置编码的cos和sin值
    标准公式: theta_i = base^(-2i/d), 其中 i = 0, 1, ..., d/2-1
    """
    assert dim % 2 == 0, "RoPE 维度必须为偶数"
    half = dim // 2
    # 修复：除以完整维度 dim，而不是 half
    inv_freq = 1.0 / (base ** (2 * torch.arange(0, half, device=device, dtype=dtype) / dim))
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.einsum('l,d->ld', t, inv_freq)  # [seq_len, half]
    # 修复：不需要 cat，直接返回
    emb = freqs.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, half]
    return emb.cos(), emb.sin()

def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    应用RoPE旋转变换
    旋转矩阵: [[cos, -sin], [sin, cos]]
    """
    x_even = x[..., ::2]   # 偶数位置
    x_odd = x[..., 1::2]   # 奇数位置
    # 修复：标准旋转公式，不需要 [..., ::2] 索引
    x_rope_even = x_even * cos - x_odd * sin
    x_rope_odd = x_even * sin + x_odd * cos  # 修复：x_even 在前
    out = torch.empty_like(x)
    out[..., ::2] = x_rope_even
    out[..., 1::2] = x_rope_odd
    return out

# Transformer网络
class TFMModel(nn.Module):
    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.in_feat_size = cfg.in_feat_size
        self.out_feat_size = cfg.out_feat_size
        self.in_seq_len = cfg.in_seq_len
        self.out_seq_len = cfg.out_seq_len
        self.hidden_feat_size = cfg.hidden_feat_size
        self.nhead = 8
        self.rope_base = 10000.0

        encoder_layer = EncoderLayer(d_model=self.hidden_feat_size, nhead=8, batch_first=True) # 单层多头自注意力
        encoder_norm = LayerNorm(self.hidden_feat_size)
        self.encoder = Encoder(encoder_layer, 6, encoder_norm)                         # transformer编码器

        decoder_layer = DecoderLayer(d_model=self.hidden_feat_size, nhead=8, batch_first=True) # 单层多头自注意力
        decoder_norm = LayerNorm(self.hidden_feat_size)
        self.decoder = Decoder(decoder_layer, 6, decoder_norm)                         # transformer解码器

        # 用FC代替embedding
        self.embedding_fc = nn.Linear(self.in_feat_size, self.hidden_feat_size)
        self.embedding_fc_t = nn.Linear(self.out_feat_size, self.hidden_feat_size)

        self.dropout = nn.Dropout(0.1)

        self.fc = nn.Linear(self.hidden_feat_size, self.out_feat_size)


    def forward_loop(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # 编码器处理
        x = self.embedding_fc(x)  # 对编码器的输入x进行embedding
        B, L, H = x.shape

        # 对encoder输入应用RoPE
        x = x.view(B, L, self.nhead, H // self.nhead).transpose(1, 2).contiguous()
        dim = H // self.nhead
        cos, sin = _rope_cos_sin(L, dim, device=x.device, dtype=x.dtype, base=self.rope_base)
        x = _apply_rope(x, cos, sin)
        x = x.transpose(1, 2).contiguous().view(B, L, H)

        x = self.dropout(x)
        x = self.encoder(x)  # [B, L, H]

        # 解码器自回归生成
        B = x.size(0)
        # 初始化：从零向量开始
        t = torch.zeros(B, 1, self.out_feat_size, device=x.device)

        for i in range(self.out_seq_len):
            B, L_cur, _ = t.shape
            ti = self.embedding_fc_t(t)  # [B, L_cur, H]

            # 对decoder输入应用RoPE
            H = ti.shape[-1]
            ti = ti.view(B, L_cur, self.nhead, H // self.nhead).transpose(1, 2).contiguous()
            dim = H // self.nhead
            cos, sin = _rope_cos_sin(L_cur, dim, device=ti.device, dtype=ti.dtype, base=self.rope_base)
            ti = _apply_rope(ti, cos, sin)
            ti = ti.transpose(1, 2).contiguous().view(B, L_cur, H)

            ti = self.dropout(ti)
            ti = self.decoder(ti, x)  # [B, L_cur, H]
            ti = self.fc(ti)  # [B, L_cur, out_feat_size]

            # 修复：每次只取最后一个预测，而不是ti[:,i:]
            next_token = ti[:, -1:, :]  # [B, 1, out_feat_size]
            t = torch.cat((t, next_token), dim=1)  # 拼接到序列

        return t[:, 1:]  # 移除初始的零向量，返回[B, out_seq_len, out_feat_size]



    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.forward_loop(x, t)
    
# 测试
if __name__ == '__main__':
    device = torch.device('cpu')

    # 创建配置对象
    class Config:
        def __init__(self):
            self.in_seq_len = 240
            self.out_seq_len = 48
            self.in_feat_size = 3
            self.out_feat_size = 1
            self.hidden_feat_size = 64

    cfg = Config()
    model = TFMModel(cfg).to(device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    src = torch.rand(2, 240, 3).to(device=device)
    tgt = torch.rand(2, 48, 1).to(device=device)
    y = torch.rand(2, 48, 1).to(device=device)

    loss_f = nn.MSELoss()

    model.train()
    for i in range(10):
        optimizer.zero_grad()
        out = model(src, tgt)
        loss = loss_f(out, y)
        print(loss.item())
        loss.backward()
        optimizer.step()