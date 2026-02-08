#!/usr/bin/env python3
"""
太阳能电站功率预测 Transformer 模型
基于 teacher 分支 ImprovedTFMModel 适配

架构: 非自回归 Encoder + Cross-Attention Decoder
位置编码: RoPE (旋转位置编码)
核心: 可学习 Query Embeddings 并行预测
"""

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm
import math


# ============== RoPE 位置编码 ==============

def _rope_cos_sin(seq_len, dim, device, dtype, base=10000.0):
    assert dim % 2 == 0
    half = dim // 2
    inv_freq = 1.0 / (base ** (2 * torch.arange(0, half, device=device, dtype=dtype) / dim))
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.einsum("l,d->ld", t, inv_freq)
    emb = freqs.unsqueeze(0).unsqueeze(0)
    return emb.cos(), emb.sin()


def _apply_rope(x, cos, sin):
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    x_rope_even = x_even * cos - x_odd * sin
    x_rope_odd = x_even * sin + x_odd * cos
    out = torch.empty_like(x)
    out[..., ::2] = x_rope_even
    out[..., 1::2] = x_rope_odd
    return out


# ============== 模型 ==============

class SolarTransformer(nn.Module):
    """
    太阳能电站功率预测 Transformer

    特点:
    1. 非自回归并行预测 (无误差累积)
    2. RoPE旋转位置编码
    3. 可学习Query Embeddings
    4. 支持气象+时间+功率多特征输入
    """

    def __init__(self, in_feat_size=13, out_feat_size=1,
                 in_seq_len=96, out_seq_len=16,
                 hidden_size=256, nhead=8,
                 num_encoder_layers=6, num_cross_attn_layers=3,
                 dropout=0.1):
        super().__init__()

        self.in_feat_size = in_feat_size
        self.out_feat_size = out_feat_size
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.hidden_size = hidden_size
        self.nhead = nhead

        # 输入投影
        self.input_projection = nn.Linear(in_feat_size, hidden_size)

        # Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=LayerNorm(hidden_size),
        )

        # 可学习的 Query Embeddings (每个未来时间步一个query)
        self.future_queries = nn.Parameter(
            torch.randn(1, out_seq_len, hidden_size) * 0.02
        )

        # Cross-Attention + FFN 层
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_size, num_heads=nhead,
                dropout=dropout, batch_first=True,
            )
            for _ in range(num_cross_attn_layers)
        ])

        self.cross_attn_norms = nn.ModuleList([
            LayerNorm(hidden_size) for _ in range(num_cross_attn_layers)
        ])

        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 4, hidden_size),
                nn.Dropout(dropout),
            )
            for _ in range(num_cross_attn_layers)
        ])

        self.ffn_norms = nn.ModuleList([
            LayerNorm(hidden_size) for _ in range(num_cross_attn_layers)
        ])

        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, out_feat_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [B, in_seq_len, in_feat_size]
        Returns:
            [B, out_seq_len]
        """
        B = x.size(0)
        dim = self.hidden_size // self.nhead

        # 投影
        x = self.input_projection(x)  # [B, 96, hidden]

        # Encoder RoPE
        L, H = x.size(1), x.size(2)
        x = x.view(B, L, self.nhead, dim).transpose(1, 2)
        cos, sin = _rope_cos_sin(L, dim, device=x.device, dtype=x.dtype)
        x = _apply_rope(x, cos, sin)
        x = x.transpose(1, 2).contiguous().view(B, L, H)
        x = self.dropout(x)

        encoder_out = self.encoder(x)  # [B, 96, hidden]

        # Query RoPE
        queries = self.future_queries.expand(B, -1, -1)
        L_q = queries.size(1)
        queries = queries.view(B, L_q, self.nhead, dim).transpose(1, 2)
        cos_q, sin_q = _rope_cos_sin(L_q, dim, device=queries.device, dtype=queries.dtype)
        queries = _apply_rope(queries, cos_q, sin_q)
        queries = queries.transpose(1, 2).contiguous().view(B, L_q, H)

        # Cross-Attention Decoder
        out = queries
        for i in range(len(self.cross_attention_layers)):
            attn_out, _ = self.cross_attention_layers[i](
                query=out, key=encoder_out, value=encoder_out
            )
            out = self.cross_attn_norms[i](out + attn_out)
            ffn_out = self.ffns[i](out)
            out = self.ffn_norms[i](out + ffn_out)

        # 输出
        predictions = self.output_projection(out)  # [B, 16, 1]
        return predictions.squeeze(-1)  # [B, 16]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    model = SolarTransformer(
        in_feat_size=13, out_seq_len=16,
        in_seq_len=96, hidden_size=256,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params:,}")

    x = torch.randn(4, 96, 13).to(device)
    with torch.no_grad():
        out = model(x)
    print(f"输入: {x.shape} → 输出: {out.shape}")
