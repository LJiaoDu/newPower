#!/usr/bin/env python3
"""
太阳能电站功率预测 - Encoder-Only Transformer + RoPE

架构思路:
  将历史 96 步 (功率+气象+时间) 和未来 16 步 (气象+时间) 分别投影到 d_model,
  拼接为 112 长序列, 通过纯 Encoder self-attention 建模全局依赖,
  取最后 16 个位置的输出预测功率.

与 Encoder-Decoder 架构的区别:
  - 没有 cross-attention, 历史和未来在同一个 self-attention 中交互
  - 结构更简洁, 参数更少 (同等 d_model/layers 下)
  - 全局 attention 让未来位置能直接关注任意历史位置 (无需经过 memory 瓶颈)

特殊设计:
  - 两个独立 input projection: 历史 13 维 → d_model, 未来 12 维 → d_model
  - Segment Embedding: 区分历史 (seg=0) / 未来 (seg=1) 两段
  - RoPE 位置: [0..111] 连续, 编码绝对时间顺序
  - forward 接口与 ED 模型一致: forward(x_enc, x_dec) -> [B, 16]
"""

import torch
import torch.nn as nn
import math

from solar_model_rope import (
    RotaryPositionEncoding,
    RoPEEncoderLayer,
)


class SolarTransformerEncOnly(nn.Module):
    """
    Encoder-Only Transformer + RoPE

    Encoder 输入: [B, 96, enc_feat_size]  历史 (功率+气象+时间)
    Decoder 输入: [B, 16, dec_feat_size]  未来 (气象+时间, 无功率)
    输出:         [B, 16]                 未来功率预测
    """

    def __init__(
        self,
        enc_feat_size: int = 13,
        dec_feat_size: int = 12,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        enc_seq_len: int = 96,
        dec_seq_len: int = 16,
        rope_base: float = 10000.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.enc_seq_len = enc_seq_len
        self.dec_seq_len = dec_seq_len
        head_dim = d_model // nhead

        # --- 输入投影 (不同特征维度 → 统一 d_model) ---
        self.enc_projection = nn.Sequential(
            nn.Linear(enc_feat_size, d_model),
            nn.LayerNorm(d_model),
        )
        self.dec_projection = nn.Sequential(
            nn.Linear(dec_feat_size, d_model),
            nn.LayerNorm(d_model),
        )

        # --- Segment Embedding: 区分历史/未来 ---
        self.segment_emb = nn.Embedding(2, d_model)

        # --- RoPE (位置 [0..111] 连续) ---
        self.rope = RotaryPositionEncoding(
            head_dim=head_dim,
            max_len=enc_seq_len + dec_seq_len + 10,
            base=rope_base,
        )

        # --- Encoder Layers ---
        self.layers = nn.ModuleList([
            RoPEEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # --- 输出投影 (仅用于未来 16 步) ---
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x_enc: torch.Tensor, x_dec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_enc: [B, enc_seq_len, enc_feat_size]  历史序列
            x_dec: [B, dec_seq_len, dec_feat_size]  未来序列
        Returns:
            [B, dec_seq_len]  功率预测
        """
        device = x_enc.device
        enc_len = x_enc.size(1)
        dec_len = x_dec.size(1)
        total_len = enc_len + dec_len

        # 1. 投影到 d_model
        enc = self.enc_projection(x_enc)  # [B, 96, d_model]
        dec = self.dec_projection(x_dec)  # [B, 16, d_model]

        # 2. 加 Segment Embedding
        seg_hist = self.segment_emb(torch.zeros(enc_len, dtype=torch.long, device=device))
        seg_fut = self.segment_emb(torch.ones(dec_len, dtype=torch.long, device=device))
        enc = enc + seg_hist  # [B, 96, d_model]
        dec = dec + seg_fut   # [B, 16, d_model]

        # 3. 拼接为完整序列
        x = torch.cat([enc, dec], dim=1)  # [B, 112, d_model]

        # 4. RoPE 位置 [0..111]
        cos, sin = self.rope(total_len, offset=0)

        # 5. Encoder layers
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)

        # 6. 取未来 16 步输出, 预测功率
        future_out = x[:, enc_len:, :]                    # [B, 16, d_model]
        pred = self.output_projection(future_out)          # [B, 16, 1]
        return pred.squeeze(-1)                            # [B, 16]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    model = SolarTransformerEncOnly(
        enc_feat_size=13,
        dec_feat_size=12,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.2,
        enc_seq_len=96,
        dec_seq_len=16,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params:,}")

    x_enc = torch.randn(4, 96, 13).to(device)
    x_dec = torch.randn(4, 16, 12).to(device)
    with torch.no_grad():
        out = model(x_enc, x_dec)
    print(f"Encoder 输入: {x_enc.shape}")
    print(f"Decoder 输入: {x_dec.shape}")
    print(f"输出: {out.shape}")
