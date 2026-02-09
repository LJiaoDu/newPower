#!/usr/bin/env python3
"""
太阳能电站功率预测 - 标准 Encoder-Decoder Transformer

改进点 (相比旧版):
1. 标准 Encoder-Decoder 架构, Decoder 带 Self-Attention + Cross-Attention
2. Decoder 输入为「未来气象预报 + 时间编码」, 而非空白可学习 Query
3. 正弦位置编码正确应用 (叠加而非替换)
4. 非自回归并行预测, Decoder Self-Attention 无 causal mask

架构:
  Encoder: 历史24h (功率+气象+时间) -> Self-Attention -> memory
  Decoder: 未来4h  (气象+时间)      -> Self-Attn + Cross-Attn -> 功率预测
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """x: [B, seq_len, d_model]"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SolarTransformer(nn.Module):
    """
    太阳能电站功率预测 - 标准 Encoder-Decoder Transformer

    Encoder 输入: 历史 24h 的 功率 + 气象 + 时间特征 [B, 96, 13]
    Decoder 输入: 未来 4h 的 气象预报 + 时间特征      [B, 16, 12]
    输出:         未来 4h 的 功率预测                  [B, 16]
    """

    def __init__(self,
                 enc_feat_size=13,
                 dec_feat_size=12,
                 d_model=256,
                 nhead=8,
                 num_encoder_layers=4,
                 num_decoder_layers=4,
                 dim_feedforward=1024,
                 dropout=0.1,
                 enc_seq_len=96,
                 dec_seq_len=16):
        super().__init__()

        self.d_model = d_model

        # --- 输入投影 ---
        self.enc_projection = nn.Sequential(
            nn.Linear(enc_feat_size, d_model),
            nn.LayerNorm(d_model),
        )
        self.dec_projection = nn.Sequential(
            nn.Linear(dec_feat_size, d_model),
            nn.LayerNorm(d_model),
        )

        # --- 位置编码 ---
        self.enc_pos = PositionalEncoding(d_model, max_len=enc_seq_len + 10, dropout=dropout)
        self.dec_pos = PositionalEncoding(d_model, max_len=dec_seq_len + 10, dropout=dropout)

        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        # --- Transformer Decoder ---
        # Self-Attention (未来时间步之间交互) + Cross-Attention (关注历史)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        # --- 输出投影 ---
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x_enc, x_dec):
        """
        Args:
            x_enc: [B, enc_seq_len, enc_feat_size]  历史序列 (功率+气象+时间)
            x_dec: [B, dec_seq_len, dec_feat_size]  未来序列 (气象+时间, 无功率)
        Returns:
            [B, dec_seq_len]  功率预测
        """
        # Encoder: 投影 + 位置编码 + Self-Attention
        enc = self.enc_projection(x_enc)   # [B, 96, d_model]
        enc = self.enc_pos(enc)
        memory = self.encoder(enc)         # [B, 96, d_model]

        # Decoder: 投影 + 位置编码 + Self-Attn + Cross-Attn
        # 非自回归: 不使用 causal mask, 所有未来时间步可以互相 attend
        dec = self.dec_projection(x_dec)   # [B, 16, d_model]
        dec = self.dec_pos(dec)
        out = self.decoder(dec, memory)    # [B, 16, d_model]

        # 输出投影
        pred = self.output_projection(out) # [B, 16, 1]
        return pred.squeeze(-1)            # [B, 16]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    model = SolarTransformer(
        enc_feat_size=13,
        dec_feat_size=12,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
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
