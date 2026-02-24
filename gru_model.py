#!/usr/bin/env python3
"""
太阳能电站功率预测 - GRU Encoder-Decoder 模型

架构:
  Encoder GRU: 历史24h (功率+气象+时间) -> hidden state
  Decoder GRU: 未来4h  (气象+时间)      -> 功率预测
"""

import torch
import torch.nn as nn


class SolarGRU(nn.Module):
    """
    GRU Encoder-Decoder 太阳能功率预测模型

    Encoder 输入: 历史 24h [B, 96, 13]  (时间6 + 气象6 + 功率1)
    Decoder 输入: 未来 4h  [B, 16, 12]  (时间6 + 气象6, 无功率)
    输出:         未来 4h 功率预测       [B, 16]
    """

    def __init__(
        self,
        enc_feat_size: int = 13,
        dec_feat_size: int = 12,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Encoder GRU: 处理历史序列, 输出 hidden state
        self.encoder_gru = nn.GRU(
            input_size=enc_feat_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Decoder GRU: 利用 encoder hidden state 处理未来特征
        self.decoder_gru = nn.GRU(
            input_size=dec_feat_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 输出投影: hidden -> 单步功率预测
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x_enc: torch.Tensor, x_dec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_enc: [B, enc_seq_len, enc_feat_size]  历史序列 (功率+气象+时间)
            x_dec: [B, dec_seq_len, dec_feat_size]  未来序列 (气象+时间, 无功率)
        Returns:
            [B, dec_seq_len]  功率预测
        """
        # 编码历史序列: 只取最终 hidden state
        _, hidden = self.encoder_gru(x_enc)  # hidden: [num_layers, B, hidden_size]

        # 解码未来序列: 以 encoder hidden state 初始化
        dec_out, _ = self.decoder_gru(x_dec, hidden)  # [B, dec_seq_len, hidden_size]

        # 投影到功率预测
        pred = self.output_proj(dec_out)  # [B, dec_seq_len, 1]
        return pred.squeeze(-1)           # [B, dec_seq_len]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    model = SolarGRU(
        enc_feat_size=13,
        dec_feat_size=12,
        hidden_size=256,
        num_layers=2,
        dropout=0.2,
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
    assert out.shape == (4, 16), f"期望 (4, 16), 实际 {out.shape}"
    print("形状验证通过")
