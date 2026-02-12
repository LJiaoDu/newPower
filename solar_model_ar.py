#!/usr/bin/env python3
"""
太阳能电站功率预测 - 自回归 Transformer (Autoregressive)

训练 & 推理都输出 16 步:
  训练 (Teacher Forcing 滑动窗口):
    step 1:  x_full[0:96]   → pred[96]     (窗口内全是真实值)
    step 2:  x_full[1:97]   → pred[97]
    step 3:  x_full[2:98]   → pred[98]
    ...
    step16:  x_full[15:111]  → pred[111]
    loss = HuberLoss(pred[0:16], y[0:16])

  推理 (自回归滑动窗口):
    step 1:  input[0:96]   → pred_96
    step 2:  input[1:97]   → pred_97     (位置96 = [已知气象, pred_96])
    step 3:  input[2:98]   → pred_98     (位置97 = [已知气象, pred_97])
    ...
    step16:  input[15:111] → pred_111
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
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class SolarTransformerAR(nn.Module):
    """
    自回归 Transformer

    单步前向: [B, 96, 13] → [B]
    训练: forward_sequence()   Teacher Forcing 滑动16步 → [B, 16]
    推理: predict_sequence()   自回归滑动16步            → [B, 16]
    """

    def __init__(
        self,
        feat_size=13,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        seq_len=96,
    ):
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len

        # 输入投影
        self.projection = nn.Sequential(
            nn.Linear(feat_size, d_model),
            nn.LayerNorm(d_model),
        )

        # 位置编码
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 10, dropout=dropout)

        # Transformer Encoder (Self-Attention)
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
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # 输出头: 取最后位置 → 预测 1 个值
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def _forward_one_step(self, x):
        """
        单步前向: [B, seq_len, feat_size] → [B]
        """
        h = self.projection(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        last = h[:, -1, :]
        return self.output_head(last).squeeze(-1)

    def forward(self, x_full, out_steps=16):
        """
        训练用: Teacher Forcing 滑动窗口, 输出 16 步预测

        Args:
            x_full:    [B, seq_len + out_steps, 13]  完整序列 (112步, 含真实功率)
            out_steps: 预测步数 (默认16)

        Returns:
            [B, out_steps]  16个预测值

        过程:
            step 0:  x_full[:, 0:96,  :] → pred[0]
            step 1:  x_full[:, 1:97,  :] → pred[1]   (位置96是真实值, teacher forcing)
            ...
            step 15: x_full[:, 15:111, :] → pred[15]
        """
        predictions = []
        for t in range(out_steps):
            window = x_full[:, t : t + self.seq_len, :]  # [B, 96, 13]
            pred = self._forward_one_step(window)  # [B]
            predictions.append(pred)
        return torch.stack(predictions, dim=1)  # [B, out_steps]

    @torch.no_grad()
    def predict_sequence(self, x_hist, future_covariates, out_steps=16):
        """
        推理用: 自回归滑动窗口, 用自己的预测值填充

        Args:
            x_hist:            [B, seq_len, 13]   初始历史数据
            future_covariates: [B, out_steps, 12]  未来气象 + 时间
            out_steps:         预测步数

        Returns:
            [B, out_steps]

        过程:
            step 0: input[0:96]   → pred_0
            step 1: input[1:97]   → pred_1  (位置96 = [future_cov[0], pred_0])
            ...
        """
        self.eval()
        predictions = []
        current = x_hist.clone()  # [B, seq_len, 13]

        for t in range(out_steps):
            pred = self._forward_one_step(current)  # [B]
            predictions.append(pred)

            if t < out_steps - 1:
                # 构造新步: [气象+时间(12维), 预测功率(1维)] = 13维
                new_step = torch.cat(
                    [future_covariates[:, t, :], pred.unsqueeze(-1)],
                    dim=-1,
                ).unsqueeze(1)  # [B, 1, 13]

                # 滑动: 丢弃最早一步, 追加新步
                current = torch.cat([current[:, 1:, :], new_step], dim=1)

        return torch.stack(predictions, dim=1)  # [B, out_steps]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    model = SolarTransformerAR(feat_size=13, d_model=256, nhead=8,
                                num_layers=6, seq_len=96).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params:,}")

    # 测试训练模式: Teacher Forcing 16步
    x_full = torch.randn(4, 112, 13).to(device)
    out = model(x_full, out_steps=16)
    print(f"训练 (TF):  输入 {x_full.shape} → 输出 {out.shape}")

    # 测试推理模式: 自回归 16步
    x_hist = torch.randn(4, 96, 13).to(device)
    future = torch.randn(4, 16, 12).to(device)
    seq = model.predict_sequence(x_hist, future, out_steps=16)
    print(f"推理 (AR):  输入 {x_hist.shape} + {future.shape} → 输出 {seq.shape}")
