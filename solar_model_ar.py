#!/usr/bin/env python3
"""
太阳能电站功率预测 - 自回归 Transformer (Autoregressive)

核心思路:
  训练: 输入 96 步 [B, 96, 13] → 预测下一步 [B, 1]
  推理: 滑动窗口逐步预测 16 步
    step 1: input[0:96]  → pred[96]
    step 2: input[1:97]  → pred[97]   (位置96 = 预测值 + 已知未来气象)
    step 3: input[2:98]  → pred[98]
    ...
    step16: input[15:111] → pred[111]

优势:
  - 每步预测都基于最新的上下文 (含之前的预测结果)
  - 训练目标简单 (单步预测), 模型更容易学习
  - Encoder-only 结构, 比 Encoder-Decoder 更轻量

劣势:
  - 推理时有误差累积 (前面的预测误差会传播到后面)
  - 推理速度较慢 (需要串行执行 16 次前向传播)
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
    自回归 Transformer: 输入 96 步, 预测下一步

    训练输入: [B, 96, 13]  (历史: 功率 + 气象 + 时间)
    训练输出: [B]           (下一步功率)

    推理时调用 predict_sequence() 逐步滚动预测 16 步
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

        # 输出头: 取最后一个位置的隐状态 → 预测 1 个值
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        """
        单步预测 (训练用)
        Args:
            x: [B, seq_len, feat_size]
        Returns:
            [B] 下一步功率预测
        """
        h = self.projection(x)  # [B, 96, d_model]
        h = self.pos_enc(h)
        h = self.encoder(h)  # [B, 96, d_model]

        last = h[:, -1, :]  # [B, d_model]  取最后位置
        out = self.output_head(last)  # [B, 1]
        return out.squeeze(-1)  # [B]

    @torch.no_grad()
    def predict_sequence(self, x_hist, future_covariates, out_steps=16):
        """
        自回归滑动窗口推理: 逐步预测 out_steps 步

        Args:
            x_hist:              [B, seq_len, 13]  初始历史数据
            future_covariates:   [B, out_steps, 12] 未来气象 + 时间特征
            out_steps:           预测步数 (默认 16 = 4小时)

        Returns:
            [B, out_steps] 预测功率序列

        推理过程:
            窗口 [0:96]  → pred_96
            窗口 [1:97]  → pred_97  (位置96 = [future_cov[0], pred_96])
            窗口 [2:98]  → pred_98  (位置97 = [future_cov[1], pred_97])
            ...
        """
        self.eval()
        predictions = []
        current = x_hist.clone()  # [B, seq_len, 13]

        for t in range(out_steps):
            # 预测下一步
            pred = self.forward(current)  # [B]
            predictions.append(pred)

            if t < out_steps - 1:
                # 构造新的一步: [未来气象+时间(12维), 预测功率(1维)] = 13维
                new_step = torch.cat(
                    [
                        future_covariates[:, t, :],  # [B, 12]
                        pred.unsqueeze(-1),  # [B, 1]
                    ],
                    dim=-1,
                )  # [B, 13]

                # 滑动窗口: 丢弃最早的一步, 追加新的一步
                current = torch.cat(
                    [
                        current[:, 1:, :],  # [B, 95, 13]
                        new_step.unsqueeze(1),  # [B, 1, 13]
                    ],
                    dim=1,
                )  # [B, 96, 13]

        return torch.stack(predictions, dim=1)  # [B, out_steps]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    model = SolarTransformerAR(
        feat_size=13,
        d_model=256,
        nhead=8,
        num_layers=6,
        seq_len=96,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params:,}")

    # 测试单步预测 (训练模式)
    x = torch.randn(4, 96, 13).to(device)
    with torch.no_grad():
        out = model(x)
    print(f"单步预测:  输入 {x.shape} → 输出 {out.shape}")

    # 测试自回归推理 (16步)
    future_cov = torch.randn(4, 16, 12).to(device)
    seq = model.predict_sequence(x, future_cov, out_steps=16)
    print(f"自回归推理: 输入 {x.shape} + 未来协变量 {future_cov.shape} → 输出 {seq.shape}")
