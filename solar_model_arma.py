#!/usr/bin/env python3
"""
太阳能电站功率预测 - 神经 ARMA 模型 (AutoRegressive Moving Average)

经典 ARMA(p,q) 形式:
  X_t = c + φ₁X_{t-1} + ... + φ_pX_{t-p}   <- AR 分量
            + θ₁ε_{t-1} + ... + θ_qε_{t-q}   <- MA 分量
            + ε_t

本实现扩展为带外生气象特征的神经 ARMAX 多步预测模型, 三分量:
  1. AR 分量: ar_multi(past_power[-p:])     -> [B, horizon]  直接多步预测
  2. MA 分量: ma_layer(ar_residuals[-q:])   -> [B, horizon]  残差修正
  3. 气象分量: weather_mlp(hist+fut_weather) -> [B, horizon]  外生调整

MA 残差计算:
  用 ar_one_step (p → 1) 对历史最近 q 个时刻做单步 AR 预测,
  ε_t = X_t - AR_one_step(X_{t-p..t-1}), 向量化 unfold, 无 Python loop.

forward 接口与其他模型完全一致:
  forward(x_enc, x_dec) -> [B, horizon]

输入格式:
  x_enc: [B, 96, 13]  历史 (时间0:6 | 气象6:12 | 功率12)
  x_dec: [B, 16, 12]  未来 (时间0:6 | 气象6:12, 无功率)
输出:
  [B, 16]  未来功率预测 (归一化)
"""

import torch
import torch.nn as nn


# ============== 气象 MLP ==============

class WeatherMLP(nn.Module):
    """
    处理历史+未来气象特征, 输出 horizon 维气象调整量.

    将历史和未来气象拼接展平后过三层 MLP.
    """

    def __init__(self, hist_len: int, fut_len: int, n_weather: int,
                 d_hidden: int, horizon: int, dropout: float = 0.1):
        super().__init__()
        input_dim = (hist_len + fut_len) * n_weather
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, horizon),
        )

    def forward(self, hist_weather: torch.Tensor,
                fut_weather: torch.Tensor) -> torch.Tensor:
        """
        hist_weather: [B, hist_len, n_weather]
        fut_weather:  [B, fut_len,  n_weather]
        returns:      [B, horizon]
        """
        B = hist_weather.shape[0]
        x = torch.cat([
            hist_weather.reshape(B, -1),
            fut_weather.reshape(B, -1),
        ], dim=-1)
        return self.net(x)


# ============== 神经 ARMA 模型 ==============

class ARMAModel(nn.Module):
    """
    神经 ARMA 模型 for 太阳能功率多步预测.

    参数:
        p        : AR 阶数, 使用最近 p 个历史功率值
        q        : MA 阶数, 使用最近 q 个 AR 单步残差
        horizon  : 预测步数 (默认 16, 即 4h / 15min)
        n_weather: 气象特征数量 (默认 6)
        d_hidden : 气象 MLP 隐层维度
        hist_len : encoder 历史序列长度 (默认 96)
        fut_len  : decoder 未来序列长度 (默认 16)
        dropout  : Dropout 比率

    约束:
        p <= hist_len
        q + p <= hist_len  (滑动窗口需要足够的历史数据)
    """

    def __init__(self,
                 p: int = 72,
                 q: int = 24,
                 horizon: int = 16,
                 n_weather: int = 6,
                 d_hidden: int = 128,
                 hist_len: int = 96,
                 fut_len: int = 16,
                 dropout: float = 0.1):
        super().__init__()

        assert p <= hist_len, \
            f"AR 阶 p={p} 不能超过历史长度 hist_len={hist_len}"

        # MA 残差需要在历史窗口内滑动，自动裁剪 q 确保 q+p <= hist_len
        effective_q = min(q, hist_len - p)
        if effective_q < q:
            import warnings
            warnings.warn(
                f"MA 阶 q={q} 超出可用范围 (hist_len={hist_len} - p={p}={hist_len-p}), "
                f"自动裁剪为 q={effective_q}"
            )

        self.p = p
        self.q = effective_q
        self.horizon = horizon

        # --- AR 单步线性层 (用于计算 MA 所需的历史残差) ---
        # 输入: [B, q, p] (q 个滑动窗口)  输出: [B, q, 1]
        self.ar_one_step = nn.Linear(p, 1, bias=True)

        # --- AR 多步线性层 (直接预测未来 horizon 步) ---
        # 输入: [B, p]  输出: [B, horizon]
        self.ar_multi = nn.Linear(p, horizon, bias=True)

        # --- MA 残差修正层 ---
        # 输入: [B, q]  输出: [B, horizon]
        self.ma_layer = nn.Linear(q, horizon, bias=True)

        # --- 气象外生变量 MLP ---
        self.weather_mlp = WeatherMLP(
            hist_len=hist_len,
            fut_len=fut_len,
            n_weather=n_weather,
            d_hidden=d_hidden,
            horizon=horizon,
            dropout=dropout,
        )

        # --- 三分量融合权重 (可学习, softmax 归一化) ---
        # α[0]*AR + α[1]*MA + α[2]*Weather
        self.log_alpha = nn.Parameter(torch.zeros(3))

        # 初始化 AR 权重 (近期更重要)
        self._init_ar_weights()

    def _init_ar_weights(self):
        """用指数衰减初始化 AR 权重: 最近时刻权重最大."""
        with torch.no_grad():
            # decay[i] 随 i 增大而增大 (索引越大 = 时间越近)
            decay = torch.exp(torch.linspace(-2.0, 0.0, self.p))
            decay = decay / decay.sum()

            # ar_one_step: [1, p]
            self.ar_one_step.weight.data = decay.unsqueeze(0)
            self.ar_one_step.bias.data.zero_()

            # ar_multi: [horizon, p]
            self.ar_multi.weight.data = decay.unsqueeze(0).repeat(self.horizon, 1)
            self.ar_multi.bias.data.zero_()

    def _compute_ar_residuals(self, past_power: torch.Tensor) -> torch.Tensor:
        """
        向量化计算最近 q 步的 AR 单步残差 (无 Python loop).

        对位置 t ∈ {L-q, ..., L-1}:
          ε_t = X_t - ar_one_step(X_{t-p : t})

        实现: 用 unfold 在 past_power 上生成滑动窗口矩阵,
        再批量通过 ar_one_step.

        past_power: [B, L]
        returns:    [B, q]
        """
        L = past_power.shape[1]

        # 取出需要的片段: 长度 = q + p - 1
        # unfold(size=p, step=1) 产生窗口数 = (q+p-1 - p)/1 + 1 = q  ✓
        # 片段覆盖位置 [L-q-p, ..., L-2]，每个窗口对应一个预测时刻
        segment = past_power[:, L - self.q - self.p : L - 1]  # [B, q+p-1]

        # 滑动窗口: 步长1, 窗口大小p -> [B, q, p]
        windows = segment.unfold(dimension=1, size=self.p, step=1)  # [B, q, p]

        # AR 单步预测: [B, q, p] -> [B, q, 1] -> [B, q]
        ar_preds = self.ar_one_step(windows).squeeze(-1)  # [B, q]

        # 对应时刻的实际功率值: 位置 [L-q, ..., L-1]
        actuals = past_power[:, L - self.q:]  # [B, q]

        return actuals - ar_preds  # [B, q]

    def forward(self, x_enc: torch.Tensor, x_dec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_enc: [B, 96, 13]  历史序列 (时间0:6 | 气象6:12 | 功率12)
            x_dec: [B, 16, 12]  未来序列 (时间0:6 | 气象6:12, 无功率)
        Returns:
            [B, 16]  未来功率预测 (归一化值, 范围约 [0, 1])
        """
        # --- 特征提取 ---
        past_power   = x_enc[:, :, -1]      # [B, 96]  归一化历史功率
        hist_weather = x_enc[:, :, 6:12]    # [B, 96, 6]
        fut_weather  = x_dec[:, :, 6:12]    # [B, 16, 6]

        # --- AR 多步预测 ---
        ar_input = past_power[:, -self.p:]   # [B, p]
        ar_pred  = self.ar_multi(ar_input)   # [B, horizon]

        # --- MA 残差修正 ---
        residuals     = self._compute_ar_residuals(past_power)  # [B, q]
        ma_correction = self.ma_layer(residuals)                # [B, horizon]

        # --- 气象外生量 ---
        weather_adj = self.weather_mlp(hist_weather, fut_weather)  # [B, horizon]

        # --- 三分量加权融合 ---
        alpha  = torch.softmax(self.log_alpha, dim=0)          # [3], 和为 1
        output = (alpha[0] * ar_pred
                  + alpha[1] * ma_correction
                  + alpha[2] * weather_adj)                    # [B, horizon]
        return output


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    model = ARMAModel(
        p=96, q=24, horizon=16,
        n_weather=6, d_hidden=128,
        hist_len=96, fut_len=16,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params:,}")

    x_enc = torch.randn(4, 96, 13).to(device)
    x_dec = torch.randn(4, 16, 12).to(device)

    with torch.no_grad():
        out = model(x_enc, x_dec)

    print(f"Encoder 输入: {x_enc.shape}")
    print(f"Decoder 输入: {x_dec.shape}")
    print(f"输出:         {out.shape}")

    alpha = torch.softmax(model.log_alpha, dim=0)
    print(f"初始分量权重: AR={alpha[0]:.3f}, MA={alpha[1]:.3f}, 气象={alpha[2]:.3f}")
