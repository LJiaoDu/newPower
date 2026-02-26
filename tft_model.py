#!/usr/bin/env python3
"""
太阳能电站功率预测 - Temporal Fusion Transformer (TFT)

论文: "Temporal Fusion Transformers for Interpretable
       Multi-horizon Time Series Forecasting" (Lim et al., 2021)

架构核心组件:
  1. GatedResidualNetwork (GRN): 门控残差网络, 所有模块的基础构件
  2. VariableSelectionNetwork (VSN): 变量选择网络, 自动学习特征重要性
  3. TemporalFusionTransformer: 完整 TFT 模型

输入三类特征:
  x_static: [B, num_static_vars]           静态协变量 (装机容量)
  x_past:   [B, enc_len, num_past_vars]    历史序列  (气象+时间+功率)
  x_future: [B, dec_len, num_future_vars]  未来已知序列 (气象+时间)

输出:
  [B, dec_len, num_quantiles]  分位数预测 (默认 P10/P50/P90)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
#  1. Gated Residual Network (GRN)
# ============================================================

class GatedResidualNetwork(nn.Module):
    """
    GRN(a, c=None):
      η = ELU( W1*a + b1 + (Wc*c, 若提供 context) )
      GLU 门控: g = σ(Wg*η),  v = Wv*η
      out = LayerNorm( skip(a) + g ⊙ v )

    支持可选的静态上下文向量 c (用于条件化变量选择与丰富化)
    """

    def __init__(self, input_size: int, hidden_size: int,
                 output_size: int = None, context_size: int = None,
                 dropout: float = 0.1):
        super().__init__()
        self.output_size = output_size if output_size is not None else input_size

        self.dense1 = nn.Linear(input_size, hidden_size)
        self.context_proj = (
            nn.Linear(context_size, hidden_size, bias=False)
            if context_size is not None else None
        )
        self.dropout = nn.Dropout(dropout)

        # GLU: 输出 2 * output_size, 前半作值, 后半作门
        self.glu = nn.Linear(hidden_size, self.output_size * 2)

        # 跳跃连接维度对齐
        self.skip = (
            nn.Linear(input_size, self.output_size, bias=False)
            if input_size != self.output_size else nn.Identity()
        )
        self.norm = nn.LayerNorm(self.output_size)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """
        x:       [..., input_size]
        context: [B, context_size]  (自动 broadcast 到时间维)
        """
        h = self.dense1(x)
        if context is not None and self.context_proj is not None:
            c = self.context_proj(context)
            if x.dim() == 3 and c.dim() == 2:
                c = c.unsqueeze(1)          # [B, 1, hidden_size]
            h = h + c
        h = F.elu(h)
        h = self.dropout(h)

        glu_out = self.glu(h)
        val  = glu_out[..., :self.output_size]
        gate = torch.sigmoid(glu_out[..., self.output_size:])
        return self.norm(self.skip(x) + gate * val)


# ============================================================
#  2. Variable Selection Network (VSN)
# ============================================================

class VariableSelectionNetwork(nn.Module):
    """
    对每个标量输入变量:
      1. 用独立线性层嵌入到 d_model 空间
      2. 用独立 GRN 精炼
    再用一个联合 GRN 计算 softmax 权重, 加权求和得到 d_model 表示

    适用于时序 (temporal=True, x: [B,T,V]) 和
              静态 (temporal=False, x: [B,V]) 两种输入
    """

    def __init__(self, num_vars: int, d_model: int,
                 dropout: float = 0.1, context_size: int = None):
        super().__init__()
        self.num_vars = num_vars
        self.d_model  = d_model

        # 每个变量: scalar -> d_model
        self.var_linear = nn.ModuleList(
            [nn.Linear(1, d_model) for _ in range(num_vars)]
        )
        # 每个变量独立 GRN (可选接受静态上下文)
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(d_model, d_model,
                                 dropout=dropout, context_size=context_size)
            for _ in range(num_vars)
        ])
        # 权重 GRN: 拼接所有嵌入 -> num_vars 维 softmax 权重
        self.weight_grn = GatedResidualNetwork(
            input_size=num_vars * d_model,
            hidden_size=d_model,
            output_size=num_vars,
            dropout=dropout,
            context_size=context_size,
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor = None):
        """
        x: [B, T, num_vars] 或 [B, num_vars]
        返回: (output, weights)
          output:  [B, T, d_model] 或 [B, d_model]
          weights: [B, T, num_vars] 或 [B, num_vars]
        """
        is_static = (x.dim() == 2)
        if is_static:
            x = x.unsqueeze(1)              # [B, 1, num_vars]

        B, T, V = x.shape

        # 嵌入 & GRN 精炼
        embeds = torch.stack(
            [self.var_linear[i](x[..., i:i+1]) for i in range(V)],
            dim=2                           # [B, T, V, d_model]
        )
        grn_out = torch.stack(
            [self.var_grns[i](embeds[:, :, i, :], context=context)
             for i in range(V)],
            dim=2                           # [B, T, V, d_model]
        )

        # 权重计算
        flat = embeds.reshape(B, T, V * self.d_model)   # [B, T, V*d_model]
        weights = torch.softmax(
            self.weight_grn(flat, context=context), dim=-1
        )                                               # [B, T, V]

        # 加权求和
        output = (weights.unsqueeze(-1) * grn_out).sum(dim=2)  # [B, T, d_model]

        if is_static:
            return output.squeeze(1), weights.squeeze(1)
        return output, weights


# ============================================================
#  3. Temporal Fusion Transformer
# ============================================================

class TemporalFusionTransformer(nn.Module):
    """
    完整 TFT 模型

    数据流:
      静态协变量 ---> StaticVSN ---> 4个上下文向量 (cs, ce_h, ce_c, cd)
                                      |
      历史序列  ---> PastVSN (条件于 ce_h)  ---> LSTM Encoder (初始化于 ce_h/ce_c)
      未来序列  ---> FutureVSN (条件于 cd)  ---> LSTM Decoder (续接 Encoder 隐状态)
                                      |
                           拼接时序 [enc+dec]
                                      |
                         静态丰富化 GRN (条件于 cs)
                                      |
                         多头自注意力 (含 skip)
                                      |
                         逐位置 GRN Feed-forward (含 skip)
                                      |
                       取 Decoder 部分 --> 分位数输出
    """

    def __init__(self,
                 num_past_vars:   int = 13,
                 num_future_vars: int = 12,
                 num_static_vars: int = 1,
                 d_model:         int = 128,
                 num_heads:       int = 4,
                 num_lstm_layers: int = 1,
                 dropout:         float = 0.1,
                 quantiles: tuple = (0.1, 0.5, 0.9)):
        super().__init__()
        self.d_model       = d_model
        self.quantiles     = quantiles
        self.num_quantiles = len(quantiles)

        # ---- 静态变量选择 ----
        self.static_vsn = VariableSelectionNetwork(
            num_vars=num_static_vars, d_model=d_model, dropout=dropout
        )
        # 4 个静态上下文向量:
        #   cs  -> 静态丰富化
        #   ce_h / ce_c -> LSTM Encoder 初始隐/细胞状态
        #   cd  -> Future VSN 上下文
        self.ctx_grns = nn.ModuleList([
            GatedResidualNetwork(d_model, d_model, dropout=dropout)
            for _ in range(4)
        ])

        # ---- 历史变量选择 ----
        self.past_vsn = VariableSelectionNetwork(
            num_vars=num_past_vars, d_model=d_model,
            dropout=dropout, context_size=d_model
        )

        # ---- 未来变量选择 ----
        self.future_vsn = VariableSelectionNetwork(
            num_vars=num_future_vars, d_model=d_model,
            dropout=dropout, context_size=d_model
        )

        # ---- LSTM 编码器 / 解码器 ----
        lstm_kwargs = dict(
            input_size=d_model, hidden_size=d_model,
            num_layers=num_lstm_layers, batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )
        self.encoder_lstm = nn.LSTM(**lstm_kwargs)
        self.decoder_lstm = nn.LSTM(**lstm_kwargs)

        # LSTM 输出后的 GLU 门 + 残差 LayerNorm
        self.lstm_glu  = nn.Linear(d_model, d_model * 2)
        self.lstm_norm = nn.LayerNorm(d_model)

        # ---- 静态丰富化 ----
        self.static_enrich = GatedResidualNetwork(
            d_model, d_model, dropout=dropout, context_size=d_model
        )

        # ---- 多头自注意力 ----
        self.attn      = nn.MultiheadAttention(d_model, num_heads,
                                               dropout=dropout, batch_first=True)
        self.attn_glu  = nn.Linear(d_model, d_model * 2)
        self.attn_norm = nn.LayerNorm(d_model)

        # ---- 逐位置 Feed-forward (GRN) ----
        self.pos_ff    = GatedResidualNetwork(d_model, d_model * 4,
                                              output_size=d_model, dropout=dropout)
        self.ff_glu    = nn.Linear(d_model, d_model * 2)
        self.ff_norm   = nn.LayerNorm(d_model)

        # ---- 分位数输出头 ----
        self.output_proj = nn.Linear(d_model, self.num_quantiles)

    # ----------------------------------------------------------
    def _glu_add_norm(self, x: torch.Tensor, residual: torch.Tensor,
                      glu_layer: nn.Linear, norm_layer: nn.LayerNorm) -> torch.Tensor:
        """GLU 门控 + 残差 + LayerNorm (在多个位置复用)"""
        glu = glu_layer(x)
        val  = glu[..., :self.d_model]
        gate = torch.sigmoid(glu[..., self.d_model:])
        return norm_layer(residual + gate * val)

    # ----------------------------------------------------------
    def forward(self, x_past:   torch.Tensor,
                      x_future: torch.Tensor,
                      x_static: torch.Tensor) -> torch.Tensor:
        """
        参数:
          x_past:   [B, enc_len, num_past_vars]
          x_future: [B, dec_len, num_future_vars]
          x_static: [B, num_static_vars]
        返回:
          [B, dec_len, num_quantiles]   P10 / P50 / P90
        """
        B, enc_len, _ = x_past.shape
        dec_len = x_future.shape[1]

        # ===== Step 1: 静态变量选择 & 上下文向量生成 =====
        static_emb, _ = self.static_vsn(x_static)          # [B, d_model]
        cs   = self.ctx_grns[0](static_emb)                 # 静态丰富化上下文
        ce_h = self.ctx_grns[1](static_emb)                 # Encoder LSTM init hidden
        ce_c = self.ctx_grns[2](static_emb)                 # Encoder LSTM init cell
        cd   = self.ctx_grns[3](static_emb)                 # Future VSN 上下文

        # ===== Step 2: 时序变量选择 =====
        past_sel,   _ = self.past_vsn(x_past,   context=ce_h)   # [B, enc_len, d_model]
        future_sel, _ = self.future_vsn(x_future, context=cd)    # [B, dec_len, d_model]

        # ===== Step 3: LSTM 序列编码 =====
        h0 = ce_h.unsqueeze(0)   # [1, B, d_model]
        c0 = ce_c.unsqueeze(0)   # [1, B, d_model]
        enc_out, (hn, cn) = self.encoder_lstm(past_sel, (h0, c0))
        dec_out, _        = self.decoder_lstm(future_sel, (hn, cn))

        # 拼接 enc + dec -> [B, enc_len+dec_len, d_model]
        temporal = torch.cat([enc_out, dec_out], dim=1)
        inputs   = torch.cat([past_sel, future_sel], dim=1)  # 残差用

        # LSTM 后 GLU 残差
        temporal = self._glu_add_norm(temporal, inputs, self.lstm_glu, self.lstm_norm)

        # ===== Step 4: 静态丰富化 =====
        enriched = self.static_enrich(temporal, context=cs)     # [B, T, d_model]

        # ===== Step 5: 多头自注意力 =====
        attn_out, _ = self.attn(enriched, enriched, enriched)   # [B, T, d_model]
        attn_out = self._glu_add_norm(attn_out, enriched, self.attn_glu, self.attn_norm)

        # ===== Step 6: 逐位置 Feed-forward =====
        ff_out = self.pos_ff(attn_out)                          # [B, T, d_model]
        ff_out = self._glu_add_norm(ff_out, attn_out, self.ff_glu, self.ff_norm)

        # ===== Step 7: 取 Decoder 部分 -> 分位数输出 =====
        dec_repr = ff_out[:, enc_len:, :]                        # [B, dec_len, d_model]
        return self.output_proj(dec_repr)                        # [B, dec_len, num_quantiles]


# ============================================================
#  快速验证
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    model = TemporalFusionTransformer(
        num_past_vars=13,
        num_future_vars=12,
        num_static_vars=1,
        d_model=128,
        num_heads=4,
        num_lstm_layers=1,
        dropout=0.1,
        quantiles=(0.1, 0.5, 0.9),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params:,}")

    B, enc_len, dec_len = 4, 96, 16
    x_past   = torch.randn(B, enc_len, 13).to(device)
    x_future = torch.randn(B, dec_len, 12).to(device)
    x_static = torch.randn(B, 1).to(device)

    with torch.no_grad():
        out = model(x_past, x_future, x_static)

    print(f"x_past:   {x_past.shape}")
    print(f"x_future: {x_future.shape}")
    print(f"x_static: {x_static.shape}")
    print(f"输出:      {out.shape}  (应为 [4, 16, 3])")
