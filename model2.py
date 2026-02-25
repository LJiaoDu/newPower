#!/usr/bin/env python3
"""
太阳能电站功率预测 - Encoder-Decoder Transformer + RoPE 位置编码

基于 model1.py 改进, 将正弦加性位置编码替换为 RoPE (Rotary Position Encoding):
  - 正弦 PE (model1.py): 位置向量叠加到 token embedding 上 (加性)
  - RoPE  (model2.py) : 位置信息直接旋转 Q/K 向量, V 不变 (乘性)

RoPE 的优势:
  1. 相对位置信息更自然: attention(q_i, k_j) 只依赖 (i-j) 的相对距离
  2. 外推性更好: 训练时未见过的位置偏移也有合理的几何解释
  3. Cross-Attention 中 Q 使用续接位置 [96..111], K 使用 [0..95],
     attention 权重天然编码了"预测步骤与历史时刻的时间距离"

架构:
  Encoder: 历史 24h [B, 96, enc_feat]  -> RoPE Self-Attn -> memory
  Decoder: 未来  4h [B, 16, dec_feat]  -> RoPE Self-Attn + Cross-Attn -> 功率预测

输入特征维度 (配合 process2.py):
  enc_feat_size = 16  (时间6 + 气象6 + TSI衍生3 + 功率1)
  dec_feat_size = 15  (时间6 + 气象6 + TSI衍生3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# ============================================================
# RoPE 核心实现
# ============================================================

class RotaryPositionEncoding(nn.Module):
    """
    预计算 RoPE 的 cos/sin 缓存表

    head_dim 必须为偶数, 每 2 维构成一个旋转对
    支持 offset 参数: cross-attention 中 decoder Q 使用续接位置
    """

    def __init__(self, head_dim: int, max_len: int = 200, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, f"head_dim 必须为偶数, 实际: {head_dim}"

        # inv_freq: [head_dim/2]
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_len)

    def _build_cache(self, max_len: int):
        pos = torch.arange(0, max_len, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq)           # [max_len, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)           # [max_len, head_dim]
        # 广播形状: [1, 1, max_len, head_dim]
        cos_cached = emb.cos().unsqueeze(0).unsqueeze(0)
        sin_cached = emb.sin().unsqueeze(0).unsqueeze(0)
        self.register_buffer("cos_cached", cos_cached)
        self.register_buffer("sin_cached", sin_cached)

    def forward(self, seq_len: int, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回 [1, 1, seq_len, head_dim] 的 cos/sin

        Args:
            seq_len: 目标序列长度
            offset:  位置起始偏移 (cross-attention Q 用 offset=enc_seq_len)
        """
        return (
            self.cos_cached[:, :, offset: offset + seq_len, :],
            self.sin_cached[:, :, offset: offset + seq_len, :],
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """RoPE 旋转: 将向量后半与前半互换, 前半取反"""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor,
    cos_q: torch.Tensor, sin_q: torch.Tensor,
    cos_k: torch.Tensor, sin_k: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对 Q 和 K 分别应用 RoPE (V 不旋转)

    q: [B, nhead, q_len, head_dim]
    k: [B, nhead, k_len, head_dim]
    cos/sin: [1, 1, seq_len, head_dim]
    """
    q_rot = q * cos_q + rotate_half(q) * sin_q
    k_rot = k * cos_k + rotate_half(k) * sin_k
    return q_rot, k_rot


# ============================================================
# 支持 RoPE 的多头注意力
# ============================================================

class RoPEMultiheadAttention(nn.Module):
    """
    多头注意力: Q/K 投影后旋转 RoPE, V 不旋转
    参数结构与 nn.MultiheadAttention 相同
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,        # [B, q_len, d_model]
        key:   torch.Tensor,        # [B, k_len, d_model]
        value: torch.Tensor,        # [B, k_len, d_model]
        cos_q: torch.Tensor,        # [1, 1, q_len, head_dim]
        sin_q: torch.Tensor,
        cos_k: torch.Tensor,        # [1, 1, k_len, head_dim]
        sin_k: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, q_len, d = query.shape
        k_len = key.shape[1]

        def split_heads(x, length):
            return x.view(B, length, self.nhead, self.head_dim).transpose(1, 2)

        Q = split_heads(self.q_proj(query), q_len)   # [B, h, q_len, hd]
        K = split_heads(self.k_proj(key),   k_len)   # [B, h, k_len, hd]
        V = split_heads(self.v_proj(value), k_len)

        Q, K = apply_rotary_pos_emb(Q, K, cos_q, sin_q, cos_k, sin_k)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, h, q_len, k_len]
        if attn_mask is not None:
            scores = scores + attn_mask
        attn = self.attn_drop(F.softmax(scores, dim=-1))

        out = torch.matmul(attn, V)                   # [B, h, q_len, hd]
        out = out.transpose(1, 2).contiguous().view(B, q_len, d)
        return self.out_proj(out)


# ============================================================
# Encoder / Decoder Layer
# ============================================================

class RoPEEncoderLayer(nn.Module):
    """Transformer Encoder Layer 带 RoPE (Post-LN)"""

    def __init__(self, d_model: int, nhead: int,
                 dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.self_attn = RoPEMultiheadAttention(d_model, nhead, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # Self-Attention + Residual + LayerNorm
        src = self.norm1(src + self.drop1(
            self.self_attn(src, src, src, cos, sin, cos, sin)
        ))
        # Feedforward + Residual + LayerNorm
        src = self.norm2(src + self.drop2(self.ff(src)))
        return src


class RoPEDecoderLayer(nn.Module):
    """Transformer Decoder Layer 带 RoPE (Post-LN)"""

    def __init__(self, d_model: int, nhead: int,
                 dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.self_attn  = RoPEMultiheadAttention(d_model, nhead, dropout)
        self.cross_attn = RoPEMultiheadAttention(d_model, nhead, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt:         torch.Tensor,   # [B, dec_len, d_model]
        memory:      torch.Tensor,   # [B, enc_len, d_model]
        cos_self:    torch.Tensor,   # decoder 局部位置 [0..dec-1]
        sin_self:    torch.Tensor,
        cos_cross_q: torch.Tensor,   # decoder 绝对位置 [enc..enc+dec-1]
        sin_cross_q: torch.Tensor,
        cos_cross_k: torch.Tensor,   # encoder 位置 [0..enc-1]
        sin_cross_k: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Self-Attention (decoder 内部)
        tgt = self.norm1(tgt + self.drop1(
            self.self_attn(tgt, tgt, tgt, cos_self, sin_self, cos_self, sin_self)
        ))
        # 2. Cross-Attention (Q 续接位置 → 编码时间距离)
        tgt = self.norm2(tgt + self.drop2(
            self.cross_attn(tgt, memory, memory,
                            cos_cross_q, sin_cross_q,
                            cos_cross_k, sin_cross_k)
        ))
        # 3. Feedforward
        tgt = self.norm3(tgt + self.drop3(self.ff(tgt)))
        return tgt


# ============================================================
# 顶层模型: SolarTransformerRoPE
# ============================================================

class SolarTransformerRoPE(nn.Module):
    """
    太阳能电站功率预测 Encoder-Decoder Transformer + RoPE

    接口与 model1.py 的 SolarTransformer 完全一致:
        forward(x_enc, x_dec) -> [B, dec_seq_len]

    默认输入维度 (配合 process2.py):
        enc_feat_size = 16  (时间6 + 气象6 + tsi_diff1/2/std_4 + 功率1)
        dec_feat_size = 15  (时间6 + 气象6 + tsi_diff1/2/std_4)
    """

    def __init__(
        self,
        enc_feat_size: int = 16,
        dec_feat_size: int = 15,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        enc_seq_len: int = 96,
        dec_seq_len: int = 16,
        rope_base: float = 10000.0,
    ):
        super().__init__()

        self.enc_seq_len = enc_seq_len
        self.dec_seq_len = dec_seq_len
        head_dim = d_model // nhead

        # --- 输入投影 (Linear + LayerNorm) ---
        self.enc_projection = nn.Sequential(
            nn.Linear(enc_feat_size, d_model),
            nn.LayerNorm(d_model),
        )
        self.dec_projection = nn.Sequential(
            nn.Linear(dec_feat_size, d_model),
            nn.LayerNorm(d_model),
        )

        # --- RoPE (共享一张频率表, 覆盖 encoder + decoder 全部位置) ---
        self.rope = RotaryPositionEncoding(
            head_dim=head_dim,
            max_len=enc_seq_len + dec_seq_len + 10,
            base=rope_base,
        )

        # --- Encoder ---
        self.encoder_layers = nn.ModuleList([
            RoPEEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # --- Decoder ---
        self.decoder_layers = nn.ModuleList([
            RoPEDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)

        # --- 输出投影 ---
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x_enc: torch.Tensor, x_dec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_enc: [B, enc_seq_len, enc_feat_size]
            x_dec: [B, dec_seq_len, dec_feat_size]
        Returns:
            [B, dec_seq_len]
        """
        enc_len = x_enc.size(1)
        dec_len = x_dec.size(1)

        # 预计算 RoPE cos/sin
        enc_cos, enc_sin = self.rope(enc_len, offset=0)

        # Decoder Self-Attn 用局部位置 [0..dec-1]
        dec_self_cos, dec_self_sin = self.rope(dec_len, offset=0)

        # Decoder Cross-Attn Q 用续接位置 [enc..enc+dec-1]
        # → attention(q_i, k_j) 天然感知"预测步 i 距历史步 j 的时间距离"
        dec_cross_q_cos, dec_cross_q_sin = self.rope(dec_len, offset=enc_len)
        dec_cross_k_cos, dec_cross_k_sin = enc_cos, enc_sin  # [0..enc-1]

        # Encoder
        enc = self.enc_projection(x_enc)
        for layer in self.encoder_layers:
            enc = layer(enc, enc_cos, enc_sin)
        memory = self.encoder_norm(enc)

        # Decoder
        dec = self.dec_projection(x_dec)
        for layer in self.decoder_layers:
            dec = layer(
                dec, memory,
                dec_self_cos, dec_self_sin,
                dec_cross_q_cos, dec_cross_q_sin,
                dec_cross_k_cos, dec_cross_k_sin,
            )
        out = self.decoder_norm(dec)

        pred = self.output_projection(out)   # [B, dec_len, 1]
        return pred.squeeze(-1)              # [B, dec_len]


# ============================================================
# 快速测试
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    model = SolarTransformerRoPE(
        enc_feat_size=16,
        dec_feat_size=15,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        enc_seq_len=96,
        dec_seq_len=16,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params:,}")

    x_enc = torch.randn(4, 96, 16).to(device)
    x_dec = torch.randn(4, 16, 15).to(device)
    with torch.no_grad():
        out = model(x_enc, x_dec)
    print(f"Encoder 输入: {x_enc.shape}")
    print(f"Decoder 输入: {x_dec.shape}")
    print(f"输出:         {out.shape}")   # 期望 [4, 16]
    assert out.shape == (4, 16), "输出形状不对!"
    print("测试通过!")
