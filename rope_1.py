#!/usr/bin/env python3
"""
太阳能电站功率预测 - Encoder-Decoder Transformer + RoPE

改进点 (相比 solar_model.py):
1. 用 RoPE (Rotary Position Encoding) 替代正弦位置编码
2. RoPE 正确应用在 attention 的 Q/K 上, 而非 additive 加到输入
3. 自定义 Encoder/Decoder Layer 以支持 RoPE 注入
4. Cross-Attention 中 Q 用绝对续接位置 [96..111], K 用 [0..95], 编码时间距离

架构:
  Encoder: 历史24h (功率+气象+时间) [B, 96, 13] -> RoPE Self-Attention -> memory
  Decoder: 未来4h  (气象+时间)      [B, 16, 12] -> RoPE Self-Attn + Cross-Attn -> 功率预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# ============== RoPE 核心 ==============

class RotaryPositionEncoding(nn.Module):
    """
    预计算 RoPE 的 cos/sin 缓存表

    head_dim 必须为偶数, 每 2 个维度组成一对做旋转
    支持 offset 参数, 用于 cross-attention 时 Q 使用续接位置
    """

    def __init__(self, head_dim: int, max_len: int = 200, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"

        # inv_freq: [head_dim/2]
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

        # 预计算缓存
        self._build_cache(max_len)

    def _build_cache(self, max_len: int):
        pos = torch.arange(0, max_len, dtype=torch.float32)
        # freqs: [max_len, head_dim/2]
        freqs = torch.outer(pos, self.inv_freq)
        # emb: [max_len, head_dim]  (每对频率重复, 与 rotate_half 对应)
        emb = torch.cat([freqs, freqs], dim=-1)
        # [1, 1, max_len, head_dim] 方便广播到 [B, nhead, seq_len, head_dim]
        cos_cached = emb.cos().unsqueeze(0).unsqueeze(0)
        sin_cached = emb.sin().unsqueeze(0).unsqueeze(0)
        self.register_buffer("cos_cached", cos_cached)
        self.register_buffer("sin_cached", sin_cached)

    def forward(self, seq_len: int, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回 [1, 1, seq_len, head_dim] 的 cos/sin

        Args:
            seq_len: 序列长度
            offset: 位置偏移量 (cross-attention 时 Q 用 offset=enc_seq_len)
        """
        return (
            self.cos_cached[:, :, offset:offset + seq_len, :],
            self.sin_cached[:, :, offset:offset + seq_len, :],
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """将 x 的前半和后半交换并取反后半, 用于 RoPE 旋转"""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor,
    cos_q: torch.Tensor, sin_q: torch.Tensor,
    cos_k: torch.Tensor, sin_k: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对 Q 和 K 分别应用 RoPE (V 不旋转)

    Args:
        q: [B, nhead, q_len, head_dim]
        k: [B, nhead, k_len, head_dim]
        cos_q/sin_q: [1, 1, q_len, head_dim]
        cos_k/sin_k: [1, 1, k_len, head_dim]
    """
    q_rot = q * cos_q + rotate_half(q) * sin_q
    k_rot = k * cos_k + rotate_half(k) * sin_k
    return q_rot, k_rot


# ============== 自定义多头注意力 (支持 RoPE) ==============

class RoPEMultiheadAttention(nn.Module):
    """
    多头注意力, Q/K 投影后先应用 RoPE 再计算 attention

    与 nn.MultiheadAttention 的区别:
    - Q, K, V 使用独立的 Linear 投影 (参数量相同)
    - forward 额外接收 cos_q, sin_q, cos_k, sin_k
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        query: torch.Tensor,       # [B, q_len, d_model]
        key: torch.Tensor,         # [B, k_len, d_model]
        value: torch.Tensor,       # [B, k_len, d_model]
        cos_q: torch.Tensor,       # [1, 1, q_len, head_dim]
        sin_q: torch.Tensor,
        cos_k: torch.Tensor,       # [1, 1, k_len, head_dim]
        sin_k: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, q_len, _ = query.shape
        k_len = key.shape[1]

        # 投影
        Q = self.q_proj(query).view(B, q_len, self.nhead, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, k_len, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, k_len, self.nhead, self.head_dim).transpose(1, 2)
        # Q/K/V: [B, nhead, seq_len, head_dim]

        # 应用 RoPE (仅 Q 和 K, V 不旋转)
        Q, K = apply_rotary_pos_emb(Q, K, cos_q, sin_q, cos_k, sin_k)

        # Scaled Dot-Product Attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # attn_weights: [B, nhead, q_len, k_len]

        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 加权求和
        out = torch.matmul(attn_weights, V)  # [B, nhead, q_len, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, q_len, self.d_model)

        return self.out_proj(out)


# ============== 自定义 Encoder/Decoder Layer ==============

class RoPEEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer + RoPE
    Post-LN 残差结构 (匹配 PyTorch 默认行为)
    """

    def __init__(self, d_model: int, nhead: int,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attn = RoPEMultiheadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: [B, seq_len, d_model]
            cos/sin: [1, 1, seq_len, head_dim]
        """
        # Self-Attention + Residual + LayerNorm
        src2 = self.self_attn(src, src, src, cos, sin, cos, sin)
        src = self.norm1(src + self.dropout1(src2))

        # Feedforward + Residual + LayerNorm
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout3(src2))

        return src


class RoPEDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer + RoPE
    Self-Attention + Cross-Attention + Feedforward
    Post-LN 残差结构
    """

    def __init__(self, d_model: int, nhead: int,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attn = RoPEMultiheadAttention(d_model, nhead, dropout)
        self.cross_attn = RoPEMultiheadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        tgt: torch.Tensor,                   # [B, dec_len, d_model]
        memory: torch.Tensor,                 # [B, enc_len, d_model]
        cos_self: torch.Tensor,               # decoder self-attn 位置
        sin_self: torch.Tensor,
        cos_cross_q: torch.Tensor,            # cross-attn Q 位置 (decoder 绝对位置)
        sin_cross_q: torch.Tensor,
        cos_cross_k: torch.Tensor,            # cross-attn K 位置 (encoder 位置)
        sin_cross_k: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Self-Attention (decoder 内部, 局部位置)
        tgt2 = self.self_attn(tgt, tgt, tgt,
                               cos_self, sin_self, cos_self, sin_self)
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        # 2. Cross-Attention (Q=decoder绝对位置, K=encoder位置 → 编码时间距离)
        tgt2 = self.cross_attn(tgt, memory, memory,
                                cos_cross_q, sin_cross_q,
                                cos_cross_k, sin_cross_k)
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        # 3. Feedforward
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout4(tgt2))

        return tgt


# ============== 顶层模型 ==============

class SolarTransformerRoPE(nn.Module):
    """
    太阳能电站功率预测 - Encoder-Decoder Transformer + RoPE

    forward 接口与 SolarTransformer 完全一致:
        forward(x_enc, x_dec) -> [B, 16]

    Encoder 输入: [B, 96, 13]  历史 (功率+气象+时间)
    Decoder 输入: [B, 16, 12]  未来 (气象+时间, 无功率)
    输出:         [B, 16]      未来功率预测
    """

    def __init__(self,
                 enc_feat_size: int = 13,
                 dec_feat_size: int = 12,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 4,
                 num_decoder_layers: int = 4,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 enc_seq_len: int = 96,
                 dec_seq_len: int = 16,
                 rope_base: float = 10000.0):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.enc_seq_len = enc_seq_len
        self.dec_seq_len = dec_seq_len
        head_dim = d_model // nhead

        # --- 输入投影 ---
        self.enc_projection = nn.Sequential(
            nn.Linear(enc_feat_size, d_model),
            nn.LayerNorm(d_model),
        )
        self.dec_projection = nn.Sequential(
            nn.Linear(dec_feat_size, d_model),
            nn.LayerNorm(d_model),
        )

        # --- RoPE (共享频率表, 覆盖 encoder + decoder 全长) ---
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
            x_enc: [B, enc_seq_len, enc_feat_size]  历史序列
            x_dec: [B, dec_seq_len, dec_feat_size]  未来序列
        Returns:
            [B, dec_seq_len]  功率预测
        """
        enc_len = x_enc.size(1)
        dec_len = x_dec.size(1)

        # 1. 预计算 RoPE cos/sin
        enc_cos, enc_sin = self.rope(enc_len, offset=0)          # [0..95]
        dec_self_cos, dec_self_sin = self.rope(dec_len, offset=0) # [0..15] 局部位置
        dec_cross_q_cos, dec_cross_q_sin = self.rope(dec_len, offset=enc_len)  # [96..111]
        dec_cross_k_cos, dec_cross_k_sin = enc_cos, enc_sin      # [0..95]

        # 2. Encoder
        enc = self.enc_projection(x_enc)     # [B, 96, d_model]
        for layer in self.encoder_layers:
            enc = layer(enc, enc_cos, enc_sin)
        memory = self.encoder_norm(enc)      # [B, 96, d_model]

        # 3. Decoder
        dec = self.dec_projection(x_dec)     # [B, 16, d_model]
        for layer in self.decoder_layers:
            dec = layer(dec, memory,
                        dec_self_cos, dec_self_sin,
                        dec_cross_q_cos, dec_cross_q_sin,
                        dec_cross_k_cos, dec_cross_k_sin)
        out = self.decoder_norm(dec)         # [B, 16, d_model]

        # 4. 输出
        pred = self.output_projection(out)   # [B, 16, 1]
        return pred.squeeze(-1)              # [B, 16]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    model = SolarTransformerRoPE(
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
