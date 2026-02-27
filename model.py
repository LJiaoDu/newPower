#!/usr/bin/env python3
"""
改进的Transformer模型 - 非自回归版本
解决原TFMModel的问题：
1. 使用并行预测代替autoregressive解码（避免误差累积）
2. 使用可学习的query embeddings（不从零向量开始）
3. 支持11个特征（10个时间特征 + 1个历史功率）
"""

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder as Encoder
from torch.nn import TransformerEncoderLayer as EncoderLayer
from torch.nn import LayerNorm
import math


# ============== RoPE位置编码（复用原代码） ==============
def _rope_cos_sin(seq_len: int, dim: int, device, dtype, base: float = 10000.0):

    assert dim % 2 == 0, "RoPE 维度必须为偶数"
    half = dim // 2
    inv_freq = 1.0 / (base ** (2 * torch.arange(0, half, device=device, dtype=dtype) / dim))
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.einsum('l,d->ld', t, inv_freq)  # [seq_len, half]
    emb = freqs.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, half]
    return emb.cos(), emb.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):

    x_even = x[..., ::2]   # 偶数位置
    x_odd = x[..., 1::2]   # 奇数位置
    x_rope_even = x_even * cos - x_odd * sin
    x_rope_odd = x_even * sin + x_odd * cos
    out = torch.empty_like(x)
    out[..., ::2] = x_rope_even
    out[..., 1::2] = x_rope_odd
    return out


class ImprovedTFMModel(nn.Module):


    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.in_feat_size     = cfg.in_feat_size
        self.out_feat_size    = cfg.out_feat_size
        self.in_seq_len       = cfg.in_seq_len
        self.out_seq_len      = cfg.out_seq_len
        self.hidden_feat_size = cfg.hidden_feat_size
        self.nhead            = cfg.nhead
        self.rope_base        = 10000.0

        num_enc_layers   = cfg.num_enc_layers
        num_cross_layers = cfg.num_cross_layers
        ffn_dim          = self.hidden_feat_size * cfg.ffn_multiplier

        # ========== Encoder: 处理历史序列 ==========
        self.input_projection = nn.Linear(self.in_feat_size, self.hidden_feat_size)

        encoder_layer = EncoderLayer(
            d_model=self.hidden_feat_size,
            nhead=self.nhead,
            dim_feedforward=ffn_dim,
            dropout=0.1,
            batch_first=True
        )
        encoder_norm = LayerNorm(self.hidden_feat_size)
        self.encoder = Encoder(encoder_layer, num_layers=num_enc_layers, norm=encoder_norm)

        # ========== 可学习的Query Embeddings（关键改进！） ==========
        self.future_queries = nn.Parameter(
            torch.randn(1, self.out_seq_len, self.hidden_feat_size) * 0.02
        )

        # ========== Cross-Attention Layers ==========
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.hidden_feat_size,
                num_heads=self.nhead,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(num_cross_layers)
        ])

        # Layer Norms
        self.cross_attn_norms = nn.ModuleList([
            LayerNorm(self.hidden_feat_size) for _ in range(num_cross_layers)
        ])

        # Feed-Forward Networks
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_feat_size, ffn_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(ffn_dim, self.hidden_feat_size),
                nn.Dropout(0.1)
            )
            for _ in range(num_cross_layers)
        ])

        self.ffn_norms = nn.ModuleList([
            LayerNorm(self.hidden_feat_size) for _ in range(num_cross_layers)
        ])

        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_feat_size, self.hidden_feat_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_feat_size // 2, self.out_feat_size)
        )

        self.dropout = nn.Dropout(0.1)


    def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播

        参数:
            x: 历史序列 [B, 240, 11]
            t: 占位符（保持接口兼容，但不使用）

        返回:
            预测序列 [B, 48, 1]
        """
        B = x.size(0)


        x = self.input_projection(x)  # [B, 240, 256]


        L, H = x.size(1), x.size(2)
        x = x.view(B, L, self.nhead, H // self.nhead).transpose(1, 2)
        dim = H // self.nhead
        cos, sin = _rope_cos_sin(L, dim, device=x.device, dtype=x.dtype, base=self.rope_base)
        x = _apply_rope(x, cos, sin)
        x = x.transpose(1, 2).contiguous().view(B, L, H)

        x = self.dropout(x)

        encoder_out = self.encoder(x)  # [B, 240, 256]


        queries = self.future_queries.expand(B, -1, -1)  # [B, 48, 256]

  
        L_q, H_q = queries.size(1), queries.size(2)
        queries = queries.view(B, L_q, self.nhead, H_q // self.nhead).transpose(1, 2)
        cos_q, sin_q = _rope_cos_sin(L_q, dim, device=queries.device, dtype=queries.dtype, base=self.rope_base)
        queries = _apply_rope(queries, cos_q, sin_q)
        queries = queries.transpose(1, 2).contiguous().view(B, L_q, H_q)

   
        out = queries
        for i in range(len(self.cross_attention_layers)):
            # Cross-Attention
            attn_out, _ = self.cross_attention_layers[i](
                query=out,
                key=encoder_out,
                value=encoder_out
            )
            out = self.cross_attn_norms[i](out + attn_out)  # 残差连接

            ffn_out = self.ffns[i](out)
            out = self.ffn_norms[i](out + ffn_out)  # 残差连接

        predictions = self.output_projection(out)  # [B, 48, 1]

        return predictions


if __name__ == '__main__':
    print("=" * 70)
    print("测试改进的Transformer模型")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}\n")

    # 创建配置对象
    class Config:
        def __init__(self):
            self.in_seq_len = 240      # 20小时历史
            self.out_seq_len = 48      # 4小时未来
            self.in_feat_size = 11     # 10时间特征 + 1功率特征
            self.out_feat_size = 1     # 功率值
            self.hidden_feat_size = 256

    cfg = Config()
    model = ImprovedTFMModel(cfg).to(device=device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}\n")

    # 测试前向传播
    src = torch.rand(4, 240, 11).to(device=device)  # 批次大小4

    print("输入形状:", src.shape)

    model.eval()
    with torch.no_grad():
        out = model(src, None)

    print("输出形状:", out.shape)
    print(f"输出范围: [{out.min().item():.4f}, {out.max().item():.4f}]")

    # 测试训练
    print("\n" + "=" * 70)
    print("测试训练流程")
    print("=" * 70)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_f = nn.MSELoss()

    model.train()
    for i in range(5):
        optimizer.zero_grad()

        src = torch.rand(4, 240, 11).to(device=device)
        tgt = torch.rand(4, 48, 1).to(device=device)

        out = model(src, None)
        loss = loss_f(out, tgt)

        loss.backward()
        optimizer.step()

        print(f"Iteration {i+1}: Loss = {loss.item():.6f}")

    print("\n✓ 模型测试通过！")
    print("\n优势对比原TFMModel:")
    print("  1. ✅ 并行预测（不是autoregressive）- 更快、无误差累积")
    print("  2. ✅ 可学习的query embeddings - 不从零开始")
    print("  3. ✅ 支持11个特征（包括历史功率）")
    print("  4. ✅ 更适合时间序列的连续性预测")
    print("=" * 70)