#!/usr/bin/env python3
"""
太阳能电站多站点功率预测模型 (基于 model2.SolarTransformerRoPE)

新增 Site Embedding 层：
  - 为每个站点维护一个可学习的嵌入向量 (site_embed_dim 维, 默认 8)
  - 嵌入向量沿时间维度广播后与 encoder/decoder 输入特征拼接
  - 模型同时学习跨站点共享的时序规律 + 各站点特有的出力特性

输入变化 (相对于 model2):
  原始特征: enc=16 维, dec=15 维
  拼接嵌入: enc=16+8=24 维, dec=15+8=23 维（模型内部自动处理）

接口:
  forward(x_enc, x_dec, site_id) -> [B, dec_seq_len]
  site_id: [B] int64 tensor，站点索引 0~(num_sites-1)
"""

import torch
import torch.nn as nn

from model2 import SolarTransformerRoPE


class SolarTransformerMultiSite(SolarTransformerRoPE):
    """
    多站点太阳能功率预测 Transformer (继承 SolarTransformerRoPE)

    核心改动：在 forward() 开头将 site_id 查嵌入表后拼接到输入特征，
    其余 Encoder/Decoder/RoPE 逻辑与父类完全一致，无需重复实现。

    参数:
        num_sites      : 站点总数
        site_embed_dim : 站点嵌入维度 (推荐 4~16, 默认 8)
        enc_feat_size  : 原始 Encoder 特征维度 (不含嵌入, 默认 16)
        dec_feat_size  : 原始 Decoder 特征维度 (不含嵌入, 默认 15)
        **kwargs       : 其余参数透传给 SolarTransformerRoPE
                         (d_model, nhead, num_encoder_layers, ...)
    """

    def __init__(
        self,
        num_sites: int,
        site_embed_dim: int = 8,
        enc_feat_size: int = 16,
        dec_feat_size: int = 15,
        **kwargs,
    ):
        # 父类接收到的特征维度 = 原始特征 + 站点嵌入
        super().__init__(
            enc_feat_size=enc_feat_size + site_embed_dim,
            dec_feat_size=dec_feat_size + site_embed_dim,
            **kwargs,
        )
        self.num_sites      = num_sites
        self.site_embed_dim = site_embed_dim

        self.site_embedding = nn.Embedding(num_sites, site_embed_dim)
        # 小值初始化：避免初始阶段站点嵌入对预测产生过大扰动
        nn.init.normal_(self.site_embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x_enc:   torch.Tensor,   # [B, enc_len, enc_feat_size]
        x_dec:   torch.Tensor,   # [B, dec_len, dec_feat_size]
        site_id: torch.Tensor,   # [B] int64
    ) -> torch.Tensor:
        """
        Returns:
            [B, dec_len]
        """
        site_emb = self.site_embedding(site_id)   # [B, site_embed_dim]

        # 沿时间维度广播，拼接到每个时间步
        enc_exp = site_emb.unsqueeze(1).expand(-1, x_enc.size(1), -1)  # [B, T_enc, E]
        dec_exp = site_emb.unsqueeze(1).expand(-1, x_dec.size(1), -1)  # [B, T_dec, E]

        x_enc = torch.cat([x_enc, enc_exp], dim=-1)   # [B, T_enc, 16+E]
        x_dec = torch.cat([x_dec, dec_exp], dim=-1)   # [B, T_dec, 15+E]

        return super().forward(x_enc, x_dec)


# ============================================================
# 快速测试
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    NUM_SITES     = 5
    SITE_EMBED    = 8
    ENC_FEAT      = 16
    DEC_FEAT      = 15

    model = SolarTransformerMultiSite(
        num_sites=NUM_SITES,
        site_embed_dim=SITE_EMBED,
        enc_feat_size=ENC_FEAT,
        dec_feat_size=DEC_FEAT,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        enc_seq_len=96,
        dec_seq_len=16,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    embed_params = model.site_embedding.weight.numel()
    print(f"总参数量:     {total_params:,}")
    print(f"站点嵌入参数: {embed_params:,}  ({NUM_SITES} 站点 × {SITE_EMBED} 维)")

    B = 8
    x_enc   = torch.randn(B, 96, ENC_FEAT).to(device)
    x_dec   = torch.randn(B, 16, DEC_FEAT).to(device)
    site_id = torch.randint(0, NUM_SITES, (B,)).to(device)

    with torch.no_grad():
        out = model(x_enc, x_dec, site_id)

    print(f"\nEncoder 输入: {x_enc.shape}  + site_emb [{SITE_EMBED}] -> 内部 {ENC_FEAT+SITE_EMBED} 维")
    print(f"Decoder 输入: {x_dec.shape}  + site_emb [{SITE_EMBED}] -> 内部 {DEC_FEAT+SITE_EMBED} 维")
    print(f"Site ID:      {site_id.tolist()}")
    print(f"输出:         {out.shape}")    # 期望 [8, 16]
    assert out.shape == (B, 16), f"输出形状错误: {out.shape}"
    print("测试通过!")
