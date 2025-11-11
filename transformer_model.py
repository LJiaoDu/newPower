import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class RotaryPositionalEmbedding(nn.Module):
    """
    RoPE (Rotary Position Embedding) 位置编码

    参考论文: RoFormer: Enhanced Transformer with Rotary Position Embedding
    https://arxiv.org/abs/2104.09864

    优势:
    1. 直接编码相对位置信息
    2. 更好的外推能力
    3. 计算效率高
    """

    def __init__(self, dim, max_len=5000, base=10000):
        """
        Args:
            dim: 特征维度（必须是偶数）
            max_len: 最大序列长度
            base: 基数，用于计算频率
        """
        super(RotaryPositionalEmbedding, self).__init__()
        assert dim % 2 == 0, "RoPE维度必须是偶数"

        self.dim = dim
        self.max_len = max_len
        self.base = base

        # 计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # 预计算旋转矩阵
        self._build_cache(max_len)

    def _build_cache(self, max_len):
        """预计算旋转矩阵"""
        t = torch.arange(max_len).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # [max_len, dim/2]

        # 生成cos和sin
        emb = torch.cat((freqs, freqs), dim=-1)  # [max_len, dim]
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def rotate_half(self, x):
        """旋转操作：将向量切分并旋转"""
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, x, seq_len):
        """
        应用旋转位置编码

        Args:
            x: 输入张量 [seq_len, batch_size, dim]
            seq_len: 序列长度

        Returns:
            应用RoPE后的张量
        """
        cos = self.cos_cached[:seq_len, ...]
        sin = self.sin_cached[:seq_len, ...]

        # 扩展维度以匹配batch
        cos = cos.unsqueeze(1)  # [seq_len, 1, dim]
        sin = sin.unsqueeze(1)  # [seq_len, 1, dim]

        # 应用旋转
        return (x * cos) + (self.rotate_half(x) * sin)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, dim]
        """
        seq_len = x.size(0)
        return self.apply_rotary_pos_emb(x, seq_len)


class TransformerPowerPredictor(nn.Module):
    """基于Transformer的功率预测模型"""

    def __init__(self,
                 input_dim,
                 d_model=128,
                 nhead=8,
                 num_encoder_layers=4,
                 num_decoder_layers=4,
                 dim_feedforward=512,
                 dropout=0.1,
                 output_len=48):
        """
        Args:
            input_dim: 输入特征维度
            d_model: Transformer模型维度
            nhead: 多头注意力的头数
            num_encoder_layers: 编码器层数
            num_decoder_layers: 解码器层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout率
            output_len: 输出序列长度（预测的时间步数）
        """
        super(TransformerPowerPredictor, self).__init__()

        self.d_model = d_model
        self.output_len = output_len

        # 输入投影层
        self.input_projection = nn.Linear(input_dim, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_encoder_layers
        )

        # Transformer解码器
        decoder_layers = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=False
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layers,
            num_layers=num_decoder_layers
        )

        # 输出投影层
        self.output_projection = nn.Linear(d_model, 1)

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_square_subsequent_mask(self, sz):
        """生成解码器的掩码矩阵，防止看到未来信息"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt=None):
        """
        Args:
            src: 输入序列 [batch_size, seq_len, input_dim]
            tgt: 目标序列(训练时使用) [batch_size, tgt_len, 1]

        Returns:
            预测的功率序列 [batch_size, output_len]
        """
        # 转换为 [seq_len, batch_size, input_dim]
        src = src.transpose(0, 1)

        # 输入投影
        src = self.input_projection(src) * math.sqrt(self.d_model)

        # 添加位置编码
        src = self.pos_encoder(src)

        # 编码器
        memory = self.transformer_encoder(src)

        # 解码器预测
        if self.training and tgt is not None:
            # 训练模式：使用teacher forcing
            tgt = tgt.unsqueeze(-1) if tgt.dim() == 2 else tgt
            tgt = tgt.transpose(0, 1)  # [tgt_len, batch_size, 1]

            # 投影到d_model维度
            tgt_embed = torch.zeros(tgt.size(0), tgt.size(1), self.d_model).to(tgt.device)
            tgt_embed = self.pos_encoder(tgt_embed)

            # 生成掩码
            tgt_mask = self._generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)

            # 解码
            output = self.transformer_decoder(tgt_embed, memory, tgt_mask=tgt_mask)
            output = self.output_projection(output)  # [tgt_len, batch_size, 1]
            output = output.transpose(0, 1).squeeze(-1)  # [batch_size, tgt_len]

        else:
            # 推理模式：自回归生成
            batch_size = src.size(1)
            device = src.device

            # 初始化输出
            outputs = []

            # 初始化解码器输入
            tgt_embed = torch.zeros(1, batch_size, self.d_model).to(device)

            for i in range(self.output_len):
                # 生成掩码
                tgt_mask = self._generate_square_subsequent_mask(tgt_embed.size(0)).to(device)

                # 解码
                output = self.transformer_decoder(tgt_embed, memory, tgt_mask=tgt_mask)
                output = self.output_projection(output[-1:])  # [1, batch_size, 1]

                outputs.append(output.squeeze(0))

                # 准备下一步的输入
                next_embed = torch.zeros(1, batch_size, self.d_model).to(device)
                tgt_embed = torch.cat([tgt_embed, next_embed], dim=0)

            output = torch.stack(outputs, dim=1).squeeze(-1)  # [batch_size, output_len]

        return output


class SimplerTransformerPredictor(nn.Module):
    """简化版Transformer预测器（仅使用编码器）"""

    def __init__(self,
                 input_dim,
                 d_model=128,
                 nhead=8,
                 num_layers=4,
                 dim_feedforward=512,
                 dropout=0.1,
                 output_len=48,
                 pos_encoding_type='sinusoidal'):
        """
        Args:
            input_dim: 输入特征维度
            d_model: Transformer模型维度
            nhead: 多头注意力的头数
            num_layers: 编码器层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout率
            output_len: 输出序列长度
            pos_encoding_type: 位置编码类型 ('sinusoidal' 或 'rope')
        """
        super(SimplerTransformerPredictor, self).__init__()

        self.d_model = d_model
        self.output_len = output_len
        self.pos_encoding_type = pos_encoding_type

        # 输入投影层
        self.input_projection = nn.Linear(input_dim, d_model)

        # 位置编码
        if pos_encoding_type == 'rope':
            self.pos_encoder = RotaryPositionalEmbedding(d_model)
            self.dropout = nn.Dropout(dropout)
        else:  # sinusoidal
            self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
            self.dropout = None

        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )

        # 输出层
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_len)
        )

        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src):
        """
        Args:
            src: 输入序列 [batch_size, seq_len, input_dim]

        Returns:
            预测的功率序列 [batch_size, output_len]
        """
        # 转换为 [seq_len, batch_size, input_dim]
        src = src.transpose(0, 1)

        # 输入投影
        src = self.input_projection(src) * math.sqrt(self.d_model)

        # 添加位置编码
        src = self.pos_encoder(src)

        # RoPE不包含dropout，需要单独添加
        if self.pos_encoding_type == 'rope' and self.dropout is not None:
            src = self.dropout(src)

        # 编码器
        memory = self.transformer_encoder(src)

        # 使用最后一个时间步的输出
        output = memory[-1, :, :]  # [batch_size, d_model]

        # 全连接层输出预测
        output = self.fc_out(output)  # [batch_size, output_len]

        return output


def create_model(model_type='simple', input_dim=None, **kwargs):
    """
    创建模型的工厂函数

    Args:
        model_type: 'simple' 或 'full'
        input_dim: 输入特征维度
        **kwargs: 其他模型参数

    Returns:
        模型实例
    """
    if input_dim is None:
        raise ValueError("必须提供input_dim参数")

    if model_type == 'simple':
        return SimplerTransformerPredictor(input_dim=input_dim, **kwargs)
    elif model_type == 'full':
        return TransformerPowerPredictor(input_dim=input_dim, **kwargs)
    else:
        raise ValueError(f"未知的模型类型: {model_type}")


if __name__ == '__main__':
    # 测试模型
    batch_size = 32
    seq_len = 240  # 20小时
    input_dim = 25  # 假设有25个特征
    output_len = 48  # 4小时

    # 测试简化版模型
    print("=== 测试简化版Transformer模型 ===")
    model_simple = SimplerTransformerPredictor(
        input_dim=input_dim,
        d_model=128,
        nhead=8,
        num_layers=4,
        output_len=output_len
    )

    x = torch.randn(batch_size, seq_len, input_dim)
    y_pred = model_simple(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y_pred.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model_simple.parameters()):,}")

    # 测试完整版模型
    print("\n=== 测试完整版Transformer模型 ===")
    model_full = TransformerPowerPredictor(
        input_dim=input_dim,
        d_model=128,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        output_len=output_len
    )

    y_pred_full = model_full(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y_pred_full.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model_full.parameters()):,}")
