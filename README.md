# Transformer功率预测系统

基于Transformer的电力功率预测系统，使用20小时的历史数据预测未来4小时的功率。

## 功能特点

- **数据预处理**: 自动从JSON文件中提取时间特征、功率统计特征
- **Transformer模型**: 使用标准正弦位置编码和多头注意力机制
- **双重评估指标**:
  - **ACC1 (趋势准确度)**: 评估预测趋势是否与真实趋势一致
  - **ACC2 (阈值准确度)**: 评估预测值是否在真实值的阈值范围内
- **可视化分析**: 生成训练曲线、预测对比图、误差分析图等

## 文件说明

```
newPower/
├── data_preprocessing.py   # 数据预处理模块
├── transformer_model.py    # Transformer模型定义
├── train.py               # 训练脚本
├── evaluate.py            # 评估脚本
├── main.py               # 主程序入口
├── requirements.txt      # 依赖包列表
└── README.md            # 说明文档

772_*.json               # 原始数据文件

生成的文件:
├── X_train.npy          # 训练集输入
├── y_train.npy          # 训练集标签
├── X_val.npy            # 验证集输入
├── y_val.npy            # 验证集标签
├── X_test.npy           # 测试集输入
├── y_test.npy           # 测试集标签
├── scaler.pkl           # 数据标准化器
├── feature_cols.pkl     # 特征列名
├── checkpoints/         # 模型检查点目录
│   ├── best_model.pth
│   └── training_curves.png
└── results/             # 评估结果目录
    ├── test_metrics.json
    ├── prediction_samples.png
    ├── error_analysis.png
    └── horizon_analysis.png
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 完整流程（推荐）

运行完整的数据预处理、训练和评估流程：

```bash
python main.py --mode all
```

### 2. 分步运行

**步骤1: 数据预处理**
```bash
python main.py --mode preprocess
```

**步骤2: 模型训练**
```bash
python main.py --mode train
```

**步骤3: 模型评估**
```bash
python main.py --mode evaluate
```

### 3. 单独运行各模块

```bash
# 仅数据预处理
python data_preprocessing.py

# 仅训练
python train.py

# 仅评估
python evaluate.py
```

## 数据特征

系统会自动从JSON数据中提取以下特征：

### 1. 时间特征
- **周期性编码**: 小时、分钟、星期、月份、年度的正弦-余弦编码
- **离散特征**: 是否周末等

### 2. 功率统计特征
- **滚动统计**: 不同时间窗口的均值、标准差、最大值、最小值
- **变化率**: 功率变化、百分比变化等

### 3. 原始功率值
- 发电功率(generationPower)

## 模型架构

### Transformer配置
- **模型维度**: 128
- **注意力头数**: 8
- **编码器层数**: 4
- **前馈网络维度**: 512
- **Dropout率**: 0.1

### 位置编码
使用标准的正弦-余弦位置编码：
- PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
- PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

### 多头注意力
8个注意力头，每个头维度为16（128/8）

## 评估指标

### 主要指标

1. **ACC1 (趋势准确度)**
   - 定义: 预测值与真实值的变化趋势一致性
   - 计算: (预测趋势正确的点数) / (总点数)
   - 范围: 0-1，越高越好

2. **ACC2 (阈值准确度)**
   - 定义: 预测值在真实值阈值范围内的比例
   - 计算: (相对误差 ≤ 阈值的点数) / (总点数)
   - 阈值: 默认10%，可配置
   - 范围: 0-1，越高越好

### 辅助指标

3. **RMSE (均方根误差)**
   - 衡量预测值与真实值的总体偏差

4. **MAE (平均绝对误差)**
   - 平均预测误差

5. **MAPE (平均绝对百分比误差)**
   - 相对误差的百分比

## 模型参数

### 数据配置
- **输入序列长度**: 240个时间步（20小时 × 12个点/小时）
- **输出序列长度**: 48个时间步（4小时 × 12个点/小时）
- **采样间隔**: 5分钟

### 训练配置
- **批次大小**: 64
- **学习率**: 0.001
- **优化器**: Adam
- **权重衰减**: 1e-5
- **学习率调度**: ReduceLROnPlateau
- **Early Stopping**: 10个epoch无改善则停止

### 数据划分
- **训练集**: 70%
- **验证集**: 15%
- **测试集**: 15%

## 输出结果

### 训练过程
- `checkpoints/best_model.pth`: 最佳模型
- `checkpoints/training_curves.png`: 训练曲线图
- `checkpoints/training_history.json`: 训练历史数据

### 评估结果
- `results/test_metrics.json`: 详细评估指标
- `results/prediction_samples.png`: 预测样本对比图
- `results/error_analysis.png`: 误差分布和散点图
- `results/horizon_analysis.png`: 不同预测时间范围的性能

## 示例输出

```
测试集评估结果
============================================================
ACC1 (趋势准确度)            : 0.8542 (85.42%)
ACC2 (阈值准确度 10%)        : 0.7321 (73.21%)
ACC2 (阈值准确度 15%)        : 0.8156 (81.56%)
ACC2 (阈值准确度 20%)        : 0.8743 (87.43%)
RMSE                        : 1245.67
MAE                         : 892.34
MAPE (%)                    : 12.45%
============================================================
```

## 自定义配置

### 修改模型参数

编辑 `train.py` 中的模型创建部分：

```python
model = create_model(
    model_type='simple',  # 'simple' 或 'full'
    input_dim=input_dim,
    d_model=128,          # 模型维度
    nhead=8,              # 注意力头数
    num_layers=4,         # 层数
    dim_feedforward=512,  # 前馈网络维度
    dropout=0.1,          # Dropout率
    output_len=output_len
)
```

### 修改训练参数

编辑 `train.py` 中的训练配置：

```python
batch_size = 64           # 批次大小
learning_rate = 0.001     # 学习率
num_epochs = 100          # 训练轮数
```

### 修改数据划分

编辑 `data_preprocessing.py` 中的 `prepare_data_for_training()` 函数：

```python
prepare_data_for_training(
    data_dir='.',
    train_ratio=0.7,      # 训练集比例
    val_ratio=0.15        # 验证集比例
)
```

## 常见问题

### Q: 为什么选择Transformer而不是LSTM？
A: Transformer的多头注意力机制可以更好地捕捉长距离依赖关系，而且训练速度更快（可并行化）。

### Q: 位置编码为什么使用正弦编码？
A: 正弦位置编码可以让模型更好地学习相对位置信息，并且可以外推到训练时未见过的序列长度。

### Q: ACC1和ACC2的区别是什么？
A:
- ACC1关注趋势：预测上升/下降是否正确
- ACC2关注精度：预测值是否接近真实值（在阈值内）

### Q: 如何提高预测准确度？
A:
1. 增加模型维度和层数
2. 添加更多特征（如天气数据）
3. 增加训练数据量
4. 调整学习率和批次大小
5. 使用数据增强技术

## 技术细节

### 为什么使用多头注意力？
多头注意力允许模型同时关注不同的特征子空间，可以捕捉更丰富的模式。8个注意力头可以：
- 关注不同的时间尺度
- 学习不同的特征组合
- 提高模型的表达能力

### 数据标准化的重要性
使用StandardScaler进行标准化可以：
- 加速模型收敛
- 避免梯度消失/爆炸
- 使不同特征处于同一尺度

### Early Stopping
当验证损失连续10个epoch没有改善时停止训练，避免过拟合。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer原始论文
- [PyTorch Transformer文档](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)

## 作者

Claude AI Assistant

## 许可证

MIT License
