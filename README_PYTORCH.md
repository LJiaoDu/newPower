# PyTorch Lightning版本使用说明

## 🚀 快速开始

### 1. 安装依赖

```bash
# GPU版本 (推荐，需要NVIDIA显卡 + CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-lightning pandas numpy scikit-learn matplotlib tqdm

# 或者CPU版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pytorch-lightning pandas numpy scikit-learn matplotlib tqdm
```

### 2. 运行训练

```bash
python train24_lstm_pytorch.py
```

## 📊 训练输出示例

```
============================================================
GPU配置
============================================================
✓ 检测到GPU: NVIDIA GeForce RTX 3080
✓ CUDA版本: 11.8
✓ GPU数量: 1
============================================================

训练配置
============================================================
✓ batch_size: 128
✓ 训练batches: 712
✓ 验证batches: 177
============================================================

开始训练...
每个epoch包含:
  1. 训练阶段 - 有进度条显示 train_loss 和 train_mae
  2. 验证阶段 - 有进度条显示 val_loss 和 val_mae

Epoch 1/100:
Training:   100%|██████████| 712/712 [03:45<00:00, 3.16batch/s, train_loss=0.523, train_mae=0.398]
Validation: 100%|██████████| 177/177 [00:28<00:00, 6.21batch/s, val_loss=0.567, val_mae=0.421, val_acc_=85.32, val_acc_mae=87.45]

Epoch 2/100:
Training:   100%|██████████| 712/712 [03:42<00:00, 3.20batch/s, train_loss=0.498, train_mae=0.385]
Validation: 100%|██████████| 177/177 [00:27<00:00, 6.35batch/s, val_loss=0.543, val_mae=0.408, val_acc_=86.12, val_acc_mae=88.23]
```

## ✨ 主要特性

### 1. 自动GPU加速
- 自动检测并使用GPU
- 如果没有GPU，自动切换到CPU

### 2. 双进度条
- **训练进度条**: 显示 train_loss 和 train_mae
- **验证进度条**: 显示 val_loss、val_mae、val_acc_、val_acc_mae

### 3. 早停机制
- 验证loss连续10个epoch不下降自动停止
- 自动恢复最佳权重

### 4. 学习率衰减
- 验证loss连续5个epoch不下降，学习率减半
- 最小学习率: 0.00001

### 5. 自动保存最佳模型
- 保存路径: `./checkpoints/`
- 文件名: `lstm-{epoch:02d}-{val_loss:.4f}.ckpt`

### 6. ACC指标
- **ACC_**: 基于RMSE的相对误差
- **ACC_MAE**: 基于MAE的相对误差
- 自动忽略真实值为0的样本

## 📁 输出文件

```
checkpoints/
  └── lstm-15-0.4523.ckpt          # 最佳模型
scalers_pytorch.pkl                 # 数据归一化器
lightning_logs/                     # TensorBoard日志
  └── version_0/
      ├── checkpoints/
      ├── events.out.tfevents...
      └── hparams.yaml
```

## 🔍 查看训练日志（TensorBoard）

```bash
tensorboard --logdir=lightning_logs
```

然后在浏览器打开: http://localhost:6006

## ⚙️ 修改训练参数

### 修改batch_size

```python
# train24_lstm_pytorch.py 第308行
batch_size = 128  # 改成你想要的值
```

### 修改epochs

```python
# train24_lstm_pytorch.py 第392行
trainer = pl.Trainer(
    max_epochs=100,  # 改成你想要的值
    ...
)
```

### 修改早停patience

```python
# train24_lstm_pytorch.py 第370行
EarlyStopping(
    monitor='val_loss',
    patience=10,  # 改成你想要的值
    ...
)
```

## 🆚 与Keras版本对比

| 特性 | Keras版本 | PyTorch Lightning版本 |
|------|----------|---------------------|
| 框架 | TensorFlow/Keras | PyTorch Lightning |
| GPU支持 | ✅ | ✅ |
| 进度条 | tqdm | 内置 + tqdm |
| 早停 | ✅ | ✅ |
| 学习率衰减 | ✅ | ✅ |
| 模型结构 | 相同 | 相同 |
| 代码行数 | ~350行 | ~450行 |
| 灵活性 | 中等 | 高 |
| TensorBoard | 需要配置 | 自动生成 |

## 💡 使用建议

1. **首次使用**: 先用默认参数运行，看看效果
2. **调参**: 根据val_loss曲线调整学习率、batch_size等
3. **GPU显存不够**: 减小batch_size（如64）
4. **训练太慢**: 增大batch_size（如256）
5. **想要更多控制**: PyTorch Lightning提供了丰富的hooks和callbacks

## 🐛 常见问题

### Q: 如何加载训练好的模型？

```python
from train24_lstm_pytorch import LSTMPowerPredictor

# 从checkpoint加载
model = LSTMPowerPredictor.load_from_checkpoint('checkpoints/lstm-15-0.4523.ckpt')

# 预测
model.eval()
with torch.no_grad():
    predictions = model(X_test)
```

### Q: 如何在多GPU上训练？

```python
trainer = pl.Trainer(
    max_epochs=100,
    accelerator='gpu',
    devices=2,  # 使用2个GPU
    strategy='ddp'  # 分布式数据并行
)
```

### Q: 训练中断了怎么恢复？

```python
# 从最后一个checkpoint恢复
trainer.fit(model, train_loader, val_loader, ckpt_path='last')
```

## 📚 更多资源

- [PyTorch Lightning文档](https://lightning.ai/docs/pytorch/stable/)
- [PyTorch文档](https://pytorch.org/docs/stable/index.html)
- [TensorBoard教程](https://www.tensorflow.org/tensorboard)
