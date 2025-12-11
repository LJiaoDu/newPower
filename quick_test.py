#!/usr/bin/env python3
"""
快速测试脚本 - 用小数据集验证代码是否正常
"""

import torch
from torch import nn
from model import ImprovedTFMModel
import numpy as np

print("="*60)
print("🧪 快速测试：模型是否能正确拟合简单数据")
print("="*60)

# 创建配置
class Config:
    in_seq_len = 240
    out_seq_len = 48
    in_feat_size = 11
    out_feat_size = 1
    hidden_feat_size = 256

cfg = Config()

# 创建模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImprovedTFMModel(cfg).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

print(f"\n使用设备: {device}")
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# 生成简单的正弦波数据（模拟光伏发电曲线）
print("\n生成简单测试数据（正弦波）...")
def generate_sine_data(n_samples=100):
    data = []
    for i in range(n_samples):
        # 生成一条正弦曲线 + 噪声
        t = np.linspace(0, 4*np.pi, 240 + 48)
        power = (np.sin(t) + 1) / 2  # 归一化到 [0, 1]
        power += np.random.normal(0, 0.05, len(power))  # 添加噪声
        power = np.clip(power, 0, 1)

        # 构造11个特征（简化版）
        features = np.zeros((240 + 48, 11))
        features[:, -1] = power  # 最后一列是功率

        # 分割输入和目标
        x = features[:240]
        y = power[240:240+48]

        data.append((x, y))

    return data

train_data = generate_sine_data(100)
print(f"生成了 {len(train_data)} 个训练样本")

# 训练测试
print("\n开始训练（如果能拟合简单数据，说明模型本身没问题）...")
model.train()

for epoch in range(50):
    total_loss = 0
    for x, y in train_data:
        x_tensor = torch.FloatTensor(x).unsqueeze(0).to(device)
        y_tensor = torch.FloatTensor(y).unsqueeze(0).to(device)

        optimizer.zero_grad()
        pred = model(x_tensor, None)
        loss = loss_fn(pred.squeeze(-1), y_tensor)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_data)

    if epoch % 10 == 0 or epoch == 49:
        print(f"Epoch {epoch:2d}: Loss = {avg_loss:.6f}")

# 测试
print("\n测试模型...")
model.eval()
test_data = generate_sine_data(10)
test_losses = []

with torch.no_grad():
    for x, y in test_data:
        x_tensor = torch.FloatTensor(x).unsqueeze(0).to(device)
        y_tensor = torch.FloatTensor(y).unsqueeze(0).to(device)

        pred = model(x_tensor, None)
        loss = loss_fn(pred.squeeze(-1), y_tensor)
        test_losses.append(loss.item())

avg_test_loss = np.mean(test_losses)
print(f"测试Loss: {avg_test_loss:.6f}")

# 判断
print("\n" + "="*60)
if avg_test_loss < 0.01:
    print("✅ 测试通过！模型能正确拟合简单数据")
    print("   → 问题出在真实数据上（数据量/质量/特征）")
elif avg_test_loss < 0.05:
    print("⚠️  模型勉强能拟合简单数据")
    print("   → 可能需要更多训练或调整超参数")
else:
    print("❌ 模型无法拟合简单数据")
    print("   → 模型实现可能有问题，或学习率设置不当")
print("="*60)

print("\n建议:")
print("  1. 运行 python diagnose.py 检查真实数据")
print("  2. 如果数据量充足，尝试增加训练epochs")
print("  3. 如果数据量不足，收集更多JSON文件")
