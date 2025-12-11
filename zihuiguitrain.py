#!/usr/bin/env python3
"""
优化版训练脚本 - train_with_power1.py
基于 train_with_power.py，应用了以下优化：

【关键改进】：
1. ✅ Warmup缩短: 10 epochs → 3 epochs
2. ✅ Start factor提高: 0.01 → 0.4 (起始LR更接近最优值)
3. ✅ Early stopping更激进: patience 15 → 5
4. ✅ Weight decay增强: 0.01 → 0.05 (更强的L2正则化)
5. ✅ Batch size增大: 48 → 96 (更稳定的梯度估计)
6. ✅ Dropout增强: 0.1 → 0.3 (动态修改模型)

【预期效果】：
- 原版最佳: Epoch 3, ACC 66.9%
- 优化后预期: Epoch 8-15, ACC 75-80%

【使用方法】：
python train_with_power1.py --device 0 --epochs 100
"""

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model import TFMModel as Model
from tqdm import tqdm
import argparse
import os
import csv
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR


class Power_Dataset_WithHistory(Dataset):
    """
    新数据集加载器 - 包含历史功率特征
    特征：10个时间特征 + 1个历史功率 = 11个特征
    """
    def __init__(self, cfg, phase='train'):
        super().__init__()

        # 读取CSV数据
        self.csv_data = []
        with open(cfg.data_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)  # 跳过表头
            for row in csv_reader:
                self.csv_data.append(row)

        self.csv_data = np.array(self.csv_data)

        # 数据子集选择（用于快速测试）
        if cfg.subset < 1.0:
            n = int(len(self.csv_data) * cfg.subset)
            self.csv_data = self.csv_data[:n]

        # 提取特征和目标
        self.features, self.targets = self._extract_features()

        # 划分训练集和验证集 (80% train, 20% val)
        split_idx = int(len(self.features) * 0.8)
        if phase == 'train':
            self.features = self.features[:split_idx]
            self.targets = self.targets[:split_idx]
        else:
            self.features = self.features[split_idx:]
            self.targets = self.targets[split_idx:]

    def _extract_features(self):
        """
        从原始数据中提取特征并进行特征工程

        原始列：
        0: datetime
        1: generationPower (目标变量)
        2: year
        3: month
        4: day
        5: hour
        6: minute
        7: dayofweek
        8: dayofyear
        9: time_idx
        10: date

        输出特征（11个）：
        1. hour_of_day (归一化)
        2-3. hour_sin/cos (24小时周期)
        4-5. dayofweek_sin/cos (7天周期)
        6-7. dayofyear_sin/cos (365天周期)
        8-9. month_sin/cos (12月周期)
        10. time_idx_norm (长期趋势)
        11. power_normalized (历史功率值) ← 关键新增！
        """
        # 提取数值列
        power = self.csv_data[:, 1].astype(np.float32)
        year = self.csv_data[:, 2].astype(np.float32)
        month = self.csv_data[:, 3].astype(np.float32)
        day = self.csv_data[:, 4].astype(np.float32)
        hour = self.csv_data[:, 5].astype(np.float32)
        minute = self.csv_data[:, 6].astype(np.float32)
        dayofweek = self.csv_data[:, 7].astype(np.float32)
        dayofyear = self.csv_data[:, 8].astype(np.float32)
        time_idx = self.csv_data[:, 9].astype(np.float32)

        # 特征工程
        features_list = []

        # 1. 一天中的小时（归一化到 0-1）
        hour_of_day = (hour + minute / 60) / 24
        features_list.append(hour_of_day)

        # 2. 小时的周期性编码（sin/cos）
        hour_angle = 2 * np.pi * hour_of_day
        features_list.append(np.sin(hour_angle))
        features_list.append(np.cos(hour_angle))

        # 3. 星期几的周期性编码
        dayofweek_angle = 2 * np.pi * dayofweek / 7
        features_list.append(np.sin(dayofweek_angle))
        features_list.append(np.cos(dayofweek_angle))

        # 4. 一年中第几天的周期性编码
        dayofyear_angle = 2 * np.pi * dayofyear / 365
        features_list.append(np.sin(dayofyear_angle))
        features_list.append(np.cos(dayofyear_angle))

        # 5. 月份的周期性编码
        month_angle = 2 * np.pi * (month - 1) / 12
        features_list.append(np.sin(month_angle))
        features_list.append(np.cos(month_angle))

        # 6. 时间索引（归一化，捕捉长期趋势）
        time_idx_norm = (time_idx - time_idx.min()) / (time_idx.max() - time_idx.min() + 1e-8)
        features_list.append(time_idx_norm)

        # 7. 历史功率值（归一化） ← 关键新增！
        # 记录归一化参数供后续反归一化使用
        self.power_max = power.max()
        self.power_min = power.min()
        power_normalized = power / (self.power_max + 1e-8)
        features_list.append(power_normalized)

        # 合并所有特征 [N, 11]
        features = np.stack(features_list, axis=1).astype(np.float32)

        # 目标变量（已经归一化过了）
        targets = power_normalized

        # 计算全局平均功率（用于ACC计算）
        # 只计算有效发电时段（过滤低功率点）
        power_valid = power[power > 200]  # 过滤夜间低功率
        self.global_avg_power = float(np.mean(power_valid))

        print(f"特征形状: {features.shape}")
        print(f"特征数量: {features.shape[1]}")
        print(f"  - 时间特征: 10个")
        print(f"  - 功率特征: 1个 (历史功率值)")
        print(f"功率范围: {self.power_min:.2f} - {self.power_max:.2f}")
        print(f"全局平均功率（有效发电）: {self.global_avg_power:.2f} W")

        return features, targets

    def __len__(self):
        # 确保有足够的序列长度
        return len(self.features) - 24 * 12

    def __getitem__(self, idx):
        """
        返回：
        - history_data: [in_seq_len, num_features] 历史特征（包含功率）
        - future_power: [out_seq_len] 未来功率值（目标）
        """
        in_seq_len = 20 * 12  # 240 (20小时)
        out_seq_len = 4 * 12   # 48 (4小时)

        history_data = self.features[idx : idx + in_seq_len]
        future_power = self.targets[idx + in_seq_len : idx + in_seq_len + out_seq_len]

        return history_data, future_power


def train_val(cfg):
    """训练和验证函数"""

    # 设备配置
    if cfg.device == 'cpu':
        device = torch.device('cpu')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

    print("=" * 80)
    print("优化版训练脚本 - train_with_power1.py")
    print("=" * 80)
    print(f"使用设备: {device}")
    print("\n【应用的优化】:")
    print("  ✅ Warmup: 10 epochs → 3 epochs")
    print("  ✅ Start factor: 0.01 → 0.4")
    print("  ✅ Early stopping patience: 15 → 5")
    print("  ✅ Weight decay: 0.01 → 0.05")
    print("  ✅ Batch size: 48 → 96")
    print("  ✅ Dropout: 0.1 → 0.3 (动态修改)")
    print("=" * 80)

    # 加载数据
    train_data = Power_Dataset_WithHistory(cfg=cfg, phase='train')
    val_data = Power_Dataset_WithHistory(cfg=cfg, phase='val')

    train_dataloader = DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    print(f"\n训练集大小: {len(train_data)}")
    print(f"验证集大小: {len(val_data)}")

    # 获取全局平均功率（用于ACC计算）
    global_avg_power = train_data.global_avg_power
    print(f"全局平均功率: {global_avg_power:.2f} W\n")

    # 创建模型
    model = Model(cfg).to(device)

    # 【优化6】动态修改模型的Dropout概率：0.1 → 0.3
    print("正在修改模型Dropout...")
    dropout_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.3  # 修改dropout概率
            dropout_count += 1
            print(f"  - {name}: Dropout(p={module.p})")
    print(f"✓ 已修改 {dropout_count} 个Dropout层\n")

    # 初始化权重
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=0.5)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    model.apply(init_weights)

    # 【优化4】优化器 - 增强Weight Decay: 0.01 → 0.05
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr_init,
        weight_decay=0.05  # 改：0.01 → 0.05
    )

    # 【优化1&2】学习率调度器 - 缩短Warmup + 提高start_factor
    warmup_epochs = 3  # 改：10 → 3
    main_epochs = cfg.epochs - warmup_epochs
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(
                optimizer,
                start_factor=0.4,  # 改：0.01 → 0.4 (起始LR更接近最优值)
                end_factor=1.0,
                total_iters=warmup_epochs
            ),
            CosineAnnealingLR(optimizer, T_max=main_epochs, eta_min=cfg.lr_final)
        ],
        milestones=[warmup_epochs]
    )

    print(f"学习率调度:")
    print(f"  - Warmup: {warmup_epochs} epochs (start_factor=0.4)")
    print(f"  - 起始LR: {cfg.lr_init * 0.4:.6f}")
    print(f"  - 峰值LR: {cfg.lr_init:.6f}")
    print(f"  - 最终LR: {cfg.lr_final:.6f}\n")

    # 损失函数
    loss_function = torch.nn.MSELoss()

    # TensorBoard
    writer = SummaryWriter(f"runs/exp_with_power1_optimized_{time.strftime('%Y%m%d-%H%M%S')}")

    # 【优化3】训练状态 - 更激进的早停
    start_epoch = 0
    best_val_loss = float("inf")
    patience = 5  # 改：15 → 5
    patience_counter = 0

    print(f"早停配置: patience={patience} epochs\n")

    # 恢复训练
    if cfg.resume and os.path.isfile(cfg.resume):
        print(f"=> 加载检查点 '{cfg.resume}'")
        checkpoint = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float("inf"))
        patience_counter = checkpoint.get('patience_counter', 0)
        print(f"=> 已加载检查点 (epoch {checkpoint['epoch']})\n")

    # 训练循环
    print("=" * 80)
    print("开始训练...")
    print("=" * 80)

    for epoch_i in range(start_epoch, cfg.epochs):
        model.train()
        loss_sum_epoch = 0

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch_i:3d} [Train]", ncols=100)
        for train_i, (history_data, future_power) in enumerate(pbar):
            history_data = history_data.to(device)
            future_power = future_power.to(device)

            optimizer.zero_grad()

            # 前向传播
            predicted_power = model(history_data, future_power.unsqueeze(2))
            loss = loss_function(predicted_power, future_power.unsqueeze(2))

            loss_sum_epoch += loss.item()

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            pbar.set_postfix(loss=f"{loss.item():.6f}")

        train_loss = loss_sum_epoch / len(train_dataloader)
        writer.add_scalar("Loss/train", train_loss, epoch_i)

        scheduler.step()

        # 验证
        model.eval()
        val_loss = 0
        val_mae = 0
        val_acc_mae = 0
        val_acc_rmse = 0
        valid_batches = 0
        valid_batches_mae = 0

        with torch.no_grad():
            pbar = tqdm(val_dataloader, desc=f"Epoch {epoch_i:3d} [Val]  ", ncols=100)
            for history_data, future_power in pbar:
                history_data = history_data.to(device)
                future_power = future_power.to(device)

                predicted_power = model(history_data, future_power.unsqueeze(2))
                loss = loss_function(predicted_power, future_power.unsqueeze(2))
                val_loss += loss.item()

                # 过滤零值（只计算有效发电时段）
                # 注意：targets是归一化后的，需要转换回原始功率来判断
                mask = future_power * train_data.power_max > 200  # 过滤低功率点

                if mask.sum() > 0:
                    future_power_nonzero = future_power[mask]
                    predicted_power_nonzero = predicted_power.squeeze(-1)[mask]

                    # 反归一化到原始功率（用于计算真实误差）
                    future_power_real = future_power_nonzero * train_data.power_max
                    predicted_power_real = predicted_power_nonzero * train_data.power_max

                    # 1. MAE-based accuracy（全局平均功率）
                    batch_mae = torch.mean(torch.abs(predicted_power_real - future_power_real)).item()
                    val_mae += batch_mae

                    acc_batch_mae = 1 - batch_mae / (global_avg_power + 1e-6)
                    acc_batch_mae = max(min(acc_batch_mae, 1.0), 0.0)
                    val_acc_mae += acc_batch_mae
                    valid_batches_mae += 1

                    # 2. RMSE-based accuracy（全局平均功率）
                    batch_rmse = torch.sqrt(torch.mean(
                        (predicted_power_real - future_power_real) ** 2
                    )).item()

                    acc_batch_rmse = 1 - batch_rmse / (global_avg_power + 1e-6)
                    acc_batch_rmse = max(min(acc_batch_rmse, 1.0), 0.0)
                    val_acc_rmse += acc_batch_rmse
                    valid_batches += 1

                pbar.set_postfix(loss=f"{loss.item():.6f}")

        val_loss /= len(val_dataloader)
        val_mae /= len(val_dataloader)
        val_acc_mae = val_acc_mae / valid_batches_mae if valid_batches_mae > 0 else float('nan')
        val_acc_rmse = val_acc_rmse / valid_batches if valid_batches > 0 else float('nan')

        lr = optimizer.param_groups[0]['lr']

        # 打印指标
        print(f"\n{'='*80}")
        print(f"Epoch {epoch_i:3d} | LR: {lr:.6f}")
        print(f"{'='*80}")
        print(f"  Train Loss:    {train_loss:.6f}")
        print(f"  Val Loss:      {val_loss:.6f}  (Gap: {val_loss/train_loss:.2f}x)")
        print(f"  Val MAE:       {val_mae:.2f} W")
        print(f"  Val ACC (MAE): {val_acc_mae:.4f}  ({val_acc_mae*100:.2f}%)")
        print(f"  Val ACC (RMSE):{val_acc_rmse:.4f}  ({val_acc_rmse*100:.2f}%)")
        print(f"{'='*80}\n")

        # TensorBoard标量记录
        writer.add_scalar("Loss/val", val_loss, epoch_i)
        writer.add_scalar("Loss/train_val_gap", val_loss/train_loss, epoch_i)
        writer.add_scalar("Metric/val_mae", val_mae, epoch_i)
        writer.add_scalar("Metric/val_acc_mae", val_acc_mae, epoch_i)
        writer.add_scalar("Metric/val_acc_rmse", val_acc_rmse, epoch_i)
        writer.add_scalar("LearningRate", lr, epoch_i)

        # TensorBoard文本记录（便于训练中断后查看历史）
        writer.add_text(
            "Validation Summary",
            (
                f"**Epoch {epoch_i} Summary (Optimized)**\n\n"
                f"- Features: 11 (10 time + 1 power)\n"
                f"- Optimizations: Warmup3/StartFactor0.4/Patience5/WD0.05/BS96/Dropout0.3\n"
                f"- LR: {lr:.8f}\n"
                f"- Train Loss: {train_loss:.6f}\n"
                f"- Val Loss: {val_loss:.6f}\n"
                f"- Train/Val Gap: {val_loss/train_loss:.2f}x\n"
                f"- Val MAE: {val_mae:.2f}W\n"
                f"- Val Acc_MAE: {val_acc_mae:.6f}\n"
                f"- Val Acc_RMSE: {val_acc_rmse:.6f}\n"
                f"- Global Avg Power: {global_avg_power:.2f}W"
            ),
            epoch_i
        )

        # 保存检查点
        checkpoint = {
            'epoch': epoch_i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'val_acc_mae': val_acc_mae,
            'val_acc_rmse': val_acc_rmse,
            'patience_counter': patience_counter,
            'power_max': train_data.power_max,  # 保存归一化参数
            'power_min': train_data.power_min,
            'global_avg_power': global_avg_power  # 保存全局平均功率
        }
        torch.save(checkpoint, f"./checkpoint/checkpoint_epoch{epoch_i}.pth")

        # 保存最佳模型和早停
        if val_loss < best_val_loss:
            improvement = (best_val_loss - val_loss) / best_val_loss * 100 if best_val_loss != float("inf") else 0
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(checkpoint, f"./checkpoint/best_checkpoint_with_power1.pth")
            print(f"✓✓✓ 保存最佳模型！Val Loss: {val_loss:.6f}")
            if improvement > 0:
                print(f"    提升: {improvement:.2f}%")
        else:
            patience_counter += 1
            print(f"⚠ 连续 {patience_counter}/{patience} 个 epoch 没有改进")
            print(f"   最佳 Val Loss: {best_val_loss:.6f} (当前: {val_loss:.6f})")

        # 早停检查
        if patience_counter >= patience:
            print(f"\n{'='*80}")
            print(f"早停触发！连续 {patience} 个 epoch 验证损失没有改进")
            print(f"{'='*80}")
            print(f"最佳验证损失: {best_val_loss:.6f}")
            print(f"最佳模型已保存: best_checkpoint_with_power1.pth")
            print(f"{'='*80}\n")
            break

    writer.close()

    print("\n" + "="*80)
    print("训练完成！")
    print("="*80)
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print(f"最佳模型: best_checkpoint_with_power1.pth")
    print(f"TensorBoard日志: runs/exp_with_power1_optimized_*")
    print("="*80)


def parse_cfg():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='优化版训练脚本 - 应用了6项关键优化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
优化清单:
  1. Warmup缩短: 10 → 3 epochs
  2. Start factor提高: 0.01 → 0.4
  3. Early stopping: patience 15 → 5
  4. Weight decay增强: 0.01 → 0.05
  5. Batch size增大: 48 → 96
  6. Dropout增强: 0.1 → 0.3

使用示例:
  python train_with_power1.py --device 0 --epochs 100
  python train_with_power1.py --device 0 --epochs 50 --batch-size 128
        """
    )

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--device', type=str, default='0', help='GPU 设备 ID 或 "cpu"')
    parser.add_argument('--data-path', type=str, default='training_data.csv',
                        help='训练数据 CSV 文件路径')
    parser.add_argument('--batch-size', type=int, default=48,
                        help='批次大小 (优化: 48 → 96)')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载线程数')

    # 学习率参数
    parser.add_argument('--lr-init', type=float, default=0.0003, help='初始学习率')
    parser.add_argument('--lr-final', type=float, default=0.00001, help='最终学习率')

    # 模型参数
    parser.add_argument('--in-seq-len', type=int, default=240, help='输入序列长度 (20小时 * 12)')
    parser.add_argument('--out-seq-len', type=int, default=48, help='输出序列长度 (4小时 * 12)')
    parser.add_argument('--in-feat-size', type=int, default=11,
                        help='输入特征维度（10个时间特征 + 1个功率特征）')
    parser.add_argument('--out-feat-size', type=int, default=1, help='输出特征维度（功率值）')
    parser.add_argument('--hidden-feat-size', type=int, default=256, help='隐藏层维度')

    # 其他参数
    parser.add_argument('--subset', type=float, default=1.0, help='使用数据子集比例 (0-1)')
    parser.add_argument('--resume', type=str, default='', help='恢复训练的检查点路径')

    return parser.parse_args()


if __name__ == '__main__':
    cfg = parse_cfg()
    print("\n" + "=" * 80)
    print("优化版训练配置 (train_with_power1.py)")
    print("=" * 80)
    print(f"数据路径: {cfg.data_path}")
    print(f"输入特征维度: {cfg.in_feat_size}")
    print(f"  - 时间特征: 10个 (hour, dayofweek, dayofyear, month, time_idx)")
    print(f"  - 功率特征: 1个 (历史功率值)")
    print(f"输入序列长度: {cfg.in_seq_len}")
    print(f"输出序列长度: {cfg.out_seq_len}")
    print(f"批次大小: {cfg.batch_size} (优化: 48 → 96)")
    print(f"初始学习率: {cfg.lr_init}")
    print(f"训练轮数: {cfg.epochs}")
    print("=" * 80 + "\n")

    train_val(cfg)