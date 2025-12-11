#!/usr/bin/env python3
"""
改进版训练脚本 - 针对60%准确率的优化

主要改进：
1. Log归一化 - 处理大范围功率数据（0-3.9MW）
2. 只用白天数据计算Loss - Loss和ACC优化目标一致
3. 默认使用512维隐藏层 - 更大的模型容量
4. 增加历史统计特征 - 过去1小时均值和变化率
"""

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model import ImprovedTFMModel as Model
from tqdm import tqdm
import argparse
import os
import csv
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import time
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Power_Dataset_Improved(Dataset):
    """
    改进的数据集加载器

    改进点：
    1. Log归一化处理大范围功率
    2. 增加历史统计特征
    """
    def __init__(self, cfg, phase='train'):
        super().__init__()

        # 读取CSV数据
        self.csv_data = []
        with open(cfg.data_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)
            for row in csv_reader:
                self.csv_data.append(row)

        self.csv_data = np.array(self.csv_data)

        if cfg.subset < 1.0:
            n = int(len(self.csv_data) * cfg.subset)
            self.csv_data = self.csv_data[:n]

        # 提取特征和目标
        self.features, self.targets = self._extract_features()

        # 划分训练集和验证集
        split_idx = int(len(self.features) * 0.8)
        if phase == 'train':
            self.features = self.features[:split_idx]
            self.targets = self.targets[:split_idx]
        else:
            self.features = self.features[split_idx:]
            self.targets = self.targets[split_idx:]

    def _extract_features(self):
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

        # 【改进1】Log归一化 - 处理大范围数据
        print("\n使用Log归一化处理大范围功率数据...")
        power_log = np.log1p(power)  # log(1 + x)
        self.power_max = power.max()
        self.power_min = power.min()
        self.power_log_max = power_log.max()
        self.power_log_min = power_log.min()

        power_normalized = (power_log - self.power_log_min) / (self.power_log_max - self.power_log_min + 1e-8)

        print(f"功率范围: {self.power_min:.2f} - {self.power_max:.2f} W")
        print(f"Log后范围: {self.power_log_min:.2f} - {self.power_log_max:.2f}")

        # 【改进2】增加历史统计特征
        print("\n计算历史统计特征...")
        power_series = pd.Series(power)

        # 过去1小时平均功率（12个点，每个5分钟）
        power_1h_avg = power_series.rolling(12, min_periods=1).mean().values
        power_1h_avg_log = np.log1p(power_1h_avg)
        power_1h_avg_normalized = (power_1h_avg_log - self.power_log_min) / (self.power_log_max - self.power_log_min + 1e-8)

        # 功率变化率
        power_diff = power_series.diff().fillna(0).values
        power_diff_log = np.sign(power_diff) * np.log1p(np.abs(power_diff))
        power_diff_max = np.max(np.abs(power_diff_log))
        power_diff_normalized = power_diff_log / (power_diff_max + 1e-8)

        self.power_diff_max = power_diff_max

        # 特征工程
        features_list = []

        # 时间特征（10个）
        hour_of_day = (hour + minute / 60) / 24
        features_list.append(hour_of_day)

        hour_angle = 2 * np.pi * hour_of_day
        features_list.append(np.sin(hour_angle))
        features_list.append(np.cos(hour_angle))

        dayofweek_angle = 2 * np.pi * dayofweek / 7
        features_list.append(np.sin(dayofweek_angle))
        features_list.append(np.cos(dayofweek_angle))

        dayofyear_angle = 2 * np.pi * dayofyear / 365
        features_list.append(np.sin(dayofyear_angle))
        features_list.append(np.cos(dayofyear_angle))

        month_angle = 2 * np.pi * (month - 1) / 12
        features_list.append(np.sin(month_angle))
        features_list.append(np.cos(month_angle))

        time_idx_norm = (time_idx - time_idx.min()) / (time_idx.max() - time_idx.min() + 1e-8)
        features_list.append(time_idx_norm)

        # 功率特征（3个）
        features_list.append(power_normalized)        # 当前功率
        features_list.append(power_1h_avg_normalized) # 过去1小时均值
        features_list.append(power_diff_normalized)   # 变化率

        # 合并所有特征 [N, 13]
        features = np.stack(features_list, axis=1).astype(np.float32)

        # 目标变量
        targets = power_normalized

        # 计算全局平均功率
        power_valid = power[power > 200]
        self.global_avg_power = float(np.mean(power_valid))

        print(f"\n特征形状: {features.shape}")
        print(f"特征数量: {features.shape[1]}")
        print(f"  - 时间特征: 10个")
        print(f"  - 功率特征: 3个 (当前功率 + 1小时均值 + 变化率)")
        print(f"全局平均功率（有效发电）: {self.global_avg_power:.2f} W")

        return features, targets

    def denormalize_power(self, power_normalized):
        """反归一化：从归一化值恢复到原始功率"""
        power_log = power_normalized * (self.power_log_max - self.power_log_min) + self.power_log_min
        power = np.expm1(power_log)  # exp(x) - 1
        return power

    def __len__(self):
        return len(self.features) - 24 * 12

    def __getitem__(self, idx):
        in_seq_len = 20 * 12  # 240
        out_seq_len = 4 * 12   # 48

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

    print(f"使用设备: {device}")
    print(f"模型: ImprovedTFMModel (非自回归)")
    print(f"特征: {cfg.in_feat_size}个 (10时间 + 3功率)")
    print(f"隐藏层维度: {cfg.hidden_feat_size}")
    print(f"数据文件: {cfg.data_path}")

    # 加载数据
    train_data = Power_Dataset_Improved(cfg=cfg, phase='train')
    val_data = Power_Dataset_Improved(cfg=cfg, phase='val')

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

    global_avg_power = train_data.global_avg_power
    print(f"全局平均功率: {global_avg_power:.2f} W\n")

    # 创建模型
    model = Model(cfg).to(device)

    # 修改Dropout
    dropout_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.2
            dropout_count += 1
    print(f"✓ 已修改 {dropout_count} 个Dropout层\n")

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}\n")

    # 初始化权重
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=0.5)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    model.apply(init_weights)

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr_init,
        weight_decay=0.04,
        betas=(0.9, 0.999)
    )

    # 学习率调度器
    warmup_epochs = 3
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=cfg.lr_final)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    # 损失函数
    loss_function = nn.MSELoss()

    # TensorBoard
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=f'runs/te_improved_{timestamp}')

    # 早停参数
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5

    # 创建检查点目录
    os.makedirs('./te_checkpoint_improved', exist_ok=True)

    # 从检查点恢复（可选）
    start_epoch = 0
    if cfg.resume:
        checkpoint = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        patience_counter = checkpoint.get('patience_counter', 0)
        print(f"从epoch {start_epoch} 恢复训练\n")

    # 训练循环
    print("=" * 80)
    print("开始训练（改进版）...")
    print("=" * 80 + "\n")

    for epoch_i in range(start_epoch, cfg.epochs):
        # ========== 训练阶段 ==========
        model.train()
        train_loss = 0.0
        train_loss_daytime = 0.0
        train_batches_daytime = 0

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch_i:3d} [Train]", ncols=100)
        for history_data, future_power in pbar:
            history_data = history_data.to(device)
            future_power = future_power.to(device)

            optimizer.zero_grad()

            predicted_power = model(history_data, None)

            # 【改进3】只用白天数据计算loss
            # 阈值：200W对应的归一化值
            power_threshold_norm = (np.log1p(200) - train_data.power_log_min) / (train_data.power_log_max - train_data.power_log_min)
            mask = future_power > power_threshold_norm

            if mask.sum() > 0:
                loss = loss_function(
                    predicted_power.squeeze(-1)[mask],
                    future_power[mask]
                )
                train_loss_daytime += loss.item()
                train_batches_daytime += 1
            else:
                # 如果全是夜间，使用全部数据（避免梯度为0）
                loss = loss_function(predicted_power.squeeze(-1), future_power)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})

        train_loss /= len(train_dataloader)
        train_loss_daytime /= max(train_batches_daytime, 1)

        # ========== 验证阶段 ==========
        model.eval()
        val_loss = 0.0
        val_loss_daytime = 0.0
        val_batches_daytime = 0
        val_mae = 0.0
        val_acc_mae = 0.0
        val_acc_rmse = 0.0
        valid_batches_mae = 0
        valid_batches = 0

        all_true_values = []
        all_pred_values = []

        with torch.no_grad():
            pbar = tqdm(val_dataloader, desc=f"Epoch {epoch_i:3d} [Val]  ", ncols=100)
            for history_data, future_power in pbar:
                history_data = history_data.to(device)
                future_power = future_power.to(device)

                predicted_power = model(history_data, None)

                # 总loss
                loss = loss_function(predicted_power.squeeze(-1), future_power)
                val_loss += loss.item()

                # 白天loss
                mask = future_power > power_threshold_norm
                if mask.sum() > 0:
                    loss_daytime = loss_function(
                        predicted_power.squeeze(-1)[mask],
                        future_power[mask]
                    )
                    val_loss_daytime += loss_daytime.item()
                    val_batches_daytime += 1

                    # 反归一化到原始功率
                    future_power_norm = future_power[mask].cpu().numpy()
                    predicted_power_norm = predicted_power.squeeze(-1)[mask].cpu().numpy()

                    future_power_real = train_data.denormalize_power(future_power_norm)
                    predicted_power_real = train_data.denormalize_power(predicted_power_norm)

                    # 收集数据
                    all_true_values.extend(future_power_real.tolist())
                    all_pred_values.extend(predicted_power_real.tolist())

                    # MAE-based accuracy
                    batch_mae = np.mean(np.abs(predicted_power_real - future_power_real))
                    val_mae += batch_mae

                    acc_batch_mae = 1 - batch_mae / (global_avg_power + 1e-6)
                    acc_batch_mae = max(min(acc_batch_mae, 1.0), 0.0)
                    val_acc_mae += acc_batch_mae
                    valid_batches_mae += 1

                    # RMSE-based accuracy
                    batch_rmse = np.sqrt(np.mean((predicted_power_real - future_power_real) ** 2))
                    acc_batch_rmse = 1 - batch_rmse / (global_avg_power + 1e-6)
                    acc_batch_rmse = max(min(acc_batch_rmse, 1.0), 0.0)
                    val_acc_rmse += acc_batch_rmse
                    valid_batches += 1

                pbar.set_postfix({'loss': f"{loss.item():.6f}"})

        val_loss /= len(val_dataloader)
        val_loss_daytime /= max(val_batches_daytime, 1)
        val_mae /= len(val_dataloader)
        val_acc_mae = val_acc_mae / valid_batches_mae if valid_batches_mae > 0 else float('nan')
        val_acc_rmse = val_acc_rmse / valid_batches if valid_batches > 0 else float('nan')

        lr = optimizer.param_groups[0]['lr']

        # 打印指标
        print(f"\n{'='*80}")
        print(f"Epoch {epoch_i:3d} | LR: {lr:.6f}")
        print(f"{'='*80}")
        print(f"  Train Loss (All):     {train_loss:.6f}")
        print(f"  Train Loss (Daytime): {train_loss_daytime:.6f}")
        print(f"  Val Loss (All):       {val_loss:.6f}")
        print(f"  Val Loss (Daytime):   {val_loss_daytime:.6f}")
        print(f"  Val MAE:              {val_mae:.2f} W")
        print(f"  Val ACC (MAE):        {val_acc_mae:.4f}  ({val_acc_mae*100:.2f}%)")
        print(f"  Val ACC (RMSE):       {val_acc_rmse:.4f}  ({val_acc_rmse*100:.2f}%)")
        print(f"{'='*80}\n")

        # TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch_i)
        writer.add_scalar("Loss/train_daytime", train_loss_daytime, epoch_i)
        writer.add_scalar("Loss/val", val_loss, epoch_i)
        writer.add_scalar("Loss/val_daytime", val_loss_daytime, epoch_i)
        writer.add_scalar("Metric/val_mae", val_mae, epoch_i)
        writer.add_scalar("Metric/val_acc_mae", val_acc_mae, epoch_i)
        writer.add_scalar("Metric/val_acc_rmse", val_acc_rmse, epoch_i)
        writer.add_scalar("LearningRate", lr, epoch_i)

        # 绘图（代码与原版相同，省略...）
        if len(all_true_values) > 0:
            max_points = 2000
            if len(all_true_values) > max_points:
                indices = np.linspace(0, len(all_true_values)-1, max_points, dtype=int)
                plot_true = [all_true_values[i] for i in indices]
                plot_pred = [all_pred_values[i] for i in indices]
            else:
                plot_true = all_true_values
                plot_pred = all_pred_values

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

            x_axis = range(len(plot_true))
            ax1.plot(x_axis, plot_true, 'b-', label='真实值', alpha=0.7, linewidth=1.5)
            ax1.plot(x_axis, plot_pred, 'r-', label='预测值', alpha=0.7, linewidth=1.5)
            ax1.set_xlabel('数据点索引', fontsize=12)
            ax1.set_ylabel('功率 (W)', fontsize=12)
            ax1.set_title(f'Epoch {epoch_i} - 改进版\nACC: {val_acc_mae:.2%}, MAE: {val_mae:.2f}W', fontsize=14)
            ax1.legend(loc='upper right', fontsize=10)
            ax1.grid(True, alpha=0.3)

            ax2.scatter(plot_true, plot_pred, alpha=0.3, s=10, c='blue')
            min_val = min(min(plot_true), min(plot_pred))
            max_val = max(max(plot_true), max(plot_pred))
            ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='完美预测线', linewidth=2)
            ax2.set_xlabel('真实值 (W)', fontsize=12)
            ax2.set_ylabel('预测值 (W)', fontsize=12)
            ax2.set_title(f'预测值 vs 真实值散点图', fontsize=14)
            ax2.legend(loc='upper left', fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.set_aspect('equal', adjustable='box')

            plt.tight_layout()

            plot_dir = 'te_prediction_plots_improved'
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, f'epoch_{epoch_i:03d}_prediction.png')
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            print(f"  预测曲线已保存: {plot_path}")

            writer.add_figure('Predictions/comparison', fig, epoch_i)
            plt.close(fig)

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
            'power_log_max': train_data.power_log_max,
            'power_log_min': train_data.power_log_min,
            'power_max': train_data.power_max,
            'global_avg_power': global_avg_power
        }
        torch.save(checkpoint, f"./te_checkpoint_improved/checkpoint_epoch{epoch_i}.pth")

        # 保存最佳模型
        if val_acc_mae > best_val_loss:  # 注意：ACC越大越好
            improvement = (val_acc_mae - best_val_loss) / max(best_val_loss, 0.01) * 100
            best_val_loss = val_acc_mae
            patience_counter = 0
            torch.save(checkpoint, f"./te_checkpoint_improved/best_epoch{epoch_i}.pth")
            print(f"✅ 保存最佳模型！Val ACC: {val_acc_mae:.4f} ({val_acc_mae*100:.2f}%)")
            if improvement > 0:
                print(f"    提升: {improvement:.2f}%\n")
        else:
            patience_counter += 1
            print(f"早停计数器: {patience_counter}/{patience}\n")

            if patience_counter >= patience:
                print(f"{'='*80}")
                print(f"早停触发！连续 {patience} 个 epoch ACC没有改进")
                print(f"{'='*80}")
                print(f"最佳ACC: {best_val_loss:.4f} ({best_val_loss*100:.2f}%)")
                print(f"{'='*80}\n")
                break

        scheduler.step()

    writer.close()
    print("\n" + "="*80)
    print("训练完成！")
    print("="*80)
    print(f"最佳ACC: {best_val_loss:.4f} ({best_val_loss*100:.2f}%)")
    print(f"检查点目录: ./te_checkpoint_improved/")
    print("="*80)


def get_args():
    parser = argparse.ArgumentParser(description='改进版训练脚本 - 针对大型光伏电站')

    parser.add_argument('--epochs', type=int, default=150, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=64, help='批次大小')
    parser.add_argument('--device', type=str, default='0', help='GPU设备号')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--data-path', type=str, default='./te_training_data.csv', help='训练数据路径')

    parser.add_argument('--lr-init', type=float, default=0.0003, help='初始学习率')
    parser.add_argument('--lr-final', type=float, default=0.00001, help='最终学习率')

    parser.add_argument('--in-seq-len', type=int, default=240, help='输入序列长度')
    parser.add_argument('--out-seq-len', type=int, default=48, help='输出序列长度')
    parser.add_argument('--in-feat-size', type=int, default=13, help='输入特征维度（10时间+3功率）')
    parser.add_argument('--out-feat-size', type=int, default=1, help='输出特征维度')
    parser.add_argument('--hidden-feat-size', type=int, default=512, help='隐藏层维度（改进：256→512）')

    parser.add_argument('--subset', type=float, default=1.0, help='数据子集比例')
    parser.add_argument('--resume', type=str, default='', help='恢复训练的检查点路径')

    return parser.parse_args()


if __name__ == '__main__':
    cfg = get_args()
    train_val(cfg)
