#!/usr/bin/env python3
"""
【原有优化】：
1. Warmup缩短: 5 epochs → 2 epochs
2. 初始LR降低: 0.0005 → 0.0003
3. Dropout增强: 0.1 → 0.2
4. Weight decay增强: 0.01 → 0.03
5. Early stopping收紧: patience 15 → 5

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
from torch.utils.tensorboard import SummaryWriter
import time
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合服务器环境
import matplotlib.pyplot as plt


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

        # 7. 历史功率值（归一化） ← 关键特征！
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



    print(f"使用设备: {device}")
    print(f"模型: ImprovedTFMModel (非自回归，并行预测)")
    print(f"特征: 11个 (10时间 + 1历史功率)")

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

    # 【优化3】动态修改模型的Dropout概率：0.1 → 0.2
    print("正在修改模型Dropout...")
    dropout_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.25  # 改：0.1 → 0.2 (适度增强)
            dropout_count += 1
            print(f"  - {name}: Dropout(p={module.p})")
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

    # 【优化4】优化器 - 增强Weight Decay: 0.01 → 0.03
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr_init,
        weight_decay=0.04,  # 改：0.01 → 0.03
        betas=(0.9, 0.999)
    )

    # 【优化1&2】学习率调度器 - 缩短Warmup + 降低初始LR
    warmup_epochs = 2  # 改：5 → 2
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,  # 保持0.1
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=30,
        eta_min=cfg.lr_final
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    print(f"学习率调度:")
    print(f"  - Warmup: {warmup_epochs} epochs (start_factor=0.1)")
    print(f"  - 起始LR: {cfg.lr_init * 0.1:.6f}")
    print(f"  - 峰值LR: {cfg.lr_init:.6f} (Epoch {warmup_epochs})")
    print(f"  - 最终LR: {cfg.lr_final:.6f}\n")

    # 损失函数
    loss_function = nn.MSELoss()

    # TensorBoard
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=f'runs/improved1_optimized_{timestamp}')

    # 【优化5】早停参数 - 更激进
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 3  # 改：15 → 5

    print(f"早停配置: patience={patience} epochs\n")

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
        print(f"从epoch {start_epoch} 恢复训练，最佳验证损失: {best_val_loss:.6f}\n")

    # 训练循环
    print("=" * 80)
    print("开始训练...")
    print("=" * 80 + "\n")

    for epoch_i in range(start_epoch, cfg.epochs):
        # ========== 训练阶段 ==========
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch_i:3d} [Train]", ncols=100)
        for history_data, future_power in pbar:
            history_data = history_data.to(device)
            future_power = future_power.to(device)

            optimizer.zero_grad()

            # 前向传播（不需要t参数）
            predicted_power = model(history_data, None)

            # 计算损失
            loss = loss_function(predicted_power.squeeze(-1), future_power)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})

        train_loss /= len(train_dataloader)

        # ========== 验证阶段 ==========
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_acc_mae = 0.0
        val_acc_rmse = 0.0
        valid_batches_mae = 0
        valid_batches = 0

        # 【新增】收集数据用于可视化（只保存白天数据）
        all_true_values = []
        all_pred_values = []

        with torch.no_grad():
            pbar = tqdm(val_dataloader, desc=f"Epoch {epoch_i:3d} [Val]  ", ncols=100)
            for history_data, future_power in pbar:
                history_data = history_data.to(device)
                future_power = future_power.to(device)

                predicted_power = model(history_data, None)
                loss = loss_function(predicted_power.squeeze(-1), future_power)
                val_loss += loss.item()

                # 过滤零值（只计算有效发电时段）
                mask = future_power * train_data.power_max > 200

                if mask.sum() > 0:
                    future_power_nonzero = future_power[mask]
                    predicted_power_nonzero = predicted_power.squeeze(-1)[mask]

                    # 反归一化到原始功率
                    future_power_real = future_power_nonzero * train_data.power_max
                    predicted_power_real = predicted_power_nonzero * train_data.power_max

                    # 【新增】收集白天数据（功率>200W）用于绘图
                    all_true_values.extend(future_power_real.cpu().numpy().tolist())
                    all_pred_values.extend(predicted_power_real.cpu().numpy().tolist())

                    # 1. MAE-based accuracy
                    batch_mae = torch.mean(torch.abs(predicted_power_real - future_power_real)).item()
                    val_mae += batch_mae

                    acc_batch_mae = 1 - batch_mae / (global_avg_power + 1e-6)
                    acc_batch_mae = max(min(acc_batch_mae, 1.0), 0.0)
                    val_acc_mae += acc_batch_mae
                    valid_batches_mae += 1

                    # 2. RMSE-based accuracy
                    batch_rmse = torch.sqrt(torch.mean(
                        (predicted_power_real - future_power_real) ** 2
                    )).item()

                    acc_batch_rmse = 1 - batch_rmse / (global_avg_power + 1e-6)
                    acc_batch_rmse = max(min(acc_batch_rmse, 1.0), 0.0)
                    val_acc_rmse += acc_batch_rmse
                    valid_batches += 1

                pbar.set_postfix({'loss': f"{loss.item():.6f}"})

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
        writer.add_scalar("Loss/train", train_loss, epoch_i)
        writer.add_scalar("Loss/val", val_loss, epoch_i)
        writer.add_scalar("Loss/train_val_gap", val_loss/train_loss, epoch_i)
        writer.add_scalar("Metric/val_mae", val_mae, epoch_i)
        writer.add_scalar("Metric/val_acc_mae", val_acc_mae, epoch_i)
        writer.add_scalar("Metric/val_acc_rmse", val_acc_rmse, epoch_i)
        writer.add_scalar("LearningRate", lr, epoch_i)

        # TensorBoard文本记录
        writer.add_text(
            "Validation Summary",
            (
                f"**Epoch {epoch_i} Summary (Optimized v1)**\n\n"
                f"- Model: ImprovedTFMModel (non-autoregressive)\n"
                f"- Optimizations: Warmup2/LR0.0003/Patience5/Dropout0.2/WD0.03\n"
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

        # 【新增】绘制真实值vs预测值曲线（只显示白天数据）
        if len(all_true_values) > 0:
            # 限制绘制的数据点数量（避免图片太密集）
            max_points = 2000  # 最多显示2000个点
            if len(all_true_values) > max_points:
                # 均匀采样
                indices = np.linspace(0, len(all_true_values)-1, max_points, dtype=int)
                plot_true = [all_true_values[i] for i in indices]
                plot_pred = [all_pred_values[i] for i in indices]
            else:
                plot_true = all_true_values
                plot_pred = all_pred_values

            # 创建图表
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

            # 子图1: 真实值 vs 预测值曲线
            x_axis = range(len(plot_true))
            ax1.plot(x_axis, plot_true, 'b-', label='真实值 (True)', alpha=0.7, linewidth=1.5)
            ax1.plot(x_axis, plot_pred, 'r-', label='预测值 (Predicted)', alpha=0.7, linewidth=1.5)
            ax1.set_xlabel('数据点索引', fontsize=12)
            ax1.set_ylabel('功率 (W)', fontsize=12)
            ax1.set_title(f'Epoch {epoch_i} - 真实值 vs 预测值 (白天数据, 功率>200W)\n'
                         f'ACC: {val_acc_mae:.2%}, MAE: {val_mae:.2f}W', fontsize=14)
            ax1.legend(loc='upper right', fontsize=10)
            ax1.grid(True, alpha=0.3)

            # 子图2: 散点图（对角线表示完美预测）
            ax2.scatter(plot_true, plot_pred, alpha=0.3, s=10, c='blue')
            min_val = min(min(plot_true), min(plot_pred))
            max_val = max(max(plot_true), max(plot_pred))
            ax2.plot([min_val, max_val], [min_val, max_val], 'r--',
                    label='完美预测线', linewidth=2)
            ax2.set_xlabel('真实值 (W)', fontsize=12)
            ax2.set_ylabel('预测值 (W)', fontsize=12)
            ax2.set_title(f'预测值 vs 真实值散点图', fontsize=14)
            ax2.legend(loc='upper left', fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.set_aspect('equal', adjustable='box')

            plt.tight_layout()

            # 保存图片到文件
            plot_dir = 'prediction_plots'
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, f'epoch_{epoch_i:03d}_prediction.png')
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            print(f"  预测曲线已保存: {plot_path}")

            # 保存到TensorBoard
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
            'power_max': train_data.power_max,
            'power_min': train_data.power_min,
            'global_avg_power': global_avg_power
        }
        torch.save(checkpoint, f"./checkpoint/checkpoint_epoch{epoch_i}.pth")

        # 保存最佳模型和早停
        if val_loss < best_val_loss:
            improvement = (best_val_loss - val_loss) / best_val_loss * 100 if best_val_loss != float("inf") else 0
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(checkpoint, f"./checkpoint/best_epoch{epoch_i}.pth")
            print(f"保存最佳模型！Val Loss: {val_loss:.6f}")
            if improvement > 0:
                print(f"    提升: {improvement:.2f}%\n")
        else:
            patience_counter += 1
            print(f"早停计数器: {patience_counter}/{patience}\n")

            if patience_counter >= patience:
                print(f"{'='*80}")
                print(f"早停触发！连续 {patience} 个 epoch 验证损失没有改进")
                print(f"{'='*80}")
                print(f"最佳验证损失: {best_val_loss:.6f}")
                print(f"最佳模型: best_checkpoint_improved1.pth")
                print(f"{'='*80}\n")
                break

        # 更新学习率
        scheduler.step()

    writer.close()
    print("\n" + "="*80)
    print("训练完成！")
    print("="*80)
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print(f"最佳模型: best_checkpoint_improved1.pth")
    print("="*80)


def get_args():
    parser = argparse.ArgumentParser(
        description='优化版改进Transformer训练',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        
    )

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=64, help='批次大小')
    parser.add_argument('--device', type=str, default='0', help='GPU设备号 (或 "cpu")')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--data-path', type=str, default='./training_data.csv',
                        help='训练数据路径')

    # 学习率参数（优化：默认值降低）
    parser.add_argument('--lr-init', type=float, default=0.0004,
                        help='初始学习率 (优化: 0.0005 → 0.0003)')
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
    cfg = get_args()
    train_val(cfg)