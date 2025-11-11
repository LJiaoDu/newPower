"""
PyTorch Lightning版本：20小时历史数据预测未来4小时功率
输入：过去20小时的功率数据（240个点，5分钟间隔）
输出：未来4小时的功率预测（48个点，5分钟间隔）
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pickle
import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks import TQDMProgressBar


class PowerDataset(Dataset):
    """电力数据集"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMPowerPredictor(pl.LightningModule):
    """LSTM电力预测模型"""

    def __init__(self, y_scaler=None):
        super().__init__()

        # 保存超参数
        self.save_hyperparameters(ignore=['y_scaler'])
        self.y_scaler = y_scaler

        # 模型架构（和Keras版本完全一样）
        self.lstm1 = nn.LSTM(
            input_size=1,
            hidden_size=128,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(0.2)

        self.lstm2 = nn.LSTM(
            input_size=256,  # bidirectional所以是128*2
            hidden_size=64,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.dropout2 = nn.Dropout(0.2)

        self.fc1 = nn.Linear(128, 128)  # 64*2
        self.dropout3 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, FORECAST_POINTS)

        # 用于累积验证集预测结果
        self.validation_step_outputs = []
        self.validation_step_targets = []

    def forward(self, x):
        # x: (batch, 240, 1)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # 取最后一个时间步
        x = self.dropout2(x)

        x = torch.relu(self.fc1(x))
        x = self.dropout3(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def training_step(self, batch, batch_idx):
        """训练步骤"""
        X, y = batch
        y_pred = self(X)

        # 计算损失
        loss = nn.functional.mse_loss(y_pred, y)
        mae = torch.mean(torch.abs(y_pred - y))

        # 记录指标（会自动显示在进度条）
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_mae', mae, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        X, y = batch
        y_pred = self(X)

        # 计算损失
        loss = nn.functional.mse_loss(y_pred, y)
        mae = torch.mean(torch.abs(y_pred - y))

        # 记录指标
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mae', mae, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # 保存预测结果用于计算ACC指标
        self.validation_step_outputs.append(y_pred.detach().cpu())
        self.validation_step_targets.append(y.detach().cpu())

        return loss

    def on_validation_epoch_end(self):
        """验证epoch结束后计算ACC指标"""
        if len(self.validation_step_outputs) == 0:
            return

        # 合并所有batch的预测结果
        all_preds = torch.cat(self.validation_step_outputs, dim=0).numpy()
        all_targets = torch.cat(self.validation_step_targets, dim=0).numpy()

        # 如果有scaler，逆归一化
        if self.y_scaler is not None:
            all_preds = self.y_scaler.inverse_transform(all_preds)
            all_targets = self.y_scaler.inverse_transform(all_targets)

        # 计算ACC_指标
        acc_ = self.calculate_acc_(all_targets, all_preds)
        acc_mae = self.calculate_acc_mae(all_targets, all_preds)

        # 记录
        self.log('val_acc_', acc_, prog_bar=True, logger=True)
        self.log('val_acc_mae', acc_mae, prog_bar=True, logger=True)

        # 清空
        self.validation_step_outputs.clear()
        self.validation_step_targets.clear()

    def calculate_acc_(self, y_actual, y_pred):
        """计算ACC_指标（基于RMSE）"""
        y_actual_flat = y_actual.flatten()
        y_pred_flat = y_pred.flatten()

        mask = y_actual_flat != 0
        y_actual_nonzero = y_actual_flat[mask]
        y_pred_nonzero = y_pred_flat[mask]

        if len(y_actual_nonzero) == 0:
            return 0.0

        relative_errors = (y_actual_nonzero - y_pred_nonzero) / y_actual_nonzero
        squared_errors = relative_errors ** 2
        rmse_relative = np.sqrt(np.mean(squared_errors))
        acc_ = 1 - rmse_relative

        return acc_ * 100

    def calculate_acc_mae(self, y_actual, y_pred):
        """计算ACC_MAE指标（基于MAE）"""
        y_actual_flat = y_actual.flatten()
        y_pred_flat = y_pred.flatten()

        mask = y_actual_flat != 0
        y_actual_nonzero = y_actual_flat[mask]
        y_pred_nonzero = y_pred_flat[mask]

        if len(y_actual_nonzero) == 0:
            return 0.0

        relative_errors = np.abs((y_actual_nonzero - y_pred_nonzero) / y_actual_nonzero)
        mae_relative = np.mean(relative_errors)
        acc_mae = 1 - mae_relative

        return acc_mae * 100

    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        # 学习率衰减
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=True
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }


def load_data():
    """加载数据"""
    print("="*60)
    print("加载数据...")
    print("="*60)
    df = pd.read_csv('/home/user/newPower/training_data.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"总数据点: {len(df)}")
    print(f"时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
    return df


def split_raw_data(df, train_ratio=0.8):
    """划分原始数据集"""
    print("\n划分原始数据集...")
    split_idx = int(len(df) * train_ratio)

    df_train = df[:split_idx].copy()
    df_val = df[split_idx:].copy()

    print(f"训练集原始数据: {len(df_train)} 点 ({df_train['datetime'].min()} 到 {df_train['datetime'].max()})")
    print(f"验证集原始数据: {len(df_val)} 点 ({df_val['datetime'].min()} 到 {df_val['datetime'].max()})")

    return df_train, df_val


def create_sequences(df, dataset_name="训练"):
    """创建LSTM序列数据"""
    print(f"\n创建{dataset_name}集序列...")
    print(f"输入窗口: {LOOKBACK_HOURS}小时 ({LOOKBACK_POINTS}个点)")
    print(f"输出窗口: {FORECAST_HOURS}小时 ({FORECAST_POINTS}个点)")

    power_values = df['generationPower'].values

    X = []
    y = []

    total_window = LOOKBACK_POINTS + FORECAST_POINTS

    for i in range(len(power_values) - total_window + 1):
        lookback_power = power_values[i:i+LOOKBACK_POINTS]
        forecast_power = power_values[i+LOOKBACK_POINTS:i+total_window]

        X.append(lookback_power)
        y.append(forecast_power)

    X = np.array(X)
    y = np.array(y)

    # LSTM需要3D输入：(samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    print(f"生成样本数: {len(X)}")
    print(f"输入形状: {X.shape} (样本数, 时间步, 特征数)")
    print(f"输出形状: {y.shape} (样本数, 预测点数)")

    return X, y


def normalize_data(X_train, y_train, X_val, y_val):
    """归一化数据"""
    print("\n归一化数据...")

    # 输入数据归一化
    X_scaler = StandardScaler()
    X_train_scaled = X_train.reshape(-1, 1)
    X_train_scaled = X_scaler.fit_transform(X_train_scaled)
    X_train_scaled = X_train_scaled.reshape(X_train.shape)

    X_val_scaled = X_val.reshape(-1, 1)
    X_val_scaled = X_scaler.transform(X_val_scaled)
    X_val_scaled = X_val_scaled.reshape(X_val.shape)

    # 输出数据归一化
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_val_scaled = y_scaler.transform(y_val)

    print("✓ 数据归一化完成")

    return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_scaler, y_scaler


def main(args):
    """主函数"""
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # GPU配置
    print("="*60)
    print("GPU配置")
    print("="*60)
    if torch.cuda.is_available():
        print(f"✓ 检测到GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA版本: {torch.version.cuda}")
        print(f"✓ GPU数量: {torch.cuda.device_count()}")
    else:
        print("✗ 未检测到GPU，将使用CPU训练")
    print("="*60)
    print()

    print("="*60)
    print("PyTorch Lightning版本：LSTM神经网络训练")
    print("="*60)
    print("\n训练配置:")
    print(f"  数据路径: {args.data_path}")
    print(f"  输入窗口: {args.lookback_hours}小时")
    print(f"  预测窗口: {args.forecast_hours}小时")
    print(f"  训练轮数: {args.epochs}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  学习率: {args.lr}")
    print(f"  LSTM隐藏层: [{args.lstm1_hidden}, {args.lstm2_hidden}]")
    print(f"  全连接层: [{args.fc1_hidden}, {args.fc2_hidden}]")
    print(f"  Dropout: {args.dropout}")
    print(f"  早停patience: {args.early_stop_patience}")
    print()

    # 计算时间点数
    LOOKBACK_POINTS = args.lookback_hours * 60 // 5  # 5分钟间隔
    FORECAST_POINTS = args.forecast_hours * 60 // 5

    # 1. 加载数据
    df = load_data(args.data_path)

    # 2. 划分数据集
    df_train, df_val = split_raw_data(df, args.train_ratio)

    # 3. 创建序列
    X_train, y_train = create_sequences(df_train, LOOKBACK_POINTS, FORECAST_POINTS, "训练")
    X_val, y_val = create_sequences(df_val, LOOKBACK_POINTS, FORECAST_POINTS, "验证")

    # 4. 归一化
    X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_scaler, y_scaler = \
        normalize_data(X_train, y_train, X_val, y_val)

    # 5. 创建PyTorch Dataset和DataLoader
    print("\n创建DataLoader...")

    train_dataset = PowerDataset(X_train_scaled, y_train_scaled)
    val_dataset = PowerDataset(X_val_scaled, y_val_scaled)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"✓ batch_size: {args.batch_size}")
    print(f"✓ 训练batches: {len(train_loader)}")
    print(f"✓ 验证batches: {len(val_loader)}")

    # 6. 创建模型
    print("\n创建模型...")
    model = LSTMPowerPredictor(
        args=args,
        y_scaler=y_scaler,
        output_size=FORECAST_POINTS
    )

    # 打印模型结构
    print("\n模型结构:")
    print(model)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 7. 配置Callbacks
    callbacks = [
        # 早停
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min',
            verbose=True
        ),

        # 保存最佳模型
        ModelCheckpoint(
            monitor='val_loss',
            dirpath='./checkpoints',
            filename='lstm-{epoch:02d}-{val_loss:.4f}',
            save_top_k=1,
            mode='min'
        ),

        # 学习率监控
        LearningRateMonitor(logging_interval='epoch'),

        # 自定义进度条（显示更多指标）
        TQDMProgressBar(refresh_rate=10)
    ]

    # 8. 创建Trainer
    print("\n" + "="*60)
    print("训练配置")
    print("="*60)

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=True,
        precision=32,
    )

    print(f"✓ 最大epoch数: 100")
    print(f"✓ 设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"✓ 早停patience: 10")
    print(f"✓ 学习率衰减patience: 5")
    print("="*60)

    # 9. 开始训练
    print("\n开始训练...")
    print("每个epoch包含:")
    print("  1. 训练阶段 - 有进度条显示 train_loss 和 train_mae")
    print("  2. 验证阶段 - 有进度条显示 val_loss 和 val_mae")
    print()

    trainer.fit(model, train_loader, val_loader)

    # 10. 训练完成
    print("\n" + "="*60)
    print("✓ 训练完成！")
    print("="*60)
    print(f"最佳模型保存在: {trainer.checkpoint_callback.best_model_path}")
    print(f"最佳验证loss: {trainer.checkpoint_callback.best_model_score:.4f}")

    # 11. 保存scaler
    print("\n保存数据归一化器...")
    with open('scalers_pytorch.pkl', 'wb') as f:
        pickle.dump({'X_scaler': X_scaler, 'y_scaler': y_scaler}, f)
    print("✓ 已保存到 scalers_pytorch.pkl")

    print("\n全部完成！")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='LSTM电力预测训练')

    # 数据参数
    parser.add_argument('--data-path', type=str, default='/home/user/newPower/training_data.csv',
                        help='训练数据路径')
    parser.add_argument('--lookback-hours', type=int, default=20,
                        help='输入历史小时数 (default: 20)')
    parser.add_argument('--forecast-hours', type=int, default=4,
                        help='预测未来小时数 (default: 4)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='训练集比例 (default: 0.8)')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数 (default: 100)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='批次大小 (default: 128)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='初始学习率 (default: 0.001)')
    parser.add_argument('--lr-min', type=float, default=0.00001,
                        help='最小学习率 (default: 0.00001)')

    # 模型参数
    parser.add_argument('--lstm1-hidden', type=int, default=128,
                        help='第一层LSTM隐藏单元数 (default: 128)')
    parser.add_argument('--lstm2-hidden', type=int, default=64,
                        help='第二层LSTM隐藏单元数 (default: 64)')
    parser.add_argument('--fc1-hidden', type=int, default=128,
                        help='第一层全连接隐藏单元数 (default: 128)')
    parser.add_argument('--fc2-hidden', type=int, default=64,
                        help='第二层全连接隐藏单元数 (default: 64)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout比例 (default: 0.2)')

    # 回调参数
    parser.add_argument('--early-stop-patience', type=int, default=10,
                        help='早停patience (default: 10)')
    parser.add_argument('--lr-patience', type=int, default=5,
                        help='学习率衰减patience (default: 5)')
    parser.add_argument('--lr-factor', type=float, default=0.5,
                        help='学习率衰减因子 (default: 0.5)')

    # 系统参数
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader的workers数量 (default: 4)')
    parser.add_argument('--device', type=str, default='auto',
                        help='设备: auto, cpu, gpu (default: auto)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (default: 42)')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='模型保存目录 (default: ./checkpoints)')
    parser.add_argument('--resume', type=str, default='',
                        help='从checkpoint恢复训练 (default: "")')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
