"""
纯PyTorch版本：20小时历史数据预测未来4小时功率
手动训练循环 + tqdm进度条
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pickle
import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

torch.set_float32_matmul_precision('high')


class PowerDataset(Dataset):
    """电力数据集"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMPowerPredictor(nn.Module):
    """LSTM电力预测模型"""

    def __init__(self, args, output_size=48):
        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=1,
            hidden_size=args.lstm1_hidden,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(args.dropout)

        self.lstm2 = nn.LSTM(
            input_size=args.lstm1_hidden * 2,
            hidden_size=args.lstm2_hidden,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.dropout2 = nn.Dropout(args.dropout)

        self.fc1 = nn.Linear(args.lstm2_hidden * 2, args.fc1_hidden)
        self.dropout3 = nn.Dropout(args.dropout)
        self.fc2 = nn.Linear(args.fc1_hidden, args.fc2_hidden)
        self.fc3 = nn.Linear(args.fc2_hidden, output_size)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.dropout2(x)

        x = torch.relu(self.fc1(x))
        x = self.dropout3(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def calculate_acc_(y_actual, y_pred):
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


def calculate_acc_mae(y_actual, y_pred):
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


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()

    train_losses = []
    train_maes = []

    pbar = tqdm(train_loader, desc='训练', ncols=100)
    for X, y in pbar:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        y_pred = model(X)

        loss = criterion(y_pred, y)
        mae = torch.mean(torch.abs(y_pred - y))

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        train_maes.append(mae.item())

        # 更新进度条显示
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = np.mean(train_losses)
    avg_mae = np.mean(train_maes)

    return avg_loss, avg_mae


def validate(model, val_loader, criterion, device, y_scaler):
    """验证模型"""
    model.eval()

    val_losses = []
    val_maes = []
    all_preds = []
    all_targets = []

    pbar = tqdm(val_loader, desc='验证', ncols=100)
    with torch.no_grad():
        for X, y in pbar:
            X, y = X.to(device), y.to(device)

            y_pred = model(X)

            loss = criterion(y_pred, y)
            mae = torch.mean(torch.abs(y_pred - y))

            val_losses.append(loss.item())
            val_maes.append(mae.item())

            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

            # 更新进度条显示
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = np.mean(val_losses)
    avg_mae = np.mean(val_maes)

    # 计算ACC指标（在原始尺度上）
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    if y_scaler is not None:
        all_preds = y_scaler.inverse_transform(all_preds)
        all_targets = y_scaler.inverse_transform(all_targets)

    acc_ = calculate_acc_(all_targets, all_preds)
    acc_mae = calculate_acc_mae(all_targets, all_preds)

    return avg_loss, avg_mae, acc_, acc_mae


def load_data(data_path):
    """加载数据"""
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"总数据点: {len(df)}")
    print(f"时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
    return df


def split_raw_data(df, train_ratio=0.8):
    """划分原始数据集"""
    split_idx = int(len(df) * train_ratio)

    df_train = df[:split_idx].copy()
    df_val = df[split_idx:].copy()
    print(f"训练集原始数据: {len(df_train)} 点 ({df_train['datetime'].min()} 到 {df_train['datetime'].max()})")
    print(f"验证集原始数据: {len(df_val)} 点 ({df_val['datetime'].min()} 到 {df_val['datetime'].max()})")

    return df_train, df_val


def create_sequences(df, lookback_points, forecast_points, dataset_name="训练"):
    """创建LSTM序列数据"""
    power_values = df['generationPower'].values

    X = []
    y = []

    total_window = lookback_points + forecast_points

    for i in range(len(power_values) - total_window + 1):
        lookback_power = power_values[i:i+lookback_points]
        forecast_power = power_values[i+lookback_points:i+total_window]

        X.append(lookback_power)
        y.append(forecast_power)

    X = np.array(X)
    y = np.array(y)

    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y


def normalize_data(X_train, y_train, X_val, y_val):
    """归一化数据"""
    X_scaler = StandardScaler()
    X_train_scaled = X_train.reshape(-1, 1)
    X_train_scaled = X_scaler.fit_transform(X_train_scaled)
    X_train_scaled = X_train_scaled.reshape(X_train.shape)

    X_val_scaled = X_val.reshape(-1, 1)
    X_val_scaled = X_scaler.transform(X_val_scaled)
    X_val_scaled = X_val_scaled.reshape(X_val.shape)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_val_scaled = y_scaler.transform(y_val)

    print("数据归一化完成")

    return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_scaler, y_scaler


def main(args):
    """主函数"""
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif args.device == 'gpu':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if torch.cuda.is_available():
        print(f"检测到GPU: {torch.cuda.get_device_name(0)}")
    print(f"使用设备: {device}\n")

    # 计算时间点数
    LOOKBACK_POINTS = args.input_hours * 60 // 5
    FORECAST_POINTS = args.pre_hours * 60 // 5

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

    # 5. 创建DataLoader
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

    print(f"batch_size: {args.batch_size}")
    print(f"训练batches: {len(train_loader)}")
    print(f"验证batches: {len(val_loader)}\n")

    # 6. 创建模型
    model = LSTMPowerPredictor(args=args, output_size=FORECAST_POINTS)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}\n")

    # 7. 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.lr_min,
    )

    # 8. 训练循环
    print("=" * 60)
    print("开始训练")
    print("=" * 60)

    best_val_loss = float('inf')
    patience_counter = 0

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # 训练阶段
        train_loss, train_mae = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"[训练完成] train_loss={train_loss:.4f}, train_mae={train_mae:.4f}")

        # 验证阶段
        val_loss, val_mae, acc_, acc_mae = validate(model, val_loader, criterion, device, y_scaler)
        print(f"[验证完成] val_loss={val_loss:.4f}, val_mae={val_mae:.4f}, ACC_={acc_:.2f}%, ACC_MAE={acc_mae:.2f}%")

        # 学习率调度
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            best_model_path = os.path.join(args.checkpoint_dir, f'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'acc_': acc_,
                'acc_mae': acc_mae,
            }, best_model_path)
            print(f"✓ 保存最佳模型 (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"早停计数: {patience_counter}/{args.early_stop_patience}")

        # 早停
        if patience_counter >= args.early_stop_patience:
            print(f"\n早停触发！已经{args.early_stop_patience}个epoch没有改进")
            break

    # 9. 保存scaler
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"最佳验证loss: {best_val_loss:.4f}")

    print("\n保存数据归一化器...")
    with open('scalers_pytorch.pkl', 'wb') as f:
        pickle.dump({'X_scaler': X_scaler, 'y_scaler': y_scaler}, f)
    print("已保存到 scalers_pytorch.pkl")
    print(f"最佳模型已保存到 {best_model_path}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='LSTM电力预测训练')
    parser.add_argument('--data-path', type=str, default='training_data.csv')
    parser.add_argument('--input-hours', type=int, default=20)
    parser.add_argument('--pre-hours', type=int, default=4)
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr-min', type=float, default=0.00001)
    parser.add_argument('--lstm1-hidden', type=int, default=128)
    parser.add_argument('--lstm2-hidden', type=int, default=64)
    parser.add_argument('--fc1-hidden', type=int, default=128)
    parser.add_argument('--fc2-hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--early-stop-patience', type=int, default=10)
    parser.add_argument('--lr-patience', type=int, default=5)
    parser.add_argument('--lr-factor', type=float, default=0.5)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    parser.add_argument('--resume', type=str, default='')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
