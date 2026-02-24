#!/usr/bin/env python3
"""
太阳能电站功率预测 - GRU 模型

数据:  solar_station_1.csv (15分钟粒度, 含气象特征)
模型:  GRU 编码器 + 全连接预测头
预测:  使用 96 步 (24h) 历史数据预测未来 16 步 (4h) 功率

准确度计算参考 solar_train_ed.py:
  ACC1 (MAE-based): 1 - MAE / mean(y_true)
  ACC2 (国标):      1 - sqrt((1/N) * sum(((P_M - P_P) / max(P_M, 0.2*Cap))^2))
  注: 不过滤夜间零功率时刻, 所有时刻均参与计算

使用方式:
  python solar_gru.py                          # 完整流程: 预处理 + 训练 + 评估
  python solar_gru.py --mode train             # 仅训练 (需先完成预处理)
  python solar_gru.py --mode evaluate          # 仅评估
  python solar_gru.py --csv-path /path/to/solar_station_1.csv
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import pandas as pd
import pickle
import os
import argparse


# ============================================================
# 评估指标 (不过滤夜间零功率时刻)
# ============================================================

def calc_acc1(y_true, y_pred, cap=1.0):
    """
    ACC1 (MAE-based): 1 - MAE / mean(y_true)
    参考 solar_train_ed.py 的 calc_acc_mae, 但不屏蔽夜间零功率时刻
    """
    y_t = y_true.flatten() * cap
    y_p = y_pred.flatten() * cap
    mae = np.mean(np.abs(y_t - y_p))
    avg = np.mean(y_t)
    return max(0.0, 1.0 - mae / (avg + 1e-6))


def calc_acc2(y_true, y_pred, cap=1.0):
    """
    ACC2 (国标): 1 - sqrt((1/N) * sum(((P_M - P_P) / max(P_M, 0.2*Cap))^2))
    参考 solar_train_ed.py 的 calc_acc2, 不过滤夜间零功率时刻
    """
    p_m = y_true.flatten() * cap
    p_p = y_pred.flatten() * cap
    denom = np.maximum(p_m, 0.2 * cap)
    return max(0.0, 1.0 - np.sqrt(np.mean(((p_m - p_p) / denom) ** 2)))


def calc_rmse(y_true, y_pred, cap=1.0):
    return np.sqrt(np.mean((y_true.flatten() * cap - y_pred.flatten() * cap) ** 2))


def calc_mae(y_true, y_pred, cap=1.0):
    return np.mean(np.abs(y_true.flatten() * cap - y_pred.flatten() * cap))


# ============================================================
# 数据预处理
# ============================================================

def load_and_clean(csv_path):
    """加载 solar_station_1.csv 并清洗异常值"""
    df = pd.read_csv(csv_path)
    print(f"原始数据: {len(df)} 条记录")
    print(f"列: {list(df.columns)}")

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    weather_cols = ["tsi", "dni", "ghi", "temp", "atm", "rh"]

    # 将 -99 替换为 NaN (传感器无效值)
    for col in weather_cols:
        n_bad = (df[col] == -99).sum()
        if n_bad > 0:
            df.loc[df[col] == -99, col] = np.nan
            print(f"  {col}: 替换 {n_bad} 个 -99 值")

    # 修复 rh 异常值 (>100% 视为传感器故障)
    n_rh_bad = (df["rh"] > 100).sum()
    if n_rh_bad > 0:
        df.loc[df["rh"] > 100, "rh"] = np.nan
        print(f"  rh: 替换 {n_rh_bad} 个 >100% 异常值")

    # 线性插值填充缺失值
    df[weather_cols] = df[weather_cols].interpolate(method="linear")
    df[weather_cols] = df[weather_cols].bfill().ffill()

    # 功率不能为负
    df["power"] = df["power"].clip(lower=0)

    print(f"清洗后缺失值: {df.isnull().sum().sum()}")
    return df


def build_features(df):
    """
    构建 13 维特征:
      时间特征 (6): hour_sin, hour_cos, month_sin, month_cos, dayofyear_sin, dayofyear_cos
      气象特征 (6): tsi, dni, ghi, temp, atm, rh  (Min-Max 归一化)
      功率特征 (1): power / cap
    """
    # --- 时间特征 (sin/cos 周期编码) ---
    hour = df["time"].dt.hour + df["time"].dt.minute / 60
    month = df["time"].dt.month
    dayofyear = df["time"].dt.dayofyear

    time_feats = np.column_stack([
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * (month - 1) / 12),
        np.cos(2 * np.pi * (month - 1) / 12),
        np.sin(2 * np.pi * dayofyear / 365),
        np.cos(2 * np.pi * dayofyear / 365),
    ])

    # --- 气象特征 (Min-Max 归一化) ---
    weather_cols = ["tsi", "dni", "ghi", "temp", "atm", "rh"]
    norm_params = {}
    weather_normed = []

    for col in weather_cols:
        vmin = float(df[col].min())
        vmax = float(df[col].max())
        norm_params[col] = {"min": vmin, "max": vmax}
        normed = (df[col].values - vmin) / (vmax - vmin + 1e-8)
        weather_normed.append(normed)

    # rh 归一化范围验证 (确认异常值已修复)
    print(f"  rh 归一化范围: min={norm_params['rh']['min']:.1f}, max={norm_params['rh']['max']:.1f}")

    weather_feats = np.column_stack(weather_normed)

    # --- 功率特征 (按标称容量归一化) ---
    cap = float(df["cap"].iloc[0])
    power_normed = (df["power"].values / cap).astype(np.float32)
    norm_params["power"] = {"cap": cap}

    # --- 拼接: [时间(6), 气象(6), 功率(1)] = 13维 ---
    features = np.column_stack([time_feats, weather_feats, power_normed]).astype(np.float32)
    targets = power_normed  # shape: [N]

    print(f"特征维度: {features.shape[1]} (时间6 + 气象6 + 功率1)")
    print(f"标称容量: {cap} MW")
    return features, targets, norm_params


def create_sequences(features, targets, in_steps=96, out_steps=16):
    """
    创建滑动窗口序列 (15分钟粒度):
      in_steps=96  -> 24h 历史作为输入
      out_steps=16 -> 4h 预测作为目标

    返回:
      X: [N, in_steps, feat_size]  输入序列
      y: [N, out_steps]            预测目标
    """
    X_list, y_list = [], []
    total = in_steps + out_steps

    for i in range(len(features) - total + 1):
        X_list.append(features[i: i + in_steps])
        y_list.append(targets[i + in_steps: i + total])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    print(f"序列创建完成:")
    print(f"  输入: {X.shape}  ({in_steps}步 = {in_steps * 15 / 60:.0f}h 历史)")
    print(f"  目标: {y.shape}  ({out_steps}步 = {out_steps * 15 / 60:.0f}h 预测)")
    return X, y


def preprocess(csv_path, in_steps=96, out_steps=16,
               train_ratio=0.7, val_ratio=0.15):
    """完整预处理流程: 加载 -> 清洗 -> 特征提取 -> 序列 -> 划分保存"""
    print("=" * 60)
    print("数据预处理")
    print("=" * 60)

    df = load_and_clean(csv_path)
    features, targets, norm_params = build_features(df)
    X, y = create_sequences(features, targets, in_steps, out_steps)

    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    splits = {
        "train": (X[:train_end],      y[:train_end]),
        "val":   (X[train_end:val_end], y[train_end:val_end]),
        "test":  (X[val_end:],         y[val_end:]),
    }

    print(f"\n数据集划分 (总计 {n} 样本):")
    for name, (xb, yb) in splits.items():
        np.save(f"X_{name}.npy", xb)
        np.save(f"y_{name}.npy", yb)
        print(f"  {name}: {len(xb)} 样本")

    with open("norm_params.pkl", "wb") as f:
        pickle.dump(norm_params, f)

    print("\n预处理完成!")
    return norm_params


# ============================================================
# GRU 模型
# ============================================================

class SolarGRU(nn.Module):
    """
    GRU 太阳能功率预测模型

    架构:
      1. 多层双向 GRU 编码历史序列
      2. LayerNorm 稳定训练
      3. 全连接预测头输出未来 out_steps 步功率

    输入: [batch, in_steps, feat_size]
    输出: [batch, out_steps]  (归一化功率, 值域 [0, 1])
    """

    def __init__(self, feat_size=13, hidden_size=256, num_layers=2,
                 out_steps=16, dropout=0.1):
        super().__init__()

        self.gru = nn.GRU(
            input_size=feat_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # 预测头: hidden -> out_steps
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, out_steps),
            nn.Sigmoid(),   # 输出值域 [0, 1] (对应归一化功率)
        )

    def forward(self, x):
        # x: [B, T, F]
        out, _ = self.gru(x)             # out: [B, T, H]
        h = self.norm(out[:, -1, :])     # 取最后时间步的隐状态
        h = self.dropout(h)
        return self.head(h)              # [B, out_steps]


# ============================================================
# 训练
# ============================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    X_train = np.load("X_train.npy")
    y_train = np.load("y_train.npy")
    X_val   = np.load("X_val.npy")
    y_val   = np.load("y_val.npy")

    with open("norm_params.pkl", "rb") as f:
        norm_params = pickle.load(f)
    cap = norm_params["power"]["cap"]

    feat_size = X_train.shape[2]
    out_steps = y_train.shape[1]

    print(f"训练集: X={X_train.shape}, y={y_train.shape}")
    print(f"验证集: X={X_val.shape},   y={y_val.shape}")
    print(f"标称容量: {cap} MW | 特征维度: {feat_size} | 预测步数: {out_steps}")

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    model = SolarGRU(
        feat_size=feat_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        out_steps=out_steps,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    os.makedirs("gru_checkpoints", exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n{'='*70}")
    print(f"开始训练 | Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"模型: hidden={args.hidden_size}, layers={args.num_layers}, dropout={args.dropout}")
    print(f"评估: ACC1/ACC2 不过滤夜间零功率时刻")
    print(f"{'='*70}\n")

    for epoch in range(args.epochs):
        # --- 训练阶段 ---
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                pred = model(X_b.to(device))
                val_loss += criterion(pred, y_b.to(device)).item()
                all_preds.append(pred.cpu().numpy())
                all_targets.append(y_b.numpy())
        val_loss /= len(val_loader)

        all_preds   = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        acc1 = calc_acc1(all_targets, all_preds, cap)
        acc2 = calc_acc2(all_targets, all_preds, cap)
        rmse = calc_rmse(all_targets, all_preds, cap)
        mae  = calc_mae(all_targets, all_preds, cap)
        lr   = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | LR: {lr:.2e} | "
            f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
            f"ACC1: {acc1:.4f} | ACC2: {acc2:.4f} | "
            f"RMSE: {rmse:.2f} MW | MAE: {mae:.2f} MW"
        )

        # --- 保存最佳模型 ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "acc1": acc1,
                "acc2": acc2,
                "norm_params": norm_params,
                "args": vars(args),
            }, "gru_checkpoints/best_model.pth")
            print(f"  -> 保存最佳模型 (Val Loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping: 连续 {args.patience} 轮无改善")
                break

        scheduler.step()

    print(f"\n训练完成! 最佳验证损失: {best_val_loss:.6f}")


# ============================================================
# 评估
# ============================================================

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")

    with open("norm_params.pkl", "rb") as f:
        norm_params = pickle.load(f)
    cap = norm_params["power"]["cap"]

    feat_size = X_test.shape[2]
    out_steps = y_test.shape[1]

    # 从 checkpoint 恢复模型结构参数
    checkpoint = torch.load("gru_checkpoints/best_model.pth",
                            map_location=device, weights_only=False)
    saved_args = checkpoint["args"]

    model = SolarGRU(
        feat_size=feat_size,
        hidden_size=saved_args["hidden_size"],
        num_layers=saved_args["num_layers"],
        out_steps=out_steps,
        dropout=saved_args["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"加载模型: epoch={checkpoint['epoch']+1}, "
          f"val_loss={checkpoint['val_loss']:.6f}, "
          f"训练集 ACC2={checkpoint['acc2']:.4f}")

    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            pred = model(X_b.to(device))
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_b.numpy())

    all_preds   = np.clip(np.concatenate(all_preds), 0, None)
    all_targets = np.concatenate(all_targets)

    acc1 = calc_acc1(all_targets, all_preds, cap)
    acc2 = calc_acc2(all_targets, all_preds, cap)
    rmse = calc_rmse(all_targets, all_preds, cap)
    mae  = calc_mae(all_targets, all_preds, cap)

    print(f"\n{'='*60}")
    print(f"测试集评估结果 (不过滤夜间零功率时刻)")
    print(f"{'='*60}")
    print(f"  ACC1 (MAE-based): {acc1:.4f} ({acc1*100:.2f}%)")
    print(f"  ACC2 (国标):      {acc2:.4f} ({acc2*100:.2f}%)")
    print(f"  RMSE:             {rmse:.2f} MW")
    print(f"  MAE:              {mae:.2f} MW")

    # 按预测时间范围逐小时分析
    steps_per_hour = 4   # 15min 粒度, 4步 = 1h
    n_hours = out_steps // steps_per_hour
    print(f"\n按预测时间范围 (逐小时):")
    for h in range(n_hours):
        start, end = h * steps_per_hour, (h + 1) * steps_per_hour
        yt = all_targets[:, start:end]
        yp = all_preds[:, start:end]
        h_acc1 = calc_acc1(yt, yp, cap)
        h_acc2 = calc_acc2(yt, yp, cap)
        h_rmse = calc_rmse(yt, yp, cap)
        h_mae  = calc_mae(yt, yp, cap)
        print(f"  {h}-{h+1}h: ACC1={h_acc1:.4f}, ACC2={h_acc2:.4f}, "
              f"RMSE={h_rmse:.2f} MW, MAE={h_mae:.2f} MW")

    print(f"{'='*60}")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="太阳能电站功率预测 GRU 模型",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- 运行模式 ---
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "preprocess", "train", "evaluate"],
                        help="运行模式")
    parser.add_argument("--csv-path", type=str,
                        default="solar_station_1.csv",
                        help="solar_station_1.csv 文件路径")

    # --- 序列参数 ---
    parser.add_argument("--in-steps", type=int, default=96,
                        help="输入历史步数 (96步=24h, 15分钟粒度)")
    parser.add_argument("--out-steps", type=int, default=16,
                        help="预测步数 (16步=4h, 15分钟粒度)")

    # --- 训练参数 ---
    parser.add_argument("--epochs", type=int, default=100,
                        help="最大训练轮次")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="初始学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="L2 正则化系数")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early Stopping 耐心轮数")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="梯度裁剪阈值")

    # --- 模型结构 ---
    parser.add_argument("--hidden-size", type=int, default=256,
                        help="GRU 隐层维度")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="GRU 层数")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout 比率")

    args = parser.parse_args()

    if args.mode in ["all", "preprocess"]:
        print("\n[1/3] 数据预处理")
        preprocess(args.csv_path, args.in_steps, args.out_steps)

    if args.mode in ["all", "train"]:
        print("\n[2/3] 模型训练")
        train(args)

    if args.mode in ["all", "evaluate"]:
        print("\n[3/3] 模型评估")
        evaluate(args)


if __name__ == "__main__":
    main()
