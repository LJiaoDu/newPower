#!/usr/bin/env python3
"""
太阳能电站功率预测 - 简单 LSTM Baseline

架构 (对比 solar_lstm_train.py Seq2Seq):
  Seq2Seq:  Encoder LSTM (288步) + Decoder LSTM (48步, 需未来时间特征)
  Baseline: 单 LSTM (288步) -> 取最后隐状态 -> FC -> 直接输出48步

输入: [B, 288, 7]  (时间sin/cos×3=6 + 历史功率=1)
输出: [B, 48]      (未来 4h 每5分钟功率, 归一化)

归一化: 功率 / max_power  (与 solar_lstm_train.py 相同)
.npy 数据文件: 与 solar_lstm_train.py 共用 (X_enc_*.npy, y_*.npy)

使用方式:
  python solar_lstm_baseline.py                    # 完整流程 (需先有 training_data.csv)
  python solar_lstm_baseline.py --mode train       # 仅训练 (需先预处理)
  python solar_lstm_baseline.py --mode evaluate    # 仅评估
  python solar_lstm_baseline.py --csv-path /path/to/training_data.csv
"""

import os
import argparse
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# =====================================================================
# 数据预处理 (与 solar_lstm_train.py 相同逻辑, 共用 norm_params_v2.pkl)
# =====================================================================

def preprocess(csv_path="training_data.csv", output_dir=".", in_steps=288, out_steps=48):
    """完整预处理: 加载 -> 特征提取 -> 切序列 -> 保存 .npy"""
    print("=" * 60)
    print("数据预处理")
    print("=" * 60)

    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"原始数据: {len(df)} 条记录")

    # 负值置零
    df.loc[df["generationPower"] < 0, "generationPower"] = 0.0

    # 时间特征 (sin/cos 周期编码)
    hour      = df["datetime"].dt.hour + df["datetime"].dt.minute / 60
    month     = df["datetime"].dt.month
    dayofyear = df["datetime"].dt.dayofyear

    time_feats = np.column_stack([
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * (month - 1) / 12),
        np.cos(2 * np.pi * (month - 1) / 12),
        np.sin(2 * np.pi * dayofyear / 365),
        np.cos(2 * np.pi * dayofyear / 365),
    ]).astype(np.float32)   # [N, 6]

    # 功率归一化: 用历史最大值代替标称容量
    max_power = float(df["generationPower"].max())
    power_normed = (df["generationPower"].values / max_power).astype(np.float32)

    features = np.hstack([time_feats, power_normed.reshape(-1, 1)])  # [N, 7]
    targets  = power_normed                                           # [N]

    print(f"max_power: {max_power:.2f} W  ({max_power/1000:.2f} kW)")
    print(f"特征维度: {features.shape[1]}")

    # 滑动窗口
    total = in_steps + out_steps
    n = len(features) - total + 1

    X = np.lib.stride_tricks.sliding_window_view(
        features, (total, features.shape[1])
    )[:, 0, :in_steps, :]          # [n, in_steps, 7]
    y = np.lib.stride_tricks.sliding_window_view(
        targets, total
    )[:n, in_steps:]               # [n, out_steps]

    X = X[:n].astype(np.float32)
    y = y[:n].astype(np.float32)
    print(f"序列: X={X.shape}, y={y.shape}")

    # 划分数据集
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    splits = {
        "train": (X[:train_end],       y[:train_end]),
        "val":   (X[train_end:val_end], y[train_end:val_end]),
        "test":  (X[val_end:],          y[val_end:]),
    }
    print("\n数据集划分:")
    for name, (xi, yi) in splits.items():
        np.save(os.path.join(output_dir, f"X_enc_{name}.npy"), xi)
        np.save(os.path.join(output_dir, f"y_{name}.npy"),     yi)
        print(f"  {name}: {len(xi)} 样本")

    norm_params = {
        "power":     {"max": max_power},
        "in_steps":  in_steps,
        "out_steps": out_steps,
    }
    with open(os.path.join(output_dir, "norm_params_v2.pkl"), "wb") as f:
        pickle.dump(norm_params, f)

    print("预处理完成!")


# =====================================================================
# 评估指标
# =====================================================================

def calc_acc1(y_true, y_pred, cap):
    mask = y_true > 0.01
    if mask.sum() == 0:
        return float("nan")
    t, p = y_true[mask] * cap, y_pred[mask] * cap
    return max(0.0, 1.0 - np.mean(np.abs(t - p)) / (np.mean(t) + 1e-6))


def calc_acc2(y_true, y_pred, cap):
    p_m = y_true.flatten() * cap
    p_p = y_pred.flatten() * cap
    denom = np.maximum(p_m, 0.2 * cap)
    return max(0.0, 1.0 - np.sqrt(np.mean(((p_m - p_p) / denom) ** 2)))


def calc_rmse(y_true, y_pred, cap):
    return np.sqrt(np.mean((y_true * cap - y_pred * cap) ** 2))


def calc_mae(y_true, y_pred, cap):
    return np.mean(np.abs(y_true * cap - y_pred * cap))


def _time_label(step_idx, step_min=5):
    total = (step_idx + 1) * step_min
    if total % 60 == 0:
        return f"+{total // 60}h"
    h, m = total // 60, total % 60
    return f"+{h}h{m:02d}m" if h else f"+{m}min"


def print_metrics(y_true, y_pred, max_power, out_steps=48, label="评估结果"):
    acc1 = calc_acc1(y_true, y_pred, max_power)
    acc2 = calc_acc2(y_true, y_pred, max_power)
    rmse = calc_rmse(y_true, y_pred, max_power)
    mae  = calc_mae(y_true, y_pred, max_power)

    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    print(f"  ACC1 (MAE-based): {acc1:.4f}  ({acc1*100:.2f}%)")
    print(f"  ACC2 (国标):      {acc2:.4f}  ({acc2*100:.2f}%)")
    print(f"  RMSE:             {rmse/1000:.4f} kW  ({rmse:.2f} W)")
    print(f"  MAE:              {mae/1000:.4f} kW  ({mae:.2f} W)")

    print(f"\n  按{out_steps}个预测点 (每点5分钟):")
    for i in range(out_steps):
        yt, yp = y_true[:, i:i+1], y_pred[:, i:i+1]
        print(f"    点{i+1:2d} ({_time_label(i):>7s}): "
              f"ACC1={calc_acc1(yt, yp, max_power):.4f}, "
              f"ACC2={calc_acc2(yt, yp, max_power):.4f}, "
              f"RMSE={calc_rmse(yt, yp, max_power)/1000:.2f} kW")
    print(f"{'='*60}")
    return acc1, acc2, rmse, mae


# =====================================================================
# 模型: 简单 LSTM Baseline
# =====================================================================

class SimpleLSTM(nn.Module):
    """
    LSTM Baseline:
      1. LSTM 处理输入序列 [B, in_steps, feat]
      2. 取最后时间步隐状态 [B, hidden]
      3. 全连接层直接输出 out_steps 步预测 [B, out_steps]
    """
    def __init__(self, input_size=7, hidden_size=128, num_layers=2,
                 out_steps=48, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_steps),
        )

    def forward(self, x):
        # x: [B, in_steps, input_size]
        out, _ = self.lstm(x)      # [B, in_steps, hidden]
        last   = out[:, -1, :]    # [B, hidden]  取最后时间步
        return self.fc(last)       # [B, out_steps]


# =====================================================================
# 训练
# =====================================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    X_train = np.load("X_enc_train.npy")
    y_train = np.load("y_train.npy")
    X_val   = np.load("X_enc_val.npy")
    y_val   = np.load("y_val.npy")

    with open("norm_params_v2.pkl", "rb") as f:
        norm_params = pickle.load(f)
    max_power = norm_params["power"]["max"]
    out_steps = norm_params.get("out_steps", y_train.shape[1])

    print(f"训练集: {X_train.shape[0]} 样本, 输入={X_train.shape[1:]},"
          f" 输出={y_train.shape[1:]}")
    print(f"验证集: {X_val.shape[0]} 样本")
    print(f"max_power: {max_power:.2f} W  ({max_power/1000:.2f} kW)")

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=args.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        batch_size=args.batch_size, shuffle=False,
    )

    feat_size = X_train.shape[2]   # 7
    model = SimpleLSTM(
        input_size=feat_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        out_steps=out_steps,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params:,}")
    print(f"结构: SimpleLSTM(hidden={args.hidden_size}, layers={args.num_layers}) + FC")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.MSELoss()

    os.makedirs("lstm_baseline_ckpt", exist_ok=True)

    best_val_loss    = float("inf")
    patience_counter = 0

    print(f"\n{'='*70}")
    print(f"开始训练 | Epochs={args.epochs} | BatchSize={args.batch_size} | LR={args.lr}")
    print(f"{'='*70}\n")

    epoch_bar = tqdm(range(args.epochs), desc="训练进度", unit="epoch")
    for epoch in epoch_bar:
        # --- 训练 ---
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"  Ep{epoch+1:3d} 训练",
                         leave=False, unit="batch")
        for X_b, y_b in train_bar:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss /= len(train_loader)

        # --- 验证 ---
        model.eval()
        val_loss  = 0.0
        all_preds = []
        all_trues = []
        val_bar = tqdm(val_loader, desc=f"  Ep{epoch+1:3d} 验证",
                       leave=False, unit="batch")
        with torch.no_grad():
            for X_b, y_b in val_bar:
                X_b, y_b = X_b.to(device), y_b.to(device)
                pred = model(X_b)
                bvl  = criterion(pred, y_b).item()
                val_loss += bvl
                val_bar.set_postfix(loss=f"{bvl:.4f}")
                all_preds.append(pred.cpu().numpy())
                all_trues.append(y_b.cpu().numpy())
        val_loss /= len(val_loader)

        all_preds = np.concatenate(all_preds)
        all_trues = np.concatenate(all_trues)

        acc1 = calc_acc1(all_trues, all_preds, max_power)
        acc2 = calc_acc2(all_trues, all_preds, max_power)
        rmse = calc_rmse(all_trues, all_preds, max_power)
        mae  = calc_mae(all_trues,  all_preds, max_power)
        lr   = optimizer.param_groups[0]["lr"]

        epoch_bar.set_postfix(
            val=f"{val_loss:.4f}", acc2=f"{acc2:.4f}", best=f"{best_val_loss:.4f}"
        )
        tqdm.write(
            f"Epoch {epoch+1:3d}/{args.epochs} | LR: {lr:.6f} | "
            f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
            f"ACC1: {acc1:.4f} | ACC2: {acc2:.4f} | "
            f"RMSE: {rmse/1000:.2f} kW | MAE: {mae/1000:.2f} kW"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_loss":         val_loss,
                "acc1":             acc1,
                "acc2":             acc2,
                "norm_params":      norm_params,
                "args":             vars(args),
            }, "lstm_baseline_ckpt/best_model.pth")
            tqdm.write(f"  -> 保存最佳模型 (val_loss={val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                tqdm.write(f"\nEarly stopping: {args.patience} epochs 无改善")
                break

        scheduler.step()

    print(f"\n训练完成! 最佳验证 Loss: {best_val_loss:.6f}")


# =====================================================================
# 评估
# =====================================================================

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_test = np.load("X_enc_test.npy")
    y_test = np.load("y_test.npy")

    with open("norm_params_v2.pkl", "rb") as f:
        norm_params = pickle.load(f)
    max_power = norm_params["power"]["max"]
    out_steps = norm_params.get("out_steps", y_test.shape[1])

    ckpt = torch.load("lstm_baseline_ckpt/best_model.pth",
                      map_location=device, weights_only=False)
    saved = ckpt["args"]
    model = SimpleLSTM(
        input_size=X_test.shape[2],
        hidden_size=saved.get("hidden_size", args.hidden_size),
        num_layers=saved.get("num_layers",  args.num_layers),
        out_steps=out_steps,
        dropout=saved.get("dropout", args.dropout),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"加载模型: epoch={ckpt['epoch']+1}, val_loss={ckpt['val_loss']:.6f}, "
          f"acc1={ckpt['acc1']:.4f}, acc2={ckpt['acc2']:.4f}")

    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
        batch_size=args.batch_size, shuffle=False,
    )

    all_preds, all_trues = [], []
    with torch.no_grad():
        for X_b, y_b in tqdm(loader, desc="测试推理"):
            all_preds.append(model(X_b.to(device)).cpu().numpy())
            all_trues.append(y_b.numpy())

    all_preds = np.concatenate(all_preds)
    all_trues = np.concatenate(all_trues)

    print_metrics(all_trues, all_preds, max_power, out_steps=out_steps,
                  label="测试集评估结果 (SimpleLSTM Baseline)")


# =====================================================================
# 主入口
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="简单 LSTM Baseline - training_data.csv")

    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "preprocess", "train", "evaluate"])
    parser.add_argument("--csv-path",  type=str, default="training_data.csv")
    parser.add_argument("--in-steps",  type=int, default=288,
                        help="输入步数 (288 = 24h @ 5min)")
    parser.add_argument("--out-steps", type=int, default=48,
                        help="预测步数 (48 = 4h @ 5min)")

    # 训练参数
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--patience",   type=int,   default=15)

    # 模型结构
    parser.add_argument("--hidden-size", type=int,   default=128,
                        help="LSTM 隐藏层大小 (Seq2Seq 用 256, Baseline 用 128)")
    parser.add_argument("--num-layers",  type=int,   default=2)
    parser.add_argument("--dropout",     type=float, default=0.2)

    args = parser.parse_args()

    if args.mode in ["all", "preprocess"]:
        print("\n[1/3] 数据预处理")
        preprocess(csv_path=args.csv_path, in_steps=args.in_steps, out_steps=args.out_steps)

    if args.mode in ["all", "train"]:
        print("\n[2/3] 模型训练")
        train(args)

    if args.mode in ["all", "evaluate"]:
        print("\n[3/3] 模型评估")
        evaluate(args)


if __name__ == "__main__":
    main()
