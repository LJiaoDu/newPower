#!/usr/bin/env python3
"""
太阳能电站功率预测 - LSTM Seq2Seq 训练脚本

任务: 用过去 24h (96步, 15分钟粒度) 预测未来 4h (16步)

模型结构:
  Encoder LSTM: 输入 96步 × 13特征 (时间6 + 气象6 + 历史功率1)
  Decoder LSTM: 逐步生成, 输入 (时间6 + 气象6) + Encoder 上下文
  输出: 16步预测功率 (归一化后)

准确度计算参考: claude/analyze-solar-station-model-n01ct 的 solar_train_ed.py
  ACC1 = 1 - MAE / mean(y_true)   [仅非零样本]
  ACC2 = 1 - sqrt(mean(((P_M - P_P) / max(P_M, 0.2*Cap))^2))  [国标]

使用方式:
  python solar_lstm.py                         # 完整流程
  python solar_lstm.py --mode train            # 仅训练
  python solar_lstm.py --mode evaluate         # 仅评估
  python solar_lstm.py --csv-path /path/to/solar_station_1.csv
"""

import os
import argparse
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# =====================================================================
# 数据预处理
# =====================================================================

def load_and_clean(csv_path):
    """加载 solar_station_1.csv 并清洗异常值"""
    df = pd.read_csv(csv_path)
    print(f"原始数据: {len(df)} 条记录, 列: {list(df.columns)}")

    df["time"] = pd.to_datetime(df["time"])
    weather_cols = ["tsi", "dni", "ghi", "temp", "atm", "rh"]

    # 将 -99 替换为 NaN
    for col in weather_cols:
        n_bad = (df[col] == -99).sum()
        if n_bad > 0:
            df.loc[df[col] == -99, col] = np.nan
            print(f"  {col}: 替换 {n_bad} 个 -99 值为 NaN")

    # rh > 100% 视为传感器故障
    n_rh_bad = (df["rh"] > 100).sum()
    if n_rh_bad > 0:
        df.loc[df["rh"] > 100, "rh"] = np.nan
        print(f"  rh: 替换 {n_rh_bad} 个 >100 的异常值为 NaN")

    # 线性插值
    df[weather_cols] = df[weather_cols].interpolate(method="linear")
    df[weather_cols] = df[weather_cols].bfill().ffill()

    print(f"清洗后缺失值总数: {df.isnull().sum().sum()}")
    return df


def extract_features(df):
    """
    特征提取:
      enc_features: [N, 13]  时间(6) + 气象(6) + 功率(1)
      dec_features: [N, 12]  时间(6) + 气象(6)   (无功率)
      targets:      [N]      归一化功率 (0~1)
    """
    # 时间特征 (sin/cos 周期编码)
    hour      = df["time"].dt.hour + df["time"].dt.minute / 60
    month     = df["time"].dt.month
    dayofyear = df["time"].dt.dayofyear

    time_feats = np.column_stack([
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * (month - 1) / 12),
        np.cos(2 * np.pi * (month - 1) / 12),
        np.sin(2 * np.pi * dayofyear / 365),
        np.cos(2 * np.pi * dayofyear / 365),
    ]).astype(np.float32)

    # 气象特征 (Min-Max 归一化)
    weather_cols = ["tsi", "dni", "ghi", "temp", "atm", "rh"]
    norm_params = {}
    weather_normed = []
    for col in weather_cols:
        vmin = float(df[col].min())
        vmax = float(df[col].max())
        norm_params[col] = {"min": vmin, "max": vmax}
        normed = (df[col].values - vmin) / (vmax - vmin + 1e-8)
        weather_normed.append(normed.astype(np.float32))
    weather_feats = np.column_stack(weather_normed)

    # 功率 (按标称容量归一化)
    cap = float(df["cap"].iloc[0])
    power_normed = (df["power"].values / cap).astype(np.float32)
    norm_params["power"] = {"cap": cap}

    dec_features = np.hstack([time_feats, weather_feats])                      # [N, 12]
    enc_features = np.hstack([dec_features, power_normed.reshape(-1, 1)])      # [N, 13]
    targets = power_normed                                                       # [N]

    print(f"特征: enc={enc_features.shape}, dec={dec_features.shape}, cap={cap} MW")
    print(f"rh 范围: min={norm_params['rh']['min']:.1f}, max={norm_params['rh']['max']:.1f}")
    return enc_features, dec_features, targets, norm_params


def create_sequences(enc_features, dec_features, targets, in_steps=96, out_steps=16):
    """
    滑动窗口: 96步历史 -> 16步预测 (15分钟粒度: 24h -> 4h)
    返回:
      X_enc: [N, 96, 13]
      X_dec: [N, 16, 12]
      y:     [N, 16]
    """
    total = in_steps + out_steps
    n = len(enc_features) - total + 1
    X_enc = np.lib.stride_tricks.sliding_window_view(
        enc_features, (total, enc_features.shape[1])
    )[:, 0, :in_steps, :]
    X_dec = np.lib.stride_tricks.sliding_window_view(
        dec_features, (total, dec_features.shape[1])
    )[:, 0, in_steps:, :]
    y_arr = np.lib.stride_tricks.sliding_window_view(
        targets, total
    )[:n, in_steps:]

    X_enc = X_enc[:n].astype(np.float32)
    X_dec = X_dec[:n].astype(np.float32)
    y_arr = y_arr[:n].astype(np.float32)

    print(f"序列: X_enc={X_enc.shape}, X_dec={X_dec.shape}, y={y_arr.shape}")
    return X_enc, X_dec, y_arr


def filter_nighttime(X_enc, X_dec, y, threshold=0.01):
    """过滤纯夜间样本 (16步全部 < threshold)"""
    n_before = len(y)
    mask = np.any(y > threshold, axis=1)
    n_after = mask.sum()
    print(f"  夜间过滤: {n_before} -> {n_after} (移除 {n_before-n_after} 纯夜间样本, "
          f"{(n_before-n_after)/n_before*100:.1f}%)")
    return X_enc[mask], X_dec[mask], y[mask]


def preprocess(csv_path="solar_station_1.csv", output_dir="."):
    """完整预处理流程, 输出 .npy 文件和 norm_params.pkl"""
    print("=" * 60)
    print("数据预处理")
    print("=" * 60)

    df = load_and_clean(csv_path)
    enc_feats, dec_feats, targets, norm_params = extract_features(df)
    X_enc, X_dec, y = create_sequences(enc_feats, dec_feats, targets)

    n = len(X_enc)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    splits = {
        "train": (X_enc[:train_end],      X_dec[:train_end],      y[:train_end]),
        "val":   (X_enc[train_end:val_end], X_dec[train_end:val_end], y[train_end:val_end]),
        "test":  (X_enc[val_end:],         X_dec[val_end:],         y[val_end:]),
    }

    print("\n数据集划分:")
    for name, (xe, xd, yt) in splits.items():
        np.save(os.path.join(output_dir, f"X_enc_{name}.npy"), xe)
        np.save(os.path.join(output_dir, f"X_dec_{name}.npy"), xd)
        np.save(os.path.join(output_dir, f"y_{name}.npy"),     yt)
        print(f"  {name}: {len(xe)} 样本")

    with open(os.path.join(output_dir, "norm_params.pkl"), "wb") as f:
        pickle.dump(norm_params, f)

    print("\n预处理完成!")


# =====================================================================
# 评估指标  (与 solar_train_ed.py 相同)
# =====================================================================

def calc_acc1(y_true, y_pred, cap=1.0):
    """ACC1 (MAE-based): 1 - MAE / mean(y_true), 仅非零样本"""
    mask = y_true > 0.01
    if mask.sum() == 0:
        return float("nan")
    y_t = y_true[mask] * cap
    y_p = y_pred[mask] * cap
    mae = np.mean(np.abs(y_t - y_p))
    avg = np.mean(y_t)
    return max(0.0, 1.0 - mae / (avg + 1e-6))


def calc_acc2(y_true, y_pred, cap=1.0):
    """ACC2 (国标): 1 - sqrt(mean(((P_M - P_P) / max(P_M, 0.2*Cap))^2))"""
    p_m = y_true.flatten() * cap
    p_p = y_pred.flatten() * cap
    denom = np.maximum(p_m, 0.2 * cap)
    return max(0.0, 1.0 - np.sqrt(np.mean(((p_m - p_p) / denom) ** 2)))


def calc_rmse(y_true, y_pred, cap=1.0):
    return np.sqrt(np.mean((y_true * cap - y_pred * cap) ** 2))


def calc_mae(y_true, y_pred, cap=1.0):
    return np.mean(np.abs(y_true * cap - y_pred * cap))


def print_metrics(y_true, y_pred, cap, label="评估结果"):
    acc1 = calc_acc1(y_true, y_pred, cap)
    acc2 = calc_acc2(y_true, y_pred, cap)
    rmse = calc_rmse(y_true, y_pred, cap)
    mae  = calc_mae(y_true, y_pred, cap)
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    print(f"  ACC1 (MAE-based): {acc1:.4f}  ({acc1*100:.2f}%)")
    print(f"  ACC2 (国标):      {acc2:.4f}  ({acc2*100:.2f}%)")
    print(f"  RMSE:             {rmse:.4f}  ({rmse*cap:.2f} MW)")
    print(f"  MAE:              {mae:.4f}  ({mae*cap:.2f} MW)")

    # 按16个预测点逐一细分 (每点15分钟)
    print(f"\n  按16个预测点 (每点15分钟):")
    for i in range(16):
        yt = y_true[:, i:i+1]
        yp = y_pred[:, i:i+1]
        h_acc1 = calc_acc1(yt, yp, cap)
        h_acc2 = calc_acc2(yt, yp, cap)
        h_rmse = calc_rmse(yt, yp, cap)
        total_min = (i + 1) * 15
        if total_min % 60 == 0:
            time_label = f"+{total_min // 60}h"
        else:
            h = total_min // 60
            m = total_min % 60
            time_label = f"+{h}h{m:02d}m" if h > 0 else f"+{m}min"
        print(f"    点{i+1:2d} ({time_label:>7s}): ACC1={h_acc1:.4f}, ACC2={h_acc2:.4f}, RMSE={h_rmse*cap:.2f} MW")
    print(f"{'='*60}")
    return acc1, acc2, rmse, mae


# =====================================================================
# LSTM 模型  (Seq2Seq)
# =====================================================================

class LSTMEncoder(nn.Module):
    """
    编码 96步历史 (功率 + 气象 + 时间) -> 隐状态
    """
    def __init__(self, input_size=13, hidden_size=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x):
        # x: [B, 96, 13]
        output, (h_n, c_n) = self.lstm(x)
        # output: [B, 96, hidden_size]  (所有时间步的输出)
        # h_n, c_n: [num_layers, B, hidden_size]
        return output, h_n, c_n


class LSTMDecoder(nn.Module):
    """
    逐步解码 16步 (气象 + 时间) -> 预测功率
    输入: encoder 隐状态 + 未来气象时间特征
    """
    def __init__(self, input_size=12, hidden_size=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x_dec, h_n, c_n):
        # x_dec: [B, 16, 12]
        output, _ = self.lstm(x_dec, (h_n, c_n))
        # output: [B, 16, hidden_size]
        pred = self.fc(output).squeeze(-1)    # [B, 16]
        return pred


class SolarLSTM(nn.Module):
    """
    LSTM Seq2Seq 模型

    Encoder: 处理历史序列 (96步, 13特征)
    Decoder: 生成预测 (16步, 输入12特征 + encoder 上下文)
    """
    def __init__(self, enc_input=13, dec_input=12, hidden_size=256,
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.encoder = LSTMEncoder(enc_input, hidden_size, num_layers, dropout)
        self.decoder = LSTMDecoder(dec_input, hidden_size, num_layers, dropout)

    def forward(self, x_enc, x_dec):
        # x_enc: [B, 96, 13]
        # x_dec: [B, 16, 12]
        _, h_n, c_n = self.encoder(x_enc)
        pred = self.decoder(x_dec, h_n, c_n)  # [B, 16]
        return pred


# =====================================================================
# 混合 Loss  (与 solar_train_ed.py 相同)
# =====================================================================

class ACC2Loss(nn.Module):
    """基于国标 ACC2 的损失函数"""
    def __init__(self, cap_norm=1.0):
        super().__init__()
        self.cap_norm = cap_norm

    def forward(self, pred, target):
        denom = torch.clamp(target, min=0.2 * self.cap_norm)
        return torch.mean(((pred - target) / denom) ** 2)


class MixedLoss(nn.Module):
    """混合 Loss = λ_mse * MSE + λ_acc2 * ACC2Loss"""
    def __init__(self, lambda_mse=1.0, lambda_acc2=0.5, cap_norm=1.0):
        super().__init__()
        self.lambda_mse  = lambda_mse
        self.lambda_acc2 = lambda_acc2
        self.mse_loss    = nn.MSELoss()
        self.acc2_loss   = ACC2Loss(cap_norm)

    def forward(self, pred, target):
        return self.lambda_mse * self.mse_loss(pred, target) \
             + self.lambda_acc2 * self.acc2_loss(pred, target)


# =====================================================================
# 训练
# =====================================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载数据
    X_enc_train = np.load("X_enc_train.npy")
    X_dec_train = np.load("X_dec_train.npy")
    y_train     = np.load("y_train.npy")
    X_enc_val   = np.load("X_enc_val.npy")
    X_dec_val   = np.load("X_dec_val.npy")
    y_val       = np.load("y_val.npy")

    with open("norm_params.pkl", "rb") as f:
        norm_params = pickle.load(f)
    cap = norm_params["power"]["cap"]

    print(f"训练集: {X_enc_train.shape[0]} 样本")
    print(f"验证集: {X_enc_val.shape[0]} 样本")
    print(f"标称容量: {cap} MW")

    enc_feat_size = X_enc_train.shape[2]   # 13
    dec_feat_size = X_dec_train.shape[2]   # 12

    train_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_enc_train),
            torch.FloatTensor(X_dec_train),
            torch.FloatTensor(y_train),
        ),
        batch_size=args.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_enc_val),
            torch.FloatTensor(X_dec_val),
            torch.FloatTensor(y_val),
        ),
        batch_size=args.batch_size, shuffle=False,
    )

    # 模型
    model = SolarLSTM(
        enc_input=enc_feat_size,
        dec_input=dec_feat_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params:,}")
    print(f"结构: hidden={args.hidden_size}, layers={args.num_layers}, dropout={args.dropout}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup_sched  = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs)
    cosine_sched  = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.warmup_epochs), eta_min=1e-6)
    scheduler     = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[args.warmup_epochs])

    criterion = MixedLoss(lambda_mse=args.lambda_mse, lambda_acc2=args.lambda_acc2)
    print(f"Loss: MixedLoss(λ_mse={args.lambda_mse}, λ_acc2={args.lambda_acc2})")

    os.makedirs("lstm_checkpoints", exist_ok=True)

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
        for batch_enc, batch_dec, batch_y in train_bar:
            batch_enc = batch_enc.to(device)
            batch_dec = batch_dec.to(device)
            batch_y   = batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_enc, batch_dec)
            loss = criterion(pred, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
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
            for batch_enc, batch_dec, batch_y in val_bar:
                batch_enc = batch_enc.to(device)
                batch_dec = batch_dec.to(device)
                batch_y   = batch_y.to(device)
                pred = model(batch_enc, batch_dec)
                batch_val_loss = criterion(pred, batch_y).item()
                val_loss += batch_val_loss
                val_bar.set_postfix(loss=f"{batch_val_loss:.4f}")
                all_preds.append(pred.cpu().numpy())
                all_trues.append(batch_y.cpu().numpy())
        val_loss /= len(val_loader)

        all_preds = np.concatenate(all_preds)
        all_trues = np.concatenate(all_trues)

        acc1 = calc_acc1(all_trues, all_preds, cap)
        acc2 = calc_acc2(all_trues, all_preds, cap)
        rmse = calc_rmse(all_trues, all_preds, cap)
        mae  = calc_mae(all_trues,  all_preds, cap)
        lr   = optimizer.param_groups[0]["lr"]

        epoch_bar.set_postfix(
            val=f"{val_loss:.4f}", acc2=f"{acc2:.4f}", best=f"{best_val_loss:.4f}"
        )
        tqdm.write(
            f"Epoch {epoch+1:3d}/{args.epochs} | LR: {lr:.6f} | "
            f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
            f"ACC1: {acc1:.4f} | ACC2: {acc2:.4f} | "
            f"RMSE: {rmse:.4f} ({rmse*cap:.2f} MW) | MAE: {mae:.4f} ({mae*cap:.2f} MW)"
        )

        # 保存最佳模型
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
            }, "lstm_checkpoints/best_model_lstm.pth")
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

    X_enc_test = np.load("X_enc_test.npy")
    X_dec_test = np.load("X_dec_test.npy")
    y_test     = np.load("y_test.npy")

    with open("norm_params.pkl", "rb") as f:
        norm_params = pickle.load(f)
    cap = norm_params["power"]["cap"]

    enc_feat_size = X_enc_test.shape[2]
    dec_feat_size = X_dec_test.shape[2]

    # 从 checkpoint 恢复 args
    ckpt = torch.load("lstm_checkpoints/best_model_lstm.pth",
                      map_location=device, weights_only=False)
    saved_args = ckpt["args"]
    hidden_size = saved_args.get("hidden_size", args.hidden_size)
    num_layers  = saved_args.get("num_layers",  args.num_layers)
    dropout     = saved_args.get("dropout",     args.dropout)

    model = SolarLSTM(
        enc_input=enc_feat_size,
        dec_input=dec_feat_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"加载模型: epoch={ckpt['epoch']+1}, val_loss={ckpt['val_loss']:.6f}, "
          f"acc1={ckpt['acc1']:.4f}, acc2={ckpt['acc2']:.4f}")

    test_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_enc_test),
            torch.FloatTensor(X_dec_test),
            torch.FloatTensor(y_test),
        ),
        batch_size=args.batch_size, shuffle=False,
    )

    all_preds = []
    all_trues = []
    with torch.no_grad():
        for batch_enc, batch_dec, batch_y in test_loader:
            pred = model(batch_enc.to(device), batch_dec.to(device))
            all_preds.append(pred.cpu().numpy())
            all_trues.append(batch_y.numpy())

    all_preds = np.concatenate(all_preds)
    all_trues = np.concatenate(all_trues)

    print_metrics(all_trues, all_preds, cap, label="测试集评估结果 (LSTM Seq2Seq)")


# =====================================================================
# 主入口
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="太阳能电站功率预测 - LSTM Seq2Seq")

    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "preprocess", "train", "evaluate"],
                        help="运行模式")
    parser.add_argument("--csv-path", type=str,
                        default="solar_station_1.csv",
                        help="solar_station_1.csv 路径")

    # 训练参数
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch-size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience",     type=int,   default=15)
    parser.add_argument("--warmup-epochs",type=int,   default=5)
    parser.add_argument("--grad-clip",    type=float, default=1.0)

    # 模型结构
    parser.add_argument("--hidden-size",  type=int,   default=256,
                        help="LSTM 隐藏层大小")
    parser.add_argument("--num-layers",   type=int,   default=2,
                        help="LSTM 层数")
    parser.add_argument("--dropout",      type=float, default=0.2,
                        help="Dropout 比率")

    # 混合 Loss 权重
    parser.add_argument("--lambda-mse",  type=float, default=1.0,
                        help="MSE Loss 权重")
    parser.add_argument("--lambda-acc2", type=float, default=0.5,
                        help="ACC2 Loss 权重")

    args = parser.parse_args()

    if args.mode in ["all", "preprocess"]:
        print("\n[1/3] 数据预处理")
        preprocess(csv_path=args.csv_path)

    if args.mode in ["all", "train"]:
        print("\n[2/3] 模型训练")
        train(args)

    if args.mode in ["all", "evaluate"]:
        print("\n[3/3] 模型评估")
        evaluate(args)


if __name__ == "__main__":
    main()
