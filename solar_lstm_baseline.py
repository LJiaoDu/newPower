#!/usr/bin/env python3
"""
太阳能电站功率预测 - 简单 LSTM Baseline (solar_station_1.csv)

架构对比:
  Seq2Seq (solar_lstm.py):
    Encoder LSTM(96步,13feat) -> (h,c) -> Decoder LSTM(16步,12feat) -> FC -> [B,16]
    未来天气逐步输入解码器

  Baseline (本文件):
    LSTM(96步,13feat) -> 最后隐状态[B,H]
                              +
    未来天气展平[B,16*12=192]
                              ↓
    Concat -> FC -> [B,16]
    未来天气一次性拼接, 无解码器

Loss: 混合 Loss = λ_mse * MSE + λ_acc2 * ACC2_Loss
  - MSE: 绝对误差, 稳定收敛
  - ACC2 Loss: 相对误差, 与国标 ACC2 评估指标直接对齐

数据: 与 solar_lstm.py 共用预处理文件 (X_enc_*.npy, X_dec_*.npy, y_*.npy, norm_params.pkl)
      如无预处理文件, --mode all 时会自动预处理 solar_station_1.csv

使用方式:
  python solar_lstm_baseline.py                                          # 完整流程
  python solar_lstm_baseline.py --mode train                             # 仅训练
  python solar_lstm_baseline.py --mode train --lambda-mse 1.0 --lambda-acc2 0.5
  python solar_lstm_baseline.py --mode evaluate                          # 仅评估
"""

import os
import argparse
import pickle
import time
import atexit
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# =========================================================
# 仅在程序结束时保存“关键终端输出”（避免 tqdm 刷新残影写入）
# =========================================================

LOG_BUFFER = []

def log_print(*args, **kwargs):
    """
    替代 print：正常打印到终端，同时把文本缓存起来，程序结束后写入 txt
    """
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    message = sep.join(str(a) for a in args) + end

    print(*args, **kwargs)
    LOG_BUFFER.append(message.rstrip("\n"))

def log_tqdm_write(message: str):
    """
    替代 tqdm.write：写到终端（不破坏进度条），同时缓存
    """
    tqdm.write(message)
    LOG_BUFFER.append(str(message))

def save_log_to_file():
    """
    程序退出时，把 LOG_BUFFER 写入文件。
    不会记录 tqdm 的动态刷新，只记录你显式输出的文本（log_print/log_tqdm_write）。
    """
    try:
        os.makedirs("solar_logs", exist_ok=True)
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"solar_logs/final_log_lstm_baseline_{time_str}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(LOG_BUFFER) + "\n")
        print(f"\n最终日志已保存至: {filename}")
    except Exception as e:
        print(f"\n[WARN] 保存最终日志失败: {e}")

atexit.register(save_log_to_file)


# =====================================================================
# 数据预处理 (与 solar_lstm.py 相同逻辑, 共用 norm_params.pkl)
# =====================================================================

def preprocess(csv_path="solar_station_1.csv", output_dir="."):
    log_print("=" * 60)
    log_print("数据预处理")
    log_print("=" * 60)

    df = pd.read_csv(csv_path)
    log_print(f"原始数据: {len(df)} 条记录")
    df["time"] = pd.to_datetime(df["time"])

    weather_cols = ["tsi", "dni", "ghi", "temp", "atm", "rh"]
    for col in weather_cols:
        n_bad = (df[col] == -99).sum()
        if n_bad > 0:
            df.loc[df[col] == -99, col] = np.nan
            log_print(f"  {col}: 替换 {n_bad} 个 -99 -> NaN")

    n_rh = (df["rh"] > 100).sum()
    if n_rh > 0:
        df.loc[df["rh"] > 100, "rh"] = np.nan
        log_print(f"  rh: 替换 {n_rh} 个 >100 -> NaN")

    df[weather_cols] = df[weather_cols].interpolate(method="linear").bfill().ffill()

    # 时间特征
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

    # 气象特征 Min-Max
    norm_params = {}
    weather_normed = []
    for col in weather_cols:
        vmin, vmax = float(df[col].min()), float(df[col].max())
        norm_params[col] = {"min": vmin, "max": vmax}
        weather_normed.append(((df[col].values - vmin) / (vmax - vmin + 1e-8)).astype(np.float32))
    weather_feats = np.column_stack(weather_normed)

    # 功率
    cap = float(df["cap"].iloc[0])
    power_normed = (df["power"].values / cap).astype(np.float32)
    norm_params["power"] = {"cap": cap}

    dec_features = np.hstack([time_feats, weather_feats])                   # [N, 12]
    enc_features = np.hstack([dec_features, power_normed.reshape(-1, 1)])   # [N, 13]
    targets = power_normed

    log_print(f"特征: enc={enc_features.shape}, dec={dec_features.shape}, cap={cap} MW")

    # 滑动窗口 96→16
    in_steps, out_steps = 96, 16
    total = in_steps + out_steps
    n = len(enc_features) - total + 1

    X_enc = np.lib.stride_tricks.sliding_window_view(
        enc_features, (total, enc_features.shape[1])
    )[:, 0, :in_steps, :]
    X_dec = np.lib.stride_tricks.sliding_window_view(
        dec_features, (total, dec_features.shape[1])
    )[:, 0, in_steps:, :]
    y_arr = np.lib.stride_tricks.sliding_window_view(targets, total)[:n, in_steps:]

    X_enc = X_enc[:n].astype(np.float32)
    X_dec = X_dec[:n].astype(np.float32)
    y_arr = y_arr[:n].astype(np.float32)

    log_print(f"序列: X_enc={X_enc.shape}, X_dec={X_dec.shape}, y={y_arr.shape}")

    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)
    splits = {
        "train": (X_enc[:train_end],          X_dec[:train_end],          y_arr[:train_end]),
        "val":   (X_enc[train_end:val_end],   X_dec[train_end:val_end],   y_arr[train_end:val_end]),
        "test":  (X_enc[val_end:],            X_dec[val_end:],            y_arr[val_end:]),
    }

    log_print("\n数据集划分:")
    for name, (xe, xd, yt) in splits.items():
        np.save(os.path.join(output_dir, f"X_enc_{name}.npy"), xe)
        np.save(os.path.join(output_dir, f"X_dec_{name}.npy"), xd)
        np.save(os.path.join(output_dir, f"y_{name}.npy"), yt)
        log_print(f"  {name}: {len(xe)} 样本")

    with open(os.path.join(output_dir, "norm_params.pkl"), "wb") as f:
        pickle.dump(norm_params, f)

    log_print("预处理完成!")


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
    return max(0.0, 1.0 - np.sqrt(np.mean(((p_m - p_p) / np.maximum(p_m, 0.2 * cap)) ** 2)))

def calc_rmse(y_true, y_pred, cap):
    return np.sqrt(np.mean((y_true * cap - y_pred * cap) ** 2))

def calc_mae(y_true, y_pred, cap):
    return np.mean(np.abs(y_true * cap - y_pred * cap))

def print_metrics(y_true, y_pred, cap, label="评估结果"):
    acc1 = calc_acc1(y_true, y_pred, cap)
    acc2 = calc_acc2(y_true, y_pred, cap)
    rmse = calc_rmse(y_true, y_pred, cap)
    mae  = calc_mae(y_true, y_pred, cap)

    log_print(f"\n{'='*60}")
    log_print(f"{label}")
    log_print(f"{'='*60}")
    log_print(f"  ACC1 (MAE-based): {acc1:.4f}  ({acc1*100:.2f}%)")
    log_print(f"  ACC2 (国标):      {acc2:.4f}  ({acc2*100:.2f}%)")
    log_print(f"  RMSE:             {rmse:.4f}  ({rmse*cap:.2f} MW)")
    log_print(f"  MAE:              {mae:.4f}  ({mae*cap:.2f} MW)")

    log_print(f"\n  按16个预测点 (每点15分钟):")
    for i in range(16):
        yt, yp = y_true[:, i:i+1], y_pred[:, i:i+1]
        total_min = (i + 1) * 15
        if total_min % 60 == 0:
            tlabel = f"+{total_min // 60}h"
        else:
            h, m = total_min // 60, total_min % 60
            tlabel = f"+{h}h{m:02d}m" if h else f"+{m}min"

        s_acc1 = calc_acc1(yt, yp, cap)
        s_acc2 = calc_acc2(yt, yp, cap)
        s_rmse = calc_rmse(yt, yp, cap)

        log_print(
            f"    点{i+1:2d} ({tlabel:>7s}): "
            f"ACC1={s_acc1:.4f}, "
            f"ACC2={s_acc2:.4f}, "
            f"RMSE={s_rmse*cap:.2f} MW"
        )

    log_print(f"{'='*60}")
    return acc1, acc2, rmse, mae


# =====================================================================
# 混合 Loss
# =====================================================================

class ACC2Loss(nn.Module):
    """
    基于国标 ACC2 的损失函数
    L = mean( ((pred - target) / max(target, 0.2 * cap_norm))^2 )
    """
    def __init__(self, cap_norm=1.0):
        super().__init__()
        self.cap_norm = cap_norm

    def forward(self, pred, target):
        denom = torch.clamp(target, min=0.2 * self.cap_norm)
        return torch.mean(((pred - target) / denom) ** 2)


class MixedLoss(nn.Module):
    """混合 Loss = λ_mse * MSE + λ_acc2 * ACC2_Loss"""
    def __init__(self, lambda_mse=1.0, lambda_acc2=1.0, cap_norm=1.0):
        super().__init__()
        self.lambda_mse  = lambda_mse
        self.lambda_acc2 = lambda_acc2
        self.mse_loss    = nn.MSELoss()
        self.acc2_loss   = ACC2Loss(cap_norm=cap_norm)

    def forward(self, pred, target):
        return self.lambda_mse * self.mse_loss(pred, target) \
             + self.lambda_acc2 * self.acc2_loss(pred, target)


# =====================================================================
# 模型: 简单 LSTM Baseline (历史LSTM + 未来天气拼接)
# =====================================================================

class SimpleLSTM(nn.Module):
    """
    步骤:
      1. LSTM 处理历史序列 X_enc [B, 96, 13] -> 最后隐状态 [B, hidden]
      2. 展平未来天气 X_dec [B, 16, 12]      -> [B, 16*12=192]
      3. 拼接 [B, hidden+192] -> FC -> [B, 16]
    """
    def __init__(self, enc_input=13, dec_flat=192, hidden_size=128,
                 num_layers=2, out_steps=16, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=enc_input,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + dec_flat, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_steps),
        )

    def forward(self, x_enc, x_dec):
        out, _ = self.lstm(x_enc)                      # [B, 96, hidden]
        last = out[:, -1, :]                           # [B, hidden]
        dec_flat = x_dec.reshape(x_dec.size(0), -1)    # [B, 192]
        return self.fc(torch.cat([last, dec_flat], dim=1))  # [B, 16]


# =====================================================================
# 训练
# =====================================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print(f"设备: {device}")

    X_enc_train = np.load("X_enc_train.npy")
    X_dec_train = np.load("X_dec_train.npy")
    y_train     = np.load("y_train.npy")
    X_enc_val   = np.load("X_enc_val.npy")
    X_dec_val   = np.load("X_dec_val.npy")
    y_val       = np.load("y_val.npy")

    with open("norm_params.pkl", "rb") as f:
        norm_params = pickle.load(f)
    cap = norm_params["power"]["cap"]

    log_print(f"训练集: {X_enc_train.shape[0]} 样本  enc={X_enc_train.shape[1:]}, dec={X_dec_train.shape[1:]}")
    log_print(f"验证集: {X_enc_val.shape[0]} 样本")
    log_print(f"cap: {cap} MW")

    enc_input = X_enc_train.shape[2]  # 13
    dec_flat  = X_dec_train.shape[1] * X_dec_train.shape[2]  # 192
    out_steps = y_train.shape[1]      # 16

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

    model = SimpleLSTM(
        enc_input=enc_input,
        dec_flat=dec_flat,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        out_steps=out_steps,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log_print(f"\n模型参数量: {total_params:,}")
    log_print(f"结构: LSTM(enc={enc_input}, h={args.hidden_size}, L={args.num_layers}) + concat(dec_flat={dec_flat}) + FC -> {out_steps}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = MixedLoss(
        lambda_mse=args.lambda_mse,
        lambda_acc2=args.lambda_acc2,
        cap_norm=1.0,  # 功率已归一化到 [0,1]
    )

    os.makedirs("lstm_baseline_ckpt", exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0

    log_print(f"\n{'='*70}")
    log_print(f"开始训练 | Epochs={args.epochs} | BatchSize={args.batch_size} | LR={args.lr}")
    log_print(f"Loss: MixedLoss(λ_mse={args.lambda_mse}, λ_acc2={args.lambda_acc2})")
    log_print(f"{'='*70}\n")

    epoch_bar = tqdm(range(args.epochs), desc="训练进度", unit="epoch")
    for epoch in epoch_bar:
        # --- 训练 ---
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"  Ep{epoch+1:3d} 训练", leave=False, unit="batch")
        for xe, xd, yb in train_bar:
            xe, xd, yb = xe.to(device), xd.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xe, xd), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss /= len(train_loader)

        # --- 验证 ---
        model.eval()
        val_loss = 0.0
        all_preds, all_trues = [], []
        val_bar = tqdm(val_loader, desc=f"  Ep{epoch+1:3d} 验证", leave=False, unit="batch")
        with torch.no_grad():
            for xe, xd, yb in val_bar:
                xe, xd, yb = xe.to(device), xd.to(device), yb.to(device)
                pred = model(xe, xd)
                bvl = criterion(pred, yb).item()
                val_loss += bvl
                val_bar.set_postfix(loss=f"{bvl:.4f}")
                all_preds.append(pred.cpu().numpy())
                all_trues.append(yb.cpu().numpy())
        val_loss /= len(val_loader)

        all_preds = np.concatenate(all_preds)
        all_trues = np.concatenate(all_trues)

        acc1 = calc_acc1(all_trues, all_preds, cap)
        acc2 = calc_acc2(all_trues, all_preds, cap)
        rmse = calc_rmse(all_trues, all_preds, cap)
        mae  = calc_mae(all_trues,  all_preds, cap)
        lr   = optimizer.param_groups[0]["lr"]

        epoch_bar.set_postfix(val=f"{val_loss:.4f}", acc2=f"{acc2:.4f}", best=f"{best_val_loss:.4f}")

        log_tqdm_write(
            f"Epoch {epoch+1:3d}/{args.epochs} | LR: {lr:.6f} | "
            f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
            f"ACC1: {acc1:.4f} | ACC2: {acc2:.4f} | "
            f"RMSE: {rmse:.4f} ({rmse*cap:.2f} MW) | MAE: {mae:.4f} ({mae*cap:.2f} MW)"
        )

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
            }, "lstm_baseline_ckpt/best_model.pth")
            log_tqdm_write(f"  -> 保存最佳模型 (val_loss={val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log_tqdm_write(f"\nEarly stopping: {args.patience} epochs 无改善")
                break

        scheduler.step()

    log_print(f"\n训练完成! 最佳验证 Loss: {best_val_loss:.6f}")


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

    ckpt = torch.load("lstm_baseline_ckpt/best_model.pth", map_location=device, weights_only=False)
    saved = ckpt.get("args", {})

    model = SimpleLSTM(
        enc_input=X_enc_test.shape[2],
        dec_flat=X_dec_test.shape[1] * X_dec_test.shape[2],
        hidden_size=saved.get("hidden_size", args.hidden_size),
        num_layers=saved.get("num_layers", args.num_layers),
        out_steps=y_test.shape[1],
        dropout=saved.get("dropout", args.dropout),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    log_print(
        f"加载模型: epoch={ckpt['epoch']+1}, val_loss={ckpt['val_loss']:.6f}, "
        f"acc1={ckpt['acc1']:.4f}, acc2={ckpt['acc2']:.4f}"
    )

    loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_enc_test),
            torch.FloatTensor(X_dec_test),
            torch.FloatTensor(y_test),
        ),
        batch_size=args.batch_size, shuffle=False,
    )

    all_preds, all_trues = [], []
    with torch.no_grad():
        for xe, xd, yb in tqdm(loader, desc="测试推理"):
            all_preds.append(model(xe.to(device), xd.to(device)).cpu().numpy())
            all_trues.append(yb.numpy())

    all_preds = np.concatenate(all_preds)
    all_trues = np.concatenate(all_trues)

    print_metrics(all_trues, all_preds, cap, label="测试集评估结果 (SimpleLSTM Baseline)")


# =====================================================================
# 主入口
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="简单 LSTM Baseline - solar_station_1.csv")

    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "preprocess", "train", "evaluate"])
    parser.add_argument("--csv-path", type=str, default="solar_station_1.csv")

    # 训练参数
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch-size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--patience",   type=int,   default=15)

    # 模型结构
    parser.add_argument("--hidden-size", type=int,   default=128)
    parser.add_argument("--num-layers",  type=int,   default=2)
    parser.add_argument("--dropout",     type=float, default=0.2)

    # 混合 Loss 权重
    parser.add_argument("--lambda-mse",  type=float, default=1.0,
                        help="MSE Loss 权重")
    parser.add_argument("--lambda-acc2", type=float, default=1.0,
                        help="ACC2 Loss 权重 (与国标 ACC2 评估指标对齐)")

    args = parser.parse_args()

    # 记录本次运行信息
    log_print("=" * 80)
    log_print(f"Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"Args: {vars(args)}")
    log_print("=" * 80)

    if args.mode in ["all", "preprocess"]:
        log_print("\n[1/3] 数据预处理")
        preprocess(csv_path=args.csv_path)

    if args.mode in ["all", "train"]:
        log_print("\n[2/3] 模型训练")
        train(args)

    if args.mode in ["all", "evaluate"]:
        log_print("\n[3/3] 模型评估")
        evaluate(args)


if __name__ == "__main__":
    main()
