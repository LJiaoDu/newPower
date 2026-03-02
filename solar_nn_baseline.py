#!/usr/bin/env python3
"""
太阳能电站功率预测 - 最近邻 (NN) Baseline (solar_station_1.csv)

算法:
  1. 模板库: 训练集所有 96 步窗口的时间特征 (6维: hour/month/doy 的 sin+cos)
  2. 查询:   对每个测试样本的 96 步时间特征, 在模板库中找 MSE 最小的窗口
  3. 预测:   取该最相似窗口在训练集中紧随的 16 步功率值作为预测
  4. 评估:   计算 ACC1 / ACC2 / RMSE / MAE

特点:
  - 无需训练, 纯检索
  - 只用时间特征匹配 (不用功率/气象), 避免 data leakage
  - 时间特征 (6维 per step) 捕捉日内周期 + 季节周期

数据: 共用预处理文件 (X_enc_*.npy, y_*.npy, norm_params.pkl)
      如无预处理文件, --mode all 时会自动预处理 solar_station_1.csv

使用方式:
  python solar_nn_baseline.py             # 完整流程 (预处理 + 评估)
  python solar_nn_baseline.py --mode run  # 仅检索+评估 (需先有 .npy 文件)
  python solar_nn_baseline.py --batch-size 512  # 调整查询批大小 (影响内存/速度)
"""

import os
import argparse
import pickle
import atexit
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm


# =========================================================
# 日志缓冲 (与其他 baseline 保持一致)
# =========================================================

LOG_BUFFER = []

def log_print(*args, **kwargs):
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    message = sep.join(str(a) for a in args) + end
    print(*args, **kwargs)
    LOG_BUFFER.append(message.rstrip("\n"))

def save_log_to_file():
    try:
        os.makedirs("solar_logs", exist_ok=True)
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"solar_logs/final_log_nn_baseline_{time_str}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(LOG_BUFFER) + "\n")
        print(f"\n最终日志已保存至: {filename}")
    except Exception as e:
        print(f"\n[WARN] 保存最终日志失败: {e}")

atexit.register(save_log_to_file)


# =====================================================================
# 数据预处理 (与其他 baseline 相同逻辑, 共用 norm_params.pkl)
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

    # 时间特征 (6维)
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

    # 气象特征 (归一化)
    norm_params = {}
    weather_normed = []
    for col in weather_cols:
        vmin, vmax = float(df[col].min()), float(df[col].max())
        norm_params[col] = {"min": vmin, "max": vmax}
        weather_normed.append(
            ((df[col].values - vmin) / (vmax - vmin + 1e-8)).astype(np.float32)
        )
    weather_feats = np.column_stack(weather_normed)

    # 功率
    cap = float(df["cap"].iloc[0])
    power_normed = (df["power"].values / cap).astype(np.float32)
    norm_params["power"] = {"cap": cap}

    dec_features = np.hstack([time_feats, weather_feats])                  # [N, 12]
    enc_features = np.hstack([dec_features, power_normed.reshape(-1, 1)])  # [N, 13]
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
        "train": (X_enc[:train_end],        X_dec[:train_end],        y_arr[:train_end]),
        "val":   (X_enc[train_end:val_end], X_dec[train_end:val_end], y_arr[train_end:val_end]),
        "test":  (X_enc[val_end:],          X_dec[val_end:],          y_arr[val_end:]),
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
    log_print(f"  RMSE:             {rmse:.4f} MW")
    log_print(f"  MAE:              {mae:.4f} MW")

    log_print(f"\n  按16个预测点 (每点15分钟):")
    for i in range(16):
        yt, yp = y_true[:, i:i+1], y_pred[:, i:i+1]
        total_min = (i + 1) * 15
        if total_min % 60 == 0:
            tlabel = f"+{total_min // 60}h"
        else:
            h, m = total_min // 60, total_min % 60
            tlabel = f"+{h}h{m:02d}m" if h else f"+{m}min"
        log_print(
            f"    点{i+1:2d} ({tlabel:>7s}): "
            f"ACC1={calc_acc1(yt, yp, cap):.4f}, "
            f"ACC2={calc_acc2(yt, yp, cap):.4f}, "
            f"RMSE={calc_rmse(yt, yp, cap):.2f} MW"
        )
    log_print(f"{'='*60}")
    return acc1, acc2, rmse, mae


# =====================================================================
# 最近邻检索
# =====================================================================

def retrieve_nn(X_enc_train, y_train, X_enc_test, batch_size=256):
    """
    对每个测试样本, 在训练集中找时间特征 MSE 最小的窗口, 返回对应的 y_train。

    参数:
      X_enc_train : [N_train, 96, 13]  训练集编码输入
      y_train     : [N_train, 16]      训练集目标
      X_enc_test  : [N_test,  96, 13]  测试集编码输入
      batch_size  : 每批查询样本数 (控制内存用量)

    匹配特征: X_enc 的前 6 列 (时间特征)
      col 0-1: sin/cos(hour/24)   — 日内周期
      col 2-3: sin/cos(month/12)  — 月份周期
      col 4-5: sin/cos(doy/365)   — 年内周期

    返回:
      y_pred : [N_test, 16]  预测值 (最近邻的后续16步功率)
      nn_mse : [N_test]      每个测试样本匹配到的最小 MSE
    """
    # 提取时间特征并展平: [N, 96*6]
    train_time = X_enc_train[:, :, :6].reshape(len(X_enc_train), -1)  # [N_train, 576]
    test_time  = X_enc_test[:, :, :6].reshape(len(X_enc_test),  -1)   # [N_test,  576]

    N_test  = len(test_time)
    N_train = len(train_time)
    feat_dim = train_time.shape[1]  # 96 * 6 = 576

    y_pred = np.zeros((N_test, y_train.shape[1]), dtype=np.float32)
    nn_mse = np.zeros(N_test, dtype=np.float32)

    for start in tqdm(range(0, N_test, batch_size), desc="NN 检索", unit="batch"):
        end  = min(start + batch_size, N_test)
        q    = test_time[start:end]               # [B, 576]

        # MSE[i, j] = mean((q[i] - train[j])^2)
        # 展开: ||q-t||^2 / D = (||q||^2 + ||t||^2 - 2*q@t^T) / D
        q_sq  = np.sum(q ** 2, axis=1, keepdims=True)           # [B, 1]
        t_sq  = np.sum(train_time ** 2, axis=1, keepdims=True)  # [N_train, 1]
        dot   = q @ train_time.T                                 # [B, N_train]
        mse_mat = (q_sq + t_sq.T - 2 * dot) / feat_dim          # [B, N_train]

        best_idx = np.argmin(mse_mat, axis=1)   # [B]
        y_pred[start:end] = y_train[best_idx]
        nn_mse[start:end] = mse_mat[np.arange(end - start), best_idx]

    return y_pred, nn_mse


# =====================================================================
# 主流程
# =====================================================================

def run(args):
    log_print("=" * 60)
    log_print("加载数据")
    log_print("=" * 60)

    X_enc_train = np.load("X_enc_train.npy")
    y_train     = np.load("y_train.npy")
    X_enc_test  = np.load("X_enc_test.npy")
    y_test      = np.load("y_test.npy")

    with open("norm_params.pkl", "rb") as f:
        norm_params = pickle.load(f)
    cap = norm_params["power"]["cap"]

    log_print(f"训练集模板: {len(X_enc_train)} 个窗口")
    log_print(f"测试集查询: {len(X_enc_test)} 个窗口")
    log_print(f"cap: {cap} MW")
    log_print(f"匹配特征: 时间特征 (前6列) — 96步 × 6维 = 576维")
    log_print(f"查询批大小: {args.batch_size}")

    log_print(f"\n{'='*60}")
    log_print("开始 NN 检索...")
    log_print(f"{'='*60}")

    y_pred, nn_mse = retrieve_nn(
        X_enc_train, y_train, X_enc_test, batch_size=args.batch_size
    )

    log_print(f"\n检索完成!")
    log_print(f"  匹配 MSE — 均值: {nn_mse.mean():.6f}, "
              f"中位数: {np.median(nn_mse):.6f}, "
              f"最大: {nn_mse.max():.6f}")

    print_metrics(y_test, y_pred, cap,
                  label="测试集评估结果 (NN Baseline, 时间特征匹配)")


# =====================================================================
# 主入口
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="最近邻 Baseline (时间特征 MSE 匹配) - solar_station_1.csv"
    )
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "preprocess", "run"],
                        help="all=预处理+检索+评估, preprocess=仅预处理, run=仅检索+评估")
    parser.add_argument("--csv-path",   type=str, default="solar_station_1.csv")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="查询批大小, 越大越快但占更多内存 (默认 256)")

    args = parser.parse_args()

    log_print("=" * 80)
    log_print(f"Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"Args: {vars(args)}")
    log_print("=" * 80)

    if args.mode in ["all", "preprocess"]:
        log_print("\n[1/2] 数据预处理")
        preprocess(csv_path=args.csv_path)

    if args.mode in ["all", "run"]:
        log_print("\n[2/2] NN 检索 + 评估")
        run(args)


if __name__ == "__main__":
    main()
