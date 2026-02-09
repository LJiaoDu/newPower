#!/usr/bin/env python3
"""
太阳能电站数据预处理脚本 (改进版)
适配 solar_station_1.csv (15分钟粒度, 含气象特征)

改进点:
1. 修复 rh 异常值 (>100% 的传感器故障数据)
2. 生成 Encoder 输入 (历史: 功率+气象+时间) 和 Decoder 输入 (未来: 气象+时间, 无功率)
3. Robust 归一化 (对 rh 使用 clip 后的统计量)

特征说明:
  Encoder 输入 (13个特征):
    - 时间特征 (6): hour_sin, hour_cos, month_sin, month_cos, dayofyear_sin, dayofyear_cos
    - 气象特征 (6): tsi, dni, ghi, temp, atm, rh
    - 功率特征 (1): power (按标称容量归一化)

  Decoder 输入 (12个特征):
    - 时间特征 (6): 同上
    - 气象特征 (6): 同上 (未来气象, 训练时用真实值模拟预报)
    - 无功率 (这是要预测的目标)
"""

import pandas as pd
import numpy as np
import pickle
import os


def load_and_clean(csv_path):
    """加载CSV数据并清洗"""
    df = pd.read_csv(csv_path)
    print(f"原始数据: {len(df)} 条记录")
    print(f"列: {list(df.columns)}")

    df["time"] = pd.to_datetime(df["time"])

    weather_cols = ["tsi", "dni", "ghi", "temp", "atm", "rh"]

    # 1. 将 -99 替换为 NaN
    for col in weather_cols:
        n_bad = (df[col] == -99).sum()
        if n_bad > 0:
            df.loc[df[col] == -99, col] = np.nan
            print(f"  {col}: 替换 {n_bad} 个 -99 值")

    # 2. 修复 rh 异常值: >100% 全部视为传感器故障
    n_rh_bad = (df["rh"] > 100).sum()
    if n_rh_bad > 0:
        df.loc[df["rh"] > 100, "rh"] = np.nan
        print(f"  rh: 替换 {n_rh_bad} 个 >100% 传感器故障值")

    # 3. 线性插值填充
    df[weather_cols] = df[weather_cols].interpolate(method="linear")
    df[weather_cols] = df[weather_cols].bfill().ffill()

    print(f"清洗后缺失值: {df.isnull().sum().sum()}")
    return df


def extract_features(df):
    """
    提取训练特征, 生成两套特征矩阵:
      - enc_features: 13维 (时间6 + 气象6 + 功率1) 用于 Encoder
      - dec_features: 12维 (时间6 + 气象6)          用于 Decoder
    """

    # --- 时间特征 (sin/cos 周期编码, 6个) ---
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

    # --- 气象特征 (Min-Max 归一化, 6个) ---
    weather_cols = ["tsi", "dni", "ghi", "temp", "atm", "rh"]
    norm_params = {}
    weather_normed = []

    for col in weather_cols:
        vmin = float(df[col].min())
        vmax = float(df[col].max())
        norm_params[col] = {"min": vmin, "max": vmax}
        normed = (df[col].values - vmin) / (vmax - vmin + 1e-8)
        weather_normed.append(normed)

    weather_feats = np.column_stack(weather_normed)

    # --- 功率特征 (按标称容量归一化, 1个) ---
    cap = float(df["cap"].iloc[0])
    power_normed = (df["power"].values / cap).astype(np.float32)
    norm_params["power"] = {"cap": cap}

    # --- 组装 ---
    # Decoder 输入: 时间(6) + 气象(6) = 12维 (不含功率)
    dec_features = np.hstack([time_feats, weather_feats]).astype(np.float32)

    # Encoder 输入: 时间(6) + 气象(6) + 功率(1) = 13维
    enc_features = np.hstack([dec_features, power_normed.reshape(-1, 1)]).astype(np.float32)

    targets = power_normed

    enc_feature_names = [
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "dayofyear_sin", "dayofyear_cos",
        "tsi", "dni", "ghi", "temp", "atm", "rh",
        "power",
    ]
    dec_feature_names = [
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "dayofyear_sin", "dayofyear_cos",
        "tsi", "dni", "ghi", "temp", "atm", "rh",
    ]

    print(f"\n特征矩阵:")
    print(f"  Encoder 输入: {enc_features.shape} (时间6 + 气象6 + 功率1)")
    print(f"  Decoder 输入: {dec_features.shape} (时间6 + 气象6)")
    print(f"  标称容量: {cap} MW")

    # 打印 rh 归一化范围 (验证异常值已修复)
    print(f"\n  rh 归一化范围验证: min={norm_params['rh']['min']:.1f}, max={norm_params['rh']['max']:.1f}")

    return enc_features, dec_features, targets, norm_params, enc_feature_names, dec_feature_names


def create_sequences(enc_features, dec_features, targets, in_steps=96, out_steps=16):
    """
    创建滑动窗口序列

    15分钟粒度:
      in_steps=96  -> 24小时历史 (Encoder)
      out_steps=16 -> 4小时预测  (Decoder)

    返回:
      X_enc: [N, 96, 13]  Encoder 输入 (历史功率+气象+时间)
      X_dec: [N, 16, 12]  Decoder 输入 (未来气象+时间, 无功率)
      y:     [N, 16]      预测目标 (未来功率)
    """
    X_enc_list = []
    X_dec_list = []
    y_list = []

    total_len = in_steps + out_steps

    for i in range(len(enc_features) - total_len + 1):
        X_enc_list.append(enc_features[i : i + in_steps])
        X_dec_list.append(dec_features[i + in_steps : i + total_len])
        y_list.append(targets[i + in_steps : i + total_len])

    X_enc = np.array(X_enc_list, dtype=np.float32)
    X_dec = np.array(X_dec_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    print(f"\n序列创建完成:")
    print(f"  Encoder 输入: {X_enc.shape}  (样本, {in_steps}步={in_steps*15/60:.0f}h, 13特征)")
    print(f"  Decoder 输入: {X_dec.shape}  (样本, {out_steps}步={out_steps*15/60:.0f}h, 12特征)")
    print(f"  预测目标:     {y.shape}  (样本, {out_steps}步)")

    return X_enc, X_dec, y


def split_and_save(X_enc, X_dec, y, norm_params,
                   enc_feature_names, dec_feature_names,
                   train_ratio=0.7, val_ratio=0.15,
                   output_dir="."):
    """按时间顺序划分数据集并保存"""
    n = len(X_enc)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    splits = {
        "train": (X_enc[:train_end], X_dec[:train_end], y[:train_end]),
        "val":   (X_enc[train_end:val_end], X_dec[train_end:val_end], y[train_end:val_end]),
        "test":  (X_enc[val_end:], X_dec[val_end:], y[val_end:]),
    }

    print(f"\n数据集划分:")
    for name, (xe, xd, yt) in splits.items():
        np.save(os.path.join(output_dir, f"X_enc_{name}.npy"), xe)
        np.save(os.path.join(output_dir, f"X_dec_{name}.npy"), xd)
        np.save(os.path.join(output_dir, f"y_{name}.npy"), yt)
        print(f"  {name}: {len(xe)} 样本")

    with open(os.path.join(output_dir, "norm_params.pkl"), "wb") as f:
        pickle.dump(norm_params, f)

    with open(os.path.join(output_dir, "feature_names.pkl"), "wb") as f:
        pickle.dump({
            "enc": enc_feature_names,
            "dec": dec_feature_names,
        }, f)

    print(f"\n文件已保存到: {output_dir}")


def main():
    csv_path = "solar_station_1.csv"

    print("=" * 60)
    print("太阳能电站数据预处理 (改进版)")
    print("=" * 60)

    # 1. 加载与清洗 (含 rh 异常值修复)
    df = load_and_clean(csv_path)

    # 2. 特征提取 (Encoder + Decoder 双输入)
    enc_features, dec_features, targets, norm_params, enc_names, dec_names = extract_features(df)

    # 3. 创建序列 (24h 输入 -> 4h 预测)
    X_enc, X_dec, y = create_sequences(enc_features, dec_features, targets,
                                        in_steps=96, out_steps=16)

    # 4. 划分并保存
    split_and_save(X_enc, X_dec, y, norm_params, enc_names, dec_names)

    print("\n预处理完成!")


if __name__ == "__main__":
    main()
