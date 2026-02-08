#!/usr/bin/env python3
"""
太阳能电站数据预处理脚本
适配 solar_station_1.csv (15分钟粒度, 含气象特征)

特征说明:
- 时间特征 (6个): hour_sin, hour_cos, month_sin, month_cos, dayofyear_sin, dayofyear_cos
- 气象特征 (6个): tsi, dni, ghi, temp, atm, rh (归一化后)
- 功率特征 (1个): power (按标称容量归一化)
- 总计: 13个特征
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

    # 解析时间
    df["time"] = pd.to_datetime(df["time"])

    # 1. 将 -99 替换为 NaN, 然后线性插值
    weather_cols = ["tsi", "dni", "ghi", "temp", "atm", "rh"]
    for col in weather_cols:
        n_bad = (df[col] == -99).sum()
        if n_bad > 0:
            df.loc[df[col] == -99, col] = np.nan
            print(f"  {col}: 替换 {n_bad} 个 -99 值")

    # 2. 湿度异常值处理: >100% 视为异常
    n_rh_bad = (df["rh"] > 100).sum()
    if n_rh_bad > 0:
        df.loc[df["rh"] > 100, "rh"] = np.nan
        print(f"  rh: 替换 {n_rh_bad} 个 >100% 异常值")

    # 3. 线性插值填充
    df[weather_cols] = df[weather_cols].interpolate(method="linear")
    df[weather_cols] = df[weather_cols].bfill().ffill()

    print(f"清洗后缺失值: {df.isnull().sum().sum()}")
    return df


def extract_features(df):
    """提取训练特征"""

    # --- 时间特征 (sin/cos周期编码, 6个) ---
    hour = df["time"].dt.hour + df["time"].dt.minute / 60
    month = df["time"].dt.month
    dayofyear = df["time"].dt.dayofyear

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month_sin = np.sin(2 * np.pi * (month - 1) / 12)
    month_cos = np.cos(2 * np.pi * (month - 1) / 12)
    dayofyear_sin = np.sin(2 * np.pi * dayofyear / 365)
    dayofyear_cos = np.cos(2 * np.pi * dayofyear / 365)

    # --- 气象特征 (Min-Max归一化, 6个) ---
    weather_cols = ["tsi", "dni", "ghi", "temp", "atm", "rh"]
    norm_params = {}
    weather_normed = {}

    for col in weather_cols:
        vmin = df[col].min()
        vmax = df[col].max()
        norm_params[col] = {"min": float(vmin), "max": float(vmax)}
        weather_normed[col] = (df[col] - vmin) / (vmax - vmin + 1e-8)

    # --- 功率特征 (按标称容量归一化, 1个) ---
    cap = df["cap"].iloc[0]
    power_normed = df["power"] / cap
    norm_params["power"] = {"cap": float(cap)}

    # --- 组装特征矩阵 [N, 13] ---
    features = np.column_stack([
        hour_sin.values,
        hour_cos.values,
        month_sin.values,
        month_cos.values,
        dayofyear_sin.values,
        dayofyear_cos.values,
        weather_normed["tsi"].values,
        weather_normed["dni"].values,
        weather_normed["ghi"].values,
        weather_normed["temp"].values,
        weather_normed["atm"].values,
        weather_normed["rh"].values,
        power_normed.values,
    ]).astype(np.float32)

    targets = power_normed.values.astype(np.float32)

    feature_names = [
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "dayofyear_sin", "dayofyear_cos",
        "tsi", "dni", "ghi", "temp", "atm", "rh",
        "power"
    ]

    print(f"\n特征矩阵: {features.shape}")
    print(f"  时间特征: 6个")
    print(f"  气象特征: 6个")
    print(f"  功率特征: 1个")
    print(f"  标称容量: {cap} MW")

    return features, targets, norm_params, feature_names


def create_sequences(features, targets, in_steps=96, out_steps=16):
    """
    创建滑动窗口序列

    15分钟粒度:
      in_steps=96  → 24小时历史
      out_steps=16 → 4小时预测
    """
    X_list = []
    y_list = []

    total_len = in_steps + out_steps

    for i in range(len(features) - total_len + 1):
        X_list.append(features[i : i + in_steps])
        y_list.append(targets[i + in_steps : i + total_len])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    print(f"\n序列创建完成:")
    print(f"  输入: {X.shape}  (样本数, {in_steps}步={in_steps*15/60:.0f}h, 特征数)")
    print(f"  输出: {y.shape}  (样本数, {out_steps}步={out_steps*15/60:.0f}h)")

    return X, y


def split_and_save(X, y, norm_params, feature_names,
                   train_ratio=0.7, val_ratio=0.15,
                   output_dir="."):
    """划分数据集并保存"""
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"\n数据集划分:")
    print(f"  训练集: {len(X_train)}")
    print(f"  验证集: {len(X_val)}")
    print(f"  测试集: {len(X_test)}")

    # 保存
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)

    with open(os.path.join(output_dir, "norm_params.pkl"), "wb") as f:
        pickle.dump(norm_params, f)

    with open(os.path.join(output_dir, "feature_names.pkl"), "wb") as f:
        pickle.dump(feature_names, f)

    print(f"\n文件已保存到: {output_dir}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
    csv_path = "solar_station_1.csv"

    print("=" * 60)
    print("太阳能电站数据预处理")
    print("=" * 60)

    # 1. 加载与清洗
    df = load_and_clean(csv_path)

    # 2. 特征提取
    features, targets, norm_params, feature_names = extract_features(df)

    # 3. 创建序列 (24h输入 → 4h预测)
    X, y = create_sequences(features, targets, in_steps=96, out_steps=16)

    # 4. 划分并保存
    split_and_save(X, y, norm_params, feature_names)

    print("\n预处理完成!")


if __name__ == "__main__":
    main()
