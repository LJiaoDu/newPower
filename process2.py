#!/usr/bin/env python3
"""
太阳能电站数据预处理脚本 v3 (新增 tsi_diff1 / tsi_diff2 / tsi_std_4 特征)

基于 solar_preprocess.py 改进, 在原有 13/12 维特征基础上新增 3 个 TSI 衍生特征:
  tsi_diff1  = df["tsi"].diff()                     一阶差分: 辐射变化方向
  tsi_diff2  = df["tsi_diff1"].diff()               二阶差分: 辐射变化加速度
  tsi_std_4  = df["tsi"].rolling(4).std()           1小时滚动标准差: 短期波动性

特征说明 (最终维度):
  Encoder 输入 (16个特征):
    时间特征 (6): hour_sin, hour_cos, month_sin, month_cos, dayofyear_sin, dayofyear_cos
    气象特征 (6): tsi, dni, ghi, temp, atm, rh
    TSI衍生  (3): tsi_diff1, tsi_diff2, tsi_std_4
    功率特征 (1): power (按标称容量归一化)

  Decoder 输入 (15个特征):
    时间特征 (6): 同上
    气象特征 (6): 同上 (未来气象预报)
    TSI衍生  (3): tsi_diff1, tsi_diff2, tsi_std_4
    无功率        (功率是预测目标)

使用方式:
  from process2 import main as preprocess_main
  preprocess_main(csv_path="solar_station_1.csv")
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

    # 2. 修复 rh 异常值: >100% 视为传感器故障
    n_rh_bad = (df["rh"] > 100).sum()
    if n_rh_bad > 0:
        df.loc[df["rh"] > 100, "rh"] = np.nan
        print(f"  rh: 替换 {n_rh_bad} 个 >100% 传感器故障值")

    # 3. 线性插值填充
    df[weather_cols] = df[weather_cols].interpolate(method="linear")
    df[weather_cols] = df[weather_cols].bfill().ffill()

    print(f"清洗后缺失值: {df.isnull().sum().sum()}")
    return df


def compute_tsi_features(df):
    """
    计算三个 TSI 衍生特征 (作用于原始、已清洗的 tsi 值)

    tsi_diff1: 一阶差分, 反映辐射变化方向和速率
    tsi_diff2: 二阶差分, 反映辐射变化的加速度 (云遮挡起止时变化最大)
    tsi_std_4: 4步(1小时)滚动标准差, 反映短期辐射波动性
               高值 → 间歇性云遮; 低值 → 稳定晴天或夜间

    NaN 处理:
      tsi_diff1 第 1 个时刻: 填 0
      tsi_diff2 前 2 个时刻: 填 0
      tsi_std_4 前 3 个时刻: 填 0
    """
    df = df.copy()

    df["tsi_diff1"] = df["tsi"].diff().fillna(0.0)
    df["tsi_diff2"] = df["tsi_diff1"].diff().fillna(0.0)
    df["tsi_std_4"] = df["tsi"].rolling(4).std().fillna(0.0)

    # 统计信息
    print(f"\n  tsi_diff1: min={df['tsi_diff1'].min():.2f}, "
          f"max={df['tsi_diff1'].max():.2f}, "
          f"std={df['tsi_diff1'].std():.2f}")
    print(f"  tsi_diff2: min={df['tsi_diff2'].min():.2f}, "
          f"max={df['tsi_diff2'].max():.2f}, "
          f"std={df['tsi_diff2'].std():.2f}")
    print(f"  tsi_std_4: min={df['tsi_std_4'].min():.2f}, "
          f"max={df['tsi_std_4'].max():.2f}, "
          f"std={df['tsi_std_4'].std():.2f}")

    return df


def normalize_tsi_features(df, norm_params, tsi_range):
    """
    对三个 TSI 衍生特征做归一化
    差分特征 (tsi_diff1/2): 除以 tsi 量程 (max-min), 映射到约 [-1, 1]
    标准差特征 (tsi_std_4): 除以 tsi 量程, 映射到约 [0, 0.5]

    tsi_range: tsi 的 (max - min), 用作统一的缩放因子
    """
    scale = tsi_range + 1e-8

    diff1_scaled = (df["tsi_diff1"].values / scale).astype(np.float32)
    diff2_scaled = (df["tsi_diff2"].values / scale).astype(np.float32)
    std4_scaled  = (df["tsi_std_4"].values  / scale).astype(np.float32)

    norm_params["tsi_diff1"] = {"scale": float(scale), "type": "diff_scale"}
    norm_params["tsi_diff2"] = {"scale": float(scale), "type": "diff_scale"}
    norm_params["tsi_std_4"] = {"scale": float(scale), "type": "diff_scale"}

    return diff1_scaled, diff2_scaled, std4_scaled


def extract_features(df):
    """
    提取训练特征, 生成两套特征矩阵:
      enc_features: 16维 (时间6 + 气象6 + TSI衍生3 + 功率1)
      dec_features: 15维 (时间6 + 气象6 + TSI衍生3)
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
    ]).astype(np.float32)

    # --- 气象特征 (Min-Max 归一化, 6个) ---
    weather_cols = ["tsi", "dni", "ghi", "temp", "atm", "rh"]
    norm_params = {}
    weather_normed = []

    for col in weather_cols:
        vmin = float(df[col].min())
        vmax = float(df[col].max())
        norm_params[col] = {"min": vmin, "max": vmax}
        normed = (df[col].values - vmin) / (vmax - vmin + 1e-8)
        weather_normed.append(normed.astype(np.float32))

    weather_feats = np.column_stack(weather_normed).astype(np.float32)

    # --- TSI 衍生特征 (3个) ---
    tsi_range = norm_params["tsi"]["max"] - norm_params["tsi"]["min"]
    diff1_scaled, diff2_scaled, std4_scaled = normalize_tsi_features(
        df, norm_params, tsi_range
    )
    tsi_extra_feats = np.column_stack([
        diff1_scaled, diff2_scaled, std4_scaled
    ]).astype(np.float32)

    # --- 功率特征 (1个) ---
    cap = float(df["cap"].iloc[0])
    power_normed = (df["power"].values / cap).astype(np.float32)
    norm_params["power"] = {"cap": cap}

    # --- 组装特征矩阵 ---
    # Decoder: 时间(6) + 气象(6) + TSI衍生(3) = 15维
    dec_features = np.hstack([
        time_feats,
        weather_feats,
        tsi_extra_feats,
    ]).astype(np.float32)

    # Encoder: Decoder特征(15) + 功率(1) = 16维
    enc_features = np.hstack([
        dec_features,
        power_normed.reshape(-1, 1),
    ]).astype(np.float32)

    targets = power_normed

    enc_feature_names = [
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "dayofyear_sin", "dayofyear_cos",
        "tsi", "dni", "ghi", "temp", "atm", "rh",
        "tsi_diff1", "tsi_diff2", "tsi_std_4",
        "power",
    ]
    dec_feature_names = [
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "dayofyear_sin", "dayofyear_cos",
        "tsi", "dni", "ghi", "temp", "atm", "rh",
        "tsi_diff1", "tsi_diff2", "tsi_std_4",
    ]

    print(f"\n特征矩阵:")
    print(f"  Encoder 输入: {enc_features.shape} (时间6 + 气象6 + TSI衍生3 + 功率1)")
    print(f"  Decoder 输入: {dec_features.shape} (时间6 + 气象6 + TSI衍生3)")
    print(f"  标称容量: {cap} MW")
    print(f"  rh 归一化范围验证: min={norm_params['rh']['min']:.1f}, "
          f"max={norm_params['rh']['max']:.1f}")

    return enc_features, dec_features, targets, norm_params, enc_feature_names, dec_feature_names


def create_sequences(enc_features, dec_features, targets, in_steps=96, out_steps=16):
    """
    创建滑动窗口序列 (15分钟粒度)

    in_steps=96  -> 24小时历史 (Encoder)
    out_steps=16 -> 4小时预测  (Decoder)

    返回:
      X_enc: [N, 96, 16]  Encoder 输入
      X_dec: [N, 16, 15]  Decoder 输入
      y:     [N, 16]      预测目标
    """
    X_enc_list = []
    X_dec_list = []
    y_list = []

    total_len = in_steps + out_steps

    for i in range(len(enc_features) - total_len + 1):
        X_enc_list.append(enc_features[i: i + in_steps])
        X_dec_list.append(dec_features[i + in_steps: i + total_len])
        y_list.append(targets[i + in_steps: i + total_len])

    X_enc = np.array(X_enc_list, dtype=np.float32)
    X_dec = np.array(X_dec_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    enc_dim = enc_features.shape[1]
    dec_dim = dec_features.shape[1]
    print(f"\n序列创建完成:")
    print(f"  Encoder 输入: {X_enc.shape}  (样本, {in_steps}步={in_steps*15//60}h, {enc_dim}特征)")
    print(f"  Decoder 输入: {X_dec.shape}  (样本, {out_steps}步={out_steps*15//60}h, {dec_dim}特征)")
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
        "train": (X_enc[:train_end],         X_dec[:train_end],         y[:train_end]),
        "val":   (X_enc[train_end:val_end],   X_dec[train_end:val_end],   y[train_end:val_end]),
        "test":  (X_enc[val_end:],            X_dec[val_end:],            y[val_end:]),
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
        pickle.dump({"enc": enc_feature_names, "dec": dec_feature_names}, f)

    print(f"\n文件已保存到: {output_dir}")


def main(csv_path="solar_station_1.csv"):

    print("=" * 60)
    print("太阳能电站数据预处理 v3 (含 tsi_diff1/diff2/std_4 特征)")
    print("=" * 60)

    # 1. 加载与清洗
    df = load_and_clean(csv_path)

    # 2. 计算 TSI 衍生特征 (在原始 tsi 上操作, 清洗之后)
    print("\nTSI 衍生特征统计:")
    df = compute_tsi_features(df)

    # 3. 特征提取
    enc_features, dec_features, targets, norm_params, enc_names, dec_names = extract_features(df)

    # 4. 创建序列 (24h 输入 -> 4h 预测)
    X_enc, X_dec, y = create_sequences(enc_features, dec_features, targets,
                                        in_steps=96, out_steps=16)

    # 5. 划分并保存
    split_and_save(X_enc, X_dec, y, norm_params, enc_names, dec_names)

    print("\n预处理完成!")


if __name__ == "__main__":
    main()
