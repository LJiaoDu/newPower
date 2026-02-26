#!/usr/bin/env python3
"""
太阳能电站数据预处理脚本 (v3 - 含 TSI 衍生特征)
适配 solar_station_1.csv (15分钟粒度, 含气象特征)

在 v1 基础上新增三个 TSI 衍生特征:
  - tsi_diff1 : TSI 一阶差分 (变化速率)
  - tsi_diff2 : TSI 二阶差分 (加速度)
  - tsi_std_4 : TSI 滚动标准差 (窗口=4, 即1小时波动)

特征说明:
  Encoder 输入 (16个特征):
    - 时间特征 (6): hour_sin, hour_cos, month_sin, month_cos, dayofyear_sin, dayofyear_cos
    - 气象特征 (6): tsi, dni, ghi, temp, atm, rh
    - TSI衍生特征 (3): tsi_diff1, tsi_diff2, tsi_std_4
    - 功率特征 (1): power (按标称容量归一化)

  Decoder 输入 (15个特征):
    - 时间特征 (6): 同上
    - 气象特征 (6): 同上 (未来气象, 训练时用真实值模拟预报)
    - TSI衍生特征 (3): 同上 (未来已知的 TSI 统计量)
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
      - enc_features: 16维 (时间6 + 气象6 + TSI衍生3 + 功率1) 用于 Encoder
      - dec_features: 15维 (时间6 + 气象6 + TSI衍生3)          用于 Decoder
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

    # --- TSI 衍生特征 (3个) ---
    # 使用原始 tsi 值计算差分和滚动标准差, 然后独立归一化

    tsi_raw = df["tsi"].values.astype(np.float64)

    # 1. tsi_diff1: 一阶差分 (TSI 变化速率), 第一个时刻补 0
    tsi_diff1 = np.diff(tsi_raw, prepend=tsi_raw[0])

    # 2. tsi_diff2: 二阶差分 (TSI 变化加速度), 第一个时刻补 0
    tsi_diff2 = np.diff(tsi_diff1, prepend=tsi_diff1[0])

    # 3. tsi_std_4: 滚动标准差, 窗口=4 (约1小时), 前3行 NaN → 0
    tsi_std_4 = df["tsi"].rolling(4).std().values
    tsi_std_4 = np.nan_to_num(tsi_std_4, nan=0.0)

    # 归一化辅助函数
    def maxabs_scale(arr, name):
        """MaxAbs 归一化: 缩放到 [-1, 1], 保留 0 点对称性"""
        m = max(float(np.abs(arr).max()), 1e-8)
        norm_params[name] = {"scale": m}
        print(f"  {name}: range=[{arr.min():.3f}, {arr.max():.3f}], scale={m:.3f}")
        return (arr / m).astype(np.float32)

    def minmax_scale_feat(arr, name):
        """Min-Max 归一化: 缩放到 [0, 1]"""
        vmin = float(arr.min())
        vmax = float(arr.max())
        norm_params[name] = {"min": vmin, "max": vmax}
        print(f"  {name}: range=[{vmin:.3f}, {vmax:.3f}]")
        return ((arr - vmin) / (vmax - vmin + 1e-8)).astype(np.float32)

    print("\n  TSI 衍生特征统计:")
    tsi_diff1_scaled = maxabs_scale(tsi_diff1, "tsi_diff1")
    tsi_diff2_scaled = maxabs_scale(tsi_diff2, "tsi_diff2")
    tsi_std4_scaled  = minmax_scale_feat(tsi_std_4, "tsi_std_4")

    # 组合为 [N, 3] 矩阵
    tsi_extra_feats = np.column_stack([
        tsi_diff1_scaled,
        tsi_diff2_scaled,
        tsi_std4_scaled,
    ])

    # --- 功率特征 (按标称容量归一化, 1个) ---
    cap = float(df["cap"].iloc[0])
    power_normed = (df["power"].values / cap).astype(np.float32)
    norm_params["power"] = {"cap": cap}

    # --- 组装 ---
    # Decoder 输入: 时间(6) + 气象(6) + TSI衍生(3) = 15维 (不含功率)
    dec_features = np.hstack([
        time_feats,
        weather_feats,
        tsi_extra_feats,
    ]).astype(np.float32)

    # Encoder 输入: 时间(6) + 气象(6) + TSI衍生(3) + 功率(1) = 16维
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
    print(f"  rh 归一化范围验证: min={norm_params['rh']['min']:.1f}, max={norm_params['rh']['max']:.1f}")

    return enc_features, dec_features, targets, norm_params, enc_feature_names, dec_feature_names


def create_sequences(enc_features, dec_features, targets, in_steps=96, out_steps=16):
    """
    创建滑动窗口序列

    15分钟粒度:
      in_steps=96  -> 24小时历史 (Encoder)
      out_steps=16 -> 4小时预测  (Decoder)

    返回:
      X_enc: [N, 96, 16]  Encoder 输入 (历史功率+气象+时间+TSI衍生)
      X_dec: [N, 16, 15]  Decoder 输入 (未来气象+时间+TSI衍生, 无功率)
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

    enc_feat = X_enc.shape[2]
    dec_feat = X_dec.shape[2]
    print(f"\n序列创建完成:")
    print(f"  Encoder 输入: {X_enc.shape}  (样本, {in_steps}步={in_steps*15/60:.0f}h, {enc_feat}特征)")
    print(f"  Decoder 输入: {X_dec.shape}  (样本, {out_steps}步={out_steps*15/60:.0f}h, {dec_feat}特征)")
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


def main(csv_path="/media/zlg/Data1/Longjiao/TF208/solar_station_1.csv"):

    print("=" * 60)
    print("太阳能电站数据预处理 (v3 - 含 TSI 衍生特征)")
    print("=" * 60)

    # 1. 加载与清洗 (含 rh 异常值修复)
    df = load_and_clean(csv_path)

    # 2. 特征提取 (Encoder 16维 + Decoder 15维)
    enc_features, dec_features, targets, norm_params, enc_names, dec_names = extract_features(df)

    # 3. 创建序列 (24h 输入 -> 4h 预测)
    X_enc, X_dec, y = create_sequences(enc_features, dec_features, targets,
                                        in_steps=96, out_steps=16)

    # 4. 划分并保存
    split_and_save(X_enc, X_dec, y, norm_params, enc_names, dec_names)

    print("\n预处理完成!")


if __name__ == "__main__":
    main()
