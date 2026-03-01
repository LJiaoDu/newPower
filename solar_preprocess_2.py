#!/usr/bin/env python3
"""
太阳能电站数据预处理脚本 (training_data.csv 版本)
适配 training_data.csv (5分钟粒度, 仅含时间+功率特征)

与 solar_preprocess_1.py 的主要区别:
1. 数据源: training_data.csv (无气象特征 tsi/dni/ghi/temp/atm/rh)
2. 粒度: 5分钟 (原为15分钟)
3. 序列长度: enc_steps=288 (24h), dec_steps=48 (4h)
4. 特征维度: Encoder=7 (时间6+功率1), Decoder=6 (时间6)
5. 功率归一化: 按全局最大值归一化 (无 cap 列)

特征说明:
  Encoder 输入 (7个特征):
    - 时间特征 (6): hour_sin, hour_cos, month_sin, month_cos, dayofyear_sin, dayofyear_cos
    - 功率特征 (1): generationPower (按最大值归一化)

  Decoder 输入 (6个特征):
    - 时间特征 (6): 同上
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

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # 1. 将负功率替换为 0 (夜间传感器噪声)
    n_neg = (df["generationPower"] < 0).sum()
    if n_neg > 0:
        df.loc[df["generationPower"] < 0, "generationPower"] = 0.0
        print(f"  generationPower: 替换 {n_neg} 个负值为 0")

    # 2. 线性插值填充缺失值
    n_null = df["generationPower"].isnull().sum()
    if n_null > 0:
        df["generationPower"] = df["generationPower"].interpolate(method="linear")
        df["generationPower"] = df["generationPower"].bfill().ffill()
        print(f"  generationPower: 插值填充 {n_null} 个缺失值")

    print(f"清洗后缺失值: {df.isnull().sum().sum()}")
    print(f"时间范围: {df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}")
    return df


def extract_features(df):
    """
    提取训练特征, 生成两套特征矩阵:
      - enc_features: 7维 (时间6 + 功率1) 用于 Encoder
      - dec_features: 6维 (时间6)          用于 Decoder
    """

    # --- 时间特征 (sin/cos 周期编码, 6个) ---
    hour = df["datetime"].dt.hour + df["datetime"].dt.minute / 60
    month = df["datetime"].dt.month
    dayofyear = df["datetime"].dt.dayofyear

    time_feats = np.column_stack([
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * (month - 1) / 12),
        np.cos(2 * np.pi * (month - 1) / 12),
        np.sin(2 * np.pi * dayofyear / 365),
        np.cos(2 * np.pi * dayofyear / 365),
    ]).astype(np.float32)

    # --- 功率特征 (按全局最大值归一化, 1个) ---
    max_power = float(df["generationPower"].max())
    power_normed = (df["generationPower"].values / (max_power + 1e-8)).astype(np.float32)
    norm_params = {"power": {"max_power": max_power}}

    print(f"\n功率归一化:")
    print(f"  max_power = {max_power:.2f} W")
    print(f"  归一化后范围: [{power_normed.min():.4f}, {power_normed.max():.4f}]")

    # --- 组装 ---
    # Decoder 输入: 时间(6) = 6维 (不含功率)
    dec_features = time_feats.copy()

    # Encoder 输入: 时间(6) + 功率(1) = 7维
    enc_features = np.hstack([time_feats, power_normed.reshape(-1, 1)]).astype(np.float32)

    targets = power_normed

    enc_feature_names = [
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "dayofyear_sin", "dayofyear_cos",
        "power",
    ]
    dec_feature_names = [
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "dayofyear_sin", "dayofyear_cos",
    ]

    print(f"\n特征矩阵:")
    print(f"  Encoder 输入: {enc_features.shape} (时间6 + 功率1)")
    print(f"  Decoder 输入: {dec_features.shape} (时间6)")

    return enc_features, dec_features, targets, norm_params, enc_feature_names, dec_feature_names


def create_sequences(enc_features, dec_features, targets, in_steps=288, out_steps=48):
    """
    创建滑动窗口序列

    5分钟粒度:
      in_steps=288  -> 24小时历史 (Encoder)
      out_steps=48  -> 4小时预测  (Decoder)

    返回:
      X_enc: [N, 288, 7]  Encoder 输入 (历史功率+时间)
      X_dec: [N, 48,  6]  Decoder 输入 (未来时间, 无功率)
      y:     [N, 48]      预测目标 (未来功率)
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
    print(f"  Encoder 输入: {X_enc.shape}  (样本, {in_steps}步={in_steps*5/60:.0f}h, 7特征)")
    print(f"  Decoder 输入: {X_dec.shape}  (样本, {out_steps}步={out_steps*5/60:.0f}h, 6特征)")
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


def main(csv_path="training_data.csv"):

    print("=" * 60)
    print("太阳能电站数据预处理 (training_data.csv 版本)")
    print("=" * 60)

    # 1. 加载与清洗
    df = load_and_clean(csv_path)

    # 2. 特征提取 (Encoder 7维 + Decoder 6维)
    enc_features, dec_features, targets, norm_params, enc_names, dec_names = extract_features(df)

    # 3. 创建序列 (24h 输入 -> 4h 预测, 5分钟粒度)
    X_enc, X_dec, y = create_sequences(enc_features, dec_features, targets,
                                        in_steps=288, out_steps=48)

    # 4. 划分并保存
    split_and_save(X_enc, X_dec, y, norm_params, enc_names, dec_names)

    print("\n预处理完成!")


if __name__ == "__main__":
    main()
