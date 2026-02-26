#!/usr/bin/env python3
"""
TFT 数据预处理脚本
适配 solar_station_1.csv (15分钟粒度)

相比标准 Encoder-Decoder 预处理新增:
  - 静态特征 X_static: 装机容量 (归一化后为 1.0, 支持多站扩展)

三类输入 (TFT 标准划分):
  静态协变量  (1个): cap_norm
  历史观测序列 (13个特征, enc_len=96 步 = 24h):
      时间编码 (6): hour_sin/cos, month_sin/cos, dayofyear_sin/cos
      气象特征 (6): tsi, dni, ghi, temp, atm, rh
      功率特征 (1): power (按标称容量归一化)
  未来已知序列 (12个特征, dec_len=16 步 = 4h):
      时间编码 (6) + 气象预报 (6)  [训练时用真实值模拟预报]

输出文件 (与 solar_preprocess.py 兼容):
  X_enc_{train/val/test}.npy   [N, 96, 13]
  X_dec_{train/val/test}.npy   [N, 16, 12]
  X_static_{train/val/test}.npy [N, 1]
  y_{train/val/test}.npy        [N, 16]
  norm_params_tft.pkl
"""

import pandas as pd
import numpy as np
import pickle
import os


# ============================================================
#  数据加载与清洗
# ============================================================

def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"原始数据: {len(df)} 条记录")
    print(f"列名: {list(df.columns)}")

    df["time"] = pd.to_datetime(df["time"])

    weather_cols = ["tsi", "dni", "ghi", "temp", "atm", "rh"]

    # 将缺测值 -99 替换为 NaN
    for col in weather_cols:
        n_bad = (df[col] == -99).sum()
        if n_bad > 0:
            df.loc[df[col] == -99, col] = np.nan
            print(f"  {col}: 替换 {n_bad} 个 -99 值")

    # 修复 rh 传感器故障 (>100%)
    n_rh = (df["rh"] > 100).sum()
    if n_rh > 0:
        df.loc[df["rh"] > 100, "rh"] = np.nan
        print(f"  rh: 替换 {n_rh} 个 >100% 异常值")

    df[weather_cols] = df[weather_cols].interpolate(method="linear")
    df[weather_cols] = df[weather_cols].bfill().ffill()

    print(f"清洗后缺失值总数: {df.isnull().sum().sum()}")
    return df


# ============================================================
#  特征提取
# ============================================================

def extract_features(df: pd.DataFrame):
    """
    返回:
      enc_features: [N, 13]  Encoder 输入特征
      dec_features: [N, 12]  Decoder 输入特征
      static_feat:  scalar   归一化后的静态装机容量 (=1.0)
      targets:      [N]      归一化功率
      norm_params:  dict
    """
    # --- 时间周期编码 (6维) ---
    hour       = df["time"].dt.hour + df["time"].dt.minute / 60
    month      = df["time"].dt.month
    dayofyear  = df["time"].dt.dayofyear

    time_feats = np.column_stack([
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * (month - 1) / 12),
        np.cos(2 * np.pi * (month - 1) / 12),
        np.sin(2 * np.pi * dayofyear / 365),
        np.cos(2 * np.pi * dayofyear / 365),
    ]).astype(np.float32)

    # --- 气象特征 Min-Max 归一化 (6维) ---
    weather_cols = ["tsi", "dni", "ghi", "temp", "atm", "rh"]
    norm_params  = {}
    weather_normed = []

    for col in weather_cols:
        vmin = float(df[col].min())
        vmax = float(df[col].max())
        norm_params[col] = {"min": vmin, "max": vmax}
        normed = (df[col].values - vmin) / (vmax - vmin + 1e-8)
        weather_normed.append(normed.astype(np.float32))

    weather_feats = np.column_stack(weather_normed)   # [N, 6]

    # --- 功率归一化 (1维) ---
    cap = float(df["cap"].iloc[0])
    power_normed = (df["power"].values / cap).astype(np.float32)
    norm_params["power"] = {"cap": cap}

    # --- 静态特征 ---
    # cap 归一化为 1.0 (单站). 多站时可用 cap/max_cap
    norm_params["cap_max"] = cap
    static_feat = np.float32(cap / cap)               # = 1.0

    # --- 组装 ---
    dec_features = np.hstack([time_feats, weather_feats])           # [N, 12]
    enc_features = np.hstack([dec_features,
                               power_normed.reshape(-1, 1)])         # [N, 13]
    targets = power_normed                                           # [N]

    print(f"\n特征维度:")
    print(f"  Encoder  输入: {enc_features.shape}  (时间6 + 气象6 + 功率1)")
    print(f"  Decoder  输入: {dec_features.shape}  (时间6 + 气象6)")
    print(f"  静态特征:      scalar={static_feat:.4f}  (装机容量归一化)")
    print(f"  标称容量: {cap} MW")
    print(f"  rh 范围: [{norm_params['rh']['min']:.1f}, {norm_params['rh']['max']:.1f}]%")

    return enc_features, dec_features, static_feat, targets, norm_params


# ============================================================
#  滑动窗口序列构建
# ============================================================

def create_sequences(enc_features: np.ndarray,
                     dec_features: np.ndarray,
                     static_feat:  float,
                     targets:      np.ndarray,
                     in_steps:  int = 96,
                     out_steps: int = 16):
    """
    15分钟粒度:
      in_steps=96  -> 24h 历史 (Encoder)
      out_steps=16 -> 4h  预测 (Decoder)

    返回:
      X_enc:    [N, 96, 13]
      X_dec:    [N, 16, 12]
      X_static: [N, 1]       每个样本的静态特征 (重复广播)
      y:        [N, 16]
    """
    X_enc_list, X_dec_list, y_list = [], [], []
    total = in_steps + out_steps

    for i in range(len(enc_features) - total + 1):
        X_enc_list.append(enc_features[i: i + in_steps])
        X_dec_list.append(dec_features[i + in_steps: i + total])
        y_list.append(targets[i + in_steps: i + total])

    X_enc    = np.array(X_enc_list, dtype=np.float32)
    X_dec    = np.array(X_dec_list, dtype=np.float32)
    y        = np.array(y_list,     dtype=np.float32)
    # 静态特征对所有样本一致, shape [N, 1]
    X_static = np.full((len(X_enc), 1), static_feat, dtype=np.float32)

    print(f"\n序列样本数: {len(X_enc)}")
    print(f"  X_enc:    {X_enc.shape}  (N, {in_steps}步={in_steps*15//60}h, 13特征)")
    print(f"  X_dec:    {X_dec.shape}  (N, {out_steps}步={out_steps*15//60}h, 12特征)")
    print(f"  X_static: {X_static.shape}  (N, 1)")
    print(f"  y:        {y.shape}")

    return X_enc, X_dec, X_static, y


# ============================================================
#  数据集划分与保存
# ============================================================

def split_and_save(X_enc, X_dec, X_static, y, norm_params,
                   train_ratio=0.7, val_ratio=0.15,
                   output_dir="."):
    n         = len(X_enc)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))

    splits = {
        "train": (X_enc[:train_end],      X_dec[:train_end],
                  X_static[:train_end],   y[:train_end]),
        "val":   (X_enc[train_end:val_end], X_dec[train_end:val_end],
                  X_static[train_end:val_end], y[train_end:val_end]),
        "test":  (X_enc[val_end:],         X_dec[val_end:],
                  X_static[val_end:],      y[val_end:]),
    }

    print(f"\n数据集划分 (按时间顺序):")
    for name, (xe, xd, xs, yt) in splits.items():
        np.save(os.path.join(output_dir, f"X_enc_{name}.npy"),    xe)
        np.save(os.path.join(output_dir, f"X_dec_{name}.npy"),    xd)
        np.save(os.path.join(output_dir, f"X_static_{name}.npy"), xs)
        np.save(os.path.join(output_dir, f"y_{name}.npy"),        yt)
        print(f"  {name:5s}: {len(xe):6d} 样本")

    with open(os.path.join(output_dir, "norm_params_tft.pkl"), "wb") as f:
        pickle.dump(norm_params, f)

    print(f"\n文件已保存至: {output_dir}")
    print("生成文件: X_enc/X_dec/X_static/y _{{train,val,test}}.npy  +  norm_params_tft.pkl")


# ============================================================
#  主入口
# ============================================================

def main(csv_path: str = "solar_station_1.csv",
         in_steps:  int = 96,
         out_steps: int = 16,
         output_dir: str = "."):

    print("=" * 60)
    print("TFT 数据预处理")
    print("=" * 60)

    df = load_and_clean(csv_path)

    enc_feat, dec_feat, static_feat, targets, norm_params = extract_features(df)

    X_enc, X_dec, X_static, y = create_sequences(
        enc_feat, dec_feat, static_feat, targets,
        in_steps=in_steps, out_steps=out_steps
    )

    split_and_save(X_enc, X_dec, X_static, y, norm_params,
                   output_dir=output_dir)

    print("\nTFT 预处理完成!")
    return norm_params


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", default="solar_station_1.csv")
    parser.add_argument("--in-steps",  type=int, default=96)
    parser.add_argument("--out-steps", type=int, default=16)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    main(csv_path=args.csv_path,
         in_steps=args.in_steps,
         out_steps=args.out_steps,
         output_dir=args.output_dir)
