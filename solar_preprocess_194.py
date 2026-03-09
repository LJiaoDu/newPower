#!/usr/bin/env python3
"""
太阳能电站数据预处理脚本 - 适配 194.csv

数据特点:
  - 无气象数据 (tsi/dni/ghi/temp/atm/rh 均不存在)
  - 时间间隔约 5 分钟 (不规则) -> 重采样到 5 分钟
  - 功率单位: 瓦 (W), 最大约 1.81MW
  - 位置: lat=22.832779, lng=108.306903 (广西南宁)
  - 时间范围: 2024-06-01 ~ 2026-03-03

特征设计 (以天文计算太阳高度角代替气象数据):
  Encoder 输入 (8个特征):
    - 时间特征 (6): hour_sin, hour_cos, month_sin, month_cos, dayofyear_sin, dayofyear_cos
    - 太阳高度角 (1): solar_elevation (基于纬度/经度/时间计算)
    - 功率特征 (1): power_normed (按标称容量归一化)

  Decoder 输入 (7个特征):
    - 时间特征 (6): 同上
    - 太阳高度角 (1): solar_elevation (预测时刻的理论值)
    - 无功率

序列长度: 288步 (24h @5min) -> 48步 (4h @5min)
"""

import pandas as pd
import numpy as np
import pickle
import os
import math


# ============================================================
# 天文计算: 太阳高度角 (不依赖外部库)
# ============================================================

def solar_elevation_angle(dt_local, lat, lng, utc_offset=8.0):
    """
    计算太阳高度角 (度)

    Args:
        dt_local: datetime 对象, 本地时间 (北京时间 UTC+8)
        lat: 纬度 (度)
        lng: 经度 (度)
        utc_offset: UTC 偏移小时数 (北京时间=8)

    Returns:
        太阳高度角 (度), 夜间为 0
    """
    # 年积日 (day of year)
    doy = dt_local.timetuple().tm_yday

    # 太阳赤纬 (Spencer 公式, 精度约 ±0.01°)
    gamma = 2 * math.pi * (doy - 1) / 365
    decl = (0.006918
            - 0.399912 * math.cos(gamma)
            + 0.070257 * math.sin(gamma)
            - 0.006758 * math.cos(2 * gamma)
            + 0.000907 * math.sin(2 * gamma)
            - 0.002697 * math.cos(3 * gamma)
            + 0.001480 * math.sin(3 * gamma))  # 弧度

    # 时间方程 (equation of time, 单位: 分钟)
    eot = 229.18 * (0.000075
                    + 0.001868 * math.cos(gamma)
                    - 0.032077 * math.sin(gamma)
                    - 0.014615 * math.cos(2 * gamma)
                    - 0.04089 * math.sin(2 * gamma))

    # 当地太阳时 (小时)
    local_hour = dt_local.hour + dt_local.minute / 60.0 + dt_local.second / 3600.0
    solar_time = local_hour + (4 * lng + eot - utc_offset * 60) / 60.0

    # 时角 (弧度), 正午为 0
    hour_angle = math.radians((solar_time - 12.0) * 15.0)

    # 太阳高度角
    lat_r = math.radians(lat)
    sin_elev = (math.sin(lat_r) * math.sin(decl)
                + math.cos(lat_r) * math.cos(decl) * math.cos(hour_angle))
    elev_deg = math.degrees(math.asin(max(-1.0, min(1.0, sin_elev))))

    return max(0.0, elev_deg)   # 夜间 (高度角<0) 取 0


def compute_solar_elevation_series(datetimes, lat, lng):
    """批量计算太阳高度角序列"""
    return np.array([solar_elevation_angle(dt, lat, lng) for dt in datetimes],
                    dtype=np.float32)


# ============================================================
# 数据加载与预处理
# ============================================================

def load_and_resample(csv_path, resample_interval="15min"):
    """
    加载 194.csv 并重采样到规则时间间隔

    Returns:
        df: 重采样后的 DataFrame, 含 datetime(索引), power, lat, lng
    """
    df = pd.read_csv(csv_path)
    print(f"原始数据: {len(df)} 条记录")
    print(f"列名: {list(df.columns)}")

    # 处理 power 空值
    df["power"] = pd.to_numeric(df["power"], errors="coerce")
    n_null = df["power"].isna().sum()
    if n_null > 0:
        print(f"  power: {n_null} 个空值 -> NaN")

    # 解析 datetime
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()

    # 位置信息 (固定值)
    lat = float(df["locationLat"].iloc[0])
    lng = float(df["locationLng"].iloc[0])
    print(f"  位置: lat={lat}, lng={lng}")

    # 重采样: 取 15min 均值
    df_power = df["power"].resample(resample_interval).mean()

    # 线性插值填充空值 (不超过 2 个相邻缺失点)
    df_power = df_power.interpolate(method="linear", limit=2)
    df_power = df_power.bfill().ffill()

    # 过滤负功率 (传感器异常)
    df_power = df_power.clip(lower=0)

    print(f"\n重采样后 ({resample_interval}): {len(df_power)} 个时间点")
    print(f"  时间范围: {df_power.index[0]} ~ {df_power.index[-1]}")
    print(f"  power 范围: {df_power.min():.1f} ~ {df_power.max():.1f} W")
    print(f"  缺失值: {df_power.isna().sum()}")

    result = pd.DataFrame({"power": df_power})
    result.attrs["lat"] = lat
    result.attrs["lng"] = lng
    return result


def extract_features(df):
    """
    提取训练特征

    Returns:
        enc_features: [N, 8]  Encoder 输入 (时间6 + 高度角1 + 功率1)
        dec_features: [N, 7]  Decoder 输入 (时间6 + 高度角1)
        targets:      [N]     功率 (归一化)
        norm_params:  dict    归一化参数
    """
    lat = df.attrs["lat"]
    lng = df.attrs["lng"]
    datetimes = df.index.to_pydatetime()

    # --- 时间特征 (sin/cos 周期编码, 6维) ---
    hour = np.array([dt.hour + dt.minute / 60.0 for dt in datetimes])
    month = np.array([dt.month for dt in datetimes])
    doy = np.array([dt.timetuple().tm_yday for dt in datetimes])

    time_feats = np.column_stack([
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * (month - 1) / 12),
        np.cos(2 * np.pi * (month - 1) / 12),
        np.sin(2 * np.pi * doy / 365),
        np.cos(2 * np.pi * doy / 365),
    ]).astype(np.float32)

    # --- 太阳高度角 (1维, [0, ~75°] -> 归一化到 [0, 1]) ---
    print("\n计算太阳高度角序列 (共 %d 个点)..." % len(datetimes))
    solar_elev = compute_solar_elevation_series(datetimes, lat, lng)
    elev_max = 90.0   # 理论最大值
    solar_elev_normed = (solar_elev / elev_max).astype(np.float32)

    elev_actual_max = float(solar_elev.max())
    print(f"  高度角: max={elev_actual_max:.2f}°, "
          f"nonzero={np.sum(solar_elev > 0)} / {len(solar_elev)}")
    norm_params = {
        "solar_elev_max": elev_max,
        "solar_elev_actual_max": elev_actual_max,
    }

    # --- 功率特征 (按标称容量归一化) ---
    # 标称容量 = 功率最大值 (四舍五入到整 MW)
    raw_power = df["power"].values.astype(np.float32)
    cap_w = float(raw_power.max())
    cap_mw = max(round(cap_w / 1e6, 1), 0.1)  # MW, 至少 0.1MW
    cap_w_round = cap_mw * 1e6                  # 统一用 W 归一化
    power_normed = (raw_power / cap_w_round).astype(np.float32)
    power_normed = np.clip(power_normed, 0.0, 1.2)  # 允许轻微超过额定

    norm_params["cap_w"] = cap_w_round
    norm_params["cap_mw"] = cap_mw
    norm_params["power"] = {"cap": cap_mw}  # 兼容旧评估代码 (MW 单位)

    print(f"\n  功率: max_raw={cap_w:.0f} W ({cap_w/1e6:.3f} MW), "
          f"标称容量={cap_mw} MW")

    # --- 组装特征矩阵 ---
    # Decoder: 时间(6) + 高度角(1) = 7维
    dec_features = np.hstack([
        time_feats,
        solar_elev_normed.reshape(-1, 1),
    ]).astype(np.float32)

    # Encoder: 时间(6) + 高度角(1) + 功率(1) = 8维
    enc_features = np.hstack([
        dec_features,
        power_normed.reshape(-1, 1),
    ]).astype(np.float32)

    targets = power_normed

    enc_names = [
        "hour_sin", "hour_cos",
        "month_sin", "month_cos",
        "dayofyear_sin", "dayofyear_cos",
        "solar_elevation",
        "power",
    ]
    dec_names = [
        "hour_sin", "hour_cos",
        "month_sin", "month_cos",
        "dayofyear_sin", "dayofyear_cos",
        "solar_elevation",
    ]

    print(f"\n特征矩阵:")
    print(f"  Encoder 输入: {enc_features.shape}  (时间6 + 高度角1 + 功率1)")
    print(f"  Decoder 输入: {dec_features.shape}  (时间6 + 高度角1)")

    return enc_features, dec_features, targets, norm_params, enc_names, dec_names


# ============================================================
# 序列创建 + 数据集划分
# ============================================================

def create_sequences(enc_features, dec_features, targets,
                     in_steps=288, out_steps=48):
    """创建滑动窗口序列 (288步@5min=24h 输入, 48步@5min=4h 预测)"""
    X_enc_list, X_dec_list, y_list = [], [], []
    total_len = in_steps + out_steps

    for i in range(len(enc_features) - total_len + 1):
        X_enc_list.append(enc_features[i : i + in_steps])
        X_dec_list.append(dec_features[i + in_steps : i + total_len])
        y_list.append(targets[i + in_steps : i + total_len])

    X_enc = np.array(X_enc_list, dtype=np.float32)
    X_dec = np.array(X_dec_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    print(f"\n序列创建完成:")
    print(f"  Encoder 输入: {X_enc.shape}  ({in_steps}步={in_steps*5//60}h, {X_enc.shape[2]}特征)")
    print(f"  Decoder 输入: {X_dec.shape}  ({out_steps}步={out_steps*5//60}h, {X_dec.shape[2]}特征)")
    print(f"  预测目标:     {y.shape}")
    return X_enc, X_dec, y


def split_and_save(X_enc, X_dec, y, norm_params,
                   enc_names, dec_names,
                   train_ratio=0.7, val_ratio=0.15,
                   output_dir="."):
    """按时间顺序划分并保存"""
    n = len(X_enc)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    splits = {
        "train": (X_enc[:train_end],      X_dec[:train_end],      y[:train_end]),
        "val":   (X_enc[train_end:val_end], X_dec[train_end:val_end], y[train_end:val_end]),
        "test":  (X_enc[val_end:],         X_dec[val_end:],         y[val_end:]),
    }

    print(f"\n数据集划分 (train:{train_ratio}, val:{val_ratio}, test:{1-train_ratio-val_ratio:.2f}):")
    for name, (xe, xd, yt) in splits.items():
        np.save(os.path.join(output_dir, f"X_enc_{name}.npy"), xe)
        np.save(os.path.join(output_dir, f"X_dec_{name}.npy"), xd)
        np.save(os.path.join(output_dir, f"y_{name}.npy"), yt)
        print(f"  {name:5s}: {len(xe):6d} 样本")

    with open(os.path.join(output_dir, "norm_params.pkl"), "wb") as f:
        pickle.dump(norm_params, f)

    with open(os.path.join(output_dir, "feature_names.pkl"), "wb") as f:
        pickle.dump({"enc": enc_names, "dec": dec_names}, f)

    print(f"\n文件已保存到: {output_dir}")


# ============================================================
# 主入口
# ============================================================

def main(csv_path="194.csv", output_dir="."):
    print("=" * 60)
    print("太阳能电站数据预处理 - 194.csv (天文高度角特征)")
    print("=" * 60)

    # 1. 加载 & 重采样到 5min
    df = load_and_resample(csv_path, resample_interval="5min")

    # 2. 特征提取
    enc_features, dec_features, targets, norm_params, enc_names, dec_names = \
        extract_features(df)

    # 3. 创建序列 (24h 输入 -> 4h 预测, @5min)
    X_enc, X_dec, y = create_sequences(enc_features, dec_features, targets,
                                        in_steps=288, out_steps=48)

    # 4. 划分并保存
    split_and_save(X_enc, X_dec, y, norm_params, enc_names, dec_names,
                   output_dir=output_dir)

    print("\n预处理完成!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="194.csv 数据预处理")
    parser.add_argument("--csv-path", type=str, default="194.csv")
    parser.add_argument("--output-dir", type=str, default=".")
    args = parser.parse_args()
    main(csv_path=args.csv_path, output_dir=args.output_dir)
