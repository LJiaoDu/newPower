#!/usr/bin/env python3
"""
太阳能电站多站点数据预处理脚本 v2 (基于 process2.py)

支持同时处理多个站点的数据文件，格式兼容：
  - CSV  文件: 列名需满足 process2.py 要求 (time/tsi/dni/ghi/temp/atm/rh/power/cap)
  - XLSX 文件: 列名自动映射，标称容量从文件名 "...XXX MW..." 自动提取

各站点气象特征独立归一化（使用各自的 min/max 和 cap），保证特征量纲一致。
训练集在各站点内部按时间顺序划分后再混合，不跨站点泄露未来信息。

输出文件 (保存在 output_dir 下):
  X_enc_{train,val,test}.npy    [N, 96, 16]   Encoder 输入
  X_dec_{train,val,test}.npy    [N, 16, 15]   Decoder 输入
  y_{train,val,test}.npy        [N, 16]       预测目标 (归一化)
  site_id_{train,val,test}.npy  [N]           int64, 站点索引 0~(num_sites-1)
  norm_params_multi.pkl          dict {site_id: norm_params}
  site_configs.pkl               dict {site_id: {name, cap, path, n_samples}}

使用方式:
  # xlsx 文件 (推荐，列名自动映射)
  python process3.py --paths "site1.xlsx" "site2.xlsx" "site3.xlsx"

  # csv 文件
  python process3.py --paths site1.csv site2.csv site3.csv

  # 混合也可以
  python process3.py --paths site1.xlsx site2.csv --output-dir ./data_multi
"""

import argparse
import os
import pickle
import re

import numpy as np

from process2 import (
    compute_tsi_features,
    create_sequences,
    extract_features,
    load_and_clean,
)


# ============================================================
# XLSX → 标准 DataFrame 转换
# ============================================================

# xlsx 列名 → process2.py 期望的列名
_XLSX_COL_MAP = {
    "Time(year-month-day h:m:s)":          "time",
    "Total solar irradiance (W/m2)":        "tsi",
    "Direct normal irradiance (W/m2)":      "dni",
    "Global horizontal irradiance (W/m2)":  "ghi",
    "Air temperature  (°C) ":              "temp",   # 原始有多余空格
    "Air temperature (°C)":                "temp",   # 无多余空格的变体
    "Air temperature  (°C)":               "temp",   # 一个多余空格的变体
    "Atmosphere (hpa)":                    "atm",
    "Relative humidity (%)":               "rh",
    "Power (MW)":                          "power",
}


def _extract_cap_from_filename(path: str) -> float:
    """从文件名中提取标称容量，例如 '...50MW...' → 50.0"""
    name = os.path.basename(path)
    m = re.search(r"(\d+(?:\.\d+)?)\s*MW", name, re.IGNORECASE)
    if m:
        return float(m.group(1))
    raise ValueError(
        f"无法从文件名提取标称容量(MW)，请确认文件名包含 'XXXMW' 格式: {name}"
    )


def load_xlsx_as_standard_df(xlsx_path: str):
    """
    读取 xlsx 文件，将列名映射到 process2.py 期望的格式，
    并添加 cap 列（从文件名提取）。

    Returns:
        pandas.DataFrame  包含 time/tsi/dni/ghi/temp/atm/rh/power/cap 列
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("请先安装 pandas 和 openpyxl: pip install pandas openpyxl")

    df = pd.read_excel(xlsx_path, sheet_name=0)
    print(f"  读取 xlsx: {len(df)} 行, 原始列: {list(df.columns)}")

    # 对列名做 strip 处理再匹配，应对多余空格
    col_rename = {}
    for col in df.columns:
        stripped = col.strip()
        # 先尝试精确匹配
        if stripped in _XLSX_COL_MAP:
            col_rename[col] = _XLSX_COL_MAP[stripped]
        else:
            # 再尝试原始列名
            if col in _XLSX_COL_MAP:
                col_rename[col] = _XLSX_COL_MAP[col]

    df = df.rename(columns=col_rename)

    # 检查必要列是否全部存在
    required = {"time", "tsi", "dni", "ghi", "temp", "atm", "rh", "power"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"xlsx 列名映射后仍缺少: {missing}\n"
            f"当前列: {list(df.columns)}\n"
            f"请检查 _XLSX_COL_MAP 是否覆盖了所有列名"
        )

    # 从文件名提取容量并写入 cap 列
    cap = _extract_cap_from_filename(xlsx_path)
    df["cap"] = cap
    print(f"  标称容量: {cap} MW (从文件名提取)")

    return df


# ============================================================
# 单站点处理
# ============================================================

def load_site_df(path: str):
    """
    根据文件扩展名自动选择加载方式：
      .xlsx / .xls → load_xlsx_as_standard_df (列名自动映射 + cap 从文件名提取)
      .csv         → load_and_clean (process2 原始流程，需文件内含 cap 列)
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df_raw = load_xlsx_as_standard_df(path)
        # load_and_clean 期望 CSV 路径，此处直接传已加载的 df 的清洗逻辑
        # 复现 load_and_clean 核心逻辑（不重复读文件）
        import pandas as pd
        weather_cols = ["tsi", "dni", "ghi", "temp", "atm", "rh"]
        df_raw["time"] = pd.to_datetime(df_raw["time"])
        # 强制将天气列转为数值类型（xlsx 可能混有字符串如 "-"、"N/A" 等）
        for col in weather_cols:
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
        for col in weather_cols:
            n_bad = (df_raw[col] == -99).sum()
            if n_bad > 0:
                df_raw.loc[df_raw[col] == -99, col] = float("nan")
                print(f"  {col}: 替换 {n_bad} 个 -99 值")
        n_rh_bad = (df_raw["rh"] > 100).sum()
        if n_rh_bad > 0:
            df_raw.loc[df_raw["rh"] > 100, "rh"] = float("nan")
            print(f"  rh: 替换 {n_rh_bad} 个 >100% 传感器故障值")
        df_raw[weather_cols] = df_raw[weather_cols].interpolate(method="linear")
        df_raw[weather_cols] = df_raw[weather_cols].bfill().ffill()
        print(f"  清洗后缺失值: {df_raw.isnull().sum().sum()}")
        return df_raw
    else:
        return load_and_clean(path)


def process_one_site(path: str, site_id: int,
                     in_steps: int = 96, out_steps: int = 16):
    """
    处理单个站点的完整预处理流程，兼容 xlsx 和 csv。

    Returns:
        X_enc      : [N, in_steps, 16]
        X_dec      : [N, out_steps, 15]
        y          : [N, out_steps]
        site_ids   : [N]  全部填充为 site_id
        norm_params: dict  该站点的归一化参数
        cap        : float 标称容量 (MW)
    """
    print(f"\n{'─' * 60}")
    print(f"站点 {site_id}: {os.path.basename(path)}")
    print(f"{'─' * 60}")

    df = load_site_df(path)
    df = compute_tsi_features(df)

    enc_features, dec_features, targets, norm_params, _, _ = extract_features(df)
    X_enc, X_dec, y = create_sequences(
        enc_features, dec_features, targets, in_steps, out_steps
    )

    site_ids = np.full(len(X_enc), site_id, dtype=np.int64)
    cap = norm_params["power"]["cap"]

    return X_enc, X_dec, y, site_ids, norm_params, cap


# ============================================================
# 多站点合并主流程
# ============================================================

def main(paths, output_dir=".", train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    多站点数据预处理主流程

    Args:
        paths      : list[str]  各站点文件路径列表（xlsx 或 csv，顺序即为站点编号）
        output_dir : str        输出目录
        train_ratio: float      训练集占比（按各站点时间顺序划分）
        val_ratio  : float      验证集占比
        seed       : int        训练集随机打乱的随机种子
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(f"多站点数据预处理 | 站点数: {len(paths)}")
    print("=" * 70)

    # 各站点分别预处理并按时间划分
    split_data = {"train": [], "val": [], "test": []}
    norm_params_multi = {}
    site_configs = {}

    for site_id, path in enumerate(paths):
        X_enc, X_dec, y, site_ids, norm_params, cap = process_one_site(
            path, site_id
        )
        norm_params_multi[site_id] = norm_params
        site_configs[site_id] = {
            "name":      os.path.splitext(os.path.basename(path))[0],
            "cap":       cap,
            "path":      path,
            "n_samples": len(X_enc),
        }

        # 按时间顺序划分（保证时序完整性，不跨站点泄露）
        n         = len(X_enc)
        train_end = int(n * train_ratio)
        val_end   = int(n * (train_ratio + val_ratio))

        split_data["train"].append((
            X_enc[:train_end], X_dec[:train_end],
            y[:train_end],     site_ids[:train_end],
        ))
        split_data["val"].append((
            X_enc[train_end:val_end], X_dec[train_end:val_end],
            y[train_end:val_end],     site_ids[train_end:val_end],
        ))
        split_data["test"].append((
            X_enc[val_end:], X_dec[val_end:],
            y[val_end:],     site_ids[val_end:],
        ))

    # 合并、打乱（仅训练集），保存
    print(f"\n{'=' * 70}")
    print("数据集合并与保存:")
    rng = np.random.default_rng(seed)

    for split_name in ["train", "val", "test"]:
        X_enc_all = np.concatenate([s[0] for s in split_data[split_name]], axis=0)
        X_dec_all = np.concatenate([s[1] for s in split_data[split_name]], axis=0)
        y_all     = np.concatenate([s[2] for s in split_data[split_name]], axis=0)
        sid_all   = np.concatenate([s[3] for s in split_data[split_name]], axis=0)

        # 训练集打乱：各站点样本充分混合，避免模型按顺序拟合单一站点
        if split_name == "train":
            idx       = rng.permutation(len(X_enc_all))
            X_enc_all = X_enc_all[idx]
            X_dec_all = X_dec_all[idx]
            y_all     = y_all[idx]
            sid_all   = sid_all[idx]

        np.save(os.path.join(output_dir, f"X_enc_{split_name}.npy"),   X_enc_all)
        np.save(os.path.join(output_dir, f"X_dec_{split_name}.npy"),   X_dec_all)
        np.save(os.path.join(output_dir, f"y_{split_name}.npy"),        y_all)
        np.save(os.path.join(output_dir, f"site_id_{split_name}.npy"), sid_all)

        # 统计各站点样本量
        site_counts = {
            sid: int((sid_all == sid).sum())
            for sid in range(len(paths))
        }
        count_str = ", ".join(f"站点{k}:{v}" for k, v in site_counts.items())
        print(f"  {split_name:5s}: {len(X_enc_all):7d} 样本  ({count_str})")

    # 保存归一化参数和站点配置
    with open(os.path.join(output_dir, "norm_params_multi.pkl"), "wb") as f:
        pickle.dump(norm_params_multi, f)
    with open(os.path.join(output_dir, "site_configs.pkl"), "wb") as f:
        pickle.dump(site_configs, f)

    print(f"\n站点配置汇总:")
    for sid, cfg in site_configs.items():
        print(f"  站点{sid} ({cfg['name']}): "
              f"标称容量={cfg['cap']} MW, 总样本={cfg['n_samples']}")

    print(f"\n文件已保存到: {os.path.abspath(output_dir)}")
    print("多站点预处理完成!")

    return norm_params_multi, site_configs


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多站点太阳能数据预处理 (支持 xlsx/csv)")
    parser.add_argument(
        "--paths", nargs="+", required=True,
        help="各站点文件路径（xlsx 或 csv，空格分隔），顺序即为站点编号"
    )
    parser.add_argument(
        "--output-dir", type=str, default=".",
        help="输出目录 (默认: 当前目录)"
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio",   type=float, default=0.15)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    main(args.paths, args.output_dir, args.train_ratio, args.val_ratio, args.seed)
