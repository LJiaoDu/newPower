#!/usr/bin/env python3
"""
太阳能电站多站点数据预处理脚本 v1 (基于 process2.py)

支持同时处理多个站点的 CSV 文件，为每个样本附加 site_id 标识。
各站点气象特征独立归一化（使用各自的 min/max 和 cap），保证特征量纲一致。
训练集在各站点内部按时间顺序划分后再混合，不跨站点泄露未来信息。

输出文件 (保存在 output_dir 下):
  X_enc_{train,val,test}.npy    [N, 96, 16]   Encoder 输入
  X_dec_{train,val,test}.npy    [N, 16, 15]   Decoder 输入
  y_{train,val,test}.npy        [N, 16]       预测目标 (归一化)
  site_id_{train,val,test}.npy  [N]           int64, 站点索引 0~(num_sites-1)
  norm_params_multi.pkl          dict {site_id: norm_params}
  site_configs.pkl               dict {site_id: {name, cap, csv_path, n_samples}}

使用方式:
  python process3.py --csv-paths site1.csv site2.csv site3.csv site4.csv site5.csv
  python process3.py --csv-paths site*.csv --output-dir ./data_multi
"""

import argparse
import os
import pickle

import numpy as np

from process2 import (
    compute_tsi_features,
    create_sequences,
    extract_features,
    load_and_clean,
)


# ============================================================
# 单站点处理
# ============================================================

def process_one_site(csv_path: str, site_id: int,
                     in_steps: int = 96, out_steps: int = 16):
    """
    处理单个站点的完整预处理流程 (复用 process2 的所有函数)

    Returns:
        X_enc      : [N, in_steps, 16]
        X_dec      : [N, out_steps, 15]
        y          : [N, out_steps]
        site_ids   : [N]  全部填充为 site_id
        norm_params: dict  该站点的归一化参数
        cap        : float 标称容量 (MW)
    """
    print(f"\n{'─' * 60}")
    print(f"站点 {site_id}: {os.path.basename(csv_path)}")
    print(f"{'─' * 60}")

    df = load_and_clean(csv_path)
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

def main(csv_paths, output_dir=".", train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    多站点数据预处理主流程

    Args:
        csv_paths  : list[str]  各站点 CSV 文件路径列表（顺序即为站点编号）
        output_dir : str        输出目录
        train_ratio: float      训练集占比（按各站点时间顺序划分）
        val_ratio  : float      验证集占比
        seed       : int        训练集随机打乱的随机种子
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(f"多站点数据预处理 | 站点数: {len(csv_paths)}")
    print("=" * 70)

    # 各站点分别预处理并按时间划分
    split_data = {"train": [], "val": [], "test": []}
    norm_params_multi = {}
    site_configs = {}

    for site_id, csv_path in enumerate(csv_paths):
        X_enc, X_dec, y, site_ids, norm_params, cap = process_one_site(
            csv_path, site_id
        )
        norm_params_multi[site_id] = norm_params
        site_configs[site_id] = {
            "name":      os.path.splitext(os.path.basename(csv_path))[0],
            "cap":       cap,
            "csv_path":  csv_path,
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
            for sid in range(len(csv_paths))
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
    parser = argparse.ArgumentParser(description="多站点太阳能数据预处理")
    parser.add_argument(
        "--csv-paths", nargs="+", required=True,
        help="各站点 CSV 文件路径（空格分隔），路径顺序即为站点编号"
    )
    parser.add_argument(
        "--output-dir", type=str, default=".",
        help="输出目录 (默认: 当前目录)"
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio",   type=float, default=0.15)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    main(args.csv_paths, args.output_dir, args.train_ratio, args.val_ratio, args.seed)
