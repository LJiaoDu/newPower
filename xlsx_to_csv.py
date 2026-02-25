#!/usr/bin/env python3
"""
xlsx_to_csv.py  —  将太阳能电站 xlsx 数据集转为 process2.py 兼容的标准 CSV

输出 CSV 列: time, tsi, dni, ghi, temp, atm, rh, power, cap
所有数值列均已完成清洗（字符串→NaN、-99→NaN、rh>100→NaN、线性插值补全）

用法:
    python xlsx_to_csv.py "Solar station site 1 (Nominal capacity-50MW).xlsx"
    python xlsx_to_csv.py *.xlsx
    python xlsx_to_csv.py site1.xlsx site2.xlsx --output-dir ./csv_data
"""

import argparse
import os
import re
import sys

import pandas as pd

# xlsx 列名 → 标准列名（与 process3.py 保持一致）
_XLSX_COL_MAP = {
    "Time(year-month-day h:m:s)":          "time",
    "Total solar irradiance (W/m2)":        "tsi",
    "Direct normal irradiance (W/m2)":      "dni",
    "Global horizontal irradiance (W/m2)":  "ghi",
    "Air temperature  (°C) ":              "temp",
    "Air temperature (°C)":                "temp",
    "Air temperature  (°C)":               "temp",
    "Atmosphere (hpa)":                    "atm",
    "Relative humidity (%)":               "rh",
    "Power (MW)":                          "power",
}

_NUMERIC_COLS = ["tsi", "dni", "ghi", "temp", "atm", "rh", "power"]


def _extract_cap(path: str) -> float:
    name = os.path.basename(path)
    m = re.search(r"(\d+(?:\.\d+)?)\s*MW", name, re.IGNORECASE)
    if m:
        return float(m.group(1))
    raise ValueError(
        f"无法从文件名提取容量(MW)，确认文件名含 'XXXMW': {name}"
    )


def convert(xlsx_path: str, output_dir: str) -> str:
    print(f"\n处理: {os.path.basename(xlsx_path)}")

    df = pd.read_excel(xlsx_path, sheet_name=0)
    print(f"  读取: {len(df)} 行, 原始列: {list(df.columns)}")

    # 重命名列
    col_rename = {}
    for col in df.columns:
        stripped = col.strip()
        if stripped in _XLSX_COL_MAP:
            col_rename[col] = _XLSX_COL_MAP[stripped]
        elif col in _XLSX_COL_MAP:
            col_rename[col] = _XLSX_COL_MAP[col]
    df = df.rename(columns=col_rename)

    required = {"time", "tsi", "dni", "ghi", "temp", "atm", "rh", "power"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"列名映射后仍缺少: {missing}\n当前列: {list(df.columns)}"
        )

    # 添加 cap 列
    cap = _extract_cap(xlsx_path)
    df["cap"] = cap
    print(f"  标称容量: {cap} MW")

    # 时间列转换
    df["time"] = pd.to_datetime(df["time"])

    # 所有数值列强制转为 float（字符串/空值 → NaN）
    for col in _NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 替换异常传感器值
    for col in _NUMERIC_COLS:
        n = int((df[col] == -99).sum())
        if n:
            df[col] = df[col].replace(-99, float("nan"))
            print(f"  {col}: 替换 {n} 个 -99")

    n_rh = int((df["rh"] > 100).sum())
    if n_rh:
        df["rh"] = df["rh"].where(df["rh"] <= 100, other=float("nan"))
        print(f"  rh: 替换 {n_rh} 个 >100%")

    # 所有数值列（含 power）插值补全
    df[_NUMERIC_COLS] = df[_NUMERIC_COLS].interpolate(method="linear")
    df[_NUMERIC_COLS] = df[_NUMERIC_COLS].bfill().ffill()

    remaining_nan = df[_NUMERIC_COLS].isnull().sum().sum()
    if remaining_nan:
        print(f"  警告: 清洗后仍有 {remaining_nan} 个 NaN（可能整列缺失）")
    else:
        print(f"  清洗后 NaN: 0 ✓")

    # 保存 CSV
    stem = os.path.splitext(os.path.basename(xlsx_path))[0]
    out_path = os.path.join(output_dir, stem + ".csv")
    df.to_csv(out_path, index=False)
    print(f"  已保存: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="将太阳能站点 xlsx 转为标准 CSV (兼容 process2/process3)"
    )
    parser.add_argument("xlsx_files", nargs="+", help="xlsx 文件路径（支持多个）")
    parser.add_argument(
        "--output-dir", "-o", default=None,
        help="输出目录（默认与 xlsx 同目录）"
    )
    args = parser.parse_args()

    converted = []
    for xlsx_path in args.xlsx_files:
        if not os.path.exists(xlsx_path):
            print(f"文件不存在，跳过: {xlsx_path}", file=sys.stderr)
            continue
        out_dir = args.output_dir or os.path.dirname(os.path.abspath(xlsx_path))
        os.makedirs(out_dir, exist_ok=True)
        out_path = convert(xlsx_path, out_dir)
        converted.append(out_path)

    print(f"\n完成！共转换 {len(converted)} 个文件:")
    for p in converted:
        print(f"  {p}")

    if converted:
        print("\n使用转换后的 CSV 训练:")
        paths_str = " ".join(f'"{p}"' for p in converted)
        print(f"  python train3.py --mode all --csv-paths {paths_str}")


if __name__ == "__main__":
    main()
