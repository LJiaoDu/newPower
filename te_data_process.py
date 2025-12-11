#!/usr/bin/env python3
"""
新数据格式处理脚本 - 适配2025-03-13.json格式

与原process_data.py的区别：
1. 支持新的JSON结构 (powerInfos 代替 stationStatisticPowerList)
2. 时间字段从 dateTime (毫秒) 改为 time (秒)
3. 功率字段从 generationPower 改为 power
4. 增加对经纬度信息的处理
"""

import json
import pandas as pd
from datetime import datetime
import os


def load_json_files_new_format(directory):
    """加载新格式JSON文件并提取功率数据

    新格式特点：
    - 使用 powerInfos 代替 stationStatisticPowerList
    - 时间字段为 time (秒时间戳)
    - 功率字段为 power
    - 包含 locationLat 和 locationLng
    """
    all_data = []
    location_info = {}

    # 加载所有JSON文件 (支持新格式命名: YYYY-MM-DD.json)
    json_files = sorted([f for f in os.listdir(directory)
                        if f.endswith('.json')])

    if not json_files:
        print(f"警告: 在 {directory} 中未找到JSON文件")
        return pd.DataFrame(), location_info

    print(f"找到 {len(json_files)} 个JSON文件\n")

    for filename in json_files:
        filepath = os.path.join(directory, filename)
        print(f"处理文件: {filename}")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 提取位置信息 (只在第一个文件提取)
            if not location_info and 'locationLat' in data:
                location_info = {
                    'systemId': data.get('systemId'),
                    'latitude': data.get('locationLat'),
                    'longitude': data.get('locationLng')
                }
                print(f"  系统ID: {location_info['systemId']}")
                print(f"  位置: ({location_info['latitude']}, {location_info['longitude']})")

            # 提取功率列表数据 (新格式)
            power_list = data.get('powerInfos', [])

            if not power_list:
                print(f"  警告: 文件中没有powerInfos数据")
                continue

            for item in power_list:
                # 新格式字段: time (秒), power
                timestamp_sec = item.get('time')
                power = item.get('power')

                if timestamp_sec is not None and power is not None:
                    # 从秒时间戳转换为datetime对象
                    dt = datetime.fromtimestamp(timestamp_sec)
                    all_data.append({
                        'datetime': dt,
                        'generationPower': power  # 统一使用generationPower保持与后续代码兼容
                    })

            print(f"  - 提取了 {len(power_list)} 个数据点")

        except json.JSONDecodeError as e:
            print(f"  错误: JSON解析失败 - {e}")
        except Exception as e:
            print(f"  错误: {e}")

    print(f"\n总共提取了 {len(all_data)} 个数据点")
    return pd.DataFrame(all_data), location_info


def resample_to_5min(df):
    """将数据重采样为标准5分钟间隔"""
    if df.empty:
        print("错误: 数据为空，无法重采样")
        return df

    print("\n开始重采样到5分钟间隔...")

    # 设置datetime为索引
    df = df.set_index('datetime')

    # 排序
    df = df.sort_index()

    print(f"原始数据范围: {df.index.min()} 到 {df.index.max()}")
    print(f"原始数据点数: {len(df)}")

    # 重采样到5分钟间隔,使用平均值
    df_resampled = df.resample('5min').mean()

    # 统计缺失数据
    missing_count = df_resampled['generationPower'].isna().sum()
    print(f"\n重采样后的数据点数: {len(df_resampled)}")
    print(f"缺失数据点数: {missing_count} ({missing_count/len(df_resampled)*100:.2f}%)")

    return df_resampled


def fill_missing_data(df, method='interpolate'):
    """填充缺失数据

    method: 'interpolate' - 线性插值
            'zero' - 填充为0
            'forward' - 前向填充
    """
    if df.empty:
        return df

    print(f"\n使用 '{method}' 方法填充缺失数据...")

    if method == 'interpolate':
        # 线性插值
        df['generationPower'] = df['generationPower'].interpolate(method='linear')
    elif method == 'zero':
        # 填充为0
        df['generationPower'] = df['generationPower'].fillna(0)
    elif method == 'forward':
        # 前向填充
        df['generationPower'] = df['generationPower'].fillna(method='ffill')

    # 处理首尾可能还存在的NaN
    df['generationPower'] = df['generationPower'].fillna(0)

    remaining_missing = df['generationPower'].isna().sum()
    print(f"填充后剩余缺失数据: {remaining_missing}")

    return df


def add_time_features(df):
    """添加时间特征，用于机器学习"""
    if df.empty:
        return df

    df = df.reset_index()
    df.rename(columns={'index': 'datetime'}, inplace=True)

    # 添加时间特征
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['dayofweek'] = df['datetime'].dt.dayofweek  # 0=周一, 6=周日
    df['dayofyear'] = df['datetime'].dt.dayofyear

    # 时间序列的秒数(从第一个时间点开始)
    df['time_idx'] = (df['datetime'] - df['datetime'].min()).dt.total_seconds() / 60  # 分钟数

    return df


def analyze_data(df_original, df_processed, location_info):
    """分析数据质量"""
    print("\n" + "="*60)
    print("数据分析报告 (新格式)")
    print("="*60)

    # 位置信息
    if location_info:
        print("\n位置信息:")
        print(f"  系统ID: {location_info.get('systemId', 'N/A')}")
        print(f"  纬度: {location_info.get('latitude', 'N/A')}")
        print(f"  经度: {location_info.get('longitude', 'N/A')}")

    if df_original.empty:
        print("\n警告: 原始数据为空")
        return df_processed

    # 确保df_original有datetime索引用于分析
    if 'datetime' in df_original.columns:
        df_orig_indexed = df_original.set_index('datetime').sort_index()
    else:
        df_orig_indexed = df_original

    print("\n原始数据统计:")
    print(f"  数据点数: {len(df_orig_indexed)}")
    time_span = (df_orig_indexed.index.max() - df_orig_indexed.index.min())
    print(f"  时间跨度: {time_span.days} 天 {time_span.seconds // 3600} 小时")
    avg_interval = df_orig_indexed.index.to_series().diff().mean()
    print(f"  平均间隔: {avg_interval}")

    if df_processed.empty:
        print("\n警告: 处理后数据为空")
        return df_processed

    print("\n处理后数据统计:")
    print(f"  数据点数: {len(df_processed)}")
    time_span_processed = (df_processed['datetime'].max() - df_processed['datetime'].min())
    print(f"  时间跨度: {time_span_processed.days} 天 {time_span_processed.seconds // 3600} 小时")
    print(f"  理论数据点数(5分钟间隔): {time_span_processed.total_seconds() / 300 + 1:.0f}")

    print("\n发电功率统计:")
    print(f"  最小值: {df_processed['generationPower'].min():.2f}")
    print(f"  最大值: {df_processed['generationPower'].max():.2f}")
    print(f"  平均值: {df_processed['generationPower'].mean():.2f}")
    print(f"  标准差: {df_processed['generationPower'].std():.2f}")

    # 每天的数据点数统计
    df_processed['date'] = df_processed['datetime'].dt.date
    daily_counts = df_processed.groupby('date').size()
    print(f"\n每天数据点数统计:")
    print(f"  期望值: 288 (24小时 * 12个5分钟)")
    print(f"  实际平均: {daily_counts.mean():.1f}")
    print(f"  最小值: {daily_counts.min()}")
    print(f"  最大值: {daily_counts.max()}")

    return df_processed


def main():
    """主函数"""
    # 数据目录 (可以根据需要修改)
    directory = '.'  # 当前目录

    print("="*60)
    print("新格式电功率数据处理脚本")
    print("支持格式: powerInfos + time(秒) + power")
    print("="*60)

    # 1. 加载所有JSON文件 (新格式)
    df_original, location_info = load_json_files_new_format(directory)

    if df_original.empty:
        print("\n错误: 未能加载任何数据，请检查:")
        print("  1. 目录中是否有JSON文件")
        print("  2. JSON文件格式是否正确 (需要包含powerInfos字段)")
        return

    # 2. 重采样到5分钟间隔
    df_resampled = resample_to_5min(df_original)

    # 3. 填充缺失数据
    df_filled = fill_missing_data(df_resampled.copy(), method='interpolate')

    # 4. 添加时间特征
    df_final = add_time_features(df_filled)

    # 5. 数据分析
    df_final = analyze_data(df_original, df_final, location_info)

    # 6. 保存处理后的数据
    # 保存为CSV格式
    output_csv = 'te_processed_power_data.csv'
    df_final.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"\n已保存处理后的数据到: {output_csv}")

    # 保存为带完整特征的训练数据
    output_train = 'te_training_data.csv'
    df_final.to_csv(output_train, index=False, encoding='utf-8')
    print(f"已保存训练数据到: {output_train}")

    # 7. 生成简化版本(只包含日期时间和功率)
    df_simple = df_final[['datetime', 'generationPower']].copy()
    output_simple = 'te_simple_power_data.csv'
    df_simple.to_csv(output_simple, index=False, encoding='utf-8')
    print(f"已保存简化版本到: {output_simple}")

    # 保存位置信息到JSON
    if location_info:
        output_location = 'te_location_info.json'
        with open(output_location, 'w', encoding='utf-8') as f:
            json.dump(location_info, f, indent=2, ensure_ascii=False)
        print(f"已保存位置信息到: {output_location}")

    print("\n" + "="*60)
    print("数据处理完成!")
    print("="*60)
    print("\n生成的文件:")
    print("1. te_processed_power_data.csv - 带时间特征的完整数据")
    print("2. te_training_data.csv - 用于训练的数据")
    print("3. te_simple_power_data.csv - 只包含时间和功率的简化数据")
    print("4. te_location_info.json - 位置信息 (如果有)")

    # 显示前几行数据
    print("\n数据预览(前10行):")
    print(df_final[['datetime', 'generationPower', 'year', 'month', 'day', 'hour', 'minute']].head(10))


if __name__ == "__main__":
    main()
