import json
import pandas as pd
from datetime import datetime, timedelta
import os

def fix_timestamp_format(data, filename):
    """检查并修复时间戳格式

    如果时间戳是秒级(10位)，自动转换为毫秒级(13位)
    """
    fixed = False

    # 修复功率列表中的时间戳
    power_list = data.get('stationStatisticPowerList', [])

    for item in power_list:
        if 'dateTime' in item:
            old_time = item['dateTime']
            # 如果小于2000000000（约2033-05-18的秒级时间戳），说明是秒级
            if old_time < 2000000000:
                item['dateTime'] = old_time * 1000
                fixed = True

        if 'updateTime' in item:
            old_time = item['updateTime']
            if old_time < 2000000000:
                item['updateTime'] = old_time * 1000

    if fixed:
        print(f"  ⚠️  检测到秒级时间戳，已自动转换为毫秒级")

    return data

def load_json_files(directory):
    """加载所有JSON文件并提取功率数据"""
    all_data = []

    # 只加载数据文件，跳过模型文件
    json_files = sorted([f for f in os.listdir(directory)
                        if f.endswith('.json') and f.startswith('772_')])

    for filename in json_files:
        filepath = os.path.join(directory, filename)
        print(f"处理文件: {filename}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 检查并修复时间戳格式
        data = fix_timestamp_format(data, filename)

        # 提取功率列表数据
        power_list = data.get('stationStatisticPowerList', [])

        for item in power_list:
            # 提取时间戳(毫秒)和发电功率
            timestamp_ms = item.get('dateTime')
            power = item.get('generationPower')

            if timestamp_ms is not None and power is not None:
                # 转换为datetime对象
                dt = datetime.fromtimestamp(timestamp_ms / 1000)
                all_data.append({
                    'datetime': dt,
                    'generationPower': power
                })

        print(f"  - 提取了 {len(power_list)} 个数据点")

    return pd.DataFrame(all_data)

def resample_to_5min(df):
    """将数据重采样为标准5分钟间隔"""
    print("\n开始重采样到5分钟间隔...")

    # 设置datetime为索引
    df = df.set_index('datetime')

    # 排序
    df = df.sort_index()

    print(f"原始数据范围: {df.index.min()} 到 {df.index.max()}")
    print(f"原始数据点数: {len(df)}")

    # 重采样到5分钟间隔,使用平均值
    # 如果同一个5分钟窗口内有多个值,取平均
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

    # 处理首尾可能还存在的NaN(插值无法处理的边界情况)
    df['generationPower'] = df['generationPower'].fillna(0)

    remaining_missing = df['generationPower'].isna().sum()
    print(f"填充后剩余缺失数据: {remaining_missing}")

    return df

def add_time_features(df):
    """添加时间特征,用于机器学习"""
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

def analyze_data(df_original, df_processed):
    """分析数据质量"""
    print("\n" + "="*60)
    print("数据分析报告")
    print("="*60)

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
    # 当前目录
    directory = '/home/user/newPower'

    print("="*60)
    print("电功率数据处理脚本")
    print("="*60)

    # 1. 加载所有JSON文件
    df_original = load_json_files(directory)

    # 2. 重采样到5分钟间隔
    df_resampled = resample_to_5min(df_original)

    # 3. 填充缺失数据(可以选择不同的方法)
    # 对于发电功率,我们使用线性插值,因为功率变化通常是连续的
    df_filled = fill_missing_data(df_resampled.copy(), method='interpolate')

    # 4. 添加时间特征
    df_final = add_time_features(df_filled)

    # 5. 数据分析
    df_final = analyze_data(df_original, df_final)

    # 6. 保存处理后的数据
    # 保存为CSV格式
    output_csv = os.path.join(directory, 'processed_power_data.csv')
    df_final.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"\n已保存处理后的数据到: {output_csv}")

    # 保存为带完整特征的训练数据
    output_train = os.path.join(directory, 'training_data.csv')
    df_final.to_csv(output_train, index=False, encoding='utf-8')
    print(f"已保存训练数据到: {output_train}")

    # 7. 生成简化版本(只包含日期时间和功率)
    df_simple = df_final[['datetime', 'generationPower']].copy()
    output_simple = os.path.join(directory, 'simple_power_data.csv')
    df_simple.to_csv(output_simple, index=False, encoding='utf-8')
    print(f"已保存简化版本到: {output_simple}")

    print("\n" + "="*60)
    print("数据处理完成!")
    print("="*60)
    print("\n生成的文件:")
    print("1. processed_power_data.csv - 带时间特征的完整数据")
    print("2. training_data.csv - 用于训练的数据(与processed_power_data.csv相同)")
    print("3. simple_power_data.csv - 只包含时间和功率的简化数据")

    # 显示前几行数据
    print("\n数据预览(前10行):")
    print(df_final[['datetime', 'generationPower', 'year', 'month', 'day', 'hour', 'minute']].head(10))

if __name__ == "__main__":
    main()
