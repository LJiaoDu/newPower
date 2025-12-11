#!/usr/bin/env python3
"""
诊断脚本 - 找出60%准确率的原因
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

print("="*80)
print("🔍 数据和模型诊断脚本")
print("="*80)

# ==================== 1. 检查JSON原始数据 ====================
print("\n" + "="*80)
print("1️⃣  检查JSON原始数据")
print("="*80)

json_files = [f for f in os.listdir('.') if f.endswith('.json')]
print(f"\n找到 {len(json_files)} 个JSON文件:")
for f in json_files[:5]:  # 只显示前5个
    print(f"  - {f}")
if len(json_files) > 5:
    print(f"  ... 还有 {len(json_files)-5} 个文件")

if len(json_files) == 0:
    print("❌ 错误：没有找到JSON文件！")
    exit(1)

# 分析第一个JSON文件
first_json = json_files[0]
print(f"\n分析第一个文件: {first_json}")
with open(first_json, 'r') as f:
    data = json.load(f)

print("\nJSON结构:")
print(f"  - 顶层字段: {list(data.keys())}")

# 检查是新格式还是旧格式
if 'powerInfos' in data:
    print("  ✅ 识别为新格式 (powerInfos)")
    power_list = data['powerInfos']
    time_field = 'time'
    power_field = 'power'
elif 'stationStatisticPowerList' in data:
    print("  ✅ 识别为旧格式 (stationStatisticPowerList)")
    power_list = data['stationStatisticPowerList']
    time_field = 'dateTime'
    power_field = 'generationPower'
else:
    print("  ❌ 错误：无法识别格式！")
    exit(1)

print(f"  - 功率数据点数: {len(power_list)}")

# 检查第一个数据点
if len(power_list) > 0:
    first_point = power_list[0]
    print(f"\n第一个数据点示例:")
    print(f"  {json.dumps(first_point, indent=2, ensure_ascii=False)}")

    # 检查时间戳
    timestamp = first_point.get(time_field)
    if time_field == 'time':
        dt = datetime.fromtimestamp(timestamp)
    else:
        dt = datetime.fromtimestamp(timestamp / 1000)
    print(f"\n  时间戳: {timestamp}")
    print(f"  转换后时间: {dt}")
    print(f"  功率: {first_point.get(power_field)}")

# 统计所有JSON文件的数据量
total_points = 0
time_range = []
for json_file in json_files:
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        if 'powerInfos' in data:
            points = data['powerInfos']
        elif 'stationStatisticPowerList' in data:
            points = data['stationStatisticPowerList']
        else:
            continue
        total_points += len(points)

        # 提取时间范围
        for p in points:
            ts = p.get('time') or p.get('dateTime')
            if ts:
                if ts > 2000000000:  # 毫秒时间戳
                    ts = ts / 1000
                time_range.append(ts)
    except:
        pass

if time_range:
    time_range = sorted(time_range)
    start_time = datetime.fromtimestamp(time_range[0])
    end_time = datetime.fromtimestamp(time_range[-1])
    time_span = (end_time - start_time).days

    print(f"\n所有JSON文件统计:")
    print(f"  - 总数据点: {total_points:,}")
    print(f"  - 时间范围: {start_time} 到 {end_time}")
    print(f"  - 时间跨度: {time_span} 天")

    # 判断数据量是否足够
    if total_points < 5000:
        print(f"  ⚠️  警告：数据点太少！建议至少10000点 (约35天)")
    elif total_points < 10000:
        print(f"  ⚠️  数据量偏少，建议增加到10000+点")
    else:
        print(f"  ✅ 数据量充足")

    if time_span < 15:
        print(f"  ⚠️  警告：时间跨度太短！建议至少30天")
    elif time_span < 30:
        print(f"  ⚠️  时间跨度偏短，建议增加到30天+")
    else:
        print(f"  ✅ 时间跨度充足")

# ==================== 2. 检查CSV训练数据 ====================
print("\n" + "="*80)
print("2️⃣  检查CSV训练数据")
print("="*80)

csv_files = ['te_training_data.csv', 'training_data.csv']
csv_file = None
for f in csv_files:
    if os.path.exists(f):
        csv_file = f
        break

if csv_file is None:
    print("❌ 错误：没有找到训练数据CSV文件！")
    print("   请先运行: python te_data_process.py")
    exit(1)

print(f"\n读取文件: {csv_file}")
df = pd.read_csv(csv_file)

print(f"\n数据概览:")
print(f"  - 总行数: {len(df):,}")
print(f"  - 列数: {len(df.columns)}")
print(f"  - 列名: {list(df.columns)}")

print(f"\n时间范围:")
df['datetime'] = pd.to_datetime(df['datetime'])
print(f"  - 开始: {df['datetime'].min()}")
print(f"  - 结束: {df['datetime'].max()}")
print(f"  - 跨度: {(df['datetime'].max() - df['datetime'].min()).days} 天")

print(f"\n功率统计 (generationPower):")
power_stats = df['generationPower'].describe()
print(f"  - 最小值: {power_stats['min']:.2f} W")
print(f"  - 25%分位: {power_stats['25%']:.2f} W")
print(f"  - 中位数: {power_stats['50%']:.2f} W")
print(f"  - 75%分位: {power_stats['75%']:.2f} W")
print(f"  - 最大值: {power_stats['max']:.2f} W")
print(f"  - 平均值: {power_stats['mean']:.2f} W")
print(f"  - 标准差: {power_stats['std']:.2f} W")

# 分析白天/夜间分布
daytime = df[df['generationPower'] > 200]
nighttime = df[df['generationPower'] <= 200]
print(f"\n白天/夜间分布:")
print(f"  - 白天数据点 (>200W): {len(daytime):,} ({len(daytime)/len(df)*100:.1f}%)")
print(f"  - 夜间数据点 (≤200W): {len(nighttime):,} ({len(nighttime)/len(df)*100:.1f}%)")
print(f"  - 白天平均功率: {daytime['generationPower'].mean():.2f} W")

# 检查缺失值
missing = df.isnull().sum()
if missing.sum() > 0:
    print(f"\n⚠️  缺失值检测:")
    for col in missing[missing > 0].index:
        print(f"  - {col}: {missing[col]} ({missing[col]/len(df)*100:.2f}%)")
else:
    print(f"\n✅ 无缺失值")

# 检查异常值
print(f"\n异常值检测:")
zero_power = len(df[df['generationPower'] == 0])
print(f"  - 功率为0: {zero_power} ({zero_power/len(df)*100:.1f}%)")

negative_power = len(df[df['generationPower'] < 0])
if negative_power > 0:
    print(f"  ⚠️  负功率: {negative_power} (不合理!)")

# 检查数据连续性
df_sorted = df.sort_values('datetime')
time_diffs = df_sorted['datetime'].diff()
print(f"\n时间间隔统计:")
print(f"  - 平均间隔: {time_diffs.mean()}")
print(f"  - 最小间隔: {time_diffs.min()}")
print(f"  - 最大间隔: {time_diffs.max()}")

large_gaps = time_diffs[time_diffs > pd.Timedelta(minutes=10)]
if len(large_gaps) > 0:
    print(f"  ⚠️  大间隔 (>10分钟): {len(large_gaps)} 处")

# ==================== 3. 检查模型检查点 ====================
print("\n" + "="*80)
print("3️⃣  检查模型训练情况")
print("="*80)

checkpoint_dirs = ['te_checkpoint', 'checkpoint']
checkpoint_dir = None
for d in checkpoint_dirs:
    if os.path.exists(d):
        checkpoint_dir = d
        break

if checkpoint_dir is None:
    print("❌ 没有找到检查点目录，模型还未训练")
else:
    print(f"\n检查点目录: {checkpoint_dir}/")
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    print(f"找到 {len(checkpoints)} 个检查点文件")

    if len(checkpoints) > 0:
        # 找最新的检查点
        latest = max(checkpoints, key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
        print(f"\n最新检查点: {latest}")

        import torch
        checkpoint = torch.load(os.path.join(checkpoint_dir, latest), map_location='cpu')

        print(f"\n训练信息:")
        print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  - Val Loss: {checkpoint.get('val_loss', 'N/A'):.6f}")
        print(f"  - Val ACC (MAE): {checkpoint.get('val_acc_mae', 'N/A'):.4f}")
        print(f"  - Val ACC (RMSE): {checkpoint.get('val_acc_rmse', 'N/A'):.4f}")
        print(f"  - Power Max: {checkpoint.get('power_max', 'N/A'):.2f} W")
        print(f"  - Global Avg Power: {checkpoint.get('global_avg_power', 'N/A'):.2f} W")

        # 反推MAE
        val_acc_mae = checkpoint.get('val_acc_mae')
        global_avg_power = checkpoint.get('global_avg_power')
        if val_acc_mae and global_avg_power:
            mae = (1 - val_acc_mae) * global_avg_power
            print(f"\n反推的MAE: {mae:.2f} W")
            print(f"相对误差: {mae/global_avg_power*100:.1f}%")

# ==================== 4. 检查预测图 ====================
print("\n" + "="*80)
print("4️⃣  检查预测图")
print("="*80)

plot_dirs = ['te_prediction_plots', 'prediction_plots']
plot_dir = None
for d in plot_dirs:
    if os.path.exists(d):
        plot_dir = d
        break

if plot_dir is None:
    print("❌ 没有找到预测图目录")
else:
    plots = [f for f in os.listdir(plot_dir) if f.endswith('.png')]
    print(f"\n找到 {len(plots)} 张预测图")
    if len(plots) > 0:
        latest_plot = max(plots, key=lambda x: os.path.getmtime(os.path.join(plot_dir, x)))
        print(f"最新预测图: {plot_dir}/{latest_plot}")
        print(f"  查看命令: open {plot_dir}/{latest_plot}")

# ==================== 5. 基准测试 ====================
print("\n" + "="*80)
print("5️⃣  基准测试 - 简单模型能达到什么效果？")
print("="*80)

if csv_file:
    print("\n使用简单的持续性预测 (Persistence Model):")
    print("  假设: 未来功率 = 当前功率")

    # 模拟持续性预测的MAE
    daytime_data = df[df['generationPower'] > 200]['generationPower'].values
    if len(daytime_data) > 240 + 48:
        # 简单测试：用前240个预测后48个
        errors = []
        for i in range(0, len(daytime_data) - 240 - 48, 100):
            current = daytime_data[i + 239]  # 最后一个历史点
            future = daytime_data[i + 240 : i + 240 + 48]  # 未来48个点
            error = np.mean(np.abs(future - current))
            errors.append(error)

        if errors:
            baseline_mae = np.mean(errors)
            baseline_acc = 1 - baseline_mae / daytime['generationPower'].mean()
            print(f"\n  持续性模型 MAE: {baseline_mae:.2f} W")
            print(f"  持续性模型 ACC: {baseline_acc:.4f} ({baseline_acc*100:.1f}%)")
            print(f"\n  💡 你的模型应该超过这个基准！")

# ==================== 6. 问题总结和建议 ====================
print("\n" + "="*80)
print("6️⃣  问题总结和改进建议")
print("="*80)

issues = []
suggestions = []

# 数据量检查
if total_points < 10000:
    issues.append("❌ 数据量不足")
    suggestions.append("1. 收集更多JSON文件（至少30天数据）")

if time_span < 30:
    issues.append("❌ 时间跨度太短")
    suggestions.append("2. 增加数据时间跨度到30天以上")

# CSV数据检查
if len(df) < 8000:
    issues.append("❌ CSV训练数据太少")
    suggestions.append("3. 检查te_data_process.py是否正确处理了所有JSON文件")

if len(daytime) / len(df) < 0.3:
    issues.append("⚠️  白天数据占比太低")

# 特征检查
if len(df.columns) <= 10:
    issues.append("⚠️  特征数量较少")
    suggestions.append("4. 增加更多特征（历史统计、功率变化率、天气数据）")

print("\n发现的问题:")
if issues:
    for issue in issues:
        print(f"  {issue}")
else:
    print("  ✅ 数据质量良好")

print("\n改进建议（按优先级）:")
if suggestions:
    for sug in suggestions:
        print(f"  {sug}")
else:
    print("  数据层面没有明显问题，建议:")
    print("  1. 增加模型容量 (--hidden-feat-size 512)")
    print("  2. 训练更多epochs (--epochs 200)")
    print("  3. 尝试更长的历史窗口 (--in-seq-len 480)")
    print("  4. 添加天气特征（太阳辐射、云量等）")

print("\n" + "="*80)
print("诊断完成！")
print("="*80)
print("\n下一步:")
print("  1. 如果数据量不足 → 收集更多JSON文件")
print("  2. 如果数据充足 → 运行改进版训练脚本")
print("  3. 分享诊断结果给我 → 我给出精确的解决方案")
