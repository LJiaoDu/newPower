"""
深入分析电功率数据规律
帮助发现可以提高预测准确率的特征和模式
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy import stats
import seaborn as sns

def load_data():
    """加载数据"""
    df = pd.read_csv('/home/user/newPower/processed_power_data.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

def analyze_daily_pattern(df):
    """分析每日功率模式"""
    print("\n" + "="*60)
    print("1. 每日功率模式分析")
    print("="*60)

    df['hour_decimal'] = df['hour'] + df['minute'] / 60
    df['date'] = df['datetime'].dt.date

    # 每个小时的平均功率和标准差
    hourly_stats = df.groupby('hour')['generationPower'].agg(['mean', 'std', 'min', 'max'])
    print("\n每小时统计:")
    print(hourly_stats.round(2))

    # 发现峰值时段
    peak_hour = hourly_stats['mean'].idxmax()
    peak_power = hourly_stats['mean'].max()
    print(f"\n峰值发电时段: {peak_hour}:00, 平均功率: {peak_power:.2f} W")

    # 发现低谷时段
    valley_hour = hourly_stats['mean'].idxmin()
    valley_power = hourly_stats['mean'].min()
    print(f"低谷时段: {valley_hour}:00, 平均功率: {valley_power:.2f} W")

    return df

def analyze_day_of_week_pattern(df):
    """分析星期几的影响"""
    print("\n" + "="*60)
    print("2. 星期模式分析")
    print("="*60)

    weekday_stats = df.groupby('dayofweek')['generationPower'].agg(['mean', 'std'])
    weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    weekday_stats.index = [weekday_names[i] for i in weekday_stats.index]

    print("\n各星期的平均功率:")
    print(weekday_stats.round(2))

    # 判断是否有显著的星期效应
    f_stat, p_value = stats.f_oneway(*[df[df['dayofweek']==i]['generationPower']
                                        for i in range(7)])
    print(f"\nANOVA检验 (星期效应): F={f_stat:.2f}, p={p_value:.4f}")
    if p_value < 0.05:
        print("✓ 不同星期之间有显著差异")
    else:
        print("✗ 不同星期之间无显著差异")

def analyze_power_transitions(df):
    """分析功率变化规律"""
    print("\n" + "="*60)
    print("3. 功率变化规律分析")
    print("="*60)

    # 计算功率变化率
    df['power_change'] = df['generationPower'].diff()
    df['power_change_pct'] = df['generationPower'].pct_change() * 100

    print("\n功率变化统计:")
    print(f"平均变化: {df['power_change'].mean():.2f} W")
    print(f"标准差: {df['power_change'].std():.2f} W")
    print(f"最大增长: {df['power_change'].max():.2f} W")
    print(f"最大下降: {df['power_change'].min():.2f} W")

    # 识别快速变化的时段
    rapid_changes = df[abs(df['power_change']) > df['power_change'].std() * 2]
    print(f"\n快速变化时段数量: {len(rapid_changes)} ({len(rapid_changes)/len(df)*100:.1f}%)")

    return df

def analyze_zero_power_pattern(df):
    """分析零功率模式"""
    print("\n" + "="*60)
    print("4. 零功率模式分析")
    print("="*60)

    df['is_zero'] = (df['generationPower'] == 0).astype(int)

    # 按小时统计零功率比例
    zero_by_hour = df.groupby('hour')['is_zero'].mean() * 100
    print("\n各小时零功率比例:")
    for hour, pct in zero_by_hour.items():
        print(f"  {hour:02d}:00 - {pct:5.1f}%")

    # 发现零功率时段
    high_zero_hours = zero_by_hour[zero_by_hour > 50].index.tolist()
    print(f"\n零功率集中时段 (>50%): {high_zero_hours}")

def analyze_correlations(df):
    """分析特征相关性"""
    print("\n" + "="*60)
    print("5. 特征相关性分析")
    print("="*60)

    # 选择数值特征
    numeric_features = ['generationPower', 'hour', 'minute', 'dayofweek',
                       'dayofyear', 'month', 'day']
    corr_matrix = df[numeric_features].corr()

    print("\n与功率最相关的特征:")
    power_corr = corr_matrix['generationPower'].sort_values(ascending=False)
    print(power_corr[1:].round(3))  # 排除自己

    return corr_matrix

def identify_peak_periods(df):
    """识别发电高峰和低谷时段"""
    print("\n" + "="*60)
    print("6. 发电时段划分")
    print("="*60)

    # 定义高峰、正常、低谷
    q75 = df['generationPower'].quantile(0.75)
    q25 = df['generationPower'].quantile(0.25)

    df['power_level'] = pd.cut(df['generationPower'],
                                bins=[-np.inf, q25, q75, np.inf],
                                labels=['低谷', '正常', '高峰'])

    print(f"\n功率区间定义:")
    print(f"  低谷: 0 - {q25:.2f} W")
    print(f"  正常: {q25:.2f} - {q75:.2f} W")
    print(f"  高峰: {q75:.2f}+ W")

    # 各时段的小时分布
    print("\n各时段在不同小时的分布:")
    period_hour = pd.crosstab(df['hour'], df['power_level'], normalize='index') * 100
    print(period_hour.round(1))

    return df

def suggest_features(df):
    """建议新特征"""
    print("\n" + "="*60)
    print("7. 特征工程建议")
    print("="*60)

    suggestions = []

    # 1. 时间周期特征
    print("\n建议添加的特征:")
    print("1. 周期性编码 (消除时间的离散性)")
    suggestions.append("hour_sin, hour_cos = np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)")
    suggestions.append("minute_sin, minute_cos = np.sin(2*np.pi*minute/60), np.cos(2*np.pi*minute/60)")

    # 2. 滞后特征
    print("2. 扩展滞后特征")
    suggestions.append("power_lag_24 (前2小时)")
    suggestions.append("power_lag_288 (前一天同一时刻)")

    # 3. 滚动统计
    print("3. 滚动窗口特征")
    suggestions.append("rolling_mean_24 (2小时平均)")
    suggestions.append("rolling_std_24 (2小时波动)")
    suggestions.append("rolling_max_12, rolling_min_12")

    # 4. 差分特征
    print("4. 差分特征")
    suggestions.append("power_diff_1 (与上一时刻的差值)")
    suggestions.append("power_diff_12 (与1小时前的差值)")

    # 5. 时段标记
    print("5. 时段分类特征")
    suggestions.append("is_peak_hour (是否高峰时段)")
    suggestions.append("is_zero_hour (是否零功率时段)")

    return suggestions

def visualize_patterns(df, corr_matrix):
    """可视化分析结果"""
    print("\n生成可视化图表...")

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    # 1. 每日功率曲线
    df['date'] = df['datetime'].dt.date
    for date in df['date'].unique():
        day_data = df[df['date'] == date]
        axes[0, 0].plot(day_data['hour'] + day_data['minute']/60,
                       day_data['generationPower'],
                       alpha=0.6, linewidth=1.5, label=str(date))
    axes[0, 0].set_title('Daily Power Patterns', fontweight='bold')
    axes[0, 0].set_xlabel('Hour of Day')
    axes[0, 0].set_ylabel('Power (W)')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 每小时箱线图
    hourly_data = [df[df['hour']==h]['generationPower'].values for h in range(24)]
    axes[0, 1].boxplot(hourly_data, labels=range(24))
    axes[0, 1].set_title('Hourly Power Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Hour')
    axes[0, 1].set_ylabel('Power (W)')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 功率自相关图
    from pandas.plotting import autocorrelation_plot
    autocorrelation_plot(df['generationPower'], ax=axes[1, 0])
    axes[1, 0].set_title('Power Autocorrelation', fontweight='bold')
    axes[1, 0].set_xlim([0, 300])

    # 4. 功率变化分布
    axes[1, 1].hist(df['power_change'].dropna(), bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Power Change Distribution', fontweight='bold')
    axes[1, 1].set_xlabel('Power Change (W)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)

    # 5. 星期模式
    weekday_mean = df.groupby('dayofweek')['generationPower'].mean()
    weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    axes[2, 0].bar(range(7), weekday_mean.values)
    axes[2, 0].set_xticks(range(7))
    axes[2, 0].set_xticklabels(weekday_names)
    axes[2, 0].set_title('Average Power by Day of Week', fontweight='bold')
    axes[2, 0].set_ylabel('Power (W)')
    axes[2, 0].grid(True, alpha=0.3, axis='y')

    # 6. 相关性热图
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=axes[2, 1], cbar_kws={'label': 'Correlation'})
    axes[2, 1].set_title('Feature Correlation Heatmap', fontweight='bold')

    plt.tight_layout()
    plt.savefig('/home/user/newPower/pattern_analysis.png', dpi=150, bbox_inches='tight')
    print("分析图表已保存到: pattern_analysis.png")

def main():
    print("="*60)
    print("电功率数据深度分析")
    print("目标: 发现规律，提高预测准确率")
    print("="*60)

    # 加载数据
    df = load_data()
    print(f"\n加载数据: {len(df)} 行")

    # 1. 每日模式分析
    df = analyze_daily_pattern(df)

    # 2. 星期模式分析
    analyze_day_of_week_pattern(df)

    # 3. 功率变化规律
    df = analyze_power_transitions(df)

    # 4. 零功率模式
    analyze_zero_power_pattern(df)

    # 5. 相关性分析
    corr_matrix = analyze_correlations(df)

    # 6. 时段划分
    df = identify_peak_periods(df)

    # 7. 特征建议
    suggest_features(df)

    # 8. 可视化
    visualize_patterns(df, corr_matrix)

    print("\n" + "="*60)
    print("分析完成!")
    print("="*60)
    print("\n关键发现将帮助你:")
    print("1. 理解数据的内在规律")
    print("2. 设计更好的特征")
    print("3. 选择合适的模型")
    print("4. 提高预测准确率")

if __name__ == "__main__":
    main()
