import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

def visualize_power_data():
    """可视化电功率数据"""
    print("加载数据...")
    df = pd.read_csv('/home/user/newPower/simple_power_data.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])

    # 设置中文字体(如果有的话)
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建多个子图
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))

    # 1. 时间序列图
    print("生成时间序列图...")
    axes[0].plot(df['datetime'], df['generationPower'], linewidth=0.5, alpha=0.7)
    axes[0].set_title('Generation Power Time Series', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Datetime')
    axes[0].set_ylabel('Generation Power (W)')
    axes[0].grid(True, alpha=0.3)

    # 2. 每天的功率分布
    print("生成每日功率分布图...")
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour + df['datetime'].dt.minute / 60

    # 按日期分组,绘制每天的功率曲线
    for date in df['date'].unique():
        day_data = df[df['date'] == date]
        axes[1].plot(day_data['hour'], day_data['generationPower'],
                    label=str(date), alpha=0.6, linewidth=1.5)

    axes[1].set_title('Daily Generation Power Pattern', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Hour of Day')
    axes[1].set_ylabel('Generation Power (W)')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # 3. 功率分布直方图
    print("生成功率分布直方图...")
    axes[2].hist(df['generationPower'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[2].set_title('Generation Power Distribution', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Generation Power (W)')
    axes[2].set_ylabel('Frequency')
    axes[2].grid(True, alpha=0.3, axis='y')

    # 调整布局
    plt.tight_layout()

    # 保存图片
    output_file = '/home/user/newPower/power_data_visualization.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n可视化结果已保存到: {output_file}")

    # 统计信息
    print("\n" + "="*60)
    print("数据统计摘要")
    print("="*60)
    print(f"总数据点: {len(df)}")
    print(f"时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
    print(f"\n发电功率统计:")
    print(f"  均值: {df['generationPower'].mean():.2f} W")
    print(f"  中位数: {df['generationPower'].median():.2f} W")
    print(f"  标准差: {df['generationPower'].std():.2f} W")
    print(f"  最小值: {df['generationPower'].min():.2f} W")
    print(f"  最大值: {df['generationPower'].max():.2f} W")
    print(f"  25分位数: {df['generationPower'].quantile(0.25):.2f} W")
    print(f"  75分位数: {df['generationPower'].quantile(0.75):.2f} W")

    # 零值统计
    zero_count = (df['generationPower'] == 0).sum()
    print(f"\n零功率数据点: {zero_count} ({zero_count/len(df)*100:.2f}%)")

    # 按小时统计平均功率
    print("\n每小时平均功率:")
    df['hour_only'] = df['datetime'].dt.hour
    hourly_avg = df.groupby('hour_only')['generationPower'].mean()
    for hour, power in hourly_avg.items():
        print(f"  {hour:02d}:00 - {power:8.2f} W")

if __name__ == "__main__":
    visualize_power_data()
