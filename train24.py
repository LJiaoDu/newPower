"""
基于20小时历史数据预测未来4小时功率
输入：过去20小时的功率数据（240个点，5分钟间隔）
输出：未来4小时的功率预测（48个点，5分钟间隔）
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import xgboost as xgb
import pickle

# 配置参数
LOOKBACK_HOURS = 20      # 输入：过去20小时
FORECAST_HOURS = 4       # 输出：未来4小时
INTERVAL_MINUTES = 5     # 数据间隔：5分钟

LOOKBACK_POINTS = LOOKBACK_HOURS * 60 // INTERVAL_MINUTES  # 20h = 240个点
FORECAST_POINTS = FORECAST_HOURS * 60 // INTERVAL_MINUTES  # 4h = 48个点

def load_data():
    """加载数据"""
    print("="*60)
    print("加载数据...")
    print("="*60)
    df = pd.read_csv('/home/user/newPower/processed_power_data.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"总数据点: {len(df)}")
    print(f"时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
    return df

def create_sequences(df):
    """创建序列数据

    输入：过去20小时（240个点）
    输出：未来4小时（48个点）
    """
    print("\n创建训练序列...")
    print(f"输入窗口: {LOOKBACK_HOURS}小时 ({LOOKBACK_POINTS}个点)")
    print(f"输出窗口: {FORECAST_HOURS}小时 ({FORECAST_POINTS}个点)")

    power_values = df['generationPower'].values
    datetimes = df['datetime'].values
    hours = df['hour'].values
    dayofweek = df['dayofweek'].values

    X = []  # 输入特征
    y = []  # 输出目标
    timestamps = []  # 记录每个样本的起始时间

    # 滑动窗口创建样本
    total_window = LOOKBACK_POINTS + FORECAST_POINTS

    for i in range(len(power_values) - total_window + 1):
        # 输入：过去20小时的功率
        lookback_power = power_values[i:i+LOOKBACK_POINTS]

        # 输出：未来4小时的功率
        forecast_power = power_values[i+LOOKBACK_POINTS:i+total_window]

        # 添加时间特征（预测起始时刻的时间信息）
        start_hour = hours[i+LOOKBACK_POINTS]
        start_dow = dayofweek[i+LOOKBACK_POINTS]

        # 创建特征向量
        # 包含：240个历史功率 + 统计特征 + 时间特征
        features = list(lookback_power)

        # 添加统计特征
        features.extend([
            np.mean(lookback_power),           # 均值
            np.std(lookback_power),            # 标准差
            np.max(lookback_power),            # 最大值
            np.min(lookback_power),            # 最小值
            np.median(lookback_power),         # 中位数
            lookback_power[-1],                # 最近一个点的值
            np.mean(lookback_power[-12:]),     # 最近1小时均值
            np.mean(lookback_power[-60:]),     # 最近5小时均值
        ])

        # 添加时间特征（周期性编码）
        features.extend([
            start_hour,
            start_dow,
            np.sin(2 * np.pi * start_hour / 24),
            np.cos(2 * np.pi * start_hour / 24),
            np.sin(2 * np.pi * start_dow / 7),
            np.cos(2 * np.pi * start_dow / 7),
        ])

        X.append(features)
        y.append(forecast_power)
        timestamps.append(datetimes[i+LOOKBACK_POINTS])

    X = np.array(X)
    y = np.array(y)

    print(f"\n生成样本数: {len(X)}")
    print(f"输入特征维度: {X.shape[1]}")
    print(f"  - 历史功率点: {LOOKBACK_POINTS}")
    print(f"  - 统计特征: 8")
    print(f"  - 时间特征: 6")
    print(f"输出维度: {y.shape[1]} (未来{FORECAST_HOURS}小时)")

    return X, y, np.array(timestamps)

def split_data(X, y, timestamps, train_ratio=0.8):
    """划分训练集和测试集"""
    print("\n划分数据集...")
    split_idx = int(len(X) * train_ratio)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    ts_train, ts_test = timestamps[:split_idx], timestamps[split_idx:]

    print(f"训练集: {len(X_train)} 样本")
    print(f"测试集: {len(X_test)} 样本")

    return X_train, X_test, y_train, y_test, ts_train, ts_test

def train_model(X_train, y_train):
    """训练多输出XGBoost模型"""
    print("\n" + "="*60)
    print("训练模型...")
    print("="*60)

    # 基础XGBoost模型
    base_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    # 使用MultiOutputRegressor支持多输出
    model = MultiOutputRegressor(base_model, n_jobs=-1)

    print("开始训练（预测48个未来时间点）...")
    model.fit(X_train, y_train)
    print("✓ 模型训练完成")

    return model

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """评估模型"""
    print("\n" + "="*60)
    print("模型评估")
    print("="*60)

    # 训练集预测
    y_train_pred = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)

    print("\n训练集:")
    print(f"  MAE:  {train_mae:.2f} W")
    print(f"  RMSE: {train_rmse:.2f} W")
    print(f"  R²:   {train_r2:.4f}")

    # 测试集预测
    y_test_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)

    print("\n测试集:")
    print(f"  MAE:  {test_mae:.2f} W")
    print(f"  RMSE: {test_rmse:.2f} W")
    print(f"  R²:   {test_r2:.4f}")

    # 计算每个时间步的误差
    print("\n各时间步误差分析:")
    timestep_mae = np.mean(np.abs(y_test - y_test_pred), axis=0)

    for hour in [1, 2, 3, 4]:
        idx = hour * 12 - 1  # 每小时的最后一个点
        print(f"  未来第{hour}小时末: MAE = {timestep_mae[idx]:.2f} W")

    mean_power = y_test.mean()
    relative_error = (test_mae / mean_power) * 100
    accuracy = 100 - relative_error

    print(f"\n整体预测准确率: {accuracy:.2f}%")
    print(f"相对误差: {relative_error:.2f}%")

    return y_test_pred

def plot_results(y_test, y_pred, timestamps):
    """可视化预测结果"""
    print("\n生成可视化...")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # 1. 选择一个样本展示完整的4小时预测
    sample_idx = len(y_test) // 2
    time_points = np.arange(FORECAST_POINTS) * INTERVAL_MINUTES / 60  # 转换为小时

    axes[0, 0].plot(time_points, y_test[sample_idx], 'b-o', label='Actual', markersize=3, linewidth=1.5)
    axes[0, 0].plot(time_points, y_pred[sample_idx], 'r--s', label='Predicted', markersize=3, linewidth=1.5)
    axes[0, 0].set_title(f'Sample Prediction (Starting at {timestamps[sample_idx]})',
                         fontweight='bold', fontsize=12)
    axes[0, 0].set_xlabel('Hours Ahead')
    axes[0, 0].set_ylabel('Power (W)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 多个样本的预测对比（每隔100个样本取一个）
    axes[0, 1].set_title('Multiple Sample Predictions', fontweight='bold', fontsize=12)
    for i in range(0, min(len(y_test), 500), 100):
        axes[0, 1].plot(time_points, y_test[i], 'b-', alpha=0.3, linewidth=1)
        axes[0, 1].plot(time_points, y_pred[i], 'r--', alpha=0.3, linewidth=1)
    axes[0, 1].set_xlabel('Hours Ahead')
    axes[0, 1].set_ylabel('Power (W)')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 每个时间步的MAE
    timestep_mae = np.mean(np.abs(y_test - y_pred), axis=0)
    axes[1, 0].plot(time_points, timestep_mae, 'g-o', markersize=4, linewidth=2)
    axes[1, 0].set_title('MAE by Forecast Horizon', fontweight='bold', fontsize=12)
    axes[1, 0].set_xlabel('Hours Ahead')
    axes[1, 0].set_ylabel('MAE (W)')
    axes[1, 0].grid(True, alpha=0.3)
    # 添加水平线标记每小时
    for hour in [1, 2, 3, 4]:
        axes[1, 0].axvline(x=hour, color='gray', linestyle='--', alpha=0.5)
        axes[1, 0].text(hour, max(timestep_mae)*0.9, f'{hour}h', ha='center')

    # 4. 误差分布
    errors = (y_pred - y_test).flatten()
    axes[1, 1].hist(errors, bins=100, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_title('Prediction Error Distribution', fontweight='bold', fontsize=12)
    axes[1, 1].set_xlabel('Error (W)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].text(0.05, 0.95, f'Mean: {errors.mean():.2f} W\nStd: {errors.std():.2f} W',
                   transform=axes[1, 1].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/newPower/train24_results.png', dpi=150)
    print("✓ 可视化已保存: train24_results.png")

def save_model(model):
    """保存模型"""
    model_path = '/home/user/newPower/power_model_24h.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n✓ 模型已保存: {model_path}")

    # 保存模型配置信息
    config = {
        'lookback_hours': LOOKBACK_HOURS,
        'forecast_hours': FORECAST_HOURS,
        'lookback_points': LOOKBACK_POINTS,
        'forecast_points': FORECAST_POINTS,
        'interval_minutes': INTERVAL_MINUTES,
    }
    config_path = '/home/user/newPower/model_24h_config.pkl'
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    print(f"✓ 配置已保存: {config_path}")

def main():
    print("="*60)
    print("20小时历史数据 → 4小时功率预测模型")
    print("="*60)

    # 1. 加载数据
    df = load_data()

    # 2. 创建序列
    X, y, timestamps = create_sequences(df)

    # 3. 划分数据
    X_train, X_test, y_train, y_test, ts_train, ts_test = split_data(X, y, timestamps)

    # 4. 训练模型
    model = train_model(X_train, y_train)

    # 5. 评估模型
    y_pred = evaluate_model(model, X_train, y_train, X_test, y_test)

    # 6. 可视化
    plot_results(y_test, y_pred, ts_test)

    # 7. 保存模型
    save_model(model)

    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)
    print("\n使用说明:")
    print("1. 输入：过去20小时的功率数据（240个点，5分钟间隔）")
    print("2. 输出：未来4小时的功率预测（48个点，5分钟间隔）")
    print("3. 加载模型：")
    print("   import pickle")
    print("   with open('power_model_24h.pkl', 'rb') as f:")
    print("       model = pickle.load(f)")
    print("4. 预测：")
    print("   predictions = model.predict(X_new)")

if __name__ == "__main__":
    main()
