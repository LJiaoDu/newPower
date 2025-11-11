"""
改进的训练脚本
基于数据分析发现的规律，优化特征工程，提高预测准确率
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import xgboost as xgb

def load_data():
    """加载数据"""
    print("加载数据...")
    df = pd.read_csv('/home/user/newPower/processed_power_data.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"原始数据: {df.shape}")
    return df

def create_advanced_features(df):
    """创建高级特征 - 基于数据分析的发现"""
    print("\n创建高级特征...")
    df = df.copy()

    # ============ 1. 周期性编码 (消除时间的线性假设) ============
    print("  1. 周期性编码...")
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    # ============ 2. 时段分类特征 (基于发现的模式) ============
    print("  2. 时段分类特征...")
    # 峰值时段 (4:00-8:00)
    df['is_peak_hour'] = ((df['hour'] >= 4) & (df['hour'] <= 8)).astype(int)
    # 零功率时段 (12:00-21:00)
    df['is_zero_hour'] = ((df['hour'] >= 12) & (df['hour'] <= 21)).astype(int)
    # 过渡时段 (21:00-4:00)
    df['is_transition_hour'] = (((df['hour'] >= 21) | (df['hour'] <= 4)) &
                                 (df['is_peak_hour'] == 0)).astype(int)

    # ============ 3. 滞后特征 (更多历史信息) ============
    print("  3. 滞后特征...")
    # 短期滞后 (前1小时)
    for i in [1, 2, 3, 6, 12]:
        df[f'power_lag_{i}'] = df['generationPower'].shift(i)

    # 长期滞后 (前一天同一时刻)
    df['power_lag_288'] = df['generationPower'].shift(288)  # 24小时 * 12个5分钟

    # ============ 4. 滚动统计特征 (捕捉趋势) ============
    print("  4. 滚动统计特征...")
    windows = [6, 12, 24, 36]  # 30分钟, 1小时, 2小时, 3小时
    for w in windows:
        df[f'rolling_mean_{w}'] = df['generationPower'].rolling(window=w, min_periods=1).mean()
        df[f'rolling_std_{w}'] = df['generationPower'].rolling(window=w, min_periods=1).std()
        df[f'rolling_max_{w}'] = df['generationPower'].rolling(window=w, min_periods=1).max()
        df[f'rolling_min_{w}'] = df['generationPower'].rolling(window=w, min_periods=1).min()

    # ============ 5. 差分特征 (变化率) ============
    print("  5. 差分特征...")
    df['power_diff_1'] = df['generationPower'].diff(1)
    df['power_diff_12'] = df['generationPower'].diff(12)  # 与1小时前的差
    df['power_pct_change_1'] = df['generationPower'].pct_change(1)

    # ============ 6. 交互特征 (基于相关性分析) ============
    print("  6. 交互特征...")
    df['hour_dayofweek'] = df['hour'] * df['dayofweek']
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['weekend_hour'] = df['is_weekend'] * df['hour']

    # ============ 7. 统计特征 ============
    print("  7. 统计特征...")
    # 当前功率相对于各窗口平均的偏差
    df['power_vs_6h'] = df['generationPower'] - df['rolling_mean_6']
    df['power_vs_12h'] = df['generationPower'] - df['rolling_mean_12']

    # 删除NaN和inf (由于滞后和滚动窗口产生)
    original_len = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)  # 将inf替换为nan
    df = df.dropna()
    print(f"  删除NaN/inf后: {len(df)} 行 (原始: {original_len})")

    return df

def select_features(df):
    """选择特征"""
    # 基础时间特征
    base_features = [
        'hour', 'minute', 'dayofweek', 'dayofyear',
        'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos',
        'dayofweek_sin', 'dayofweek_cos'
    ]

    # 时段特征
    period_features = [
        'is_peak_hour', 'is_zero_hour', 'is_transition_hour',
        'is_weekend', 'weekend_hour'
    ]

    # 滞后特征
    lag_features = [col for col in df.columns if 'lag' in col]

    # 滚动特征
    rolling_features = [col for col in df.columns if 'rolling' in col]

    # 差分特征
    diff_features = ['power_diff_1', 'power_diff_12', 'power_pct_change_1']

    # 统计特征
    stat_features = ['power_vs_6h', 'power_vs_12h']

    # 交互特征
    interaction_features = ['hour_dayofweek']

    all_features = (base_features + period_features + lag_features +
                   rolling_features + diff_features + stat_features +
                   interaction_features)

    print(f"\n总特征数: {len(all_features)}")
    print(f"  基础特征: {len(base_features)}")
    print(f"  时段特征: {len(period_features)}")
    print(f"  滞后特征: {len(lag_features)}")
    print(f"  滚动特征: {len(rolling_features)}")
    print(f"  差分特征: {len(diff_features)}")
    print(f"  统计特征: {len(stat_features)}")
    print(f"  交互特征: {len(interaction_features)}")

    return df, all_features

def train_improved_model(X_train, y_train, X_test, y_test):
    """训练改进的模型"""
    print("\n训练改进的XGBoost模型...")

    # 优化的超参数
    model = xgb.XGBRegressor(
        n_estimators=200,      # 增加树的数量
        max_depth=8,           # 增加深度
        learning_rate=0.05,    # 降低学习率
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,         # L1正则化
        reg_lambda=1.0,        # L2正则化
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train,
             eval_set=[(X_test, y_test)],
             verbose=False)

    print("模型训练完成!")
    return model

def evaluate_improved_model(model, X_train, y_train, X_test, y_test):
    """评估改进模型"""
    print("\n" + "="*60)
    print("改进模型评估结果")
    print("="*60)

    # 训练集
    y_train_pred = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)

    print("\n训练集:")
    print(f"  MAE:  {train_mae:.2f} W")
    print(f"  RMSE: {train_rmse:.2f} W")
    print(f"  R²:   {train_r2:.4f}")

    # 测试集
    y_test_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)

    print("\n测试集:")
    print(f"  MAE:  {test_mae:.2f} W")
    print(f"  RMSE: {test_rmse:.2f} W")
    print(f"  R²:   {test_r2:.4f}")

    mean_power = y_test.mean()
    relative_mae = (test_mae / mean_power) * 100
    print(f"\n相对误差: {relative_mae:.2f}%")

    # 与基准模型对比
    print("\n" + "="*60)
    print("与基准模型对比")
    print("="*60)
    print("基准模型 (example_train.py):")
    print("  测试集 MAE:  651.07 W")
    print("  测试集 R²:   0.9303")
    print("  相对误差:    16.79%")
    print("\n改进模型:")
    print(f"  测试集 MAE:  {test_mae:.2f} W")
    print(f"  测试集 R²:   {test_r2:.4f}")
    print(f"  相对误差:    {relative_mae:.2f}%")
    print("\n改进:")
    print(f"  MAE 降低:    {651.07 - test_mae:.2f} W ({(651.07-test_mae)/651.07*100:.1f}%)")
    print(f"  R² 提升:     {test_r2 - 0.9303:.4f}")
    print(f"  误差率降低:  {16.79 - relative_mae:.2f}%")

    return y_test_pred

def plot_improved_results(y_test, y_pred, feature_importance, feature_names):
    """绘制改进模型结果"""
    print("\n生成可视化...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 预测 vs 实际
    indices = range(len(y_test))
    axes[0, 0].plot(indices, y_test.values, label='Actual', alpha=0.7, linewidth=1)
    axes[0, 0].plot(indices, y_pred, label='Predicted', alpha=0.7, linewidth=1)
    axes[0, 0].set_title('Improved Model: Actual vs Predicted', fontweight='bold')
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('Power (W)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 散点图
    axes[0, 1].scatter(y_test, y_pred, alpha=0.5, s=10)
    axes[0, 1].plot([y_test.min(), y_test.max()],
                   [y_test.min(), y_test.max()],
                   'r--', linewidth=2)
    axes[0, 1].set_title('Prediction Scatter Plot', fontweight='bold')
    axes[0, 1].set_xlabel('Actual Power (W)')
    axes[0, 1].set_ylabel('Predicted Power (W)')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 误差分布
    errors = y_pred - y_test.values
    axes[1, 0].hist(errors, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_title('Prediction Error Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('Error (W)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Top 15 特征重要性
    top_n = 15
    top_indices = np.argsort(feature_importance)[-top_n:]
    axes[1, 1].barh(range(top_n), feature_importance[top_indices])
    axes[1, 1].set_yticks(range(top_n))
    axes[1, 1].set_yticklabels([feature_names[i] for i in top_indices], fontsize=9)
    axes[1, 1].set_title(f'Top {top_n} Feature Importances', fontweight='bold')
    axes[1, 1].set_xlabel('Importance')
    axes[1, 1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('/home/user/newPower/improved_model_results.png', dpi=150)
    print("结果已保存到: improved_model_results.png")

def main():
    print("="*60)
    print("改进的电功率预测模型训练")
    print("基于数据分析发现的规律优化")
    print("="*60)

    # 1. 加载数据
    df = load_data()

    # 2. 创建高级特征
    df = create_advanced_features(df)

    # 3. 选择特征
    df, feature_names = select_features(df)

    # 4. 准备训练数据
    X = df[feature_names]
    y = df['generationPower']

    # 5. 划分数据集 (时间序列,按顺序)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\n训练集: {len(X_train)} 样本")
    print(f"测试集: {len(X_test)} 样本")

    # 6. 训练模型
    model = train_improved_model(X_train, y_train, X_test, y_test)

    # 7. 评估模型
    y_pred = evaluate_improved_model(model, X_train, y_train, X_test, y_test)

    # 8. 可视化
    plot_improved_results(y_test, y_pred, model.feature_importances_, feature_names)

    # 9. 保存模型
    model_path = '/home/user/newPower/improved_power_model.json'
    model.save_model(model_path)
    print(f"\n改进模型已保存到: {model_path}")

    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)
    print("\n关键改进:")
    print("✓ 周期性编码 - 更好地表示时间循环")
    print("✓ 时段分类 - 基于发现的峰值/零功率时段")
    print("✓ 扩展滞后特征 - 包含前一天同一时刻")
    print("✓ 多窗口滚动统计 - 捕捉不同尺度的趋势")
    print("✓ 差分特征 - 捕捉变化率")
    print("✓ 交互特征 - 周末和小时的组合效应")
    print("✓ 优化超参数 - 更深的树、正则化")

if __name__ == "__main__":
    main()
