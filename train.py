"""
电功率预测模型训练脚本
使用XGBoost + 43个优化特征实现高精度预测
"""

import pandas as pd
import numpy as np
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

def create_features(df):
    """创建特征"""
    print("创建特征...")
    df = df.copy()

    # 1. 周期性编码
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    # 2. 时段分类特征
    df['is_peak_hour'] = ((df['hour'] >= 4) & (df['hour'] <= 8)).astype(int)
    df['is_zero_hour'] = ((df['hour'] >= 12) & (df['hour'] <= 21)).astype(int)
    df['is_transition_hour'] = (((df['hour'] >= 21) | (df['hour'] <= 4)) &
                                 (df['is_peak_hour'] == 0)).astype(int)

    # 3. 滞后特征
    for i in [1, 2, 3, 6, 12]:
        df[f'power_lag_{i}'] = df['generationPower'].shift(i)
    df['power_lag_288'] = df['generationPower'].shift(288)

    # 4. 滚动统计特征
    windows = [6, 12, 24, 36]
    for w in windows:
        df[f'rolling_mean_{w}'] = df['generationPower'].rolling(window=w, min_periods=1).mean()
        df[f'rolling_std_{w}'] = df['generationPower'].rolling(window=w, min_periods=1).std()
        df[f'rolling_max_{w}'] = df['generationPower'].rolling(window=w, min_periods=1).max()
        df[f'rolling_min_{w}'] = df['generationPower'].rolling(window=w, min_periods=1).min()

    # 5. 差分特征
    df['power_diff_1'] = df['generationPower'].diff(1)
    df['power_diff_12'] = df['generationPower'].diff(12)
    df['power_pct_change_1'] = df['generationPower'].pct_change(1)

    # 6. 交互特征
    df['hour_dayofweek'] = df['hour'] * df['dayofweek']
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['weekend_hour'] = df['is_weekend'] * df['hour']

    # 7. 统计特征
    df['power_vs_6h'] = df['generationPower'] - df['rolling_mean_6']
    df['power_vs_12h'] = df['generationPower'] - df['rolling_mean_12']

    # 删除NaN和inf
    original_len = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    print(f"有效样本: {len(df)} (原始: {original_len})")

    return df

def get_feature_names(df):
    """获取特征名称"""
    base_features = [
        'hour', 'minute', 'dayofweek', 'dayofyear',
        'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos',
        'dayofweek_sin', 'dayofweek_cos'
    ]
    period_features = [
        'is_peak_hour', 'is_zero_hour', 'is_transition_hour',
        'is_weekend', 'weekend_hour'
    ]
    lag_features = [col for col in df.columns if 'lag' in col]
    rolling_features = [col for col in df.columns if 'rolling' in col]
    diff_features = ['power_diff_1', 'power_diff_12', 'power_pct_change_1']
    stat_features = ['power_vs_6h', 'power_vs_12h']
    interaction_features = ['hour_dayofweek']

    all_features = (base_features + period_features + lag_features +
                   rolling_features + diff_features + stat_features +
                   interaction_features)

    print(f"总特征数: {len(all_features)}")
    return all_features

def train_model(X_train, y_train, X_test, y_test):
    """训练XGBoost模型"""
    print("\n训练模型...")

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train,
             eval_set=[(X_test, y_test)],
             verbose=False)

    print("✓ 模型训练完成")
    return model

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """评估模型"""
    print("\n" + "="*60)
    print("模型评估结果")
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
    relative_error = (test_mae / mean_power) * 100
    accuracy = 100 - relative_error

    print(f"\n预测准确率: {accuracy:.2f}%")
    print(f"相对误差:   {relative_error:.2f}%")

    return y_test_pred

def plot_results(y_test, y_pred, feature_importance, feature_names):
    """绘制结果"""
    print("\n生成可视化...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 预测 vs 实际
    indices = range(len(y_test))
    axes[0, 0].plot(indices, y_test.values, label='Actual', alpha=0.7, linewidth=1)
    axes[0, 0].plot(indices, y_pred, label='Predicted', alpha=0.7, linewidth=1)
    axes[0, 0].set_title('Actual vs Predicted', fontweight='bold', fontsize=14)
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('Power (W)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 散点图
    axes[0, 1].scatter(y_test, y_pred, alpha=0.5, s=10)
    axes[0, 1].plot([y_test.min(), y_test.max()],
                   [y_test.min(), y_test.max()],
                   'r--', linewidth=2, label='Perfect Prediction')
    axes[0, 1].set_title('Prediction Scatter Plot', fontweight='bold', fontsize=14)
    axes[0, 1].set_xlabel('Actual Power (W)')
    axes[0, 1].set_ylabel('Predicted Power (W)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 误差分布
    errors = y_pred - y_test.values
    axes[1, 0].hist(errors, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_title('Prediction Error Distribution', fontweight='bold', fontsize=14)
    axes[1, 0].set_xlabel('Error (W)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].text(0.05, 0.95, f'Mean Error: {errors.mean():.2f} W\nStd Error: {errors.std():.2f} W',
                   transform=axes[1, 0].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Top 15 特征重要性
    top_n = 15
    top_indices = np.argsort(feature_importance)[-top_n:]
    axes[1, 1].barh(range(top_n), feature_importance[top_indices])
    axes[1, 1].set_yticks(range(top_n))
    axes[1, 1].set_yticklabels([feature_names[i] for i in top_indices], fontsize=9)
    axes[1, 1].set_title(f'Top {top_n} Feature Importances', fontweight='bold', fontsize=14)
    axes[1, 1].set_xlabel('Importance')
    axes[1, 1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('/home/user/newPower/model_results.png', dpi=150)
    print("✓ 可视化已保存: model_results.png")

def main():
    print("="*60)
    print("电功率预测模型训练")
    print("="*60)

    # 1. 加载数据
    df = load_data()

    # 2. 创建特征
    df = create_features(df)

    # 3. 选择特征
    feature_names = get_feature_names(df)
    X = df[feature_names]
    y = df['generationPower']

    # 4. 划分数据集
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\n训练集: {len(X_train)} 样本")
    print(f"测试集: {len(X_test)} 样本")

    # 5. 训练模型
    model = train_model(X_train, y_train, X_test, y_test)

    # 6. 评估模型
    y_pred = evaluate_model(model, X_train, y_train, X_test, y_test)

    # 7. 可视化
    plot_results(y_test, y_pred, model.feature_importances_, feature_names)

    # 8. 保存模型
    model_path = '/home/user/newPower/power_model.json'
    model.save_model(model_path)
    print(f"\n✓ 模型已保存: {model_path}")

    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)

if __name__ == "__main__":
    main()
