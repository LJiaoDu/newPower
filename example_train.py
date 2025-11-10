"""
示例代码: 使用处理好的数据训练简单的预测模型

这个脚本展示了如何使用 training_data.csv 来训练一个基本的时间序列预测模型。
我们将使用 XGBoost 作为示例,因为它对于时间序列预测效果不错且易于理解。
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 注意: 需要安装 xgboost: pip install xgboost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("警告: xgboost 未安装,将使用线性回归作为替代")
    from sklearn.linear_model import LinearRegression

def load_and_prepare_data(filepath):
    """加载并准备数据"""
    print("加载数据...")
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'])

    print(f"数据形状: {df.shape}")
    print(f"数据列: {df.columns.tolist()}")

    return df

def create_lag_features(df, n_lags=12):
    """创建滞后特征

    n_lags: 创建多少个滞后特征 (默认12个,即1小时前的数据)
    """
    print(f"\n创建 {n_lags} 个滞后特征...")

    df = df.copy()
    for i in range(1, n_lags + 1):
        df[f'power_lag_{i}'] = df['generationPower'].shift(i)

    # 创建滚动窗口特征
    df['power_rolling_mean_6'] = df['generationPower'].rolling(window=6).mean()  # 30分钟平均
    df['power_rolling_mean_12'] = df['generationPower'].rolling(window=12).mean()  # 1小时平均
    df['power_rolling_std_12'] = df['generationPower'].rolling(window=12).std()  # 1小时标准差

    # 删除包含NaN的行
    df = df.dropna()

    print(f"创建特征后的数据形状: {df.shape}")

    return df

def prepare_features(df):
    """准备特征和目标变量"""
    # 特征列
    feature_cols = [
        'year', 'month', 'day', 'hour', 'minute',
        'dayofweek', 'dayofyear', 'time_idx'
    ]

    # 添加滞后特征
    lag_features = [col for col in df.columns if 'lag' in col or 'rolling' in col]
    feature_cols.extend(lag_features)

    X = df[feature_cols]
    y = df['generationPower']

    return X, y, feature_cols

def train_model(X_train, y_train, X_test, y_test):
    """训练模型"""
    print("\n开始训练模型...")

    if XGBOOST_AVAILABLE:
        print("使用 XGBoost 模型")
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
    else:
        print("使用线性回归模型")
        model = LinearRegression()

    model.fit(X_train, y_train)
    print("模型训练完成!")

    return model

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """评估模型"""
    print("\n" + "="*60)
    print("模型评估结果")
    print("="*60)

    # 训练集预测
    y_train_pred = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)

    print("\n训练集表现:")
    print(f"  MAE:  {train_mae:.2f} W")
    print(f"  RMSE: {train_rmse:.2f} W")
    print(f"  R²:   {train_r2:.4f}")

    # 测试集预测
    y_test_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)

    print("\n测试集表现:")
    print(f"  MAE:  {test_mae:.2f} W")
    print(f"  RMSE: {test_rmse:.2f} W")
    print(f"  R²:   {test_r2:.4f}")

    # 计算相对误差
    mean_power = y_test.mean()
    relative_mae = (test_mae / mean_power) * 100
    print(f"\n相对误差 (MAE/均值): {relative_mae:.2f}%")

    return y_test_pred

def plot_predictions(y_test, y_pred, output_path):
    """绘制预测结果"""
    print("\n生成预测结果可视化...")

    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # 1. 实际值 vs 预测值时间序列图
    indices = range(len(y_test))
    axes[0].plot(indices, y_test.values, label='Actual', alpha=0.7, linewidth=1)
    axes[0].plot(indices, y_pred, label='Predicted', alpha=0.7, linewidth=1)
    axes[0].set_title('Actual vs Predicted Power', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('Generation Power (W)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. 散点图
    axes[1].scatter(y_test, y_pred, alpha=0.5, s=10)
    axes[1].plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()],
                 'r--', linewidth=2, label='Perfect Prediction')
    axes[1].set_title('Predicted vs Actual Power (Scatter)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Actual Power (W)')
    axes[1].set_ylabel('Predicted Power (W)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"预测结果已保存到: {output_path}")

def plot_feature_importance(model, feature_names, output_path):
    """绘制特征重要性"""
    if not XGBOOST_AVAILABLE:
        print("\n线性回归模型不支持特征重要性分析")
        return

    print("\n生成特征重要性可视化...")

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]  # 取前20个最重要的特征

    plt.figure(figsize=(12, 8))
    plt.title('Top 20 Feature Importances', fontsize=14, fontweight='bold')
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"特征重要性图已保存到: {output_path}")

def main():
    """主函数"""
    print("="*60)
    print("电功率预测模型训练示例")
    print("="*60)

    # 1. 加载数据
    df = load_and_prepare_data('/home/user/newPower/training_data.csv')

    # 2. 创建滞后特征
    df = create_lag_features(df, n_lags=12)

    # 3. 准备特征和目标
    X, y, feature_names = prepare_features(df)

    print(f"\n特征数量: {len(feature_names)}")
    print(f"样本数量: {len(X)}")

    # 4. 划分训练集和测试集(时间序列要按顺序划分)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\n训练集样本数: {len(X_train)}")
    print(f"测试集样本数: {len(X_test)}")

    # 5. 训练模型
    model = train_model(X_train, y_train, X_test, y_test)

    # 6. 评估模型
    y_pred = evaluate_model(model, X_train, y_train, X_test, y_test)

    # 7. 可视化结果
    plot_predictions(y_test, y_pred, '/home/user/newPower/prediction_results.png')
    plot_feature_importance(model, feature_names, '/home/user/newPower/feature_importance.png')

    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)
    print("\n生成的文件:")
    print("1. prediction_results.png - 预测结果对比图")
    if XGBOOST_AVAILABLE:
        print("2. feature_importance.png - 特征重要性图")

    # 8. 保存模型(可选)
    if XGBOOST_AVAILABLE:
        model_path = '/home/user/newPower/power_prediction_model.json'
        model.save_model(model_path)
        print(f"3. {model_path} - 训练好的模型")

    print("\n下一步建议:")
    print("- 调整模型超参数以提升性能")
    print("- 尝试其他模型(LSTM, Prophet等)")
    print("- 添加更多特征(天气数据等)")
    print("- 收集更多历史数据")

if __name__ == "__main__":
    main()
