"""
LSTM神经网络：20小时历史数据预测未来4小时功率
输入：过去20小时的功率数据（240个点，5分钟间隔）
输出：未来4小时的功率预测（48个点，5分钟间隔）
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
import pickle
import os

# 配置参数
LOOKBACK_HOURS = 20      # 输入：过去20小时
FORECAST_HOURS = 4       # 输出：未来4小时
INTERVAL_MINUTES = 5     # 数据间隔：5分钟

LOOKBACK_POINTS = LOOKBACK_HOURS * 60 // INTERVAL_MINUTES  # 20h = 240个点
FORECAST_POINTS = FORECAST_HOURS * 60 // INTERVAL_MINUTES  # 4h = 48个点

# 设置随机种子以保证可复现性
np.random.seed(42)
tf.random.set_seed(42)

def load_data():
    """加载数据"""
    print("="*60)
    print("加载数据...")
    print("="*60)
    df = pd.read_csv('/home/user/newPower/training_data.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"总数据点: {len(df)}")
    print(f"时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
    return df

def split_raw_data(df, train_ratio=0.8):
    """先将原始数据按时间顺序划分为训练集和验证集"""
    print("\n划分原始数据集...")
    split_idx = int(len(df) * train_ratio)

    df_train = df[:split_idx].copy()
    df_val = df[split_idx:].copy()

    print(f"训练集原始数据: {len(df_train)} 点 ({df_train['datetime'].min()} 到 {df_train['datetime'].max()})")
    print(f"验证集原始数据: {len(df_val)} 点 ({df_val['datetime'].min()} 到 {df_val['datetime'].max()})")

    return df_train, df_val

def create_sequences(df, dataset_name="训练"):
    """创建LSTM序列数据

    输入：过去20小时（240个点）
    输出：未来4小时（48个点）
    """
    print(f"\n创建{dataset_name}序列...")
    print(f"输入窗口: {LOOKBACK_HOURS}小时 ({LOOKBACK_POINTS}个点)")
    print(f"输出窗口: {FORECAST_HOURS}小时 ({FORECAST_POINTS}个点)")

    power_values = df['generationPower'].values
    datetimes = df['datetime'].values

    X = []  # 输入序列
    y = []  # 输出序列
    timestamps = []

    # 滑动窗口创建样本
    total_window = LOOKBACK_POINTS + FORECAST_POINTS

    for i in range(len(power_values) - total_window + 1):
        # 输入：过去20小时的功率序列
        lookback_power = power_values[i:i+LOOKBACK_POINTS]

        # 输出：未来4小时的功率序列
        forecast_power = power_values[i+LOOKBACK_POINTS:i+total_window]

        X.append(lookback_power)
        y.append(forecast_power)
        timestamps.append(datetimes[i+LOOKBACK_POINTS])

    X = np.array(X)
    y = np.array(y)

    # LSTM需要3D输入：(samples, timesteps, features)
    # 当前是(samples, timesteps)，需要reshape为(samples, timesteps, 1)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    print(f"生成样本数: {len(X)}")
    print(f"输入形状: {X.shape} (样本数, 时间步, 特征数)")
    print(f"输出形状: {y.shape} (样本数, 预测点数)")

    return X, y, np.array(timestamps)

def normalize_data(X_train, y_train, X_val, y_val):
    """归一化数据（LSTM训练必需）"""
    print("\n归一化数据...")

    # 输入数据归一化
    X_scaler = StandardScaler()
    X_train_scaled = X_train.reshape(-1, 1)
    X_train_scaled = X_scaler.fit_transform(X_train_scaled)
    X_train_scaled = X_train_scaled.reshape(X_train.shape)

    X_val_scaled = X_val.reshape(-1, 1)
    X_val_scaled = X_scaler.transform(X_val_scaled)
    X_val_scaled = X_val_scaled.reshape(X_val.shape)

    # 输出数据归一化
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_val_scaled = y_scaler.transform(y_val)

    print("✓ 数据归一化完成")

    return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_scaler, y_scaler

def build_lstm_model():
    """构建LSTM模型"""
    print("\n构建LSTM模型...")

    model = Sequential([
        # 第一层：双向LSTM，128个单元
        Bidirectional(LSTM(128, return_sequences=True),
                     input_shape=(LOOKBACK_POINTS, 1)),
        Dropout(0.2),

        # 第二层：双向LSTM，64个单元
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.2),

        # 全连接层
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),

        # 输出层
        Dense(FORECAST_POINTS)
    ])

    # 编译模型
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    print("\n模型结构:")
    model.summary()

    return model

def train_model(model, X_train, y_train, X_val, y_val):
    """训练LSTM模型"""
    print("\n" + "="*60)
    print("训练模型...")
    print("="*60)

    # 回调函数
    callbacks = [
        # 早停：验证集loss连续10个epoch不下降则停止
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # 学习率衰减：验证集loss 5个epoch不下降则降低学习率
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
    ]

    # 训练
    print(f"\n开始训练（预测{FORECAST_POINTS}个未来时间点）...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    print("✓ 模型训练完成")

    return model, history

def evaluate_model(model, X_train, y_train, X_val, y_val, y_scaler):
    """评估模型"""
    print("\n" + "="*60)
    print("模型评估")
    print("="*60)

    # 训练集预测
    y_train_pred_scaled = model.predict(X_train, verbose=0)
    y_train_pred = y_scaler.inverse_transform(y_train_pred_scaled)
    y_train_actual = y_scaler.inverse_transform(y_train)

    train_mae = mean_absolute_error(y_train_actual, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
    train_r2 = r2_score(y_train_actual, y_train_pred)

    print("\n训练集:")
    print(f"  MAE:  {train_mae:.2f} W")
    print(f"  RMSE: {train_rmse:.2f} W")
    print(f"  R²:   {train_r2:.4f}")

    mean_power_train = y_train_actual.mean()
    train_relative_error = (train_mae / mean_power_train) * 100
    train_accuracy = 100 - train_relative_error
    print(f"  准确率: {train_accuracy:.2f}%")

    # 验证集预测
    y_val_pred_scaled = model.predict(X_val, verbose=0)
    y_val_pred = y_scaler.inverse_transform(y_val_pred_scaled)
    y_val_actual = y_scaler.inverse_transform(y_val)

    val_mae = mean_absolute_error(y_val_actual, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val_actual, y_val_pred))
    val_r2 = r2_score(y_val_actual, y_val_pred)

    print("\n验证集:")
    print(f"  MAE:  {val_mae:.2f} W")
    print(f"  RMSE: {val_rmse:.2f} W")
    print(f"  R²:   {val_r2:.4f}")

    mean_power_val = y_val_actual.mean()
    val_relative_error = (val_mae / mean_power_val) * 100
    val_accuracy = 100 - val_relative_error
    print(f"  准确率: {val_accuracy:.2f}%")

    # 计算每个时间步的误差（验证集）
    print("\n验证集各时间步误差分析:")
    timestep_mae = np.mean(np.abs(y_val_actual - y_val_pred), axis=0)

    for hour in [1, 2, 3, 4]:
        idx = hour * 12 - 1  # 每小时的最后一个点
        print(f"  未来第{hour}小时末: MAE = {timestep_mae[idx]:.2f} W")

    print("\n" + "="*60)
    print(f"最终验证集准确率: {val_accuracy:.2f}%")
    print(f"最终验证集误差率: {val_relative_error:.2f}%")
    print("="*60)

    return y_val_pred, y_val_actual

def plot_results(y_val, y_pred, timestamps, history):
    """可视化预测结果"""
    print("\n生成可视化...")

    fig = plt.figure(figsize=(20, 12))

    # 创建2x3的子图布局
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. 训练历史（损失）
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss During Training', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 训练历史（MAE）
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_title('Model MAE During Training', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE (W)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 单个样本预测展示
    ax3 = fig.add_subplot(gs[0, 2])
    sample_idx = len(y_val) // 2
    time_points = np.arange(FORECAST_POINTS) * INTERVAL_MINUTES / 60
    ax3.plot(time_points, y_val[sample_idx], 'b-o', label='Actual', markersize=3, linewidth=1.5)
    ax3.plot(time_points, y_pred[sample_idx], 'r--s', label='Predicted', markersize=3, linewidth=1.5)
    ax3.set_title(f'Sample Prediction (Starting at {timestamps[sample_idx]})',
                  fontweight='bold', fontsize=10)
    ax3.set_xlabel('Hours Ahead')
    ax3.set_ylabel('Power (W)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 多个样本预测对比
    ax4 = fig.add_subplot(gs[1, 0])
    for i in range(0, min(len(y_val), 500), 100):
        ax4.plot(time_points, y_val[i], 'b-', alpha=0.3, linewidth=1)
        ax4.plot(time_points, y_pred[i], 'r--', alpha=0.3, linewidth=1)
    ax4.set_title('Multiple Sample Predictions', fontweight='bold')
    ax4.set_xlabel('Hours Ahead')
    ax4.set_ylabel('Power (W)')
    ax4.grid(True, alpha=0.3)

    # 5. 每个时间步的MAE
    ax5 = fig.add_subplot(gs[1, 1])
    timestep_mae = np.mean(np.abs(y_val - y_pred), axis=0)
    ax5.plot(time_points, timestep_mae, 'g-o', markersize=4, linewidth=2)
    ax5.set_title('MAE by Forecast Horizon', fontweight='bold')
    ax5.set_xlabel('Hours Ahead')
    ax5.set_ylabel('MAE (W)')
    ax5.grid(True, alpha=0.3)
    for hour in [1, 2, 3, 4]:
        ax5.axvline(x=hour, color='gray', linestyle='--', alpha=0.5)
        ax5.text(hour, max(timestep_mae)*0.9, f'{hour}h', ha='center')

    # 6. 误差分布
    ax6 = fig.add_subplot(gs[1, 2])
    errors = (y_pred - y_val).flatten()
    ax6.hist(errors, bins=100, alpha=0.7, edgecolor='black')
    ax6.axvline(0, color='r', linestyle='--', linewidth=2)
    ax6.set_title('Prediction Error Distribution', fontweight='bold')
    ax6.set_xlabel('Error (W)')
    ax6.set_ylabel('Frequency')
    ax6.text(0.05, 0.95, f'Mean: {errors.mean():.2f} W\nStd: {errors.std():.2f} W',
            transform=ax6.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax6.grid(True, alpha=0.3)

    plt.savefig('/home/user/newPower/train24_lstm_results.png', dpi=150, bbox_inches='tight')
    print("✓ 可视化已保存: train24_lstm_results.png")

def save_model(model, X_scaler, y_scaler):
    """保存模型和归一化器"""
    # 保存Keras模型
    model_path = '/home/user/newPower/power_model_24h_lstm.h5'
    model.save(model_path)
    print(f"\n✓ LSTM模型已保存: {model_path}")

    # 保存归一化器
    scalers = {
        'X_scaler': X_scaler,
        'y_scaler': y_scaler
    }
    scaler_path = '/home/user/newPower/lstm_scalers.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scalers, f)
    print(f"✓ 归一化器已保存: {scaler_path}")

    # 保存配置
    config = {
        'lookback_hours': LOOKBACK_HOURS,
        'forecast_hours': FORECAST_HOURS,
        'lookback_points': LOOKBACK_POINTS,
        'forecast_points': FORECAST_POINTS,
        'interval_minutes': INTERVAL_MINUTES,
    }
    config_path = '/home/user/newPower/model_24h_lstm_config.pkl'
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    print(f"✓ 配置已保存: {config_path}")

def main():
    print("="*60)
    print("LSTM神经网络：20小时历史 → 4小时功率预测")
    print("使用 training_data.csv，按 80/20 划分训练/验证集")
    print("="*60)

    # 1. 加载数据
    df = load_data()

    # 2. 划分原始数据（80% 训练，20% 验证）
    df_train, df_val = split_raw_data(df, train_ratio=0.8)

    # 3. 创建序列
    X_train, y_train, ts_train = create_sequences(df_train, dataset_name="训练集")
    X_val, y_val, ts_val = create_sequences(df_val, dataset_name="验证集")

    print("\n" + "="*60)
    print(f"最终样本统计:")
    print(f"  训练样本: {len(X_train)}")
    print(f"  验证样本: {len(X_val)}")
    print("="*60)

    # 4. 归一化数据
    X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_scaler, y_scaler = \
        normalize_data(X_train, y_train, X_val, y_val)

    # 5. 构建模型
    model = build_lstm_model()

    # 6. 训练模型
    model, history = train_model(model, X_train_scaled, y_train_scaled,
                                 X_val_scaled, y_val_scaled)

    # 7. 评估模型
    y_pred, y_actual = evaluate_model(model, X_train_scaled, y_train_scaled,
                                      X_val_scaled, y_val_scaled, y_scaler)

    # 8. 可视化
    plot_results(y_actual, y_pred, ts_val, history)

    # 9. 保存模型
    save_model(model, X_scaler, y_scaler)

    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)
    print("\n使用说明:")
    print("1. 输入：过去20小时的功率数据（240个点，5分钟间隔）")
    print("2. 输出：未来4小时的功率预测（48个点，5分钟间隔）")
    print("3. 加载模型：")
    print("   from tensorflow import keras")
    print("   model = keras.models.load_model('power_model_24h_lstm.h5')")
    print("4. 加载归一化器：")
    print("   import pickle")
    print("   with open('lstm_scalers.pkl', 'rb') as f:")
    print("       scalers = pickle.load(f)")
    print("5. 预测（需要先归一化输入）：")
    print("   X_new_scaled = scalers['X_scaler'].transform(X_new)")
    print("   y_pred_scaled = model.predict(X_new_scaled)")
    print("   y_pred = scalers['y_scaler'].inverse_transform(y_pred_scaled)")

if __name__ == "__main__":
    main()
