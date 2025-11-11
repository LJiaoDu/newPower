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
from tqdm.keras import TqdmCallback


LOOKBACK_HOURS = 20     
FORECAST_HOURS = 4       
INTERVAL_MINUTES = 5     

LOOKBACK_POINTS = LOOKBACK_HOURS * 60 // INTERVAL_MINUTES 
FORECAST_POINTS = FORECAST_HOURS * 60 // INTERVAL_MINUTES  

# 设置随机种子以保证可复现性
np.random.seed(42)
tf.random.set_seed(42)

# GPU配置
print("="*60)
print("GPU配置")
print("="*60)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 设置GPU内存增长（避免占用所有显存）
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # 显示可用GPU
        print(f"✓ 检测到 {len(gpus)} 个GPU:")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")

        # 设置使用第一个GPU
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"✓ 使用GPU: {logical_gpus[0].name}")
        print("✓ GPU内存增长模式: 已启用")
    except RuntimeError as e:
        print(f"✗ GPU配置错误: {e}")
else:
    print("✗ 未检测到GPU，将使用CPU训练")
    print("  (如果你有NVIDIA显卡，请确保安装了CUDA和cuDNN)")
print("="*60)
print()

def load_data():

    df = pd.read_csv('training_data.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"总数据点: {len(df)}")
    print(f"时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
    return df

def split_raw_data(df, train_ratio=0.8):

    print("划分原始数据集...")
    split_idx = int(len(df) * train_ratio)

    df_train = df[:split_idx].copy()
    df_val = df[split_idx:].copy()

    print(f"训练集原始数据: {len(df_train)} 点 ({df_train['datetime'].min()} 到 {df_train['datetime'].max()})")
    print(f"验证集原始数据: {len(df_val)} 点 ({df_val['datetime'].min()} 到 {df_val['datetime'].max()})")

    return df_train, df_val

def create_sequences(df, dataset_name="训练"):
    power_values = df['generationPower'].values
    datetimes = df['datetime'].values

    X = [] 
    y = []  
    timestamps = []


    total_window = LOOKBACK_POINTS + FORECAST_POINTS

    for i in range(len(power_values) - total_window + 1):

        lookback_power = power_values[i:i+LOOKBACK_POINTS]

 
        forecast_power = power_values[i+LOOKBACK_POINTS:i+total_window]

        X.append(lookback_power)
        y.append(forecast_power)
        timestamps.append(datetimes[i+LOOKBACK_POINTS])

    X = np.array(X)
    y = np.array(y)

    # LSTM需要3D输入：(samples, timesteps, features)
    # 当前是(samples, timesteps)，需要reshape为(samples, timesteps, 1)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y, np.array(timestamps)

def normalize_data(X_train, y_train, X_val, y_val):

    print("归一化数据...")


    X_scaler = StandardScaler()
    X_train_scaled = X_train.reshape(-1, 1)
    X_train_scaled = X_scaler.fit_transform(X_train_scaled)
    X_train_scaled = X_train_scaled.reshape(X_train.shape)

    X_val_scaled = X_val.reshape(-1, 1)
    X_val_scaled = X_scaler.transform(X_val_scaled)
    X_val_scaled = X_val_scaled.reshape(X_val.shape)


    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_val_scaled = y_scaler.transform(y_val)


    return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_scaler, y_scaler

def build_lstm_model():


    model = Sequential([

        Bidirectional(LSTM(128, return_sequences=True),
                     input_shape=(LOOKBACK_POINTS, 1)),
        Dropout(0.2),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.2),

        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(FORECAST_POINTS)
    ])


    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    return model

def train_model(model, X_train, y_train, X_val, y_val):
    """训练LSTM模型"""
    print("\n" + "="*60)
    print("训练配置")
    print("="*60)

    # 计算batch_size以获得64个batch
    num_batches = 64
    batch_size = int(np.ceil(len(X_train) / num_batches))

    print(f"训练样本数: {len(X_train)}")
    print(f"验证样本数: {len(X_val)}")
    print(f"目标batch数: {num_batches}")
    print(f"batch_size: {batch_size}")
    print(f"实际训练batch数: {int(np.ceil(len(X_train) / batch_size))}")
    print(f"实际验证batch数: {int(np.ceil(len(X_val) / batch_size))}")
    print("="*60)

    callbacks = [
        # 早停
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # 学习率衰减
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        ),
        # tqdm进度条
        TqdmCallback(verbose=2)
    ]

    print(f"\n开始训练（预测{FORECAST_POINTS}个未来时间点）...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )

    print("\n✓ 模型训练完成")

    return model, history

def calculate_acc_(y_actual, y_pred):

    y_actual_flat = y_actual.flatten()
    y_pred_flat = y_pred.flatten()
    mask = y_actual_flat != 0
    y_actual_nonzero = y_actual_flat[mask]
    y_pred_nonzero = y_pred_flat[mask]
    if len(y_actual_nonzero) == 0:
        return 0.0  
    relative_errors = (y_actual_nonzero - y_pred_nonzero) / y_actual_nonzero
    squared_errors = relative_errors ** 2
    rmse_relative = np.sqrt(np.mean(squared_errors))
    acc_ = 1 - rmse_relative
    return acc_ * 100  

def calculate_acc_mae(y_actual, y_pred):
  
    y_actual_flat = y_actual.flatten()
    y_pred_flat = y_pred.flatten()

    mask = y_actual_flat != 0
    y_actual_nonzero = y_actual_flat[mask]
    y_pred_nonzero = y_pred_flat[mask]

    if len(y_actual_nonzero) == 0:
        return 0.0  

    relative_errors = np.abs((y_actual_nonzero - y_pred_nonzero) / y_actual_nonzero)

    mae_relative = np.mean(relative_errors)

    acc_mae = 1 - mae_relative

    return acc_mae * 100  

def evaluate_model(model, X_train, y_train, X_val, y_val, y_scaler):


    y_train_pred_scaled = model.predict(X_train, verbose=0)
    y_train_pred = y_scaler.inverse_transform(y_train_pred_scaled)
    y_train_actual = y_scaler.inverse_transform(y_train)

    train_mae = mean_absolute_error(y_train_actual, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
    train_r2 = r2_score(y_train_actual, y_train_pred)
    train_acc_ = calculate_acc_(y_train_actual, y_train_pred)
    train_acc_mae = calculate_acc_mae(y_train_actual, y_train_pred)

    print("\n训练集:")
    print(f"  MAE:  {train_mae:.2f} W")
    print(f"  RMSE: {train_rmse:.2f} W")
    print(f"  ACC_:  {train_acc_:.2f}%")
    print(f"  ACC_MAE: {train_acc_mae:.2f}%")


    y_val_pred_scaled = model.predict(X_val, verbose=0)
    y_val_pred = y_scaler.inverse_transform(y_val_pred_scaled)
    y_val_actual = y_scaler.inverse_transform(y_val)

    val_mae = mean_absolute_error(y_val_actual, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val_actual, y_val_pred))
    val_r2 = r2_score(y_val_actual, y_val_pred)
    val_acc_ = calculate_acc_(y_val_actual, y_val_pred)
    val_acc_mae = calculate_acc_mae(y_val_actual, y_val_pred)

    print("验证集:")
    print(f"  MAE:  {val_mae:.2f} W")
    print(f"  RMSE: {val_rmse:.2f} W")
    print(f"  ACC_:  {val_acc_:.2f}%")
    print(f"  ACC_MAE: {val_acc_mae:.2f}%")

    return y_val_pred, y_val_actual

def plot_results(y_val, y_pred, timestamps, history):


    fig = plt.figure(figsize=(20, 12))


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

    ax4 = fig.add_subplot(gs[1, 0])
    for i in range(0, min(len(y_val), 500), 100):
        ax4.plot(time_points, y_val[i], 'b-', alpha=0.3, linewidth=1)
        ax4.plot(time_points, y_pred[i], 'r--', alpha=0.3, linewidth=1)
    ax4.set_title('Multiple Sample Predictions', fontweight='bold')
    ax4.set_xlabel('Hours Ahead')
    ax4.set_ylabel('Power (W)')
    ax4.grid(True, alpha=0.3)


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

    plt.savefig('train24_lstm_results.png', dpi=150, bbox_inches='tight')
    print("可视化已保存: train24_lstm_results.png")

def save_model(model, X_scaler, y_scaler):


    model_path = 'power_model_24h_lstm.h5'
    model.save(model_path)
    print(f"LSTM模型已保存: {model_path}")


    scalers = {
        'X_scaler': X_scaler,
        'y_scaler': y_scaler
    }
    scaler_path = 'lstm_scalers.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scalers, f)
    print(f"归一化器已保存: {scaler_path}")

    # 保存配置
    config = {
        'lookback_hours': LOOKBACK_HOURS,
        'forecast_hours': FORECAST_HOURS,
        'lookback_points': LOOKBACK_POINTS,
        'forecast_points': FORECAST_POINTS,
        'interval_minutes': INTERVAL_MINUTES,
    }
    config_path = 'model_24h_lstm_config.pkl'
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    print(f"配置已保存: {config_path}")

def main():

    df = load_data()
    df_train, df_val = split_raw_data(df, train_ratio=0.8)
    X_train, y_train, ts_train = create_sequences(df_train, dataset_name="训练集")
    X_val, y_val, ts_val = create_sequences(df_val, dataset_name="验证集")

    print(f"最终样本统计:")
    print(f"  训练样本: {len(X_train)}")
    print(f"  验证样本: {len(X_val)}")


    X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_scaler, y_scaler = \
        normalize_data(X_train, y_train, X_val, y_val)

    model = build_lstm_model()

    model, history = train_model(model, X_train_scaled, y_train_scaled,
                                 X_val_scaled, y_val_scaled)

    y_pred, y_actual = evaluate_model(model, X_train_scaled, y_train_scaled,
                                      X_val_scaled, y_val_scaled, y_scaler)

    plot_results(y_actual, y_pred, ts_val, history)

    save_model(model, X_scaler, y_scaler)


if __name__ == "__main__":
    main()
