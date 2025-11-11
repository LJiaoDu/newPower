"""
LSTM训练 - 手动循环版本（PyTorch风格）
展示如果不用model.fit()，训练循环应该怎么写
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

# ... (前面的数据加载代码相同)

def train_model_manual_loop(model, X_train, y_train, X_val, y_val, y_scaler):
    """
    手动训练循环版本 - PyTorch风格
    完全控制训练过程的每一步
    """
    # 配置
    epochs = 100
    batch_size = 128
    patience = 10  # 早停patience
    lr_patience = 5  # 学习率衰减patience

    # 优化器和损失函数
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = keras.losses.MeanSquaredError()
    mae_metric = keras.metrics.MeanAbsoluteError()

    # 计算batch数量
    num_train_samples = len(X_train)
    num_val_samples = len(X_val)
    train_batches = int(np.ceil(num_train_samples / batch_size))
    val_batches = int(np.ceil(num_val_samples / batch_size))

    # 早停和学习率衰减的变量
    best_val_loss = float('inf')
    patience_counter = 0
    lr_patience_counter = 0

    print(f"训练配置: {epochs} epochs, {batch_size} batch_size")
    print(f"训练batches: {train_batches}, 验证batches: {val_batches}")
    print()

    # ========== 开始训练 ==========
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # ========== 训练阶段 ==========
        train_loss_total = 0
        mae_metric.reset_states()

        # 打乱训练数据
        indices = np.random.permutation(num_train_samples)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        # 训练进度条
        train_pbar = tqdm(range(train_batches), desc="Training", leave=False)

        for batch_idx in train_pbar:
            # 获取当前batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_train_samples)
            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]

            # 前向传播 + 反向传播
            with tf.GradientTape() as tape:
                # 前向传播（训练模式）
                y_pred = model(X_batch, training=True)
                # 计算损失
                loss = loss_fn(y_batch, y_pred)

            # 反向传播：计算梯度
            gradients = tape.gradient(loss, model.trainable_variables)

            # 更新权重
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # 累积损失
            train_loss_total += loss.numpy()
            mae_metric.update_state(y_batch, y_pred)

            # 更新进度条
            current_loss = train_loss_total / (batch_idx + 1)
            current_mae = mae_metric.result().numpy()
            train_pbar.set_postfix({'loss': f'{current_loss:.4f}', 'mae': f'{current_mae:.4f}'})

        # 计算训练集平均指标
        avg_train_loss = train_loss_total / train_batches
        avg_train_mae = mae_metric.result().numpy()

        # ========== 验证阶段 ==========
        val_loss_total = 0
        mae_metric.reset_states()

        # 验证进度条
        val_pbar = tqdm(range(val_batches), desc="Validation", leave=False)

        for batch_idx in val_pbar:
            # 获取当前batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_val_samples)
            X_batch = X_val[start_idx:end_idx]
            y_batch = y_val[start_idx:end_idx]

            # 前向传播（验证模式，不计算梯度）
            y_pred = model(X_batch, training=False)
            loss = loss_fn(y_batch, y_pred)

            # 累积损失
            val_loss_total += loss.numpy()
            mae_metric.update_state(y_batch, y_pred)

            # 更新进度条
            current_loss = val_loss_total / (batch_idx + 1)
            current_mae = mae_metric.result().numpy()
            val_pbar.set_postfix({'val_loss': f'{current_loss:.4f}', 'val_mae': f'{current_mae:.4f}'})

        # 计算验证集平均指标
        avg_val_loss = val_loss_total / val_batches
        avg_val_mae = mae_metric.result().numpy()

        # 打印epoch结果
        print(f"  train_loss: {avg_train_loss:.4f} - train_mae: {avg_train_mae:.4f} - "
              f"val_loss: {avg_val_loss:.4f} - val_mae: {avg_val_mae:.4f}")

        # ========== 早停检查 ==========
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            lr_patience_counter = 0
            # 保存最佳权重
            best_weights = model.get_weights()
            print(f"  ✓ val_loss improved to {best_val_loss:.4f}")
        else:
            patience_counter += 1
            lr_patience_counter += 1

            # 学习率衰减
            if lr_patience_counter >= lr_patience:
                old_lr = optimizer.learning_rate.numpy()
                new_lr = old_lr * 0.5
                if new_lr >= 0.00001:  # min_lr
                    optimizer.learning_rate.assign(new_lr)
                    print(f"  ⚠ Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
                    lr_patience_counter = 0

            # 早停
            if patience_counter >= patience:
                print(f"\n早停触发！验证loss连续{patience}个epoch未改善")
                print(f"恢复最佳权重 (val_loss={best_val_loss:.4f})")
                model.set_weights(best_weights)
                break

    print("\n✓ 训练完成")
    return model


if __name__ == "__main__":
    print("="*70)
    print("这个文件展示了如果不用 model.fit() 应该怎么写训练循环")
    print("="*70)
    print()
    print("优点:")
    print("  ✓ 完全控制训练过程")
    print("  ✓ 可以自定义任何细节")
    print("  ✓ 类似PyTorch的写法")
    print()
    print("缺点:")
    print("  ✗ 代码更长（~150行 vs 1行）")
    print("  ✗ 需要手动处理很多细节")
    print("  ✗ 容易出错")
    print()
    print("我们当前使用 model.fit() 是因为:")
    print("  1. 代码更简洁")
    print("  2. Keras已经优化好了")
    print("  3. 对于标准训练流程足够用")
    print("  4. 通过callbacks可以自定义行为")
    print()
