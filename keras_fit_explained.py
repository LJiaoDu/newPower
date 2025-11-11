"""
Keras model.fit() 内部实现原理解析
展示 model.fit() 背后实际做了什么
"""

import tensorflow as tf
import numpy as np

# 假设的数据
X_train = np.random.randn(1000, 10)
y_train = np.random.randn(1000, 1)
X_val = np.random.randn(200, 10)
y_val = np.random.randn(200, 1)

# 创建模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("="*70)
print("方法1: 使用 model.fit() - Keras高层API")
print("="*70)
print("""
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    verbose=1
)
""")
print("✓ 一行代码，Keras自动处理所有细节！")
print()

print("="*70)
print("方法2: 手动训练循环 - 相当于 model.fit() 内部实现")
print("="*70)
print()

# 创建另一个相同的模型用于对比
model_manual = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# 定义优化器和损失函数
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.MeanSquaredError()
mae_metric = tf.keras.metrics.MeanAbsoluteError()

# 参数
epochs = 3
batch_size = 128
num_train_samples = len(X_train)
num_val_samples = len(X_val)

# 计算batch数量
train_batches = int(np.ceil(num_train_samples / batch_size))
val_batches = int(np.ceil(num_val_samples / batch_size))

print("手动训练循环代码：")
print("""
# model.fit() 背后实际执行的代码
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    # ========== 训练阶段 ==========
    train_loss = 0
    train_mae = 0

    # 1. 打乱训练数据
    indices = np.random.permutation(num_train_samples)
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]

    # 2. 分批次训练
    for batch_idx in range(train_batches):
        # 获取当前batch的数据
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_train_samples)
        X_batch = X_train_shuffled[start_idx:end_idx]
        y_batch = y_train_shuffled[start_idx:end_idx]

        # 3. 前向传播 + 反向传播 + 权重更新
        with tf.GradientTape() as tape:
            # 前向传播
            y_pred = model_manual(X_batch, training=True)
            # 计算损失
            loss = loss_fn(y_batch, y_pred)

        # 反向传播：计算梯度
        gradients = tape.gradient(loss, model_manual.trainable_variables)

        # 更新权重
        optimizer.apply_gradients(zip(gradients, model_manual.trainable_variables))

        # 累积指标
        train_loss += loss.numpy()
        mae_metric.update_state(y_batch, y_pred)

    # 计算平均训练损失
    avg_train_loss = train_loss / train_batches
    avg_train_mae = mae_metric.result().numpy()
    mae_metric.reset_states()

    # ========== 验证阶段 ==========
    val_loss = 0
    val_mae = 0

    # 验证时不打乱数据
    for batch_idx in range(val_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_val_samples)
        X_batch = X_val[start_idx:end_idx]
        y_batch = y_val[start_idx:end_idx]

        # 前向传播（不计算梯度）
        y_pred = model_manual(X_batch, training=False)
        loss = loss_fn(y_batch, y_pred)

        # 累积指标
        val_loss += loss.numpy()
        mae_metric.update_state(y_batch, y_pred)

    # 计算平均验证损失
    avg_val_loss = val_loss / val_batches
    avg_val_mae = mae_metric.result().numpy()
    mae_metric.reset_states()

    # 打印结果
    print(f"  loss: {avg_train_loss:.4f} - mae: {avg_train_mae:.4f} - "
          f"val_loss: {avg_val_loss:.4f} - val_mae: {avg_val_mae:.4f}")
""")

print("\n" + "="*70)
print("总结")
print("="*70)
print("""
model.fit() 自动做了这些事：
✓ 1. 外层epoch循环 (for epoch in range(epochs))
✓ 2. 数据打乱 (shuffle)
✓ 3. 分批次处理 (batch循环)
✓ 4. 前向传播 (forward pass)
✓ 5. 计算损失 (loss calculation)
✓ 6. 反向传播 (backward pass)
✓ 7. 权重更新 (optimizer.apply_gradients)
✓ 8. 验证评估 (validation)
✓ 9. 指标计算 (metrics)
✓ 10. 进度条显示 (progress bar)
✓ 11. 回调函数处理 (callbacks)

所以我们不需要手动写这些循环！
""")

print("="*70)
print("PyTorch vs Keras/TensorFlow")
print("="*70)
print("""
PyTorch:
  - 低层API为主
  - 需要手动写训练循环
  - 更灵活，但代码更长
  - 适合研究、自定义算法

Keras/TensorFlow:
  - 高层API为主
  - model.fit() 自动处理一切
  - 代码简洁，但灵活性稍低
  - 适合快速开发、生产部署

都可以完成相同的任务！
""")
