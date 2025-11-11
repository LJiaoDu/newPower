import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from datetime import datetime

from transformer_model import create_model


class MetricsCalculator:
    """评估指标计算器"""

    @staticmethod
    def calculate_acc1(y_true, y_pred):
        """
        ACC1: 趋势准确度
        计算预测值和真实值的变化趋势是否一致

        Args:
            y_true: 真实值 [batch_size, seq_len]
            y_pred: 预测值 [batch_size, seq_len]

        Returns:
            趋势准确度 (0-1之间)
        """
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()

        # 计算变化趋势（当前值 - 前一个值）
        # 对于第一个点，与输入序列的最后一个点比较
        true_diff = np.diff(y_true, axis=1)  # [batch_size, seq_len-1]
        pred_diff = np.diff(y_pred, axis=1)  # [batch_size, seq_len-1]

        # 判断趋势是否一致（同号）
        # 上升：diff > 0, 下降：diff < 0, 不变：diff == 0
        trend_match = (true_diff * pred_diff) > 0  # 同号为True

        # 处理真实值不变的情况（diff == 0）
        # 如果真实值不变，只要预测值变化不大就算正确
        true_flat = np.abs(true_diff) < 1e-6
        pred_flat = np.abs(pred_diff) < 1e-6
        flat_match = true_flat & pred_flat

        # 综合判断
        correct = trend_match | flat_match

        acc1 = np.mean(correct)
        return acc1

    @staticmethod
    def calculate_acc2(y_true, y_pred, threshold=0.1):
        """
        ACC2: 阈值准确度
        计算预测值在真实值一定阈值范围内的比例

        Args:
            y_true: 真实值 [batch_size, seq_len]
            y_pred: 预测值 [batch_size, seq_len]
            threshold: 相对误差阈值，默认10%

        Returns:
            阈值准确度 (0-1之间)
        """
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()

        # 计算相对误差
        # 避免除以0，加一个小的常数
        epsilon = 1e-6
        relative_error = np.abs(y_true - y_pred) / (np.abs(y_true) + epsilon)

        # 判断是否在阈值范围内
        within_threshold = relative_error <= threshold

        acc2 = np.mean(within_threshold)
        return acc2

    @staticmethod
    def calculate_rmse(y_true, y_pred):
        """计算RMSE"""
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()

        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        return rmse

    @staticmethod
    def calculate_mae(y_true, y_pred):
        """计算MAE"""
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()

        mae = np.mean(np.abs(y_true - y_pred))
        return mae

    @staticmethod
    def calculate_mape(y_true, y_pred):
        """计算MAPE"""
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()

        epsilon = 1e-6
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
        return mape


class Trainer:
    """训练器"""

    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 criterion,
                 optimizer,
                 device,
                 save_dir='checkpoints'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir

        os.makedirs(save_dir, exist_ok=True)

        self.metrics_calc = MetricsCalculator()
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc1': [],
            'val_acc2': [],
            'val_rmse': [],
            'val_mae': []
        }

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(batch_x, batch_y if self.model.training else None)

            # 计算损失
            loss = self.criterion(outputs, batch_y)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # 前向传播
                outputs = self.model(batch_x)

                # 计算损失
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                num_batches += 1

                all_preds.append(outputs.cpu())
                all_targets.append(batch_y.cpu())

        # 合并所有批次的预测和目标
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # 计算评估指标
        avg_loss = total_loss / num_batches
        acc1 = self.metrics_calc.calculate_acc1(all_targets, all_preds)
        acc2 = self.metrics_calc.calculate_acc2(all_targets, all_preds, threshold=0.1)
        rmse = self.metrics_calc.calculate_rmse(all_targets, all_preds)
        mae = self.metrics_calc.calculate_mae(all_targets, all_preds)
        mape = self.metrics_calc.calculate_mape(all_targets, all_preds)

        metrics = {
            'loss': avg_loss,
            'acc1': acc1,
            'acc2': acc2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }

        return metrics

    def train(self, num_epochs, scheduler=None):
        """完整训练流程"""
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*50}")

            # 训练
            train_loss = self.train_epoch()
            print(f"训练损失: {train_loss:.4f}")

            # 验证
            val_metrics = self.validate()
            print(f"验证损失: {val_metrics['loss']:.4f}")
            print(f"验证ACC1 (趋势准确度): {val_metrics['acc1']:.4f}")
            print(f"验证ACC2 (阈值准确度): {val_metrics['acc2']:.4f}")
            print(f"验证RMSE: {val_metrics['rmse']:.2f}")
            print(f"验证MAE: {val_metrics['mae']:.2f}")
            print(f"验证MAPE: {val_metrics['mape']:.2f}%")

            # 更新学习率
            if scheduler is not None:
                scheduler.step(val_metrics['loss'])
                print(f"当前学习率: {self.optimizer.param_groups[0]['lr']:.6f}")

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc1'].append(val_metrics['acc1'])
            self.history['val_acc2'].append(val_metrics['acc2'])
            self.history['val_rmse'].append(val_metrics['rmse'])
            self.history['val_mae'].append(val_metrics['mae'])

            # 保存最佳模型
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                print(f"✓ 保存最佳模型 (验证损失: {best_val_loss:.4f})")
            else:
                patience_counter += 1

            # 定期保存检查点
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, val_metrics, is_best=False)

            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping: 验证损失已经{patience}个epoch没有改善")
                break

        print(f"\n{'='*50}")
        print("训练完成!")
        print(f"最佳验证损失: {best_val_loss:.4f}")
        print(f"{'='*50}")

        # 保存训练历史
        self.save_history()

        # 绘制训练曲线
        self.plot_history()

        return self.history

    def save_checkpoint(self, epoch, metrics, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history
        }

        if is_best:
            path = os.path.join(self.save_dir, 'best_model.pth')
        else:
            path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pth')

        torch.save(checkpoint, path)

    def save_history(self):
        """保存训练历史"""
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"训练历史已保存到: {history_path}")

    def plot_history(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 损失曲线
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # ACC1曲线
        axes[0, 1].plot(self.history['val_acc1'], label='ACC1 (Trend)', color='green')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('ACC1')
        axes[0, 1].set_title('Validation ACC1 (Trend Accuracy)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # ACC2曲线
        axes[1, 0].plot(self.history['val_acc2'], label='ACC2 (Threshold)', color='orange')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('ACC2')
        axes[1, 0].set_title('Validation ACC2 (Threshold Accuracy)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # RMSE曲线
        axes[1, 1].plot(self.history['val_rmse'], label='RMSE', color='red')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('RMSE')
        axes[1, 1].set_title('Validation RMSE')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plot_path = os.path.join(self.save_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存到: {plot_path}")
        plt.close()


def train_with_config(pos_encoding_type='sinusoidal', save_dir=None):
    """使用指定配置训练模型"""
    if save_dir is None:
        save_dir = f'checkpoints_{pos_encoding_type}'

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    print(f"位置编码类型: {pos_encoding_type.upper()}")

    # 加载数据
    print("\n=== 加载数据 ===")
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_val = np.load('X_val.npy')
    y_val = np.load('y_val.npy')

    print(f"训练集: X={X_train.shape}, y={y_train.shape}")
    print(f"验证集: X={X_val.shape}, y={y_val.shape}")

    # 转换为Tensor
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)

    # 创建DataLoader
    batch_size = 64
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型
    print("\n=== 创建模型 ===")
    input_dim = X_train.shape[-1]
    output_len = y_train.shape[-1]

    model = create_model(
        model_type='simple',
        input_dim=input_dim,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        output_len=output_len,
        pos_encoding_type=pos_encoding_type
    )

    model = model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_dir=save_dir
    )

    # 开始训练
    print("\n=== 开始训练 ===")
    history = trainer.train(num_epochs=100, scheduler=scheduler)

    print(f"\n{pos_encoding_type.upper()} 训练完成!")

    return history


def main():
    """主训练函数 - 对比两种位置编码"""
    import argparse

    parser = argparse.ArgumentParser(description='Transformer功率预测训练')
    parser.add_argument('--pos_encoding', type=str, default='both',
                       choices=['sinusoidal', 'rope', 'both'],
                       help='位置编码类型: sinusoidal, rope, 或 both (对比两种)')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')

    args = parser.parse_args()

    if args.pos_encoding == 'both':
        print("\n" + "="*70)
        print("对比训练: Sinusoidal vs RoPE 位置编码")
        print("="*70)

        # 训练标准正弦位置编码模型
        print("\n" + "="*70)
        print("训练 1/2: 标准正弦位置编码 (Sinusoidal)")
        print("="*70)
        history_sin = train_with_config(pos_encoding_type='sinusoidal')

        # 训练RoPE位置编码模型
        print("\n" + "="*70)
        print("训练 2/2: 旋转位置编码 (RoPE)")
        print("="*70)
        history_rope = train_with_config(pos_encoding_type='rope')

        # 对比结果
        print("\n" + "="*70)
        print("对比结果")
        print("="*70)
        print(f"\n最终验证指标对比:")
        print(f"{'指标':<20s} {'Sinusoidal':>15s} {'RoPE':>15s} {'提升':>15s}")
        print("-" * 70)

        # ACC1
        sin_acc1 = history_sin['val_acc1'][-1]
        rope_acc1 = history_rope['val_acc1'][-1]
        improvement_acc1 = ((rope_acc1 - sin_acc1) / sin_acc1 * 100) if sin_acc1 > 0 else 0
        print(f"{'ACC1 (趋势准确度)':<20s} {sin_acc1:>15.4f} {rope_acc1:>15.4f} {improvement_acc1:>14.2f}%")

        # ACC2
        sin_acc2 = history_sin['val_acc2'][-1]
        rope_acc2 = history_rope['val_acc2'][-1]
        improvement_acc2 = ((rope_acc2 - sin_acc2) / sin_acc2 * 100) if sin_acc2 > 0 else 0
        print(f"{'ACC2 (阈值准确度)':<20s} {sin_acc2:>15.4f} {rope_acc2:>15.4f} {improvement_acc2:>14.2f}%")

        # RMSE
        sin_rmse = history_sin['val_rmse'][-1]
        rope_rmse = history_rope['val_rmse'][-1]
        improvement_rmse = ((sin_rmse - rope_rmse) / sin_rmse * 100) if sin_rmse > 0 else 0
        print(f"{'RMSE':<20s} {sin_rmse:>15.2f} {rope_rmse:>15.2f} {improvement_rmse:>14.2f}%")

        # MAE
        sin_mae = history_sin['val_mae'][-1]
        rope_mae = history_rope['val_mae'][-1]
        improvement_mae = ((sin_mae - rope_mae) / sin_mae * 100) if sin_mae > 0 else 0
        print(f"{'MAE':<20s} {sin_mae:>15.2f} {rope_mae:>15.2f} {improvement_mae:>14.2f}%")

        print("\n" + "="*70)
        print(f"推荐使用: {'RoPE' if rope_acc1 > sin_acc1 else 'Sinusoidal'} 位置编码")
        print("="*70)

    else:
        # 单独训练
        train_with_config(pos_encoding_type=args.pos_encoding)


if __name__ == '__main__':
    main()
