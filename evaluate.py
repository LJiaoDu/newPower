import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import os
import json

from transformer_model import create_model
from train import MetricsCalculator


class Evaluator:
    """模型评估器"""

    def __init__(self, model, device, save_dir='results'):
        self.model = model
        self.device = device
        self.save_dir = save_dir
        self.metrics_calc = MetricsCalculator()

        os.makedirs(save_dir, exist_ok=True)

    def evaluate(self, test_loader):
        """评估模型"""
        self.model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # 预测
                outputs = self.model(batch_x)

                all_preds.append(outputs.cpu())
                all_targets.append(batch_y.cpu())

        # 合并所有批次
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()

        # 计算所有指标
        metrics = {
            'ACC1 (趋势准确度)': self.metrics_calc.calculate_acc1(all_targets, all_preds),
            'ACC2 (阈值准确度 10%)': self.metrics_calc.calculate_acc2(all_targets, all_preds, threshold=0.1),
            'ACC2 (阈值准确度 15%)': self.metrics_calc.calculate_acc2(all_targets, all_preds, threshold=0.15),
            'ACC2 (阈值准确度 20%)': self.metrics_calc.calculate_acc2(all_targets, all_preds, threshold=0.2),
            'RMSE': self.metrics_calc.calculate_rmse(all_targets, all_preds),
            'MAE': self.metrics_calc.calculate_mae(all_targets, all_preds),
            'MAPE (%)': self.metrics_calc.calculate_mape(all_targets, all_preds)
        }

        return metrics, all_preds, all_targets

    def print_metrics(self, metrics):
        """打印评估指标"""
        print("\n" + "="*60)
        print("测试集评估结果")
        print("="*60)
        for name, value in metrics.items():
            if 'ACC' in name:
                print(f"{name:30s}: {value:.4f} ({value*100:.2f}%)")
            elif '%' in name:
                print(f"{name:30s}: {value:.2f}%")
            else:
                print(f"{name:30s}: {value:.2f}")
        print("="*60)

    def save_metrics(self, metrics, filename='test_metrics.json'):
        """保存评估指标"""
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"\n评估指标已保存到: {filepath}")

    def plot_predictions(self, y_true, y_pred, num_samples=5):
        """绘制预测结果"""
        num_samples = min(num_samples, len(y_true))

        fig, axes = plt.subplots(num_samples, 1, figsize=(15, 3*num_samples))
        if num_samples == 1:
            axes = [axes]

        for i in range(num_samples):
            ax = axes[i]

            # 时间步
            time_steps = np.arange(len(y_true[i]))

            # 绘制真实值和预测值
            ax.plot(time_steps, y_true[i], label='真实值', linewidth=2, marker='o', markersize=4)
            ax.plot(time_steps, y_pred[i], label='预测值', linewidth=2, marker='s', markersize=4)

            # 计算该样本的指标
            acc1 = self.metrics_calc.calculate_acc1(y_true[i:i+1], y_pred[i:i+1])
            acc2 = self.metrics_calc.calculate_acc2(y_true[i:i+1], y_pred[i:i+1], threshold=0.1)
            mae = self.metrics_calc.calculate_mae(y_true[i:i+1], y_pred[i:i+1])

            ax.set_title(f'样本 {i+1} - ACC1: {acc1:.3f}, ACC2: {acc2:.3f}, MAE: {mae:.2f}')
            ax.set_xlabel('时间步 (每步5分钟)')
            ax.set_ylabel('功率 (W)')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(self.save_dir, 'prediction_samples.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"预测样本图已保存到: {filepath}")
        plt.close()

    def plot_error_distribution(self, y_true, y_pred):
        """绘制误差分布"""
        errors = y_true - y_pred
        relative_errors = np.abs(errors) / (np.abs(y_true) + 1e-6)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 绝对误差分布
        axes[0, 0].hist(errors.flatten(), bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('误差 (W)')
        axes[0, 0].set_ylabel('频数')
        axes[0, 0].set_title('绝对误差分布')
        axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].grid(True, alpha=0.3)

        # 相对误差分布
        axes[0, 1].hist(relative_errors.flatten(), bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[0, 1].set_xlabel('相对误差')
        axes[0, 1].set_ylabel('频数')
        axes[0, 1].set_title('相对误差分布')
        axes[0, 1].axvline(x=0.1, color='red', linestyle='--', linewidth=2, label='10%阈值')
        axes[0, 1].axvline(x=0.2, color='green', linestyle='--', linewidth=2, label='20%阈值')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 真实值 vs 预测值散点图
        sample_indices = np.random.choice(y_true.size, size=min(5000, y_true.size), replace=False)
        axes[1, 0].scatter(y_true.flatten()[sample_indices],
                          y_pred.flatten()[sample_indices],
                          alpha=0.5, s=1)
        # 添加理想线
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想预测')
        axes[1, 0].set_xlabel('真实值 (W)')
        axes[1, 0].set_ylabel('预测值 (W)')
        axes[1, 0].set_title('真实值 vs 预测值')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 按时间步的平均误差
        mae_per_step = np.mean(np.abs(errors), axis=0)
        time_steps = np.arange(len(mae_per_step))
        axes[1, 1].plot(time_steps, mae_per_step, linewidth=2, marker='o')
        axes[1, 1].set_xlabel('预测时间步 (每步5分钟)')
        axes[1, 1].set_ylabel('平均绝对误差 (W)')
        axes[1, 1].set_title('不同预测时间步的MAE')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(self.save_dir, 'error_analysis.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"误差分析图已保存到: {filepath}")
        plt.close()

    def analyze_by_horizon(self, y_true, y_pred):
        """按预测时间范围分析性能"""
        print("\n" + "="*60)
        print("按预测时间范围的性能分析")
        print("="*60)

        horizons = [
            (0, 12, "0-1小时"),
            (12, 24, "1-2小时"),
            (24, 36, "2-3小时"),
            (36, 48, "3-4小时")
        ]

        results = []
        for start, end, name in horizons:
            y_true_slice = y_true[:, start:end]
            y_pred_slice = y_pred[:, start:end]

            acc1 = self.metrics_calc.calculate_acc1(y_true_slice, y_pred_slice)
            acc2 = self.metrics_calc.calculate_acc2(y_true_slice, y_pred_slice, threshold=0.1)
            rmse = self.metrics_calc.calculate_rmse(y_true_slice, y_pred_slice)
            mae = self.metrics_calc.calculate_mae(y_true_slice, y_pred_slice)

            print(f"\n{name}:")
            print(f"  ACC1: {acc1:.4f} ({acc1*100:.2f}%)")
            print(f"  ACC2: {acc2:.4f} ({acc2*100:.2f}%)")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  MAE:  {mae:.2f}")

            results.append({
                'horizon': name,
                'ACC1': acc1,
                'ACC2': acc2,
                'RMSE': rmse,
                'MAE': mae
            })

        print("="*60)

        # 绘制图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        horizon_names = [r['horizon'] for r in results]
        acc1_values = [r['ACC1'] for r in results]
        acc2_values = [r['ACC2'] for r in results]
        rmse_values = [r['RMSE'] for r in results]
        mae_values = [r['MAE'] for r in results]

        axes[0, 0].bar(horizon_names, acc1_values, color='skyblue', edgecolor='black')
        axes[0, 0].set_ylabel('ACC1')
        axes[0, 0].set_title('ACC1 vs 预测时间范围')
        axes[0, 0].grid(True, alpha=0.3, axis='y')

        axes[0, 1].bar(horizon_names, acc2_values, color='lightgreen', edgecolor='black')
        axes[0, 1].set_ylabel('ACC2')
        axes[0, 1].set_title('ACC2 vs 预测时间范围')
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        axes[1, 0].bar(horizon_names, rmse_values, color='salmon', edgecolor='black')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].set_title('RMSE vs 预测时间范围')
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        axes[1, 1].bar(horizon_names, mae_values, color='gold', edgecolor='black')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].set_title('MAE vs 预测时间范围')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        filepath = os.path.join(self.save_dir, 'horizon_analysis.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"\n时间范围分析图已保存到: {filepath}")
        plt.close()

        return results


def main():
    """主评估函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载测试数据
    print("\n=== 加载测试数据 ===")
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')

    print(f"测试集: X={X_test.shape}, y={y_test.shape}")

    # 转换为Tensor
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    # 创建DataLoader
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 加载模型
    print("\n=== 加载模型 ===")
    input_dim = X_test.shape[-1]
    output_len = y_test.shape[-1]

    model = create_model(
        model_type='simple',
        input_dim=input_dim,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        output_len=output_len
    )

    # 加载最佳模型权重
    checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print("✓ 已加载最佳模型")

    # 创建评估器
    evaluator = Evaluator(model, device, save_dir='results')

    # 评估模型
    print("\n=== 评估模型 ===")
    metrics, y_pred, y_true = evaluator.evaluate(test_loader)

    # 打印指标
    evaluator.print_metrics(metrics)

    # 保存指标
    evaluator.save_metrics(metrics)

    # 绘制预测结果
    print("\n=== 生成可视化结果 ===")
    evaluator.plot_predictions(y_true, y_pred, num_samples=5)
    evaluator.plot_error_distribution(y_true, y_pred)

    # 按时间范围分析
    evaluator.analyze_by_horizon(y_true, y_pred)

    print("\n评估完成!")


if __name__ == '__main__':
    main()
