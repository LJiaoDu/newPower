#!/usr/bin/env python3
"""
Transformer功率预测主程序

功能：
1. 数据预处理：从JSON文件中提取特征并创建训练序列
2. 模型训练：使用Transformer模型进行功率预测
3. 模型评估：计算ACC1、ACC2等多项指标并生成可视化结果

使用方法：
python main.py --mode all                    # 运行完整流程
python main.py --mode preprocess             # 仅数据预处理
python main.py --mode train                  # 仅训练
python main.py --mode evaluate               # 仅评估
"""

import argparse
import sys
import os

from data_preprocessing import prepare_data_for_training
from train import main as train_main
from evaluate import main as evaluate_main


def check_files_exist(files):
    """检查文件是否存在"""
    missing = []
    for f in files:
        if not os.path.exists(f):
            missing.append(f)
    return missing


def preprocess_data():
    """数据预处理"""
    print("\n" + "="*70)
    print("步骤 1/3: 数据预处理")
    print("="*70)

    # 检查是否有JSON文件
    json_files = [f for f in os.listdir('.') if f.endswith('.json')]
    if not json_files:
        print("错误: 未找到JSON数据文件!")
        print("请确保当前目录包含 772_YYYY-MM-DD.json 格式的数据文件")
        sys.exit(1)

    print(f"找到 {len(json_files)} 个JSON文件")

    # 运行数据预处理
    try:
        prepare_data_for_training()
        print("\n✓ 数据预处理完成!")
    except Exception as e:
        print(f"\n✗ 数据预处理失败: {e}")
        sys.exit(1)


def train_model():
    """训练模型"""
    print("\n" + "="*70)
    print("步骤 2/3: 模型训练")
    print("="*70)

    # 检查预处理数据是否存在
    required_files = [
        'X_train.npy', 'y_train.npy',
        'X_val.npy', 'y_val.npy',
        'X_test.npy', 'y_test.npy',
        'scaler.pkl'
    ]

    missing = check_files_exist(required_files)
    if missing:
        print("错误: 缺少预处理数据文件:")
        for f in missing:
            print(f"  - {f}")
        print("\n请先运行数据预处理: python main.py --mode preprocess")
        sys.exit(1)

    # 运行训练
    try:
        train_main()
        print("\n✓ 模型训练完成!")
    except Exception as e:
        print(f"\n✗ 模型训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def evaluate_model():
    """评估模型"""
    print("\n" + "="*70)
    print("步骤 3/3: 模型评估")
    print("="*70)

    # 检查模型文件是否存在
    if not os.path.exists('checkpoints/best_model.pth'):
        print("错误: 未找到训练好的模型!")
        print("请先运行模型训练: python main.py --mode train")
        sys.exit(1)

    # 检查测试数据是否存在
    if not os.path.exists('X_test.npy') or not os.path.exists('y_test.npy'):
        print("错误: 未找到测试数据!")
        print("请先运行数据预处理: python main.py --mode preprocess")
        sys.exit(1)

    # 运行评估
    try:
        evaluate_main()
        print("\n✓ 模型评估完成!")
        print("\n结果文件:")
        print("  - results/test_metrics.json      (评估指标)")
        print("  - results/prediction_samples.png (预测样本)")
        print("  - results/error_analysis.png     (误差分析)")
        print("  - results/horizon_analysis.png   (时间范围分析)")
    except Exception as e:
        print(f"\n✗ 模型评估失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Transformer功率预测系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --mode all                 # 运行完整流程
  python main.py --mode preprocess          # 仅数据预处理
  python main.py --mode train               # 仅训练
  python main.py --mode evaluate            # 仅评估

说明:
  - 数据预处理: 从JSON文件提取特征，创建训练/验证/测试集
  - 模型训练: 使用Transformer进行功率预测训练
  - 模型评估: 计算ACC1、ACC2等指标并生成可视化结果

输出文件:
  - checkpoints/: 模型检查点和训练历史
  - results/: 评估结果和可视化图表
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['all', 'preprocess', 'train', 'evaluate'],
        default='all',
        help='运行模式 (默认: all)'
    )

    args = parser.parse_args()

    print("="*70)
    print("Transformer功率预测系统")
    print("="*70)
    print(f"模式: {args.mode}")

    if args.mode in ['all', 'preprocess']:
        preprocess_data()

    if args.mode in ['all', 'train']:
        train_model()

    if args.mode in ['all', 'evaluate']:
        evaluate_model()

    print("\n" + "="*70)
    print("所有任务完成!")
    print("="*70)


if __name__ == '__main__':
    main()
