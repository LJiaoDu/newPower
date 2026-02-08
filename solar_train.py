#!/usr/bin/env python3
"""
太阳能电站功率预测 - 训练与评估脚本

使用方式:
  python solar_train.py                     # 完整流程: 预处理 + 训练 + 评估
  python solar_train.py --mode preprocess   # 仅预处理
  python solar_train.py --mode train        # 仅训练
  python solar_train.py --mode evaluate     # 仅评估
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import argparse
import pickle
import time

from solar_model import SolarTransformer
from solar_preprocess import main as preprocess_main


# ============== 评估指标 ==============

def calc_acc_mae(y_true, y_pred, cap=1.0):
    """ACC (MAE-based): 1 - MAE / mean(y_true)"""
    mask = y_true > 0.01  # 过滤夜间零值
    if mask.sum() == 0:
        return float("nan")
    y_t = y_true[mask] * cap
    y_p = y_pred[mask] * cap
    mae = np.mean(np.abs(y_t - y_p))
    avg = np.mean(y_t)
    acc = max(0.0, 1.0 - mae / (avg + 1e-6))
    return acc


def calc_rmse(y_true, y_pred, cap=1.0):
    return np.sqrt(np.mean((y_true * cap - y_pred * cap) ** 2))


def calc_mae(y_true, y_pred, cap=1.0):
    return np.mean(np.abs(y_true * cap - y_pred * cap))


def calc_acc2(y_true, y_pred, cap=1.0):
    """ACC2 (国标): 1 - sqrt( (1/N) * Σ ((P_M - P_P) / max(P_M, 0.2*Cap))^2 )"""
    p_m = y_true.flatten() * cap
    p_p = y_pred.flatten() * cap
    denom = np.maximum(p_m, 0.2 * cap)
    acc2 = 1.0 - np.sqrt(np.mean(((p_m - p_p) / denom) ** 2))
    return max(0.0, acc2)


# ============== 训练 ==============

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载数据
    X_train = np.load("X_train.npy")
    y_train = np.load("y_train.npy")
    X_val = np.load("X_val.npy")
    y_val = np.load("y_val.npy")

    with open("norm_params.pkl", "rb") as f:
        norm_params = pickle.load(f)
    cap = norm_params["power"]["cap"]

    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}")
    print(f"标称容量: {cap} MW")

    in_seq_len = X_train.shape[1]
    in_feat_size = X_train.shape[2]
    out_seq_len = y_train.shape[1]

    # DataLoader
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=args.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        batch_size=args.batch_size, shuffle=False,
    )

    # 模型
    model = SolarTransformer(
        in_feat_size=in_feat_size,
        in_seq_len=in_seq_len,
        out_seq_len=out_seq_len,
        hidden_size=args.hidden_size,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_cross_attn_layers=args.num_cross_attn_layers,
        dropout=args.dropout,
    ).to(device)

    # 初始化权重
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p, gain=0.5)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 优化器 + 调度器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[args.warmup_epochs])

    criterion = nn.MSELoss()

    os.makedirs("solar_checkpoints", exist_ok=True)

    # TensorBoard
    log_dir = os.path.join("runs", time.strftime("%Y%m%d_%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard 日志: {log_dir}")
    print(f"  启动命令: tensorboard --logdir=runs")

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"开始训练 | Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        # --- 训练 ---
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # --- 验证 ---
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                val_loss += criterion(pred, batch_y).item()
                all_preds.append(pred.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
        val_loss /= len(val_loader)

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        acc = calc_acc_mae(all_targets, all_preds, cap)
        acc2 = calc_acc2(all_targets, all_preds, cap)
        rmse = calc_rmse(all_targets, all_preds, cap)
        mae = calc_mae(all_targets, all_preds, cap)
        lr = optimizer.param_groups[0]["lr"]

        # TensorBoard 记录所有指标
        writer.add_scalars("Loss", {"Train": train_loss, "Val": val_loss}, epoch + 1)
        writer.add_scalar("Accuracy/ACC1", acc, epoch + 1)
        writer.add_scalar("Accuracy/ACC2", acc2, epoch + 1)
        writer.add_scalar("Error/RMSE_MW", rmse, epoch + 1)
        writer.add_scalar("Error/MAE_MW", mae, epoch + 1)
        writer.add_scalar("LearningRate", lr, epoch + 1)

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"LR: {lr:.6f} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f} | "
            f"ACC1: {acc:.4f} | "
            f"ACC2: {acc2:.4f} | "
            f"RMSE: {rmse:.2f} MW | "
            f"MAE: {mae:.2f} MW"
        )

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "acc": acc,
                "norm_params": norm_params,
            }, "solar_checkpoints/best_model.pth")
            print(f"  -> 保存最佳模型 (Val Loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping: {args.patience} epochs 无改善")
                break

        scheduler.step()

    writer.close()
    print(f"\n训练完成! 最佳验证损失: {best_val_loss:.6f}")
    print(f"TensorBoard 查看: tensorboard --logdir=runs")


# ============== 评估 ==============

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")

    with open("norm_params.pkl", "rb") as f:
        norm_params = pickle.load(f)
    cap = norm_params["power"]["cap"]

    in_feat_size = X_test.shape[2]
    in_seq_len = X_test.shape[1]
    out_seq_len = y_test.shape[1]

    model = SolarTransformer(
        in_feat_size=in_feat_size,
        in_seq_len=in_seq_len,
        out_seq_len=out_seq_len,
        hidden_size=args.hidden_size,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_cross_attn_layers=args.num_cross_attn_layers,
        dropout=args.dropout,
    ).to(device)

    checkpoint = torch.load("solar_checkpoints/best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
        batch_size=args.batch_size, shuffle=False,
    )

    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            pred = model(batch_x)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(batch_y.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    acc = calc_acc_mae(all_targets, all_preds, cap)
    acc2 = calc_acc2(all_targets, all_preds, cap)
    rmse = calc_rmse(all_targets, all_preds, cap)
    mae = calc_mae(all_targets, all_preds, cap)

    print(f"\n{'='*60}")
    print(f"测试集评估结果")
    print(f"{'='*60}")
    print(f"  ACC1 (MAE-based): {acc:.4f} ({acc*100:.2f}%)")
    print(f"  ACC2 (国标):      {acc2:.4f} ({acc2*100:.2f}%)")
    print(f"  RMSE:             {rmse:.2f} MW")
    print(f"  MAE:              {mae:.2f} MW")

    # 按时间范围分析
    steps_per_hour = 4  # 15分钟粒度
    horizons = [
        (0, steps_per_hour, "0-1h"),
        (steps_per_hour, 2 * steps_per_hour, "1-2h"),
        (2 * steps_per_hour, 3 * steps_per_hour, "2-3h"),
        (3 * steps_per_hour, 4 * steps_per_hour, "3-4h"),
    ]

    print(f"\n按预测时间范围:")
    for start, end, name in horizons:
        yt = all_targets[:, start:end]
        yp = all_preds[:, start:end]
        h_acc = calc_acc_mae(yt, yp, cap)
        h_acc2 = calc_acc2(yt, yp, cap)
        h_rmse = calc_rmse(yt, yp, cap)
        print(f"  {name}: ACC1={h_acc:.4f}, ACC2={h_acc2:.4f}, RMSE={h_rmse:.2f} MW")

    print(f"{'='*60}")


# ============== 主入口 ==============

def main():
    parser = argparse.ArgumentParser(description="太阳能电站功率预测")
    # --- 运行模式 ---
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "preprocess", "train", "evaluate"])

    # --- 训练参数 ---
    parser.add_argument("--epochs", type=int, default=80,         help="最大训练轮次")
    parser.add_argument("--batch-size", type=int, default=64,     help="批大小")
    parser.add_argument("--lr", type=float, default=0.0003,       help="学习率")
    parser.add_argument("--weight-decay", type=float, default=0.03, help="L2正则化系数")
    parser.add_argument("--patience", type=int, default=10,       help="Early Stopping 耐心值")
    parser.add_argument("--warmup-epochs", type=int, default=3,   help="Warmup 轮次")
    parser.add_argument("--grad-clip", type=float, default=1.0,   help="梯度裁剪阈值")

    # --- 模型结构 ---
    parser.add_argument("--hidden-size", type=int, default=256,   help="Transformer 隐藏层维度")
    parser.add_argument("--nhead", type=int, default=8,           help="注意力头数")
    parser.add_argument("--num-encoder-layers", type=int, default=6,  help="Encoder 层数")
    parser.add_argument("--num-cross-attn-layers", type=int, default=3, help="Cross-Attention 层数")
    parser.add_argument("--dropout", type=float, default=0.15,    help="Dropout 比率")

    args = parser.parse_args()

    if args.mode in ["all", "preprocess"]:
        print("\n[1/3] 数据预处理")
        preprocess_main()

    if args.mode in ["all", "train"]:
        print("\n[2/3] 模型训练")
        train(args)

    if args.mode in ["all", "evaluate"]:
        print("\n[3/3] 模型评估")
        evaluate(args)


if __name__ == "__main__":
    main()
