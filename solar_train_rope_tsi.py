#!/usr/bin/env python3
"""
太阳能电站功率预测 - RoPE ED + Δtsi 特征 训练脚本

基于 solar_train_rope.py (rope v1), 改动:
1. 使用 solar_preprocess_v2 (含 Δtsi 特征, Encoder 14维, Decoder 13维)
2. 模型自动适配新特征维度 (enc_feat_size / dec_feat_size 从数据读取)
3. 不过滤夜间样本

使用方式:
  python solar_train_rope_tsi.py                          # 完整流程
  python solar_train_rope_tsi.py --mode train             # 仅训练 (需先跑 preprocess)
  python solar_train_rope_tsi.py --mode evaluate          # 仅评估
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
from tqdm import tqdm

from solar_model_rope import SolarTransformerRoPE
from solar_preprocess_v2 import main as preprocess_main


# ============== 混合 Loss ==============

class ACC2Loss(nn.Module):
    """
    基于国标 ACC2 的损失函数
    L = mean( ((pred - target) / max(target, 0.2 * cap_norm))^2 )
    """

    def __init__(self, cap_norm=1.0):
        super().__init__()
        self.cap_norm = cap_norm

    def forward(self, pred, target):
        denom = torch.clamp(target, min=0.2 * self.cap_norm)
        relative_error = (pred - target) / denom
        return torch.mean(relative_error ** 2)


class MixedLoss(nn.Module):
    """
    混合 Loss = λ_mse * MSE + λ_acc2 * ACC2_Loss
    """

    def __init__(self, lambda_mse=1.0, lambda_acc2=0.5, cap_norm=1.0):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_acc2 = lambda_acc2
        self.mse_loss = nn.MSELoss()
        self.acc2_loss = ACC2Loss(cap_norm=cap_norm)

    def forward(self, pred, target):
        l_mse = self.mse_loss(pred, target)
        l_acc2 = self.acc2_loss(pred, target)
        return self.lambda_mse * l_mse + self.lambda_acc2 * l_acc2


# ============== 评估指标 ==============

def calc_acc_mae(y_true, y_pred, cap=1.0):
    """ACC1 (MAE-based): 1 - MAE / mean(y_true)"""
    mask = y_true > 0.01
    if mask.sum() == 0:
        return float("nan")
    y_t = y_true[mask] * cap
    y_p = y_pred[mask] * cap
    mae = np.mean(np.abs(y_t - y_p))
    avg = np.mean(y_t)
    return max(0.0, 1.0 - mae / (avg + 1e-6))


def calc_rmse(y_true, y_pred, cap=1.0):
    return np.sqrt(np.mean((y_true * cap - y_pred * cap) ** 2))


def calc_mae(y_true, y_pred, cap=1.0):
    return np.mean(np.abs(y_true * cap - y_pred * cap))


def calc_acc2(y_true, y_pred, cap=1.0):
    """ACC2 (国标): 1 - sqrt( (1/N) * sum( ((P_M - P_P) / max(P_M, 0.2*Cap))^2 ) )"""
    p_m = y_true.flatten() * cap
    p_p = y_pred.flatten() * cap
    denom = np.maximum(p_m, 0.2 * cap)
    return max(0.0, 1.0 - np.sqrt(np.mean(((p_m - p_p) / denom) ** 2)))


# ============== 训练 ==============

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载数据 (不过滤夜间样本)
    X_enc_train = np.load("X_enc_train.npy")
    X_dec_train = np.load("X_dec_train.npy")
    y_train = np.load("y_train.npy")
    X_enc_val = np.load("X_enc_val.npy")
    X_dec_val = np.load("X_dec_val.npy")
    y_val = np.load("y_val.npy")

    with open("norm_params.pkl", "rb") as f:
        norm_params = pickle.load(f)
    cap = norm_params["power"]["cap"]

    print(f"训练集: enc={X_enc_train.shape}, dec={X_dec_train.shape}, y={y_train.shape}")
    print(f"验证集: enc={X_enc_val.shape}, dec={X_dec_val.shape}, y={y_val.shape}")
    print(f"标称容量: {cap} MW")

    enc_seq_len = X_enc_train.shape[1]
    enc_feat_size = X_enc_train.shape[2]
    dec_seq_len = X_dec_train.shape[1]
    dec_feat_size = X_dec_train.shape[2]

    print(f"特征维度: Encoder={enc_feat_size}, Decoder={dec_feat_size}")

    # DataLoader
    train_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_enc_train),
            torch.FloatTensor(X_dec_train),
            torch.FloatTensor(y_train),
        ),
        batch_size=args.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_enc_val),
            torch.FloatTensor(X_dec_val),
            torch.FloatTensor(y_val),
        ),
        batch_size=args.batch_size, shuffle=False,
    )

    # 模型 (自动适配 14/13 维特征)
    model = SolarTransformerRoPE(
        enc_feat_size=enc_feat_size,
        dec_feat_size=dec_feat_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.d_model * 4,
        dropout=args.dropout,
        enc_seq_len=enc_seq_len,
        dec_seq_len=dec_seq_len,
    ).to(device)

    # Xavier 初始化
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p, gain=0.5)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[args.warmup_epochs])

    # 混合 Loss: MSE + ACC2
    criterion = MixedLoss(
        lambda_mse=args.lambda_mse,
        lambda_acc2=args.lambda_acc2,
        cap_norm=1.0,
    )
    print(f"Loss: MixedLoss(λ_mse={args.lambda_mse}, λ_acc2={args.lambda_acc2})")

    os.makedirs("solar_checkpoints", exist_ok=True)

    # TensorBoard
    log_dir = os.path.join("runs", f"rope_tsi_{time.strftime('%Y%m%d_%H%M%S')}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard 日志: {log_dir}")

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n{'='*70}")
    print(f"开始训练 | Epochs: {args.epochs} | Batch: {args.batch_size} | "
          f"LR: {args.lr}")
    print(f"Loss: λ_mse={args.lambda_mse} * MSE + λ_acc2={args.lambda_acc2} * ACC2Loss")
    print(f"模型: SolarTransformerRoPE | d_model={args.d_model}, heads={args.nhead}, "
          f"enc_layers={args.num_encoder_layers}, dec_layers={args.num_decoder_layers}")
    print(f"特征: Encoder {enc_feat_size}维 (含Δtsi), Decoder {dec_feat_size}维 (含Δtsi)")
    print(f"{'='*70}\n")

    epoch_bar = tqdm(range(args.epochs), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        # --- 训练 ---
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]",
                         leave=False, unit="batch")
        for batch_enc, batch_dec, batch_y in train_bar:
            batch_enc = batch_enc.to(device)
            batch_dec = batch_dec.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_enc, batch_dec)
            loss = criterion(pred, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.6f}")
        train_loss /= len(train_loader)

        # --- 验证 ---
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]",
                       leave=False, unit="batch")
        with torch.no_grad():
            for batch_enc, batch_dec, batch_y in val_bar:
                batch_enc = batch_enc.to(device)
                batch_dec = batch_dec.to(device)
                batch_y = batch_y.to(device)

                pred = model(batch_enc, batch_dec)
                val_loss += criterion(pred, batch_y).item()
                all_preds.append(pred.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
        val_loss /= len(val_loader)

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        acc1 = calc_acc_mae(all_targets, all_preds, cap)
        acc2 = calc_acc2(all_targets, all_preds, cap)
        rmse = calc_rmse(all_targets, all_preds, cap)
        mae = calc_mae(all_targets, all_preds, cap)
        lr = optimizer.param_groups[0]["lr"]

        # TensorBoard
        writer.add_scalars("Loss", {"Train": train_loss, "Val": val_loss}, epoch + 1)
        writer.add_scalar("Accuracy/ACC1", acc1, epoch + 1)
        writer.add_scalar("Accuracy/ACC2", acc2, epoch + 1)
        writer.add_scalar("Error/RMSE_MW", rmse, epoch + 1)
        writer.add_scalar("Error/MAE_MW", mae, epoch + 1)
        writer.add_scalar("LearningRate", lr, epoch + 1)

        epoch_bar.set_postfix(train=f"{train_loss:.5f}", val=f"{val_loss:.5f}",
                              ACC1=f"{acc1:.4f}", ACC2=f"{acc2:.4f}")
        tqdm.write(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"LR: {lr:.6f} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f} | "
            f"ACC1: {acc1:.4f} | "
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
                "acc1": acc1,
                "acc2": acc2,
                "norm_params": norm_params,
                "args": vars(args),
            }, "solar_checkpoints/best_model_rope_tsi.pth")
            tqdm.write(f"  -> 保存最佳模型 (Val Loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                tqdm.write(f"\nEarly stopping: {args.patience} epochs 无改善")
                break

        scheduler.step()

    writer.close()
    print(f"\n训练完成! 最佳验证损失: {best_val_loss:.6f}")


# ============== 评估 ==============

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_enc_test = np.load("X_enc_test.npy")
    X_dec_test = np.load("X_dec_test.npy")
    y_test = np.load("y_test.npy")

    with open("norm_params.pkl", "rb") as f:
        norm_params = pickle.load(f)
    cap = norm_params["power"]["cap"]

    enc_feat_size = X_enc_test.shape[2]
    dec_feat_size = X_dec_test.shape[2]
    enc_seq_len = X_enc_test.shape[1]
    dec_seq_len = X_dec_test.shape[1]

    model = SolarTransformerRoPE(
        enc_feat_size=enc_feat_size,
        dec_feat_size=dec_feat_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.d_model * 4,
        dropout=args.dropout,
        enc_seq_len=enc_seq_len,
        dec_seq_len=dec_seq_len,
    ).to(device)

    checkpoint = torch.load("solar_checkpoints/best_model_rope_tsi.pth",
                            map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"加载模型: epoch={checkpoint['epoch']+1}, "
          f"val_loss={checkpoint['val_loss']:.6f}")

    test_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_enc_test),
            torch.FloatTensor(X_dec_test),
            torch.FloatTensor(y_test),
        ),
        batch_size=args.batch_size, shuffle=False,
    )

    all_preds, all_targets = [], []
    test_bar = tqdm(test_loader, desc="Testing", unit="batch")
    with torch.no_grad():
        for batch_enc, batch_dec, batch_y in test_bar:
            batch_enc = batch_enc.to(device)
            batch_dec = batch_dec.to(device)
            pred = model(batch_enc, batch_dec)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(batch_y.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    acc1 = calc_acc_mae(all_targets, all_preds, cap)
    acc2 = calc_acc2(all_targets, all_preds, cap)
    rmse = calc_rmse(all_targets, all_preds, cap)
    mae = calc_mae(all_targets, all_preds, cap)

    print(f"\n{'='*60}")
    print(f"测试集评估结果")
    print(f"{'='*60}")
    print(f"  ACC1 (MAE-based): {acc1:.4f} ({acc1*100:.2f}%)")
    print(f"  ACC2 (国标):      {acc2:.4f} ({acc2*100:.2f}%)")
    print(f"  RMSE:             {rmse:.2f} MW")
    print(f"  MAE:              {mae:.2f} MW")

    # 按预测时间范围分析
    steps_per_hour = 4
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
        h_acc1 = calc_acc_mae(yt, yp, cap)
        h_acc2 = calc_acc2(yt, yp, cap)
        h_rmse = calc_rmse(yt, yp, cap)
        h_mae = calc_mae(yt, yp, cap)
        print(f"  {name}: ACC1={h_acc1:.4f}, ACC2={h_acc2:.4f}, "
              f"RMSE={h_rmse:.2f} MW, MAE={h_mae:.2f} MW")

    print(f"{'='*60}")


# ============== 主入口 ==============

def main():
    parser = argparse.ArgumentParser(description="太阳能电站功率预测 (RoPE ED + Δtsi 特征)")

    # --- 运行模式 ---
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "preprocess", "train", "evaluate"])
    parser.add_argument("--csv-path", type=str,
                        default="/media/zlg/Data1/Longjiao/TF208/solar_station_1.csv",
                        help="solar_station_1.csv 文件路径")

    # --- 训练参数 ---
    parser.add_argument("--epochs", type=int, default=100,
                        help="最大训练轮次")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="批大小")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="学习率")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="L2正则化系数")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early Stopping 耐心值")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="Warmup 轮次")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="梯度裁剪阈值")

    # --- 模型结构 ---
    parser.add_argument("--d-model", type=int, default=256,
                        help="Transformer 隐藏层维度")
    parser.add_argument("--nhead", type=int, default=8,
                        help="注意力头数")
    parser.add_argument("--num-encoder-layers", type=int, default=4,
                        help="Encoder 层数")
    parser.add_argument("--num-decoder-layers", type=int, default=4,
                        help="Decoder 层数")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout 比率")

    # --- 混合 Loss 权重 ---
    parser.add_argument("--lambda-mse", type=float, default=1.0,
                        help="MSE Loss 权重")
    parser.add_argument("--lambda-acc2", type=float, default=0.5,
                        help="ACC2 Loss 权重")

    args = parser.parse_args()

    if args.mode in ["all", "preprocess"]:
        print("\n[1/3] 数据预处理 (v2, 含 Δtsi)")
        preprocess_main(csv_path=args.csv_path)

    if args.mode in ["all", "train"]:
        print("\n[2/3] 模型训练")
        train(args)

    if args.mode in ["all", "evaluate"]:
        print("\n[3/3] 模型评估")
        evaluate(args)


if __name__ == "__main__":
    main()
