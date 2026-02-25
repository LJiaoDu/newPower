#!/usr/bin/env python3
"""
太阳能电站功率预测 - 训练脚本 v2

改进点 (相比 train1.py):
1. 使用 model2.SolarTransformerRoPE (RoPE 位置编码)
2. 使用 process2.main (含 tsi_diff1 / tsi_diff2 / tsi_std_4 特征)
   Encoder: 16维, Decoder: 15维 (较原来各增加 3 个 TSI 衍生特征)
3. 特征维度从数据文件自动读取, 无需手动配置

使用方式:
  python train2.py                          # 完整流程: 预处理 + 训练 + 评估
  python train2.py --mode train             # 仅训练 (需先运行 preprocess)
  python train2.py --mode evaluate          # 仅评估
  python train2.py --lambda-mse 1.0 --lambda-acc2 0.5
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

from model2 import SolarTransformerRoPE
from process2 import main as preprocess_main


# ============================================================
# 损失函数
# ============================================================

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
        return torch.mean(((pred - target) / denom) ** 2)


class MixedLoss(nn.Module):
    """
    混合 Loss = λ_mse * MSE + λ_acc2 * ACC2Loss

    MSE:      绝对误差, 对高功率段敏感
    ACC2Loss: 相对误差, 与国标评估指标对齐
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


# ============================================================
# 评估指标
# ============================================================

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
    """ACC2 (国标): 1 - sqrt( mean( ((P_M - P_P) / max(P_M, 0.2*Cap))^2 ) )"""
    p_m = y_true.flatten() * cap
    p_p = y_pred.flatten() * cap
    denom = np.maximum(p_m, 0.2 * cap)
    return max(0.0, 1.0 - np.sqrt(np.mean(((p_m - p_p) / denom) ** 2)))


# ============================================================
# 训练
# ============================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载数据
    X_enc_train = np.load("X_enc_train.npy")
    X_dec_train = np.load("X_dec_train.npy")
    y_train     = np.load("y_train.npy")
    X_enc_val   = np.load("X_enc_val.npy")
    X_dec_val   = np.load("X_dec_val.npy")
    y_val       = np.load("y_val.npy")

    with open("norm_params.pkl", "rb") as f:
        norm_params = pickle.load(f)
    cap = norm_params["power"]["cap"]

    enc_seq_len  = X_enc_train.shape[1]
    enc_feat_size = X_enc_train.shape[2]
    dec_seq_len  = X_dec_train.shape[1]
    dec_feat_size = X_dec_train.shape[2]

    print(f"训练集: enc={X_enc_train.shape}, dec={X_dec_train.shape}, y={y_train.shape}")
    print(f"验证集: enc={X_enc_val.shape},   dec={X_dec_val.shape},   y={y_val.shape}")
    print(f"特征维度: Encoder={enc_feat_size}, Decoder={dec_feat_size}")
    print(f"标称容量: {cap} MW")

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

    # 模型 (RoPE + 自动适配特征维度)
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

    # 优化器 + 学习率调度
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    warmup_epochs = min(args.warmup_epochs, max(1, args.epochs - 1))
    warmup_sched  = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                             total_iters=warmup_epochs)
    cosine_sched  = CosineAnnealingLR(optimizer,
                                      T_max=max(1, args.epochs - warmup_epochs),
                                      eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched],
                              milestones=[warmup_epochs])

    # 损失函数
    criterion = MixedLoss(
        lambda_mse=args.lambda_mse,
        lambda_acc2=args.lambda_acc2,
        cap_norm=1.0,
    )

    os.makedirs("solar_checkpoints", exist_ok=True)

    # TensorBoard
    log_dir = os.path.join("runs", f"rope_tsi3_{time.strftime('%Y%m%d_%H%M%S')}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard 日志: {log_dir}")

    best_val_loss = float("inf")  # MixedLoss 最佳 (对照)
    best_acc2     = -1.0          # ACC2 最佳 (主要指标)
    patience_counter = 0

    print(f"\n{'='*70}")
    print(f"模型: SolarTransformerRoPE (RoPE 位置编码)")
    print(f"特征: Encoder {enc_feat_size}维 (含 tsi_diff1/diff2/std_4), "
          f"Decoder {dec_feat_size}维")
    print(f"Loss: λ_mse={args.lambda_mse} * MSE + λ_acc2={args.lambda_acc2} * ACC2Loss")
    print(f"Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"保存&早停指标: ACC2 (同时额外保存 MixedLoss 最佳)")
    print(f"{'='*70}\n")

    epoch_bar = tqdm(range(args.epochs), desc="Training", unit="epoch")
    for epoch in epoch_bar:

        # ---- 训练 ----
        model.train()
        tr_mixed = tr_mse = tr_acc2l = 0.0

        tr_bar = tqdm(train_loader,
                      desc=f"Epoch {epoch+1}/{args.epochs} [Train]",
                      leave=False, unit="batch")
        for enc_b, dec_b, y_b in tr_bar:
            enc_b, dec_b, y_b = enc_b.to(device), dec_b.to(device), y_b.to(device)

            optimizer.zero_grad()
            pred = model(enc_b, dec_b)

            l_mse  = criterion.mse_loss(pred, y_b)
            l_acc2 = criterion.acc2_loss(pred, y_b)
            loss   = args.lambda_mse * l_mse + args.lambda_acc2 * l_acc2

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            tr_mixed += loss.item()
            tr_mse   += l_mse.item()
            tr_acc2l += l_acc2.item()
            tr_bar.set_postfix(mixed=f"{loss.item():.5f}",
                               mse=f"{l_mse.item():.5f}",
                               acc2l=f"{l_acc2.item():.5f}")

        n_tr = len(train_loader)
        tr_mixed /= n_tr
        tr_mse   /= n_tr
        tr_acc2l /= n_tr

        # ---- 验证 ----
        model.eval()
        va_mixed = va_mse = va_acc2l = 0.0
        all_preds, all_targets = [], []

        va_bar = tqdm(val_loader,
                      desc=f"Epoch {epoch+1}/{args.epochs} [Val]",
                      leave=False, unit="batch")
        with torch.no_grad():
            for enc_b, dec_b, y_b in va_bar:
                enc_b, dec_b, y_b = enc_b.to(device), dec_b.to(device), y_b.to(device)
                pred = model(enc_b, dec_b)

                l_mse  = criterion.mse_loss(pred, y_b)
                l_acc2 = criterion.acc2_loss(pred, y_b)
                loss   = args.lambda_mse * l_mse + args.lambda_acc2 * l_acc2

                va_mixed += loss.item()
                va_mse   += l_mse.item()
                va_acc2l += l_acc2.item()

                all_preds.append(pred.cpu().numpy())
                all_targets.append(y_b.cpu().numpy())

        n_va = len(val_loader)
        va_mixed /= n_va
        va_mse   /= n_va
        va_acc2l /= n_va

        all_preds   = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        acc1 = calc_acc_mae(all_targets, all_preds, cap)
        acc2 = calc_acc2(all_targets, all_preds, cap)
        rmse = calc_rmse(all_targets, all_preds, cap)
        mae  = calc_mae(all_targets, all_preds, cap)
        lr   = optimizer.param_groups[0]["lr"]

        # TensorBoard
        writer.add_scalars("Loss/Mixed",   {"Train": tr_mixed,  "Val": va_mixed},  epoch + 1)
        writer.add_scalars("Loss/MSE",     {"Train": tr_mse,    "Val": va_mse},    epoch + 1)
        writer.add_scalars("Loss/ACC2Loss",{"Train": tr_acc2l,  "Val": va_acc2l},  epoch + 1)
        writer.add_scalar("Accuracy/ACC1", acc1, epoch + 1)
        writer.add_scalar("Accuracy/ACC2", acc2, epoch + 1)
        writer.add_scalar("Error/RMSE_MW", rmse, epoch + 1)
        writer.add_scalar("Error/MAE_MW",  mae,  epoch + 1)
        writer.add_scalar("LearningRate",  lr,   epoch + 1)

        epoch_bar.set_postfix(
            tr=f"{tr_mixed:.5f}", va=f"{va_mixed:.5f}", ACC2=f"{acc2:.4f}"
        )
        tqdm.write(
            f"Epoch {epoch+1:3d}/{args.epochs} | LR: {lr:.6f} | "
            f"Train(Mix/MSE/ACC2L): {tr_mixed:.6f}/{tr_mse:.6f}/{tr_acc2l:.6f} | "
            f"Val(Mix/MSE/ACC2L):   {va_mixed:.6f}/{va_mse:.6f}/{va_acc2l:.6f} | "
            f"ACC1: {acc1:.4f} | ACC2: {acc2:.4f} | "
            f"RMSE: {rmse:.2f} MW | MAE: {mae:.2f} MW"
        )

        ckpt_base = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_mixed": va_mixed,
            "val_mse": va_mse,
            "val_acc2loss": va_acc2l,
            "acc1": acc1,
            "acc2": acc2,
            "norm_params": norm_params,
            "args": vars(args),
        }

        # 保存 MixedLoss 最佳 (对照用)
        if va_mixed < best_val_loss:
            best_val_loss = va_mixed
            torch.save(ckpt_base,
                       "solar_checkpoints/best_model_rope_tsi3_mixedloss.pth")
            tqdm.write(f"  -> 保存 MixedLoss 最佳 (Val MixedLoss: {va_mixed:.6f})")

        # 保存 ACC2 最佳 (主要指标)
        improved = acc2 > best_acc2
        if improved:
            best_acc2 = acc2
            torch.save(ckpt_base,
                       "solar_checkpoints/best_model_rope_tsi3_acc2.pth")
            tqdm.write(f"  -> 保存 ACC2 最佳 (ACC2: {acc2:.6f})")

        # Early Stopping (按 ACC2)
        if improved:
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                tqdm.write(f"\nEarly stopping: {args.patience} epochs ACC2 无提升")
                break

        scheduler.step()

    writer.close()
    print(f"\n训练完成!")
    print(f"  MixedLoss 最佳: {best_val_loss:.6f}  "
          f"-> solar_checkpoints/best_model_rope_tsi3_mixedloss.pth")
    print(f"  ACC2     最佳: {best_acc2:.6f}  "
          f"-> solar_checkpoints/best_model_rope_tsi3_acc2.pth")


# ============================================================
# 评估
# ============================================================

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_enc_test = np.load("X_enc_test.npy")
    X_dec_test = np.load("X_dec_test.npy")
    y_test     = np.load("y_test.npy")

    with open("norm_params.pkl", "rb") as f:
        norm_params = pickle.load(f)
    cap = norm_params["power"]["cap"]

    enc_feat_size = X_enc_test.shape[2]
    dec_feat_size = X_dec_test.shape[2]
    enc_seq_len   = X_enc_test.shape[1]
    dec_seq_len   = X_dec_test.shape[1]

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

    ckpt_path = "solar_checkpoints/best_model_rope_tsi3_acc2.pth"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"加载模型: {ckpt_path}")
    print(f"  epoch={checkpoint['epoch']+1}, "
          f"val_mixed={checkpoint['val_mixed']:.6f}, "
          f"acc2={checkpoint['acc2']:.6f}")

    test_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_enc_test),
            torch.FloatTensor(X_dec_test),
            torch.FloatTensor(y_test),
        ),
        batch_size=args.batch_size, shuffle=False,
    )

    all_preds, all_targets = [], []
    with torch.no_grad():
        for enc_b, dec_b, y_b in tqdm(test_loader, desc="Testing"):
            enc_b, dec_b = enc_b.to(device), dec_b.to(device)
            pred = model(enc_b, dec_b)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_b.numpy())

    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    acc1 = calc_acc_mae(all_targets, all_preds, cap)
    acc2 = calc_acc2(all_targets, all_preds, cap)
    rmse = calc_rmse(all_targets, all_preds, cap)
    mae  = calc_mae(all_targets, all_preds, cap)

    print(f"\n{'='*60}")
    print(f"测试集评估结果 (RoPE + tsi_diff1/diff2/std_4)")
    print(f"{'='*60}")
    print(f"  ACC1 (MAE-based): {acc1:.4f} ({acc1*100:.2f}%)")
    print(f"  ACC2 (国标):      {acc2:.4f} ({acc2*100:.2f}%)")
    print(f"  RMSE:             {rmse:.2f} MW")
    print(f"  MAE:              {mae:.2f} MW")

    # 按预测时间范围分析
    steps_per_hour = 4
    horizons = [
        (0,                   steps_per_hour,     "0-1h"),
        (steps_per_hour,      2*steps_per_hour,   "1-2h"),
        (2*steps_per_hour,    3*steps_per_hour,   "2-3h"),
        (3*steps_per_hour,    4*steps_per_hour,   "3-4h"),
    ]
    print(f"\n按预测时间范围:")
    for start, end, name in horizons:
        yt = all_targets[:, start:end]
        yp = all_preds[:, start:end]
        print(f"  {name}: ACC1={calc_acc_mae(yt,yp,cap):.4f}, "
              f"ACC2={calc_acc2(yt,yp,cap):.4f}, "
              f"RMSE={calc_rmse(yt,yp,cap):.2f} MW, "
              f"MAE={calc_mae(yt,yp,cap):.2f} MW")
    print(f"{'='*60}")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="太阳能电站功率预测 (RoPE + tsi_diff1/diff2/std_4)"
    )

    # 运行模式
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "preprocess", "train", "evaluate"])
    parser.add_argument("--csv-path", type=str, default="solar_station_1.csv",
                        help="solar_station_1.csv 文件路径")

    # 训练参数
    parser.add_argument("--epochs",        type=int,   default=100)
    parser.add_argument("--batch-size",    type=int,   default=64)
    parser.add_argument("--lr",            type=float, default=3e-4)
    parser.add_argument("--weight-decay",  type=float, default=0.01)
    parser.add_argument("--patience",      type=int,   default=15,
                        help="Early Stopping 耐心值 (按ACC2)")
    parser.add_argument("--warmup-epochs", type=int,   default=5)
    parser.add_argument("--grad-clip",     type=float, default=1.0)

    # 模型结构
    parser.add_argument("--d-model",             type=int,   default=256)
    parser.add_argument("--nhead",               type=int,   default=8)
    parser.add_argument("--num-encoder-layers",  type=int,   default=4)
    parser.add_argument("--num-decoder-layers",  type=int,   default=4)
    parser.add_argument("--dropout",             type=float, default=0.1)

    # 混合 Loss 权重
    parser.add_argument("--lambda-mse",  type=float, default=0.0,
                        help="MSE Loss 权重")
    parser.add_argument("--lambda-acc2", type=float, default=1.0,
                        help="ACC2Loss 权重")

    args = parser.parse_args()

    if args.mode in ["all", "preprocess"]:
        print("\n[1/3] 数据预处理 (含 tsi_diff1/diff2/std_4)")
        preprocess_main(csv_path=args.csv_path)

    if args.mode in ["all", "train"]:
        print("\n[2/3] 模型训练 (RoPE)")
        train(args)

    if args.mode in ["all", "evaluate"]:
        print("\n[3/3] 模型评估")
        evaluate(args)


if __name__ == "__main__":
    main()
