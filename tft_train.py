#!/usr/bin/env python3
"""
TFT 训练与评估脚本

损失函数: 分位数损失 (Quantile / Pinball Loss), 覆盖 P10 / P50 / P90
评估指标: ACC1 (MAE-based) / ACC2 (国标) / RMSE / MAE  --  均使用 P50 预测值
优化策略: AdamW + Warmup + CosineAnnealingLR + EarlyStopping + 梯度裁剪

使用方式:
  python tft_train.py                          # 完整流程: 预处理 + 训练 + 评估
  python tft_train.py --mode preprocess        # 仅数据预处理
  python tft_train.py --mode train             # 仅训练
  python tft_train.py --mode evaluate          # 仅评估 (需已有 checkpoint)
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

from tft_model import TemporalFusionTransformer
from tft_preprocess import main as preprocess_main


# ============================================================
#  分位数损失 (Pinball Loss)
# ============================================================

def quantile_loss(y_pred: torch.Tensor, y_true: torch.Tensor,
                  quantiles: tuple) -> torch.Tensor:
    """
    y_pred: [B, T, num_quantiles]
    y_true: [B, T]
    返回标量损失 (所有分位数的平均)
    """
    y_true = y_true.unsqueeze(-1)          # [B, T, 1]
    q = torch.tensor(quantiles, dtype=y_pred.dtype,
                     device=y_pred.device)  # [num_quantiles]
    errors = y_true - y_pred               # [B, T, num_quantiles]
    loss = torch.max(q * errors, (q - 1) * errors)  # [B, T, num_quantiles]
    return loss.mean()


# ============================================================
#  评估指标 (与 solar_train.py 一致)
# ============================================================

def calc_acc1(y_true, y_pred, cap=1.0):
    """ACC1 (MAE-based): 1 - MAE / mean(y_true)"""
    mask = y_true > 0.01
    if mask.sum() == 0:
        return float("nan")
    yt = y_true[mask] * cap
    yp = y_pred[mask] * cap
    mae = np.mean(np.abs(yt - yp))
    return max(0.0, 1.0 - mae / (np.mean(yt) + 1e-6))


def calc_acc2(y_true, y_pred, cap=1.0):
    """ACC2 (国标): 1 - sqrt( mean( ((P_M-P_P)/max(P_M, 0.2*Cap))^2 ) )"""
    pm = y_true.flatten() * cap
    pp = y_pred.flatten() * cap
    denom = np.maximum(pm, 0.2 * cap)
    return max(0.0, 1.0 - np.sqrt(np.mean(((pm - pp) / denom) ** 2)))


def calc_rmse(y_true, y_pred, cap=1.0):
    return np.sqrt(np.mean((y_true * cap - y_pred * cap) ** 2))


def calc_mae(y_true, y_pred, cap=1.0):
    return np.mean(np.abs(y_true * cap - y_pred * cap))


# ============================================================
#  训练
# ============================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # --- 加载数据 ---
    X_enc_tr    = np.load("X_enc_train.npy")
    X_dec_tr    = np.load("X_dec_train.npy")
    X_static_tr = np.load("X_static_train.npy")
    y_tr        = np.load("y_train.npy")

    X_enc_val    = np.load("X_enc_val.npy")
    X_dec_val    = np.load("X_dec_val.npy")
    X_static_val = np.load("X_static_val.npy")
    y_val        = np.load("y_val.npy")

    with open("norm_params_tft.pkl", "rb") as f:
        norm_params = pickle.load(f)
    cap = norm_params["power"]["cap"]

    print(f"训练集: enc={X_enc_tr.shape}, dec={X_dec_tr.shape}, "
          f"static={X_static_tr.shape}, y={y_tr.shape}")
    print(f"验证集: enc={X_enc_val.shape}, dec={X_dec_val.shape}, "
          f"static={X_static_val.shape}, y={y_val.shape}")
    print(f"标称容量: {cap} MW")

    enc_feat_size    = X_enc_tr.shape[2]
    dec_feat_size    = X_dec_tr.shape[2]
    static_feat_size = X_static_tr.shape[1]

    # --- DataLoader ---
    def make_loader(xe, xd, xs, yt, shuffle):
        return DataLoader(
            TensorDataset(
                torch.FloatTensor(xe),
                torch.FloatTensor(xd),
                torch.FloatTensor(xs),
                torch.FloatTensor(yt),
            ),
            batch_size=args.batch_size, shuffle=shuffle,
            num_workers=0, pin_memory=(device.type == "cuda"),
        )

    train_loader = make_loader(X_enc_tr,  X_dec_tr,  X_static_tr,  y_tr,  True)
    val_loader   = make_loader(X_enc_val, X_dec_val, X_static_val, y_val, False)

    # --- 模型 ---
    quantiles = (0.1, 0.5, 0.9)
    model = TemporalFusionTransformer(
        num_past_vars=enc_feat_size,
        num_future_vars=dec_feat_size,
        num_static_vars=static_feat_size,
        d_model=args.d_model,
        num_heads=args.nhead,
        num_lstm_layers=args.num_lstm_layers,
        dropout=args.dropout,
        quantiles=quantiles,
    ).to(device)

    # Xavier 初始化
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p, gain=0.5)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTFT 参数量: {total_params:,}")

    # --- 优化器 & 学习率调度 ---
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)
    warmup  = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                       total_iters=args.warmup_epochs)
    cosine  = CosineAnnealingLR(optimizer,
                                T_max=args.epochs - args.warmup_epochs,
                                eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup, cosine],
                             milestones=[args.warmup_epochs])

    os.makedirs("tft_checkpoints", exist_ok=True)
    log_dir = os.path.join("runs_tft", time.strftime("%Y%m%d_%H%M%S"))
    writer  = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard 日志: {log_dir}")

    print(f"\n{'='*72}")
    print(f"开始训练 TFT | Epochs: {args.epochs} | Batch: {args.batch_size} | "
          f"LR: {args.lr}")
    print(f"模型: d_model={args.d_model}, heads={args.nhead}, "
          f"lstm_layers={args.num_lstm_layers}, dropout={args.dropout}")
    print(f"分位数预测: P10 / P50 / P90")
    print(f"{'='*72}\n")

    best_val_loss  = float("inf")
    patience_count = 0
    p50_idx        = list(quantiles).index(0.5)   # P50 对应的输出索引

    epoch_bar = tqdm(range(args.epochs), desc="Training", unit="epoch")
    for epoch in epoch_bar:

        # ===== 训练 =====
        model.train()
        train_loss = 0.0
        for xe, xd, xs, yt in tqdm(train_loader,
                                    desc=f"Epoch {epoch+1} [Train]",
                                    leave=False):
            xe, xd, xs, yt = (t.to(device) for t in (xe, xd, xs, yt))
            optimizer.zero_grad()
            pred = model(xe, xd, xs)                  # [B, T, 3]
            loss = quantile_loss(pred, yt, quantiles)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ===== 验证 =====
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for xe, xd, xs, yt in tqdm(val_loader,
                                        desc=f"Epoch {epoch+1} [Val]",
                                        leave=False):
                xe, xd, xs, yt = (t.to(device) for t in (xe, xd, xs, yt))
                pred = model(xe, xd, xs)              # [B, T, 3]
                val_loss += quantile_loss(pred, yt, quantiles).item()
                all_preds.append(pred[:, :, p50_idx].cpu().numpy())  # P50
                all_targets.append(yt.cpu().numpy())
        val_loss /= len(val_loader)

        preds   = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)

        acc1 = calc_acc1(targets, preds, cap)
        acc2 = calc_acc2(targets, preds, cap)
        rmse = calc_rmse(targets, preds, cap)
        mae  = calc_mae(targets, preds, cap)
        lr   = optimizer.param_groups[0]["lr"]

        # TensorBoard
        writer.add_scalars("Loss",         {"Train": train_loss, "Val": val_loss}, epoch + 1)
        writer.add_scalar("Accuracy/ACC1", acc1, epoch + 1)
        writer.add_scalar("Accuracy/ACC2", acc2, epoch + 1)
        writer.add_scalar("Error/RMSE_MW", rmse, epoch + 1)
        writer.add_scalar("Error/MAE_MW",  mae,  epoch + 1)
        writer.add_scalar("LearningRate",  lr,   epoch + 1)

        epoch_bar.set_postfix(train=f"{train_loss:.5f}", val=f"{val_loss:.5f}",
                              ACC1=f"{acc1:.4f}", ACC2=f"{acc2:.4f}")
        tqdm.write(
            f"Epoch {epoch+1:3d}/{args.epochs} | LR: {lr:.2e} | "
            f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | "
            f"ACC1: {acc1:.4f} | ACC2: {acc2:.4f} | "
            f"RMSE: {rmse:.2f} MW | MAE: {mae:.2f} MW"
        )

        # 保存最佳 checkpoint
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "acc1": acc1, "acc2": acc2,
                "norm_params": norm_params,
                "args": vars(args),
                "quantiles": quantiles,
            }, "tft_checkpoints/best_model.pth")
            tqdm.write(f"  -> 保存最佳模型 (Val Loss: {val_loss:.6f})")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                tqdm.write(f"\nEarly Stopping: {args.patience} 轮无改善")
                break

        scheduler.step()

    writer.close()
    print(f"\n训练完成! 最佳验证损失: {best_val_loss:.6f}")


# ============================================================
#  评估
# ============================================================

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_enc_te    = np.load("X_enc_test.npy")
    X_dec_te    = np.load("X_dec_test.npy")
    X_static_te = np.load("X_static_test.npy")
    y_te        = np.load("y_test.npy")

    ckpt = torch.load("tft_checkpoints/best_model.pth",
                      map_location=device, weights_only=False)
    norm_params = ckpt["norm_params"]
    cap         = norm_params["power"]["cap"]
    quantiles   = ckpt.get("quantiles", (0.1, 0.5, 0.9))
    p50_idx     = list(quantiles).index(0.5)

    enc_feat_size    = X_enc_te.shape[2]
    dec_feat_size    = X_dec_te.shape[2]
    static_feat_size = X_static_te.shape[1]

    ckpt_args = ckpt.get("args", {})
    model = TemporalFusionTransformer(
        num_past_vars=enc_feat_size,
        num_future_vars=dec_feat_size,
        num_static_vars=static_feat_size,
        d_model=ckpt_args.get("d_model", args.d_model),
        num_heads=ckpt_args.get("nhead", args.nhead),
        num_lstm_layers=ckpt_args.get("num_lstm_layers", args.num_lstm_layers),
        dropout=0.0,
        quantiles=quantiles,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"加载 checkpoint: epoch={ckpt['epoch']+1}, "
          f"val_loss={ckpt['val_loss']:.6f}")

    test_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_enc_te),
            torch.FloatTensor(X_dec_te),
            torch.FloatTensor(X_static_te),
            torch.FloatTensor(y_te),
        ),
        batch_size=args.batch_size, shuffle=False,
    )

    all_preds_q, all_targets = [], []
    with torch.no_grad():
        for xe, xd, xs, yt in tqdm(test_loader, desc="Testing"):
            xe, xd, xs = xe.to(device), xd.to(device), xs.to(device)
            pred = model(xe, xd, xs)          # [B, T, 3]
            all_preds_q.append(pred.cpu().numpy())
            all_targets.append(yt.numpy())

    preds_q  = np.concatenate(all_preds_q)   # [N, 16, 3]
    targets  = np.concatenate(all_targets)    # [N, 16]
    preds_p50 = preds_q[:, :, p50_idx]       # [N, 16]

    # --- 整体指标 ---
    acc1 = calc_acc1(targets, preds_p50, cap)
    acc2 = calc_acc2(targets, preds_p50, cap)
    rmse = calc_rmse(targets, preds_p50, cap)
    mae  = calc_mae(targets,  preds_p50, cap)

    print(f"\n{'='*60}")
    print(f"TFT 测试集评估结果 (P50 预测)")
    print(f"{'='*60}")
    print(f"  ACC1 (MAE-based): {acc1:.4f}  ({acc1*100:.2f}%)")
    print(f"  ACC2 (国标):      {acc2:.4f}  ({acc2*100:.2f}%)")
    print(f"  RMSE:             {rmse:.2f} MW")
    print(f"  MAE:              {mae:.2f} MW")

    # --- 分预测时段分析 ---
    steps_per_hour = 4
    horizons = [
        (0 * steps_per_hour, 1 * steps_per_hour, "0-1h"),
        (1 * steps_per_hour, 2 * steps_per_hour, "1-2h"),
        (2 * steps_per_hour, 3 * steps_per_hour, "2-3h"),
        (3 * steps_per_hour, 4 * steps_per_hour, "3-4h"),
    ]
    print(f"\n按预测时段 (P50):")
    for s, e, name in horizons:
        yt_h = targets[:, s:e]
        yp_h = preds_p50[:, s:e]
        print(f"  {name}: ACC1={calc_acc1(yt_h, yp_h, cap):.4f}  "
              f"ACC2={calc_acc2(yt_h, yp_h, cap):.4f}  "
              f"RMSE={calc_rmse(yt_h, yp_h, cap):.2f} MW  "
              f"MAE={calc_mae(yt_h, yp_h, cap):.2f} MW")

    # --- 分位数区间覆盖率 ---
    p10 = preds_q[:, :, 0]
    p90 = preds_q[:, :, 2]
    within = ((targets >= p10) & (targets <= p90)).mean()
    print(f"\nP10-P90 区间覆盖率: {within*100:.1f}%  (期望 ~80%)")
    print(f"{'='*60}")


# ============================================================
#  主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="TFT 太阳能功率预测")

    parser.add_argument("--mode", default="all",
                        choices=["all", "preprocess", "train", "evaluate"])
    parser.add_argument("--csv-path", default="solar_station_1.csv",
                        help="solar_station_1.csv 路径")

    # 训练超参数
    parser.add_argument("--epochs",        type=int,   default=100)
    parser.add_argument("--batch-size",    type=int,   default=64)
    parser.add_argument("--lr",            type=float, default=3e-4)
    parser.add_argument("--weight-decay",  type=float, default=0.01)
    parser.add_argument("--patience",      type=int,   default=15)
    parser.add_argument("--warmup-epochs", type=int,   default=5)
    parser.add_argument("--grad-clip",     type=float, default=1.0)

    # 模型结构
    parser.add_argument("--d-model",          type=int,   default=128)
    parser.add_argument("--nhead",            type=int,   default=4)
    parser.add_argument("--num-lstm-layers",  type=int,   default=1)
    parser.add_argument("--dropout",          type=float, default=0.1)

    args = parser.parse_args()

    if args.mode in ("all", "preprocess"):
        print("\n[1/3] TFT 数据预处理")
        preprocess_main(csv_path=args.csv_path)

    if args.mode in ("all", "train"):
        print("\n[2/3] TFT 模型训练")
        train(args)

    if args.mode in ("all", "evaluate"):
        print("\n[3/3] TFT 模型评估")
        evaluate(args)


if __name__ == "__main__":
    main()
