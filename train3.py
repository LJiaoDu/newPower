#!/usr/bin/env python3
"""
太阳能电站多站点功率预测 - 训练脚本 v3

基于 train2.py 改进，新增多站点支持:
  - 使用 SolarTransformerMultiSite (model3.py)，含可学习站点嵌入
  - 训练数据为多站点混合，各站点独立归一化后拼接
  - 验证/测试时同时输出全局指标和各站点独立指标
  - 默认超参数已针对多站点场景调优 (更低 LR、更强正则化)

使用方式:
  # 完整流程: 预处理 + 训练 + 评估
  python train3.py --csv-paths site1.csv site2.csv site3.csv site4.csv site5.csv

  # 仅训练 (已有预处理数据)
  python train3.py --mode train

  # 仅评估
  python train3.py --mode evaluate

  # 自定义超参数
  python train3.py --csv-paths s*.csv --lr 1e-4 --dropout 0.2 --batch-size 128
"""

import argparse
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model3 import SolarTransformerMultiSite
from process3 import main as preprocess_main


# ============================================================
# 损失函数
# ============================================================

class ACC2Loss(nn.Module):
    """
    基于国标 ACC2 的损失函数
    L = mean( ((pred - target) / max(target, 0.2))^2 )
    分母 0.2 对应归一化容量的 20%，防止夜间小值引起数值爆炸
    """

    def __init__(self, cap_norm: float = 1.0):
        super().__init__()
        self.cap_norm = cap_norm

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred  = torch.clamp(pred, min=0.0)                         # 功率物理约束 ≥ 0
        denom = torch.clamp(target, min=0.2 * self.cap_norm)
        return torch.mean(((pred - target) / denom) ** 2)


class MixedLoss(nn.Module):
    def __init__(self, lambda_mse: float = 0.0,
                 lambda_acc2: float = 1.0, cap_norm: float = 1.0):
        super().__init__()
        self.lambda_mse  = lambda_mse
        self.lambda_acc2 = lambda_acc2
        self.mse_loss  = nn.MSELoss()
        self.acc2_loss = ACC2Loss(cap_norm=cap_norm)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l_mse  = self.mse_loss(pred, target)
        l_acc2 = self.acc2_loss(pred, target)
        return self.lambda_mse * l_mse + self.lambda_acc2 * l_acc2


# ============================================================
# 评估指标
# ============================================================

def calc_acc_mae(y_true: np.ndarray, y_pred: np.ndarray, cap: float = 1.0) -> float:
    """ACC1 (MAE-based): 1 - MAE / mean(y_true)，只统计功率 > 1% 容量的时刻"""
    mask = y_true > 0.01
    if mask.sum() == 0:
        return float("nan")
    y_t = y_true[mask] * cap
    y_p = y_pred[mask] * cap
    return max(0.0, 1.0 - np.mean(np.abs(y_t - y_p)) / (np.mean(y_t) + 1e-6))


def calc_rmse(y_true: np.ndarray, y_pred: np.ndarray, cap: float = 1.0) -> float:
    return float(np.sqrt(np.mean((y_true * cap - y_pred * cap) ** 2)))


def calc_mae(y_true: np.ndarray, y_pred: np.ndarray, cap: float = 1.0) -> float:
    return float(np.mean(np.abs(y_true * cap - y_pred * cap)))


def calc_acc2(y_true: np.ndarray, y_pred: np.ndarray, cap: float = 1.0) -> float:
    """ACC2 (国标): 1 - sqrt( mean( ((PM-PP) / max(PM, 0.2*Cap))^2 ) )"""
    p_m   = y_true.flatten() * cap
    p_p   = y_pred.flatten() * cap
    denom = np.maximum(p_m, 0.2 * cap)
    return max(0.0, 1.0 - float(np.sqrt(np.mean(((p_m - p_p) / denom) ** 2))))


# ============================================================
# 训练
# ============================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # ---- 加载数据 ----
    X_enc_train = np.load("X_enc_train.npy")
    X_dec_train = np.load("X_dec_train.npy")
    y_train     = np.load("y_train.npy")
    sid_train   = np.load("site_id_train.npy")

    X_enc_val   = np.load("X_enc_val.npy")
    X_dec_val   = np.load("X_dec_val.npy")
    y_val       = np.load("y_val.npy")
    sid_val     = np.load("site_id_val.npy")

    with open("site_configs.pkl", "rb") as f:
        site_configs = pickle.load(f)

    num_sites     = len(site_configs)
    enc_seq_len   = X_enc_train.shape[1]
    enc_feat_size = X_enc_train.shape[2]
    dec_seq_len   = X_dec_train.shape[1]
    dec_feat_size = X_dec_train.shape[2]

    print(f"站点数量: {num_sites}")
    print(f"训练集: enc={X_enc_train.shape}, dec={X_dec_train.shape}, y={y_train.shape}")
    print(f"验证集: enc={X_enc_val.shape},   dec={X_dec_val.shape},   y={y_val.shape}")
    print(f"特征维度: Encoder={enc_feat_size}, Decoder={dec_feat_size}")
    for sid, cfg in site_configs.items():
        n_tr = int((sid_train == sid).sum())
        n_va = int((sid_val   == sid).sum())
        print(f"  站点{sid} ({cfg['name']}): {cfg['cap']} MW | "
              f"train={n_tr}, val={n_va}")

    # ---- DataLoader (含 site_id) ----
    train_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_enc_train),
            torch.FloatTensor(X_dec_train),
            torch.FloatTensor(y_train),
            torch.LongTensor(sid_train),
        ),
        batch_size=args.batch_size, shuffle=True, drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_enc_val),
            torch.FloatTensor(X_dec_val),
            torch.FloatTensor(y_val),
            torch.LongTensor(sid_val),
        ),
        batch_size=args.batch_size, shuffle=False,
    )

    # ---- 模型 ----
    model = SolarTransformerMultiSite(
        num_sites=num_sites,
        site_embed_dim=args.site_embed_dim,
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

    # Xavier 初始化（跳过 site_embedding，已单独初始化为 N(0, 0.02)）
    for name, p in model.named_parameters():
        if p.dim() > 1 and "site_embedding" not in name:
            nn.init.xavier_uniform_(p, gain=0.5)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # ---- 优化器 + LR 调度 ----
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

    criterion = MixedLoss(
        lambda_mse=args.lambda_mse,
        lambda_acc2=args.lambda_acc2,
        cap_norm=1.0,
    )

    os.makedirs("solar_checkpoints", exist_ok=True)
    log_dir = os.path.join("runs", f"multisite_{time.strftime('%Y%m%d_%H%M%S')}")
    writer  = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard 日志: {log_dir}")

    best_val_loss    = float("inf")
    best_acc2        = -1.0
    patience_counter = 0

    print(f"\n{'=' * 70}")
    print(f"模型: SolarTransformerMultiSite "
          f"({num_sites} 站点, site_embed={args.site_embed_dim})")
    print(f"Loss: λ_mse={args.lambda_mse} * MSE + λ_acc2={args.lambda_acc2} * ACC2Loss")
    print(f"Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"Dropout: {args.dropout} | WeightDecay: {args.weight_decay} | "
          f"Patience: {args.patience}")
    print(f"{'=' * 70}\n")

    epoch_bar = tqdm(range(args.epochs), desc="Training", unit="epoch")

    for epoch in epoch_bar:

        # ---- 训练阶段 ----
        model.train()
        tr_mixed = tr_mse = tr_acc2l = 0.0

        tr_bar = tqdm(train_loader,
                      desc=f"Epoch {epoch+1}/{args.epochs} [Train]",
                      leave=False, unit="batch")
        for enc_b, dec_b, y_b, sid_b in tr_bar:
            enc_b = enc_b.to(device)
            dec_b = dec_b.to(device)
            y_b   = y_b.to(device)
            sid_b = sid_b.to(device)

            optimizer.zero_grad()
            pred   = model(enc_b, dec_b, sid_b)
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
                               acc2l=f"{l_acc2.item():.5f}")

        n_tr      = len(train_loader)
        tr_mixed /= n_tr
        tr_mse   /= n_tr
        tr_acc2l /= n_tr

        # ---- 验证阶段 ----
        model.eval()
        va_mixed = va_mse = va_acc2l = 0.0
        all_preds, all_targets, all_sids = [], [], []

        va_bar = tqdm(val_loader,
                      desc=f"Epoch {epoch+1}/{args.epochs} [Val]",
                      leave=False, unit="batch")
        with torch.no_grad():
            for enc_b, dec_b, y_b, sid_b in va_bar:
                enc_b = enc_b.to(device)
                dec_b = dec_b.to(device)
                y_b   = y_b.to(device)
                sid_b = sid_b.to(device)

                pred   = model(enc_b, dec_b, sid_b)
                l_mse  = criterion.mse_loss(pred, y_b)
                l_acc2 = criterion.acc2_loss(pred, y_b)
                loss   = args.lambda_mse * l_mse + args.lambda_acc2 * l_acc2

                va_mixed += loss.item()
                va_mse   += l_mse.item()
                va_acc2l += l_acc2.item()

                all_preds.append(pred.cpu().numpy())
                all_targets.append(y_b.cpu().numpy())
                all_sids.append(sid_b.cpu().numpy())

        n_va      = len(val_loader)
        va_mixed /= n_va
        va_mse   /= n_va
        va_acc2l /= n_va

        all_preds   = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        all_sids    = np.concatenate(all_sids)

        # 全局指标 (cap=1.0: ACC2 公式 scale-invariant，RMSE/MAE 为归一化值)
        acc1 = calc_acc_mae(all_targets, all_preds, cap=1.0)
        acc2 = calc_acc2(all_targets, all_preds, cap=1.0)
        rmse = calc_rmse(all_targets, all_preds, cap=1.0)
        mae  = calc_mae(all_targets, all_preds, cap=1.0)
        lr   = optimizer.param_groups[0]["lr"]

        # 各站点 ACC2
        site_acc2 = {}
        for sid in range(num_sites):
            mask = all_sids == sid
            if mask.sum() > 0:
                cap = site_configs[sid]["cap"]
                site_acc2[sid] = calc_acc2(all_targets[mask], all_preds[mask], cap)
                writer.add_scalar(f"SiteACC2/site_{sid}", site_acc2[sid], epoch + 1)

        # TensorBoard
        writer.add_scalars("Loss/Mixed",   {"Train": tr_mixed,  "Val": va_mixed},  epoch + 1)
        writer.add_scalars("Loss/MSE",     {"Train": tr_mse,    "Val": va_mse},    epoch + 1)
        writer.add_scalars("Loss/ACC2Loss",{"Train": tr_acc2l,  "Val": va_acc2l},  epoch + 1)
        writer.add_scalar("Accuracy/ACC1", acc1, epoch + 1)
        writer.add_scalar("Accuracy/ACC2", acc2, epoch + 1)
        writer.add_scalar("Error/RMSE",    rmse, epoch + 1)
        writer.add_scalar("Error/MAE",     mae,  epoch + 1)
        writer.add_scalar("LearningRate",  lr,   epoch + 1)

        site_acc2_str = " | ".join(
            f"S{i}={v:.4f}" for i, v in site_acc2.items()
        )
        epoch_bar.set_postfix(
            tr=f"{tr_mixed:.5f}", va=f"{va_mixed:.5f}", ACC2=f"{acc2:.4f}"
        )
        tqdm.write(
            f"Epoch {epoch+1:3d}/{args.epochs} | LR: {lr:.6f} | "
            f"Train(Mix/MSE/ACC2L): {tr_mixed:.6f}/{tr_mse:.6f}/{tr_acc2l:.6f} | "
            f"Val(Mix/MSE/ACC2L): {va_mixed:.6f}/{va_mse:.6f}/{va_acc2l:.6f} | "
            f"ACC1: {acc1:.4f} | ACC2: {acc2:.4f} | "
            f"RMSE: {rmse:.4f} | MAE: {mae:.4f}\n"
            f"  [各站点 ACC2] {site_acc2_str}"
        )

        # ---- Checkpoint ----
        ckpt_base = {
            "epoch":            epoch,
            "model_state_dict": model.state_dict(),
            "val_mixed":        va_mixed,
            "acc1":             acc1,
            "acc2":             acc2,
            "site_acc2":        site_acc2,
            "site_configs":     site_configs,
            "args":             vars(args),
        }

        if va_mixed < best_val_loss:
            best_val_loss = va_mixed
            torch.save(ckpt_base,
                       "solar_checkpoints/best_multisite_mixedloss.pth")
            tqdm.write(f"  -> 保存 MixedLoss 最佳 (Val MixedLoss: {va_mixed:.6f})")

        improved = acc2 > best_acc2
        if improved:
            best_acc2 = acc2
            torch.save(ckpt_base,
                       "solar_checkpoints/best_multisite_acc2.pth")
            tqdm.write(f"  -> 保存 ACC2 最佳 (ACC2: {acc2:.6f})")
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
          f"-> solar_checkpoints/best_multisite_mixedloss.pth")
    print(f"  ACC2     最佳: {best_acc2:.6f}  "
          f"-> solar_checkpoints/best_multisite_acc2.pth")


# ============================================================
# 评估
# ============================================================

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_enc_test = np.load("X_enc_test.npy")
    X_dec_test = np.load("X_dec_test.npy")
    y_test     = np.load("y_test.npy")
    sid_test   = np.load("site_id_test.npy")

    with open("site_configs.pkl", "rb") as f:
        site_configs = pickle.load(f)
    num_sites = len(site_configs)

    enc_feat_size = X_enc_test.shape[2]
    dec_feat_size = X_dec_test.shape[2]
    enc_seq_len   = X_enc_test.shape[1]
    dec_seq_len   = X_dec_test.shape[1]

    model = SolarTransformerMultiSite(
        num_sites=num_sites,
        site_embed_dim=args.site_embed_dim,
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

    ckpt_path = "solar_checkpoints/best_multisite_acc2.pth"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"加载模型: {ckpt_path}")
    print(f"  epoch={checkpoint['epoch']+1}, acc2={checkpoint['acc2']:.6f}")

    test_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_enc_test),
            torch.FloatTensor(X_dec_test),
            torch.FloatTensor(y_test),
            torch.LongTensor(sid_test),
        ),
        batch_size=args.batch_size, shuffle=False,
    )

    all_preds, all_targets, all_sids = [], [], []
    with torch.no_grad():
        for enc_b, dec_b, y_b, sid_b in tqdm(test_loader, desc="Testing"):
            enc_b = enc_b.to(device)
            dec_b = dec_b.to(device)
            sid_b = sid_b.to(device)
            pred  = model(enc_b, dec_b, sid_b)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_b.numpy())
            all_sids.append(sid_b.cpu().numpy())

    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_sids    = np.concatenate(all_sids)

    print(f"\n{'=' * 70}")
    print(f"测试集评估结果 (多站点 SiteEmbedding + RoPE)")
    print(f"{'=' * 70}")

    # 全局指标
    print(f"[全局]")
    print(f"  ACC1 (MAE-based): {calc_acc_mae(all_targets, all_preds, 1.0):.4f}")
    print(f"  ACC2 (国标):      {calc_acc2(all_targets, all_preds, 1.0):.4f}")
    print(f"  RMSE (归一化):    {calc_rmse(all_targets, all_preds, 1.0):.4f}")
    print(f"  MAE  (归一化):    {calc_mae(all_targets, all_preds, 1.0):.4f}")

    # 各站点指标（使用实际 MW 容量）
    print(f"\n[各站点]")
    for sid in range(num_sites):
        mask = all_sids == sid
        if mask.sum() == 0:
            continue
        cap  = site_configs[sid]["cap"]
        name = site_configs[sid]["name"]
        yt   = all_targets[mask]
        yp   = all_preds[mask]
        print(f"  站点{sid} ({name}, {cap} MW): "
              f"ACC1={calc_acc_mae(yt, yp, cap):.4f}, "
              f"ACC2={calc_acc2(yt, yp, cap):.4f}, "
              f"RMSE={calc_rmse(yt, yp, cap):.2f} MW, "
              f"MAE={calc_mae(yt, yp, cap):.2f} MW")

    # 按预测时间范围分析（全局）
    print(f"\n[按预测时间范围 (全局)]")
    horizons = [
        (0,  4,  "0-1h"),
        (4,  8,  "1-2h"),
        (8,  12, "2-3h"),
        (12, 16, "3-4h"),
    ]
    for start, end, label in horizons:
        yt = all_targets[:, start:end]
        yp = all_preds[:, start:end]
        print(f"  {label}: ACC2={calc_acc2(yt, yp, 1.0):.4f}, "
              f"RMSE={calc_rmse(yt, yp, 1.0):.4f}, "
              f"MAE={calc_mae(yt, yp, 1.0):.4f}")

    print(f"{'=' * 70}")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="太阳能多站点功率预测 (SiteEmbedding + RoPE)"
    )
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["all", "preprocess", "train", "evaluate"],
    )
    parser.add_argument(
        "--csv-paths", nargs="+", default=[],
        help="各站点 CSV 路径（mode=all/preprocess 时必须提供）"
    )

    # 训练超参数（默认值已针对多站点优化）
    parser.add_argument("--epochs",        type=int,   default=100)
    parser.add_argument("--batch-size",    type=int,   default=128,
                        help="多站点数据量约 5x，建议 128 或 256")
    parser.add_argument("--lr",            type=float, default=1e-4,
                        help="峰值学习率（比单站点更低以避免训练初期震荡）")
    parser.add_argument("--weight-decay",  type=float, default=0.05)
    parser.add_argument("--patience",      type=int,   default=30)
    parser.add_argument("--warmup-epochs", type=int,   default=10)
    parser.add_argument("--grad-clip",     type=float, default=1.0)

    # 模型结构
    parser.add_argument("--d-model",             type=int,   default=256)
    parser.add_argument("--nhead",               type=int,   default=8)
    parser.add_argument("--num-encoder-layers",  type=int,   default=4)
    parser.add_argument("--num-decoder-layers",  type=int,   default=4)
    parser.add_argument("--dropout",             type=float, default=0.2)
    parser.add_argument("--site-embed-dim",      type=int,   default=8,
                        help="站点嵌入向量维度（推荐 4~16）")

    # 损失权重
    parser.add_argument("--lambda-mse",  type=float, default=0.0)
    parser.add_argument("--lambda-acc2", type=float, default=1.0)

    args = parser.parse_args()

    if args.mode in ["all", "preprocess"]:
        if not args.csv_paths:
            raise ValueError("--mode all/preprocess 时必须提供 --csv-paths")
        print("\n[1/3] 多站点数据预处理")
        preprocess_main(args.csv_paths)

    if args.mode in ["all", "train"]:
        print("\n[2/3] 多站点模型训练")
        train(args)

    if args.mode in ["all", "evaluate"]:
        print("\n[3/3] 模型评估")
        evaluate(args)


if __name__ == "__main__":
    main()
