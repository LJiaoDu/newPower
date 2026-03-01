#!/usr/bin/env python3
"""
太阳能电站功率预测 - RoPE ED Transformer 训练脚本 (training_data.csv 版本)

对应关系:
  数据预处理: solar_preprocess_2.py  -> X_enc_*.npy, X_dec_*.npy, y_*.npy
  模型定义:   rope_model_2.py        -> SolarTransformerRoPE
  本脚本:     rope_2.py              -> 训练 + 验证 + 评估

与 rope_1.py 的主要差异:
  1. 数据源: training_data.csv (5分钟粒度, 无气象特征)
  2. 序列长度: enc=288 (24h), dec=48 (4h)
  3. 特征维度: Encoder=7, Decoder=6
  4. 归一化参数: norm_params["power"]["max_power"] (无 cap 列)
  5. 逐步时刻标签: 每步 5 分钟
  6. 默认 CSV: training_data.csv
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

from rope_model_2 import SolarTransformerRoPE
from solar_preprocess_2 import main as preprocess_main

from datetime import datetime
import atexit


# =========================================================
# 仅在程序结束时保存"关键终端输出"（避免 tqdm 进度条刷屏写入）
# =========================================================

LOG_BUFFER = []

def log_print(*args, **kwargs):
    """替代 print：正常打印到终端，同时把文本缓存起来，程序结束后写入 txt"""
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    message = sep.join(str(a) for a in args) + end

    print(*args, **kwargs)
    LOG_BUFFER.append(message.rstrip("\n"))

def log_tqdm_write(message: str):
    """替代 tqdm.write：写到终端（不破坏进度条），同时缓存"""
    tqdm.write(message)
    LOG_BUFFER.append(str(message))

def save_log_to_file():
    """程序退出时，把 LOG_BUFFER 写入文件"""
    try:
        os.makedirs("solar_logs", exist_ok=True)
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"solar_logs/final_log_rope2_{time_str}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(LOG_BUFFER) + "\n")
        print(f"\n最终日志已保存至: {filename}")
    except Exception as e:
        print(f"\n[WARN] 保存最终日志失败: {e}")

atexit.register(save_log_to_file)


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
    """混合 Loss = λ_mse * MSE + λ_acc2 * ACC2_Loss"""
    def __init__(self, lambda_mse=1.0, lambda_acc2=1.0, cap_norm=1.0):
        super().__init__()
        self.lambda_mse  = lambda_mse
        self.lambda_acc2 = lambda_acc2
        self.mse_loss    = nn.MSELoss()
        self.acc2_loss   = ACC2Loss(cap_norm=cap_norm)

    def forward(self, pred, target):
        l_mse  = self.mse_loss(pred, target)
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
    p_m   = y_true.flatten() * cap
    p_p   = y_pred.flatten() * cap
    denom = np.maximum(p_m, 0.2 * cap)
    return max(0.0, 1.0 - np.sqrt(np.mean(((p_m - p_p) / denom) ** 2)))


# ============== 训练 ==============

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print(f"设备: {device}")

    # 加载数据
    X_enc_train = np.load("X_enc_train.npy")
    X_dec_train = np.load("X_dec_train.npy")
    y_train     = np.load("y_train.npy")
    X_enc_val   = np.load("X_enc_val.npy")
    X_dec_val   = np.load("X_dec_val.npy")
    y_val       = np.load("y_val.npy")

    with open("norm_params.pkl", "rb") as f:
        norm_params = pickle.load(f)
    # training_data.csv 使用 max_power 而非 cap
    cap = norm_params["power"]["max_power"]

    log_print(f"训练集: enc={X_enc_train.shape}, dec={X_dec_train.shape}, y={y_train.shape}")
    log_print(f"验证集: enc={X_enc_val.shape}, dec={X_dec_val.shape}, y={y_val.shape}")
    log_print(f"最大功率 (归一化基准): {cap:.2f} W")

    enc_seq_len   = X_enc_train.shape[1]   # 288
    enc_feat_size = X_enc_train.shape[2]   # 7
    dec_seq_len   = X_dec_train.shape[1]   # 48
    dec_feat_size = X_dec_train.shape[2]   # 6

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

    # 模型
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
    log_print(f"模型参数量: {total_params:,}")

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6)
    scheduler    = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[args.warmup_epochs])

    # Loss
    criterion = MixedLoss(
        lambda_mse=args.lambda_mse,
        lambda_acc2=args.lambda_acc2,
        cap_norm=1.0,
    )

    os.makedirs("solar_checkpoints", exist_ok=True)

    # TensorBoard
    log_dir = os.path.join("runs", f"rope2_{time.strftime('%Y%m%d_%H%M%S')}")
    writer  = SummaryWriter(log_dir=log_dir)
    log_print(f"TensorBoard 日志: {log_dir}")

    best_acc2        = -1.0
    patience_counter = 0

    log_print(f"\n{'='*70}")
    log_print(f"开始训练 | Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    log_print(f"Loss: λ_mse={args.lambda_mse} * MSE + λ_acc2={args.lambda_acc2} * ACC2Loss")
    log_print(f"模型: SolarTransformerRoPE | d_model={args.d_model}, heads={args.nhead}, "
              f"enc_layers={args.num_encoder_layers}, dec_layers={args.num_decoder_layers}")
    log_print(f"正则: dropout={args.dropout}, weight_decay={args.weight_decay}")
    log_print(f"输入噪声: std={args.noise_std}")
    log_print(f"保存标准: ACC2 最高")
    log_print(f"{'='*70}\n")

    epoch_bar = tqdm(range(args.epochs), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        # --- 训练 ---
        model.train()
        train_loss = 0.0
        train_bar  = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]",
                          leave=False, unit="batch")
        for batch_enc, batch_dec, batch_y in train_bar:
            batch_enc = batch_enc.to(device)
            batch_dec = batch_dec.to(device)
            batch_y   = batch_y.to(device)

            if args.noise_std > 0:
                batch_enc = batch_enc + torch.randn_like(batch_enc) * args.noise_std
                batch_dec = batch_dec + torch.randn_like(batch_dec) * args.noise_std

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
                batch_y   = batch_y.to(device)

                pred      = model(batch_enc, batch_dec)
                val_loss += criterion(pred, batch_y).item()
                all_preds.append(pred.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
        val_loss /= len(val_loader)

        all_preds   = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        acc1 = calc_acc_mae(all_targets, all_preds, cap)
        acc2 = calc_acc2(all_targets, all_preds, cap)
        rmse = calc_rmse(all_targets, all_preds, cap)
        mae  = calc_mae(all_targets, all_preds, cap)
        lr   = optimizer.param_groups[0]["lr"]

        # TensorBoard
        writer.add_scalars("Loss",              {"Train": train_loss, "Val": val_loss}, epoch + 1)
        writer.add_scalar("Accuracy/ACC1",      acc1,  epoch + 1)
        writer.add_scalar("Accuracy/ACC2",      acc2,  epoch + 1)
        writer.add_scalar("Error/RMSE_W",       rmse,  epoch + 1)
        writer.add_scalar("Error/MAE_W",        mae,   epoch + 1)
        writer.add_scalar("LearningRate",       lr,    epoch + 1)

        epoch_bar.set_postfix(train=f"{train_loss:.5f}", val=f"{val_loss:.5f}",
                              ACC1=f"{acc1:.4f}", ACC2=f"{acc2:.4f}")

        log_tqdm_write(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"LR: {lr:.6f} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f} | "
            f"ACC1: {acc1:.4f} | "
            f"ACC2: {acc2:.4f} | "
            f"RMSE: {rmse:.2f} W | "
            f"MAE: {mae:.2f} W"
        )

        # 保存最佳模型 (按 ACC2 最高)
        if acc2 > best_acc2:
            best_acc2        = acc2
            patience_counter = 0
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_loss":         val_loss,
                "acc1":             acc1,
                "acc2":             acc2,
                "norm_params":      norm_params,
                "args":             vars(args),
            }, "solar_checkpoints/best_model_rope2.pth")
            log_tqdm_write(f"  -> 保存最佳模型 (ACC2: {acc2:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log_tqdm_write(f"\nEarly stopping: {args.patience} epochs ACC2 无改善")
                break

        scheduler.step()

    writer.close()
    log_print(f"\n训练完成! 最佳 ACC2: {best_acc2:.4f}")


# ============== 评估 ==============

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_enc_test = np.load("X_enc_test.npy")
    X_dec_test = np.load("X_dec_test.npy")
    y_test     = np.load("y_test.npy")

    with open("norm_params.pkl", "rb") as f:
        norm_params = pickle.load(f)
    cap = norm_params["power"]["max_power"]

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

    checkpoint = torch.load("solar_checkpoints/best_model_rope2.pth",
                            map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    log_print(f"加载模型: epoch={checkpoint['epoch']+1}, "
              f"val_loss={checkpoint['val_loss']:.6f}, "
              f"ACC2={checkpoint['acc2']:.4f}")

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
            pred      = model(batch_enc, batch_dec)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(batch_y.numpy())

    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    acc1 = calc_acc_mae(all_targets, all_preds, cap)
    acc2 = calc_acc2(all_targets, all_preds, cap)
    rmse = calc_rmse(all_targets, all_preds, cap)
    mae  = calc_mae(all_targets, all_preds, cap)

    log_print(f"\n{'='*60}")
    log_print(f"测试集评估结果")
    log_print(f"{'='*60}")
    log_print(f"  ACC1 (MAE-based): {acc1:.4f} ({acc1*100:.2f}%)")
    log_print(f"  ACC2 (国标):      {acc2:.4f} ({acc2*100:.2f}%)")
    log_print(f"  RMSE:             {rmse:.2f} W")
    log_print(f"  MAE:              {mae:.2f} W")

    # 逐步指标 (每步 5 分钟)
    num_steps = all_targets.shape[1]
    log_print(f"\n按预测步 (每步5分钟, 共{num_steps}步, 预测时域={num_steps*5//60}h):")
    log_print(f"  {'步':>3s}  {'时刻':>7s}  {'ACC1':>7s}  {'ACC2':>7s}  {'RMSE(W)':>8s}  {'MAE(W)':>7s}")
    log_print(f"  {'-'*3}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*7}")
    for step in range(num_steps):
        yt         = all_targets[:, step:step+1]
        yp         = all_preds[:,   step:step+1]
        s_acc1     = calc_acc_mae(yt, yp, cap)
        s_acc2     = calc_acc2(yt, yp, cap)
        s_rmse     = calc_rmse(yt, yp, cap)
        s_mae      = calc_mae(yt, yp, cap)
        minutes    = (step + 1) * 5           # 5分钟粒度
        time_label = f"+{minutes}min"
        log_print(f"  {step+1:3d}  {time_label:>7s}  {s_acc1:.4f}  {s_acc2:.4f}  {s_rmse:8.2f}  {s_mae:7.2f}")

    log_print(f"{'='*60}")


# ============== 主入口 ==============

def main():
    parser = argparse.ArgumentParser(description="太阳能电站功率预测 (RoPE, training_data.csv 版本)")

    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "preprocess", "train", "evaluate"])
    parser.add_argument("--csv-path", type=str,
                        default="training_data.csv",
                        help="training_data.csv 文件路径")

    parser.add_argument("--epochs",        type=int,   default=100,  help="最大训练轮次")
    parser.add_argument("--batch-size",    type=int,   default=64,   help="批大小")
    parser.add_argument("--lr",            type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight-decay",  type=float, default=0.05, help="L2正则化系数")
    parser.add_argument("--patience",      type=int,   default=10,   help="Early Stopping 耐心值 (基于 ACC2)")
    parser.add_argument("--warmup-epochs", type=int,   default=5,    help="Warmup 轮次")
    parser.add_argument("--grad-clip",     type=float, default=1.0,  help="梯度裁剪阈值")

    parser.add_argument("--d-model",           type=int,   default=128, help="Transformer 隐藏层维度")
    parser.add_argument("--nhead",             type=int,   default=4,   help="注意力头数")
    parser.add_argument("--num-encoder-layers",type=int,   default=2,   help="Encoder 层数")
    parser.add_argument("--num-decoder-layers",type=int,   default=2,   help="Decoder 层数")
    parser.add_argument("--dropout",           type=float, default=0.2, help="Dropout 比率")

    parser.add_argument("--lambda-mse",  type=float, default=0.5, help="MSE Loss 权重")
    parser.add_argument("--lambda-acc2", type=float, default=1.0, help="ACC2 Loss 权重")

    parser.add_argument("--noise-std", type=float, default=0.01,
                        help="训练时输入高斯噪声标准差 (0=关闭)")

    args = parser.parse_args()

    log_print("=" * 80)
    log_print(f"Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"Args: {vars(args)}")
    log_print("=" * 80)

    if args.mode in ["all", "preprocess"]:
        log_print("\n[1/3] 数据预处理")
        preprocess_main(csv_path=args.csv_path)

    if args.mode in ["all", "train"]:
        log_print("\n[2/3] 模型训练")
        train(args)

    if args.mode in ["all", "evaluate"]:
        log_print("\n[3/3] 模型评估")
        evaluate(args)


if __name__ == "__main__":
    main()
