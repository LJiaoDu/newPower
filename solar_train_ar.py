#!/usr/bin/env python3
"""
太阳能电站功率预测 - 自回归训练与评估脚本

训练: Teacher Forcing 滑动窗口, 每个样本输出 16 步, loss 算在 16 步上
评估: 自回归滑动窗口推理 (用自己的预测值填充), 输出 16 步

数据格式:
  训练: X_full [N, 112, 13] (96步历史 + 16步未来, 含真实功率) → y [N, 16]
  评估: X_enc [N, 96, 13] + X_dec [N, 16, 12] → y [N, 16]

使用方式:
  python solar_train_ar.py                     # 预处理 + 训练 + 评估
  python solar_train_ar.py --mode train        # 仅训练
  python solar_train_ar.py --mode evaluate     # 仅评估
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import os
import argparse
import pickle
import time

from solar_model_ar import SolarTransformerAR
from solar_preprocess import load_and_clean, extract_features


# ============== 数据准备 ==============

def create_ar_sequences(enc_features, dec_features, targets,
                        in_steps=96, out_steps=16, daytime_only=False):
    """
    创建自回归训练 + 评估序列

    Args:
      daytime_only: 若为True, 只保留目标中有发电(非全零)的样本

    返回:
      X_full: [N, 112, 13]  完整序列 (训练用, 含真实功率, teacher forcing 滑窗)
      X_enc:  [N, 96, 13]   历史序列 (推理用, 自回归起点)
      X_dec:  [N, 16, 12]   未来协变量 (推理用, 气象+时间, 无功率)
      y:      [N, 16]       预测目标
    """
    total_len = in_steps + out_steps
    X_full_list, X_enc_list, X_dec_list, y_list = [], [], [], []
    n_skipped = 0

    for i in range(len(enc_features) - total_len + 1):
        target = targets[i + in_steps : i + total_len]

        # 过滤: 目标全零(纯夜间)的样本跳过
        if daytime_only and (target < 0.001).all():
            n_skipped += 1
            continue

        X_full_list.append(enc_features[i : i + total_len])       # [112, 13]
        X_enc_list.append(enc_features[i : i + in_steps])         # [96, 13]
        X_dec_list.append(dec_features[i + in_steps : i + total_len])  # [16, 12]
        y_list.append(target)                                      # [16]

    X_full = np.array(X_full_list, dtype=np.float32)
    X_enc = np.array(X_enc_list, dtype=np.float32)
    X_dec = np.array(X_dec_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    print(f"  X_full={X_full.shape}, X_enc={X_enc.shape}, "
          f"X_dec={X_dec.shape}, y={y.shape}")
    if n_skipped > 0:
        print(f"  过滤掉 {n_skipped} 个纯夜间样本")
    return X_full, X_enc, X_dec, y


def preprocess(csv_path, in_steps=96, out_steps=16,
               train_ratio=0.7, val_ratio=0.15, output_dir="."):
    """预处理: 生成训练 + 评估数据"""
    print("=" * 60)
    print("自回归模型数据预处理")
    print("=" * 60)

    df = load_and_clean(csv_path)
    enc_features, dec_features, targets, norm_params, enc_names, dec_names = \
        extract_features(df)

    n = len(enc_features)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    print("\n训练集 (仅白天):")
    Xf_train, Xe_train, Xd_train, y_train = create_ar_sequences(
        enc_features[:train_end], dec_features[:train_end],
        targets[:train_end], in_steps, out_steps, daytime_only=True)

    print("验证集 (仅白天):")
    Xf_val, Xe_val, Xd_val, y_val = create_ar_sequences(
        enc_features[train_end:val_end], dec_features[train_end:val_end],
        targets[train_end:val_end], in_steps, out_steps, daytime_only=True)

    print("测试集:")
    Xf_test, Xe_test, Xd_test, y_test = create_ar_sequences(
        enc_features[val_end:], dec_features[val_end:],
        targets[val_end:], in_steps, out_steps)

    save = lambda name, arr: np.save(os.path.join(output_dir, f"{name}.npy"), arr)
    for prefix, (xf, xe, xd, y) in [
        ("ar_train", (Xf_train, Xe_train, Xd_train, y_train)),
        ("ar_val",   (Xf_val,   Xe_val,   Xd_val,   y_val)),
        ("ar_test",  (Xf_test,  Xe_test,  Xd_test,  y_test)),
    ]:
        save(f"{prefix}_Xfull", xf)
        save(f"{prefix}_Xenc", xe)
        save(f"{prefix}_Xdec", xd)
        save(f"{prefix}_y", y)

    with open(os.path.join(output_dir, "norm_params.pkl"), "wb") as f:
        pickle.dump(norm_params, f)

    print(f"\n预处理完成! 文件保存到: {output_dir}")


# ============== 评估指标 ==============

def calc_acc_mae(y_true, y_pred, cap=1.0):
    """ACC1 (MAE-based)"""
    mask = y_true > 0.01
    if mask.sum() == 0:
        return float("nan")
    y_t, y_p = y_true[mask] * cap, y_pred[mask] * cap
    return max(0.0, 1.0 - np.mean(np.abs(y_t - y_p)) / (np.mean(y_t) + 1e-6))


def calc_acc2(y_true, y_pred, cap=1.0):
    """ACC2 (国标)"""
    p_m = y_true.flatten() * cap
    p_p = y_pred.flatten() * cap
    denom = np.maximum(p_m, 0.2 * cap)
    return max(0.0, 1.0 - np.sqrt(np.mean(((p_m - p_p) / denom) ** 2)))


def calc_rmse(y_true, y_pred, cap=1.0):
    return np.sqrt(np.mean((y_true * cap - y_pred * cap) ** 2))


def calc_mae(y_true, y_pred, cap=1.0):
    return np.mean(np.abs(y_true * cap - y_pred * cap))


def eval_metrics(y_true, y_pred, cap):
    return {
        "acc1": calc_acc_mae(y_true, y_pred, cap),
        "acc2": calc_acc2(y_true, y_pred, cap),
        "rmse": calc_rmse(y_true, y_pred, cap),
        "mae":  calc_mae(y_true, y_pred, cap),
    }


# ============== 自回归多步推理评估 ==============

def ar_evaluate_multistep(model, X_enc, X_dec, y_true, cap, device,
                          batch_size=256, out_steps=16, desc="评估"):
    """自回归滑动窗口推理 → 评估16步"""
    model.eval()
    all_preds = []
    n_batches = (len(X_enc) + batch_size - 1) // batch_size

    for start in tqdm(range(0, len(X_enc), batch_size), total=n_batches,
                      desc=desc, leave=False, unit="batch"):
        end = min(start + batch_size, len(X_enc))
        x_enc = torch.FloatTensor(X_enc[start:end]).to(device)
        x_dec = torch.FloatTensor(X_dec[start:end]).to(device)
        preds = model.predict_sequence(x_enc, x_dec, out_steps=out_steps)
        all_preds.append(preds.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    return all_preds, eval_metrics(y_true, all_preds, cap)


# ============== 训练 ==============

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 训练数据: X_full [N, 112, 13] + y [N, 16]
    Xf_train = np.load("ar_train_Xfull.npy")
    y_train = np.load("ar_train_y.npy")

    # 验证数据: Teacher Forcing 验证 loss + 自回归推理评估
    Xf_val = np.load("ar_val_Xfull.npy")
    y_val = np.load("ar_val_y.npy")
    Xe_val = np.load("ar_val_Xenc.npy")
    Xd_val = np.load("ar_val_Xdec.npy")

    with open("norm_params.pkl", "rb") as f:
        norm_params = pickle.load(f)
    cap = norm_params["power"]["cap"]

    in_steps = Xf_train.shape[1] - args.out_steps  # 96
    feat_size = Xf_train.shape[2]  # 13

    print(f"训练集: Xfull={Xf_train.shape}, y={y_train.shape}")
    print(f"验证集: Xfull={Xf_val.shape}, Xenc={Xe_val.shape}, Xdec={Xd_val.shape}")

    # DataLoader: 训练用 X_full + y
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(Xf_train), torch.FloatTensor(y_train)),
        batch_size=args.batch_size, shuffle=True,
    )
    val_tf_loader = DataLoader(
        TensorDataset(torch.FloatTensor(Xf_val), torch.FloatTensor(y_val)),
        batch_size=args.batch_size, shuffle=False,
    )

    # 模型
    model = SolarTransformerAR(
        feat_size=feat_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.d_model * 4,
        dropout=args.dropout,
        seq_len=in_steps,
    ).to(device)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p, gain=0.5)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                            total_iters=args.warmup_epochs)
    cosine_sched = CosineAnnealingLR(optimizer,
                                      T_max=args.epochs - args.warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched],
                             milestones=[args.warmup_epochs])

    criterion = nn.HuberLoss(delta=args.huber_delta)

    os.makedirs("solar_checkpoints", exist_ok=True)
    log_dir = os.path.join("runs", "ar_" + time.strftime("%Y%m%d_%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n{'='*70}")
    print(f"自回归训练 (Teacher Forcing, 16步 loss)")
    print(f"Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"模型: d={args.d_model}, heads={args.nhead}, layers={args.num_layers}")
    print(f"{'='*70}\n")

    epoch_bar = tqdm(range(args.epochs), desc="训练进度", unit="epoch")
    for epoch in epoch_bar:
        # --- 训练: Teacher Forcing 16步, loss 算在 [B, 16] 上 ---
        model.train()
        train_loss = 0.0
        batch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}", leave=False,
                         unit="batch")
        for batch_xf, batch_y in batch_bar:
            batch_xf = batch_xf.to(device)  # [B, 112, 13]
            batch_y = batch_y.to(device)     # [B, 16]

            optimizer.zero_grad()
            pred = model(batch_xf, out_steps=args.out_steps)  # [B, 16]
            loss = criterion(pred, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            train_loss += loss.item()
            batch_bar.set_postfix(loss=f"{loss.item():.6f}")
        train_loss /= len(train_loader)

        # --- 验证: Teacher Forcing loss ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_xf, batch_y in val_tf_loader:
                batch_xf = batch_xf.to(device)
                batch_y = batch_y.to(device)
                pred = model(batch_xf, out_steps=args.out_steps)
                val_loss += criterion(pred, batch_y).item()
        val_loss /= len(val_tf_loader)

        epoch_bar.set_postfix(train=f"{train_loss:.6f}", val=f"{val_loss:.6f}")

        lr = optimizer.param_groups[0]["lr"]
        writer.add_scalars("Loss", {"Train": train_loss, "Val": val_loss}, epoch + 1)
        writer.add_scalar("LearningRate", lr, epoch + 1)

        # --- 每个 epoch 做自回归推理评估 ---
        if True:
            _, metrics = ar_evaluate_multistep(
                model, Xe_val, Xd_val, y_val, cap, device,
                batch_size=args.batch_size, out_steps=args.out_steps,
                desc="验证集AR推理")
            writer.add_scalar("AR/ACC1", metrics["acc1"], epoch + 1)
            writer.add_scalar("AR/ACC2", metrics["acc2"], epoch + 1)
            writer.add_scalar("AR/RMSE", metrics["rmse"], epoch + 1)

            print(
                f"Epoch {epoch+1:3d}/{args.epochs} | LR: {lr:.6f} | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                f"AR-ACC1: {metrics['acc1']:.4f} | AR-ACC2: {metrics['acc2']:.4f} | "
                f"AR-RMSE: {metrics['rmse']:.2f} MW"
            )

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "norm_params": norm_params,
                "args": vars(args),
            }, "solar_checkpoints/best_model_ar.pth")
            print(f"  -> 保存最佳模型 (Val Loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping: {args.patience} epochs 无改善")
                break

        scheduler.step()

    writer.close()
    print(f"\n训练完成! 最佳验证损失: {best_val_loss:.6f}")


# ============== 测试集评估 ==============

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xe_test = np.load("ar_test_Xenc.npy")
    Xd_test = np.load("ar_test_Xdec.npy")
    y_test = np.load("ar_test_y.npy")

    with open("norm_params.pkl", "rb") as f:
        norm_params = pickle.load(f)
    cap = norm_params["power"]["cap"]

    feat_size = Xe_test.shape[2]
    seq_len = Xe_test.shape[1]

    model = SolarTransformerAR(
        feat_size=feat_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.d_model * 4,
        dropout=args.dropout,
        seq_len=seq_len,
    ).to(device)

    checkpoint = torch.load("solar_checkpoints/best_model_ar.pth",
                            map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"加载模型: epoch={checkpoint['epoch']+1}, "
          f"val_loss={checkpoint['val_loss']:.6f}")

    all_preds, metrics = ar_evaluate_multistep(
        model, Xe_test, Xd_test, y_test, cap, device,
        batch_size=args.batch_size, out_steps=args.out_steps,
        desc="测试集AR推理")

    print(f"\n{'='*60}")
    print(f"测试集评估 (自回归滑动窗口推理)")
    print(f"{'='*60}")
    print(f"  ACC1 (MAE-based): {metrics['acc1']:.4f} ({metrics['acc1']*100:.2f}%)")
    print(f"  ACC2 (国标):      {metrics['acc2']:.4f} ({metrics['acc2']*100:.2f}%)")
    print(f"  RMSE:             {metrics['rmse']:.2f} MW")
    print(f"  MAE:              {metrics['mae']:.2f} MW")

    # 按时间范围
    print(f"\n按预测时间范围:")
    for start, end, name in [(0,4,"0-1h"),(4,8,"1-2h"),(8,12,"2-3h"),(12,16,"3-4h")]:
        m = eval_metrics(y_test[:, start:end], all_preds[:, start:end], cap)
        print(f"  {name}: ACC2={m['acc2']:.4f}, RMSE={m['rmse']:.2f} MW")

    # 逐步 RMSE
    print(f"\n逐步 RMSE (观察误差累积):")
    for t in range(args.out_steps):
        step_rmse = np.sqrt(np.mean((y_test[:, t] * cap - all_preds[:, t] * cap) ** 2))
        bar = "#" * int(step_rmse / 2)
        print(f"  step {t+1:2d} (+{(t+1)*15:3d}min): RMSE={step_rmse:5.2f} MW  {bar}")

    print(f"{'='*60}")


# ============== 主入口 ==============

def main():
    parser = argparse.ArgumentParser(
        description="太阳能电站功率预测 (自回归 Transformer)")

    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "preprocess", "train", "evaluate"])
    parser.add_argument("--csv-path", type=str,
                        default="/media/zlg/Data1/Longjiao/TF208/solar_station_1.csv",
                        help="solar_station_1.csv 文件路径")
    parser.add_argument("--out-steps", type=int, default=16,
                        help="预测步数 (默认16 = 4小时)")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--huber-delta", type=float, default=1.0)

    # 模型结构
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)

    args = parser.parse_args()

    if args.mode in ["all", "preprocess"]:
        print("\n[1/3] 数据预处理")
        preprocess(args.csv_path, out_steps=args.out_steps)

    if args.mode in ["all", "train"]:
        print("\n[2/3] 模型训练")
        train(args)

    if args.mode in ["all", "evaluate"]:
        print("\n[3/3] 模型评估")
        evaluate(args)


if __name__ == "__main__":
    main()
