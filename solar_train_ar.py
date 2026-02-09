#!/usr/bin/env python3
"""
太阳能电站功率预测 - 自回归训练与评估脚本

训练: 单步预测 (输入96步 → 预测第97步)
评估: 滑动窗口自回归推理 (输入96步 → 逐步预测16步)

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
import numpy as np
import os
import argparse
import pickle
import time

from solar_model_ar import SolarTransformerAR
from solar_preprocess import load_and_clean, extract_features


# ============== 数据准备 ==============

def create_ar_train_sequences(enc_features, targets, in_steps=96):
    """
    单步预测训练数据:
      X: [N, 96, 13]  → y: [N]  (第97步的功率)
    """
    X_list, y_list = [], []
    for i in range(len(enc_features) - in_steps):
        X_list.append(enc_features[i : i + in_steps])
        y_list.append(targets[i + in_steps])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    print(f"  单步训练序列: X={X.shape}, y={y.shape}")
    return X, y


def create_ar_eval_sequences(enc_features, dec_features, targets,
                              in_steps=96, out_steps=16):
    """
    自回归推理评估数据:
      X_enc: [N, 96, 13]   初始历史
      X_dec: [N, 16, 12]   未来气象+时间 (用于滑动窗口中构造新输入)
      y:     [N, 16]        真实功率 (16步)
    """
    X_enc_list, X_dec_list, y_list = [], [], []
    total_len = in_steps + out_steps

    for i in range(len(enc_features) - total_len + 1):
        X_enc_list.append(enc_features[i : i + in_steps])
        X_dec_list.append(dec_features[i + in_steps : i + total_len])
        y_list.append(targets[i + in_steps : i + total_len])

    X_enc = np.array(X_enc_list, dtype=np.float32)
    X_dec = np.array(X_dec_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    print(f"  评估序列: X_enc={X_enc.shape}, X_dec={X_dec.shape}, y={y.shape}")
    return X_enc, X_dec, y


def preprocess(csv_path, in_steps=96, out_steps=16,
               train_ratio=0.7, val_ratio=0.15, output_dir="."):
    """预处理: 生成单步训练数据 + 多步评估数据"""
    print("=" * 60)
    print("自回归模型数据预处理")
    print("=" * 60)

    df = load_and_clean(csv_path)
    enc_features, dec_features, targets, norm_params, enc_names, dec_names = extract_features(df)

    n = len(enc_features)
    # 按时间划分 (在原始序列上切分, 再创建滑动窗口)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # --- 训练集: 单步预测 ---
    print("\n训练集:")
    X_train, y_train = create_ar_train_sequences(
        enc_features[:train_end], targets[:train_end], in_steps)

    # --- 验证集: 单步预测 (用于训练时的 loss 监控) ---
    print("验证集 (单步):")
    X_val, y_val = create_ar_train_sequences(
        enc_features[train_end:val_end], targets[train_end:val_end], in_steps)

    # --- 验证集: 多步评估 (用于自回归推理评估) ---
    print("验证集 (多步推理):")
    X_enc_val, X_dec_val, y_val_ms = create_ar_eval_sequences(
        enc_features[train_end:val_end], dec_features[train_end:val_end],
        targets[train_end:val_end], in_steps, out_steps)

    # --- 测试集: 多步评估 ---
    print("测试集 (多步推理):")
    X_enc_test, X_dec_test, y_test_ms = create_ar_eval_sequences(
        enc_features[val_end:], dec_features[val_end:],
        targets[val_end:], in_steps, out_steps)

    # 保存
    save = lambda name, arr: np.save(os.path.join(output_dir, f"{name}.npy"), arr)
    save("ar_X_train", X_train)
    save("ar_y_train", y_train)
    save("ar_X_val", X_val)
    save("ar_y_val", y_val)
    save("ar_X_enc_val", X_enc_val)
    save("ar_X_dec_val", X_dec_val)
    save("ar_y_val_ms", y_val_ms)
    save("ar_X_enc_test", X_enc_test)
    save("ar_X_dec_test", X_dec_test)
    save("ar_y_test_ms", y_test_ms)

    with open(os.path.join(output_dir, "norm_params.pkl"), "wb") as f:
        pickle.dump(norm_params, f)

    print(f"\n预处理完成! 文件保存到: {output_dir}")
    return norm_params


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


# ============== 自回归多步评估 ==============

def ar_evaluate_multistep(model, X_enc, X_dec, y_true, cap, device,
                          batch_size=256, out_steps=16):
    """
    自回归滑动窗口推理 → 评估16步预测
    """
    model.eval()
    all_preds = []
    n = len(X_enc)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x_enc = torch.FloatTensor(X_enc[start:end]).to(device)
        x_dec = torch.FloatTensor(X_dec[start:end]).to(device)

        preds = model.predict_sequence(x_enc, x_dec, out_steps=out_steps)
        all_preds.append(preds.cpu().numpy())

    all_preds = np.concatenate(all_preds)

    acc1 = calc_acc_mae(y_true, all_preds, cap)
    acc2 = calc_acc2(y_true, all_preds, cap)
    rmse = calc_rmse(y_true, all_preds, cap)
    mae = calc_mae(y_true, all_preds, cap)

    return all_preds, {"acc1": acc1, "acc2": acc2, "rmse": rmse, "mae": mae}


# ============== 训练 ==============

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载数据
    X_train = np.load("ar_X_train.npy")
    y_train = np.load("ar_y_train.npy")
    X_val = np.load("ar_X_val.npy")
    y_val = np.load("ar_y_val.npy")

    # 多步评估数据
    X_enc_val = np.load("ar_X_enc_val.npy")
    X_dec_val = np.load("ar_X_dec_val.npy")
    y_val_ms = np.load("ar_y_val_ms.npy")

    with open("norm_params.pkl", "rb") as f:
        norm_params = pickle.load(f)
    cap = norm_params["power"]["cap"]

    feat_size = X_train.shape[2]
    seq_len = X_train.shape[1]

    print(f"训练集: {X_train.shape} → {y_train.shape} (单步)")
    print(f"验证集: {X_val.shape} → {y_val.shape} (单步)")
    print(f"验证集 (多步): enc={X_enc_val.shape}, dec={X_dec_val.shape}, y={y_val_ms.shape}")

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
    model = SolarTransformerAR(
        feat_size=feat_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.d_model * 4,
        dropout=args.dropout,
        seq_len=seq_len,
    ).to(device)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p, gain=0.5)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[args.warmup_epochs])

    criterion = nn.HuberLoss(delta=args.huber_delta)

    os.makedirs("solar_checkpoints", exist_ok=True)
    log_dir = os.path.join("runs", "ar_" + time.strftime("%Y%m%d_%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n{'='*70}")
    print(f"自回归训练 | Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"模型: d_model={args.d_model}, heads={args.nhead}, layers={args.num_layers}")
    print(f"{'='*70}\n")

    for epoch in range(args.epochs):
        # --- 训练 (单步预测) ---
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_x)  # [B]
            loss = criterion(pred, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # --- 验证 (单步预测 loss) ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred = model(batch_x)
                val_loss += criterion(pred, batch_y).item()
        val_loss /= len(val_loader)

        lr = optimizer.param_groups[0]["lr"]
        writer.add_scalars("Loss", {"Train": train_loss, "Val": val_loss}, epoch + 1)
        writer.add_scalar("LearningRate", lr, epoch + 1)

        # --- 每 5 个 epoch 做一次多步自回归评估 ---
        if (epoch + 1) % 5 == 0 or epoch == 0:
            _, metrics = ar_evaluate_multistep(
                model, X_enc_val, X_dec_val, y_val_ms, cap, device,
                batch_size=args.batch_size)
            writer.add_scalar("AR_Accuracy/ACC1", metrics["acc1"], epoch + 1)
            writer.add_scalar("AR_Accuracy/ACC2", metrics["acc2"], epoch + 1)
            writer.add_scalar("AR_Error/RMSE_MW", metrics["rmse"], epoch + 1)

            print(
                f"Epoch {epoch+1:3d}/{args.epochs} | LR: {lr:.6f} | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                f"AR-ACC1: {metrics['acc1']:.4f} | AR-ACC2: {metrics['acc2']:.4f} | "
                f"AR-RMSE: {metrics['rmse']:.2f} MW"
            )
        else:
            print(
                f"Epoch {epoch+1:3d}/{args.epochs} | LR: {lr:.6f} | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f}"
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

    X_enc_test = np.load("ar_X_enc_test.npy")
    X_dec_test = np.load("ar_X_dec_test.npy")
    y_test = np.load("ar_y_test_ms.npy")

    with open("norm_params.pkl", "rb") as f:
        norm_params = pickle.load(f)
    cap = norm_params["power"]["cap"]

    feat_size = X_enc_test.shape[2]
    seq_len = X_enc_test.shape[1]

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
    print(f"加载模型: epoch={checkpoint['epoch']+1}, val_loss={checkpoint['val_loss']:.6f}")

    all_preds, metrics = ar_evaluate_multistep(
        model, X_enc_test, X_dec_test, y_test, cap, device,
        batch_size=args.batch_size)

    print(f"\n{'='*60}")
    print(f"测试集评估 (自回归滑动窗口推理)")
    print(f"{'='*60}")
    print(f"  ACC1 (MAE-based): {metrics['acc1']:.4f} ({metrics['acc1']*100:.2f}%)")
    print(f"  ACC2 (国标):      {metrics['acc2']:.4f} ({metrics['acc2']*100:.2f}%)")
    print(f"  RMSE:             {metrics['rmse']:.2f} MW")
    print(f"  MAE:              {metrics['mae']:.2f} MW")

    # 按预测步数分析误差累积
    print(f"\n按预测步数 (观察误差累积):")
    steps_per_hour = 4
    horizons = [
        (0, steps_per_hour, "0-1h"),
        (steps_per_hour, 2 * steps_per_hour, "1-2h"),
        (2 * steps_per_hour, 3 * steps_per_hour, "2-3h"),
        (3 * steps_per_hour, 4 * steps_per_hour, "3-4h"),
    ]
    for start, end, name in horizons:
        yt = y_test[:, start:end]
        yp = all_preds[:, start:end]
        h_acc2 = calc_acc2(yt, yp, cap)
        h_rmse = calc_rmse(yt, yp, cap)
        print(f"  {name}: ACC2={h_acc2:.4f}, RMSE={h_rmse:.2f} MW")

    # 逐步误差 (观察累积趋势)
    print(f"\n逐步 RMSE (15分钟粒度):")
    for t in range(16):
        step_rmse = np.sqrt(np.mean((y_test[:, t] * cap - all_preds[:, t] * cap) ** 2))
        bar = "#" * int(step_rmse / 2)
        print(f"  step {t+1:2d} (+{(t+1)*15:3d}min): RMSE={step_rmse:5.2f} MW  {bar}")

    print(f"{'='*60}")


# ============== 主入口 ==============

def main():
    parser = argparse.ArgumentParser(description="太阳能电站功率预测 (自回归 Transformer)")

    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "preprocess", "train", "evaluate"])
    parser.add_argument("--csv-path", type=str,
                        default="/media/zlg/Data1/Longjiao/TF208/solar_station_1.csv",
                        help="solar_station_1.csv 文件路径")

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
        preprocess(args.csv_path)

    if args.mode in ["all", "train"]:
        print("\n[2/3] 模型训练")
        train(args)

    if args.mode in ["all", "evaluate"]:
        print("\n[3/3] 模型评估")
        evaluate(args)


if __name__ == "__main__":
    main()
