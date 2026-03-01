#!/usr/bin/env python3
"""
太阳能电站功率预测 - 神经 ARMA 模型训练脚本

模型: ARMAModel (solar_model_arma.py)
  - AR 分量: 线性 AR(p), 用过去 p 步功率直接预测未来 16 步
  - MA 分量: 线性 MA(q), 用过去 q 步 AR 残差做误差修正
  - 气象分量: MLP 处理历史+未来气象特征做外生调整
  - 三分量权重可学习 (softmax 归一化)

使用方式:
  python solar_train_arma.py                      # 完整流程 (预处理+训练+评估)
  python solar_train_arma.py --mode train         # 仅训练
  python solar_train_arma.py --mode evaluate      # 仅评估
  python solar_train_arma.py --p 48 --q 12        # 调整 ARMA 阶数
  python solar_train_arma.py --d-hidden 256       # 更大气象 MLP
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

from solar_model_arma import ARMAModel
from solar_preprocess import main as preprocess_main


# ============== 混合 Loss ==============

class ACC2Loss(nn.Module):
    """
    基于国标 ACC2 的损失函数
    L = mean( ((pred - target) / max(target, 0.2 * cap_norm))^2 )
    """

    def __init__(self, cap_norm: float = 1.0):
        super().__init__()
        self.cap_norm = cap_norm

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        denom = torch.clamp(target, min=0.2 * self.cap_norm)
        return torch.mean(((pred - target) / denom) ** 2)


class MixedLoss(nn.Module):
    """混合 Loss = λ_mse * MSE + λ_acc2 * ACC2_Loss"""

    def __init__(self, lambda_mse: float = 1.0, lambda_acc2: float = 1.0,
                 cap_norm: float = 1.0):
        super().__init__()
        self.lambda_mse  = lambda_mse
        self.lambda_acc2 = lambda_acc2
        self.mse_loss    = nn.MSELoss()
        self.acc2_loss   = ACC2Loss(cap_norm=cap_norm)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (self.lambda_mse  * self.mse_loss(pred, target)
                + self.lambda_acc2 * self.acc2_loss(pred, target))


# ============== 评估指标 ==============

def calc_acc_mae(y_true: np.ndarray, y_pred: np.ndarray, cap: float = 1.0) -> float:
    """ACC1 (MAE-based): 1 - MAE / mean(y_true), 仅计算功率 > 0 的样本"""
    mask = y_true > 0.01
    if mask.sum() == 0:
        return float("nan")
    y_t = y_true[mask] * cap
    y_p = y_pred[mask] * cap
    mae = np.mean(np.abs(y_t - y_p))
    avg = np.mean(y_t)
    return max(0.0, 1.0 - mae / (avg + 1e-6))


def calc_rmse(y_true: np.ndarray, y_pred: np.ndarray, cap: float = 1.0) -> float:
    return float(np.sqrt(np.mean((y_true * cap - y_pred * cap) ** 2)))


def calc_mae(y_true: np.ndarray, y_pred: np.ndarray, cap: float = 1.0) -> float:
    return float(np.mean(np.abs(y_true * cap - y_pred * cap)))


def calc_acc2(y_true: np.ndarray, y_pred: np.ndarray, cap: float = 1.0) -> float:
    """ACC2 (国标): 1 - sqrt( mean( ((P_M - P_P) / max(P_M, 0.2*Cap))^2 ) )"""
    p_m   = y_true.flatten() * cap
    p_p   = y_pred.flatten() * cap
    denom = np.maximum(p_m, 0.2 * cap)
    return max(0.0, 1.0 - float(np.sqrt(np.mean(((p_m - p_p) / denom) ** 2))))


# ============== 训练 ==============

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # --- 加载数据 ---
    X_enc_train = np.load("X_enc_train.npy")
    X_dec_train = np.load("X_dec_train.npy")
    y_train     = np.load("y_train.npy")
    X_enc_val   = np.load("X_enc_val.npy")
    X_dec_val   = np.load("X_dec_val.npy")
    y_val       = np.load("y_val.npy")

    with open("norm_params.pkl", "rb") as f:
        norm_params = pickle.load(f)
    cap = norm_params["power"]["cap"]

    print(f"训练集: enc={X_enc_train.shape}, dec={X_dec_train.shape}, y={y_train.shape}")
    print(f"验证集: enc={X_enc_val.shape},   dec={X_dec_val.shape},   y={y_val.shape}")
    print(f"标称容量: {cap} MW")

    hist_len  = X_enc_train.shape[1]   # 96
    fut_len   = X_dec_train.shape[1]   # 16
    n_weather = 6

    # --- DataLoader ---
    train_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_enc_train),
            torch.FloatTensor(X_dec_train),
            torch.FloatTensor(y_train),
        ),
        batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_enc_val),
            torch.FloatTensor(X_dec_val),
            torch.FloatTensor(y_val),
        ),
        batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # --- 模型 ---
    model = ARMAModel(
        p=args.p,
        q=args.q,
        horizon=fut_len,
        n_weather=n_weather,
        d_hidden=args.d_hidden,
        hist_len=hist_len,
        fut_len=fut_len,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # --- 优化器与调度器 ---
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    warmup_sched  = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                              total_iters=args.warmup_epochs)
    cosine_sched  = CosineAnnealingLR(optimizer,
                                       T_max=args.epochs - args.warmup_epochs,
                                       eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched],
                              milestones=[args.warmup_epochs])

    criterion = MixedLoss(
        lambda_mse=args.lambda_mse,
        lambda_acc2=args.lambda_acc2,
        cap_norm=1.0,
    )

    os.makedirs("solar_checkpoints", exist_ok=True)

    # --- TensorBoard ---
    log_dir = os.path.join("runs", f"arma_{time.strftime('%Y%m%d_%H%M%S')}")
    writer  = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard 日志: {log_dir}")

    best_acc2       = -1.0
    patience_counter = 0

    print(f"\n{'='*70}")
    print(f"开始训练 | Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"ARMA 阶数: p={args.p} (AR), q={args.q} (MA) | d_hidden={args.d_hidden}")
    print(f"Loss: λ_mse={args.lambda_mse} * MSE + λ_acc2={args.lambda_acc2} * ACC2Loss")
    print(f"输入噪声: std={args.noise_std} | dropout={args.dropout}")
    print(f"{'='*70}\n")

    epoch_bar = tqdm(range(args.epochs), desc="Training", unit="epoch")
    for epoch in epoch_bar:

        # ---- 训练 ----
        model.train()
        train_loss = 0.0
        train_bar  = tqdm(train_loader,
                          desc=f"Epoch {epoch+1}/{args.epochs} [Train]",
                          leave=False, unit="batch")
        for batch_enc, batch_dec, batch_y in train_bar:
            batch_enc = batch_enc.to(device)
            batch_dec = batch_dec.to(device)
            batch_y   = batch_y.to(device)

            # 输入噪声增强 (仅训练时)
            if args.noise_std > 0:
                batch_enc = batch_enc + torch.randn_like(batch_enc) * args.noise_std

            optimizer.zero_grad()
            pred = model(batch_enc, batch_dec)
            loss = criterion(pred, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.6f}")
        train_loss /= len(train_loader)

        # ---- 验证 (无噪声) ----
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch_enc, batch_dec, batch_y in val_loader:
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
        writer.add_scalars("Loss", {"Train": train_loss, "Val": val_loss}, epoch + 1)
        writer.add_scalar("Accuracy/ACC1",   acc1, epoch + 1)
        writer.add_scalar("Accuracy/ACC2",   acc2, epoch + 1)
        writer.add_scalar("Error/RMSE_MW",   rmse, epoch + 1)
        writer.add_scalar("Error/MAE_MW",    mae,  epoch + 1)
        writer.add_scalar("LearningRate",    lr,   epoch + 1)

        # 记录三分量权重变化
        alpha = torch.softmax(model.log_alpha.detach().cpu(), dim=0)
        writer.add_scalars("ComponentWeights", {
            "AR":      alpha[0].item(),
            "MA":      alpha[1].item(),
            "Weather": alpha[2].item(),
        }, epoch + 1)

        epoch_bar.set_postfix(
            train=f"{train_loss:.5f}", val=f"{val_loss:.5f}",
            ACC1=f"{acc1:.4f}", ACC2=f"{acc2:.4f}"
        )
        tqdm.write(
            f"Epoch {epoch+1:3d}/{args.epochs} | LR: {lr:.6f} | "
            f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
            f"ACC1: {acc1:.4f} | ACC2: {acc2:.4f} | "
            f"RMSE: {rmse:.2f} MW | MAE: {mae:.2f} MW | "
            f"α(AR/MA/X)={alpha[0]:.2f}/{alpha[1]:.2f}/{alpha[2]:.2f}"
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
            }, "solar_checkpoints/best_model_arma.pth")
            tqdm.write(f"  -> 保存最佳模型 (ACC2: {acc2:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                tqdm.write(f"\nEarly stopping: {args.patience} epochs ACC2 无改善")
                break

        scheduler.step()

    writer.close()
    print(f"\n训练完成! 最佳 ACC2: {best_acc2:.4f}")


# ============== 评估 ==============

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_enc_test = np.load("X_enc_test.npy")
    X_dec_test = np.load("X_dec_test.npy")
    y_test     = np.load("y_test.npy")

    with open("norm_params.pkl", "rb") as f:
        norm_params = pickle.load(f)
    cap = norm_params["power"]["cap"]

    hist_len = X_enc_test.shape[1]   # 96
    fut_len  = X_dec_test.shape[1]   # 16

    model = ARMAModel(
        p=args.p, q=args.q, horizon=fut_len,
        n_weather=6, d_hidden=args.d_hidden,
        hist_len=hist_len, fut_len=fut_len,
        dropout=args.dropout,
    ).to(device)

    checkpoint = torch.load(
        "solar_checkpoints/best_model_arma.pth",
        map_location=device, weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"加载模型: epoch={checkpoint['epoch']+1}, "
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
    with torch.no_grad():
        for batch_enc, batch_dec, batch_y in tqdm(test_loader, desc="Testing"):
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

    print(f"\n{'='*60}")
    print(f"测试集评估结果 (神经 ARMA 模型)")
    print(f"{'='*60}")
    print(f"  ACC1 (MAE-based): {acc1:.4f} ({acc1*100:.2f}%)")
    print(f"  ACC2 (国标):      {acc2:.4f} ({acc2*100:.2f}%)")
    print(f"  RMSE:             {rmse:.2f} MW")
    print(f"  MAE:              {mae:.2f} MW")

    # 显示三分量学习到的权重
    alpha = torch.softmax(model.log_alpha.cpu(), dim=0)
    print(f"\n学习到的分量权重:")
    print(f"  AR  分量 (φ): {alpha[0]:.4f} ({alpha[0]*100:.1f}%)")
    print(f"  MA  分量 (θ): {alpha[1]:.4f} ({alpha[1]*100:.1f}%)")
    print(f"  气象分量 (X): {alpha[2]:.4f} ({alpha[2]*100:.1f}%)")

    # 逐步分析 16 个预测点
    num_steps = all_targets.shape[1]
    print(f"\n按预测步 (每步15分钟, 共{num_steps}步):")
    print(f"  {'步':>3s}  {'时刻':>7s}  {'ACC1':>7s}  {'ACC2':>7s}"
          f"  {'RMSE(MW)':>9s}  {'MAE(MW)':>8s}")
    print(f"  {'-'*3}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*9}  {'-'*8}")
    for step in range(num_steps):
        yt     = all_targets[:, step:step+1]
        yp     = all_preds[:, step:step+1]
        s_acc1 = calc_acc_mae(yt, yp, cap)
        s_acc2 = calc_acc2(yt, yp, cap)
        s_rmse = calc_rmse(yt, yp, cap)
        s_mae  = calc_mae(yt, yp, cap)
        minutes = (step + 1) * 15
        print(f"  {step+1:3d}  +{minutes:4d}min  {s_acc1:.4f}  {s_acc2:.4f}"
              f"  {s_rmse:9.2f}  {s_mae:8.2f}")
    print(f"{'='*60}")


# ============== 主入口 ==============

def main():
    parser = argparse.ArgumentParser(
        description="太阳能电站功率预测 - 神经 ARMA 模型"
    )

    # --- 运行模式 ---
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "preprocess", "train", "evaluate"],
                        help="运行模式")
    parser.add_argument("--csv-path", type=str,
                        default="/media/zlg/Data1/Longjiao/TF208/solar_station_1.csv",
                        help="solar_station_1.csv 文件路径")

    # --- ARMA 超参数 ---
    parser.add_argument("--p", type=int, default=96,
                        help="AR 阶数: 使用最近 p 步历史功率 (≤96)")
    parser.add_argument("--q", type=int, default=24,
                        help="MA 阶数: 使用最近 q 步 AR 残差 (q+p ≤96)")
    parser.add_argument("--d-hidden", type=int, default=128,
                        help="气象 MLP 隐层维度")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout 比率")

    # --- 训练参数 ---
    parser.add_argument("--epochs", type=int, default=100,
                        help="最大训练轮次")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="L2 正则化系数")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early Stopping 耐心值 (基于 ACC2)")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="Warmup 轮次")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="梯度裁剪阈值")

    # --- 混合 Loss 权重 ---
    parser.add_argument("--lambda-mse", type=float, default=1.0,
                        help="MSE Loss 权重")
    parser.add_argument("--lambda-acc2", type=float, default=1.0,
                        help="ACC2 Loss 权重")

    # --- 数据增强 ---
    parser.add_argument("--noise-std", type=float, default=0.01,
                        help="训练时输入高斯噪声标准差 (0=关闭)")

    args = parser.parse_args()

    if args.mode in ["all", "preprocess"]:
        print("\n[1/3] 数据预处理")
        preprocess_main(csv_path=args.csv_path)

    if args.mode in ["all", "train"]:
        print("\n[2/3] 模型训练")
        train(args)

    if args.mode in ["all", "evaluate"]:
        print("\n[3/3] 模型评估")
        evaluate(args)


if __name__ == "__main__":
    main()
