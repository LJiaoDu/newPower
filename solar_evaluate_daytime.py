#!/usr/bin/env python3
"""
太阳能电站功率预测 - 带白天 Mask 的评估脚本

对比两种 ACC2 计算方式：
  - ACC2_all:  原始国标，包含夜晚（target≈0 导致指标虚高）
  - ACC2_day:  仅白天样本（target > threshold），消除夜晚刷分
  - ACC1:      MAE-based，原本就只含白天，用于参考

使用方式:
  # 完整流程（加载模型推理 + 评估）
  python solar_evaluate_daytime.py

  # 推理结果存到文件，下次直接加载（省去 torch 推理环境）
  python solar_evaluate_daytime.py --save-preds preds_test.npy

  # 直接用已保存的预测结果做评估（不需要 torch）
  python solar_evaluate_daytime.py --load-preds preds_test.npy

  # 调整参数
  python solar_evaluate_daytime.py --threshold 0.02 --step-minutes 15
"""

import os
import argparse
import pickle

import numpy as np


# ============================================================
# 指标函数
# ============================================================

def daytime_mask(y_true_norm, threshold=0.01):
    """返回白天 bool mask：归一化功率 > threshold"""
    return y_true_norm.flatten() > threshold


def calc_acc1(y_true_norm, y_pred_norm, cap, threshold=0.01):
    """ACC1 (MAE-based): 1 - MAE/mean(y_true), 仅白天"""
    mask = y_true_norm.flatten() > threshold
    if mask.sum() == 0:
        return float("nan")
    yt = y_true_norm.flatten()[mask] * cap
    yp = y_pred_norm.flatten()[mask] * cap
    return max(0.0, 1.0 - np.mean(np.abs(yt - yp)) / (np.mean(yt) + 1e-6))


def calc_acc2_all(y_true_norm, y_pred_norm, cap):
    """ACC2 原始国标：全部样本（含夜晚）"""
    p_m = y_true_norm.flatten() * cap
    p_p = y_pred_norm.flatten() * cap
    denom = np.maximum(p_m, 0.2 * cap)
    return max(0.0, 1.0 - np.sqrt(np.mean(((p_m - p_p) / denom) ** 2)))


def calc_acc2_day(y_true_norm, y_pred_norm, cap, threshold=0.01):
    """ACC2 白天版：仅白天样本（target > threshold）"""
    mask = y_true_norm.flatten() > threshold
    if mask.sum() == 0:
        return float("nan")
    p_m = y_true_norm.flatten()[mask] * cap
    p_p = y_pred_norm.flatten()[mask] * cap
    denom = np.maximum(p_m, 0.2 * cap)
    return max(0.0, 1.0 - np.sqrt(np.mean(((p_m - p_p) / denom) ** 2)))


def calc_rmse(y_true_norm, y_pred_norm, cap, threshold=None):
    """RMSE (MW)，可选仅白天"""
    yt = y_true_norm.flatten() * cap
    yp = y_pred_norm.flatten() * cap
    if threshold is not None:
        mask = y_true_norm.flatten() > threshold
        if mask.sum() == 0:
            return float("nan")
        yt, yp = yt[mask], yp[mask]
    return np.sqrt(np.mean((yt - yp) ** 2))


def calc_mae(y_true_norm, y_pred_norm, cap, threshold=None):
    """MAE (MW)，可选仅白天"""
    yt = y_true_norm.flatten() * cap
    yp = y_pred_norm.flatten() * cap
    if threshold is not None:
        mask = y_true_norm.flatten() > threshold
        if mask.sum() == 0:
            return float("nan")
        yt, yp = yt[mask], yp[mask]
    return np.mean(np.abs(yt - yp))


# ============================================================
# 加载模型 & 推理（依赖 torch，仅在无 --load-preds 时调用）
# ============================================================

def run_model_inference(args):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm
    from solar_model_rope import SolarTransformerRoPE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"推理设备: {device}")

    X_enc = np.load("X_enc_test.npy")
    X_dec = np.load("X_dec_test.npy")
    y     = np.load("y_test.npy")

    model = SolarTransformerRoPE(
        enc_feat_size=X_enc.shape[2],
        dec_feat_size=X_dec.shape[2],
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.d_model * 4,
        dropout=0.0,
        enc_seq_len=X_enc.shape[1],
        dec_seq_len=X_dec.shape[1],
    ).to(device)

    ckpt_path = "solar_checkpoints/best_model_194.pth"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"加载模型: epoch={ckpt['epoch']+1}, 训练时ACC2={ckpt['acc2']:.4f}")

    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_enc),
                      torch.FloatTensor(X_dec),
                      torch.FloatTensor(y)),
        batch_size=256, shuffle=False,
    )
    all_preds, all_targets = [], []
    with torch.no_grad():
        for be, bd, by in tqdm(loader, desc="推理中"):
            pred = model(be.to(device), bd.to(device))
            all_preds.append(pred.cpu().numpy())
            all_targets.append(by.numpy())

    preds   = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    if args.save_preds:
        np.save(args.save_preds, preds)
        print(f"预测结果已保存至: {args.save_preds}")

    return preds, targets


def load_data(args):
    """加载预测结果和真实值"""
    with open("norm_params.pkl", "rb") as f:
        norm_params = pickle.load(f)
    cap = norm_params["power"]["cap"]

    targets = np.load("y_test.npy")

    if args.load_preds:
        preds = np.load(args.load_preds)
        print(f"加载预测结果: {args.load_preds}, shape={preds.shape}")
    else:
        preds, targets = run_model_inference(args)

    return preds, targets, cap


# ============================================================
# 主评估逻辑
# ============================================================

def evaluate(args):
    thr = args.threshold  # 归一化阈值
    step_min = args.step_minutes

    preds, targets, cap = load_data(args)
    num_steps = targets.shape[1]

    # ---------- 整体指标 ----------
    acc1       = calc_acc1(targets, preds, cap, thr)
    acc2_all   = calc_acc2_all(targets, preds, cap)
    acc2_day   = calc_acc2_day(targets, preds, cap, thr)
    rmse_all   = calc_rmse(targets, preds, cap)
    rmse_day   = calc_rmse(targets, preds, cap, thr)
    mae_all    = calc_mae(targets, preds, cap)
    mae_day    = calc_mae(targets, preds, cap, thr)

    # 白天样本占比
    day_ratio = (targets.flatten() > thr).mean()

    print(f"\n{'='*68}")
    print(f"  测试集评估结果 (标称容量 {cap} MW, 白天阈值={thr})")
    print(f"{'='*68}")
    print(f"  白天样本占比:          {day_ratio*100:.1f}%")
    print(f"  ACC1  (MAE, 仅白天):   {acc1:.4f}  ({acc1*100:.2f}%)")
    print(f"  ACC2  (国标, 全部):    {acc2_all:.4f}  ({acc2_all*100:.2f}%)  ← 含夜晚, 偏高")
    print(f"  ACC2  (国标, 仅白天):  {acc2_day:.4f}  ({acc2_day*100:.2f}%)  ← 真实水平")
    print(f"  虚高幅度:              {(acc2_all - acc2_day)*100:.2f} 个百分点")
    print(f"  RMSE (全部):           {rmse_all:.4f} MW")
    print(f"  RMSE (仅白天):         {rmse_day:.4f} MW")
    print(f"  MAE  (全部):           {mae_all:.4f} MW")
    print(f"  MAE  (仅白天):         {mae_day:.4f} MW")
    print(f"{'='*68}")

    # ---------- 逐步指标 ----------
    print(f"\n按预测步 (每步{step_min}分钟, 共{num_steps}步):")

    hdr = f"  {'步':>3}  {'时刻':>8}  {'白天占比':>6}  {'ACC1':>6}  {'ACC2_全':>7}  {'ACC2_白':>7}  {'虚高':>6}  {'RMSE_d':>7}  {'MAE_d':>7}"
    sep = "  " + "-"*3 + "  " + "-"*8 + "  " + "-"*6 + "  " + "-"*6 + "  " + "-"*7 + "  " + "-"*7 + "  " + "-"*6 + "  " + "-"*7 + "  " + "-"*7
    print(hdr)
    print(sep)

    gaps = []
    for step in range(num_steps):
        yt = targets[:, step:step+1]
        yp = preds[:, step:step+1]
        day_pct  = (yt.flatten() > thr).mean() * 100
        a1       = calc_acc1(yt, yp, cap, thr)
        a2_all   = calc_acc2_all(yt, yp, cap)
        a2_day   = calc_acc2_day(yt, yp, cap, thr)
        r_day    = calc_rmse(yt, yp, cap, thr)
        m_day    = calc_mae(yt, yp, cap, thr)
        gap      = a2_all - a2_day
        gaps.append((step + 1, (step+1)*step_min, a2_all, a2_day, gap))
        print(
            f"  {step+1:>3}  {'+%dmin'%((step+1)*step_min):>8}  {day_pct:>5.1f}%"
            f"  {a1:>6.4f}  {a2_all:>7.4f}  {a2_day:>7.4f}"
            f"  {gap*100:>+5.2f}pp  {r_day:>7.4f}  {m_day:>7.4f}"
        )

    print(f"{'='*68}")

    # ---------- 夜晚刷分最严重的步 ----------
    gaps.sort(key=lambda x: -x[4])
    print("\nACC2 虚高最严重的 10 个预测步:")
    print(f"  {'步':>5}  {'时刻':>8}  {'ACC2_全':>7}  {'ACC2_白':>7}  {'虚高':>8}")
    print(f"  {'-----':>5}  {'--------':>8}  {'-------':>7}  {'-------':>7}  {'--------':>8}")
    for g in gaps[:10]:
        print(f"  step{g[0]:>2}  {'+%dmin'%g[1]:>8}  {g[2]:>7.4f}  {g[3]:>7.4f}  {g[4]*100:>+7.2f}pp")


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="白天 Mask 评估脚本")
    parser.add_argument("--threshold",    type=float, default=0.01,
                        help="白天判断阈值（归一化功率，默认 0.01）")
    parser.add_argument("--step-minutes", type=int,   default=15,
                        help="每预测步的分钟数（默认 15）")
    # 预测缓存（两种互斥）
    preds_group = parser.add_mutually_exclusive_group()
    preds_group.add_argument("--load-preds", type=str, default=None,
                        help="直接加载已保存的预测 .npy 文件（不需要 torch）")
    preds_group.add_argument("--save-preds", type=str, default=None,
                        help="推理完成后将预测结果保存到指定 .npy 文件")
    # 模型结构（需与训练时一致，仅推理时用到）
    parser.add_argument("--d-model",             type=int,   default=128)
    parser.add_argument("--nhead",               type=int,   default=4)
    parser.add_argument("--num-encoder-layers",  type=int,   default=3)
    parser.add_argument("--num-decoder-layers",  type=int,   default=2)
    parser.add_argument("--dropout",             type=float, default=0.0)

    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
