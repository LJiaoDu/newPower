#!/usr/bin/env python3
"""
Solar Station 1 — 15分钟粒度训练脚本
基于 ImprovedTFMModel（非自回归 Transformer）

数据集: solar_station_1.csv
列:  sid, time, tsi, dni, ghi, temp, atm, rh, power, cap

特征（共 17 维）:
  时间特征 × 10: hour_norm, hour_sin/cos, weekday_sin/cos,
                  dayofyear_sin/cos, month_sin/cos, time_idx_norm
  气象特征 × 6 : tsi_norm, dni_norm, ghi_norm, temp_norm, atm_norm, rh_norm
  功率特征 × 1 : power_norm (= power / cap)

序列长度（15分钟粒度）:
  in_seq_len  = 80  (20小时 × 4步/时)
  out_seq_len = 16  ( 4小时 × 4步/时)

评估指标:
  ACC_MAE   = 1 - MAE  / global_avg_power
  ACC_RMSE  = 1 - RMSE / global_avg_power
  ACC2(国标) = 1 - sqrt( mean( ((P_M-P_P)/max(P_M, 0.2*Cap))^2 ) )
"""

import os
import time
import argparse
from datetime import datetime
import atexit

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import numpy as np
import pandas as pd

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

from model import ImprovedTFMModel as Model


# =========================================================
# 程序结束时保存“关键终端输出”到 txt（避免 tqdm 刷新残影写入）
# =========================================================
LOG_BUFFER = []

def log_print(*args, **kwargs):
    """
    替代 print：正常打印到终端，同时把文本缓存起来，程序结束后写入 txt
    """
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    message = sep.join(str(a) for a in args) + end

    print(*args, **kwargs)
    LOG_BUFFER.append(message.rstrip("\n"))

def log_tqdm_write(message: str):
    """
    替代 tqdm.write：写到终端（不破坏进度条），同时缓存
    """
    tqdm.write(str(message))
    LOG_BUFFER.append(str(message))

def save_log_to_file():
    """
    程序退出时，把 LOG_BUFFER 写入文件。
    不会记录 tqdm 的动态刷新，只记录你显式输出的文本（log_print/log_tqdm_write）。
    """
    try:
        os.makedirs("solar_logs", exist_ok=True)
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"solar_logs/final_log_solar15min_{time_str}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(LOG_BUFFER) + "\n")
        print(f"\n最终日志已保存至: {filename}")
    except Exception as e:
        print(f"\n[WARN] 保存最终日志失败: {e}")

atexit.register(save_log_to_file)


# ============================================================
# 评估指标
# ============================================================

def calc_acc2(y_true: np.ndarray, y_pred: np.ndarray, cap: float) -> float:
    """
    ACC2（国标）:
        1 - sqrt( (1/N) * sum( ((P_M - P_P) / max(P_M, 0.2*Cap))^2 ) )

    y_true / y_pred : 原始功率值（与 cap 同单位，kW）
    cap             : 装机容量（kW）
    """
    p_m   = y_true.flatten()
    p_p   = y_pred.flatten()
    denom = np.maximum(p_m, 0.2 * cap)
    return max(0.0, float(1.0 - np.sqrt(np.mean(((p_m - p_p) / denom) ** 2))))



# ============================================================
# 数据预处理工具
# ============================================================

def load_and_preprocess(data_path: str) -> pd.DataFrame:
    """
    读取 solar_station_1.csv，返回干净的 DataFrame。

    处理：
    - 解析 time 列，派生时间特征
    - 将 -99（缺测占位符）替换为 NaN，然后线性插值
    - rh 超出 [0, 100] 的异常值截断到 100
    """
    df = pd.read_csv(data_path, parse_dates=['time'])
    df = df.sort_values('time').reset_index(drop=True)

    # -99 → NaN，然后插值
    SENTINEL = -99
    weather_cols = ['tsi', 'dni', 'ghi', 'temp', 'atm', 'rh']
    for col in weather_cols:
        df[col] = df[col].replace(SENTINEL, np.nan)
    df[weather_cols] = df[weather_cols].interpolate(method='linear', limit_direction='both')

    # rh 截断到 [0, 100]（数据集中存在 >100 的异常值）
    df['rh'] = df['rh'].clip(0.0, 100.0)

    # 派生时间特征
    df['hour']      = df['time'].dt.hour
    df['minute']    = df['time'].dt.minute
    df['month']     = df['time'].dt.month
    df['dayofweek'] = df['time'].dt.dayofweek      # 0=周一
    df['dayofyear'] = df['time'].dt.dayofyear
    df['time_idx']  = np.arange(len(df), dtype=np.float32)

    return df


def normalize_weather(df: pd.DataFrame, stats: dict = None):
    """
    对气象特征做归一化，返回归一化后的数组和归一化参数 stats。

    tsi / dni / ghi : / 理论最大值（1400 / 1000 / 1000 W/m²）
    temp            : (x - min) / (max - min)  基于训练集
    atm             : (x - min) / (max - min)  基于训练集
    rh              : / 100
    """
    if stats is None:
        stats = {
            'tsi_max':  1400.0,
            'dni_max':  1000.0,
            'ghi_max':  1000.0,
            'temp_min': float(df['temp'].min()),
            'temp_max': float(df['temp'].max()),
            'atm_min':  float(df['atm'].min()),
            'atm_max':  float(df['atm'].max()),
        }

    tsi_n  = df['tsi'].values  / stats['tsi_max']
    dni_n  = df['dni'].values  / stats['dni_max']
    ghi_n  = df['ghi'].values  / stats['ghi_max']
    temp_n = (df['temp'].values - stats['temp_min']) / (stats['temp_max'] - stats['temp_min'] + 1e-8)
    atm_n  = (df['atm'].values  - stats['atm_min'])  / (stats['atm_max']  - stats['atm_min']  + 1e-8)
    rh_n   = df['rh'].values   / 100.0

    weather = np.stack([tsi_n, dni_n, ghi_n, temp_n, atm_n, rh_n], axis=1).astype(np.float32)
    return weather, stats


# ============================================================
# 数据集
# ============================================================

class SolarDataset15min(Dataset):
    """
    输入  : [80, 17]  (20小时历史，17维特征)
    输出  : [16]      (未来4小时功率，原始 kW)
    """

    IN_LEN  = 20 * 4   # 80 步
    OUT_LEN =  4 * 4   # 16 步

    def __init__(self, df: pd.DataFrame, weather_stats: dict, phase: str = 'train'):
        super().__init__()

        # ---------- 时间序列 70/15/15 切分 ----------
        n       = len(df)
        n_train = int(n * 0.70)
        n_val   = int(n * 0.85)   # 70%~85% 为验证集，85%~100% 为测试集
        if phase == 'train':
            sub = df.iloc[:n_train]
        elif phase == 'val':
            sub = df.iloc[n_train:n_val]
        else:                      # phase == 'test'
            sub = df.iloc[n_val:]
        sub = sub.reset_index(drop=True)

        cap_val = float(df['cap'].iloc[0])

        self.cap           = cap_val
        self.power_max     = float(sub['power'].max())
        self.power_min     = float(sub['power'].min())

        # 有效发电时段均值（用于 ACC 分母）
        valid = sub['power'].values[sub['power'].values > 0.2]   # >0.2 kW
        self.global_avg_power = float(np.mean(valid)) if len(valid) > 0 else 1.0

        # ---------- 构造特征矩阵 ----------
        self.features = self._build_time_features(sub, weather_stats)  # [N, 17]
        self.targets  = sub['power'].values.astype(np.float32)          # [N]  原始 kW

        log_print(f"[{phase:5s}] 样本数={len(sub)}  序列数={len(self)}"
                  f"  power: {self.power_min:.2f}~{self.power_max:.2f} kW"
                  f"  cap={self.cap} kW")

    def _build_time_features(self, sub: pd.DataFrame, weather_stats: dict) -> np.ndarray:
        hour      = sub['hour'].values.astype(np.float32)
        minute    = sub['minute'].values.astype(np.float32)
        month     = sub['month'].values.astype(np.float32)
        dayofweek = sub['dayofweek'].values.astype(np.float32)
        dayofyear = sub['dayofyear'].values.astype(np.float32)
        time_idx  = sub['time_idx'].values.astype(np.float32)
        power     = sub['power'].values.astype(np.float32)

        feats = []

        # ---- 10 个时间特征 ----
        hour_of_day = (hour + minute / 60.0) / 24.0
        feats.append(hour_of_day)                                   # 1

        ha = 2 * np.pi * hour_of_day
        feats.append(np.sin(ha))                                    # 2
        feats.append(np.cos(ha))                                    # 3

        da = 2 * np.pi * dayofweek / 7.0
        feats.append(np.sin(da))                                    # 4
        feats.append(np.cos(da))                                    # 5

        ya = 2 * np.pi * dayofyear / 365.0
        feats.append(np.sin(ya))                                    # 6
        feats.append(np.cos(ya))                                    # 7

        ma = 2 * np.pi * (month - 1) / 12.0
        feats.append(np.sin(ma))                                    # 8
        feats.append(np.cos(ma))                                    # 9

        t_norm = (time_idx - time_idx.min()) / (time_idx.max() - time_idx.min() + 1e-8)
        feats.append(t_norm)                                        # 10

        # ---- 6 个气象特征 ----
        weather, _ = normalize_weather(sub, weather_stats)
        for i in range(6):
            feats.append(weather[:, i])                             # 11~16

        # ---- 1 个历史功率特征 ----
        feats.append(power / (self.cap + 1e-8))                      # 17

        return np.stack(feats, axis=1).astype(np.float32)

    def __len__(self):
        return len(self.features) - self.IN_LEN - self.OUT_LEN

    def __getitem__(self, idx):
        s, m, e = idx, idx + self.IN_LEN, idx + self.IN_LEN + self.OUT_LEN
        return (
            self.features[s:m],     # [80, 17]
            self.targets[m:e],      # [16]  kW
        )


# ============================================================
# 训练主函数
# ============================================================

def train_val(cfg):

    # ---------- 设备 ----------
    if cfg.device == 'cpu':
        device = torch.device('cpu')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    log_print("=" * 80)
    log_print(f"Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"Args: {vars(cfg)}")
    log_print("=" * 80)

    log_print("=" * 80)
    log_print(f"设备        : {device}")
    log_print(f"数据集      : {cfg.data_path}")
    log_print(f"粒度        : 15 分钟")
    log_print(f"输入序列    : {cfg.in_seq_len} 步（20h）")
    log_print(f"预测序列    : {cfg.out_seq_len} 步（4h）")
    log_print(f"输入特征    : {cfg.in_feat_size} 维（10时间 + 6气象 + 1功率）")
    log_print("=" * 80 + "\n")

    # ---------- 数据预处理 ----------
    df = load_and_preprocess(cfg.data_path)

    if cfg.subset < 1.0:
        df = df.iloc[:int(len(df) * cfg.subset)].reset_index(drop=True)

    # 用训练集（前70%）统计气象归一化参数
    split_n       = int(len(df) * 0.70)
    train_df_full = df.iloc[:split_n]
    _, weather_stats = normalize_weather(train_df_full)

    train_ds = SolarDataset15min(df, weather_stats, phase='train')
    val_ds   = SolarDataset15min(df, weather_stats, phase='val')
    test_ds  = SolarDataset15min(df, weather_stats, phase='test')

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    cap        = train_ds.cap
    global_avg = train_ds.global_avg_power

    log_print(f"\n装机容量    : {cap} kW")
    log_print(f"全局均值    : {global_avg:.3f} kW  (有效发电时段 >0.2kW)\n")

    # ---------- 模型 ----------
    model = Model(cfg).to(device)

    # 调整所有 Dropout
    for _, m in model.named_modules():
        if isinstance(m, nn.Dropout):
            m.p = cfg.dropout

    total_params = sum(p.numel() for p in model.parameters())
    log_print(f"模型参数量  : {total_params:,}\n")

    # Xavier 初始化
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    model.apply(_init)

    # ---------- 优化器 & 学习率 ----------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr_init,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999)
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=cfg.warmup_epochs
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.t_max, eta_min=cfg.lr_final
            ),
        ],
        milestones=[cfg.warmup_epochs]
    )

    # ---------- 损失 / TensorBoard ----------
    loss_fn   = nn.MSELoss()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    writer    = SummaryWriter(log_dir=f"runs/solar15min_{timestamp}")
    log_print(f"TensorBoard 日志: runs/solar15min_{timestamp}")

    # ---------- 早停 ----------
    best_val_loss  = float('inf')
    best_ckpt_path = ''
    patience_cnt   = 0
    patience       = cfg.patience
    start_epoch    = 0

    # ---------- 断点续训 ----------
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch   = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('val_loss', float('inf'))
        patience_cnt  = ckpt.get('patience_counter', 0)
        log_print(f"断点续训    : epoch {start_epoch}，最佳 val_loss={best_val_loss:.6f}\n")

    os.makedirs("./checkpoint", exist_ok=True)
    os.makedirs("./prediction_plots", exist_ok=True)

    # ============================================================
    # 训练循环
    # ============================================================
    log_print("=" * 80)
    log_print("开始训练...")
    log_print("=" * 80 + "\n")

    for ep in range(start_epoch, cfg.epochs):

        # ==================== 训练阶段 ====================
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Ep {ep:3d} [Train]", ncols=110)
        for hist, fut in pbar:
            hist = hist.to(device)
            fut  = fut.to(device)

            fut_norm = fut / cap

            optimizer.zero_grad()
            pred = model(hist, None)  # [B, 16, 1]
            loss = loss_fn(pred.squeeze(-1), fut_norm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})

        train_loss /= len(train_loader)

        # ==================== 验证阶段 ====================
        model.eval()
        val_loss = 0.0
        sum_mae = sum_acc_mae = sum_acc_rmse = 0.0
        valid_cnt = 0
        all_true, all_pred = [], []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Ep {ep:3d} [Val]  ", ncols=110)
            for hist, fut in pbar:
                hist     = hist.to(device)
                fut      = fut.to(device)
                fut_norm = fut / cap

                pred      = model(hist, None)
                pred_norm = pred.squeeze(-1)
                loss      = loss_fn(pred_norm, fut_norm)
                val_loss += loss.item()

                fut_kw  = fut
                pred_kw = pred_norm * cap

                mask = fut_kw > 0.2
                if mask.sum() > 0:
                    ft = fut_kw[mask].cpu().numpy()
                    pp = pred_kw[mask].cpu().numpy()

                    all_true.extend(ft.tolist())
                    all_pred.extend(pp.tolist())

                    mae  = float(np.mean(np.abs(pp - ft)))
                    rmse = float(np.sqrt(np.mean((pp - ft) ** 2)))

                    sum_mae      += mae
                    sum_acc_mae  += max(0.0, min(1.0, 1.0 - mae  / (global_avg + 1e-6)))
                    sum_acc_rmse += max(0.0, min(1.0, 1.0 - rmse / (global_avg + 1e-6)))
                    valid_cnt    += 1

                pbar.set_postfix({'loss': f"{loss.item():.6f}"})

        val_loss     /= len(val_loader)
        val_mae       = sum_mae      / max(valid_cnt, 1)
        val_acc_mae   = sum_acc_mae  / max(valid_cnt, 1)
        val_acc_rmse  = sum_acc_rmse / max(valid_cnt, 1)

        if len(all_true) > 0:
            arr_t    = np.array(all_true, dtype=np.float32)
            arr_p    = np.array(all_pred, dtype=np.float32)
            val_acc2 = calc_acc2(arr_t, arr_p, cap=cap)
        else:
            val_acc2 = float('nan')

        lr = optimizer.param_groups[0]['lr']

        # ---------- 关键总结输出：写终端 + 写 txt ----------
        log_tqdm_write(f"\n{'='*80}")
        log_tqdm_write(f"Epoch {ep:3d} | LR: {lr:.2e}")
        log_tqdm_write(f"{'='*80}")
        log_tqdm_write(f"  Train Loss   : {train_loss:.6f}")
        log_tqdm_write(f"  Val   Loss   : {val_loss:.6f}  (Gap {val_loss/(train_loss+1e-12):.2f}x)")
        log_tqdm_write(f"  Val   MAE    : {val_mae:.4f} kW")
        log_tqdm_write(f"  ACC_MAE      : {val_acc_mae:.4f}  ({val_acc_mae*100:.2f}%)")
        log_tqdm_write(f"  ACC_RMSE     : {val_acc_rmse:.4f}  ({val_acc_rmse*100:.2f}%)")
        log_tqdm_write(f"  ACC2 (国标)  : {val_acc2:.4f}  ({val_acc2*100:.2f}%)")
        log_tqdm_write(f"{'='*80}\n")

        # ---------- TensorBoard ----------
        writer.add_scalar("Loss/train",      train_loss,   ep)
        writer.add_scalar("Loss/val",         val_loss,     ep)
        writer.add_scalar("Loss/gap",         val_loss / (train_loss + 1e-12), ep)
        writer.add_scalar("Metric/mae_kw",   val_mae,      ep)
        writer.add_scalar("Metric/acc_mae",  val_acc_mae,  ep)
        writer.add_scalar("Metric/acc_rmse", val_acc_rmse, ep)
        writer.add_scalar("Metric/acc2", 0.0 if np.isnan(val_acc2) else val_acc2, ep)
        writer.add_scalar("LearningRate",     lr,           ep)

       
        # ---------- Checkpoint ----------
        ckpt_data = {
            'epoch':                ep,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss':             val_loss,
            'val_acc_mae':          val_acc_mae,
            'val_acc_rmse':         val_acc_rmse,
            'val_acc2':             val_acc2,
            'patience_counter':     patience_cnt,
            'cap':                  cap,
            'global_avg_power':     global_avg,
            'weather_stats':        weather_stats,
        }
        torch.save(ckpt_data, f'./checkpoint/ckpt_epoch{ep}.pth')

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_ckpt_path = f'./checkpoint/best_epoch{ep}.pth'
            patience_cnt   = 0
            torch.save(ckpt_data, best_ckpt_path)
            log_tqdm_write(f"  ✓ 保存最佳模型  val_loss={val_loss:.6f}\n")
        else:
            patience_cnt += 1
            log_tqdm_write(f"  早停计数: {patience_cnt}/{patience}\n")
            if patience_cnt >= patience:
                log_tqdm_write(f"{'='*80}")
                log_tqdm_write(f"早停触发（连续 {patience} epoch 无改善）")
                log_tqdm_write(f"最佳 val_loss: {best_val_loss:.6f}")
                log_tqdm_write(f"{'='*80}\n")
                break

        scheduler.step()

    writer.close()
    log_print("\n" + "=" * 80)
    log_print("训练完成！")
    log_print(f"最佳 val_loss : {best_val_loss:.6f}")
    log_print("=" * 80)

    # ============================================================
    # 测试集评估（加载最佳模型，跑一次 test_loader）
    # ============================================================
    if best_ckpt_path and os.path.exists(best_ckpt_path):
        log_print(f"\n加载最佳模型: {best_ckpt_path}")
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        test_loss  = 0.0
        all_true_steps = []
        all_pred_steps = []

        with torch.no_grad():
            pbar = tqdm(test_loader, desc="[Test]  ", ncols=110)
            for hist, fut in pbar:
                hist     = hist.to(device)
                fut      = fut.to(device)
                fut_norm = fut / cap

                pred      = model(hist, None)
                pred_norm = pred.squeeze(-1)
                loss      = loss_fn(pred_norm, fut_norm)
                test_loss += loss.item()

                all_true_steps.append(fut.cpu().numpy())               # [B, 16] kW
                all_pred_steps.append((pred_norm * cap).cpu().numpy()) # [B, 16] kW

        test_loss /= len(test_loader)

        true_mat = np.concatenate(all_true_steps, axis=0)   # [N, 16]
        pred_mat = np.concatenate(all_pred_steps, axis=0)   # [N, 16]
        out_len = true_mat.shape[1]

        # 逐步指标
        step_mae      = np.zeros(out_len)
        step_acc_mae  = np.zeros(out_len)
        step_acc_rmse = np.zeros(out_len)
        step_acc2     = np.zeros(out_len)

        for s in range(out_len):
            t = true_mat[:, s]
            p = pred_mat[:, s]
            mask = t > 0.2
            if mask.sum() == 0:
                step_mae[s] = step_acc_mae[s] = step_acc_rmse[s] = step_acc2[s] = float('nan')
                continue
            tf, pf = t[mask], p[mask]
            mae  = float(np.mean(np.abs(pf - tf)))
            rmse = float(np.sqrt(np.mean((pf - tf) ** 2)))
            step_mae[s]      = mae
            step_acc_mae[s]  = max(0.0, min(1.0, 1.0 - mae  / (global_avg + 1e-6)))
            step_acc_rmse[s] = max(0.0, min(1.0, 1.0 - rmse / (global_avg + 1e-6)))
            step_acc2[s]     = calc_acc2(tf, pf, cap=cap)

        # 整体（展平）
        mask_all = true_mat > 0.2
        tf_all = true_mat[mask_all]
        pf_all = pred_mat[mask_all]
        overall_mae  = float(np.mean(np.abs(pf_all - tf_all)))
        overall_rmse = float(np.sqrt(np.mean((pf_all - tf_all) ** 2)))
        overall_acc_mae  = max(0.0, min(1.0, 1.0 - overall_mae  / (global_avg + 1e-6)))
        overall_acc_rmse = max(0.0, min(1.0, 1.0 - overall_rmse / (global_avg + 1e-6)))
        overall_acc2     = calc_acc2(tf_all, pf_all, cap=cap)

        # 打印（写入 txt）
        log_print(f"\n{'='*80}")
        log_print("测试集最终结果（最佳模型）— 逐步指标")
        log_print(f"{'='*80}")
        log_print(f"  Test Loss : {test_loss:.6f}")
        log_print("")
        log_print(f"  {'步':>3}  {'时间':>6}  {'MAE(kW)':>8}  {'ACC_MAE':>8}  {'ACC_RMSE':>9}  {'ACC2':>8}")
        log_print(f"  {'-'*3}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*8}")
        for s in range(out_len):
            t_min = (s + 1) * 15
            log_print(f"  {s+1:>3}  {t_min:>4}min"
                      f"  {step_mae[s]:>8.4f}"
                      f"  {step_acc_mae[s]:>7.2%}"
                      f"  {step_acc_rmse[s]:>8.2%}"
                      f"  {step_acc2[s]:>7.2%}")
        log_print(f"  {'-'*3}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*8}")
        log_print(f"  {'均值':>3}  {'  ALL':>6}"
                  f"  {overall_mae:>8.4f}"
                  f"  {overall_acc_mae:>7.2%}"
                  f"  {overall_acc_rmse:>8.2%}"
                  f"  {overall_acc2:>7.2%}")
        log_print(f"{'='*80}\n")


# ============================================================
# 参数解析
# ============================================================

def get_args():
    p = argparse.ArgumentParser(
        description='Solar Station 1 — 15分钟粒度 Transformer 训练',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python train_solar15.py --device 0
  python train_solar15.py --resume ./checkpoint/best_epoch5.pth
  python train_solar15.py --subset 0.1 --epochs 3 --device cpu
  tensorboard --logdir runs/
        """
    )
    p.add_argument('--epochs',            type=int,   default=100)
    p.add_argument('--batch-size',        type=int,   default=64)
    p.add_argument('--device',            type=str,   default='0')
    p.add_argument('--num-workers',       type=int,   default=4)
    p.add_argument('--data-path',         type=str,   default='./solar_station_1.csv')
    p.add_argument('--lr-init',           type=float, default=0.0004)
    p.add_argument('--lr-final',          type=float, default=0.00001)

    # 15分钟粒度
    p.add_argument('--in-seq-len',        type=int,   default=80,
                   help='20h × 4步/h = 80')
    p.add_argument('--out-seq-len',       type=int,   default=16,
                   help=' 4h × 4步/h = 16')

    # 17维特征
    p.add_argument('--in-feat-size',      type=int,   default=17,
                   help='10时间 + 6气象 + 1功率')
    p.add_argument('--out-feat-size',     type=int,   default=1)
    p.add_argument('--hidden-feat-size',  type=int,   default=256)

    p.add_argument('--subset',            type=float, default=1.0)
    p.add_argument('--resume',            type=str,   default='')

    # ---- 训练超参数 ----
    p.add_argument('--dropout',           type=float, default=0.25,
                   help='所有 Dropout 层的丢弃率')
    p.add_argument('--weight-decay',      type=float, default=0.04,
                   help='AdamW 权重衰减')
    p.add_argument('--warmup-epochs',     type=int,   default=2,
                   help='线性预热轮数')
    p.add_argument('--t-max',             type=int,   default=50,
                   help='CosineAnnealingLR 的 T_max')
    p.add_argument('--patience',          type=int,   default=10,
                   help='早停耐心值：连续 N 轮 val_loss 无改善则停止')

    return p.parse_args()


if __name__ == '__main__':
    cfg = get_args()
    train_val(cfg)
