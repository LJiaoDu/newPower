#!/usr/bin/env python3
"""
Solar Station 1 — 15分钟粒度训练脚本
基于 ImprovedTFMModel（非自回归 Transformer）

相对原 train.py 的变更：
1. 数据集  : training_data.csv → solar_station_1.csv
2. 时间粒度: 5min → 15min
     in_seq_len :  240 → 80   (20小时 × 4步/时)
     out_seq_len:   48 → 16   ( 4小时 × 4步/时)
3. 新增指标: ACC2（国标）
"""

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model import ImprovedTFMModel as Model
from tqdm import tqdm
import argparse
import os
import csv
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# 评估指标
# ============================================================

def calc_acc2(y_true: np.ndarray, y_pred: np.ndarray, cap: float = 1.0) -> float:
    """
    ACC2（国标）:
        1 - sqrt( (1/N) * sum( ((P_M - P_P) / max(P_M, 0.2*Cap))^2 ) )

    参数
    ----
    y_true / y_pred : 原始功率值（单位与 cap 相同，如 W 或 kW）
    cap             : 装机容量（默认用训练集最大功率代替）
    """
    p_m = y_true.flatten()
    p_p = y_pred.flatten()
    denom = np.maximum(p_m, 0.2 * cap)
    return max(0.0, 1.0 - float(np.sqrt(np.mean(((p_m - p_p) / denom) ** 2))))


# ============================================================
# 数据集
# ============================================================

class SolarDataset15min(Dataset):
    """
    15分钟粒度太阳能发电数据集。

    期望 CSV 列顺序（与 training_data.csv / process_data.py 输出一致）：
        col 0  datetime
        col 1  generationPower
        col 2  year
        col 3  month
        col 4  day
        col 5  hour
        col 6  minute
        col 7  dayofweek
        col 8  dayofyear
        col 9  time_idx

    输入序列  : 80 步 × 11 特征（20小时历史）
    预测序列  : 16 步 × 1  特征（未来4小时功率）
    """

    IN_LEN  = 20 * 4   # 80
    OUT_LEN =  4 * 4   # 16

    def __init__(self, cfg, phase: str = 'train'):
        super().__init__()

        # ---------- 读取 CSV ----------
        rows = []
        with open(cfg.data_path, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)          # 跳过表头
            for row in reader:
                rows.append(row)

        data = np.array(rows)

        if cfg.subset < 1.0:
            data = data[:int(len(data) * cfg.subset)]

        self.features, self.targets = self._build_features(data)

        # ---------- 时间序列切分（不打乱顺序） ----------
        split = int(len(self.features) * 0.8)
        if phase == 'train':
            self.features = self.features[:split]
            self.targets  = self.targets[:split]
        else:
            self.features = self.features[split:]
            self.targets  = self.targets[split:]

    # ----------------------------------------------------------
    def _build_features(self, data: np.ndarray):
        power     = data[:, 1].astype(np.float32)
        month     = data[:, 3].astype(np.float32)
        hour      = data[:, 5].astype(np.float32)
        minute    = data[:, 6].astype(np.float32)
        dayofweek = data[:, 7].astype(np.float32)
        dayofyear = data[:, 8].astype(np.float32)
        time_idx  = data[:, 9].astype(np.float32)

        feats = []

        # 1. 小时归一化 [0,1]
        hour_of_day = (hour + minute / 60.0) / 24.0
        feats.append(hour_of_day)

        # 2. 小时周期编码（sin/cos）
        ha = 2 * np.pi * hour_of_day
        feats.append(np.sin(ha))
        feats.append(np.cos(ha))

        # 3. 星期几周期编码
        da = 2 * np.pi * dayofweek / 7.0
        feats.append(np.sin(da))
        feats.append(np.cos(da))

        # 4. 年内第几天周期编码
        ya = 2 * np.pi * dayofyear / 365.0
        feats.append(np.sin(ya))
        feats.append(np.cos(ya))

        # 5. 月份周期编码
        ma = 2 * np.pi * (month - 1) / 12.0
        feats.append(np.sin(ma))
        feats.append(np.cos(ma))

        # 6. 时间索引归一化（捕捉长期趋势）
        t_norm = (time_idx - time_idx.min()) / (time_idx.max() - time_idx.min() + 1e-8)
        feats.append(t_norm)

        # 7. 历史功率归一化
        self.power_max = float(power.max())
        self.power_min = float(power.min())
        power_norm = power / (self.power_max + 1e-8)
        feats.append(power_norm)

        features = np.stack(feats, axis=1).astype(np.float32)   # [N, 11]
        targets  = power_norm                                    # [N]

        # 全局均值（仅有效发电时段 > 200W，用于 ACC 分母）
        valid = power[power > 200]
        self.global_avg_power = float(np.mean(valid)) if len(valid) > 0 else 1.0

        print(f"样本总数   : {len(features)}")
        print(f"特征维度   : {features.shape[1]}  (10个时间特征 + 1个功率特征)")
        print(f"功率范围   : {self.power_min:.1f} ~ {self.power_max:.1f} W")
        print(f"全局均值   : {self.global_avg_power:.1f} W  (有效发电时段 >200W)")

        return features, targets

    # ----------------------------------------------------------
    def __len__(self):
        return len(self.features) - self.IN_LEN - self.OUT_LEN

    def __getitem__(self, idx):
        s = idx
        m = idx + self.IN_LEN
        e = m   + self.OUT_LEN
        return (
            self.features[s:m],   # [80, 11]  历史特征
            self.targets[m:e],    # [16]      未来功率（归一化）
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

    print("=" * 80)
    print(f"设备        : {device}")
    print(f"数据集      : {cfg.data_path}")
    print(f"粒度        : 15 分钟")
    print(f"输入序列    : {cfg.in_seq_len} 步（20h）")
    print(f"预测序列    : {cfg.out_seq_len} 步（4h）")
    print("=" * 80 + "\n")

    # ---------- 数据 ----------
    train_ds = SolarDataset15min(cfg, phase='train')
    val_ds   = SolarDataset15min(cfg, phase='val')

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    print(f"\n训练集样本  : {len(train_ds)}")
    print(f"验证集样本  : {len(val_ds)}")

    global_avg = train_ds.global_avg_power
    cap        = train_ds.power_max        # 装机容量代理值（实测最大功率）

    # ---------- 模型 ----------
    model = Model(cfg).to(device)

    # 调整所有 Dropout → 0.25
    for name, m in model.named_modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.25

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量  : {total_params:,}\n")

    # Xavier 初始化
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    model.apply(_init)

    # ---------- 优化器 ----------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr_init,
        weight_decay=0.04,
        betas=(0.9, 0.999)
    )

    # ---------- 学习率调度 ----------
    warmup_epochs = 2
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                     total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=30, eta_min=cfg.lr_final),
        ],
        milestones=[warmup_epochs]
    )

    print(f"学习率调度  : Warmup {warmup_epochs} epochs → Cosine Annealing")
    print(f"  峰值 LR   : {cfg.lr_init:.2e}")
    print(f"  最终 LR   : {cfg.lr_final:.2e}\n")

    # ---------- 损失 / TensorBoard ----------
    loss_fn   = nn.MSELoss()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    writer    = SummaryWriter(log_dir=f'runs/solar15min_{timestamp}')

    # ---------- 早停 ----------
    best_val_loss = float('inf')
    patience_cnt  = 0
    patience      = 3
    start_epoch   = 0

    # ---------- 断点续训 ----------
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch   = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('val_loss', float('inf'))
        patience_cnt  = ckpt.get('patience_counter', 0)
        print(f"断点续训    : epoch {start_epoch}，最佳 val_loss={best_val_loss:.6f}\n")

    os.makedirs('./checkpoint',      exist_ok=True)
    os.makedirs('./prediction_plots', exist_ok=True)

    # ============================================================
    # 训练循环
    # ============================================================
    print("=" * 80)
    print("开始训练...")
    print("=" * 80 + "\n")

    for ep in range(start_epoch, cfg.epochs):

        # -------------------- 训练阶段 --------------------
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Ep {ep:3d} [Train]", ncols=110)
        for hist, fut in pbar:
            hist = hist.to(device)   # [B, 80, 11]
            fut  = fut.to(device)    # [B, 16]

            optimizer.zero_grad()
            pred = model(hist, None)                         # [B, 16, 1]
            loss = loss_fn(pred.squeeze(-1), fut)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})

        train_loss /= len(train_loader)

        # -------------------- 验证阶段 --------------------
        model.eval()
        val_loss = 0.0
        sum_mae = sum_acc_mae = sum_acc_rmse = 0.0
        valid_cnt = 0
        all_true, all_pred = [], []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Ep {ep:3d} [Val]  ", ncols=110)
            for hist, fut in pbar:
                hist = hist.to(device)
                fut  = fut.to(device)

                pred = model(hist, None)
                loss = loss_fn(pred.squeeze(-1), fut)
                val_loss += loss.item()

                # 只对有效发电时段（>200W）计算准确率，避免夜间零值虚拉指标
                mask = (fut * train_ds.power_max) > 200
                if mask.sum() > 0:
                    fut_raw  = fut[mask]                  * train_ds.power_max  # W
                    pred_raw = pred.squeeze(-1)[mask]     * train_ds.power_max

                    all_true.extend(fut_raw.cpu().numpy().tolist())
                    all_pred.extend(pred_raw.cpu().numpy().tolist())

                    mae  = torch.mean(torch.abs(pred_raw - fut_raw)).item()
                    rmse = torch.sqrt(torch.mean((pred_raw - fut_raw) ** 2)).item()

                    sum_mae      += mae
                    sum_acc_mae  += max(0.0, min(1.0, 1.0 - mae  / (global_avg + 1e-6)))
                    sum_acc_rmse += max(0.0, min(1.0, 1.0 - rmse / (global_avg + 1e-6)))
                    valid_cnt    += 1

                pbar.set_postfix({'loss': f"{loss.item():.6f}"})

        val_loss     /= len(val_loader)
        val_mae       = sum_mae      / max(valid_cnt, 1)
        val_acc_mae   = sum_acc_mae  / max(valid_cnt, 1)
        val_acc_rmse  = sum_acc_rmse / max(valid_cnt, 1)

        # ---------- ACC2（国标） ----------
        if len(all_true) > 0:
            arr_true = np.array(all_true, dtype=np.float32)
            arr_pred = np.array(all_pred, dtype=np.float32)
            val_acc2 = calc_acc2(arr_true, arr_pred, cap=cap)
        else:
            val_acc2 = float('nan')

        lr = optimizer.param_groups[0]['lr']

        # ---------- 控制台输出 ----------
        print(f"\n{'='*80}")
        print(f"Epoch {ep:3d} | LR: {lr:.2e}  | 装机容量代理: {cap:.0f} W")
        print(f"{'='*80}")
        print(f"  Train Loss   : {train_loss:.6f}")
        print(f"  Val   Loss   : {val_loss:.6f}  (Gap {val_loss/train_loss:.2f}x)")
        print(f"  Val   MAE    : {val_mae:.1f} W")
        print(f"  ACC_MAE      : {val_acc_mae:.4f}  ({val_acc_mae*100:.2f}%)")
        print(f"  ACC_RMSE     : {val_acc_rmse:.4f}  ({val_acc_rmse*100:.2f}%)")
        print(f"  ACC2 (国标)  : {val_acc2:.4f}  ({val_acc2*100:.2f}%)")
        print(f"{'='*80}\n")

        # ---------- TensorBoard ----------
        writer.add_scalar("Loss/train",       train_loss,   ep)
        writer.add_scalar("Loss/val",          val_loss,     ep)
        writer.add_scalar("Loss/gap",          val_loss / train_loss, ep)
        writer.add_scalar("Metric/mae",        val_mae,      ep)
        writer.add_scalar("Metric/acc_mae",    val_acc_mae,  ep)
        writer.add_scalar("Metric/acc_rmse",   val_acc_rmse, ep)
        writer.add_scalar("Metric/acc2",       val_acc2 if not np.isnan(val_acc2) else 0.0, ep)
        writer.add_scalar("LearningRate",      lr,           ep)

        # ---------- 预测曲线图 ----------
        if len(all_true) > 0:
            max_pts = 2000
            if len(all_true) > max_pts:
                idx_s = np.linspace(0, len(all_true) - 1, max_pts, dtype=int)
                pt = [all_true[i] for i in idx_s]
                pp = [all_pred[i] for i in idx_s]
            else:
                pt, pp = all_true, all_pred

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

            ax1.plot(pt, 'b-', label='True',      alpha=0.7, linewidth=1.5)
            ax1.plot(pp, 'r-', label='Predicted', alpha=0.7, linewidth=1.5)
            ax1.set_title(
                f"Epoch {ep}  |  ACC_MAE={val_acc_mae:.2%}  "
                f"ACC2={val_acc2:.2%}  MAE={val_mae:.1f}W",
                fontsize=13
            )
            ax1.set_xlabel('样本索引');  ax1.set_ylabel('功率 (W)')
            ax1.legend(loc='upper right'); ax1.grid(alpha=0.3)

            mn = min(min(pt), min(pp));  mx = max(max(pt), max(pp))
            ax2.scatter(pt, pp, s=8, alpha=0.3, c='steelblue')
            ax2.plot([mn, mx], [mn, mx], 'r--', lw=2, label='完美预测')
            ax2.set_xlabel('真实值 (W)'); ax2.set_ylabel('预测值 (W)')
            ax2.set_title('散点图：真实值 vs 预测值', fontsize=13)
            ax2.legend(loc='upper left'); ax2.grid(alpha=0.3)
            ax2.set_aspect('equal', adjustable='box')

            plt.tight_layout()
            plot_path = f'prediction_plots/epoch_{ep:03d}.png'
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            writer.add_figure('Predictions/comparison', fig, ep)
            plt.close(fig)
            print(f"  预测图已保存: {plot_path}")

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
            'power_max':            train_ds.power_max,
            'power_min':            train_ds.power_min,
            'global_avg_power':     global_avg,
            'cap':                  cap,
        }
        torch.save(ckpt_data, f'./checkpoint/ckpt_epoch{ep}.pth')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
            torch.save(ckpt_data, f'./checkpoint/best_epoch{ep}.pth')
            print(f"  ✓ 保存最佳模型  val_loss={val_loss:.6f}\n")
        else:
            patience_cnt += 1
            print(f"  早停计数: {patience_cnt}/{patience}\n")
            if patience_cnt >= patience:
                print(f"{'='*80}")
                print(f"早停触发（连续 {patience} epoch 无改善）")
                print(f"最佳 val_loss: {best_val_loss:.6f}")
                print(f"{'='*80}\n")
                break

        scheduler.step()

    writer.close()
    print("\n" + "=" * 80)
    print("训练完成！")
    print(f"最佳 val_loss : {best_val_loss:.6f}")
    print("=" * 80)


# ============================================================
# 参数解析
# ============================================================

def get_args():
    p = argparse.ArgumentParser(
        description='Solar Station 1 — 15分钟粒度 Transformer 训练',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # GPU 训练
  python train_solar15.py --device 0

  # 断点续训
  python train_solar15.py --resume ./checkpoint/best_epoch5.pth

  # 快速冒烟测试（10%数据）
  python train_solar15.py --subset 0.1 --epochs 3 --device cpu

  # 查看 TensorBoard
  tensorboard --logdir runs/
        """
    )

    # 训练参数
    p.add_argument('--epochs',           type=int,   default=100)
    p.add_argument('--batch-size',       type=int,   default=64)
    p.add_argument('--device',           type=str,   default='0',
                   help='GPU 设备号，或 "cpu"')
    p.add_argument('--num-workers',      type=int,   default=4)
    p.add_argument('--data-path',        type=str,   default='./solar_station_1.csv')

    # 学习率
    p.add_argument('--lr-init',          type=float, default=0.0004)
    p.add_argument('--lr-final',         type=float, default=0.00001)

    # 模型结构（15min 粒度）
    p.add_argument('--in-seq-len',       type=int,   default=80,
                   help='输入序列步数（20h × 4步/h = 80）')
    p.add_argument('--out-seq-len',      type=int,   default=16,
                   help='预测序列步数（4h × 4步/h = 16）')
    p.add_argument('--in-feat-size',     type=int,   default=11)
    p.add_argument('--out-feat-size',    type=int,   default=1)
    p.add_argument('--hidden-feat-size', type=int,   default=256)

    # 其他
    p.add_argument('--subset',           type=float, default=1.0,
                   help='数据子集比例 (0~1)，<1 时用于快速测试')
    p.add_argument('--resume',           type=str,   default='',
                   help='断点续训的 checkpoint 路径')

    return p.parse_args()


if __name__ == '__main__':
    cfg = get_args()
    train_val(cfg)
