#!/usr/bin/env python3
"""
太阳能电站功率预测 - 训练脚本 (training_data.csv 版本)

对应关系:
  数据预处理: solar_preprocess_2.py  -> X_enc_*.npy, X_dec_*.npy, y_*.npy
  模型定义:   rope_model_2.py        -> SolarTransformerRoPE
  本脚本:     rope_2.py              -> 训练 + 验证 + 保存

与 rope_1.py / train.py 的主要区别:
  1. 数据格式: 三元组 (X_enc, X_dec, y) 而非二元组 (X, y)
  2. 序列长度: enc=288 (24h@5min), dec=48 (4h@5min)
  3. 特征维度: Encoder=7, Decoder=6 (无气象特征)
  4. 模型: SolarTransformerRoPE from rope_model_2
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
import argparse

from rope_model_2 import SolarTransformerRoPE


# ============== 评估指标 ==============

class MetricsCalculator:

    @staticmethod
    def calculate_acc1(y_true, y_pred):
        """趋势准确度: 预测趋势与真实趋势一致的比例"""
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()

        true_diff = np.diff(y_true, axis=1)
        pred_diff = np.diff(y_pred, axis=1)

        trend_match = (true_diff * pred_diff) > 0
        true_flat   = np.abs(true_diff) < 1e-6
        pred_flat   = np.abs(pred_diff) < 1e-6
        flat_match  = true_flat & pred_flat

        return float(np.mean(trend_match | flat_match))

    @staticmethod
    def calculate_acc2(y_true, y_pred, threshold=0.1):
        """阈值准确度: 相对误差在 threshold 以内的比例"""
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()

        rel_err = np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-6)
        return float(np.mean(rel_err <= threshold))

    @staticmethod
    def calculate_rmse(y_true, y_pred):
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def calculate_mae(y_true, y_pred):
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def calculate_mape(y_true, y_pred):
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        return float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100)


# ============== 训练器 ==============

class Trainer:

    def __init__(self, model, train_loader, val_loader,
                 criterion, optimizer, device, save_dir="checkpoints_rope2"):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.criterion    = criterion
        self.optimizer    = optimizer
        self.device       = device
        self.save_dir     = save_dir
        self.metrics      = MetricsCalculator()

        os.makedirs(save_dir, exist_ok=True)

        self.history = {
            "train_loss": [],
            "val_loss":   [],
            "val_acc1":   [],
            "val_acc2":   [],
            "val_rmse":   [],
            "val_mae":    [],
        }

    # ------------------------------------------------------------------
    def train_epoch(self):
        self.model.train()
        total_loss, n = 0.0, 0

        pbar = tqdm(self.train_loader, desc="Training")
        for x_enc, x_dec, y in pbar:
            x_enc = x_enc.to(self.device)
            x_dec = x_dec.to(self.device)
            y     = y.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(x_enc, x_dec)      # [B, 48]
            loss = self.criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n          += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / n

    # ------------------------------------------------------------------
    def validate(self):
        self.model.eval()
        total_loss, n = 0.0, 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for x_enc, x_dec, y in self.val_loader:
                x_enc = x_enc.to(self.device)
                x_dec = x_dec.to(self.device)
                y     = y.to(self.device)

                pred  = self.model(x_enc, x_dec)
                loss  = self.criterion(pred, y)
                total_loss += loss.item()
                n          += 1

                all_preds.append(pred.cpu())
                all_targets.append(y.cpu())

        all_preds   = torch.cat(all_preds,   dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        return {
            "loss": total_loss / n,
            "acc1": self.metrics.calculate_acc1(all_targets, all_preds),
            "acc2": self.metrics.calculate_acc2(all_targets, all_preds),
            "rmse": self.metrics.calculate_rmse(all_targets, all_preds),
            "mae":  self.metrics.calculate_mae(all_targets, all_preds),
            "mape": self.metrics.calculate_mape(all_targets, all_preds),
        }

    # ------------------------------------------------------------------
    def train(self, num_epochs, scheduler=None):
        best_val_loss   = float("inf")
        patience        = 10
        patience_counter = 0

        for epoch in range(num_epochs):
            print(f"\n{'='*55}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*55}")

            train_loss = self.train_epoch()
            print(f"训练损失: {train_loss:.4f}")

            val = self.validate()
            print(f"验证损失: {val['loss']:.4f}")
            print(f"验证ACC1 (趋势准确度): {val['acc1']:.4f}")
            print(f"验证ACC2 (阈值准确度): {val['acc2']:.4f}")
            print(f"验证RMSE: {val['rmse']:.6f}")
            print(f"验证MAE:  {val['mae']:.6f}")
            print(f"验证MAPE: {val['mape']:.2f}%")

            if scheduler is not None:
                scheduler.step(val["loss"])
                print(f"当前学习率: {self.optimizer.param_groups[0]['lr']:.2e}")

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val["loss"])
            self.history["val_acc1"].append(val["acc1"])
            self.history["val_acc2"].append(val["acc2"])
            self.history["val_rmse"].append(val["rmse"])
            self.history["val_mae"].append(val["mae"])

            if val["loss"] < best_val_loss:
                best_val_loss    = val["loss"]
                patience_counter = 0
                self._save_checkpoint(epoch, val, is_best=True)
                print(f"✓ 保存最佳模型 (验证损失: {best_val_loss:.4f})")
            else:
                patience_counter += 1

            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(epoch, val, is_best=False)

            if patience_counter >= patience:
                print(f"\nEarly stopping: 验证损失连续 {patience} 个 epoch 未改善")
                break

        print(f"\n{'='*55}")
        print(f"训练完成!  最佳验证损失: {best_val_loss:.4f}")
        print(f"{'='*55}")

        self._save_history()
        self._plot_history()
        return self.history

    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch, metrics, is_best):
        ckpt = {
            "epoch":               epoch,
            "model_state_dict":    self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics":             metrics,
            "history":             self.history,
        }
        fname = "best_model.pth" if is_best else f"checkpoint_epoch_{epoch + 1}.pth"
        torch.save(ckpt, os.path.join(self.save_dir, fname))

    def _save_history(self):
        path = os.path.join(self.save_dir, "training_history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=4)
        print(f"训练历史已保存: {path}")

    def _plot_history(self):
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        epochs = range(1, len(self.history["train_loss"]) + 1)

        axes[0, 0].plot(epochs, self.history["train_loss"], label="Train Loss")
        axes[0, 0].plot(epochs, self.history["val_loss"],   label="Val Loss")
        axes[0, 0].set_title("Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        axes[0, 1].plot(epochs, self.history["val_acc1"], color="green", label="ACC1")
        axes[0, 1].set_title("Validation ACC1 (趋势准确度)")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        axes[1, 0].plot(epochs, self.history["val_acc2"], color="orange", label="ACC2")
        axes[1, 0].set_title("Validation ACC2 (阈值准确度, 10%)")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        axes[1, 1].plot(epochs, self.history["val_rmse"], color="red", label="RMSE")
        axes[1, 1].set_title("Validation RMSE")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        path = os.path.join(self.save_dir, "training_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"训练曲线已保存: {path}")


# ============== 主训练流程 ==============

def main():
    parser = argparse.ArgumentParser(description="rope_2 训练脚本 (training_data.csv)")
    parser.add_argument("--data_dir",    type=str, default=".",
                        help="npy 数据文件所在目录")
    parser.add_argument("--save_dir",    type=str, default="checkpoints_rope2",
                        help="检查点保存目录")
    parser.add_argument("--epochs",      type=int, default=100)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--d_model",     type=int, default=256)
    parser.add_argument("--nhead",       type=int, default=8)
    parser.add_argument("--num_layers",  type=int, default=4)
    parser.add_argument("--ffn_dim",     type=int, default=1024)
    parser.add_argument("--dropout",     type=float, default=0.1)
    args = parser.parse_args()

    # 随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # ------------------------------------------------------------------
    # 1. 加载数据
    # ------------------------------------------------------------------
    print("\n=== 加载数据 ===")

    def load(split):
        d = args.data_dir
        return (
            np.load(os.path.join(d, f"X_enc_{split}.npy")),
            np.load(os.path.join(d, f"X_dec_{split}.npy")),
            np.load(os.path.join(d, f"y_{split}.npy")),
        )

    X_enc_tr, X_dec_tr, y_tr = load("train")
    X_enc_vl, X_dec_vl, y_vl = load("val")

    print(f"训练集: X_enc={X_enc_tr.shape}, X_dec={X_dec_tr.shape}, y={y_tr.shape}")
    print(f"验证集: X_enc={X_enc_vl.shape}, X_dec={X_dec_vl.shape}, y={y_vl.shape}")

    # ------------------------------------------------------------------
    # 2. DataLoader  (三元组 TensorDataset)
    # ------------------------------------------------------------------
    def make_loader(X_enc, X_dec, y, shuffle):
        ds = TensorDataset(
            torch.FloatTensor(X_enc),
            torch.FloatTensor(X_dec),
            torch.FloatTensor(y),
        )
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=(device.type == "cuda"))

    train_loader = make_loader(X_enc_tr, X_dec_tr, y_tr, shuffle=True)
    val_loader   = make_loader(X_enc_vl, X_dec_vl, y_vl, shuffle=False)

    # ------------------------------------------------------------------
    # 3. 模型
    # ------------------------------------------------------------------
    print("\n=== 创建模型 ===")

    # 从数据自动推断特征维度, 兜底使用预处理默认值
    enc_feat = X_enc_tr.shape[-1]   # 应为 7
    dec_feat = X_dec_tr.shape[-1]   # 应为 6
    enc_seq  = X_enc_tr.shape[1]    # 应为 288
    dec_seq  = X_dec_tr.shape[1]    # 应为 48

    model = SolarTransformerRoPE(
        enc_feat_size=enc_feat,
        dec_feat_size=dec_feat,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        dim_feedforward=args.ffn_dim,
        dropout=args.dropout,
        enc_seq_len=enc_seq,
        dec_seq_len=dec_seq,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Encoder 输入: [B, {enc_seq}, {enc_feat}]")
    print(f"Decoder 输入: [B, {dec_seq}, {dec_feat}]")
    print(f"模型参数量:   {total_params:,}")

    # ------------------------------------------------------------------
    # 4. 损失 / 优化器 / 调度器
    # ------------------------------------------------------------------
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # ------------------------------------------------------------------
    # 5. 训练
    # ------------------------------------------------------------------
    print("\n=== 开始训练 ===")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_dir=args.save_dir,
    )
    trainer.train(num_epochs=args.epochs, scheduler=scheduler)


if __name__ == "__main__":
    main()
