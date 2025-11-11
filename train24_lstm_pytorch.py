import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pickle
import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks import TQDMProgressBar

torch.set_float32_matmul_precision('high')

class PowerDataset(Dataset):
    """电力数据集"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMPowerPredictor(pl.LightningModule):


    def __init__(self, args, y_scaler=None, output_size=48):
        super().__init__()

        self.save_hyperparameters(ignore=['y_scaler', 'args'])
        self.y_scaler = y_scaler
        self.args = args
        self.output_size = output_size


        self.lstm1 = nn.LSTM(
            input_size=1,
            hidden_size=args.lstm1_hidden,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(args.dropout)

        self.lstm2 = nn.LSTM(
            input_size=args.lstm1_hidden * 2,
            hidden_size=args.lstm2_hidden,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.dropout2 = nn.Dropout(args.dropout)

        self.fc1 = nn.Linear(args.lstm2_hidden * 2, args.fc1_hidden)
        self.dropout3 = nn.Dropout(args.dropout)
        self.fc2 = nn.Linear(args.fc1_hidden, args.fc2_hidden)
        self.fc3 = nn.Linear(args.fc2_hidden, self.output_size)

        self.validation_step_outputs = []
        self.validation_step_targets = []

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.dropout2(x)

        x = torch.relu(self.fc1(x))
        x = self.dropout3(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)

        loss = nn.functional.mse_loss(y_pred, y)
        mae = torch.mean(torch.abs(y_pred - y))

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_mae', mae, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)

        loss = nn.functional.mse_loss(y_pred, y)
        mae = torch.mean(torch.abs(y_pred - y))

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mae', mae, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.validation_step_outputs.append(y_pred.detach().cpu())
        self.validation_step_targets.append(y.detach().cpu())

        return loss

    def on_validation_epoch_end(self):
        if len(self.validation_step_outputs) == 0:
            return

        all_preds = torch.cat(self.validation_step_outputs, dim=0).numpy()
        all_targets = torch.cat(self.validation_step_targets, dim=0).numpy()

        if self.y_scaler is not None:
            all_preds = self.y_scaler.inverse_transform(all_preds)
            all_targets = self.y_scaler.inverse_transform(all_targets)

        acc_ = self.calculate_acc_(all_targets, all_preds)
        acc_mae = self.calculate_acc_mae(all_targets, all_preds)

        self.log('val_acc_', acc_, prog_bar=True, logger=True)
        self.log('val_acc_mae', acc_mae, prog_bar=True, logger=True)

        self.validation_step_outputs.clear()
        self.validation_step_targets.clear()

    def calculate_acc_(self, y_actual, y_pred):
        y_actual_flat = y_actual.flatten()
        y_pred_flat = y_pred.flatten()
        mask = y_actual_flat != 0
        y_actual_nonzero = y_actual_flat[mask]
        y_pred_nonzero = y_pred_flat[mask]

        if len(y_actual_nonzero) == 0:
            return 0.0

        relative_errors = (y_actual_nonzero - y_pred_nonzero) / y_actual_nonzero
        squared_errors = relative_errors ** 2
        rmse_relative = np.sqrt(np.mean(squared_errors))
        acc_ = 1 - rmse_relative

        return acc_ * 100

    def calculate_acc_mae(self, y_actual, y_pred):
        y_actual_flat = y_actual.flatten()
        y_pred_flat = y_pred.flatten()
        mask = y_actual_flat != 0
        y_actual_nonzero = y_actual_flat[mask]
        y_pred_nonzero = y_pred_flat[mask]

        if len(y_actual_nonzero) == 0:
            return 0.0

        relative_errors = np.abs((y_actual_nonzero - y_pred_nonzero) / y_actual_nonzero)
        mae_relative = np.mean(relative_errors)
        acc_mae = 1 - mae_relative

        return acc_mae * 100

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.args.lr_factor,
            patience=self.args.lr_patience,
            min_lr=self.args.lr_min,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }


def load_data(data_path):

    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"总数据点: {len(df)}")
    print(f"时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
    return df


def split_raw_data(df, train_ratio=0.8):

    split_idx = int(len(df) * train_ratio)

    df_train = df[:split_idx].copy()
    df_val = df[split_idx:].copy()
    print(f"训练集原始数据: {len(df_train)} 点 ({df_train['datetime'].min()} 到 {df_train['datetime'].max()})")
    print(f"验证集原始数据: {len(df_val)} 点 ({df_val['datetime'].min()} 到 {df_val['datetime'].max()})")

    return df_train, df_val


def create_sequences(df, lookback_points, forecast_points, dataset_name="训练"):

    power_values = df['generationPower'].values

    X = []
    y = []

    total_window = lookback_points + forecast_points

    for i in range(len(power_values) - total_window + 1):
        lookback_power = power_values[i:i+lookback_points]
        forecast_power = power_values[i+lookback_points:i+total_window]

        X.append(lookback_power)
        y.append(forecast_power)

    X = np.array(X)
    y = np.array(y)

    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y


def normalize_data(X_train, y_train, X_val, y_val):

    X_scaler = StandardScaler()
    X_train_scaled = X_train.reshape(-1, 1)
    X_train_scaled = X_scaler.fit_transform(X_train_scaled)
    X_train_scaled = X_train_scaled.reshape(X_train.shape)

    X_val_scaled = X_val.reshape(-1, 1)
    X_val_scaled = X_scaler.transform(X_val_scaled)
    X_val_scaled = X_val_scaled.reshape(X_val.shape)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_val_scaled = y_scaler.transform(y_val)

    print("数据归一化完成")

    return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_scaler, y_scaler


def main(args):


    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if torch.cuda.is_available():
        print(f" 检测到GPU: {torch.cuda.get_device_name(0)}")

    LOOKBACK_POINTS = args.input_hours * 60 // 5
    FORECAST_POINTS = args.pre_hours * 60 // 5

    df = load_data(args.data_path)

    df_train, df_val = split_raw_data(df, args.train_ratio)

    X_train, y_train = create_sequences(df_train, LOOKBACK_POINTS, FORECAST_POINTS, "训练")
    X_val, y_val = create_sequences(df_val, LOOKBACK_POINTS, FORECAST_POINTS, "验证")


    X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_scaler, y_scaler = \
        normalize_data(X_train, y_train, X_val, y_val)

    train_dataset = PowerDataset(X_train_scaled, y_train_scaled)
    val_dataset = PowerDataset(X_val_scaled, y_val_scaled)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )


    model = LSTMPowerPredictor(args=args, y_scaler=y_scaler, output_size=FORECAST_POINTS)


    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=args.early_stop_patience,
            mode='min',
            verbose=True
        ),
        ModelCheckpoint(
            monitor='val_loss',
            dirpath=args.checkpoint_dir,
            filename='lstm-{epoch:02d}-{val_loss:.4f}',
            save_top_k=1,
            mode='min'
        ),
        LearningRateMonitor(logging_interval='epoch'),
        TQDMProgressBar(refresh_rate=10)
    ]


    if args.device == 'auto':
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'


    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_progress_bar=True,
        deterministic=True,
        precision=32,
    )


    ckpt_path = args.resume if args.resume else None
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    print("保存数据归一化器")
    with open('scalers_pytorch.pkl', 'wb') as f:
        pickle.dump({'X_scaler': X_scaler, 'y_scaler': y_scaler}, f)
    print("已保存到 scalers_pytorch.pkl")




def parse_args():

    parser = argparse.ArgumentParser(description='LSTM电力预测训练')
    parser.add_argument('--data-path', type=str, default='training_data.csv')
    parser.add_argument('--input-hours', type=int, default=20)
    parser.add_argument('--pre-hours', type=int, default=4)
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr-min', type=float, default=0.00001)
    parser.add_argument('--lstm1-hidden', type=int, default=128)
    parser.add_argument('--lstm2-hidden', type=int, default=64)
    parser.add_argument('--fc1-hidden', type=int, default=128)
    parser.add_argument('--fc2-hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--early-stop-patience', type=int, default=10)
    parser.add_argument('--lr-patience', type=int, default=5)
    parser.add_argument('--lr-factor', type=float, default=0.5)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    parser.add_argument('--resume', type=str, default='')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
