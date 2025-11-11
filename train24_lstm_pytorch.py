import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pickle
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks import TQDMProgressBar

# 配置参数
LOOKBACK_HOURS = 20     
FORECAST_HOURS = 4       
INTERVAL_MINUTES = 5    

LOOKBACK_POINTS = LOOKBACK_HOURS * 60 // INTERVAL_MINUTES  
FORECAST_POINTS = FORECAST_HOURS * 60 // INTERVAL_MINUTES  


np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class PowerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMPowerPredictor(pl.LightningModule):

    def __init__(self, y_scaler=None):
        super().__init__()

        self.save_hyperparameters(ignore=['y_scaler'])
        self.y_scaler = y_scaler

        self.lstm1 = nn.LSTM(
            input_size=1,
            hidden_size=128,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(0.2)

        self.lstm2 = nn.LSTM(
            input_size=256,  
            hidden_size=64,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(128, 128)  
        self.dropout3 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, FORECAST_POINTS)

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
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=True
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


def load_data():


    df = pd.read_csv('training_data.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"总数据点: {len(df)}")
    print(f"时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
    return df


def split_raw_data(df, train_ratio=0.8):
    print("\n划分原始数据集")
    split_idx = int(len(df) * train_ratio)

    df_train = df[:split_idx].copy()
    df_val = df[split_idx:].copy()

    print(f"训练集原始数据: {len(df_train)} 点 ({df_train['datetime'].min()} 到 {df_train['datetime'].max()})")
    print(f"验证集原始数据: {len(df_val)} 点 ({df_val['datetime'].min()} 到 {df_val['datetime'].max()})")

    return df_train, df_val


def create_sequences(df, dataset_name="训练"):

    power_values = df['generationPower'].values

    X = []
    y = []

    total_window = LOOKBACK_POINTS + FORECAST_POINTS

    for i in range(len(power_values) - total_window + 1):
        lookback_power = power_values[i:i+LOOKBACK_POINTS]
        forecast_power = power_values[i+LOOKBACK_POINTS:i+total_window]

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

    # 输出数据归一化
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_val_scaled = y_scaler.transform(y_val)

    print("数据归一化完成")

    return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_scaler, y_scaler


def main():

    df = load_data()
    df_train, df_val = split_raw_data(df)
    X_train, y_train = create_sequences(df_train, "训练")
    X_val, y_val = create_sequences(df_val, "验证")
    X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_scaler, y_scaler = \
        normalize_data(X_train, y_train, X_val, y_val)
    batch_size = 128

    train_dataset = PowerDataset(X_train_scaled, y_train_scaled)
    val_dataset = PowerDataset(X_val_scaled, y_val_scaled)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )


    model = LSTMPowerPredictor(y_scaler=y_scaler)


    callbacks = [
  
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min',
            verbose=True
        ),


        ModelCheckpoint(
            monitor='val_loss',
            dirpath='./checkpoints',
            filename='lstm-{epoch:02d}-{val_loss:.4f}',
            save_top_k=1,
            mode='min'
        ),

  
        LearningRateMonitor(logging_interval='epoch'),

        TQDMProgressBar(refresh_rate=10)
    ]



    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=True,
        precision=32,
    )



    trainer.fit(model, train_loader, val_loader)


    with open('scalers_pytorch.pkl', 'wb') as f:
        pickle.dump({'X_scaler': X_scaler, 'y_scaler': y_scaler}, f)
    print("已保存到 scalers_pytorch.pkl")


if __name__ == "__main__":
    main()
