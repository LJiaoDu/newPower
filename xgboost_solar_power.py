"""
XGBoost Solar Power Prediction
Dataset: solar_station_1.csv
Target: power (MW), Capacity: 50MW
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. Load & Clean Data
# ─────────────────────────────────────────────
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['time'])
    df = df.sort_values('time').reset_index(drop=True)

    # Replace sentinel -99 with NaN
    sentinel_cols = ['tsi', 'dni', 'ghi', 'temp', 'atm', 'rh']
    for col in sentinel_cols:
        df[col] = df[col].replace(-99, np.nan)

    # Forward-fill short gaps (≤4 steps = 1 hour), then back-fill remaining
    df[sentinel_cols] = df[sentinel_cols].ffill(limit=4).bfill(limit=4)

    # Drop rows still missing after fill (very rare)
    before = len(df)
    df = df.dropna(subset=sentinel_cols + ['power'])
    print(f"Dropped {before - len(df)} rows with unresolvable NaN")

    # Clip obviously erroneous irradiance negatives to 0
    for col in ['tsi', 'dni', 'ghi']:
        df[col] = df[col].clip(lower=0)

    return df


# ─────────────────────────────────────────────
# 2. Feature Engineering
# ─────────────────────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    t = df['time']

    # Temporal features
    df['hour']       = t.dt.hour + t.dt.minute / 60        # fractional hour
    df['hour_sin']   = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']   = np.cos(2 * np.pi * df['hour'] / 24)
    df['dayofyear']  = t.dt.dayofyear
    df['doy_sin']    = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['doy_cos']    = np.cos(2 * np.pi * df['dayofyear'] / 365)
    df['month']      = t.dt.month
    df['dayofweek']  = t.dt.dayofweek

    # Solar capacity factor (power / installed capacity)
    cap = df['cap'].iloc[0]
    df['ghi_norm']   = df['ghi'] / 1000.0   # normalised irradiance [0–1 ish]
    df['dni_norm']   = df['dni'] / 1000.0
    df['tsi_norm']   = df['tsi'] / 1361.0   # TSI ≈ solar constant

    # Interaction: irradiance × temperature
    df['ghi_x_temp'] = df['ghi_norm'] * df['temp']

    # Lag features (previous 15-min & 1-hour readings)
    for col in ['power', 'ghi', 'dni']:
        df[f'{col}_lag1']  = df[col].shift(1)   # t-15min
        df[f'{col}_lag4']  = df[col].shift(4)   # t-1h
        df[f'{col}_lag96'] = df[col].shift(96)  # t-24h (same time yesterday)

    # Rolling statistics (1-hour window)
    df['power_roll4_mean'] = df['power'].shift(1).rolling(4).mean()
    df['ghi_roll4_mean']   = df['ghi'].shift(1).rolling(4).mean()

    df = df.dropna()  # remove rows invalidated by lag creation
    return df


# ─────────────────────────────────────────────
# 3. Train / Validation / Test split (time-based)
# ─────────────────────────────────────────────
FEATURE_COLS = [
    'tsi', 'dni', 'ghi', 'temp', 'atm', 'rh',
    'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos',
    'month', 'dayofweek',
    'ghi_norm', 'dni_norm', 'tsi_norm',
    'ghi_x_temp',
    'power_lag1',  'power_lag4',  'power_lag96',
    'ghi_lag1',    'ghi_lag4',    'ghi_lag96',
    'dni_lag1',    'dni_lag4',    'dni_lag96',
    'power_roll4_mean', 'ghi_roll4_mean',
]

TARGET_COL = 'power'


def split_data(df: pd.DataFrame):
    # 2019: train+val  |  2020: test
    train_val = df[df['time'].dt.year == 2019]
    test      = df[df['time'].dt.year == 2020]

    # Within 2019: last 2 months = validation
    val_start = '2019-11-01'
    train = train_val[train_val['time'] < val_start]
    val   = train_val[train_val['time'] >= val_start]

    print(f"Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

    X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
    X_val,   y_val   = val[FEATURE_COLS],   val[TARGET_COL]
    X_test,  y_test  = test[FEATURE_COLS],  test[TARGET_COL]
    t_test           = test['time']
    return X_train, y_train, X_val, y_val, X_test, y_test, t_test


# ─────────────────────────────────────────────
# 4. Model Training
# ─────────────────────────────────────────────
def train_model(X_train, y_train, X_val, y_val):
    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective='reg:squarederror',
        tree_method='hist',
        early_stopping_rounds=50,
        eval_metric='rmse',
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )
    print(f"\nBest iteration: {model.best_iteration}")
    return model


# ─────────────────────────────────────────────
# 5. Evaluation
# ─────────────────────────────────────────────
def evaluate(model, X, y, label=''):
    pred = model.predict(X)
    pred = np.clip(pred, 0, None)   # power ≥ 0

    rmse = np.sqrt(mean_squared_error(y, pred))
    mae  = mean_absolute_error(y, pred)
    r2   = r2_score(y, pred)
    cap  = 50.0
    nrmse = rmse / cap * 100        # normalised RMSE (% of capacity)
    nmae  = mae  / cap * 100

    print(f"\n{'─'*40}")
    print(f"  {label}")
    print(f"  RMSE : {rmse:.4f} MW  ({nrmse:.2f}% of cap)")
    print(f"  MAE  : {mae:.4f} MW  ({nmae:.2f}% of cap)")
    print(f"  R²   : {r2:.6f}")
    print(f"{'─'*40}")
    return pred, rmse, mae, r2


# ─────────────────────────────────────────────
# 6. Plots
# ─────────────────────────────────────────────
def plot_results(t_test, y_test, y_pred, model, X_test, output_dir='.'):
    fig, axes = plt.subplots(3, 1, figsize=(15, 13))
    fig.suptitle('XGBoost Solar Power Prediction – Test Set (2020)', fontsize=14)

    # ── (a) One-week time-series comparison ──
    ax = axes[0]
    week_mask = (t_test >= '2020-07-01') & (t_test < '2020-07-08')
    ax.plot(t_test[week_mask].values, y_test[week_mask].values,
            label='Actual', color='steelblue', linewidth=1.5)
    ax.plot(t_test[week_mask].values, y_pred[week_mask],
            label='Predicted', color='tomato', linewidth=1.5, linestyle='--')
    ax.set_title('(a) Sample week: 2020-07-01 ~ 07-07')
    ax.set_ylabel('Power (MW)')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.grid(True, alpha=0.3)

    # ── (b) Scatter: predicted vs actual ──
    ax = axes[1]
    ax.scatter(y_test.values, y_pred, alpha=0.15, s=4, color='darkorange')
    lim = max(y_test.max(), y_pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', linewidth=1, label='Ideal')
    r2 = r2_score(y_test, y_pred)
    ax.set_title(f'(b) Predicted vs Actual  (R² = {r2:.4f})')
    ax.set_xlabel('Actual Power (MW)')
    ax.set_ylabel('Predicted Power (MW)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── (c) Feature importance (top-20) ──
    ax = axes[2]
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    importance = importance.nlargest(20).sort_values()
    importance.plot(kind='barh', ax=ax, color='teal')
    ax.set_title('(c) Feature Importance (top 20, gain)')
    ax.set_xlabel('Importance score')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    save_path = f'{output_dir}/xgboost_solar_results.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved → {save_path}")
    plt.close()

    # Monthly RMSE breakdown
    monthly = pd.DataFrame({'time': t_test.values, 'actual': y_test.values, 'pred': y_pred})
    monthly['month'] = pd.to_datetime(monthly['time']).dt.month
    monthly['se'] = (monthly['actual'] - monthly['pred']) ** 2
    monthly_rmse = monthly.groupby('month')['se'].mean().apply(np.sqrt)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    monthly_rmse.plot(kind='bar', ax=ax2, color='steelblue', edgecolor='white')
    ax2.set_title('Monthly RMSE – Test Set 2020')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('RMSE (MW)')
    ax2.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                          'Jul','Aug','Sep','Oct','Nov','Dec'], rotation=0)
    ax2.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_path2 = f'{output_dir}/xgboost_monthly_rmse.png'
    plt.savefig(save_path2, dpi=150, bbox_inches='tight')
    print(f"Monthly RMSE plot saved → {save_path2}")
    plt.close()


# ─────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  XGBoost Solar Power Prediction")
    print("=" * 50)

    # Load & preprocess
    df = load_and_clean('/home/user/newPower/solar_station_1.csv')
    df = add_features(df)
    print(f"\nFinal dataset shape: {df.shape}")

    # Split
    X_train, y_train, X_val, y_val, X_test, y_test, t_test = split_data(df)

    # Train
    print("\nTraining XGBoost model …")
    model = train_model(X_train, y_train, X_val, y_val)

    # Evaluate on all splits
    evaluate(model, X_train, y_train, 'Train Set')
    evaluate(model, X_val,   y_val,   'Validation Set')
    y_pred, rmse, mae, r2 = evaluate(model, X_test, y_test, 'Test Set (2020)')

    # Reset test index for plotting
    t_test = t_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Plots
    plot_results(t_test, y_test, y_pred, model, X_test,
                 output_dir='/home/user/newPower')

    # Save predictions
    results = pd.DataFrame({
        'time':    t_test,
        'actual':  y_test.values,
        'predicted': y_pred,
    })
    results.to_csv('/home/user/newPower/xgboost_predictions.csv', index=False)
    print("\nPredictions saved → xgboost_predictions.csv")

    print("\nDone.")


if __name__ == '__main__':
    main()
