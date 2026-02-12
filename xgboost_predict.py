"""
XGBoost 光伏发电功率预测模型
===========================
基于 solar_station_1.csv 数据集（50MW光伏电站），利用气象特征和时序特征，
使用 XGBoost 进行发电功率预测。

数据集: solar_station_1.csv
  - 70,176 行，15分钟间隔，2019-01-01 ~ 2020-12-31
  - 特征: tsi(总辐照度), dni(直接法向辐照度), ghi(全球水平辐照度),
          temp(气温), atm(气压), rh(相对湿度)
  - 目标变量: power (MW)
  - 额定容量: 50 MW

功能:
  1. 数据加载与预处理（处理-99异常值）
  2. 特征工程（气象特征 + 时序特征 + 滞后/滚动统计）
  3. XGBoost 模型训练（TimeSeriesSplit 交叉验证）
  4. 模型评估（MAE, RMSE, R², MAPE）
  5. 预测结果可视化
  6. 模型保存与加载
"""

import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

# ============================================================
# 路径配置
# ============================================================

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(DATA_DIR, "solar_station_1.csv")
MODEL_PATH = os.path.join(DATA_DIR, "xgboost_model.json")
RESULT_DIR = os.path.join(DATA_DIR, "results")

TARGET_COL = "power"
CAPACITY = 50  # 额定容量 50MW


# ============================================================
# 1. 数据加载与预处理
# ============================================================


def load_data(csv_path=CSV_PATH):
    """加载CSV数据并进行基础预处理"""
    df = pd.read_csv(csv_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # -99 是缺失值标记，替换为 NaN 后用前向填充
    for col in ["tsi", "dni", "ghi", "temp", "atm", "rh"]:
        df[col] = df[col].replace(-99, np.nan)
    df = df.ffill().bfill()

    # 辐照度不应为负数
    for col in ["tsi", "dni", "ghi"]:
        df[col] = df[col].clip(lower=0)

    # 发电功率不应为负数
    df[TARGET_COL] = df[TARGET_COL].clip(lower=0)

    print(f"数据加载完成: {len(df)} 行, 时间范围: {df['time'].min()} ~ {df['time'].max()}")
    print(f"列: {list(df.columns)}")
    return df


# ============================================================
# 2. 特征工程
# ============================================================


def add_time_features(df):
    """添加时间相关特征"""
    dt = df["time"]

    # 基础时间特征
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["minute"] = dt.dt.minute
    df["dayofweek"] = dt.dt.dayofweek
    df["dayofyear"] = dt.dt.dayofyear
    df["hour_min"] = dt.dt.hour + dt.dt.minute / 60.0
    df["date"] = dt.dt.date.astype(str)

    # 周期性编码（正弦/余弦变换，捕捉周期性规律）
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_min"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_min"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

    # 是否为周末
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    return df


def add_weather_features(df):
    """添加气象衍生特征"""
    # 晴空指数（总辐照度 / 全球水平辐照度）— 反映云层遮挡程度
    df["clearness_index"] = np.where(
        df["ghi"] > 10,
        df["tsi"] / df["ghi"],
        0.0,
    )

    # 散射辐射比例
    df["diffuse_ratio"] = np.where(
        df["tsi"] > 10,
        (df["tsi"] - df["dni"]) / df["tsi"],
        0.0,
    )

    # 辐照度变化率（当前与上一时刻的差值）
    df["irradiance_change"] = df["tsi"].diff()

    # 温度与辐照度的交互特征
    df["temp_irr_interaction"] = df["temp"] * df["tsi"] / 1000.0

    # 容量利用率（历史滞后的，避免数据泄露）
    df["capacity_ratio"] = df[TARGET_COL] / CAPACITY

    return df


def add_lag_features(df, target_col=TARGET_COL):
    """添加滞后特征和滚动统计特征"""
    # 15分钟间隔: 1步=15min, 4步=1h, 96步=1天
    lag_steps = [1, 2, 4, 8, 16, 96]  # 15min, 30min, 1h, 2h, 4h, 1day
    for lag in lag_steps:
        df[f"power_lag_{lag}"] = df[target_col].shift(lag)

    # 辐照度的滞后特征
    for lag in [1, 4]:
        df[f"irr_lag_{lag}"] = df["tsi"].shift(lag)

    # 发电功率的滚动统计
    for window in [4, 8, 24, 96]:  # 1h, 2h, 6h, 1day
        rolled = df[target_col].shift(1).rolling(window=window, min_periods=1)
        df[f"power_rmean_{window}"] = rolled.mean()
        df[f"power_rstd_{window}"] = rolled.std()
        df[f"power_rmax_{window}"] = rolled.max()

    # 辐照度的滚动统计
    for window in [4, 8]:  # 1h, 2h
        rolled = df["tsi"].shift(1).rolling(window=window, min_periods=1)
        df[f"irr_rmean_{window}"] = rolled.mean()
        df[f"irr_rstd_{window}"] = rolled.std()

    # 前一天同时刻的功率差值
    df["power_diff_1day"] = df[target_col].shift(1) - df[target_col].shift(97)

    return df


def add_daily_stats(df, target_col=TARGET_COL):
    """添加当天的累计统计特征"""
    df["daily_cumsum"] = df.groupby("date")[target_col].cumsum()
    df["daily_cumcount"] = df.groupby("date").cumcount() + 1
    df["daily_curmean"] = df["daily_cumsum"] / df["daily_cumcount"]
    return df


def build_features(df):
    """构建完整特征集"""
    df = add_time_features(df)
    df = add_weather_features(df)
    df = add_lag_features(df)
    df = add_daily_stats(df)

    # 删除 lag/diff 导致的 NaN 行
    df = df.dropna().reset_index(drop=True)
    print(f"特征工程完成: {len(df)} 行, {df.shape[1]} 列")
    return df


# ============================================================
# 3. 特征列表 & 数据集划分
# ============================================================

FEATURE_COLS = [
    # ---- 气象原始特征 ----
    "tsi",
    "dni",
    "ghi",
    "temp",
    "atm",
    "rh",
    # ---- 气象衍生特征 ----
    "clearness_index",
    "diffuse_ratio",
    "irradiance_change",
    "temp_irr_interaction",
    # ---- 时间特征 ----
    "month",
    "day",
    "hour",
    "minute",
    "dayofweek",
    "dayofyear",
    "hour_min",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    "is_weekend",
    # ---- 发电功率滞后 ----
    "power_lag_1",
    "power_lag_2",
    "power_lag_4",
    "power_lag_8",
    "power_lag_16",
    "power_lag_96",
    # ---- 辐照度滞后 ----
    "irr_lag_1",
    "irr_lag_4",
    # ---- 发电功率滚动统计 ----
    "power_rmean_4",
    "power_rstd_4",
    "power_rmax_4",
    "power_rmean_8",
    "power_rstd_8",
    "power_rmax_8",
    "power_rmean_24",
    "power_rstd_24",
    "power_rmax_24",
    "power_rmean_96",
    "power_rstd_96",
    "power_rmax_96",
    # ---- 辐照度滚动统计 ----
    "irr_rmean_4",
    "irr_rstd_4",
    "irr_rmean_8",
    "irr_rstd_8",
    # ---- 差分与日内统计 ----
    "power_diff_1day",
    "daily_curmean",
    "daily_cumcount",
]


def split_train_test(df, test_days=60):
    """按时间顺序划分训练集和测试集（最后N天作为测试集）"""
    dates = pd.to_datetime(df["date"])
    cutoff = dates.max() - pd.Timedelta(days=test_days)
    train = df[dates <= cutoff].copy()
    test = df[dates > cutoff].copy()
    print(f"训练集: {len(train)} 行 | 测试集: {len(test)} 行")
    print(f"训练期: {train['time'].min()} ~ {train['time'].max()}")
    print(f"测试期: {test['time'].min()} ~ {test['time'].max()}")
    return train, test


# ============================================================
# 4. 模型训练
# ============================================================


def train_model(train_df, feature_cols=FEATURE_COLS, target_col=TARGET_COL):
    """训练XGBoost模型，使用TimeSeriesSplit交叉验证选择最佳轮数"""
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.05,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
    }

    # TimeSeriesSplit 交叉验证确定最佳迭代轮数
    tscv = TimeSeriesSplit(n_splits=3)
    best_rounds = []

    print("\n--- 交叉验证 ---")
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train)):
        dtrain = xgb.DMatrix(X_train[tr_idx], label=y_train[tr_idx], feature_names=feature_cols)
        dval = xgb.DMatrix(X_train[va_idx], label=y_train[va_idx], feature_names=feature_cols)

        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        best_rounds.append(bst.best_iteration)
        print(f"  Fold {fold + 1}: best_iteration={bst.best_iteration}, val_rmse={bst.best_score:.4f}")

    optimal_rounds = int(np.mean(best_rounds))
    print(f"  平均最佳轮数: {optimal_rounds}")

    # 使用全部训练数据重新训练
    print("\n--- 使用全部训练数据训练最终模型 ---")
    dtrain_full = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    model = xgb.train(params, dtrain_full, num_boost_round=optimal_rounds)
    print(f"  训练完成, 共 {optimal_rounds} 轮")

    return model


# ============================================================
# 5. 模型评估
# ============================================================


def evaluate(model, test_df, feature_cols=FEATURE_COLS, target_col=TARGET_COL):
    """在测试集上评估模型"""
    X_test = test_df[feature_cols].values
    y_true = test_df[target_col].values

    dtest = xgb.DMatrix(X_test, feature_names=feature_cols)
    y_pred = model.predict(dtest)

    # 发电功率不应为负数，且不超过额定容量
    y_pred = np.clip(y_pred, 0, CAPACITY)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # MAPE（仅在实际值 > 0 时计算，避免除零）
    mask = y_true > 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = float("nan")

    print("\n======== 模型评估结果 ========")
    print(f"  MAE  : {mae:.4f} MW")
    print(f"  RMSE : {rmse:.4f} MW")
    print(f"  R²   : {r2:.4f}")
    print(f"  MAPE : {mape:.2f}%")
    print("==============================")

    return y_pred, {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


# ============================================================
# 6. 可视化
# ============================================================


def plot_results(test_df, y_pred, metrics, save_dir=RESULT_DIR):
    """绘制预测结果图表"""
    os.makedirs(save_dir, exist_ok=True)
    datetimes = pd.to_datetime(test_df["time"].values)
    y_true = test_df[TARGET_COL].values

    # --- 图1: 整体预测 vs 实际 ---
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(datetimes, y_true, label="Actual", alpha=0.7, linewidth=0.5)
    ax.plot(datetimes, y_pred, label="Predicted", alpha=0.7, linewidth=0.5)
    ax.set_title("Solar Power Generation: Actual vs Predicted (Test Set)")
    ax.set_xlabel("DateTime")
    ax.set_ylabel("Power (MW)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.tight_layout()
    path1 = os.path.join(save_dir, "prediction_overview.png")
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"  已保存: {path1}")

    # --- 图2: 逐日详细对比（每隔 10 天取一天，最多5天）---
    unique_dates = sorted(test_df["date"].unique())
    sample_dates = unique_dates[::10][:5]
    fig, axes = plt.subplots(len(sample_dates), 1, figsize=(14, 3.5 * len(sample_dates)), sharex=False)
    if len(sample_dates) == 1:
        axes = [axes]
    for ax, d in zip(axes, sample_dates):
        mask = test_df["date"].values == d
        ax.plot(datetimes[mask], y_true[mask], "o-", label="Actual", markersize=2)
        ax.plot(datetimes[mask], y_pred[mask], "s-", label="Predicted", markersize=2)
        ax.set_title(f"Date: {d}")
        ax.set_ylabel("Power (MW)")
        ax.legend(loc="upper right")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.tight_layout()
    path2 = os.path.join(save_dir, "prediction_daily_detail.png")
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"  已保存: {path2}")

    # --- 图3: 散点图 ---
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, s=1, alpha=0.3)
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1, label="y=x")
    ax.set_xlabel("Actual Power (MW)")
    ax.set_ylabel("Predicted Power (MW)")
    ax.set_title(f"Scatter Plot  (R²={metrics['R2']:.4f})")
    ax.legend()
    ax.set_aspect("equal")
    plt.tight_layout()
    path3 = os.path.join(save_dir, "prediction_scatter.png")
    fig.savefig(path3, dpi=150)
    plt.close(fig)
    print(f"  已保存: {path3}")

    return path1, path2, path3


def plot_feature_importance(model, save_dir=RESULT_DIR):
    """绘制特征重要性图"""
    os.makedirs(save_dir, exist_ok=True)
    importance = model.get_score(importance_type="gain")
    imp_df = pd.DataFrame(
        {"feature": list(importance.keys()), "importance": list(importance.values())}
    )
    imp_df = imp_df.sort_values("importance", ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(imp_df["feature"], imp_df["importance"])
    ax.set_title("Top 20 Feature Importance (Gain)")
    ax.set_xlabel("Gain")
    plt.tight_layout()
    path = os.path.join(save_dir, "feature_importance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  已保存: {path}")
    return path


# ============================================================
# 7. 模型保存与加载
# ============================================================


def save_model(model, path=MODEL_PATH):
    """保存XGBoost模型"""
    model.save_model(path)
    print(f"模型已保存: {path}")


def load_model(path=MODEL_PATH):
    """加载XGBoost模型"""
    model = xgb.Booster()
    model.load_model(path)
    print(f"模型已加载: {path}")
    return model


# ============================================================
# 8. 主流程
# ============================================================


def main():
    print("=" * 60)
    print("  XGBoost 光伏发电功率预测")
    print("  数据集: solar_station_1.csv (50MW)")
    print("=" * 60)

    # 1. 加载数据
    df = load_data()

    # 2. 特征工程
    df = build_features(df)

    # 3. 划分数据集（最后60天为测试集，约2个月）
    train_df, test_df = split_train_test(df, test_days=60)

    # 4. 训练模型
    model = train_model(train_df)

    # 5. 评估
    y_pred, metrics = evaluate(model, test_df)

    # 6. 可视化
    print("\n--- 生成可视化图表 ---")
    plot_results(test_df, y_pred, metrics)
    plot_feature_importance(model)

    # 7. 保存模型
    save_model(model)

    # 8. 保存预测结果
    os.makedirs(RESULT_DIR, exist_ok=True)
    result_df = test_df[["time", TARGET_COL]].copy()
    result_df["predicted"] = y_pred
    result_path = os.path.join(RESULT_DIR, "predictions.csv")
    result_df.to_csv(result_path, index=False)
    print(f"预测结果已保存: {result_path}")

    print("\n完成！")
    return model, metrics


if __name__ == "__main__":
    main()
