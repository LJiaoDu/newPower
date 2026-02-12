"""
XGBoost 光伏发电功率预测模型
===========================
基于772光伏电站的历史发电数据，使用XGBoost进行发电功率预测。

功能:
1. 数据加载与预处理
2. 时序特征工程（滞后特征、滚动统计、周期编码等）
3. XGBoost模型训练与超参数调优
4. 模型评估（MAE, RMSE, R², MAPE）
5. 预测结果可视化
6. 模型保存与加载
"""

import os
import json
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
import joblib

warnings.filterwarnings("ignore")

# ============================================================
# 1. 数据加载与预处理
# ============================================================

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(DATA_DIR, "training_data.csv")
MODEL_PATH = os.path.join(DATA_DIR, "xgboost_model.json")
RESULT_DIR = os.path.join(DATA_DIR, "results")


def load_data(csv_path=CSV_PATH):
    """加载CSV训练数据"""
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"数据加载完成: {len(df)} 行, 时间范围: {df['datetime'].min()} ~ {df['datetime'].max()}")
    return df


# ============================================================
# 2. 特征工程
# ============================================================


def add_time_features(df):
    """添加时间相关特征"""
    dt = df["datetime"]

    # 基础时间特征
    df["hour_min"] = dt.dt.hour + dt.dt.minute / 60.0

    # 周期性编码（正弦/余弦变换，捕捉周期性规律）
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_min"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_min"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

    # 是否为工作日
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    return df


def add_lag_features(df, target_col="generationPower"):
    """添加滞后特征和滚动统计特征"""
    # 滞后特征：前几个时间步的发电功率
    lag_steps = [1, 2, 3, 6, 12, 24, 288]  # 5min, 10min, 15min, 30min, 1h, 2h, 1day
    for lag in lag_steps:
        df[f"lag_{lag}"] = df[target_col].shift(lag)

    # 滚动统计特征
    for window in [6, 12, 36, 288]:  # 30min, 1h, 3h, 1day
        rolled = df[target_col].shift(1).rolling(window=window, min_periods=1)
        df[f"rolling_mean_{window}"] = rolled.mean()
        df[f"rolling_std_{window}"] = rolled.std()
        df[f"rolling_max_{window}"] = rolled.max()
        df[f"rolling_min_{window}"] = rolled.min()

    # 同一时刻前一天的发电量差值
    df["diff_1day"] = df[target_col].shift(1) - df[target_col].shift(289)

    return df


def add_daily_stats(df, target_col="generationPower"):
    """添加当天的历史统计特征（使用截至当前时刻的数据）"""
    df["daily_cumsum"] = df.groupby("date")[target_col].cumsum()
    df["daily_cumcount"] = df.groupby("date").cumcount() + 1
    df["daily_curmean"] = df["daily_cumsum"] / df["daily_cumcount"]
    return df


def build_features(df):
    """构建完整特征集"""
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_daily_stats(df)

    # 删除无法计算特征的行（lag导致的NaN）
    df = df.dropna().reset_index(drop=True)
    print(f"特征工程完成: {len(df)} 行, {df.shape[1]} 列")
    return df


# ============================================================
# 3. 数据集划分
# ============================================================

FEATURE_COLS = [
    # 时间特征
    "year",
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
    # 滞后特征
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "lag_24",
    "lag_288",
    # 滚动统计
    "rolling_mean_6",
    "rolling_std_6",
    "rolling_max_6",
    "rolling_min_6",
    "rolling_mean_12",
    "rolling_std_12",
    "rolling_max_12",
    "rolling_min_12",
    "rolling_mean_36",
    "rolling_std_36",
    "rolling_max_36",
    "rolling_min_36",
    "rolling_mean_288",
    "rolling_std_288",
    "rolling_max_288",
    "rolling_min_288",
    # 差分与日内统计
    "diff_1day",
    "daily_curmean",
    "daily_cumcount",
]

TARGET_COL = "generationPower"


def split_train_test(df, test_days=30):
    """按时间顺序划分训练集和测试集（最后N天作为测试集）"""
    dates = pd.to_datetime(df["date"])
    cutoff = dates.max() - pd.Timedelta(days=test_days)
    train = df[dates <= cutoff].copy()
    test = df[dates > cutoff].copy()
    print(f"训练集: {len(train)} 行 | 测试集: {len(test)} 行")
    print(f"训练期: {train['datetime'].min()} ~ {train['datetime'].max()}")
    print(f"测试期: {test['datetime'].min()} ~ {test['datetime'].max()}")
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
        print(f"  Fold {fold + 1}: best_iteration={bst.best_iteration}, val_rmse={bst.best_score:.2f}")

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

    # 发电功率不应为负数
    y_pred = np.clip(y_pred, 0, None)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # MAPE（仅在实际值>0时计算，避免除零）
    mask = y_true > 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = float("nan")

    print("\n======== 模型评估结果 ========")
    print(f"  MAE  : {mae:.2f}")
    print(f"  RMSE : {rmse:.2f}")
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
    datetimes = pd.to_datetime(test_df["datetime"].values)
    y_true = test_df[TARGET_COL].values

    # --- 图1: 整体预测 vs 实际 ---
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(datetimes, y_true, label="Actual", alpha=0.7, linewidth=0.5)
    ax.plot(datetimes, y_pred, label="Predicted", alpha=0.7, linewidth=0.5)
    ax.set_title("Solar Power Generation: Actual vs Predicted (Test Set)")
    ax.set_xlabel("DateTime")
    ax.set_ylabel("Generation Power (kW)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.tight_layout()
    path1 = os.path.join(save_dir, "prediction_overview.png")
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"  已保存: {path1}")

    # --- 图2: 取几天做详细对比 ---
    unique_dates = sorted(test_df["date"].unique())
    sample_dates = unique_dates[::7][:5]  # 每隔7天取一天，最多5天
    fig, axes = plt.subplots(len(sample_dates), 1, figsize=(14, 3.5 * len(sample_dates)), sharex=False)
    if len(sample_dates) == 1:
        axes = [axes]
    for ax, d in zip(axes, sample_dates):
        mask = test_df["date"].values == d
        ax.plot(datetimes[mask], y_true[mask], "o-", label="Actual", markersize=2)
        ax.plot(datetimes[mask], y_pred[mask], "s-", label="Predicted", markersize=2)
        ax.set_title(f"Date: {d}")
        ax.set_ylabel("Power (kW)")
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
    ax.set_xlabel("Actual Power (kW)")
    ax.set_ylabel("Predicted Power (kW)")
    ax.set_title(f"Scatter Plot  (R²={metrics['R2']:.4f})")
    ax.legend()
    ax.set_aspect("equal")
    plt.tight_layout()
    path3 = os.path.join(save_dir, "prediction_scatter.png")
    fig.savefig(path3, dpi=150)
    plt.close(fig)
    print(f"  已保存: {path3}")

    # --- 图4: 特征重要性 ---
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
    print("=" * 60)

    # 1. 加载数据
    df = load_data()

    # 2. 特征工程
    df = build_features(df)

    # 3. 划分数据集（最后30天为测试集）
    train_df, test_df = split_train_test(df, test_days=30)

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
    result_df = test_df[["datetime", TARGET_COL]].copy()
    result_df["predicted"] = y_pred
    result_path = os.path.join(RESULT_DIR, "predictions.csv")
    result_df.to_csv(result_path, index=False)
    print(f"预测结果已保存: {result_path}")

    print("\n完成！")
    return model, metrics


if __name__ == "__main__":
    main()
