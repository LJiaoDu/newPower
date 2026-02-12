#!/usr/bin/env python3
"""
太阳能电站数据分析 - 分布图像生成脚本
生成多张分析图表，保存到 plots/ 目录
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无头模式
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置中文字体（回退到 DejaVu Sans 如果没有中文字体）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

# 输出目录
PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

CSV_PATH = "/media/zlg/Data1/Longjiao/TF208/solar_station_1.csv"
# 如果本地有副本则用本地的
LOCAL_CSV = os.path.join(os.path.dirname(__file__), "solar_station_1.csv")
if os.path.exists(LOCAL_CSV):
    CSV_PATH = LOCAL_CSV


def load_data():
    df = pd.read_csv(CSV_PATH)
    df["time"] = pd.to_datetime(df["time"])
    # 处理 -99 缺失
    weather_cols = ["tsi", "dni", "ghi", "temp", "atm", "rh"]
    for col in weather_cols:
        df.loc[df[col] == -99, col] = np.nan
    # 处理 rh 异常值
    df.loc[df["rh"] > 100, "rh"] = np.nan
    df[weather_cols] = df[weather_cols].interpolate(method="linear")
    df[weather_cols] = df[weather_cols].bfill().ffill()
    df["hour"] = df["time"].dt.hour + df["time"].dt.minute / 60
    df["month"] = df["time"].dt.month
    df["date"] = df["time"].dt.date
    return df


def plot_power_distribution(df):
    """Fig 1: Power distribution histogram"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Full distribution
    axes[0].hist(df["power"], bins=100, color="#2196F3", edgecolor="white", alpha=0.8)
    axes[0].set_xlabel("Power (MW)", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title("Power Distribution (All Data)", fontsize=13)
    axes[0].axvline(x=df["power"].mean(), color="red", linestyle="--", label=f'Mean={df["power"].mean():.1f} MW')
    axes[0].legend()

    # Daytime only
    day = df[df["power"] > 0]["power"]
    axes[1].hist(day, bins=80, color="#FF9800", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("Power (MW)", fontsize=12)
    axes[1].set_ylabel("Count", fontsize=12)
    axes[1].set_title("Power Distribution (Daytime, power>0)", fontsize=13)
    axes[1].axvline(x=day.mean(), color="red", linestyle="--", label=f"Mean={day.mean():.1f} MW")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "01_power_distribution.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_weather_distributions(df):
    """Fig 2: Weather feature distributions (6 subplots)"""
    weather_cols = ["tsi", "dni", "ghi", "temp", "atm", "rh"]
    titles = [
        "TSI - Total Solar Irradiance (W/m²)",
        "DNI - Direct Normal Irradiance (W/m²)",
        "GHI - Global Horizontal Irradiance (W/m²)",
        "Temperature (°C)",
        "Atmospheric Pressure (hPa)",
        "Relative Humidity (%)",
    ]
    colors = ["#E91E63", "#9C27B0", "#3F51B5", "#009688", "#795548", "#607D8B"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, (col, title, color) in enumerate(zip(weather_cols, titles, colors)):
        ax = axes[i]
        data = df[col].dropna()
        ax.hist(data, bins=80, color=color, edgecolor="white", alpha=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Count", fontsize=10)
        # 统计
        ax.axvline(x=data.mean(), color="black", linestyle="--", linewidth=1,
                    label=f"Mean={data.mean():.1f}")
        ax.legend(fontsize=9)

    plt.suptitle("Weather Feature Distributions", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "02_weather_distributions.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_hourly_pattern(df):
    """Fig 3: Hourly average power pattern"""
    hourly = df.groupby(df["time"].dt.hour)["power"].agg(["mean", "std", "median"])

    fig, ax = plt.subplots(figsize=(12, 5))
    hours = hourly.index
    ax.fill_between(hours, hourly["mean"] - hourly["std"], hourly["mean"] + hourly["std"],
                     alpha=0.2, color="#2196F3", label="Mean ± Std")
    ax.plot(hours, hourly["mean"], "o-", color="#2196F3", linewidth=2, markersize=6, label="Mean")
    ax.plot(hours, hourly["median"], "s--", color="#FF5722", linewidth=1.5, markersize=5, label="Median")
    ax.set_xlabel("Hour of Day", fontsize=12)
    ax.set_ylabel("Power (MW)", fontsize=12)
    ax.set_title("Hourly Average Power Pattern", fontsize=13)
    ax.set_xticks(range(0, 24))
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "03_hourly_pattern.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_monthly_pattern(df):
    """Fig 4: Monthly power boxplot"""
    fig, ax = plt.subplots(figsize=(12, 5))

    # 只用白天数据
    day_df = df[df["power"] > 0].copy()
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    bp = ax.boxplot([day_df[day_df["month"] == m]["power"].values for m in range(1, 13)],
                     labels=month_names, patch_artist=True, showfliers=False)

    colors_month = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, 12))
    for patch, color in zip(bp["boxes"], colors_month):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Power (MW)", fontsize=12)
    ax.set_title("Monthly Power Distribution (Daytime Only)", fontsize=13)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "04_monthly_boxplot.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_correlation_heatmap(df):
    """Fig 5: Feature correlation heatmap"""
    cols = ["tsi", "dni", "ghi", "temp", "atm", "rh", "power"]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                mask=mask, square=True, linewidths=0.5, ax=ax,
                vmin=-1, vmax=1, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix", fontsize=13, pad=10)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "05_correlation_heatmap.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_power_vs_tsi(df):
    """Fig 6: Power vs TSI scatter plot"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # 随机采样避免点太多
    sample = df[df["power"] > 0].sample(min(5000, len(df[df["power"] > 0])), random_state=42)
    scatter = ax.scatter(sample["tsi"], sample["power"], c=sample["hour"],
                          cmap="viridis", alpha=0.5, s=10, edgecolors="none")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Hour of Day", fontsize=11)
    ax.set_xlabel("TSI - Total Solar Irradiance (W/m²)", fontsize=12)
    ax.set_ylabel("Power (MW)", fontsize=12)
    ax.set_title(f"Power vs TSI (r={df['tsi'].corr(df['power']):.4f})", fontsize=13)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "06_power_vs_tsi.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_autocorrelation(df):
    """Fig 7: Power autocorrelation"""
    power = df["power"].values
    lags = list(range(1, 97 * 3 + 1))  # up to 3 days
    acf = []
    mean_p = power.mean()
    var_p = np.sum((power - mean_p) ** 2)

    for lag in lags:
        cov = np.sum((power[lag:] - mean_p) * (power[:-lag] - mean_p))
        acf.append(cov / var_p)

    fig, ax = plt.subplots(figsize=(14, 5))
    hours = [l * 15 / 60 for l in lags]
    ax.plot(hours, acf, color="#2196F3", linewidth=1)
    ax.fill_between(hours, acf, alpha=0.15, color="#2196F3")

    # 标注关键点
    key_lags = {1: "15min", 4: "1h", 16: "4h", 96: "24h", 192: "48h"}
    for lag, label in key_lags.items():
        if lag <= len(acf):
            ax.annotate(f"{label}\nr={acf[lag-1]:.3f}",
                        xy=(lag * 15 / 60, acf[lag - 1]),
                        fontsize=9, ha="center",
                        xytext=(0, 15), textcoords="offset points",
                        arrowprops=dict(arrowstyle="->", color="red"),
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    ax.set_xlabel("Lag (hours)", fontsize=12)
    ax.set_ylabel("Autocorrelation", fontsize=12)
    ax.set_title("Power Autocorrelation Function", fontsize=13)
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color="gray", linewidth=0.5)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "07_autocorrelation.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_daily_generation(df):
    """Fig 8: Daily total generation time series"""
    daily = df.groupby("date")["power"].sum() * 0.25  # MWh (15min intervals)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Time series
    dates = pd.to_datetime(list(daily.index))
    axes[0].fill_between(dates, daily.values, alpha=0.3, color="#4CAF50")
    axes[0].plot(dates, daily.values, color="#4CAF50", linewidth=0.8)
    axes[0].set_ylabel("Daily Generation (MWh)", fontsize=11)
    axes[0].set_title("Daily Generation Time Series", fontsize=13)
    axes[0].axhline(y=daily.mean(), color="red", linestyle="--", alpha=0.7,
                     label=f"Mean={daily.mean():.1f} MWh")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Distribution
    axes[1].hist(daily.values, bins=50, color="#4CAF50", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("Daily Generation (MWh)", fontsize=11)
    axes[1].set_ylabel("Count (days)", fontsize=11)
    axes[1].set_title("Daily Generation Distribution", fontsize=13)
    axes[1].axvline(x=daily.mean(), color="red", linestyle="--",
                     label=f"Mean={daily.mean():.1f} MWh")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "08_daily_generation.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_rh_anomaly(df_raw):
    """Fig 9: rh anomaly visualization (before/after fix)"""
    df_raw2 = pd.read_csv(CSV_PATH)
    rh_raw = df_raw2["rh"].copy()
    rh_raw[rh_raw == -99] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Before fix
    rh_vals = rh_raw.dropna()
    axes[0].hist(rh_vals, bins=100, color="#F44336", edgecolor="white", alpha=0.8)
    axes[0].set_xlabel("Relative Humidity (%)", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_title(f"RH Distribution (Raw, max={rh_vals.max():.0f}%)", fontsize=12)
    axes[0].axvline(x=100, color="black", linestyle="--", linewidth=2, label="100% threshold")
    n_bad = (rh_vals > 100).sum()
    axes[0].annotate(f"{n_bad} anomaly points\n(sensor fault >100%)",
                      xy=(0.6, 0.85), xycoords="axes fraction",
                      fontsize=10, color="red",
                      bbox=dict(boxstyle="round", facecolor="lightyellow"))
    axes[0].legend()

    # After fix
    rh_fixed = df_raw["rh"]
    axes[1].hist(rh_fixed, bins=80, color="#4CAF50", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("Relative Humidity (%)", fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)
    axes[1].set_title(f"RH Distribution (Fixed, max={rh_fixed.max():.1f}%)", fontsize=12)
    axes[1].annotate("Anomalies removed\n& interpolated",
                      xy=(0.6, 0.85), xycoords="axes fraction",
                      fontsize=10, color="green",
                      bbox=dict(boxstyle="round", facecolor="lightyellow"))

    plt.suptitle("Humidity (rh) Sensor Fault Fix", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "09_rh_anomaly_fix.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_power_heatmap(df):
    """Fig 10: Power heatmap (hour vs day-of-year)"""
    df2 = df.copy()
    df2["dayofyear"] = df2["time"].dt.dayofyear
    df2["hour_int"] = df2["time"].dt.hour

    pivot = df2.pivot_table(values="power", index="hour_int", columns="dayofyear", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", origin="lower",
                    extent=[1, 366, 0, 24])
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Power (MW)", fontsize=11)
    ax.set_xlabel("Day of Year", fontsize=12)
    ax.set_ylabel("Hour of Day", fontsize=12)
    ax.set_title("Power Heatmap (Hour x Day of Year)", fontsize=13)
    ax.set_yticks(range(0, 25, 3))

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "10_power_heatmap.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_step_difficulty(df):
    """Fig 11: Prediction difficulty by step (how much power changes in 1-16 steps ahead)"""
    power = df["power"].values
    steps = range(1, 17)
    mae_list = []
    for s in steps:
        diff = np.abs(power[s:] - power[:-s])
        mae_list.append(diff.mean())

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(steps, mae_list, color=plt.cm.Reds(np.linspace(0.3, 0.9, 16)),
                   edgecolor="white")
    ax.set_xlabel("Prediction Step (x15min ahead)", fontsize=12)
    ax.set_ylabel("Mean Absolute Deviation (MW)", fontsize=12)
    ax.set_title("Naive Prediction Difficulty by Step", fontsize=13)
    ax.set_xticks(steps)
    ax.set_xticklabels([f"{s}\n({s*15}min)" for s in steps], fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # 标数值
    for bar, val in zip(bars, mae_list):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", fontsize=8)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "11_step_difficulty.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("=" * 60)
    print("Generating Distribution Plots")
    print("=" * 60)

    print("\nLoading data...")
    df = load_data()
    print(f"  Loaded {len(df)} rows")

    print("\nGenerating plots:")
    plot_power_distribution(df)
    plot_weather_distributions(df)
    plot_hourly_pattern(df)
    plot_monthly_pattern(df)
    plot_correlation_heatmap(df)
    plot_power_vs_tsi(df)
    plot_autocorrelation(df)
    plot_daily_generation(df)
    plot_rh_anomaly(df)
    plot_power_heatmap(df)
    plot_step_difficulty(df)

    print(f"\nAll plots saved to: {PLOT_DIR}/")
    print(f"Total: {len(os.listdir(PLOT_DIR))} images")


if __name__ == "__main__":
    main()
