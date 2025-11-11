import json
import pandas as pd
import numpy as np
from datetime import datetime
import os
from sklearn.preprocessing import StandardScaler
import pickle

class PowerDataProcessor:
    """功率数据预处理类"""

    def __init__(self, data_dir='.'):
        self.data_dir = data_dir
        self.scaler = StandardScaler()

    def load_csv_file(self, csv_file='training_data.csv'):
        """加载CSV文件"""
        file_path = os.path.join(self.data_dir, csv_file)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"未找到CSV文件: {file_path}")

        print(f"从CSV文件加载数据: {file_path}")
        df = pd.read_csv(file_path)

        # 统一列名（支持datetime或dateTime）
        if 'datetime' in df.columns and 'dateTime' not in df.columns:
            df.rename(columns={'datetime': 'dateTime'}, inplace=True)

        # 转换时间列
        if 'dateTime' in df.columns:
            # 如果是时间戳，转换为datetime
            if df['dateTime'].dtype in ['int64', 'float64']:
                df['dateTime'] = pd.to_datetime(df['dateTime'], unit='ms')
            else:
                df['dateTime'] = pd.to_datetime(df['dateTime'])

        df = df.sort_values('dateTime').reset_index(drop=True)
        print(f"总共加载 {len(df)} 条数据记录")
        print(f"数据时间范围: {df['dateTime'].min()} 至 {df['dateTime'].max()}")
        print(f"数据列: {list(df.columns)}")
        return df

    def load_json_files(self):
        """加载所有JSON文件并转换为DataFrame"""
        all_data = []

        json_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.json')])
        print(f"找到 {len(json_files)} 个JSON文件")

        for json_file in json_files:
            file_path = os.path.join(self.data_dir, json_file)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                power_list = data.get('stationStatisticPowerList', [])

                for entry in power_list:
                    all_data.append({
                        'dateTime': entry['dateTime'],
                        'generationPower': entry['generationPower'],
                        'seq': entry['seq'],
                        'year': entry['year'],
                        'month': entry['month'],
                        'day': entry['day']
                    })
            except Exception as e:
                print(f"读取文件 {json_file} 时出错: {e}")
                continue

        df = pd.DataFrame(all_data)
        df['dateTime'] = pd.to_datetime(df['dateTime'], unit='ms')
        df = df.sort_values('dateTime').reset_index(drop=True)

        print(f"总共加载 {len(df)} 条数据记录")
        return df

    def extract_time_features(self, df):
        """提取时间特征"""
        df = df.copy()

        # 基础时间特征
        df['hour'] = df['dateTime'].dt.hour
        df['minute'] = df['dateTime'].dt.minute
        df['dayofweek'] = df['dateTime'].dt.dayofweek  # 0=周一, 6=周日
        df['dayofyear'] = df['dateTime'].dt.dayofyear
        df['weekofyear'] = df['dateTime'].dt.isocalendar().week

        # 周期性编码（正弦-余弦变换）
        # 小时的周期性 (24小时)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # 一天中分钟的周期性 (1440分钟)
        df['minute_of_day'] = df['hour'] * 60 + df['minute']
        df['minute_sin'] = np.sin(2 * np.pi * df['minute_of_day'] / 1440)
        df['minute_cos'] = np.cos(2 * np.pi * df['minute_of_day'] / 1440)

        # 星期的周期性 (7天)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

        # 月份的周期性 (12个月)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # 年度的周期性 (365天)
        df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
        df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)

        # 是否为周末
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

        return df

    def extract_power_features(self, df, window_sizes=[12, 24, 48]):
        """提取功率统计特征

        window_sizes: 窗口大小列表，默认[12, 24, 48]表示1小时、2小时、4小时
                     (假设每5分钟一个数据点)
        """
        df = df.copy()

        for window in window_sizes:
            # 滚动平均
            df[f'power_ma_{window}'] = df['generationPower'].rolling(
                window=window, min_periods=1).mean()

            # 滚动标准差
            df[f'power_std_{window}'] = df['generationPower'].rolling(
                window=window, min_periods=1).std().fillna(0)

            # 滚动最大值
            df[f'power_max_{window}'] = df['generationPower'].rolling(
                window=window, min_periods=1).max()

            # 滚动最小值
            df[f'power_min_{window}'] = df['generationPower'].rolling(
                window=window, min_periods=1).min()

        # 功率变化率
        df['power_diff_1'] = df['generationPower'].diff(1).fillna(0)
        df['power_diff_2'] = df['generationPower'].diff(2).fillna(0)

        # 功率变化率的百分比
        df['power_pct_change'] = df['generationPower'].pct_change().fillna(0)

        return df

    def create_sequences(self, df, input_hours=20, output_hours=4, step_minutes=5):
        """创建序列数据用于训练

        Args:
            df: 数据框
            input_hours: 输入的历史小时数 (20小时)
            output_hours: 预测的未来小时数 (4小时)
            step_minutes: 采样间隔(分钟)

        Returns:
            X: 输入序列特征
            y: 目标序列(功率值)
        """
        # 计算序列长度
        input_len = int(input_hours * 60 / step_minutes)  # 20小时 = 240个点
        output_len = int(output_hours * 60 / step_minutes)  # 4小时 = 48个点

        # 选择特征列
        feature_cols = [col for col in df.columns if col not in
                       ['dateTime', 'year', 'month', 'day', 'seq',
                        'hour', 'minute', 'dayofweek', 'dayofyear',
                        'weekofyear', 'minute_of_day']]

        X_list = []
        y_list = []

        total_len = input_len + output_len

        for i in range(len(df) - total_len + 1):
            # 输入序列
            X_seq = df.iloc[i:i+input_len][feature_cols].values
            # 输出序列(只需要功率值)
            y_seq = df.iloc[i+input_len:i+total_len]['generationPower'].values

            X_list.append(X_seq)
            y_list.append(y_seq)

        X = np.array(X_list)
        y = np.array(y_list)

        print(f"创建了 {len(X)} 个训练序列")
        print(f"输入形状: {X.shape} (样本数, 时间步, 特征数)")
        print(f"输出形状: {y.shape} (样本数, 预测步数)")
        print(f"特征列: {feature_cols}")

        return X, y, feature_cols

    def normalize_data(self, X_train, X_val=None, X_test=None):
        """标准化数据"""
        # 重塑为2D进行标准化
        n_samples, n_timesteps, n_features = X_train.shape
        X_train_2d = X_train.reshape(-1, n_features)

        # 拟合标准化器
        X_train_scaled = self.scaler.fit_transform(X_train_2d)
        X_train_scaled = X_train_scaled.reshape(n_samples, n_timesteps, n_features)

        results = [X_train_scaled]

        # 转换验证集和测试集
        if X_val is not None:
            n_samples_val = X_val.shape[0]
            X_val_2d = X_val.reshape(-1, n_features)
            X_val_scaled = self.scaler.transform(X_val_2d)
            X_val_scaled = X_val_scaled.reshape(n_samples_val, n_timesteps, n_features)
            results.append(X_val_scaled)

        if X_test is not None:
            n_samples_test = X_test.shape[0]
            X_test_2d = X_test.reshape(-1, n_features)
            X_test_scaled = self.scaler.transform(X_test_2d)
            X_test_scaled = X_test_scaled.reshape(n_samples_test, n_timesteps, n_features)
            results.append(X_test_scaled)

        return tuple(results) if len(results) > 1 else results[0]

    def save_scaler(self, filepath='scaler.pkl'):
        """保存标准化器"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"标准化器已保存到 {filepath}")

    def load_scaler(self, filepath='scaler.pkl'):
        """加载标准化器"""
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"标准化器已从 {filepath} 加载")


def prepare_data_for_training(data_dir='.', train_ratio=0.7, val_ratio=0.15, use_csv=None):
    """完整的数据准备流程

    Args:
        data_dir: 数据目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        use_csv: 是否使用CSV文件（None表示自动检测，True强制使用CSV，False强制使用JSON）
    """
    processor = PowerDataProcessor(data_dir)

    # 1. 加载数据
    if use_csv is None:
        # 自动检测
        csv_file = os.path.join(data_dir, 'training_data.csv')
        use_csv = os.path.exists(csv_file)

    if use_csv:
        print("\n=== 步骤 1: 加载CSV数据 ===")
        df = processor.load_csv_file('training_data.csv')
    else:
        print("\n=== 步骤 1: 加载JSON数据 ===")
        df = processor.load_json_files()

    # 2. 提取特征
    print("\n=== 步骤 2: 提取时间特征 ===")
    df = processor.extract_time_features(df)

    print("\n=== 步骤 3: 提取功率统计特征 ===")
    df = processor.extract_power_features(df)

    # 3. 创建序列
    print("\n=== 步骤 4: 创建训练序列 ===")
    X, y, feature_cols = processor.create_sequences(df)

    # 4. 划分数据集
    print("\n=== 步骤 5: 划分训练集、验证集、测试集 ===")
    n_samples = len(X)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]

    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]

    print(f"训练集: {len(X_train)} 样本")
    print(f"验证集: {len(X_val)} 样本")
    print(f"测试集: {len(X_test)} 样本")

    # 5. 标准化
    print("\n=== 步骤 6: 数据标准化 ===")
    X_train_scaled, X_val_scaled, X_test_scaled = processor.normalize_data(
        X_train, X_val, X_test)

    # 6. 保存数据和标准化器
    print("\n=== 步骤 7: 保存处理后的数据 ===")
    np.save('X_train.npy', X_train_scaled)
    np.save('y_train.npy', y_train)
    np.save('X_val.npy', X_val_scaled)
    np.save('y_val.npy', y_val)
    np.save('X_test.npy', X_test_scaled)
    np.save('y_test.npy', y_test)

    # 保存特征列名
    with open('feature_cols.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)

    processor.save_scaler('scaler.pkl')

    print("\n=== 数据准备完成 ===")
    print(f"特征维度: {X_train_scaled.shape[-1]}")
    print(f"输入序列长度: {X_train_scaled.shape[1]}")
    print(f"输出序列长度: {y_train.shape[1]}")

    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, feature_cols


if __name__ == '__main__':
    # 运行数据准备
    X_train, y_train, X_val, y_val, X_test, y_test, features = prepare_data_for_training()
