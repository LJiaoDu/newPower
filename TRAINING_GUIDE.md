# 电功率预测训练指南

## 📚 你的问题回答

### Q1: 应该用什么方法训练这些数据？

根据你的数据特点（时间序列、5分钟间隔、电功率预测），我推荐以下方法：

#### ✅ **XGBoost (已实现，效果最好)**
- **文件**: `improved_train.py`
- **优点**: 训练快、准确率高、易于调参
- **效果**: R²=0.9896, 误差率5.62%
- **适用**: 中小规模数据，需要快速迭代

#### 🔄 **LSTM深度学习 (可选尝试)**
- **优点**: 捕捉长期时间依赖
- **缺点**: 需要更多数据、训练时间长
- **适用**: 有大量历史数据时

#### 📈 **Prophet (快速基线)**
- **优点**: 自动处理季节性
- **缺点**: 对短期数据效果一般
- **适用**: 快速原型验证

### Q2: 如何处理processed_power_data.csv发现规律？

## 🔍 数据分析流程

### 第一步: 运行模式分析
```bash
python analyze_patterns.py
```

**发现的关键规律**:
1. **日内周期明显**
   - 峰值时段: 04:00-08:00 (平均15,000-17,000 W)
   - 低谷时段: 12:00-21:00 (80%为零功率)
   - 过渡时段: 21:00-04:00 (快速上升)

2. **星期效应显著** (p < 0.0001)
   - 周六发电量最高 (9,616 W)
   - 周一发电量最低 (2,508 W)
   - 工作日vs周末有明显差异

3. **功率变化特征**
   - 平均变化: -1.56 W
   - 标准差: 2,300 W (波动大)
   - 4.3%的时段有快速变化

4. **相关性分析**
   - hour与功率: -0.522 (负相关最强)
   - dayofweek与功率: +0.319 (正相关)

### 第二步: 基于发现的特征工程

运行改进的训练:
```bash
python improved_train.py
```

**实施的改进策略**:

#### 1. 周期性编码
```python
# 将线性时间转换为周期性
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
```
**为什么**: 23:00和00:00在数值上差23，但实际只差1小时

#### 2. 时段分类特征
```python
is_peak_hour = (hour >= 4) & (hour <= 8)      # 高峰
is_zero_hour = (hour >= 12) & (hour <= 21)    # 零功率
is_transition_hour = ...                       # 过渡
```
**为什么**: 不同时段有完全不同的发电模式

#### 3. 扩展滞后特征
```python
power_lag_1, lag_2, lag_3  # 短期历史
power_lag_288              # 前一天同一时刻
```
**为什么**: 功率有短期惯性和日周期

#### 4. 多窗口滚动统计
```python
rolling_mean_6   # 30分钟趋势
rolling_mean_12  # 1小时趋势
rolling_std_24   # 2小时波动
```
**为什么**: 捕捉不同时间尺度的变化

#### 5. 差分特征
```python
power_diff_1   # 与上一时刻的差
power_diff_12  # 与1小时前的差
```
**为什么**: 变化率比绝对值更有预测价值

#### 6. 交互特征
```python
weekend_hour = is_weekend × hour
```
**为什么**: 周末的小时效应与工作日不同

## 📊 效果对比

| 指标 | 基准模型 | 改进模型 | 提升 |
|------|---------|---------|------|
| MAE | 651.07 W | 337.86 W | **↓ 48.1%** |
| RMSE | 1370.00 W | 611.29 W | **↓ 55.4%** |
| R² | 0.9303 | 0.9896 | **↑ 6.4%** |
| 相对误差 | 16.79% | 5.62% | **↓ 66.5%** |

## 🚀 训练步骤总结

### 快速开始
```bash
# 1. 数据处理
python process_data.py
# 生成: simple_power_data.csv, processed_power_data.csv, training_data.csv

# 2. 模式分析 (可选但推荐)
python analyze_patterns.py
# 生成: pattern_analysis.png (理解数据规律)

# 3. 训练改进模型
python improved_train.py
# 生成: improved_power_model.json, improved_model_results.png
```

### 完整流程
```bash
# Step 1: 理解原始数据
python visualize_data.py

# Step 2: 处理数据
python process_data.py

# Step 3: 深度分析
python analyze_patterns.py

# Step 4: 训练基准模型
python example_train.py

# Step 5: 训练改进模型
python improved_train.py
```

## 💡 提高准确率的关键发现

### 1. 时间特征最重要
- **hour** 是最重要的特征 (相关性-0.522)
- 使用sin/cos编码比直接用数值好

### 2. 历史信息很有用
- 前12步(1小时)的滞后特征
- 前288步(前一天同一时刻)
- 滚动统计(趋势和波动)

### 3. 分段建模思路
零功率时段(12:00-21:00)可以考虑:
- 单独建立二分类模型(是否为零)
- 然后对非零时段建立回归模型

### 4. 数据量是限制因素
当前只有5天数据，建议:
- 收集至少1-2个月数据
- 能捕捉更完整的周期性
- 更好地训练深度学习模型

## 🎯 下一步优化建议

### 短期改进
1. **集成学习**: 组合XGBoost、LightGBM、CatBoost
2. **交叉验证**: 使用时间序列交叉验证
3. **超参数优化**: Grid Search或Bayesian Optimization

### 中期改进
1. **LSTM模型**: 尝试深度学习
2. **特征选择**: 使用SHAP值分析
3. **异常检测**: 识别和处理异常值

### 长期改进
1. **增加外部数据**: 天气、节假日
2. **在线学习**: 模型定期更新
3. **多步预测**: 预测未来多个时刻

## 📁 文件说明

| 文件 | 用途 | 何时使用 |
|------|------|---------|
| `process_data.py` | 数据清洗和预处理 | 首次使用或有新数据 |
| `visualize_data.py` | 数据可视化 | 理解数据分布 |
| `analyze_patterns.py` | 深度模式分析 | 发现规律和特征 |
| `example_train.py` | 基准模型训练 | 快速基线 |
| `improved_train.py` | **改进模型训练** | **获得最佳效果** ⭐ |

## 🔑 核心要点

1. **数据质量 > 模型复杂度**
   - 先确保数据清洗正确
   - 再考虑复杂模型

2. **特征工程是关键**
   - 48%的准确率提升来自更好的特征
   - 不是换了更复杂的模型

3. **理解业务逻辑**
   - 分析发现峰值时段和零功率时段
   - 这些领域知识转化为特征

4. **迭代优化**
   - 基准模型 → 分析 → 改进 → 再分析
   - 持续改进的循环

## 📞 常见问题

### Q: 为什么数据从1606行变成913行？
A: 因为创建滞后特征(lag_288需要前288个数据点)和滚动窗口，前面的数据没有足够的历史信息，所以被删除了。这是正常的。

### Q: 可以直接用improved_train.py吗？
A: 可以！运行 `python improved_train.py` 即可，它会自动加载processed_power_data.csv并训练。

### Q: 如果有新数据怎么办？
A:
1. 把新的JSON文件放到同一目录
2. 运行 `python process_data.py` 重新处理
3. 运行 `python improved_train.py` 重新训练

### Q: 如何使用训练好的模型预测？
A:
```python
import xgboost as xgb
model = xgb.XGBRegressor()
model.load_model('improved_power_model.json')

# 准备新数据(需要包含所有43个特征)
predictions = model.predict(X_new)
```

---

**总结**: 使用 `improved_train.py` 可以获得最佳效果 (误差率5.62%)，它通过深入分析数据规律，设计了针对性的特征，显著提升了预测准确率。
