# 新数据格式适配说明

## 数据格式变化

### 旧格式 (772_2024-06-01.json)
```json
{
  "stationStatisticDay": { ... 日统计数据 ... },
  "stationStatisticPowerList": [
    {
      "id": "772_20240601_70",
      "seq": 70,
      "systemId": 772,
      "acceptDay": "20240601",
      "year": 2024,
      "month": 6,
      "day": 1,
      "generationPower": 2509.0,
      "dateTime": 1717192498000,  // 毫秒时间戳
      "updateTime": 1717192498000,
      "generationCapacity": null
    },
    ...
  ]
}
```

### 新格式 (2025-03-13.json)
```json
{
  "systemId": 104606,
  "locationLat": 31.033461,
  "locationLng": 112.204804,
  "powerInfos": [
    {
      "time": 1741858976,  // 秒时间戳 (注意：不是毫秒!)
      "seq": 212,
      "power": 27619.0
    },
    ...
  ]
}
```

## 关键变化总结

| 项目 | 旧格式 | 新格式 |
|------|--------|--------|
| 功率列表字段 | `stationStatisticPowerList` | `powerInfos` |
| 时间字段名 | `dateTime` | `time` |
| 时间单位 | 毫秒 | **秒** |
| 功率字段名 | `generationPower` | `power` |
| 位置信息 | 无 | `locationLat`, `locationLng` |
| 日统计 | `stationStatisticDay` | 无 |

## 新增文件

### 1. te_data_process.py
**用途**: 处理新格式JSON数据，生成训练用CSV文件

**主要改动**:
- `load_json_files_new_format()`: 支持新的JSON结构
  - 读取 `powerInfos` 代替 `stationStatisticPowerList`
  - 时间字段从 `dateTime`(毫秒) 改为 `time`(秒)
  - 功率字段从 `generationPower` 改为 `power`
  - 提取经纬度信息 `locationLat`, `locationLng`

**使用方法**:
```bash
# 将新格式的JSON文件放在当前目录
python te_data_process.py
```

**输出文件**:
- `te_processed_power_data.csv` - 带时间特征的完整数据
- `te_training_data.csv` - 用于训练的数据
- `te_simple_power_data.csv` - 简化版数据
- `te_location_info.json` - 位置信息

### 2. te_train.py
**用途**: 使用新格式数据进行模型训练

**主要改动**:
- 默认数据路径: `./training_data.csv` → `./te_training_data.csv`
- 检查点目录: `./checkpoint/` → `./te_checkpoint/`
- 预测图保存: `prediction_plots/` → `te_prediction_plots/`
- TensorBoard日志: `runs/improved1_optimized_*` → `runs/te_training_*`

**使用方法**:
```bash
# 1. 先处理数据
python te_data_process.py

# 2. 开始训练
python te_train.py --epochs 100 --batch-size 64 --device 0

# 可选参数
python te_train.py \
  --data-path ./te_training_data.csv \
  --epochs 50 \
  --batch-size 32 \
  --lr-init 0.0003 \
  --device 0
```

## 原文件保持不变

以下文件**不需要修改**，继续支持旧格式数据:
- `process_data.py` - 处理旧格式JSON
- `train.py` - 使用旧格式数据训练
- `model.py` - 模型定义（新旧格式通用）

## 完整工作流程

### 旧格式数据流程
```bash
# 数据处理
python process_data.py

# 模型训练
python train.py --data-path ./training_data.csv
```

### 新格式数据流程
```bash
# 数据处理 (新)
python te_data_process.py

# 模型训练 (新)
python te_train.py --data-path ./te_training_data.csv
```

## 目录结构

```
newPower/
├── 772_2024-06-01.json          # 旧格式数据示例
├── 2025-03-13.json              # 新格式数据示例
│
├── process_data.py              # 旧格式数据处理 (保持不变)
├── train.py                     # 旧格式训练脚本 (保持不变)
├── model.py                     # 模型定义 (通用)
│
├── te_data_process.py           # 新格式数据处理 (新增)
├── te_train.py                  # 新格式训练脚本 (新增)
│
├── checkpoint/                  # 旧格式模型检查点
├── te_checkpoint/               # 新格式模型检查点 (新增)
│
├── prediction_plots/            # 旧格式预测图
├── te_prediction_plots/         # 新格式预测图 (新增)
│
└── README_NEW_FORMAT.md         # 本说明文档
```

## 注意事项

1. **时间戳单位**: 新格式使用秒时间戳，旧格式使用毫秒时间戳
2. **数据兼容性**: 新旧格式生成的CSV文件格式完全相同，可互换使用
3. **模型通用性**: `model.py` 不需要修改，可同时用于新旧格式训练
4. **检查点隔离**: 新旧格式使用不同的检查点目录，避免混淆

## 快速测试

```bash
# 测试新格式数据处理
python te_data_process.py

# 测试新格式训练 (使用小数据集快速验证)
python te_train.py --subset 0.1 --epochs 5 --batch-size 32
```

## 模型性能指标

训练完成后查看:
- **TensorBoard**: `tensorboard --logdir runs/`
- **预测曲线**: `te_prediction_plots/` 目录
- **最佳模型**: `te_checkpoint/best_epoch*.pth`

## 常见问题

### Q: 为什么要创建新文件而不是修改原文件?
A: 保持向后兼容性，确保旧格式数据仍可正常使用。

### Q: 新旧格式能否混合使用?
A: CSV文件格式相同，可混合。但建议分开处理避免混淆。

### Q: model.py 需要修改吗?
A: 不需要。模型只处理CSV数据，与JSON格式无关。

### Q: 如何验证数据处理是否正确?
A: 运行 `te_data_process.py` 后，检查生成的CSV文件和输出的统计信息。
