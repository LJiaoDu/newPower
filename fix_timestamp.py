"""
修复时间戳格式错误的JSON文件
将秒级时间戳转换为毫秒级
"""
import json

filename = 'temp_2025-05-11.json'

print(f"修复文件: {filename}")

# 读取文件
with open(filename, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 修复 updateTime
if 'stationStatisticDay' in data and 'updateTime' in data['stationStatisticDay']:
    old_time = data['stationStatisticDay']['updateTime']
    if old_time < 2000000000:  # 如果小于这个值，说明是秒级
        data['stationStatisticDay']['updateTime'] = old_time * 1000
        print(f"  修复 stationStatisticDay.updateTime: {old_time} -> {old_time * 1000}")

# 修复功率列表中的时间戳
power_list = data.get('stationStatisticPowerList', [])
fixed_count = 0

for item in power_list:
    # 修复 dateTime
    if 'dateTime' in item:
        old_time = item['dateTime']
        if old_time < 2000000000:  # 秒级时间戳
            item['dateTime'] = old_time * 1000
            fixed_count += 1

    # 修复 updateTime
    if 'updateTime' in item:
        old_time = item['updateTime']
        if old_time < 2000000000:
            item['updateTime'] = old_time * 1000

print(f"  修复了 {fixed_count} 个功率数据点的时间戳")

# 保存修复后的文件
output_filename = '772_2025-05-11.json'
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ 修复完成！保存到: {output_filename}")

# 验证修复结果
with open(output_filename, 'r') as f:
    fixed_data = json.load(f)

sample_ts = fixed_data['stationStatisticPowerList'][0]['dateTime']
from datetime import datetime
dt = datetime.fromtimestamp(sample_ts / 1000)
print(f"\n验证: 第一个时间戳 {sample_ts} -> {dt}")
