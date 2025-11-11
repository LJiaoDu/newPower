"""
将JSON格式的功率数据转换为CSV格式
"""
import json
import pandas as pd
import os
from datetime import datetime

def convert_json_to_csv(json_dir='.', output_file='training_data.csv'):
    """
    将所有JSON文件转换为单个CSV文件

    Args:
        json_dir: JSON文件所在目录
        output_file: 输出CSV文件名
    """
    all_data = []

    # 获取所有JSON文件
    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])
    print(f"找到 {len(json_files)} 个JSON文件")

    for json_file in json_files:
        file_path = os.path.join(json_dir, json_file)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # 提取功率列表
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
            print(f"处理文件 {json_file} 时出错: {e}")
            continue

    # 转换为DataFrame
    df = pd.DataFrame(all_data)

    # 转换时间戳为datetime
    df['dateTime'] = pd.to_datetime(df['dateTime'], unit='ms')

    # 按时间排序
    df = df.sort_values('dateTime').reset_index(drop=True)

    # 保存为CSV
    df.to_csv(output_file, index=False)

    print(f"\n成功转换 {len(df)} 条记录")
    print(f"CSV文件已保存到: {output_file}")
    print(f"\n数据预览:")
    print(df.head())
    print(f"\n数据信息:")
    print(df.info())

    return df

if __name__ == '__main__':
    convert_json_to_csv()
