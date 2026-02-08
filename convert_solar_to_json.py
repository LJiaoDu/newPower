import pandas as pd
import json


def convert_excel_to_json(excel_path, output_path, station_id, nominal_capacity):
    # 读取Excel文件
    df = pd.read_excel(excel_path)

    # 加上站点号列
    df.insert(0, "station_id", station_id)

    # 加上最大Nominal capacity列
    df["nominal_capacity_MW"] = nominal_capacity

    # 转换为JSON格式并保存
    records = df.to_dict(orient="records")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"处理完成: {output_path}")
    print(f"记录数: {len(records)}")


if __name__ == "__main__":
    convert_excel_to_json(
        excel_path="Solar station site 1 (Nominal capacity-50MW).xlsx",
        output_path="solar_station_1.json",
        station_id=1,
        nominal_capacity=50,
    )
