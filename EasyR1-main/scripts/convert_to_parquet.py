#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
将JSON格式的医学影像报告数据集转换为Parquet格式
"""

import os
import json
import argparse
import pandas as pd
from pathlib import Path


def convert_json_to_parquet(input_file, output_file):
    """
    将JSON文件转换为Parquet文件
    
    Args:
        input_file: 输入的JSON文件路径
        output_file: 输出的Parquet文件路径
    """
    # 读取JSON文件
    print(f"Reading JSON file: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 转换为DataFrame
    print(f"Converting to DataFrame with {len(data)} rows")
    df = pd.DataFrame(data)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 保存为Parquet格式
    print(f"Saving to Parquet file: {output_file}")
    df.to_parquet(output_file, index=False)
    
    print(f"Conversion completed. Saved {len(data)} rows to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert JSON dataset to Parquet format")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output", type=str, required=True, help="Output Parquet file path")
    
    args = parser.parse_args()
    
    convert_json_to_parquet(args.input, args.output)


if __name__ == "__main__":
    main() 