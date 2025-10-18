#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
将医学影像报告数据集转换为EasyR1可用的格式
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm


def convert_medical_report_dataset(input_file, output_dir, split="train"):
    """
    将医学影像报告数据集转换为EasyR1可用的格式
    
    Args:
        input_file: 输入的JSON文件路径
        output_dir: 输出目录
        split: 数据集分割，默认为train
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 转换数据格式
    converted_data = []
    for item in tqdm(data, desc=f"Converting {split} data"):
        # 检查图像文件是否存在
        image_path = item["image_path"]
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}")
            continue
        
        # 构建转换后的数据项
        converted_item = {
            "prompt": f"Based on this medical X-ray image, please analyze and generate a diagnostic report.", # <img>{image_path}</img>
            "images": [image_path],
            "answer": item["report"]
        }
        converted_data.append(converted_item)
    
    # 保存转换后的数据
    output_file = os.path.join(output_dir, f"medical_report_{split}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"Converted {len(converted_data)} items, saved to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Convert medical report dataset to EasyR1 format")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (train/val)")
    
    args = parser.parse_args()
    
    convert_medical_report_dataset(args.input, args.output_dir, args.split)


if __name__ == "__main__":
    main() 