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
        # 检查数据项是否包含必要字段
        if "messages" not in item or "images" not in item:
            print(f"Warning: Item missing required fields (messages or images)")
            continue
        
        # 获取图像路径
        image_paths = item["images"]
        if not image_paths:
            print(f"Warning: No images found in item")
            continue
        
        # 检查图像文件是否存在
        valid_images = []
        for image_path in image_paths:
            if os.path.exists(image_path):
                valid_images.append(image_path)
            else:
                print(f"Warning: Image file not found: {image_path}")
        
        if not valid_images:
            print(f"Warning: No valid images found for item")
            continue
        
        # 从messages中提取用户提示和助手回答
        user_prompt = None
        assistant_answer = None
        
        for message in item["messages"]:
            if message["role"] == "user":
                # 提取用户提示，去除<image>标签
                content = message["content"]
                user_prompt = content.replace("<image>", "").strip()
            elif message["role"] == "assistant":
                assistant_answer = message["content"]
        
        if not user_prompt or not assistant_answer:
            print(f"Warning: Missing user prompt or assistant answer in messages")
            continue
        
        # 构建转换后的数据项
        converted_item = {
            "prompt": user_prompt,
            "images": valid_images,
            "answer": assistant_answer
        }
        print(converted_item)
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