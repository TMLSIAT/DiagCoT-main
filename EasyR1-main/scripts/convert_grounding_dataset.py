#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import pandas as pd
import argparse
import os

def convert_grounding_data(input_file: str, output_dir: str):
    """
    将grounding数据转换为训练格式
    Args:
        input_file: 输入的JSON文件路径
        output_dir: 输出目录
    """
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    converted_data = []
    
    for item in data:
        messages = item['messages']
        images = item['images']
        
        # 获取用户的prompt（去掉<image>标记）
        user_content = messages[0]['content']
        prompt = user_content.replace('<image>', '').strip()
        
        # 获取助手的回答
        assistant_content = messages[1]['content']
        
        converted_item = {
            'prompt': prompt,
            'images': images,
            'answer': assistant_content
        }
        converted_data.append(converted_item)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为parquet格式
    df = pd.DataFrame(converted_data)
    output_file = os.path.join(output_dir, 'grounding_train_balanced.parquet')
    df.to_parquet(output_file, index=False)
    
    print(f"转换完成！")
    print(f"总样本数: {len(converted_data)}")
    print(f"输出文件: {output_file}")
    print(f"数据字段: {df.columns.tolist()}")

def main():
    parser = argparse.ArgumentParser(description='转换grounding数据集格式')
    parser.add_argument('--input', default="/data0/zhuoxu/yihong/code/EasyR1-main/examples/data/medical_grouding_rsna/val_grounding_wo_think_augmented_Qwen2_vl.json", help='输入JSON文件路径')
    parser.add_argument('--output_dir', default="/data0/zhuoxu/yihong/code/EasyR1-main/examples/data/medical_grouding_rsna", help='输出目录')
    
    args = parser.parse_args()
    
    convert_grounding_data(args.input, args.output_dir)

if __name__ == "__main__":
    main() 