#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import re
import os
from typing import List, Dict, Any

def has_coordinates(content: str) -> bool:
    """
    检查内容是否包含坐标信息
    Args:
        content: 文本内容
    Returns:
        bool: 是否包含坐标
    """
    # 匹配 <box>(x1,y1),(x2,y2)</box> 格式的坐标
    box_pattern = r'<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>'
    return bool(re.search(box_pattern, content))

def filter_dataset_with_coordinates(input_file: str, output_file: str) -> None:
    """
    过滤数据集，只保留包含坐标的数据
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    print(f"正在读取数据集: {input_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"原始数据集大小: {len(data)} 条")
        
        # 过滤包含坐标的数据
        filtered_data = []
        
        for item in data:
            if 'messages' in item and len(item['messages']) >= 2:
                # 检查assistant的回复是否包含坐标
                assistant_message = item['messages'][1]
                if assistant_message.get('role') == 'assistant':
                    content = assistant_message.get('content', '')
                    if has_coordinates(content):
                        filtered_data.append(item)
        
        print(f"过滤后数据集大小: {len(filtered_data)} 条")
        print(f"保留比例: {len(filtered_data)/len(data)*100:.2f}%")
        
        # 保存过滤后的数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)
        
        print(f"过滤后的数据已保存到: {output_file}")
        
        # 输出一些统计信息
        coordinate_counts = []
        for item in filtered_data:
            content = item['messages'][1]['content']
            boxes = re.findall(r'<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>', content)
            coordinate_counts.append(len(boxes))
        
        if coordinate_counts:
            print(f"坐标统计:")
            print(f"  - 平均每个样本包含 {sum(coordinate_counts)/len(coordinate_counts):.2f} 个坐标")
            print(f"  - 最多包含 {max(coordinate_counts)} 个坐标")
            print(f"  - 最少包含 {min(coordinate_counts)} 个坐标")
            
            # 统计坐标数量分布
            from collections import Counter
            coord_distribution = Counter(coordinate_counts)
            print(f"  - 坐标数量分布:")
            for count, freq in sorted(coord_distribution.items()):
                print(f"    {count} 个坐标: {freq} 条数据 ({freq/len(filtered_data)*100:.1f}%)")
        
    except Exception as e:
        print(f"处理数据时出错: {e}")
        raise

def main():
    """主函数"""
    input_file = "val_grounding_wo_think.json"
    output_file = "val_grounding_wo_think_with_coordinates.json"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 {input_file} 不存在")
        return
    
    # 过滤数据集
    filter_dataset_with_coordinates(input_file, output_file)
    
    print("数据集过滤完成！")

if __name__ == "__main__":
    main() 