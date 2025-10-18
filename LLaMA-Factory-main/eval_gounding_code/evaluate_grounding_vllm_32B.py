#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用VLLM部署的32B模型进行医学影像标框任务评估
通过OpenAI API接口调用本地VLLM服务
结合标框任务评估和VLLM API调用功能
"""

import json
import os
import re
import base64
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Any, Optional
from openai import OpenAI
from retrying import retry
import warnings
warnings.filterwarnings("ignore")

def encode_image(image_path):
    """
    将图像文件编码为base64格式
    
    Args:
        image_path (str): 图像文件路径
        
    Returns:
        str: base64编码的图像数据
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

class VLLMGroundingClassifier:
    """
    使用VLLM API的医学影像标框分类器
    """
    
    def __init__(self, model_name, api_key="qwen-abc123", api_url="http://127.0.0.1:8000/v1"):
        """
        初始化VLLM标框分类器
        
        Args:
            model_name (str): 模型名称
            api_key (str): API密钥
            api_url (str): API服务地址
        """
        self.model_name = model_name
        self.api_key = api_key
        self.api_url = api_url
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_url,
        )
        print(f"初始化VLLM标框分类器，模型: {model_name}, API地址: {api_url}")

    def _construct_messages(self, image_path):
        """
        构造发送给模型的消息格式
        
        Args:
            image_path (str): 图像文件路径
            
        Returns:
            list: 格式化的消息列表
        """
        # 将图像编码为base64
        base64_image = encode_image(image_path)
        
        return [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                },
                {
                    "type": "text",
                    "text": "Please detect and locate any lung opacity regions in this chest X-ray image.\n\nDetection Guidelines:\n- Look for areas of increased density in the lung fields\n- Consider consolidation, infiltrates, or other opacity patterns\n- Ensure coordinates are within image boundaries\n- Provide precise (x1,y1),(x2,y2) coordinates\n\nOutput Format:\n\"The detected lung opacity regions are: Lung Opacity: <box>(x1,y1),(x2,y2)</box>\"\n\nIf no opacity is detected, output: \"The detected lung opacity regions are: No lung opacity regions detected.\" Output the thinking process in <think> </think>, and output the final detection result within the <answer> </answer>. Following \"<think></think>\\n<answer></answer> \" format."
                }
            ]
        }]

    @retry(wait_fixed=3000, stop_max_attempt_number=3)
    def _retry_call(self, messages):
        """
        带重试机制的API调用
        
        Args:
            messages (list): 消息列表
            
        Returns:
            str: 模型生成的响应
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            # max_tokens=4096,
        )
        return response.choices[0].message.content

    def detect(self, image_path):
        """
        对医学影像进行标框检测
        
        Args:
            image_path (str): 图像文件路径
            
        Returns:
            str: 检测结果
        """
        try:
            # 构造消息
            messages = self._construct_messages(image_path)
            
            # 调用API进行检测
            response = self._retry_call(messages)
            
            return response if response else ""
            
        except Exception as e:
            print(f"检测时出错 {image_path}: {str(e)}")
            return ""

def extract_answer_from_tags(text: str) -> str:
    """
    从文本中提取<answer></answer>标签内的内容
    如果没有找到标签，返回原始文本
    
    Args:
        text (str): 输入文本
        
    Returns:
        str: 提取的答案内容
    """
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def parse_bbox_from_text(text: str) -> List[Tuple[float, float, float, float]]:
    """
    从模型输出文本中解析bbox坐标
    格式: <box>(x1,y1),(x2,y2)</box>
    支持相对坐标(0-1)和绝对坐标(>1)
    
    Args:
        text (str): 输入文本
        
    Returns:
        List[Tuple[float, float, float, float]]: 相对坐标格式的bbox列表
    """
    # 首先提取answer标签内的内容
    answer_text = extract_answer_from_tags(text)
    print("answer_text:", answer_text)
    
    # 统一的坐标匹配格式，支持整数和小数
    box_pattern = r'<box>\(([0-9.]+),([0-9.]+)\),\(([0-9.]+),([0-9.]+)\)</box>'
    matches = re.findall(box_pattern, answer_text)
    
    bboxes = []
    
    for match in matches:
        x1, y1, x2, y2 = map(float, match)
        
        # 判断是相对坐标还是绝对坐标
        # 如果所有坐标都在0-1范围内，认为是相对坐标
        if 0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1:
            # 相对坐标，直接使用
            x1_rel, y1_rel, x2_rel, y2_rel = x1, y1, x2, y2
        else:
            # 绝对坐标，转换为相对坐标（除以1000）
            x1_rel = x1 / 1000.0
            y1_rel = y1 / 1000.0
            x2_rel = x2 / 1000.0
            y2_rel = y2 / 1000.0
            
            # 检查转换后的坐标是否合理
            if not (0 <= x1_rel <= 1 and 0 <= y1_rel <= 1 and 0 <= x2_rel <= 1 and 0 <= y2_rel <= 1):
                # 如果转换后仍不在0-1范围内，跳过这个边界框
                continue
        
        # 确保坐标顺序正确 (左上角到右下角)
        x1_rel, x2_rel = min(x1_rel, x2_rel), max(x1_rel, x2_rel)
        y1_rel, y2_rel = min(y1_rel, y2_rel), max(y1_rel, y2_rel)
        
        # 检查边界框是否有效（宽度和高度大于0）
        if x2_rel > x1_rel and y2_rel > y1_rel:
            bboxes.append((x1_rel, y1_rel, x2_rel, y2_rel))
    
    return bboxes

def parse_prediction_text_improved(text: str) -> Tuple[List[Tuple[float, float, float, float]], bool]:
    """
    改进的预测文本解析函数
    支持从<answer></answer>标签中提取答案
    
    Args:
        text (str): 预测文本
        
    Returns:
        Tuple[List[Tuple[float, float, float, float]], bool]: (bbox列表, 是否有异常)
    """
    # 首先提取answer标签内的内容
    answer_text = extract_answer_from_tags(text)
    
    # 检查是否明确表示没有检测到异常
    no_finding_patterns = [
        "no lung opacity regions detected",
        "no abnormality detected",
        "no abnormal areas found",
        "no opacity regions found",
        "normal chest x-ray",
        "no findings",
        "no lesions detected"
    ]
    
    text_lower = answer_text.lower()
    has_no_finding = any(pattern in text_lower for pattern in no_finding_patterns)
    
    # 解析bounding box
    bboxes = parse_bbox_from_text(text)
    
    # 判断是否有异常
    has_abnormality = len(bboxes) > 0 or not has_no_finding
    
    return bboxes, has_abnormality

def calculate_iou(bbox1: Tuple[float, float, float, float], 
                  bbox2: Tuple[float, float, float, float]) -> float:
    """
    计算两个bbox的IoU
    
    Args:
        bbox1 (Tuple[float, float, float, float]): 第一个bbox (x1, y1, x2, y2)
        bbox2 (Tuple[float, float, float, float]): 第二个bbox (x1, y1, x2, y2)
        
    Returns:
        float: IoU分数
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # 计算交集
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    
    # 计算各自面积
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # 计算并集
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union

def evaluate_detection_improved(pred_bboxes: List[Tuple[float, float, float, float]], 
                               gt_bboxes: List[Tuple[float, float, float, float]], 
                               pred_has_abnormality: bool,
                               gt_has_abnormality: bool,
                               iou_threshold: float = 0.5) -> Dict[str, Any]:
    """
    改进的检测评估函数
    处理两种情况：有框和无框
    
    Args:
        pred_bboxes: 预测的bbox列表
        gt_bboxes: 真实的bbox列表
        pred_has_abnormality: 预测是否有异常
        gt_has_abnormality: 真实是否有异常
        iou_threshold: IoU阈值
        
    Returns:
        Dict[str, Any]: 评估结果
    """
    # 情况1: 真实无异常，预测无异常 -> 正确
    if not gt_has_abnormality and not pred_has_abnormality:
        return {
            "correct": True,
            "case": "true_negative",
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "iou_score": 1.0
        }
    
    # 情况2: 真实无异常，预测有异常 -> 错误
    if not gt_has_abnormality and pred_has_abnormality:
        return {
            "correct": False,
            "case": "false_positive",
            "precision": 0.0,
            "recall": 1.0,
            "f1": 0.0,
            "iou_score": 0.0
        }
    
    # 情况3: 真实有异常，预测无异常 -> 错误
    if gt_has_abnormality and not pred_has_abnormality:
        return {
            "correct": False,
            "case": "false_negative",
            "precision": 1.0,
            "recall": 0.0,
            "f1": 0.0,
            "iou_score": 0.0
        }
    
    # 情况4: 真实有异常，预测有异常 -> 需要计算IoU
    if gt_has_abnormality and pred_has_abnormality:
        if len(pred_bboxes) == 0 or len(gt_bboxes) == 0:
            return {
                "correct": False,
                "case": "bbox_mismatch",
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "iou_score": 0.0
            }
        
        # 计算每个预测框与真实框的最大IoU
        matched_gt = set()
        true_positives = 0
        max_iou = 0.0
        
        for pred_bbox in pred_bboxes:
            pred_max_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx, gt_bbox in enumerate(gt_bboxes):
                if gt_idx in matched_gt:
                    continue
                
                iou = calculate_iou(pred_bbox, gt_bbox)
                if iou > pred_max_iou:
                    pred_max_iou = iou
                    best_gt_idx = gt_idx
            
            if pred_max_iou > max_iou:
                max_iou = pred_max_iou
            
            if pred_max_iou >= iou_threshold and best_gt_idx != -1:
                matched_gt.add(best_gt_idx)
                true_positives += 1
        
        precision = true_positives / len(pred_bboxes) if len(pred_bboxes) > 0 else 0.0
        recall = true_positives / len(gt_bboxes) if len(gt_bboxes) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # 判断是否正确 (IoU阈值和F1分数)
        is_correct = max_iou >= iou_threshold and f1 > 0.5
        
        return {
            "correct": is_correct,
            "case": "true_positive" if is_correct else "low_iou_or_f1",
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou_score": max_iou,
            "max_iou": max_iou,
            "matched_pairs": true_positives,
            "total_predictions": len(pred_bboxes),
            "total_ground_truths": len(gt_bboxes)
        }
    
    # 兜底返回
    return {
        "correct": False,
        "case": "unknown",
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "iou_score": 0.0
    }

def load_test_data_improved(test_file: str) -> List[Dict[str, Any]]:
    """
    改进的测试数据加载函数
    支持从<answer></answer>标签中提取答案
    
    Args:
        test_file (str): 测试文件路径
        
    Returns:
        List[Dict[str, Any]]: 处理后的测试数据
    """
    print(f"正在加载测试数据: {test_file}")
    with open(test_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # 转换为统一格式
    processed_data = []
    for item in raw_data:
        # 从assistant的回答中提取真实标注
        assistant_content = item["messages"][1]["content"]
        image_path = item["images"][0]
        
        # 提取answer标签内容（如果存在）
        extracted_answer = extract_answer_from_tags(assistant_content)
        
        # 解析真实标注
        gt_bboxes = parse_bbox_from_text(assistant_content)
        gt_has_abnormality = len(gt_bboxes) > 0
        
        # 创建统一格式的数据项
        data_item = {
            "image": image_path,
            "annotations": [{"bbox": bbox} for bbox in gt_bboxes],  # 直接使用相对坐标
            "gt_bboxes_relative": gt_bboxes,  # 保存相对坐标
            "has_abnormality": gt_has_abnormality,
            "raw_annotation": assistant_content,
            "extracted_answer": extracted_answer  # 保存提取的答案
        }
        processed_data.append(data_item)
    
    print(f"数据加载完成:")
    print(f"  总样本数: {len(processed_data)}")
    print(f"  有病变样本: {sum(1 for item in processed_data if item['has_abnormality'])}")
    print(f"  无病变样本: {sum(1 for item in processed_data if not item['has_abnormality'])}")
    
    return processed_data

def evaluate_model_vllm(model_name: str, test_file: str, api_url: str = "http://127.0.0.1:8000/v1", 
                       output_file: str = None, max_samples: int = None,
                       iou_thresholds: List[float] = [0.3, 0.5, 0.7]):
    """
    使用VLLM API评估模型性能
    
    Args:
        model_name (str): 模型名称
        test_file (str): 测试文件路径
        api_url (str): VLLM API服务地址
        output_file (str): 输出结果文件路径
        max_samples (int): 最大测试样本数量
        iou_thresholds (List[float]): IoU阈值列表
        
    Returns:
        tuple: (最终指标, 结果列表)
    """
    # 加载测试数据
    test_data = load_test_data_improved(test_file)
    
    if max_samples:
        test_data = test_data[:max_samples]
        print(f"限制测试样本数量为: {max_samples}")
    
    print(f"测试样本总数: {len(test_data)}")
    
    # 初始化VLLM分类器
    classifier = VLLMGroundingClassifier(model_name, api_url=api_url)
    
    # 存储结果
    results = []
    
    print("开始评估...")
    for i, item in enumerate(tqdm(test_data, desc="评估进度")):
        image_path = item['image']
        
        # 检查图片是否存在
        if not os.path.exists(image_path):
            print(f"图片不存在: {image_path}")
            continue
        
        # 生成预测
        prediction = classifier.detect(image_path)
        print("prediction:", prediction)
        
        # 从预测结果中提取answer
        extracted_prediction_answer = extract_answer_from_tags(prediction)
        
        result = {
            "image": item["image"],
            "ground_truth": item["annotations"],
            "gt_bboxes_relative": item["gt_bboxes_relative"],
            "gt_has_abnormality": item["has_abnormality"],
            "prediction": prediction,
            "raw_annotation": item.get("raw_annotation", ""),
            "extracted_answer": item.get("extracted_answer", ""),  # 保留GT的extracted_answer
            "extracted_prediction_answer": extracted_prediction_answer  # 新增：从预测中提取的answer
        }
        results.append(result)
        
        # 打印进度信息
        if (i + 1) % 10 == 0:
            print(f"已处理 {i + 1}/{len(test_data)} 个样本")
    
    # 评估结果
    final_metrics = evaluate_results_improved(results, iou_thresholds)
    
    # 保存结果
    if output_file:
        print(f"\n保存结果到: {output_file}")
        output_data = {
            'model_info': {
                'model_name': model_name,
                'api_url': api_url,
                'total_samples': len(results)
            },
            'final_metrics': final_metrics,
            'detailed_results': results,
            'summary': {
                'total_samples': len(results),
                'model_name': model_name,
                'test_file': test_file
            }
        }
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    return final_metrics, results

def evaluate_results_improved(results: List[Dict[str, Any]], 
                             iou_thresholds: List[float] = [0.3, 0.5, 0.7]) -> Dict[str, Any]:
    """
    改进的评估函数
    
    Args:
        results: 结果列表
        iou_thresholds: IoU阈值列表
        
    Returns:
        Dict[str, Any]: 评估指标
    """
    # 按IoU阈值统计的指标
    iou_metrics = {}
    for iou_th in iou_thresholds:
        iou_metrics[f"iou_{iou_th}"] = {
            "correct_samples": 0,
            "total_samples": 0,
            "true_negatives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "true_positives": 0,
            "low_iou_or_f1": 0
        }
    
    # 专门统计"真实有框且预测有框"情况的IoU值
    both_has_bbox_ious = []
    
    # 用于实时统计mean_iou的变量
    all_ious = []
    samples_with_bbox = 0
    
    for i, result in enumerate(results):
        # 解析预测结果
        pred_text = result["prediction"]
        pred_bboxes_relative, pred_has_abnormality = parse_prediction_text_improved(pred_text)
        
        # 获取真实标注
        gt_bboxes_relative = result["gt_bboxes_relative"]
        gt_bboxes = [bbox for bbox in gt_bboxes_relative]  # 直接使用相对坐标
        gt_has_abnormality = result["gt_has_abnormality"]
        
        # 专门计算"真实有框且预测有框"情况的IoU
        if gt_has_abnormality and pred_has_abnormality and len(gt_bboxes) > 0 and len(pred_bboxes_relative) > 0:
            sample_ious = []
            for pred_bbox in pred_bboxes_relative:
                for gt_bbox in gt_bboxes:
                    iou = calculate_iou(pred_bbox, gt_bbox)
                    sample_ious.append(iou)
                    both_has_bbox_ious.append(iou)
                    all_ious.append(iou)  # 添加到全局IoU列表
            
            # 计算该样本的最大IoU
            if sample_ious:
                max_sample_iou = max(sample_ious)
            else:
                max_sample_iou = 0.0
            
            samples_with_bbox += 1
        else:
            max_sample_iou = 0.0
        
        # 实时打印mean_iou（每10个样本打印一次）
        if (i + 1) % 10 == 0 and all_ious:
            current_mean_iou = np.mean(all_ious)
            print(f"📊 已处理 {i + 1}/{len(results)} 个样本，当前mean_iou: {current_mean_iou:.4f} (基于 {len(all_ious)} 个IoU值)")
        
        # 对每个IoU阈值进行评估
        for iou_th in iou_thresholds:
            eval_result = evaluate_detection_improved(
                pred_bboxes_relative, gt_bboxes, pred_has_abnormality, gt_has_abnormality, iou_th
            )
            
            metrics_key = f"iou_{iou_th}"
            iou_metrics[metrics_key]["total_samples"] += 1
            
            if eval_result["correct"]:
                iou_metrics[metrics_key]["correct_samples"] += 1
            
            # 统计各种情况
            case = eval_result["case"]
            if case == "true_negative":
                iou_metrics[metrics_key]["true_negatives"] += 1
            elif case == "false_positive":
                iou_metrics[metrics_key]["false_positives"] += 1
            elif case == "false_negative":
                iou_metrics[metrics_key]["false_negatives"] += 1
            elif case == "true_positive":
                iou_metrics[metrics_key]["true_positives"] += 1
            elif case == "low_iou_or_f1":
                iou_metrics[metrics_key]["low_iou_or_f1"] += 1
    
    # 计算最终指标
    final_metrics = {
        "total_samples": len(results),
        "samples_with_bbox": samples_with_bbox,
        "mean_iou_all": np.mean(all_ious) if all_ious else 0.0,
        "mean_iou_both_has_bbox": np.mean(both_has_bbox_ious) if both_has_bbox_ious else 0.0,
        "iou_metrics": iou_metrics
    }
    
    # 计算每个IoU阈值的准确率
    for iou_th in iou_thresholds:
        metrics_key = f"iou_{iou_th}"
        total = iou_metrics[metrics_key]["total_samples"]
        correct = iou_metrics[metrics_key]["correct_samples"]
        accuracy = correct / total if total > 0 else 0.0
        iou_metrics[metrics_key]["accuracy"] = accuracy
    
    # 打印结果
    print(f"\n🎯 === 最终评估结果 ===")
    print(f"总样本数量: {final_metrics['total_samples']}")
    print(f"有框样本数量: {final_metrics['samples_with_bbox']}")
    print(f"所有IoU的平均值: {final_metrics['mean_iou_all']:.4f}")
    print(f"双方都有框时的平均IoU: {final_metrics['mean_iou_both_has_bbox']:.4f}")
    
    for iou_th in iou_thresholds:
        metrics_key = f"iou_{iou_th}"
        metrics = iou_metrics[metrics_key]
        print(f"\nIoU阈值 {iou_th}:")
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  正确样本: {metrics['correct_samples']}/{metrics['total_samples']}")
        print(f"  真阴性: {metrics['true_negatives']}")
        print(f"  假阳性: {metrics['false_positives']}")
        print(f"  假阴性: {metrics['false_negatives']}")
        print(f"  真阳性: {metrics['true_positives']}")
        print(f"  低IoU/F1: {metrics['low_iou_or_f1']}")
    
    return final_metrics

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='使用VLLM评估医疗标框模型 - 32B版本')
    parser.add_argument('--model_name', type=str, 
                       default='/data0/zhuoxu/yihong/code/Qwen2.5-VL-32B-Instruct', 
                       help='VLLM部署的模型名称')
    parser.add_argument('--test_file', type=str, 
                       default='/data0/zhuoxu/yihong/code/EasyR1-main/eval_data/test_grounding_balanced.json', 
                       help='测试文件路径')
    parser.add_argument('--api_url', type=str, 
                       default='http://127.0.0.1:8000/v1', 
                       help='VLLM API服务地址')
    parser.add_argument('--output_file', type=str, 
                       default='/data0/zhuoxu/yihong/code/LLaMA-Factory-main/eval_result/Qwen2.5-VL-32B-VLLM-Grounding.json', 
                       help='输出结果文件路径')
    parser.add_argument('--max_samples', type=int, help='最大测试样本数量')
    
    args = parser.parse_args()
    
    # 运行评估
    metrics, results = evaluate_model_vllm(
        model_name=args.model_name,
        test_file=args.test_file,
        api_url=args.api_url,
        output_file=args.output_file,
        max_samples=args.max_samples
    )
    
    print(f"\n🏆 评估完成！主要指标:")
    print(f"平均IoU: {metrics['mean_iou_all']:.4f}")
    print(f"IoU@0.5准确率: {metrics['iou_metrics']['iou_0.5']['accuracy']:.4f}")
    print(f"\n结果已保存到: {args.output_file}")

if __name__ == "__main__":
    # 配置参数
    TEST_FILE = "/data0/zhuoxu/yihong/code/EasyR1-main/eval_data/test_grounding_balanced.json"
    MODEL_NAME = "/data0/zhuoxu/yihong/code/Qwen2.5-VL-32B-Instruct"  # 根据实际部署的模型名称调整
    API_URL = "http://127.0.0.1:8000/v1"  # VLLM服务地址
    OUTPUT_FILE = "/data0/zhuoxu/yihong/code/LLaMA-Factory-main/eval_result/Qwen2.5-VL-32B-VLLM-Grounding.json"
    
    # 运行评估
    main()