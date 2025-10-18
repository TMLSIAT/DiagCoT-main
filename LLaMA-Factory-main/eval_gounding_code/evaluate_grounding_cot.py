#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
改进的X-ray Grounding任务评估脚本
正确处理相对坐标格式和两种情况的评判标准
支持从<answer></answer>标签中提取答案
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
import json
import re
import argparse
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# 添加LLaMA-Factory路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def extract_answer_from_tags(text: str) -> str:
    """
    从文本中提取<answer></answer>标签内的内容
    如果没有找到标签，返回原始文本
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
    返回: [(x1, y1, x2, y2), ...] 相对坐标格式
    """
    # 首先提取answer标签内的内容
    answer_text = extract_answer_from_tags(text)
    print("answer_text:",answer_text)
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

def calculate_iou(bbox1: Tuple[float, float, float, float], 
                  bbox2: Tuple[float, float, float, float]) -> float:
    """
    计算两个bbox的IoU
    bbox格式: (x1, y1, x2, y2) 相对坐标
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
    
    # 兜底返回，修复linter错误
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

def parse_prediction_text_improved(text: str) -> Tuple[List[Tuple[float, float, float, float]], bool]:
    """
    改进的预测文本解析函数
    支持从<answer></answer>标签中提取答案
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

def load_model_and_tokenizer(model_path: str):
    """加载模型和分词器"""
    print(f"加载模型: {model_path}")
    
    try:
        from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
        from qwen_vl_utils import process_vision_info
        
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            trust_remote_code=True
        )
        processor = Qwen2VLProcessor.from_pretrained(model_path, trust_remote_code=True)
        return model, processor
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保已安装transformers和qwen_vl_utils库")
        return None, None

def generate_grounding_prediction(model, processor, image_path: str) -> str:
    """生成grounding预测结果"""
    if model is None or processor is None:
        return ""
        
    try:
        # 构建grounding任务的prompt
        prompt = "<image>Please detect and locate any lung opacity regions in this chest X-ray image.\n\nDetection Guidelines:\n- Look for areas of increased density in the lung fields\n- Consider consolidation, infiltrates, or other opacity patterns\n- Ensure coordinates are within image boundaries\n- Provide precise (x1,y1),(x2,y2) coordinates\n\nOutput Format:\n\"The detected lung opacity regions are: Lung Opacity: <box>(x1,y1),(x2,y2)</box>\"\n\nIf no opacity is detected, output: \"The detected lung opacity regions are: No lung opacity regions detected.\""
        
        image = Image.open(image_path).convert('RGB')
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        try:
            from qwen_vl_utils import process_vision_info
            vision_output = process_vision_info(messages)
            if isinstance(vision_output, tuple) and len(vision_output) >= 2:
                image_inputs, video_inputs = vision_output[0], vision_output[1]
            else:
                image_inputs, video_inputs = vision_output, None
        except ImportError:
            print("警告: 无法导入qwen_vl_utils，跳过视觉信息处理")
            return ""
            
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=4096,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return response.strip()
        
    except Exception as e:
        print(f"生成预测时出错: {e}")
        return ""

def run_inference_improved(model_path: str, test_data: List[Dict[str, Any]], 
                          output_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    改进的推理函数
    """
    print("开始加载模型进行推理...")
    
    # 加载模型
    model, processor = load_model_and_tokenizer(model_path)
    
    if model is None or processor is None:
        print("模型加载失败，跳过推理")
        return []
    
    # 运行推理
    results = []
    for i, item in enumerate(tqdm(test_data, desc="推理进度")):
        image_path = item["image"]
        
        # 检查图片是否存在
        if not os.path.exists(image_path):
            print(f"图片不存在: {image_path}")
            continue
        
        # 生成预测
        prediction = generate_grounding_prediction(model, processor, image_path)
        print("prediction:",prediction)
        
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
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    return results

def evaluate_results_improved(results: List[Dict[str, Any]], 
                             iou_thresholds: List[float] = [0.3, 0.5, 0.7]) -> Dict[str, Any]:
    """
    改进的评估函数
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
    
    # 高IoU样本收集（用于论文展示）
    high_iou_samples = []
    perfect_samples = []  # IoU > 0.9的样本
    good_samples = []     # IoU > 0.7的样本
    medium_samples = []   # IoU > 0.5的样本
    
    # 详细结果
    detailed_results = []
    
    # 用于实时统计mean_iou的变量
    all_ious = []
    samples_with_bbox = 0
    
    for i, result in enumerate(results):
        # 解析预测结果
        pred_text = result["prediction"]
        pred_bboxes_relative, pred_has_abnormality = parse_prediction_text_improved(pred_text)
        
        # 获取真实标注
        gt_bboxes_relative = result["gt_bboxes_relative"]
        gt_bboxes = [bbox for bbox in gt_bboxes_relative] # 直接使用相对坐标
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
        
        # 收集高IoU样本用于论文展示
        if gt_has_abnormality and pred_has_abnormality and len(gt_bboxes) > 0 and len(pred_bboxes_relative) > 0:
            if max_sample_iou > 0.9:
                perfect_samples.append({
                    "sample_id": len(detailed_results),
                    "image_path": result["image"],
                    "max_iou": max_sample_iou,
                    "ground_truth": {
                        "has_abnormality": gt_has_abnormality,
                        "bboxes": gt_bboxes,
                        "raw_annotation": result.get("raw_annotation", ""),
                        "extracted_answer": result.get("extracted_answer", "")
                    },
                    "prediction": {
                        "has_abnormality": pred_has_abnormality,
                        "bboxes": pred_bboxes_relative,
                        "raw_prediction": pred_text,
                        "extracted_prediction_answer": result.get("extracted_prediction_answer", "")
                    }
                })
            elif max_sample_iou > 0.7:
                good_samples.append({
                    "sample_id": len(detailed_results),
                    "image_path": result["image"],
                    "max_iou": max_sample_iou,
                    "ground_truth": {
                        "has_abnormality": gt_has_abnormality,
                        "bboxes": gt_bboxes,
                        "raw_annotation": result.get("raw_annotation", ""),
                        "extracted_answer": result.get("extracted_answer", "")
                    },
                    "prediction": {
                        "has_abnormality": pred_has_abnormality,
                        "bboxes": pred_bboxes_relative,
                        "raw_prediction": pred_text,
                        "extracted_prediction_answer": result.get("extracted_prediction_answer", "")
                    }
                })
            elif max_sample_iou > 0.5:
                medium_samples.append({
                    "sample_id": len(detailed_results),
                    "image_path": result["image"],
                    "max_iou": max_sample_iou,
                    "ground_truth": {
                        "has_abnormality": gt_has_abnormality,
                        "bboxes": gt_bboxes,
                        "raw_annotation": result.get("raw_annotation", ""),
                        "extracted_answer": result.get("extracted_answer", "")
                    },
                    "prediction": {
                        "has_abnormality": pred_has_abnormality,
                        "bboxes": pred_bboxes_relative,
                        "raw_prediction": pred_text,
                        "extracted_prediction_answer": result.get("extracted_prediction_answer", "")
                    }
                })
        
        # 对每个IoU阈值进行评估
        sample_results = {}
        for iou_th in iou_thresholds:
            eval_result = evaluate_detection_improved(
                pred_bboxes_relative, gt_bboxes, 
                pred_has_abnormality, gt_has_abnormality, 
                iou_th
            )
            
            # 更新统计
            iou_metrics[f"iou_{iou_th}"]["total_samples"] += 1
            if eval_result["correct"]:
                iou_metrics[f"iou_{iou_th}"]["correct_samples"] += 1
            
            # 按情况统计
            case = eval_result["case"]
            if case == "true_negative":
                iou_metrics[f"iou_{iou_th}"]["true_negatives"] += 1
            elif case == "false_positive":
                iou_metrics[f"iou_{iou_th}"]["false_positives"] += 1
            elif case == "false_negative":
                iou_metrics[f"iou_{iou_th}"]["false_negatives"] += 1
            elif case == "true_positive":
                iou_metrics[f"iou_{iou_th}"]["true_positives"] += 1
            elif case == "low_iou_or_f1":
                iou_metrics[f"iou_{iou_th}"]["low_iou_or_f1"] += 1
            
            sample_results[f"iou_{iou_th}"] = eval_result
        
        # 保存详细结果
        detailed_result = {
            "image_path": result["image"],
            "ground_truth": {
                "has_abnormality": gt_has_abnormality,
                "bboxes": gt_bboxes,
                "bboxes_relative": gt_bboxes_relative,
                "raw_annotation": result.get("raw_annotation", ""),
                "extracted_answer": result.get("extracted_answer", "")
            },
            "prediction": {
                "has_abnormality": pred_has_abnormality,
                "bboxes": pred_bboxes_relative,
                "bboxes_relative": pred_bboxes_relative,
                "raw_prediction": pred_text,
                "extracted_prediction_answer": result.get("extracted_prediction_answer", "")
            },
            "evaluation": sample_results,
            "max_sample_iou": max_sample_iou  # 添加该样本的最大IoU
        }
        detailed_results.append(detailed_result)
    
    # 计算最终指标
    final_metrics = {}
    
    # 添加高IoU样本统计（用于论文展示）
    final_metrics["high_iou_samples"] = {
        "perfect_samples": {
            "count": len(perfect_samples),
            "samples": perfect_samples[:10]  # 只保存前10个完美样本
        },
        "good_samples": {
            "count": len(good_samples),
            "samples": good_samples[:15]  # 保存前15个好样本
        },
        "medium_samples": {
            "count": len(medium_samples),
            "samples": medium_samples[:20]  # 保存前20个中等样本
        }
    }
    
    # 添加"真实有框且预测有框"情况的IoU统计
    if both_has_bbox_ious:
        final_metrics["both_has_bbox_iou_statistics"] = {
            "mean_iou": np.mean(both_has_bbox_ious),
            "max_iou": np.max(both_has_bbox_ious),
            "min_iou": np.min(both_has_bbox_ious),
            "median_iou": np.median(both_has_bbox_ious),
            "std_iou": np.std(both_has_bbox_ious),
            "total_iou_pairs": len(both_has_bbox_ious),
            "samples_with_both_bbox": len([r for r in detailed_results if r["ground_truth"]["has_abnormality"] and r["prediction"]["has_abnormality"] and len(r["ground_truth"]["bboxes"]) > 0 and len(r["prediction"]["bboxes"]) > 0])
        }
    else:
        final_metrics["both_has_bbox_iou_statistics"] = {
            "mean_iou": 0.0,
            "max_iou": 0.0,
            "min_iou": 0.0,
            "median_iou": 0.0,
            "std_iou": 0.0,
            "total_iou_pairs": 0,
            "samples_with_both_bbox": 0
        }
    
    # 添加全局IoU统计（包括所有有框样本的IoU）
    if all_ious:
        final_metrics["global_iou_statistics"] = {
            "mean_iou": np.mean(all_ious),
            "max_iou": np.max(all_ious),
            "min_iou": np.min(all_ious),
            "median_iou": np.median(all_ious),
            "std_iou": np.std(all_ious),
            "total_iou_values": len(all_ious),
            "samples_with_bbox": samples_with_bbox,
            "total_samples": len(results)
        }
    else:
        final_metrics["global_iou_statistics"] = {
            "mean_iou": 0.0,
            "max_iou": 0.0,
            "min_iou": 0.0,
            "median_iou": 0.0,
            "std_iou": 0.0,
            "total_iou_values": 0,
            "samples_with_bbox": 0,
            "total_samples": len(results)
        }
    
    for iou_th in iou_thresholds:
        key = f"iou_threshold_{iou_th}"
        metrics = iou_metrics[f"iou_{iou_th}"]
        
        total = metrics["total_samples"]
        correct = metrics["correct_samples"]
        accuracy = correct / total if total > 0 else 0.0
        
        # 计算精确率、召回率、F1
        tp = metrics["true_positives"]
        fp = metrics["false_positives"]
        fn = metrics["false_negatives"]
        tn = metrics["true_negatives"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        final_metrics[key] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "low_iou_or_f1": metrics["low_iou_or_f1"],
            "total_samples": total,
            "correct_samples": correct
        }
    
    final_metrics["detailed_results"] = detailed_results
    
    return final_metrics

def main():
    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    
    parser = argparse.ArgumentParser(description="改进的X-ray Grounding任务评估")
    parser.add_argument("--model_path", type=str, 
                       default="/data0/zhuoxu/yihong/code/LLaMA-Factory-main/saves/Qwen2-VL-7B-Instruct/full/train_grounding_merger_llm_use_cot_5e-6_ep3_stage2",
                       help="训练好的模型路径")
    parser.add_argument("--test_data", type=str, 
                       default="/data0/zhuoxu/yihong/code/EasyR1-main/eval_data/test_grounding_balanced.json",
                       help="测试数据文件路径")
    parser.add_argument("--output_dir", type=str, 
                       default="/data0/zhuoxu/yihong/code/LLaMA-Factory-main/task_grounding_result_test_balanced_improved_test_train_grounding_merger_llm_use_cot_5e-6_ep3_stage2",
                       help="评估结果输出目录")
    parser.add_argument("--iou_thresholds", type=float, nargs="+", 
                       default=[0.3, 0.5, 0.7],
                       help="IoU阈值列表")
    parser.add_argument("--max_samples", type=int, help="最大测试样本数量")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("改进的X-ray Grounding任务评估")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载测试数据
    print(f"加载测试数据: {args.test_data}")
    test_data = load_test_data_improved(args.test_data)
    
    if args.max_samples:
        test_data = test_data[:args.max_samples]
        print(f"限制测试样本数量为: {args.max_samples}")
    
    print(f"测试样本数量: {len(test_data)}")
    
    # 运行推理
    print(f"使用模型进行推理: {args.model_path}")
    inference_output = os.path.join(args.output_dir, "inference_results.json")
    results = run_inference_improved(args.model_path, test_data, inference_output)
    
    # 评估结果
    print("评估检测性能...")
    print("📊 将实时显示mean_iou统计信息（每10个样本更新一次）")
    metrics = evaluate_results_improved(results, args.iou_thresholds)
    
    # 打印结果
    print("\n🎯 === 改进的评估结果 ===")
    print("-" * 80)
    
    # 打印高IoU样本统计（用于论文展示）
    if "high_iou_samples" in metrics:
        high_iou_stats = metrics["high_iou_samples"]
        print(f"\n🏆 高IoU样本统计（用于论文展示）:")
        print(f"  完美样本 (IoU > 0.9): {high_iou_stats['perfect_samples']['count']} 个")
        print(f"  优秀样本 (IoU > 0.7): {high_iou_stats['good_samples']['count']} 个")
        print(f"  良好样本 (IoU > 0.5): {high_iou_stats['medium_samples']['count']} 个")
        
        # 展示一些高IoU样本的详细信息
        if high_iou_stats['perfect_samples']['samples']:
            print(f"\n🎯 完美样本示例 (IoU > 0.9):")
            for i, sample in enumerate(high_iou_stats['perfect_samples']['samples'][:3]):
                print(f"  样本 {i+1}: IoU = {sample['max_iou']:.4f}")
                print(f"    图片: {sample['image_path']}")
                print(f"    真实标注: {sample['ground_truth']['extracted_answer'][:100]}...")
                print(f"    预测提取: {sample['prediction'].get('extracted_prediction_answer', 'N/A')[:100]}...")
        
        if high_iou_stats['good_samples']['samples']:
            print(f"\n👍 优秀样本示例 (IoU > 0.7):")
            for i, sample in enumerate(high_iou_stats['good_samples']['samples'][:2]):
                print(f"  样本 {i+1}: IoU = {sample['max_iou']:.4f}")
                print(f"    图片: {sample['image_path']}")
                print(f"    真实标注: {sample['ground_truth']['extracted_answer'][:100]}...")
                print(f"    预测提取: {sample['prediction'].get('extracted_prediction_answer', 'N/A')[:100]}...")
    
    # 打印全局IoU统计（所有有框样本的IoU）
    if "global_iou_statistics" in metrics:
        global_iou_stats = metrics["global_iou_statistics"]
        print(f"\n🌍 全局IoU统计 (所有有框样本):")
        print(f"  总样本数: {global_iou_stats['total_samples']}")
        print(f"  有框样本数: {global_iou_stats['samples_with_bbox']}")
        print(f"  总IoU值数: {global_iou_stats['total_iou_values']}")
        print(f"  平均IoU: {global_iou_stats['mean_iou']:.4f}")
        print(f"  最大IoU: {global_iou_stats['max_iou']:.4f}")
        print(f"  最小IoU: {global_iou_stats['min_iou']:.4f}")
        print(f"  中位IoU: {global_iou_stats['median_iou']:.4f}")
        print(f"  IoU标准差: {global_iou_stats['std_iou']:.4f}")
    
    # 打印"真实有框且预测有框"情况的IoU统计
    if "both_has_bbox_iou_statistics" in metrics:
        bbox_iou_stats = metrics["both_has_bbox_iou_statistics"]
        print(f"\n📊 真实有框且预测有框情况的IoU统计:")
        print(f"  样本数: {bbox_iou_stats['samples_with_both_bbox']}")
        print(f"  平均IoU: {bbox_iou_stats['mean_iou']:.4f}")
        print(f"  最大IoU: {bbox_iou_stats['max_iou']:.4f}")
        print(f"  最小IoU: {bbox_iou_stats['min_iou']:.4f}")
        print(f"  中位IoU: {bbox_iou_stats['median_iou']:.4f}")
        print(f"  IoU标准差: {bbox_iou_stats['std_iou']:.4f}")
        print(f"  总IoU对数: {bbox_iou_stats['total_iou_pairs']}")
    
    # 按IoU阈值打印结果
    for iou_th in args.iou_thresholds:
        key = f"iou_threshold_{iou_th}"
        threshold_metrics = metrics[key]
        
        print(f"\n🔍 IoU阈值 {iou_th}:")
        print(f"  总样本数: {threshold_metrics['total_samples']}")
        print(f"  正确样本数: {threshold_metrics['correct_samples']}")
        print(f"  准确率: {threshold_metrics['accuracy']:.4f}")
        print(f"  精确率: {threshold_metrics['precision']:.4f}")
        print(f"  召回率: {threshold_metrics['recall']:.4f}")
        print(f"  F1分数: {threshold_metrics['f1']:.4f}")
        print(f"  真阳性(TP): {threshold_metrics['true_positives']}")
        print(f"  假阳性(FP): {threshold_metrics['false_positives']}")
        print(f"  真阴性(TN): {threshold_metrics['true_negatives']}")
        print(f"  假阴性(FN): {threshold_metrics['false_negatives']}")
        print(f"  低IoU/F1: {threshold_metrics['low_iou_or_f1']}")
    
    # 打印最终的mean_iou总结
    if "global_iou_statistics" in metrics:
        final_mean_iou = metrics["global_iou_statistics"]["mean_iou"]
        print(f"\n🎯 === 最终Mean IoU总结 ===")
        print(f"  最终平均IoU: {final_mean_iou:.4f}")
        print(f"  基于 {metrics['global_iou_statistics']['total_iou_values']} 个IoU值计算")
        print(f"  来自 {metrics['global_iou_statistics']['samples_with_bbox']} 个有框样本")
    
    # 保存评估结果
    metrics_file = os.path.join(args.output_dir, "evaluation_metrics.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    # 保存详细结果
    detailed_file = os.path.join(args.output_dir, "detailed_results.json")
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump(metrics["detailed_results"], f, ensure_ascii=False, indent=2)
    
    # 保存高IoU样本（用于论文展示）
    high_iou_file = os.path.join(args.output_dir, "high_iou_samples_for_paper.json")
    with open(high_iou_file, 'w', encoding='utf-8') as f:
        json.dump(metrics["high_iou_samples"], f, ensure_ascii=False, indent=2)
    
    # 生成错误案例分析
    error_analysis = analyze_errors(metrics["detailed_results"])
    error_file = os.path.join(args.output_dir, "error_analysis.json")
    with open(error_file, 'w', encoding='utf-8') as f:
        json.dump(error_analysis, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 评估完成！结果已保存到: {args.output_dir}")
    print(f"📊 推理结果: {inference_output}")
    print(f"📈 评估指标: {metrics_file}")
    print(f"📋 详细结果: {detailed_file}")
    print(f"🏆 高IoU样本: {high_iou_file}")
    print(f"❌ 错误分析: {error_file}")

def analyze_errors(detailed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    分析错误案例
    """
    error_analysis = {
        "false_positives": [],
        "false_negatives": [],
        "low_iou_cases": [],
        "bbox_mismatch": []
    }
    
    for i, result in enumerate(detailed_results):
        # 检查各种错误情况
        for iou_th in [0.3, 0.5, 0.7]:
            eval_key = f"iou_{iou_th}"
            if eval_key in result["evaluation"]:
                eval_result = result["evaluation"][eval_key]
                
                if eval_result["case"] == "false_positive":
                    error_analysis["false_positives"].append({
                        "sample_id": i,
                        "image_path": result["image_path"],
                        "ground_truth": result["ground_truth"],
                        "prediction": result["prediction"],
                        "iou_threshold": iou_th
                    })
                elif eval_result["case"] == "false_negative":
                    error_analysis["false_negatives"].append({
                        "sample_id": i,
                        "image_path": result["image_path"],
                        "ground_truth": result["ground_truth"],
                        "prediction": result["prediction"],
                        "iou_threshold": iou_th
                    })
                elif eval_result["case"] == "low_iou_or_f1":
                    error_analysis["low_iou_cases"].append({
                        "sample_id": i,
                        "image_path": result["image_path"],
                        "ground_truth": result["ground_truth"],
                        "prediction": result["prediction"],
                        "iou_threshold": iou_th,
                        "max_iou": eval_result.get("max_iou", 0.0),
                        "f1": eval_result.get("f1", 0.0)
                    })
                elif eval_result["case"] == "bbox_mismatch":
                    error_analysis["bbox_mismatch"].append({
                        "sample_id": i,
                        "image_path": result["image_path"],
                        "ground_truth": result["ground_truth"],
                        "prediction": result["prediction"],
                        "iou_threshold": iou_th
                    })
    
    # 统计错误数量 - 修复类型错误
    error_analysis["error_statistics"] = {
        "total_samples": len(detailed_results),
        "false_positives_count": len(error_analysis["false_positives"]),
        "false_negatives_count": len(error_analysis["false_negatives"]),
        "low_iou_cases_count": len(error_analysis["low_iou_cases"]),
        "bbox_mismatch_count": len(error_analysis["bbox_mismatch"])
    }
    
    return error_analysis

if __name__ == "__main__":
    main()