#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from scipy.optimize import linear_sum_assignment

# def format_reward(predict: str) -> float:
#     """评估生成文本是否符合预定格式 - 要求先think再answer"""
#     predict = predict.strip()
    
#     # 检查是否包含think和answer标签的完整格式
#     pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
#     if not re.fullmatch(pattern, predict):
#         return 0.0
    
#     # 提取answer部分，检查是否包含box坐标或正确的无病变格式
#     answer_match = re.search(r'<answer>(.*?)</answer>', predict, re.DOTALL)
#     if not answer_match:
#         return 0.0
    
#     answer_content = answer_match.group(1).strip()
    
#     # 检查answer中是否包含坐标格式
#     if '<box>' in answer_content and '</box>' in answer_content:
#         # 进一步检查坐标格式是否正确
#         box_pattern = r'<box>\(([0-9.]+),([0-9.]+)\),\(([0-9.]+),([0-9.]+)\)</box>'
#         if re.search(box_pattern, answer_content):
#             return 1.0
#         else:
#             return 0.7  # 有box标签但格式不正确
    
#     # # 检查是否包含"无病变"、"正常"等关键词
#     # no_lesion_keywords = ['no lesion', 'normal', 'no abnormality','No lung opacity regions detected']
#     # if any(keyword in answer_content.lower() for keyword in no_lesion_keywords):
#     #     return 1.0
    
#     return 0.5  # 格式正确但内容不够好

def format_reward(predict: str) -> float:
    """
    简化的格式奖励函数，检查 box 或无病变关键词。
    """
    predict = predict.strip()
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    if not re.fullmatch(pattern, predict):
        return 0.0

    # 提取answer部分，检查是否包含box坐标或正确的无病变格式
    answer_match = re.search(r'<answer>(.*?)</answer>', predict, re.DOTALL)
    if not answer_match:
        return 0.0
    answer_content = answer_match.group(1).strip()
    # 检查 box 格式
    box_pattern = r'<box>\(([0-9.]+),([0-9.]+)\),\(([0-9.]+),([0-9.]+)\)</box>'
    match = re.search(box_pattern, answer_content)

    if match:
        return 1.0  # 有框且格式正确

    # 检查无病变关键词
    no_lesion_keywords = ['no lesion', 'normal', 'no abnormality', 'no lung opacity regions detected']  #大小写问题! 那个N！！！！，导致格式化得分总是和iou的差异开来，一上一下的。
    predict_lower = answer_content.lower()
    if any(keyword in predict_lower for keyword in no_lesion_keywords):
        return 1.0  # 无框但正确识别为无病变

    # 有标签但格式不对
    if '<box>' in answer_content and '</box>' in answer_content:
        return 0.5

    # 无框且无关键词
    return 0.0


# def convert_normalized_to_absolute(bbox: Tuple[float, float, float, float], 
#                                  image_width: int = 1024, 
#                                  image_height: int = 1024) -> Tuple[float, float, float, float]:
#     """
#     将相对坐标转换为绝对坐标
#     输入: (x1, y1, x2, y2) 相对坐标 (0-1范围)
#     输出: (x, y, w, h) 绝对坐标
#     """
#     x1, y1, x2, y2 = bbox
    
#     # 转换为绝对坐标
#     x1_abs = x1 * image_width
#     y1_abs = y1 * image_height
#     x2_abs = x2 * image_width
#     y2_abs = y2 * image_height
    
#     # 转换为 (x, y, w, h) 格式
#     x = x1_abs
#     y = y1_abs
#     w = x2_abs - x1_abs
#     h = y2_abs - y1_abs
    
#     return (x, y, w, h)

def extract_bounding_boxes(text: str) -> List[Tuple[float, float, float, float]]:
    """
    从文本中提取边界框坐标
    Args:
        text: 输入文本
    Returns:
        List[Tuple[float, float, float, float]]: 边界框列表 [(x1, y1, x2, y2), ...] 相对坐标
    """
    boxes = []
    
    # 统一的坐标匹配格式，支持整数和小数
    box_pattern = r'<box>\(([0-9.]+),([0-9.]+)\),\(([0-9.]+),([0-9.]+)\)</box>'
    matches = re.findall(box_pattern, text)
    
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
            boxes.append((x1_rel, y1_rel, x2_rel, y2_rel))
    
    # 改进：如果没有找到坐标格式，检查是否包含无病变描述
    if not boxes:
        no_lesion_keywords = ['no lesion', 'normal', 'no abnormality', 'No lung opacity regions detected']
        text_lower = text.lower()
        
        # 如果包含无病变关键词，返回空列表（表示正确识别为无病变）
        if any(keyword in text_lower for keyword in no_lesion_keywords):
            return []
        
        # 如果包含病变描述但没有坐标，返回默认框（表示识别到病变但坐标不准确）
        lesion_keywords = ['lung opacity', 'detected', 'lesion', 'abnormality']
        if any(keyword in text_lower for keyword in lesion_keywords):
            return [(0.0, 0.0, 0.1, 0.1)]  # 极小的默认框
    
    return boxes

def calculate_iou_with_conversion(box1: Tuple[float, float, float, float], 
                                box2: Tuple[float, float, float, float],
                                image_width: int = 1024,
                                image_height: int = 1024) -> float:
    """
    计算两个边界框的IoU
    Args:
        box1: 第一个边界框 (x1, y1, x2, y2) - 相对坐标0-1
        box2: 第二个边界框 (x1, y1, x2, y2) - 相对坐标0-1
        image_width: 图像宽度（用于转换，但现在输入已经是相对坐标）
        image_height: 图像高度（用于转换，但现在输入已经是相对坐标）
    Returns:
        float: IoU值 (0-1)
    """
    # 输入的坐标已经是相对坐标，直接使用
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # 计算交集区域
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    # 如果没有交集
    if x1_inter >= x2_inter or y1_inter >= y2_inter:
        return 0.0
    
    # 计算交集面积
    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # 计算各自面积
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # 计算并集面积
    union = area1 + area2 - intersection
    
    # 避免除零
    if union <= 0:
        return 0.0
    
    return intersection / union

def match_boxes_hungarian(pred_boxes: List[Tuple[float, float, float, float]], 
                         true_boxes: List[Tuple[float, float, float, float]]) -> List[Tuple[int, int, float]]:
    """
    使用匈牙利算法进行边界框匹配
    Args:
        pred_boxes: 预测边界框列表
        true_boxes: 真实边界框列表
    Returns:
        List[Tuple[int, int, float]]: 匹配结果 [(pred_idx, true_idx, iou), ...]
    """
    if not pred_boxes or not true_boxes:
        return []
    
    # 构建IoU矩阵
    iou_matrix = np.zeros((len(pred_boxes), len(true_boxes)))
    
    for i, pred_box in enumerate(pred_boxes):
        for j, true_box in enumerate(true_boxes):
            iou_matrix[i, j] = calculate_iou_with_conversion(pred_box, true_box)
    
    # 使用匈牙利算法进行最优匹配
    # 由于linear_sum_assignment最小化成本，我们使用(1-IoU)作为成本
    cost_matrix = 1 - iou_matrix
    pred_indices, true_indices = linear_sum_assignment(cost_matrix)
    
    matches = []
    for pred_idx, true_idx in zip(pred_indices, true_indices):
        iou = iou_matrix[pred_idx, true_idx]
        if iou > 0:  # 只保留有交集的匹配
            matches.append((pred_idx, true_idx, iou))
    
    return matches

def calculate_detection_metrics(pred_boxes: List[Tuple[float, float, float, float]], 
                              true_boxes: List[Tuple[float, float, float, float]], 
                              iou_threshold: float = 0.5) -> Dict[str, Union[float, int]]:
    """
    计算检测评估指标
    Args:
        pred_boxes: 预测边界框列表
        true_boxes: 真实边界框列表
        iou_threshold: IoU阈值
    Returns:
        Dict[str, Union[float, int]]: 包含各种评估指标的字典
    """
    # 使用匈牙利算法进行最优匹配
    matches = match_boxes_hungarian(pred_boxes, true_boxes)
    
    # 计算真正例 (True Positives)
    tp = sum(1 for _, _, iou in matches if iou >= iou_threshold)
    
    # 计算假正例 (False Positives) 和假负例 (False Negatives)
    fp = len(pred_boxes) - tp
    fn = len(true_boxes) - tp
    
    # 计算精确率、召回率和F1分数
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 计算平均IoU
    avg_iou = np.mean([iou for _, _, iou in matches]) if matches else 0.0
    
    # 计算最大IoU
    max_iou = max([iou for _, _, iou in matches]) if matches else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'avg_iou': avg_iou,
        'max_iou': max_iou,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'num_matches': len(matches)
    }

def calculate_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """
    兼容性函数：计算两个边界框的IoU，自动处理坐标转换
    Args:
        box1: 第一个边界框 (x1, y1, x2, y2)
        box2: 第二个边界框 (x1, y1, x2, y2)
    Returns:
        float: IoU值 (0-1)
    """
    return calculate_iou_with_conversion(box1, box2)

def grounding_reward(predict: str, ground_truth: str) -> float:
    """
    计算grounding任务的奖励分数 - 改进版本，更平滑的奖励策略
    Args:
        predict: 模型预测结果
        ground_truth: 真实标签
    Returns:
        float: 奖励分数 (0-1)
    """
    try:
        # 首先尝试文本完全匹配（兜底机制）
        predict_clean = predict.strip()
        ground_truth_clean = ground_truth.strip()
        # print("predict_clean: ",predict_clean)
        # print("ground_truth_clean: ",ground_truth_clean)
        # 提取answer部分进行比较
        predict_match = re.search(r'<answer>(.*?)</answer>', predict_clean, re.DOTALL)
        if predict_match:
            predict_match = predict_match.group(1).strip()
        # predict_match = predict_clean
        gt_match = ground_truth_clean
        print("predict_match: ",predict_match)
        print("gt_match: ",gt_match)

        if predict_match and gt_match:
            predict_answer = predict_match.strip()
            gt_answer = gt_match.strip()
            
            # 如果答案完全匹配，给予满分
            if predict_answer == gt_answer:
                return 1.0
        
        # 提取预测和真实的边界框
        pred_boxes = extract_bounding_boxes(predict_match)
        true_boxes = extract_bounding_boxes(gt_match)
        
        # 过滤无效的边界框
        pred_boxes = [box for box in pred_boxes if box[2] > box[0] and box[3] > box[1]]
        true_boxes = [box for box in true_boxes if box[2] > box[0] and box[3] > box[1]]
        
        # 处理无检测结果的情况 - 改进的奖励策略
        if not true_boxes and not pred_boxes:
            # 都没有检测到，检查是否包含"无病变"等关键词
            no_lesion_keywords = ['no lesion', 'normal', 'no abnormality', 'No lung opacity regions detected']
            predict_lower = predict_match.lower()
            gt_lower = gt_match.lower()
            
            # 如果预测和真实都包含无病变关键词，给予高分
            pred_has_no_lesion = any(keyword in predict_lower for keyword in no_lesion_keywords)
            gt_has_no_lesion = any(keyword in gt_lower for keyword in no_lesion_keywords)
            
            if pred_has_no_lesion and gt_has_no_lesion:
                return 0.8  # 都正确识别为无病变
            elif gt_has_no_lesion and not pred_has_no_lesion:
                return 0.2  # 真实无病变但预测没说清楚
            elif not gt_has_no_lesion and pred_has_no_lesion:
                return 0.1  # 真实有病变但预测说无病变
            else:
                return 0.3  # 都没有明确说明，给予中等分数
        
        if not true_boxes and pred_boxes:
            # 真实无目标但预测有目标，给予很低分（假正例）
            return 0.05
        
        if true_boxes and not pred_boxes:
            # 真实有目标但预测无目标，给予很低分（假负例）
            return 0.0
        
        # 计算检测指标
        metrics = calculate_detection_metrics(pred_boxes, true_boxes)
        
        # 专注于IoU的奖励策略 - 改进版本，更平滑的奖励
        avg_iou = metrics['avg_iou']
        max_iou = metrics['max_iou']
        
        # 使用更平滑的IoU奖励策略，避免过于严格的阶梯式奖励
        if avg_iou >= 0.9:
            # 极高质量IoU，给予满分
            reward = 1.0
        elif avg_iou >= 0.7:
            # 高质量IoU，平滑过渡
            reward = 0.7 + 0.3 * (avg_iou - 0.7) / 0.2  # 0.7-1.0
        elif avg_iou >= 0.5:
            # 中等质量IoU，明显的奖励梯度
            reward = 0.4 + 0.3 * (avg_iou - 0.5) / 0.2  # 0.4-0.7
        elif avg_iou >= 0.3:
            # 低质量IoU，但仍给予合理奖励
            reward = 0.2 + 0.2 * (avg_iou - 0.3) / 0.2  # 0.2-0.4
        elif avg_iou >= 0.1:
            # 很低质量IoU，但至少有检测
            reward = 0.1 + 0.1 * (avg_iou - 0.1) / 0.2  # 0.1-0.2
        elif avg_iou > 0:
            # 极低质量IoU，但至少有重叠
            reward = 0.05 + 0.05 * avg_iou / 0.1  # 0.05-0.1
        else:
            # 无重叠，但至少尝试了检测
            reward = 0.02
        
        # 对于多目标检测，平衡平均IoU和最大IoU
        if len(true_boxes) > 1:
            # 多目标情况下，既要平均质量好，也要有高质量检测
            reward = 0.7 * reward + 0.3 * max_iou
        
        # 额外奖励：对于非常高的IoU给予额外激励
        if avg_iou > 0.8:
            reward = min(1.0, reward + 0.05)  # 额外5%奖励
        
        # 确保奖励在合理范围内
        reward = max(0.0, min(1.0, reward))
        
        return reward
        
    except Exception as e:
        print(f"计算grounding奖励时出错: {e}")
        print(f"预测: {predict[:100]}...")
        print(f"真实: {ground_truth[:100]}...")
        return 0.0

def analyze_iou_distribution(predicts: List[str], ground_truths: List[str]) -> Dict[str, Union[float, int]]:
    """
    分析IoU分布情况，用于调试和监控
    Args:
        predicts: 预测结果列表
        ground_truths: 真实标签列表
    Returns:
        Dict[str, Union[float, int]]: IoU分布统计
    """
    iou_values = []
    no_detection_count = 0
    false_positive_count = 0
    false_negative_count = 0
    
    for predict, ground_truth in zip(predicts, ground_truths):
        pred_boxes = extract_bounding_boxes(predict)
        true_boxes = extract_bounding_boxes(ground_truth)
        
        if not true_boxes and not pred_boxes:
            no_detection_count += 1
            continue
        elif not true_boxes and pred_boxes:
            false_positive_count += 1
            continue
        elif true_boxes and not pred_boxes:
            false_negative_count += 1
            continue
        else:
            # 计算IoU
            metrics = calculate_detection_metrics(pred_boxes, true_boxes)
            iou_values.append(metrics['avg_iou'])
    
    if iou_values:
        stats = {
            'avg_iou': np.mean(iou_values),
            'max_iou': np.max(iou_values),
            'min_iou': np.min(iou_values),
            'std_iou': np.std(iou_values),
            'median_iou': np.median(iou_values),
            'iou_above_0.5': sum(1 for iou in iou_values if iou > 0.5) / len(iou_values),
            'iou_above_0.7': sum(1 for iou in iou_values if iou > 0.7) / len(iou_values),
            'iou_above_0.8': sum(1 for iou in iou_values if iou > 0.8) / len(iou_values),
            'total_samples': len(predicts),
            'detection_samples': len(iou_values),
            'no_detection_rate': no_detection_count / len(predicts),
            'false_positive_rate': false_positive_count / len(predicts),
            'false_negative_rate': false_negative_count / len(predicts)
        }
    else:
        stats = {
            'avg_iou': 0.0,
            'max_iou': 0.0,
            'min_iou': 0.0,
            'std_iou': 0.0,
            'median_iou': 0.0,
            'iou_above_0.5': 0.0,
            'iou_above_0.7': 0.0,
            'iou_above_0.8': 0.0,
            'total_samples': len(predicts),
            'detection_samples': 0,
            'no_detection_rate': no_detection_count / len(predicts),
            'false_positive_rate': false_positive_count / len(predicts),
            'false_negative_rate': false_negative_count / len(predicts)
        }
    
    return stats

def compute_score(predicts: List[str], ground_truths: List[str], format_weight: float = 0.2) -> List[Dict[str, float]]:
    """
    批量计算奖励分数 - 改进版本，增加调试信息
    Args:
        predicts: 预测结果列表
        ground_truths: 真实标签列表
        format_weight: 格式奖励权重
    Returns:
        List[Dict[str, float]]: 奖励分数列表
    """
    results = []
    
    # 收集统计信息
    total_samples = len(predicts)
    format_scores = []
    iou_scores = []
    
    for i, (predict, ground_truth) in enumerate(zip(predicts, ground_truths)):
        # 计算格式奖励
        format_score = format_reward(predict)
        print("format_score: ",format_score)
        format_scores.append(format_score)
        
        # 计算IoU奖励
        iou_score = grounding_reward(predict, ground_truth)
        iou_scores.append(iou_score)
        
        # 综合奖励：格式20% + IoU 80%
        final_score = format_weight * format_score + (1 - format_weight) * iou_score
        
        results.append({
            'overall': final_score,
            'format_reward': format_score,
            'iou_reward': iou_score
        })
        
        # 每100个样本打印一次统计信息
        if (i + 1) % 100 == 0:
            avg_format = np.mean(format_scores)
            avg_iou = np.mean(iou_scores)
            print(f"样本 {i+1}/{total_samples}: 平均格式奖励={avg_format:.3f}, 平均IoU奖励={avg_iou:.3f}")
    
    # 最终统计
    if total_samples > 0:
        avg_format = np.mean(format_scores)
        avg_iou = np.mean(iou_scores)
        print(f"最终统计: 平均格式奖励={avg_format:.3f}, 平均IoU奖励={avg_iou:.3f}")
        print(f"IoU奖励分布: <0.1: {sum(1 for s in iou_scores if s < 0.1)/len(iou_scores):.1%}, "
              f"0.1-0.3: {sum(1 for s in iou_scores if 0.1 <= s < 0.3)/len(iou_scores):.1%}, "
              f"0.3-0.5: {sum(1 for s in iou_scores if 0.3 <= s < 0.5)/len(iou_scores):.1%}, "
              f">0.5: {sum(1 for s in iou_scores if s >= 0.5)/len(iou_scores):.1%}")
    
    return results 