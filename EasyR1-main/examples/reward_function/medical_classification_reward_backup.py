#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from typing import List, Dict
import numpy as np
from collections import Counter
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings

# 抑制sklearn的警告
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.metrics._ranking')

# 定义所有可能的标签
LABELS = [
    "enlarged cardiomediastinum",
    "cardiomegaly",
    "lung opacity",
    "edema",
    "consolidation",
    "atelectasis",
    "pleural effusion",
    "support devices",
    "pneumonia",
    "lung lesion",
    "pneumothorax",
    "pleural other",
    "fracture",
    "no finding"
]

def format_reward(predict: str) -> float:
    """
    评估生成文本是否符合预定格式
    Args:
        predict: 模型生成的文本
    Returns:
        float: 格式评分 (0 或 1)
    """
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, predict)
    return 1.0 if format_match else 0.0

def repetition_penalty(predict: str) -> float:
    """
    计算重复内容的惩罚分数
    Args:
        predict: 模型生成的文本
    Returns:
        float: 重复惩罚分数 (0.0-1.0，1.0表示无重复，0.0表示严重重复)
    """
    try:
        # 检测<think>标签重复
        think_pattern = r"<think>(.*?)</think>"
        think_matches = re.findall(think_pattern, predict, re.DOTALL)
        
        if len(think_matches) > 1:
            # 如果有多个<think>标签，严重惩罚
            return 0.1
        
        # 检测行级重复
        lines = predict.split('\n')
        if len(lines) < 5:  # 对于短文本，不进行重复检测
            return 1.0
        
        # 统计连续重复的行
        consecutive_repeats = 0
        max_consecutive = 0
        
        for i in range(1, len(lines)):
            if lines[i].strip() == lines[i-1].strip() and len(lines[i].strip()) > 5:
                consecutive_repeats += 1
                max_consecutive = max(max_consecutive, consecutive_repeats)
            else:
                consecutive_repeats = 0
        
        # 如果有超过2行连续重复，进行惩罚
        if max_consecutive > 2:
            penalty = max(0.1, 1.0 - (max_consecutive - 2) * 0.3)
            return penalty
        
        # 检测内容块重复（如整个句子或段落的重复）
        # 提取所有非空行
        non_empty_lines = [line.strip() for line in lines if len(line.strip()) > 10]
        
        if len(non_empty_lines) > 0:
            # 计算重复行的比例
            unique_lines = set(non_empty_lines)
            repetition_ratio = 1.0 - len(unique_lines) / len(non_empty_lines)
            
            # 如果重复率超过30%，进行惩罚
            if repetition_ratio > 0.3:
                penalty = max(0.2, 1.0 - repetition_ratio * 2)
                return penalty
        
        # 检测短语重复（检测重复的n-gram）
        words = predict.lower().split()
        if len(words) > 20:
            # 检测4-gram重复
            ngrams = [' '.join(words[i:i+4]) for i in range(len(words)-3)]
            ngram_counts = {}
            for ngram in ngrams:
                ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
            
            # 计算重复4-gram的比例
            repeated_ngrams = sum(1 for count in ngram_counts.values() if count > 1)
            if repeated_ngrams > len(ngrams) * 0.2:  # 如果超过20%的4-gram重复
                penalty = max(0.3, 1.0 - repeated_ngrams / len(ngrams))
                return penalty
        
        return 1.0  # 无重复
        
    except Exception as e:
        # 如果检测出错，给一个中等分数
        return 0.8

def extract_labels(text: str) -> List[str]:
    """
    从文本中提取标签 - 增强版
    Args:
        text: 输入文本
    Returns:
        List[str]: 提取的标签列表
    """
    # 移除可能的前缀文本
    if "The label of this X-ray image is:" in text:
        text = text.split("The label of this X-ray image is:")[-1]
    
    # 处理特殊情况
    if "No significant abnormalities were found in this X-ray." in text or "no finding" in text.lower():
        return ["no finding"]
    
    # 分割并清理标签
    labels = []
    
    # 按逗号分割
    parts = text.split(",")
    for part in parts:
        label = part.strip().lower()
        # 移除可能的标点符号
        label = re.sub(r'[^\w\s]', '', label).strip()
        
        # 检查是否在预定义标签中
        if label in LABELS:
            labels.append(label)
        else:
            # 模糊匹配
            for predefined_label in LABELS:
                if label in predefined_label or predefined_label in label:
                    if predefined_label not in labels:  # 避免重复
                        labels.append(predefined_label)
                    break
    
    # 如果没有提取到任何标签，尝试更宽松的匹配
    if not labels:
        text_lower = text.lower()
        for label in LABELS:
            if label in text_lower:
                labels.append(label)
    
    return labels

def convert_to_binary(labels: List[str]) -> List[int]:
    """
    将标签转换为二进制向量
    Args:
        labels: 标签列表
    Returns:
        List[int]: 二进制标签向量
    """
    return [1 if label in labels else 0 for label in LABELS]

def calculate_auc_score(y_true_batch: List[List[int]], y_pred_batch: List[List[int]], debug: bool = False) -> Dict[str, float]:
    """
    计算批次的AUC分数 - 改进版本，处理无正样本的情况
    Args:
        y_true_batch: 真实标签的二进制向量列表
        y_pred_batch: 预测标签的二进制向量列表
        debug: 是否打印调试信息
    Returns:
        Dict[str, float]: AUC相关指标
    """
    if len(y_true_batch) == 0:
        return {'macro_auc': 0.0, 'micro_auc': 0.0, 'average_precision': 0.0, 'valid_labels': 0}
    
    y_true = np.array(y_true_batch)
    y_pred = np.array(y_pred_batch)
    
    # 对于二进制预测，我们需要将其转换为概率
    # 使用更合理的概率映射：预测为1的标签给0.8的概率，预测为0的标签给0.2的概率
    y_pred_proba = y_pred * 0.6 + 0.2  # 将0,1映射到0.2,0.8
    
    try:
        # 计算macro AUC (每个标签单独计算AUC然后平均)
        macro_auc_scores = []
        valid_labels = []
        
        for i in range(len(LABELS)):
            y_true_label = y_true[:, i]
            y_pred_label = y_pred_proba[:, i]
            
            # 检查是否有正样本和负样本
            unique_true = np.unique(y_true_label)
            if len(unique_true) > 1:  # 既有0又有1
                try:
                    auc = roc_auc_score(y_true_label, y_pred_label)
                    macro_auc_scores.append(auc)
                    valid_labels.append(i)
                    if debug:
                        print(f"[DEBUG] 标签 '{LABELS[i]}' AUC: {auc:.3f}")
                except Exception as e:
                    if debug:
                        print(f"[DEBUG] 标签 '{LABELS[i]}' AUC计算失败: {str(e)}")
            else:
                # 对于只有一个类别的标签，我们给一个基准分数
                if unique_true[0] == 1:  # 全是正样本
                    # 如果预测也全是1，给高分；否则给低分
                    if np.all(y_pred_label >= 0.5):
                        base_score = 0.8
                    else:
                        base_score = 0.3
                else:  # 全是负样本
                    # 如果预测也全是0，给高分；否则给低分
                    if np.all(y_pred_label < 0.5):
                        base_score = 0.8
                    else:
                        base_score = 0.3
                
                macro_auc_scores.append(base_score)
                valid_labels.append(i)
                if debug:
                    print(f"[DEBUG] 标签 '{LABELS[i]}' 单一类别，基准分数: {base_score:.3f}")
        
        macro_auc = np.mean(macro_auc_scores) if macro_auc_scores else 0.0
        
        # 计算micro AUC (将所有标签合并后计算AUC)
        try:
            # 检查是否有足够的变化来计算micro AUC
            y_true_flat = y_true.ravel()
            y_pred_flat = y_pred_proba.ravel()
            
            if len(np.unique(y_true_flat)) > 1:
                micro_auc = roc_auc_score(y_true_flat, y_pred_flat)
            else:
                # 如果所有真实标签都相同，使用准确率作为替代
                micro_auc = np.mean(y_true_flat == (y_pred_flat >= 0.5))
        except Exception as e:
            if debug:
                print(f"[DEBUG] Micro AUC计算失败: {str(e)}")
            micro_auc = 0.0
        
        # 计算平均精度分数
        try:
            avg_precision = average_precision_score(y_true, y_pred_proba, average='macro')
        except Exception as e:
            if debug:
                print(f"[DEBUG] Average Precision计算失败: {str(e)}")
            avg_precision = 0.0
        
        if debug:
            print(f"[DEBUG] Macro AUC: {macro_auc:.3f}, Micro AUC: {micro_auc:.3f}, Avg Precision: {avg_precision:.3f}")
            print(f"[DEBUG] 有效标签数: {len(valid_labels)}/{len(LABELS)}")
        
        return {
            'macro_auc': macro_auc,
            'micro_auc': micro_auc,
            'average_precision': avg_precision,
            'valid_labels': len(valid_labels)
        }
        
    except Exception as e:
        if debug:
            print(f"[DEBUG] AUC计算出错: {str(e)}")
        return {'macro_auc': 0.0, 'micro_auc': 0.0, 'average_precision': 0.0, 'valid_labels': 0}

def calculate_basic_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    计算基本的分类指标
    Args:
        y_true: 真实标签的二进制向量
        y_pred: 预测标签的二进制向量
    Returns:
        Dict[str, float]: 基本分类指标
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Hamming准确率 (每个标签位置的准确率)
    hamming_accuracy = np.mean(y_true == y_pred)
    
    # F1分数
    tp = np.sum(y_true & y_pred)
    fp = np.sum((1 - y_true) & y_pred)
    fn = np.sum(y_true & (1 - y_pred))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'hamming_accuracy': hamming_accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def classification_reward(predict: str, ground_truth: str, debug: bool = False) -> float:
    """
    计算基于单个样本的分类奖励
    Args:
        predict: 模型生成的文本
        ground_truth: 参考答案文本
        debug: 是否打印调试信息
    Returns:
        float: 分类奖励分数
    """
    try:
        # 首先检测重复内容
        repetition_score = repetition_penalty(predict)
        
        if debug:
            print(f"[DEBUG] 重复惩罚分数: {repetition_score:.3f}")
        
        # 如果重复严重，直接返回低分
        if repetition_score < 0.3:
            if debug:
                print(f"[DEBUG] 检测到严重重复，返回低分: {repetition_score * 0.2:.3f}")
            return repetition_score * 0.2
        
        # 从预测文本中提取<answer>标签内容
        pred_match = re.search(r"<answer>(.*?)</answer>", predict, re.DOTALL)
        if not pred_match:
            if debug:
                print(f"[DEBUG] 格式错误：未找到<answer>标签")
                print(f"[DEBUG] 预测文本前100字符: {predict[:100]}...")
            return 0.1 * repetition_score  # 格式错误也要考虑重复惩罚
        
        pred_text = pred_match.group(1).strip()
        pred_labels = extract_labels(pred_text)
        true_labels = extract_labels(ground_truth)
        
        if debug:
            print(f"[DEBUG] 预测文本: {pred_text}")
            print(f"[DEBUG] 预测标签: {pred_labels}")
            print(f"[DEBUG] 真实标签: {true_labels}")
        
        # 转换为二进制向量
        pred_binary = convert_to_binary(pred_labels)
        true_binary = convert_to_binary(true_labels)
        
        if debug:
            print(f"[DEBUG] 预测二进制: {pred_binary}")
            print(f"[DEBUG] 真实二进制: {true_binary}")
        
        # 计算基本指标
        metrics = calculate_basic_metrics(true_binary, pred_binary)
        
        if debug:
            print(f"[DEBUG] 基本指标: hamming={metrics['hamming_accuracy']:.3f}, f1={metrics['f1']:.3f}, precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}")
        
        # 单个样本的奖励主要基于F1和Hamming准确率
        base_classification_score = 0.6 * metrics['f1'] + 0.4 * metrics['hamming_accuracy']
        
        # 应用重复惩罚
        final_classification_score = base_classification_score * repetition_score
        
        if debug:
            print(f"[DEBUG] 基础分类分数: {base_classification_score:.3f}")
            print(f"[DEBUG] 最终分类分数 (含重复惩罚): {final_classification_score:.3f}")
        
        return final_classification_score
        
    except Exception as e:
        print(f"[ERROR] 计算分类奖励时出错: {str(e)}")
        if debug:
            print(f"[DEBUG] 预测文本: {predict}")
            print(f"[DEBUG] 真实文本: {ground_truth}")
        return 0.1

def compute_score(predicts: List[str], ground_truths: List[str], format_weight: float = 0.1) -> List[Dict[str, float]]:
    """
    计算基于AUC的综合评分 - 增强AUC权重
    Args:
        predicts: 模型生成的文本列表
        ground_truths: 参考答案列表
        format_weight: 格式评分权重（降低到0.1）
    Returns:
        List[Dict[str, float]]: 评分结果列表
    """
    scores = []
    
    # 前3个样本打印详细调试信息
    debug_indices = list(range(min(3, len(predicts))))
    
    print(f"\n=== 开始计算 {len(predicts)} 个样本的AUC奖励分数 ===")
    
    # 收集所有样本的预测和真实标签用于批次AUC计算
    all_pred_binary = []
    all_true_binary = []
    
    for i, (predict, ground_truth) in enumerate(zip(predicts, ground_truths)):
        debug = i in debug_indices
        
        if debug:
            print(f"\n--- 样本 {i+1} 详细调试信息 ---")
        
        # 计算格式评分
        base_format_score = format_reward(predict)
        repetition_score = repetition_penalty(predict)
        format_score = base_format_score * repetition_score  # 格式分数也要考虑重复惩罚
        
        # 计算单个样本的分类分数（已包含重复惩罚）
        classification_score = classification_reward(predict, ground_truth, debug=debug)
        
        # 提取标签用于批次AUC计算
        try:
            pred_match = re.search(r"<answer>(.*?)</answer>", predict, re.DOTALL)
            if pred_match:
                pred_text = pred_match.group(1).strip()
                pred_labels = extract_labels(pred_text)
                true_labels = extract_labels(ground_truth)
                
                pred_binary = convert_to_binary(pred_labels)
                true_binary = convert_to_binary(true_labels)
                
                all_pred_binary.append(pred_binary)
                all_true_binary.append(true_binary)
        except:
            # 如果提取失败，使用全0向量
            all_pred_binary.append([0] * len(LABELS))
            all_true_binary.append([0] * len(LABELS))
        
        # 暂时使用单样本分数，后面会用批次AUC调整
        overall_score = format_weight * format_score + (1 - format_weight) * classification_score
        
        if debug:
            print(f"[DEBUG] 基础格式分数: {base_format_score:.3f}")
            print(f"[DEBUG] 重复惩罚分数: {repetition_score:.3f}")
            print(f"[DEBUG] 最终格式分数: {format_score:.3f}")
            print(f"[DEBUG] 单样本分类分数: {classification_score:.3f}")
            print(f"[DEBUG] 初始综合分数: {overall_score:.3f}")
            print("-" * 50)
        
        scores.append({
            "overall": overall_score,
            "format": format_score,
            "classification": classification_score,
            "base_format": base_format_score,
            "repetition_penalty": repetition_score
        })
    
    # 计算批次AUC分数
    if len(all_true_binary) > 1:
        auc_metrics = calculate_auc_score(all_true_binary, all_pred_binary, debug=True)
        batch_auc_score = auc_metrics['macro_auc']
        
        print(f"\n=== 批次AUC评估 ===")
        print(f"Macro AUC: {auc_metrics['macro_auc']:.3f}")
        print(f"Micro AUC: {auc_metrics['micro_auc']:.3f}")
        print(f"Average Precision: {auc_metrics['average_precision']:.3f}")
        print(f"有效标签数: {auc_metrics['valid_labels']}/{len(LABELS)}")
        
        # 大幅增加AUC权重，使其成为主要奖励
        auc_weight = 0.7  # AUC权重提高到70%
        classification_weight = 0.2  # 基础分类权重降低到20%
        
        for i, score in enumerate(scores):
            # 重新计算综合分数，以AUC为主导
            auc_based_score = (auc_weight * batch_auc_score + 
                             classification_weight * score['classification'])
            
            score['classification'] = auc_based_score
            score['overall'] = format_weight * score['format'] + (1 - format_weight) * auc_based_score
            score['batch_auc'] = batch_auc_score
            score['macro_auc'] = auc_metrics['macro_auc']
            score['micro_auc'] = auc_metrics['micro_auc']
            score['avg_precision'] = auc_metrics['average_precision']
        
        print(f"AUC权重: {auc_weight:.1%}, 基础分类权重: {classification_weight:.1%}")
        print(f"批次AUC分数: {batch_auc_score:.3f}")
    else:
        batch_auc_score = 0.0
        for score in scores:
            score['batch_auc'] = 0.0
            score['macro_auc'] = 0.0
            score['micro_auc'] = 0.0
            score['avg_precision'] = 0.0
    
    # 计算最终统计信息
    overall_scores = [s["overall"] for s in scores]
    classification_scores = [s["classification"] for s in scores]
    format_scores = [s["format"] for s in scores]
    repetition_scores = [s["repetition_penalty"] for s in scores]
    
    print(f"\n=== 最终批次统计信息 ===")
    print(f"样本数量: {len(scores)}")
    print(f"批次AUC (主要指标): {batch_auc_score:.3f}")
    print(f"重复惩罚 - 平均: {np.mean(repetition_scores):.3f}, 标准差: {np.std(repetition_scores):.3f}, 范围: [{np.min(repetition_scores):.3f}, {np.max(repetition_scores):.3f}]")
    print(f"综合分数 - 平均: {np.mean(overall_scores):.3f}, 标准差: {np.std(overall_scores):.3f}, 范围: [{np.min(overall_scores):.3f}, {np.max(overall_scores):.3f}]")
    print(f"分类分数 - 平均: {np.mean(classification_scores):.3f}, 标准差: {np.std(classification_scores):.3f}, 范围: [{np.min(classification_scores):.3f}, {np.max(classification_scores):.3f}]")
    print(f"格式分数 - 平均: {np.mean(format_scores):.3f}, 标准差: {np.std(format_scores):.3f}, 范围: [{np.min(format_scores):.3f}, {np.max(format_scores):.3f}]")
    print("=" * 50)
    
    return scores