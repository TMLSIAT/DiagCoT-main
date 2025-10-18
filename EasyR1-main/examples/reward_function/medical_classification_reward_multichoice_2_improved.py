#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import os
from typing import List, Dict, Optional
import numpy as np
from collections import Counter
import warnings
from datetime import datetime
import math

'''
改进版医疗分类奖励函数 - 解决奖励波动问题
主要改进：
1. 动态奖励调整机制
2. 更精细的评分策略
3. 训练阶段适应性
4. 改进的重复惩罚
5. 更好的奖励区分度
'''

# 抑制sklearn的警告
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.metrics._ranking')

# 简单的调试日志类
class ProcessLogger:
    def __init__(self, prefix=""):
        self.prefix = prefix
        self.log_file = f"medical_reward_{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    def log(self, message):
        if os.getenv("DEBUG_MODE") == "true":
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] {self.prefix}: {message}"
            print(log_entry)
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(log_entry + '\n')
            except:
                pass

# 创建日志实例
medical_logger = ProcessLogger("MEDICAL_REWARD_IMPROVED")

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

# 标签难度权重（基于医学复杂度和识别难度）
LABEL_DIFFICULTY_WEIGHTS = {
    "no finding": 0.3,  # 最简单，但重要
    "support devices": 0.6,
    "cardiomegaly": 0.8,
    "lung opacity": 0.9,
    "atelectasis": 1.0,
    "pleural effusion": 1.1,
    "edema": 1.2,
    "consolidation": 1.3,
    "enlarged cardiomediastinum": 1.4,
    "pneumonia": 1.5,
    "lung lesion": 1.6,
    "pneumothorax": 1.7,
    "pleural other": 1.8,
    "fracture": 2.0  # 最难识别
}

def format_reward(predict: str) -> float:
    """评估生成文本是否符合预定格式"""
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, predict)
    return 1.0 if format_match else 0.0

def improved_repetition_penalty(predict: str) -> float:
    """改进的重复内容惩罚 - 更加智能和宽松"""
    try:
        # 检测<think>标签重复
        think_pattern = r"<think>(.*?)</think>"
        think_matches = re.findall(think_pattern, predict, re.DOTALL)
        
        if len(think_matches) > 1:
            return 0.5  # 稍微提高惩罚分数
        
        # 检测行级重复，但对医学术语更宽松
        lines = predict.split('\n')
        if len(lines) < 5:
            return 1.0
        
        # 过滤掉常见的医学术语重复
        medical_terms = ["lung", "chest", "heart", "opacity", "finding", "image", "x-ray"]
        filtered_lines = []
        for line in lines:
            line_lower = line.lower().strip()
            if len(line_lower) > 10:
                # 如果行主要包含医学术语，允许一定程度的重复
                medical_word_count = sum(1 for term in medical_terms if term in line_lower)
                if medical_word_count < 3:  # 不是纯医学术语行才检查重复
                    filtered_lines.append(line_lower)
        
        if len(filtered_lines) < 2:
            return 1.0
        
        # 统计连续重复的行
        consecutive_repeats = 0
        max_consecutive = 0
        
        for i in range(1, len(filtered_lines)):
            if filtered_lines[i] == filtered_lines[i-1]:
                consecutive_repeats += 1
                max_consecutive = max(max_consecutive, consecutive_repeats)
            else:
                consecutive_repeats = 0
        
        # 更宽松的重复惩罚
        if max_consecutive > 4:  # 提高阈值
            penalty = max(0.6, 1.0 - (max_consecutive - 4) * 0.1)  # 减少惩罚力度
            return penalty
        
        # 检测内容块重复
        if len(filtered_lines) > 0:
            unique_lines = set(filtered_lines)
            repetition_ratio = 1.0 - len(unique_lines) / len(filtered_lines)
            
            if repetition_ratio > 0.7:  # 提高阈值
                penalty = max(0.7, 1.0 - repetition_ratio * 0.8)  # 减少惩罚力度
                return penalty
        
        return 1.0
        
    except Exception as e:
        return 0.95  # 出错时给予较高分数

def extract_labels_flexible(text: str) -> List[str]:
    """
    灵活提取标签 - 改进版
    """
    # 首先尝试从<answer>标签中提取
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_match:
        text = answer_match.group(1).strip()
    
    # 如果包含标准回答格式，提取标签部分
    if "The label of this X-ray image is:" in text:
        text = text.split("The label of this X-ray image is:")[-1]
    
    # 检查是否为"no finding"
    if "No significant abnormalities were found in this X-ray." in text or "no finding" in text.lower():
        return ["no finding"]
    
    labels = []
    
    # 按逗号分割并清理
    parts = text.split(",")
    for part in parts:
        label = part.strip().lower()
        # 移除标点符号
        label = re.sub(r'[^\w\s]', '', label).strip()
        
        # 精确匹配
        if label in LABELS:
            labels.append(label)
        else:
            # 模糊匹配
            for predefined_label in LABELS:
                if label in predefined_label or predefined_label in label:
                    if predefined_label not in labels:
                        labels.append(predefined_label)
                    break
    
    # 如果没有找到标签，在整个文本中搜索
    if not labels:
        text_lower = text.lower()
        for label in LABELS:
            if label in text_lower:
                labels.append(label)
    
    return labels

def calculate_dynamic_reward(pred_labels: List[str], true_labels: List[str], 
                           training_step: int = 0, total_steps: int = 1000,
                           reward_mode: str = "adaptive") -> Dict[str, float]:
    """
    动态奖励计算 - 根据训练阶段调整奖励策略
    
    Args:
        pred_labels: 预测标签列表
        true_labels: 真实标签列表
        training_step: 当前训练步数
        total_steps: 总训练步数
        reward_mode: 奖励模式
    
    Returns:
        包含奖励分数和验证信息的字典
    """
    pred_set = set(pred_labels)
    true_set = set(true_labels)
    
    # 计算训练进度
    progress = min(training_step / total_steps, 1.0) if total_steps > 0 else 0.0
    
    # 早期训练：更宽松的奖励策略
    # 后期训练：更严格的奖励策略
    early_stage = progress < 0.3
    middle_stage = 0.3 <= progress < 0.7
    late_stage = progress >= 0.7
    
    verification_method = "none"
    reward = 0.0
    
    # 完全匹配
    if pred_set == true_set:
        reward = 1.0
        verification_method = "exact_match"
    
    # 子集匹配 - 预测是真实的子集
    elif pred_set.issubset(true_set) and len(pred_set) > 0:
        base_reward = len(pred_set) / len(true_set)
        
        # 考虑标签难度权重
        weighted_correct = sum(LABEL_DIFFICULTY_WEIGHTS.get(label, 1.0) for label in pred_set)
        weighted_total = sum(LABEL_DIFFICULTY_WEIGHTS.get(label, 1.0) for label in true_set)
        difficulty_bonus = weighted_correct / weighted_total if weighted_total > 0 else 0
        
        if early_stage:
            # 早期训练：鼓励任何正确预测
            reward = 0.9 * base_reward + 0.1 * difficulty_bonus
        elif middle_stage:
            # 中期训练：平衡奖励
            reward = 0.8 * base_reward + 0.2 * difficulty_bonus
        else:
            # 后期训练：更严格，但仍给予一定奖励
            reward = 0.7 * base_reward + 0.3 * difficulty_bonus
        
        verification_method = f"subset_match_{['early', 'middle', 'late'][int(progress*3)]}"
    
    # 超集匹配 - 预测包含了所有真实标签，但有多余的
    elif true_set.issubset(pred_set) and len(true_set) > 0:
        # 计算冗余度
        redundancy = len(pred_set - true_set) / len(pred_set)
        
        # 考虑标签难度权重
        weighted_correct = sum(LABEL_DIFFICULTY_WEIGHTS.get(label, 1.0) for label in true_set)
        weighted_total = sum(LABEL_DIFFICULTY_WEIGHTS.get(label, 1.0) for label in true_set)
        difficulty_bonus = weighted_correct / weighted_total if weighted_total > 0 else 1.0
        
        if early_stage:
            # 早期训练：超集给予较高奖励，轻微惩罚冗余
            base_reward = 0.85
            reward = base_reward * (1 - redundancy * 0.3) * difficulty_bonus
        elif middle_stage:
            # 中期训练：适中惩罚
            base_reward = 0.75
            reward = base_reward * (1 - redundancy * 0.5) * difficulty_bonus
        else:
            # 后期训练：更严格惩罚冗余
            base_reward = 0.65
            reward = base_reward * (1 - redundancy * 0.7) * difficulty_bonus
        
        verification_method = f"superset_match_{['early', 'middle', 'late'][int(progress*3)]}"
    
    # 部分重叠 - 有交集但不是子集或超集关系
    elif len(pred_set & true_set) > 0:
        intersection = len(pred_set & true_set)
        union = len(pred_set | true_set)
        
        # 计算加权交集
        weighted_intersection = sum(LABEL_DIFFICULTY_WEIGHTS.get(label, 1.0) for label in pred_set & true_set)
        weighted_union = sum(LABEL_DIFFICULTY_WEIGHTS.get(label, 1.0) for label in pred_set | true_set)
        
        jaccard = intersection / union if union > 0 else 0
        weighted_jaccard = weighted_intersection / weighted_union if weighted_union > 0 else 0
        
        if early_stage:
            # 早期训练：鼓励任何正确预测
            reward = 0.6 * jaccard + 0.4 * weighted_jaccard
        elif middle_stage:
            # 中期训练：平衡
            reward = 0.4 * jaccard + 0.3 * weighted_jaccard
        else:
            # 后期训练：更严格
            reward = 0.3 * jaccard + 0.2 * weighted_jaccard
        
        verification_method = f"partial_overlap_{['early', 'middle', 'late'][int(progress*3)]}"
    
    # 完全不匹配
    else:
        reward = 0.0
        verification_method = "no_match"
    
    # 应用非线性变换，增加奖励区分度
    if reward > 0:
        # 使用sigmoid变换增加区分度
        transformed_reward = 1 / (1 + math.exp(-10 * (reward - 0.5)))
        # 保持原始奖励的相对关系，但增加区分度
        reward = 0.7 * reward + 0.3 * transformed_reward
    
    # 基本指标计算
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    tn = len(LABELS) - tp - fp - fn
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    hamming_acc = (tp + tn) / len(LABELS)
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    
    return {
        'reward': reward,
        'verification_method': verification_method,
        'training_progress': progress,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'hamming_accuracy': hamming_acc,
        'jaccard': jaccard,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'pred_count': len(pred_labels),
        'true_count': len(true_labels),
        'pred_set': pred_set,
        'true_set': true_set
    }

def compute_multichoice_score_improved(predicts: List[str], ground_truths: List[str], 
                                     training_step: int = 0,
                                     total_steps: int = 1000,
                                     reward_mode: str = "adaptive",
                                     format_weight: float = 0.1,  # 降低格式权重
                                     **kwargs) -> List[Dict[str, float]]:
    """
    改进的多选题奖励评分函数
    
    Args:
        predicts: 预测结果列表
        ground_truths: 真实标签列表
        training_step: 当前训练步数
        total_steps: 总训练步数
        reward_mode: 奖励模式
        format_weight: 格式分数权重（降低）
        **kwargs: 额外参数
    """
    progress = min(training_step / total_steps, 1.0) if total_steps > 0 else 0.0
    
    print(f"\n=== 改进版多选题奖励计算 ===")
    print(f"样本数量: {len(predicts)}")
    print(f"训练进度: {progress:.2%} ({training_step}/{total_steps})")
    print(f"奖励模式: {reward_mode}")
    print(f"格式权重: {format_weight}")
    
    scores = []
    debug_indices = list(range(min(3, len(predicts))))
    
    # 统计信息
    all_rewards = []
    all_f1_scores = []
    all_hamming_accs = []
    format_scores = []
    repetition_scores = []
    verification_methods = []
    
    for i, (predict, ground_truth) in enumerate(zip(predicts, ground_truths)):
        debug = i in debug_indices
        
        try:
            # 格式和重复分数
            base_format_score = format_reward(predict)
            repetition_score = improved_repetition_penalty(predict)
            format_score = base_format_score * repetition_score
            
            # 灵活提取答案
            pred_text = predict
            pred_match = re.search(r"<answer>(.*?)</answer>", predict, re.DOTALL)
            if pred_match:
                pred_text = pred_match.group(1).strip()
            
            # 提取真实答案
            true_text = ground_truth
            true_match = re.search(r"<answer>(.*?)</answer>", ground_truth, re.DOTALL)
            if true_match:
                true_text = true_match.group(1).strip()
            
            # 提取标签
            pred_labels = extract_labels_flexible(pred_text)
            true_labels = extract_labels_flexible(true_text)
            
            if debug or os.getenv("DEBUG_MODE") == "true":
                medical_logger.log(f"样本 {i+1}:")
                medical_logger.log(f"预测文本: {pred_text[:100]}...")
                medical_logger.log(f"预测标签: {pred_labels}")
                medical_logger.log(f"真实标签: {true_labels}")
            
            # 计算动态奖励
            dynamic_result = calculate_dynamic_reward(
                pred_labels, true_labels, 
                training_step, total_steps, reward_mode
            )
            classification_reward = dynamic_result['reward']
            verification_method = dynamic_result['verification_method']
            reward_metrics = dynamic_result
            
            if debug or os.getenv("DEBUG_MODE") == "true":
                medical_logger.log(f"动态奖励: {classification_reward:.3f}")
                medical_logger.log(f"验证方法: {verification_method}")
                medical_logger.log(f"训练进度: {reward_metrics['training_progress']:.2%}")
            
            # 应用重复惩罚
            classification_score = classification_reward * repetition_score
            
            # 计算最终分数 - 动态调整格式权重
            # 训练后期降低格式权重，更关注内容质量
            dynamic_format_weight = format_weight * (1 - progress * 0.5)
            overall_score = dynamic_format_weight * format_score + (1 - dynamic_format_weight) * classification_score
            
            if debug or os.getenv("DEBUG_MODE") == "true":
                medical_logger.log(f"分类奖励: {classification_reward:.3f}")
                medical_logger.log(f"重复惩罚后: {classification_score:.3f}")
                medical_logger.log(f"动态格式权重: {dynamic_format_weight:.3f}")
                medical_logger.log(f"最终分数: {overall_score:.3f}")
                medical_logger.log(f"F1: {reward_metrics.get('f1', 0):.3f}, Hamming Acc: {reward_metrics.get('hamming_accuracy', 0):.3f}")
                medical_logger.log("---")
            
            scores.append({
                "overall": overall_score,
                "format": format_score,
                "classification": classification_score,
                "base_format": base_format_score,
                "repetition_penalty": repetition_score,
                "reward": classification_reward,
                "f1": reward_metrics.get('f1', 0.0),
                "hamming_accuracy": reward_metrics.get('hamming_accuracy', 0.0),
                "precision": reward_metrics.get('precision', 0.0),
                "recall": reward_metrics.get('recall', 0.0),
                "jaccard": reward_metrics.get('jaccard', 0.0),
                "tp": reward_metrics.get('tp', 0),
                "fp": reward_metrics.get('fp', 0),
                "fn": reward_metrics.get('fn', 0),
                "training_progress": reward_metrics.get('training_progress', 0.0)
                # "verification_method": verification_method
            })
            
            # 收集统计信息
            all_rewards.append(classification_reward)
            all_f1_scores.append(reward_metrics.get('f1', 0.0))
            all_hamming_accs.append(reward_metrics.get('hamming_accuracy', 0.0))
            format_scores.append(format_score)
            repetition_scores.append(repetition_score)
            verification_methods.append(verification_method)
            
        except Exception as e:
            medical_logger.log(f"样本 {i+1} 处理失败: {str(e)}")
            scores.append({
                "overall": 0.0,
                "format": 0.0,
                "classification": 0.0,
                "base_format": 0.0,
                "repetition_penalty": 0.8,
                "reward": 0.0,
                "f1": 0.0,
                "hamming_accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "jaccard": 0.0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "training_progress": progress
                # "verification_method": "error"
            })
    
    # 打印统计信息
    overall_scores = [s["overall"] for s in scores]
    
    print(f"\n=== 改进版多选题奖励统计信息 ===")
    print(f"样本数量: {len(scores)}")
    print(f"训练进度: {progress:.2%}")
    print(f"分类奖励 - 平均: {np.mean(all_rewards):.3f}, 标准差: {np.std(all_rewards):.3f}, 范围: [{np.min(all_rewards):.3f}, {np.max(all_rewards):.3f}]")
    print(f"F1分数 - 平均: {np.mean(all_f1_scores):.3f}, 标准差: {np.std(all_f1_scores):.3f}")
    print(f"Hamming准确率 - 平均: {np.mean(all_hamming_accs):.3f}, 标准差: {np.std(all_hamming_accs):.3f}")
    print(f"重复惩罚 - 平均: {np.mean(repetition_scores):.3f}, 范围: [{np.min(repetition_scores):.3f}, {np.max(repetition_scores):.3f}]")
    print(f"综合分数 - 平均: {np.mean(overall_scores):.3f}, 标准差: {np.std(overall_scores):.3f}, 范围: [{np.min(overall_scores):.3f}, {np.max(overall_scores):.3f}]")
    print(f"格式分数 - 平均: {np.mean(format_scores):.3f}, 标准差: {np.std(format_scores):.3f}")
    
    # 验证方法统计
    method_counts = Counter(verification_methods)
    print(f"验证方法统计: {dict(method_counts)}")
    
    # 奖励分布分析
    reward_ranges = {
        "0.0-0.2": sum(1 for s in overall_scores if 0.0 <= s < 0.2),
        "0.2-0.4": sum(1 for s in overall_scores if 0.2 <= s < 0.4),
        "0.4-0.6": sum(1 for s in overall_scores if 0.4 <= s < 0.6),
        "0.6-0.8": sum(1 for s in overall_scores if 0.6 <= s < 0.8),
        "0.8-1.0": sum(1 for s in overall_scores if 0.8 <= s <= 1.0),
    }
    print(f"奖励分布: {reward_ranges}")
    print("=" * 50)
    
    return scores

# 便捷函数
def compute_score_adaptive(predicts: List[str], ground_truths: List[str], 
                          training_step: int = 0, total_steps: int = 1000) -> List[Dict[str, float]]:
    """
    自适应模式多选题评分（推荐使用）
    根据训练进度动态调整奖励策略
    """
    return compute_multichoice_score_improved(
        predicts, ground_truths, 
        training_step=training_step, 
        total_steps=total_steps,
        reward_mode="adaptive"
    )

def compute_score_early_stage(predicts: List[str], ground_truths: List[str]) -> List[Dict[str, float]]:
    """
    早期训练阶段评分 - 更宽松的奖励策略
    """
    return compute_multichoice_score_improved(
        predicts, ground_truths, 
        training_step=0, 
        total_steps=1000,
        reward_mode="adaptive"
    )

def compute_score_late_stage(predicts: List[str], ground_truths: List[str]) -> List[Dict[str, float]]:
    """
    后期训练阶段评分 - 更严格的奖励策略
    """
    return compute_multichoice_score_improved(
        predicts, ground_truths, 
        training_step=800, 
        total_steps=1000,
        reward_mode="adaptive"
    ) 