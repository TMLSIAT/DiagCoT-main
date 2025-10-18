#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np

# 测试数据
test_predicts = [
    "<think>这是一张胸部X光片，我需要仔细观察。</think><answer>The label of this X-ray image is: cardiomegaly, lung opacity</answer>",
    "<think>观察图像特征。</think><answer>The label of this X-ray image is: no finding</answer>",
    "<think>分析病理特征。</think><answer>The label of this X-ray image is: pneumonia, pleural effusion</answer>",
    "<think>检查异常区域。</think><answer>The label of this X-ray image is: atelectasis</answer>",
    "<think>综合判断。</think><answer>The label of this X-ray image is: consolidation, edema</answer>"
]

test_ground_truths = [
    "cardiomegaly, lung opacity",
    "no finding", 
    "pneumonia, pleural effusion",
    "atelectasis",
    "consolidation, edema"
]

def test_stability():
    """测试稳定性对比"""
    print("=== AUC稳定性测试 ===\n")
    
    try:
        # 导入稳定AUC版本
        from medical_classification_reward_stable_auc import compute_score as compute_stable_auc
        
        print("✓ 稳定AUC版本测试:")
        stable_scores = compute_stable_auc(test_predicts, test_ground_truths)
        stable_overall = [s["overall"] for s in stable_scores]
        
        print(f"稳定AUC版本 - 平均分: {np.mean(stable_overall):.3f}, 标准差: {np.std(stable_overall):.3f}")
        print(f"分数范围: [{np.min(stable_overall):.3f}, {np.max(stable_overall):.3f}]")
        
    except ImportError as e:
        print(f"✗ 无法导入稳定AUC版本: {e}")
    
    try:
        # 导入原版
        from medical_classification_reward import compute_score as compute_original
        
        print("\n✓ 原版测试:")
        original_scores = compute_original(test_predicts, test_ground_truths)
        original_overall = [s["overall"] for s in original_scores]
        
        print(f"原版 - 平均分: {np.mean(original_overall):.3f}, 标准差: {np.std(original_overall):.3f}")
        print(f"分数范围: [{np.min(original_overall):.3f}, {np.max(original_overall):.3f}]")
        
    except ImportError as e:
        print(f"✗ 无法导入原版: {e}")

if __name__ == "__main__":
    test_stability() 