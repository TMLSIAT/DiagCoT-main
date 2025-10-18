#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
对比测试：Macro AUC vs 多选题奖励方法
"""

import sys
import numpy as np
from medical_classification_reward_stable_auc import compute_score as compute_auc_score
from medical_classification_reward_multichoice import compute_multichoice_score

def create_test_samples():
    """创建测试样本"""
    
    # 测试样本：预测结果
    predicts = [
        # 样本1：完全正确
        """<think>分析X光片显示心脏扩大和肺部不透明。</think>
<answer>The label of this X-ray image is: cardiomegaly, lung opacity</answer>""",
        
        # 样本2：部分正确（预测了额外标签）
        """<think>观察到心脏扩大、肺水肿和支撑设备。</think>
<answer>The label of this X-ray image is: cardiomegaly, edema, support devices, lung opacity</answer>""",
        
        # 样本3：部分正确（遗漏了标签）
        """<think>可见心脏扩大。</think>
<answer>The label of this X-ray image is: cardiomegaly</answer>""",
        
        # 样本4：完全错误
        """<think>发现肺炎和骨折。</think>
<answer>The label of this X-ray image is: pneumonia, fracture</answer>""",
        
        # 样本5：无异常发现（正确）
        """<think>未发现明显异常。</think>
<answer>The label of this X-ray image is: no finding</answer>""",
        
        # 样本6：无异常发现（错误，实际有病变）
        """<think>未发现明显异常。</think>
<answer>The label of this X-ray image is: no finding</answer>"""
    ]
    
    # 对应的真实标签
    ground_truths = [
        "cardiomegaly, lung opacity",  # 样本1的真实标签
        "cardiomegaly, edema, support devices",  # 样本2的真实标签  
        "cardiomegaly, lung opacity, atelectasis",  # 样本3的真实标签
        "cardiomegaly, lung opacity",  # 样本4的真实标签
        "no finding",  # 样本5的真实标签
        "pneumonia, consolidation"  # 样本6的真实标签
    ]
    
    return predicts, ground_truths

def analyze_score_differences(auc_scores, mc_scores):
    """分析两种方法的分数差异"""
    print("\n=== 详细对比分析 ===")
    
    for i, (auc, mc) in enumerate(zip(auc_scores, mc_scores)):
        print(f"\n样本 {i+1}:")
        print(f"  Macro AUC方法:")
        print(f"    - 综合分数: {auc['overall']:.3f}")
        print(f"    - 分类分数: {auc['classification']:.3f}")
        print(f"    - Macro AUC: {auc['macro_auc']:.3f}")
        
        print(f"  多选题方法:")
        print(f"    - 综合分数: {mc['overall']:.3f}")
        print(f"    - 分类分数: {mc['classification']:.3f}")
        print(f"    - 分类奖励: {mc['reward']:.3f}")
        print(f"    - F1分数: {mc['f1']:.3f}")
        print(f"    - Hamming准确率: {mc['hamming_accuracy']:.3f}")
        print(f"    - TP/FP/FN: {mc['tp']}/{mc['fp']}/{mc['fn']}")
        
        print(f"  分数差异:")
        overall_diff = mc['overall'] - auc['overall']
        class_diff = mc['classification'] - auc['classification']
        print(f"    - 综合分数差异: {overall_diff:+.3f}")
        print(f"    - 分类分数差异: {class_diff:+.3f}")

def compare_reward_variance():
    """比较两种方法的奖励方差"""
    print("\n=== 奖励方差对比 ===")
    
    predicts, ground_truths = create_test_samples()
    
    # 计算AUC方法分数
    auc_scores = compute_auc_score(predicts, ground_truths, format_weight=0.1)
    
    # 计算多选题方法分数（三种模式）
    mc_balanced = compute_multichoice_score(predicts, ground_truths, 
                                          reward_type="balanced", use_weighted=True, format_weight=0.1)
    mc_strict = compute_multichoice_score(predicts, ground_truths, 
                                        reward_type="strict", use_weighted=False, format_weight=0.1)
    mc_lenient = compute_multichoice_score(predicts, ground_truths, 
                                         reward_type="lenient", use_weighted=True, format_weight=0.1)
    
    # 提取分类分数
    auc_class_scores = [s['classification'] for s in auc_scores]
    mc_balanced_scores = [s['classification'] for s in mc_balanced]
    mc_strict_scores = [s['classification'] for s in mc_strict]
    mc_lenient_scores = [s['classification'] for s in mc_lenient]
    
    print(f"分类分数方差对比:")
    print(f"  Macro AUC方法: 标准差={np.std(auc_class_scores):.4f}, 范围=[{np.min(auc_class_scores):.3f}, {np.max(auc_class_scores):.3f}]")
    print(f"  多选题-平衡模式: 标准差={np.std(mc_balanced_scores):.4f}, 范围=[{np.min(mc_balanced_scores):.3f}, {np.max(mc_balanced_scores):.3f}]")
    print(f"  多选题-严格模式: 标准差={np.std(mc_strict_scores):.4f}, 范围=[{np.min(mc_strict_scores):.3f}, {np.max(mc_strict_scores):.3f}]")
    print(f"  多选题-宽松模式: 标准差={np.std(mc_lenient_scores):.4f}, 范围=[{np.min(mc_lenient_scores):.3f}, {np.max(mc_lenient_scores):.3f}]")
    
    return auc_scores, mc_balanced, mc_strict, mc_lenient

def main():
    """主函数"""
    print("=" * 60)
    print("医学X-ray分类奖励方法对比测试")
    print("=" * 60)
    
    # 创建测试样本
    predicts, ground_truths = create_test_samples()
    
    print(f"\n测试样本描述:")
    descriptions = [
        "完全正确预测",
        "部分正确（多预测了标签）", 
        "部分正确（遗漏了标签）",
        "完全错误预测",
        "无异常发现（正确）",
        "无异常发现（错误）"
    ]
    
    for i, desc in enumerate(descriptions):
        print(f"  样本{i+1}: {desc}")
    
    # 运行对比测试
    auc_scores, mc_balanced, mc_strict, mc_lenient = compare_reward_variance()
    
    # 详细分析最佳方法（平衡模式）
    analyze_score_differences(auc_scores, mc_balanced)
    
    print("\n=== 方法优缺点总结 ===")
    print("\nMacro AUC方法:")
    print("  优点:")
    print("    - 全局视角，考虑整体分类性能")
    print("    - 对类别不平衡有一定鲁棒性")
    print("  缺点:")
    print("    - 所有样本得到相同的分类分数，缺乏个体区分度")
    print("    - 不利于GRPO算法的样本级别优化")
    print("    - 奖励信号稀疏，难以指导模型改进")
    
    print("\n多选题奖励方法:")
    print("  优点:")
    print("    - 每个样本有独立的奖励分数")
    print("    - 奖励信号丰富，有利于强化学习")
    print("    - 可以通过不同模式调节严格程度")
    print("    - 支持加权奖励，重视稀有疾病")
    print("  缺点:")
    print("    - 可能对噪声标签敏感")
    print("    - 需要调节奖励策略参数")
    
    print(f"\n=== 推荐方案 ===")
    print(f"对于GRPO算法，推荐使用多选题奖励方法（平衡模式+加权）：")
    print(f"  - 提供样本级别的差异化奖励")
    print(f"  - 奖励分数范围: [{np.min([s['classification'] for s in mc_balanced]):.3f}, {np.max([s['classification'] for s in mc_balanced]):.3f}]")
    print(f"  - 标准差: {np.std([s['classification'] for s in mc_balanced]):.4f}")
    print(f"  - 有利于模型学习和优化")

if __name__ == "__main__":
    main() 