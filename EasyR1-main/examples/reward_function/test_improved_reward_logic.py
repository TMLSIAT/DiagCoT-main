#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append('EasyR1-main/examples/reward_function')

from medical_classification_reward_multichoice_2 import compute_multichoice_score

def test_improved_reward_logic():
    """完整测试改进后的奖励逻辑，包含详细日志"""
    
    # 设置调试模式以生成详细日志
    os.environ["DEBUG_MODE"] = "true"
    
    # 测试用例 - 涵盖所有可能的匹配情况
    test_cases = [
        # 1. 完全匹配
        ("<think>分析X光片显示心脏扩大和肺部不透明。</think><answer>cardiomegaly, lung opacity</answer>", 
         "<answer>cardiomegaly, lung opacity</answer>", 
         "完全匹配"),
        
        # 2. 子集匹配 - 预测是真实的子集（预测少了）
        ("<think>可以看到明显的心脏扩大。</think><answer>cardiomegaly</answer>", 
         "<answer>cardiomegaly, lung opacity</answer>", 
         "子集匹配(预测少了)"),
        
        # 3. 超集匹配 - 预测包含所有真实标签但有多余（预测多了）
        ("<think>心脏扩大，肺部不透明，还可能有水肿。</think><answer>cardiomegaly, lung opacity, edema</answer>", 
         "<answer>cardiomegaly, lung opacity</answer>", 
         "超集匹配(预测多了)"),
        
        # 4. 部分重叠 - 有交集但不是子集或超集关系
        ("<think>心脏扩大和水肿征象。</think><answer>cardiomegaly, edema</answer>", 
         "<answer>cardiomegaly, lung opacity</answer>", 
         "部分重叠(一对一错)"),
        
        # 5. 完全不匹配
        ("<think>发现肺炎征象。</think><answer>pneumonia</answer>", 
         "<answer>cardiomegaly, lung opacity</answer>", 
         "完全不匹配"),
        
        # 6. 无发现的情况
        ("<think>X光片未见明显异常。</think><answer>no finding</answer>", 
         "<answer>no finding</answer>", 
         "无发现匹配"),
        
        # 7. 复杂的超集情况
        ("<think>多种异常征象。</think><answer>cardiomegaly, lung opacity, pneumonia, edema</answer>", 
         "<answer>cardiomegaly, lung opacity</answer>", 
         "复杂超集匹配"),
        
        # 8. 部分重叠但预测更多
        ("<think>心脏问题和其他异常。</think><answer>cardiomegaly, pneumonia, fracture</answer>", 
         "<answer>cardiomegaly, lung opacity</answer>", 
         "部分重叠(预测更多)"),
    ]
    
    print("=" * 80)
    print("完整测试改进后的医学多选题奖励逻辑")
    print("=" * 80)
    print(f"测试用例数量: {len(test_cases)}")
    print(f"测试模式: strict, balanced, partial")
    print("=" * 80)
    
    # 对每种模式进行测试
    for mode in ["strict", "balanced", "partial"]:
        print(f"\n{'='*20} {mode.upper()} 模式测试 {'='*20}")
        
        predicts = [case[0] for case in test_cases]
        ground_truths = [case[1] for case in test_cases]
        
        print(f"\n开始计算 {mode} 模式的奖励分数...")
        
        # 计算奖励分数
        results = compute_multichoice_score(
            predicts, ground_truths, 
            subset_mode=mode,
            format_weight=0.1  # 保留一定的格式权重
        )
        
        print(f"\n{mode.upper()} 模式详细结果:")
        print("-" * 60)
        
        total_reward = 0
        for i, (predict, ground_truth, desc) in enumerate(test_cases):
            result = results[i]
            total_reward += result['reward']
            
            print(f"测试 {i+1}: {desc}")
            print(f"  预测标签: {result['pred_labels']}")
            print(f"  真实标签: {result['true_labels']}")
            print(f"  分类奖励: {result['reward']:.3f}")
            print(f"  验证方法: {result['verification_method']}")
            print(f"  最终分数: {result['overall']:.3f}")
            print(f"  精确率: {result['precision']:.3f}, 召回率: {result['recall']:.3f}, F1: {result['f1']:.3f}")
            print()
        
        avg_reward = total_reward / len(test_cases)
        print(f"{mode.upper()} 模式总结:")
        print(f"  平均分类奖励: {avg_reward:.3f}")
        print(f"  总体表现: {'优秀' if avg_reward > 0.7 else '良好' if avg_reward > 0.5 else '一般' if avg_reward > 0.3 else '较差'}")
        
        # 统计验证方法分布
        method_counts = {}
        for result in results:
            method = result['verification_method']
            method_counts[method] = method_counts.get(method, 0) + 1
        
        print(f"  验证方法分布: {method_counts}")
        print("-" * 60)
    
    print(f"\n{'='*80}")
    print("测试完成！请查看上述日志了解详细的奖励计算过程。")
    print("日志文件已保存，可用于进一步分析。")
    print("=" * 80)

def compare_modes():
    """比较不同模式的奖励策略"""
    print("\n" + "="*50)
    print("奖励模式对比说明")
    print("="*50)
    
    print("\n1. STRICT 模式（严格）:")
    print("   - 完全匹配: 1.0分")
    print("   - 子集匹配: 0分")
    print("   - 超集匹配: 0分") 
    print("   - 部分重叠: 0分")
    print("   - 适用场景: 要求完全准确的诊断")
    
    print("\n2. BALANCED 模式（平衡，推荐）:")
    print("   - 完全匹配: 1.0分")
    print("   - 子集匹配: 部分分数×0.8")
    print("   - 超集匹配: 0.6分")
    print("   - 部分重叠: 召回率×0.4")
    print("   - 适用场景: 平衡准确性和召回率")
    
    print("\n3. PARTIAL 模式（部分奖励）:")
    print("   - 完全匹配: 1.0分")
    print("   - 子集匹配: 预测正确数/总正确数")
    print("   - 超集匹配: 0.8分")
    print("   - 部分重叠: Jaccard相似度×0.5")
    print("   - 适用场景: 鼓励部分正确的预测")

if __name__ == "__main__":
    test_improved_reward_logic()
    compare_modes() 