#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分类模型测试脚本
用于测试训练好的Qwen2-VL分类模型在验证集上的性能
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import sys
import json
import torch
import argparse
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, hamming_loss, roc_auc_score
from tqdm import tqdm

# 添加src路径到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from transformers import AutoProcessor, AutoConfig
from model.modeling_cls import Qwen2VLForSequenceClassification, Qwen2_5_VLForSequenceClassification

# 定义类别映射（与训练时保持一致）
# CLASS_2_ID = {
#     "no finding": 0,
#     "enlarged cardiomediastinum": 1,
#     "cardiomegaly": 2,
#     "lung opacity": 3,
#     "lung lesion": 4,
#     "edema": 5,
#     "consolidation": 6,
#     "pneumonia": 7,
#     "atelectasis": 8,
#     "pneumothorax": 9,
#     "pleural effusion": 10,
#     "pleural other": 11,
#     "fracture": 12,
#     "support devices": 13
# }
CLASS_2_ID = {
    "enlarged cardiomediastinum": 0,
    "cardiomegaly": 1,
    "lung opacity": 2,
    "edema": 3,
    "consolidation": 4,
    "atelectasis": 5,
    "pleural effusion": 6,
    "support devices": 7,
    "pneumonia": 8,
    "lung lesion": 9,
    "pneumothorax": 10,
    "pleural other": 11,
    "fracture": 12, 
    "no finding": 13,
}
ID_2_CLASS = {v: k for k, v in CLASS_2_ID.items()}

def parse_multi_labels(label_str: str) -> np.ndarray:
    """
    将标签字符串解析为多标签二进制向量
    
    Args:
        label_str: 标签字符串，如 "no finding" 或 "cardiomegaly, edema"
    
    Returns:
        长度为14的二进制向量
    """
    labels = np.zeros(len(CLASS_2_ID), dtype=np.float32)
    
    if not label_str or label_str.strip() == "":
        return labels
    
    # 分割多个标签并处理
    label_parts = [part.strip().lower() for part in label_str.split(',')]
    
    for label in label_parts:
        if label in CLASS_2_ID:
            labels[CLASS_2_ID[label]] = 1.0
    
    return labels

def load_model_and_processor(model_path: str, device: str = "cuda"):
    """
    加载模型和处理器
    
    Args:
        model_path: 模型路径
        device: 设备类型
    
    Returns:
        model, processor
    """
    print(f"正在加载模型: {model_path}")
    
    # 加载配置
    config = AutoConfig.from_pretrained(model_path)
    
    # 设置分类相关配置
    config.num_labels = 14
    config.mlp_head_hidden_dim = getattr(config, 'mlp_head_hidden_dim', 0)
    config.mlp_head_dropout = getattr(config, 'mlp_head_dropout', 0.0)
    config.use_margin_head = getattr(config, 'use_margin_head', False)
    
    # 根据模型类型加载相应的模型
    if "Qwen2.5" in config.architectures[0]:
        model = Qwen2_5_VLForSequenceClassification.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float16,
            device_map=device,
            attn_implementation="flash_attention_2"
        )
    else:
        model = Qwen2VLForSequenceClassification.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float16,
            device_map=device,
            attn_implementation="flash_attention_2"
        )
    
    # 加载处理器
    processor = AutoProcessor.from_pretrained(model_path)
    
    model.eval()
    print(f"模型加载完成，设备: {next(model.parameters()).device}")
    
    return model, processor

def load_test_data(test_file: str) -> List[Dict[str, Any]]:
    """
    加载测试数据
    
    Args:
        test_file: 测试文件路径
    
    Returns:
        测试数据列表
    """
    print(f"正在加载测试数据: {test_file}")
    
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"加载了 {len(data)} 条测试数据")
    return data

def preprocess_data(data_item: Dict[str, Any], processor) -> Dict[str, torch.Tensor]:
    """
    预处理单个数据项
    
    Args:
        data_item: 数据项
        processor: 处理器
    
    Returns:
        处理后的数据字典
    """
    # 加载图像
    image_path = data_item['image']
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    
    image = Image.open(image_path).convert('RGB')
    
    # 构建对话格式
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": data_item['prompt']}
            ]
        }
    ]
    
    # 应用聊天模板
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    
    # 处理输入
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True
    )
    
    return inputs

def predict_single(model, processor, data_item: Dict[str, Any], device: str) -> np.ndarray:
    """
    对单个样本进行预测
    
    Args:
        model: 模型
        processor: 处理器
        data_item: 数据项
        device: 设备
    
    Returns:
        预测概率向量
    """
    try:
        # 预处理数据
        inputs = preprocess_data(data_item, processor)
        
        # 移动到设备
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # 推理
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # 应用sigmoid获取概率
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        return probs
    
    except Exception as e:
        print(f"预测失败: {e}")
        return np.zeros(14, dtype=np.float32)

def evaluate_model(model, processor, test_data: List[Dict[str, Any]], device: str, threshold: float = 0.5):
    """
    评估模型性能
    
    Args:
        model: 模型
        processor: 处理器
        test_data: 测试数据
        device: 设备
        threshold: 分类阈值
    
    Returns:
        评估结果字典和样本详细结果
    """
    print("开始模型评估...")
    
    all_predictions = []
    all_labels = []
    all_probs = []
    sample_results = []  # 存储每个样本的详细结果
    
    # 逐个样本预测
    for i, data_item in enumerate(tqdm(test_data, desc="预测进度")):
        # 获取真实标签
        true_labels = parse_multi_labels(data_item['label'])
        all_labels.append(true_labels)
        
        # 预测
        probs = predict_single(model, processor, data_item, device)
        all_probs.append(probs)
        
        # 二值化预测
        predictions = (probs > threshold).astype(np.float32)
        all_predictions.append(predictions)
        
        # 转换为标签名称
        true_label_names = [ID_2_CLASS[j] for j, label in enumerate(true_labels) if label > 0]
        pred_label_names = [ID_2_CLASS[j] for j, pred in enumerate(predictions) if pred > 0]
        
        # 保存样本结果
        sample_result = {
            "sample_id": i,
            "image_path": data_item['image'],
            "true_labels": true_label_names,
            "predicted_labels": pred_label_names,
            "true_labels_binary": true_labels.tolist(),
            "predicted_labels_binary": predictions.tolist(),
            "prediction_probabilities": probs.tolist()
        }
        sample_results.append(sample_result)
        
        # 打印前几个样本的详细结果
        
        print(f"\n样本 {i+1}:")
        print(f"图像: {data_item['image']}")
        print(f"真实标签: {data_item['label']}")
        print(f"预测概率: {probs}")
        print(f"预测标签: {pred_label_names}")
    
    # 转换为numpy数组
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 计算评估指标
    results = {}
    
    # 精确匹配率（所有标签都正确）
    exact_match = np.mean(np.all(all_predictions == all_labels, axis=1))
    results['exact_match'] = exact_match
    
    # 汉明损失（错误标签的比例）
    hamming = hamming_loss(all_labels, all_predictions)
    results['hamming_loss'] = hamming
    
    # Jaccard分数（交集/并集）
    intersection = np.sum(all_predictions * all_labels, axis=1)
    union = np.sum((all_predictions + all_labels) > 0, axis=1)
    jaccard = np.mean(intersection / (union + 1e-8))
    results['jaccard_score'] = jaccard
    
    # 微平均和宏平均指标
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        all_labels.flatten(), all_predictions.flatten(), average='micro', zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels.flatten(), all_predictions.flatten(), average='macro', zero_division=0
    )
    
    results.update({
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro
    })
    
    # 每个类别的指标
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )
    
    # 计算每个类别的AUC
    auc_per_class = []
    for i in range(len(CLASS_2_ID)):
        try:
            # 检查是否有正样本和负样本
            if len(np.unique(all_labels[:, i])) > 1:
                auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
            else:
                auc = 0.0  # 如果只有一个类别，AUC设为0
        except Exception as e:
            print(f"计算类别 {ID_2_CLASS[i]} 的AUC时出错: {e}")
            auc = 0.0
        auc_per_class.append(auc)
    
    # 计算宏平均AUC
    macro_auc = np.mean(auc_per_class)
    results['macro_auc'] = macro_auc
    
    class_results = {}
    for i, class_name in ID_2_CLASS.items():
        class_results[class_name] = {
            'precision': precision_per_class[i],
            'recall': recall_per_class[i],
            'f1': f1_per_class[i],
            'support': support_per_class[i],
            'auc': auc_per_class[i]
        }
    
    results['per_class'] = class_results
    
    return results, all_probs, all_predictions, all_labels, sample_results

def print_results(results: Dict[str, Any]):
    """
    打印评估结果
    
    Args:
        results: 评估结果字典
    """
    print("\n" + "="*50)
    print("模型评估结果")
    print("="*50)
    
    print(f"精确匹配率: {results['exact_match']:.4f}")
    print(f"汉明损失: {results['hamming_loss']:.4f}")
    print(f"Jaccard分数: {results['jaccard_score']:.4f}")
    
    print("\n微平均指标:")
    print(f"  精确率: {results['precision_micro']:.4f}")
    print(f"  召回率: {results['recall_micro']:.4f}")
    print(f"  F1分数: {results['f1_micro']:.4f}")
    
    print("\n宏平均指标:")
    print(f"  精确率: {results['precision_macro']:.4f}")
    print(f"  召回率: {results['recall_macro']:.4f}")
    print(f"  F1分数: {results['f1_macro']:.4f}")
    print(f"  AUC分数: {results['macro_auc']:.4f}")
    
    print("\n各类别详细指标:")
    print(f"{'类别':<25} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'AUC':<8} {'支持数':<8}")
    print("-" * 73)
    
    for class_name, metrics in results['per_class'].items():
        print(f"{class_name:<25} {metrics['precision']:<8.4f} {metrics['recall']:<8.4f} "
              f"{metrics['f1']:<8.4f} {metrics['auc']:<8.4f} {int(metrics['support']):<8}")
    
    # 按AUC分数排序显示
    print("\n按AUC分数排序的类别:")
    sorted_classes = sorted(results['per_class'].items(), key=lambda x: x[1]['auc'], reverse=True)
    print(f"{'类别':<25} {'AUC':<8}")
    print("-" * 35)
    for class_name, metrics in sorted_classes:
        print(f"{class_name:<25} {metrics['auc']:<8.4f}")

def save_results(results: Dict[str, Any], output_file: str):
    """
    保存评估结果到文件
    
    Args:
        results: 评估结果
        output_file: 输出文件路径
    """
    # 转换numpy类型为Python原生类型以便JSON序列化
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    results_serializable = convert_numpy(results)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"\n评估结果已保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="测试Qwen2-VL分类模型")
    parser.add_argument("--model_path", type=str, default="/data0/zhuoxu/yihong/code/Qwen2-VL-Finetune/output/qwen_cls_mlp_head_2layer_use_logit_adjusted_loss_2", help="模型路径")
    parser.add_argument("--test_file", type=str, default="/data0/zhuoxu/yihong/code/Qwen2-VL-Finetune/test_cls_qwen_converted_1000.json", help="测试文件路径")
    parser.add_argument("--output_dir", type=str, default="./test_results_2_mlp_layer_use_logit_adjusted_loss_2_latest_threshold_0.5", help="输出目录")
    parser.add_argument("--device", type=str, default="cuda", help="设备类型")
    parser.add_argument("--threshold", type=float, default=0.5, help="分类阈值")
    parser.add_argument("--batch_size", type=int, default=1, help="批处理大小（当前仅支持1）")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 检查设备
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        args.device = "cpu"
    
    print(f"使用设备: {args.device}")
    
    try:
        # 加载模型和处理器
        model, processor = load_model_and_processor(args.model_path, args.device)
        
        # 加载测试数据
        test_data = load_test_data(args.test_file)
        
        # 评估模型
        results, all_probs, all_predictions, all_labels, sample_results = evaluate_model(
            model, processor, test_data, args.device, args.threshold
        )
        
        # 打印结果
        print_results(results)
        
        # 保存结果
        output_file = os.path.join(args.output_dir, "evaluation_results.json")
        save_results(results, output_file)
        
        # 保存预测结果
        predictions_file = os.path.join(args.output_dir, "predictions.npz")
        np.savez(predictions_file, 
                 probabilities=all_probs,
                 predictions=all_predictions,
                 labels=all_labels)
        print(f"预测结果已保存到: {predictions_file}")
        
        # 保存每个样本的详细结果到JSON
        sample_results_file = os.path.join(args.output_dir, "sample_predictions.json")
        with open(sample_results_file, 'w', encoding='utf-8') as f:
            json.dump(sample_results, f, indent=2, ensure_ascii=False)
        print(f"样本详细预测结果已保存到: {sample_results_file}")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())