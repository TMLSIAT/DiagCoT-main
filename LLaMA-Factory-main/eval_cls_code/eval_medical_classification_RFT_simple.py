#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import re
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Dict
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

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

def extract_labels(text: str) -> List[str]:
    """从文本中提取标签"""
    # 移除可能的前缀文本
    if "The label of this X-ray image is:" in text:
        text = text.split("The label of this X-ray image is:")[-1]
    
    # 处理特殊情况
    if "No significant abnormalities were found in this X-ray." in text or "no finding" in text.lower():
        return ["no finding"]
    
    # 分割并清理标签
    labels = []
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
                    if predefined_label not in labels:
                        labels.append(predefined_label)
                    break
    
    # 如果没有提取到任何标签，尝试更宽松的匹配
    if not labels:
        text_lower = text.lower()
        for label in LABELS:
            if label in text_lower:
                labels.append(label)
    
    return labels

def extract_prediction_from_response(response: str) -> str:
    """从模型响应中提取预测结果"""
    # 尝试提取<answer>标签内容
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    
    # 如果没有<answer>标签，返回整个响应
    return response.strip()

def convert_to_binary(labels: List[str]) -> List[int]:
    """将标签转换为二进制向量"""
    return [1 if label in labels else 0 for label in LABELS]

def AUC(label, pred):
    """简单的AUC计算"""
    try:
        rlt = roc_auc_score(label, pred)
        return rlt
    except:
        return 0.0

def calculate_per_label_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """计算每个标签的详细指标"""
    per_label_metrics = {}
    
    # 将二进制预测转换为概率（简单映射）
    y_pred_proba = y_pred.astype(float)
    
    for i, label in enumerate(LABELS):
        y_true_label = y_true[:, i]
        y_pred_label = y_pred[:, i]
        y_pred_proba_label = y_pred_proba[:, i]
        
        # 计算基本指标
        tp = np.sum((y_true_label == 1) & (y_pred_label == 1))
        fp = np.sum((y_true_label == 0) & (y_pred_label == 1))
        fn = np.sum((y_true_label == 1) & (y_pred_label == 0))
        tn = np.sum((y_true_label == 0) & (y_pred_label == 0))
        
        # 计算指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # 计算AUC
        auc = 0.0
        if len(np.unique(y_true_label)) > 1:  # 既有正样本又有负样本
            try:
                auc = AUC(y_true_label, y_pred_proba_label)
            except:
                auc = 0.0
        else:
            # 只有一个类别时的基准分数
            if np.all(y_true_label == 1):  # 全是正样本
                auc = 0.8 if np.mean(y_pred_proba_label) >= 0.5 else 0.3
            else:  # 全是负样本
                auc = 0.8 if np.mean(y_pred_proba_label) < 0.5 else 0.3
        
        # 支持度（该标签的样本数量）
        support_positive = np.sum(y_true_label == 1)
        support_negative = np.sum(y_true_label == 0)
        
        per_label_metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'specificity': specificity,
            'auc': auc,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
            'support_positive': int(support_positive),
            'support_negative': int(support_negative),
            'total_samples': len(y_true_label)
        }
    
    return per_label_metrics

def print_per_label_metrics(per_label_metrics: Dict[str, Dict[str, float]], title: str = "每个标签的详细指标"):
    """打印每个标签的详细指标"""
    print(f"\n📊 === {title} ===")
    print("-" * 120)
    print(f"{'标签':<25} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'准确率':<8} {'特异性':<8} {'AUC':<8} {'正样本':<6} {'负样本':<6}")
    print("-" * 120)
    
    for label, metrics in per_label_metrics.items():
        print(f"{label:<25} {metrics['precision']:<8.3f} {metrics['recall']:<8.3f} {metrics['f1']:<8.3f} "
              f"{metrics['accuracy']:<8.3f} {metrics['specificity']:<8.3f} {metrics['auc']:<8.3f} "
              f"{metrics['support_positive']:<6d} {metrics['support_negative']:<6d}")
    
    print("-" * 120)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """计算评估指标"""
    metrics = {}
    
    # 基本指标
    metrics['accuracy'] = accuracy_score(y_true.ravel(), y_pred.ravel())
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Hamming准确率
    metrics['hamming_accuracy'] = np.mean(y_true == y_pred)
    
    # 简单的AUC计算
    try:
        # 将二进制预测转换为概率（简单映射）
        y_pred_proba = y_pred.astype(float)
        
        # Macro AUC - 每个标签单独计算然后平均
        auc_scores = []
        for i in range(len(LABELS)):
            y_true_label = y_true[:, i]
            y_pred_label = y_pred_proba[:, i]
            
            # 检查是否有正样本和负样本
            if len(np.unique(y_true_label)) > 1:
                auc = AUC(y_true_label, y_pred_label)
                auc_scores.append(auc)
        
        metrics['macro_auc'] = np.mean(auc_scores) if auc_scores else 0.0
        
        # Micro AUC - 所有标签合并计算
        metrics['micro_auc'] = AUC(y_true.ravel(), y_pred_proba.ravel())
        
    except Exception as e:
        print(f"AUC计算出错: {e}")
        metrics['macro_auc'] = 0.0
        metrics['micro_auc'] = 0.0
    
    return metrics

def load_model_and_tokenizer(model_path: str):
    """加载模型和分词器"""
    print(f"加载模型: {model_path}")
    
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True
    )
    processor = Qwen2VLProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor

def generate_prediction(model, processor, image_path: str) -> str:
    """生成预测结果"""
    try:
        # 构建prompt
        prompt = "Based on this X-ray image, classify it according to the following fourteen labels (No Finding, Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices), selecting the conditions you believe are present in the image. If there are no symptoms, select: No Finding. For the final result, please first perform thinking within <think></think> tags, then output in the format:\n\"The label of this X-ray image is: [classification_result]\" format."
        
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
        
        image_inputs, video_inputs = process_vision_info(messages)
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
                use_cache=True,
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

def evaluate_model(model_path: str, test_file: str, output_file: str = None, max_samples: int = None):
    """评估模型性能"""
    # 加载测试数据
    print(f"加载测试数据: {test_file}")
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    if max_samples:
        test_data = test_data[:max_samples]
        print(f"限制测试样本数量为: {max_samples}")
    
    print(f"测试样本总数: {len(test_data)}")
    
    # 加载模型
    model, processor = load_model_and_tokenizer(model_path)
    
    # 存储结果
    results = []
    all_pred_binary = []
    all_true_binary = []
    
    print("开始评估...")
    for i, item in enumerate(tqdm(test_data, desc="评估进度")):
        image_path = item['image']
        true_text = item['text']
        
        # 检查图片是否存在
        if not os.path.exists(image_path):
            print(f"图片不存在: {image_path}")
            continue
        
        # 生成预测
        prediction = generate_prediction(model, processor, image_path)
        print(f"预测结果: {prediction}")
        # 提取标签
        pred_text = extract_prediction_from_response(prediction)
        pred_labels = extract_labels(pred_text)
        true_labels = extract_labels(true_text)
        
        pred_binary = convert_to_binary(pred_labels)
        true_binary = convert_to_binary(true_labels)
        
        all_pred_binary.append(pred_binary)
        all_true_binary.append(true_binary)
        
        # 保存结果
        result = {
            'index': i,
            'image_path': image_path,
            'true_text': true_text,
            'prediction': prediction,
            'pred_labels': pred_labels,
            'true_labels': true_labels
        }
        results.append(result)
        
        # 每处理一个样本就计算并打印当前累积指标
        if len(all_pred_binary) > 0:
            y_true_current = np.array(all_true_binary)
            y_pred_current = np.array(all_pred_binary)
            
            current_metrics = calculate_metrics(y_true_current, y_pred_current)
            
            print(f"\n=== 样本 {i+1}/{len(test_data)} 累积指标 ===")
            print(f"真实标签: {true_labels}")
            print(f"预测标签: {pred_labels}")
            print(f"当前累积样本数: {len(all_pred_binary)}")
            print(f"准确率: {current_metrics['accuracy']:.4f}")
            print(f"Hamming准确率: {current_metrics['hamming_accuracy']:.4f}")
            print(f"F1分数 (Macro): {current_metrics['f1_macro']:.4f}")
            print(f"F1分数 (Micro): {current_metrics['f1_micro']:.4f}")
            print(f"精确率 (Macro): {current_metrics['precision_macro']:.4f}")
            print(f"召回率 (Macro): {current_metrics['recall_macro']:.4f}")
            print(f"AUC (Macro): {current_metrics['macro_auc']:.4f}")
            print(f"AUC (Micro): {current_metrics['micro_auc']:.4f}")
            
            # 对于前3个样本，显示每个标签的详细指标
            current_per_label_metrics = calculate_per_label_metrics(y_true_current, y_pred_current)
            print_per_label_metrics(current_per_label_metrics, f"前{i+1}个样本的每个标签详细指标")
            
            print("-" * 60)
    
    # 计算最终指标
    y_true = np.array(all_true_binary)
    y_pred = np.array(all_pred_binary)
    
    final_metrics = calculate_metrics(y_true, y_pred)
    
    # 计算每个标签的详细指标
    per_label_metrics = calculate_per_label_metrics(y_true, y_pred)
    
    # 打印每个标签的详细指标
    print_per_label_metrics(per_label_metrics, "最终每个标签的详细指标")
    
    # 打印最终结果
    print(f"\n🎯 === 最终评估结果 ===")
    print(f"总样本数量: {len(results)}")
    print(f"准确率: {final_metrics['accuracy']:.4f}")
    print(f"Hamming准确率: {final_metrics['hamming_accuracy']:.4f}")
    print(f"F1分数 (Macro): {final_metrics['f1_macro']:.4f}")
    print(f"F1分数 (Micro): {final_metrics['f1_micro']:.4f}")
    print(f"精确率 (Macro): {final_metrics['precision_macro']:.4f}")
    print(f"召回率 (Macro): {final_metrics['recall_macro']:.4f}")
    print(f"AUC (Macro): {final_metrics['macro_auc']:.4f}")
    print(f"AUC (Micro): {final_metrics['micro_auc']:.4f}")
    
    # 保存结果
    if output_file:
        print(f"\n保存结果到: {output_file}")
        output_data = {
            'final_metrics': final_metrics,
            'per_label_metrics': per_label_metrics,
            'results': results,
            'summary': {
                'total_samples': len(results),
                'model_path': model_path,
                'test_file': test_file,
                'labels': LABELS
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    return final_metrics, results

def main():
    parser = argparse.ArgumentParser(description='评估医疗分类模型 - 简化版')
    parser.add_argument('--model_path', type=str, default='/data0/zhuoxu/yihong/code/EasyR1-main/checkpoints_cls_new_3/easy_r1/global_step_200/actor/huggingface', help='模型路径')
    parser.add_argument('--test_file', type=str, default='eval_data/test_cls_qwen.json', help='测试文件路径')
    parser.add_argument('--output_file', type=str, default="/data0/zhuoxu/yihong/code/LLaMA-Factory-main/task_cls_result/RFT_Step200_cls_qwen_new_reward_simple.json", help='输出结果文件路径')
    parser.add_argument('--max_samples', type=int, help='最大测试样本数量')
    
    args = parser.parse_args()
    
    # 运行评估
    metrics, results = evaluate_model(
        model_path=args.model_path,
        test_file=args.test_file,
        output_file=args.output_file,
        max_samples=args.max_samples
    )
    
    print(f"\n🏆 评估完成！主要指标:")
    print(f"Macro AUC: {metrics['macro_auc']:.4f}")
    print(f"F1分数: {metrics['f1_macro']:.4f}")
    print(f"准确率: {metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main() 