#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用VLLM部署的72B模型进行医学影像分类评估
通过OpenAI API接口调用本地VLLM服务
结合分类任务评估和VLLM API调用功能
"""

import json
import os
import re
import base64
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Dict
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from openai import OpenAI
from retrying import retry
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

class VLLMClassifier:
    """
    使用VLLM API的医学影像分类器
    """
    
    def __init__(self, model_name, api_key="qwen-abc123", api_url="http://127.0.0.1:8000/v1"):
        """
        初始化VLLM分类器
        
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
        print(f"初始化VLLM分类器，模型: {model_name}, API地址: {api_url}")

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
                    "text": "Based on this X-ray image, classify it according to the following fourteen labels (No Finding, Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices), selecting the conditions you believe are present in the image. If there are no symptoms, select: No Finding. Output the thinking process in <think> </think>, and output the final classification result within the <answer> </answer>. Following \"<think></think>\n<answer></answer> \" format. The label of this X-ray image is: [classification_result]"
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

    def classify(self, image_path):
        """
        对医学影像进行分类
        
        Args:
            image_path (str): 图像文件路径
            
        Returns:
            str: 分类结果
        """
        try:
            # 构造消息
            messages = self._construct_messages(image_path)
            
            # 调用API进行分类
            response = self._retry_call(messages)
            
            return response if response else ""
            
        except Exception as e:
            print(f"分类时出错 {image_path}: {str(e)}")
            return ""

def extract_answer_content(raw_prediction):
    """
    直接提取 <answer> 标签内的完整内容（保留原始格式）
    兼容标签大小写、前后空格等情况
    """
    # 统一处理输入格式
    text = raw_prediction[0] if isinstance(raw_prediction, list) else str(raw_prediction)
    
    # 匹配任意大小写和空格的标签
    match = re.search(r'<\s*answer\s*>([\s\S]*?)<\s*/\s*answer\s*>', text, re.IGNORECASE)
    
    if match:
        # 提取内容并去除首尾空白
        return match.group(1).strip()
    else:
        print(f"WARNING: No <answer> tag found in:\n{text}")
        return raw_prediction

def extract_labels(text: str) -> List[str]:
    """
    从文本中提取标签
    
    Args:
        text (str): 输入文本
        
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

def convert_to_binary(labels: List[str]) -> List[int]:
    """
    将标签转换为二进制向量
    
    Args:
        labels (List[str]): 标签列表
        
    Returns:
        List[int]: 二进制向量
    """
    return [1 if label in labels else 0 for label in LABELS]

def AUC(label, pred):
    """
    简单的AUC计算
    
    Args:
        label: 真实标签
        pred: 预测结果
        
    Returns:
        float: AUC分数
    """
    try:
        rlt = roc_auc_score(label, pred)
        return rlt
    except:
        return 0.0

def calculate_per_label_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    计算每个标签的详细指标
    
    Args:
        y_true (np.ndarray): 真实标签
        y_pred (np.ndarray): 预测标签
        
    Returns:
        Dict[str, Dict[str, float]]: 每个标签的详细指标
    """
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
    """
    打印每个标签的详细指标
    
    Args:
        per_label_metrics (Dict[str, Dict[str, float]]): 每个标签的指标
        title (str): 标题
    """
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
    """
    计算评估指标
    
    Args:
        y_true (np.ndarray): 真实标签
        y_pred (np.ndarray): 预测标签
        
    Returns:
        Dict[str, float]: 评估指标字典
    """
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

def evaluate_model_vllm(model_name: str, test_file: str, api_url: str = "http://127.0.0.1:8000/v1", 
                       output_file: str = None, max_samples: int = None):
    """
    使用VLLM API评估模型性能
    
    Args:
        model_name (str): 模型名称
        test_file (str): 测试文件路径
        api_url (str): VLLM API服务地址
        output_file (str): 输出结果文件路径
        max_samples (int): 最大测试样本数量
        
    Returns:
        tuple: (最终指标, 结果列表)
    """
    # 加载测试数据
    print(f"加载测试数据: {test_file}")
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    if max_samples:
        test_data = test_data[:max_samples]
        print(f"限制测试样本数量为: {max_samples}")
    
    print(f"测试样本总数: {len(test_data)}")
    
    # 初始化VLLM分类器
    classifier = VLLMClassifier(model_name, api_url=api_url)
    
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
        prediction = classifier.classify(image_path)
        
        # 提取标签
        pred_text = extract_answer_content(prediction)
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
            'CoT': prediction,
            'prediction': pred_text,
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
            if i < 3:
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
            'model_info': {
                'model_name': model_name,
                'api_url': api_url,
                'total_samples': len(results)
            },
            'final_metrics': final_metrics,
            'per_label_metrics': per_label_metrics,
            'detailed_results': results,
            'summary': {
                'total_samples': len(results),
                'model_name': model_name,
                'test_file': test_file,
                'labels': LABELS
            }
        }
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    return final_metrics, results

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='使用VLLM评估医疗分类模型 - 72B版本')
    parser.add_argument('--model_name', type=str, 
                       default='/data0/zhuoxu/yihong/code/Qwen2.5-VL-72B-Instruct-AWQ', 
                       help='VLLM部署的模型名称')
    parser.add_argument('--test_file', type=str, 
                       default='/data0/zhuoxu/yihong/code/EasyR1-main/eval_data/test_cls_qwen.json', 
                       help='测试文件路径')
    parser.add_argument('--api_url', type=str, 
                       default='http://127.0.0.1:8000/v1', 
                       help='VLLM API服务地址')
    parser.add_argument('--output_file', type=str, 
                       default='/data0/zhuoxu/yihong/code/LLaMA-Factory-main/eval_result/Qwen2.5-VL-72B-VLLM-CLS.json', 
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
    print(f"Macro AUC: {metrics['macro_auc']:.4f}")
    print(f"F1分数: {metrics['f1_macro']:.4f}")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"\n结果已保存到: {args.output_file}")

if __name__ == "__main__":
    # 配置参数
    TEST_FILE = "/data0/zhuoxu/yihong/code/EasyR1-main/eval_data/test_cls_qwen.json"
    MODEL_NAME = "/data0/zhuoxu/yihong/code/Qwen2.5-VL-72B-Instruct-AWQ"  # 根据实际部署的模型名称调整
    API_URL = "http://127.0.0.1:8000/v1"  # VLLM服务地址
    OUTPUT_FILE = "/data0/zhuoxu/yihong/code/LLaMA-Factory-main/eval_result/Qwen2.5-VL-72B-VLLM-CLS.json"
    
    # 运行评估
    main()