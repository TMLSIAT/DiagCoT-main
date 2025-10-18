#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用VLLM部署的32B模型进行IU-xray数据集医学影像报告评估
通过OpenAI API接口调用本地VLLM服务
"""

import os
import json
import nltk
import base64
import re
from tqdm import tqdm
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from openai import OpenAI
from retrying import retry

# 确保NLTK资源就绪
try:
    nltk.data.find('/data0/zhuoxu/yihong/code/tokenizers/punkt')
except:
    nltk.download('punkt', quiet=True)

def preprocess(text):
    """预处理文本，进行分词和小写化"""
    return nltk.word_tokenize(text.lower())

def compute_meteor(reference, hypothesis):
    """计算METEOR得分"""
    ref_tokens = [preprocess(reference)]
    hyp_tokens = preprocess(hypothesis)
    return meteor_score(ref_tokens, hyp_tokens)

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

class VLLMReportGenerator:
    """
    使用VLLM API的医学报告生成器
    """
    
    def __init__(self, model_name, api_key="qwen-abc123", api_url="http://127.0.0.1:8000/v1"):
        """
        初始化VLLM报告生成器
        
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
        print(f"初始化VLLM客户端，模型: {model_name}, API地址: {api_url}")

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
                    "text": "Based on this medical X-ray image, please analyze and generate a diagnostic report. Output the thinking process in <think> </think>, and output the final findings and impression within the <answer> </answer>. Following \"<think></think>\n<answer></answer> \" format."
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
            # max_tokens=12000,
        )
        return response.choices[0].message.content

    def generate(self, image_path):
        """
        生成医学影像诊断报告
        
        Args:
            image_path (str): 图像文件路径
            
        Returns:
            str: 生成的诊断报告
        """
        try:
            # 构造消息
            messages = self._construct_messages(image_path)
            
            # 调用API生成报告
            response = self._retry_call(messages)
            
            return response if response else ""
            
        except Exception as e:
            print(f"生成报告时出错 {image_path}: {str(e)}")
            return ""

def calculate_metrics(prediction, reference):
    """
    计算BLEU1-4、ROUGE-L和METEOR指标
    
    Args:
        prediction (str): 预测的报告
        reference (str): 参考报告
        
    Returns:
        dict: 包含各项指标的字典
    """
    # 预处理文本
    pred_tokens = nltk.word_tokenize(prediction.lower())
    ref_tokens = nltk.word_tokenize(reference.lower())
    
    # 平滑函数处理零值情况
    smoother = SmoothingFunction().method1
    
    # 计算BLEU1-4
    bleu_scores = {}
    for n in range(1, 5):
        weights = tuple([1.0/n]*n + [0.0]*(4-n))  # 动态权重
        bleu_scores[f'bleu-{n}'] = sentence_bleu(
            [ref_tokens], 
            pred_tokens, 
            weights=weights[:n],
            smoothing_function=smoother
        )
    
    # 计算ROUGE-L
    rouge = Rouge()
    try:
        rouge_scores = rouge.get_scores(" ".join(pred_tokens), " ".join(ref_tokens))[0]
        rouge_l = rouge_scores['rouge-l']['f']
    except:
        rouge_l = 0.0
    
    # 计算METEOR
    meteor = compute_meteor(reference, prediction)
    
    # 返回所有指标
    return {**{k: round(v, 4) for k, v in bleu_scores.items()}, 
            'rouge-l': round(rouge_l, 4), 
            'meteor': round(meteor, 4)}

def main(test_path, model_name, api_url="http://127.0.0.1:8000/v1", 
         output_path=""):
    """
    主评估函数
    
    Args:
        test_path (str): 测试数据路径
        model_name (str): 模型名称
        api_url (str): VLLM API服务地址
        output_path (str): 结果输出路径
    """
    # 初始化生成器
    generator = VLLMReportGenerator(model_name, api_url=api_url)
    
    # 加载测试数据
    print(f"加载测试数据: {test_path}")
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"共有 {len(test_data)} 个测试样本")
    
    # 处理所有样本
    results = []
    metrics_accumulator = {f'bleu-{n}': 0.0 for n in range(1, 5)}
    metrics_accumulator['rouge-l'] = 0.0
    metrics_accumulator['meteor'] = 0.0
    processed_count = 0  # 成功处理的样本计数
    
    for data in tqdm(test_data, desc="处理样本"):
        try:
            # 生成报告
            prediction = generator.generate(data['image_url'])
            prediction_answer = extract_answer_content(prediction)
            print("prediction_answer:", prediction_answer)
            
            # 计算指标
            metrics = calculate_metrics(prediction_answer, data['reference_report'])
            
            # 记录结果
            results.append({
                "process_id": data['process_id'],
                "image_url": data['image_url'],
                "CoT": prediction,
                "predicted_report": prediction_answer,
                "reference_report": data['reference_report'],
                "metrics": metrics
            })
            
            print(f'样本指标: {metrics}')
            
            # 累加指标
            for k in metrics_accumulator:
                metrics_accumulator[k] += metrics[k]
            processed_count += 1
            
            # 实时计算当前平均值
            current_avg = {
                k: round(v / processed_count, 4)
                for k, v in metrics_accumulator.items()
            }
            print(f'当前平均指标: {current_avg}')
                
        except Exception as e:
            print(f"处理样本时出错 {data['image_url']}: {str(e)}")
            # 添加失败记录
            results.append({
                "process_id": data['process_id'],
                "image_url": data['image_url'],
                "CoT": "",
                "predicted_report": "",
                "reference_report": data['reference_report'],
                "metrics": {**{f'bleu-{n}': 0.0 for n in range(1, 5)}, 'rouge-l': 0.0, 'meteor': 0.0},
                "error": str(e)
            })
    
    # 计算平均指标
    num_samples = len(results)
    final_metrics = {
        k: round(v / processed_count, 4) if processed_count > 0 else 0.0 
        for k, v in metrics_accumulator.items()
    }
    
    # 保存详细结果和全局指标
    output = {
        "model_info": {
            "model_name": model_name,
            "api_url": api_url,
            "total_samples": num_samples,
            "processed_samples": processed_count
        },
        "detailed_results": results,
        "average_metrics": final_metrics
    }
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    
    # 打印结果摘要
    print("\n=== 评估完成 ===")
    print(f"模型: {model_name}")
    print(f"总样本数: {num_samples}")
    print(f"成功处理样本数: {processed_count}")
    print("\n平均指标:")
    for metric, score in final_metrics.items():
        print(f"{metric.upper():<10}: {score:.4f}")
    print(f"\n结果已保存到: {output_path}")

if __name__ == "__main__":
    # 配置参数
    TEST_PATH = "/data0/zhuoxu/yihong/code/LLaMA-Factory-main/eval_data_dir/IU_X_ray_test_cleaned.json"  # IU-xray测试数据路径
    MODEL_NAME = "/data0/zhuoxu/yihong/code/Qwen2.5-VL-32B-Instruct"  # 根据实际部署的模型名称调整
    API_URL = "http://127.0.0.1:8001/v1"  # VLLM服务地址
    OUTPUT_PATH = "/data0/zhuoxu/yihong/code/LLaMA-Factory-main/eval_result/Qwen2.5-VL-32B-Base-IU-xray.json"
    
    # 运行评估
    main(
        test_path=TEST_PATH,
        model_name=MODEL_NAME,
        api_url=API_URL,
        output_path=OUTPUT_PATH
    )