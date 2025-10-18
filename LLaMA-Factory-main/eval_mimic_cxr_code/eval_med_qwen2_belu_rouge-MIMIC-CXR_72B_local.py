#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地直接加载72B模型进行医学影像报告评估
使用transformers库直接加载模型，避免VLLM API的网络延迟
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import json
import nltk
import torch
import re
from tqdm import tqdm
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor,Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from PIL import Image

# 确保NLTK资源就绪
# try:
#     nltk.data.find('/data0/zhuoxu/yihong/code/tokenizers/punkt')
# except:
#     nltk.download('punkt', quiet=True)

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

class LocalReportGenerator:
    """
    本地加载的医学报告生成器
    """
    
    def __init__(self, model_path, device="auto"):
        """
        初始化本地报告生成器
        
        Args:
            model_path (str): 模型路径
            device (str): 设备类型，"auto"表示自动选择
        """
        self.model_path = model_path
        self.device = device
        
        print(f"正在加载模型: {model_path}")
        print("这可能需要几分钟时间...")
        
        # 加载模型和处理器
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # 加载处理器
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        print(f"模型加载完成，设备: {self.model.device}")

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
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {
                        "type": "text", 
                        "text": "Based on this medical X-ray image, please generate a diagnostic report Output the thinking process in <think> </think>, and output the final findings and impression within the <answer> </answer>. Following \"<think></think>\n<answer></answer> \" format."
                    }
                ]
            }]
            
            # 处理输入
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            # 移动到正确的设备
            inputs = inputs.to(self.model.device)
            
            # 生成响应
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=8000
                )
            
            # 解码生成的文本
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            return output_text[0] if output_text else ""
            
        except Exception as e:
            print(f"生成报告时出错 {image_path}: {str(e)}")
            return ""

def calculate_metrics(prediction, reference):
    """
    计算BLEU1-4和ROUGE-L指标
    
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
    
    # 返回所有指标
    return {**{k: round(v, 4) for k, v in bleu_scores.items()}, 'rouge-l': round(rouge_l, 4)}

def main(test_path, model_path, device="auto",
         output_path="/data0/zhuoxu/yihong/code/LLaMA-Factory-main/eval_result/Qwen2.5-VL-72B-Local-MIMIC-CXR_only_F-I.json"):
    """
    主评估函数
    
    Args:
        test_path (str): 测试数据路径
        model_path (str): 模型路径
        device (str): 设备类型
        output_path (str): 结果输出路径
    """
    # 初始化生成器
    generator = LocalReportGenerator(model_path, device=device)
    
    # 加载测试数据
    print(f"加载测试数据: {test_path}")
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    # test_data = test_data[:3]  # 测试时可以取少量样本
    print(f"共有 {len(test_data)} 个测试样本")
    
    # 处理所有样本
    results = []
    metrics_accumulator = {f'bleu-{n}': 0.0 for n in range(1, 5)}
    metrics_accumulator['rouge-l'] = 0.0
    
    for data in tqdm(test_data, desc="处理样本"):
        try:
            # 生成报告
            prediction = generator.generate(data['image_path'])
            prediction_answer = extract_answer_content(prediction)
            print("prediction_answer:", prediction_answer)
            
            # 计算指标
            metrics = calculate_metrics(prediction_answer, data['report'])
            
            # 记录结果
            results.append({
                "subject_id": data['subject_id'],
                "study_id": data['study_id'],
                "dicom_id": data['dicom_id'],
                "image_path": data['image_path'],
                "CoT": prediction,
                "predicted_report": prediction_answer,
                "reference_report": data['report'],
                "metrics": metrics
            })
            
            print(f'样本指标: {metrics}')
            
            # 累加指标
            for k in metrics_accumulator:
                metrics_accumulator[k] += metrics[k]
                
        except Exception as e:
            print(f"处理样本时出错 {data['image_path']}: {str(e)}")
            # 添加失败记录
            results.append({
                "subject_id": data['subject_id'],
                "study_id": data['study_id'],
                "dicom_id": data['dicom_id'],
                "image_path": data['image_path'],
                "predicted_report": "",
                "reference_report": data['report'],
                "metrics": {**{f'bleu-{n}': 0.0 for n in range(1, 5)}, 'rouge-l': 0.0},
                "error": str(e)
            })
    
    # 计算平均指标
    num_samples = len(results)
    final_metrics = {
        k: round(v / num_samples, 4) if num_samples > 0 else 0.0 
        for k, v in metrics_accumulator.items()
    }
    
    # 保存详细结果和全局指标
    output = {
        "model_info": {
            "model_path": model_path,
            "device": str(generator.model.device),
            "total_samples": num_samples
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
    print(f"模型: {model_path}")
    print(f"设备: {generator.model.device}")
    print(f"总样本数: {num_samples}")
    print("\n平均指标:")
    for metric, score in final_metrics.items():
        print(f"{metric.upper():<10}: {score:.4f}")
    print(f"\n结果已保存到: {output_path}")

if __name__ == "__main__":
    # 配置参数
    TEST_PATH = "/data0/zhuoxu/yihong/code/LLaMA-Factory-main/eval_data_dir/Med_mimic_cxr_only_ref_report_test_only_F-I.json"
    MODEL_PATH = "/data0/zhuoxu/yihong/code/Qwen2.5-VL-72B-Instruct-AWQ"  # 本地模型路径
    DEVICE = "auto"  # 可以设置为 "cuda:0", "cuda:1" 等指定GPU
    OUTPUT_PATH = "/data0/zhuoxu/yihong/code/LLaMA-Factory-main/eval_result/Qwen2.5-VL-72B-Local-MIMIC-CXR_only_F-I.json"
    
    # 运行评估
    main(
        test_path=TEST_PATH,
        model_path=MODEL_PATH,
        device=DEVICE,
        output_path=OUTPUT_PATH
    )