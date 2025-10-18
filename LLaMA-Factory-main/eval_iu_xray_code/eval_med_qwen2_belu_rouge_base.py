import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import torch
import nltk
from tqdm import tqdm
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor,Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import base64
import re
# 确保NLTK资源就绪
try:
    nltk.data.find('/data0/zhuoxu/yihong/code/tokenizers/punkt')
except:
    nltk.download('punkt', quiet=True)

class ReportGenerator:
    def __init__(self, model_path, device="cuda:0"):
        self.device = torch.device(device)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        print(f"Loaded Qwen2-VL model from {model_path}")

    def generate(self, image_path):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "Based on this medical X-ray image, please generate a diagnostic report."} #Details: Generate diagnostic report. Output the thinking process in <think> . Following \"<think> thinking process \n<answer> diagnostic report </answer>)\" format.
            ]
        }]
        #"Based on this medical X-ray image, please analyze and generate a diagnostic report."
        # "Based on this medical X-ray image, please analyze and generate a diagnostic report Output the thinking process in <think> . Following \"<think> thinking process \n<answer> diagnostic report </answer>)\" format."} #Details: Generate diagnostic report. Output the thinking process in <think> . Following \"<think> thinking process \n<answer> diagnostic report </answer>)\" format.
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=1024) #max_new_tokens=2048,use_cache=True
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0] if output_text else ""

def calculate_metrics(prediction, reference):
    """计算BLEU1-4和ROUGE-L指标"""
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
    
    # 修正后的返回语句（重点检查以下部分）
    return {**{k: round(v, 4) for k, v in bleu_scores.items()}, 'rouge-l': round(rouge_l, 4)}
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

def main(test_path, model_path, output_path="/data0/zhuoxu/yihong/code/LLaMA-Factory-main/eval_result_filter/train_Med_Qwen_2_VL_7B_Proj_stage1_warmup_only_F-I_filter_rft_num624.json"):
    # 初始化生成器
    generator = ReportGenerator(model_path)
    
    # 加载测试数据
    with open(test_path) as f:
        test_data = json.load(f)
    
    # 处理所有样本
    results = []
    metrics_accumulator = {f'bleu-{n}': 0.0 for n in range(1, 5)}
    metrics_accumulator['rouge-l'] = 0.0
    
    for data in tqdm(test_data, desc="Processing samples"):
        try:
            # 生成报告
            prediction = generator.generate(data['image_url'])

            # prediction = extract_answer_content(prediction)

            # 计算指标
            metrics = calculate_metrics(prediction, data['reference_report'])
            
            # 记录结果
            results.append({
                "process_id": data['process_id'],
                "image_url": data['image_url'],
                "predicted_report": prediction,
                "reference_report": data['reference_report'],
                "metrics": metrics
            })
            print('metrics:',metrics)
            # 累加指标
            for k in metrics_accumulator:
                metrics_accumulator[k] += metrics[k]
                
        except Exception as e:
            print(f"Error processing {data['image_url']}: {str(e)}")
    
    # 计算平均指标
    num_samples = len(results)
    final_metrics = {
        k: round(v / num_samples, 4) if num_samples > 0 else 0.0 
        for k, v in metrics_accumulator.items()
    }
    
    # 保存详细结果和全局指标
    output = {
        "detailed_results": results,
        "average_metrics": final_metrics
    }
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)
    
    print("\nAverage Metrics:")
    for metric, score in final_metrics.items():
        print(f"{metric.upper():<8}: {score:.4f}")
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main(
        test_path="/data0/zhuoxu/yihong/code/LLaMA-Factory-main/eval_data_dir/Qwen2-VL-7B-Stage3-RFT-train-merger-llm-IU-xray-bleu-rouge-wo-think-pormpt-temp1_5000_1_step200-filtered-bleu1-ge-0.3.json",
        model_path="/data0/zhuoxu/yihong/code/LLaMA-Factory-main/saves/Qwen2-VL-7B-Instruct/full/train_Med_Qwen_2_VL_7B_Proj_stage1_warmup_only_F-I"
    )