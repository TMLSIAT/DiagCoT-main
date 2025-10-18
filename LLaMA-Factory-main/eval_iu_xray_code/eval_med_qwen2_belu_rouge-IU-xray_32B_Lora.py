import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import torch
import nltk
from tqdm import tqdm
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import base64
import re

from peft import PeftModel, LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

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

class ReportGenerator:
    """32B LoRA模型的医学报告生成器，适配IU-xray数据集"""
    
    def __init__(self, base_model_path, lora_path=None, device="cuda:0"):
        self.device = torch.device(device)
        
        # 严格匹配训练参数的LoRA配置
        lora_config = LoraConfig(
            r=8,  # lora_rank=8
            lora_alpha=16,  # lora_alpha=16
            # target_modules根据实际训练配置调整
            # lora_dropout=0.0,
            # bias="none",
            # task_type="CAUSAL_LM",
        )
        
        # 精确加载32B基础模型
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,  # 匹配bf16=true
            attn_implementation="flash_attention_2",  # 对应flash_attn=fa2
            device_map="auto",
            # trust_remote_code=True
        )
        
        # 加载LoRA适配器
        if lora_path:
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_path,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
        else:
            self.model = get_peft_model(self.model, lora_config)
        self.model = self.model.eval()
        
        # 多模态处理器（匹配qwen2_vl模板）
        self.processor = AutoProcessor.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            padding_side="left"  # 匹配训练设置
        )
        print(f"已加载32B模型，配置IU-xray优化的LoRA适配器")

    def generate(self, image_path):
        """生成IU-xray诊断报告"""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type":"text", "text": "Based on this medical X-ray image, please generate a diagnostic report."} # # {"type": "text", "text": "Based on this medical X-ray image, please analyze and generate a diagnostic report. Output the thinking process in <think> </think>, and output the final findings and impression within the <answer> </answer>. Following \"<think></think>\n<answer></answer> \" format."}
            ]
        }]
        
        # 严格复现训练数据预处理
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            padding_side="left"  # 与训练数据对齐
            # chat_template="qwen2_vl"  # 显式传入chat_template参数，解决No chat template错误
        )
        
        # 医学影像特殊处理
        image_inputs, _ = process_vision_info(messages)
        
        # 构建与训练一致的输入格式
        inputs = self.processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            padding=True,
            truncation=True,
            max_length=16000,
            return_tensors="pt"
        ).to(self.device)

        # 匹配训练时的生成参数
        generation_config = {
            "max_new_tokens": 4096,
            # "do_sample": True,
            # "temperature": 0.7,
            # "top_p": 0.9,
            # "repetition_penalty": 1.1,
            "pad_token_id": self.processor.tokenizer.eos_token_id,
            "eos_token_id": [self.processor.tokenizer.eos_token_id, 151645]  # 医学文本终止符
        }
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_config
            )
        
        # 严格匹配训练解码设置
        decoded_text = self.processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
            padding_side="left"  # 对齐训练设置
        )
        return decoded_text[0]

def calculate_metrics(prediction, reference):
    """计算BLEU1-4、ROUGE-L和METEOR指标"""
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

def main(test_path, model_path, lora_path=None, output_path="/data0/zhuoxu/yihong/code/LLaMA-Factory-main/eval_result/Qwen2.5-VL-32B-IU-xray-Lora.json"):
    """主评估函数"""
    # 初始化生成器
    generator = ReportGenerator(model_path, lora_path)
    
    # 加载测试数据
    print(f"加载IU-xray测试数据: {test_path}")
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"共有 {len(test_data)} 个测试样本")
    
    # 处理所有样本
    results = []
    metrics_accumulator = {f'bleu-{n}': 0.0 for n in range(1, 5)}
    metrics_accumulator['rouge-l'] = 0.0
    metrics_accumulator['meteor'] = 0.0
    processed_count = 0  # 成功处理的样本计数
    
    for data in tqdm(test_data, desc="处理IU-xray样本"):
        try:
            # 生成报告
            prediction = generator.generate(data['image_url'])
            # prediction_answer = extract_answer_content(prediction)
            
            # 计算指标
            metrics = calculate_metrics(prediction_answer, data['reference_report'])
            
            # 记录结果
            results.append({
                "process_id": data['process_id'],
                "image_url": data['image_url'],
                # "CoT": prediction,
                "predicted_report": prediction,
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
            "base_model_path": model_path,
            "lora_path": lora_path,
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
    print("\n=== IU-xray评估完成 ===")
    print(f"基础模型: {model_path}")
    print(f"LoRA路径: {lora_path}")
    print(f"总样本数: {num_samples}")
    print(f"成功处理样本数: {processed_count}")
    print("\n平均指标:")
    for metric, score in final_metrics.items():
        print(f"{metric.upper():<10}: {score:.4f}")
    print(f"\n结果已保存到: {output_path}")

if __name__ == "__main__":
    # 配置参数
    TEST_PATH = "/data0/zhuoxu/yihong/code/LLaMA-Factory-main/eval_data_dir/IU_X_ray_test_cleaned.json"  # IU-xray测试数据路径
    BASE_MODEL_PATH = "/data0/zhuoxu/yihong/code/Qwen2.5-VL-32B-Instruct"  # 32B基础模型路径
    LORA_PATH = "/data0/zhuoxu/yihong/code/LLaMA-Factory-main/saves/Qwen2.5-VL-7B-Instruct/lora/tran_Qwen2.5_vl_32B_lora"  # LoRA适配器路径
    OUTPUT_PATH = "/data0/zhuoxu/yihong/code/LLaMA-Factory-main/eval_result/Qwen2.5-VL-32B-IU-xray-Lora.json"
    
    # 运行评估
    main(
        test_path=TEST_PATH,
        model_path=BASE_MODEL_PATH,
        lora_path=LORA_PATH,
        output_path=OUTPUT_PATH
    )