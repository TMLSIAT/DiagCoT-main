import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

from peft import PeftModel, LoraConfig, get_peft_model
# # 确保NLTK资源就绪
# try:
#     nltk.data.find('/data0/zhuoxu/yihong/code/tokenizers/punkt')
# except:
#     nltk.download('punkt', quiet=True)

# class ReportGenerator:
#     def __init__(self, model_path, device="cuda:0"):
#         self.device = torch.device(device)
#         self.model = Qwen2VLForConditionalGeneration.from_pretrained(
#             model_path,
#             torch_dtype=torch.bfloat16,
#             attn_implementation="flash_attention_2",
#             device_map="auto"
#         ).eval()
#         self.processor = AutoProcessor.from_pretrained(model_path)
#         print(f"Loaded Qwen2-VL model from {model_path}")

#     def generate(self, image_path):
#         messages = [{
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": image_path},
#                 {"type": "text", "text": "Based on this medical X-ray image, please generate a diagnostic report Output the thinking process in <think> </think>, and output the final findings and impression within the <answer> </answer>. Following \"<think></think>\n<answer></answer> \" format."} #Details: Generate diagnostic report. Output the thinking process in <think> . Following \"<think> thinking process \n<answer> diagnostic report </answer>)\" format.          Based on this medical X-ray image, please analyze and generate a diagnostic report.
#             ]
#         }]
        
#         text = self.processor.apply_chat_template(
#             messages, tokenize=False, add_generation_prompt=True
#         )
#         image_inputs, _ = process_vision_info(messages)
        
#         inputs = self.processor(
#             text=[text],
#             images=image_inputs,
#             padding=True,
#             return_tensors="pt",
#         ).to(self.device)

#         with torch.no_grad():
#             outputs = self.model.generate(**inputs, max_new_tokens=12000,temperature=0.5, do_sample=True,top_p=0.9) # temperature=0.1,do_sample=True 要同时开 ,   temperature=0.3, do_sample=True,top_p=0.9 | use_cache=True RFT的方法要用
#         generated_ids_trimmed = [
#             out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
#         ]
#         output_text = self.processor.batch_decode(
#             generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#         )
#         return output_text[0] if output_text else ""


class ReportGenerator:
    def __init__(self, base_model_path, lora_path=None, device="cuda:0"):
        self.device = torch.device(device)
        
        # 严格匹配训练参数
        lora_config = LoraConfig(
            r=8,  # lora_rank=8
            lora_alpha=16,  # lora_alpha=16
            # target_modules=[  # 根据all的设定覆盖所有可训练层
            #     "q_proj", "k_proj", "v_proj", "o_proj",
            #     "gate_proj", "up_proj", "down_proj",
            #     "vision_model.encoder.layers.0.*",  # 包含视觉层
            #     "language_model.model.layers.0.*"   # 包含语言层
            # ],
            # lora_dropout=0.0,  # 对应lora_dropout=0
            # bias="none",
            # task_type="CAUSAL_LM",
            # modules_to_save=["lm_head", "vision_proj"]  # 适配Med_mimic_cxr任务
        )
        
        # 精确加载基础模型
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,  # 匹配bf16=true
            attn_implementation="flash_attention_2",  # 对应flash_attn=fa2
            device_map="auto",
            # trust_remote_code=True  # 对应trust_remote_code=true
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
            # padding_side="left"  # 匹配训练设置
        )
        print(f"Loaded model with Med_mimic_cxr optimized LoRA")

    def generate(self, image_path):  # 匹配cutoff_len=4096
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "Based on this medical X-ray image, please generate a diagnostic report Output the thinking process in <think> </think>, and output the final findings and impression within the <answer> </answer>. Following \"<think></think>\n<answer></answer> \" format."} #Details: Generate diagnostic report. Output the thinking process in <think> . Following \"<think> thinking process \n<answer> diagnostic report </answer>)\" format.          Based on this medical X-ray image, please analyze and generate a diagnostic report.
            ]
        }]
        # 严格复现训练数据预处理
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
            # padding_side="left"  # 与训练数据对齐
        )
        
        # 医学影像特殊处理
        image_inputs, _ = process_vision_info(messages)
        
        # 构建与训练一致的输入格式
        inputs = self.processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            padding=True,
            truncation=True,
            max_length=10000,
            return_tensors="pt"
        ).to(self.device)

        # 匹配训练时的生成参数
        generation_config = {
            "max_new_tokens": 5000,
            "do_sample": True,
            "temperature": 0.3,  # 推荐与训练时验证集参数一致
            "top_p": 0.9,
            "use_cache": True,  # 启用KV缓存
            # "repetition_penalty": 1.1,
            # "pad_token_id": self.processor.tokenizer.eos_token_id,
            # "eos_token_id": [self.processor.tokenizer.eos_token_id, 151645]  # 医学文本终止符
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
            clean_up_tokenization_spaces=False
            # padding_side="left"  # 对齐训练设置
        )
        return decoded_text[0]

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

def main(test_path, model_path, output_path="/data0/zhuoxu/yihong/code/LLaMA-Factory-main/eval_result/Qwen2-VL-7B-MIMIC-CXR-Stage2-lora_use_32B_CoT_new_cutoff_0.5w_temp0-3.json"):
    temp_path = "/data0/zhuoxu/yihong/code/LLaMA-Factory-main/temp_eval_result/Qwen2-VL-7B-MIMIC-CXR-Stage2-lora_use_32B_CoT_new_cutoff_0.5w_temp0-3.json"
    # 初始化生成器
    lora_path = "/data0/zhuoxu/yihong/code/LLaMA-Factory-main/saves/Qwen2-VL-7B-Instruct/lora/train_stage2_use_32B_MIMIC_CXR_new_cutoff_lora"
    generator = ReportGenerator(model_path,lora_path)
    
    # 加载测试数据
    with open(test_path) as f:
        test_data = json.load(f)
    
    # 处理所有样本
    results = []
    temp_re = {}
    metrics_accumulator = {f'bleu-{n}': 0.0 for n in range(1, 5)}
    metrics_accumulator['rouge-l'] = 0.0
    
    for data in tqdm(test_data, desc="Processing samples"):
        try:
            # 生成报告
            prediction = generator.generate(data['image_path'])
            # print(prediction)
            prediction = extract_answer_content(prediction)

            # 计算指标
            metrics = calculate_metrics(prediction, data['report'])
            
            # 记录结果
            results.append({
                # "process_id": data['process_id'],
                "subject_id": data['subject_id'],
                "study_id": data['study_id'],
                "image_path": data['image_path'],
                "predicted_report": prediction,
                "reference_report": data['report'],
                "metrics": metrics
            })
            print('metrics:',metrics)
            # 临时存放
            temp_re = {
                "subject_id": data['subject_id'],
                "study_id": data['study_id'],
                "image_path": data['image_path'],
                # "predicted_report": prediction,
                "metrics": metrics
            }
            with open(temp_path, "a") as f:
                json.dump(temp_re, f, indent=4)

            # 累加指标
            for k in metrics_accumulator:
                metrics_accumulator[k] += metrics[k]
                
        except Exception as e:
            print(f"Error processing {data['image_path']}: {str(e)}")
            
    
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
        test_path="/data0/zhuoxu/yihong/code/LLaMA-Factory-main/eval_data_dir/Med_mimic_cxr_only_ref_report_test_only_F-I.json",
        # model_path="/data0/zhuoxu/yihong/code/LLaMA-Factory-main/saves/Qwen2-VL-7B-Instruct/full/train_Med_Qwen_2_VL_7B_Proj_stage1_warmup_only_F-I"
        model_path="/data0/zhuoxu/yihong/code/LLaMA-Factory-main/saves/Qwen2-VL-7B-Instruct/full/train_Med_Qwen_2_VL_7B_Proj_stage1_warmup_only_F-I"
    )