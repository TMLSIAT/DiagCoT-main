import json
from tqdm import tqdm
import argparse
import os
import re
from openai import OpenAI
from retrying import retry
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

class CoTValidator:
    def __init__(self, args):
        self.gpt = GPT(
            model_name=args.model_name,
            api_key=args.api_key,
            api_url=args.api_url
        )
        self.input_path = args.input_path
        self.output_path = args.output_path

    def _construct_messages(self, combined_text, reference):
        system_prompt = (
            "作为放射科质控专家，请严格审核生成报告是否符合标准，主要看思考过程对不对，这个很重要！仅返回true或false，不要任何解释。判断标准：\n"
            "1. **解剖定位**：病变器官/部位是否一致（如肺叶、胸膜）\n"
            "2. **病理特征**：病变类型（积液/实变/气胸）及程度（轻度/中度）是否匹配\n"
            "3. **关键发现**：必须包含所有阳性发现（如心脏扩大）和重要阴性指标（如无气胸）\n"
            "4. **鉴别诊断**：必须体现相同的鉴别考虑（如'需排除肺炎'）\n"
            "5. **临床关联**：是否结合正确临床推理（如心衰导致肺水肿）\n"
            "处理原则：\n"
            "- 允许术语变体（如'cardiomegaly'与'enlarged heart'）\n"
            "- 忽略非关键描述差异（如'轻度' vs '轻度增加'）\n"
            "- 严格处理部位错误（如将'下叶'写成'上叶'）"
        )
        
        # 修正后的消息格式
        return [{
            "role": "system",
            "content": [
                {
                    "type": "text",  # 必须显式声明类型
                    "text": system_prompt
                }
            ]
        }, {
            "role": "user",
            "content": [
                {
                    "type": "text",  # 必须显式声明类型
                    "text": f"生成的文本：\n{combined_text}\n\n参考报告：\n{reference}"
                }
            ]
        }]

    def _parse_response(self, response):
        # 使用正则表达式严格匹配响应
        match = re.search(r'\b(true|false)\b', response, re.IGNORECASE)
        if match:
            return match.group().lower() == 'true'
        return False  # 无法解析的响应视为不匹配

    def process_entry(self, entry):
        try:
            # 拼接Complex_CoT和Response
            combined_text = f"Complex_CoT:{entry['Complex_CoT']}\nResponse:{entry['Response']}"
            
            # 构造prompt
            messages = self._construct_messages(
                combined_text, 
                entry['reference_report']
            )
            
            # 调用模型
            response = self.gpt.retry_call(messages)
            
            # 解析结果
            return self._parse_response(response)
        except Exception as e:
            print(f"处理 {entry['process_id']} 时出错: {str(e)}")
            return False

    def run(self):
        # 读取输入文件
        with open(self.input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 处理每个条目
        for entry in tqdm(data, desc="审核进度"):
            entry['is_consistent'] = self.process_entry(entry)
            print(entry['is_consistent'])
            # # 添加图片校验标记（根据实际需要）
            # entry['image_valid'] = self.validate_image(entry['image_url'])

        # 保存结果
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def validate_image(self, image_url):
        # 预留图片校验接口（根据实际需求实现）
        return True

class GPT:
    def __init__(self, model_name="qwen-vl-plus", api_key=None, api_url="http://localhost:8000/v1"):  # 默认改为本地地址
        self.model_name = model_name
        self.api_key = api_key if api_key else os.getenv("DASHSCOPE_API_KEY")
        self.api_url = api_url
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_url,
        )

    @retry(wait_fixed=3000, stop_max_attempt_number=3)
    def retry_call(self, messages):
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0,  # 确保最大确定性
            max_tokens=20
        ).choices[0].message.content

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/data0/zhuoxu/yihong/code/Test_CoT_Med_MLLM/output_data_new/Latest_result/multi_MIMIC_CXR_random_5000_2_xray_train_CoT_use_local_Qwen_VL_32B_Instruct_Lora_merger_base_result.json')
    parser.add_argument('--output_path', type=str, default='/data0/zhuoxu/yihong/code/Test_CoT_Med_MLLM/extract_cot_use_72B/clean_multi_MIMIC_CXR_random_5000_2_xray_train_CoT_use_local_Qwen_VL_32B_Instruct_Lora_merger_base_result.json')
    parser.add_argument('--model_name', type=str, default="DiagCoT-main/Qwen2.5-VL-72B-Instruct-AWQ", help='模型名称')
    parser.add_argument("--api_key", type=str, default="qwen-abc123", help="DashScope API key.")
    parser.add_argument("--api_url", type=str, default="http://127.0.0.1:8000/v1", help="DashScope API URL.")
    
    args = parser.parse_args()
    
    validator = CoTValidator(args)
    validator.run()