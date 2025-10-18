import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import traceback
from openai import OpenAI
# from tenacity import retry, wait_fixed, stop_after_attempt
from retrying import retry
import base64
import re
import random
import json
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import random
import requests
from retrying import retry
import argparse
import re
import traceback
import copy
import sys
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# GPT 类调整为 Qwen-vl API
class GPT:
    def __init__(self, model_name="qwen-vl-plus", api_key=None, api_url="https://dashscope.aliyuncs.com/compatible-mode/v1"):
        self.model_name = model_name
        self.api_key = api_key if api_key else os.getenv("DASHSCOPE_API_KEY")
        self.api_url = api_url
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_url,
        )
        print(f"Using model: {self.model_name}")

    def call(self, messages, additional_args={}):
        payload = {
            "model": self.model_name,
            "messages": messages,
            **additional_args,
        }
        completion = self.client.chat.completions.create(**payload)
        return completion.choices[0].message.content

    @retry(wait_fixed=3000, stop_max_attempt_number=3)
    def retry_call(self, messages, additional_args={"max_tokens": 8192}):
        return self.call(messages, additional_args)

# 医学相关的初始提示
query_prompt_init = """<question>
Generate a corresponding medical report based on this X-ray image.
</question>

<reference_report>
{}
</reference_report>

Please refer to the reference report I provided and generate an appropriate thought process. In addition, Please respond using the **Chain of Thought (CoT) reasoning method**. Your reasoning should consist of multiple steps, each containing the following three types of actions:

- **"Inner Thinking"**: Perform a detailed analysis. Gradually examine the X-ray image, including (but not limited to) image quality, anatomical structures, abnormal radiographic features, and potential disease indications. Each step should have a brief title.
- **"Final Conclusion"**: Summarize the correct reasoning from all previous "Inner Thinking" steps and provide the final X-ray diagnosis report. No title is needed.
- **"Verification"**: Verify the conclusion from the "Final Conclusion" step. If the conclusion is correct, end the reasoning process. If not, return to "Inner Thinking" for further analysis. No title is needed.

### **Your response must strictly follow the JSON format below:**
```json
{{
"CoT": [
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
"""

# 医学相关的验证提示
verify_prompt = """<Model Response>  
{}  
</Model Response>  

<Reference Report>  
{}  
</Reference Report>  

Compare the model-generated report with the reference report. Pay special attention to the Findings and Impression. Output "True" only if:  
1. All key findings (e.g., lung opacities, fractures) match.  
2. The clinical impression in the "Findings" and "Impression" section is identical to the reference.  
Otherwise output "False"."""
# 回溯策略提示（医学相关）
gen_prompt_rethink_Backtracking = """<question>
Generate a corresponding medical report based on this X-ray image.
</question>

<reference_report>  
{}
</reference_report>

<previous reasoning>
{}  
</previous reasoning>

<response requirements>
Please refer to the reference report I provided and generate an appropriate thought process. Your response must include the following steps, each composed of three types of actions: **"Inner Thinking"**, **"Final Conclusion"**, and **"Verification"**:

1. **Inner Thinking**: Break down the reasoning process into multiple concise steps. Each step should start with a brief title to clarify its purpose.
2. **Final Conclusion**: Summarize the correct reasoning from all previous 'Inner Thinking' steps and provide the final X-ray diagnosis report. No title is needed for this section.
3. **Verification**: Verify the accuracy of the "Final Conclusion". If it holds, conclude the process. Otherwise, return to "Inner Thinking" for further refinement.

</response requirements>

<question> represents the task to generate a medical report based on the X-ray image, and <previous reasoning> contains your prior reasoning. Your task is to continue from the current 'Verification' step. I have manually reviewed the reasoning and determined that the **Final Conclusion** is false. Your 'Verification' results must align with mine. Proceed to refine the reasoning using **backtracking** to revisit earlier points of reasoning and construct a new Final Conclusion.

### Output Format
Strictly follow the JSON structure below. You do not need to repeat your previous reasoning. Begin directly from the next 'Verification' stage.

```json
{{
"CoT": [
    {{"action": "Verification", "content": "..."}},
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
"""
gen_prompt_rethink_Exploring_New_Path = """<question>
Generate a corresponding medical report based on this X-ray image.
</question>

<reference_report>  
{}
</reference_report>

<previous reasoning>
{}  
</previous reasoning>

<response requirements>
Please refer to the reference report I provided and generate an appropriate thought process. Your response must include the following steps, each composed of three types of actions: **"Inner Thinking"**, **"Final Conclusion"**, and **"Verification"**:

1. **Inner Thinking**: Break down the reasoning process into multiple concise steps. Each step should start with a brief title to clarify its purpose.
2. **Final Conclusion**: Summarize the correct reasoning from all previous 'Inner Thinking' steps and provide the final X-ray diagnosis report. No title is needed for this section.
3. **Verification**: Verify the accuracy of the "Final Conclusion". If it holds, conclude the process. Otherwise, return to "Inner Thinking" for further refinement.

</response requirements>

<question> represents the task to generate a medical report based on the X-ray image, and <previous reasoning> contains your prior reasoning. Your task is to continue from the current 'Verification' step. I have manually reviewed the reasoning and determined that the **Final Conclusion** is false. Your 'Verification' results must align with mine. Proceed to refine the reasoning by exploring new approaches to analyzing the X-ray image and construct a new Final Conclusion.

### Output Format
Strictly follow the JSON structure below. You do not need to repeat your previous reasoning. Begin directly from the next 'Verification' stage.

```json
{{
"CoT": [
    {{"action": "Verification", "content": "..."}},
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""

gen_prompt_rethink_Verification = """<question>
Generate a corresponding medical report based on this X-ray image.
</question>

<reference_report>  
{}
</reference_report>

<previous reasoning>
{}  
</previous reasoning>

<response requirements>
Please refer to the reference report I provided and generate an appropriate thought process. Your response must include the following steps, each composed of three types of actions: **"Inner Thinking"**, **"Final Conclusion"**, and **"Verification"**:

1. **Inner Thinking**: Break down the reasoning process into multiple concise steps. Each step should start with a brief title to clarify its purpose.
2. **Final Conclusion**: Summarize the correct reasoning from all previous 'Inner Thinking' steps and provide the final X-ray diagnosis report. No title is needed for this section.
3. **Verification**: Verify the accuracy of the "Final Conclusion". If it holds, conclude the process. Otherwise, return to "Inner Thinking" for further refinement.

</response requirements>

<question> represents the task to generate a medical report based on the X-ray image, and <previous reasoning> contains your prior reasoning. Your task is to continue from the current 'Verification' step. I have manually reviewed the reasoning and determined that the **Final Conclusion** is false. Your 'Verification' results must align with mine. Proceed to refine the reasoning by conducting a thorough **validation** process to ensure the accuracy of your diagnosis and construct a new Final Conclusion.

### Output Format
Strictly follow the JSON structure below. You do not need to repeat your previous reasoning. Begin directly from the next 'Verification' stage.

```json
{{
"CoT": [
    {{"action": "Verification", "content": "..."}},
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""

gen_prompt_rethink_Correction = """<question>
Generate a corresponding medical report based on this X-ray image.
</question>

<reference_report>  
{}
</reference_report>

<previous reasoning>
{}  
</previous reasoning>

<response requirements>
Please refer to the reference report I provided and generate an appropriate thought process. Your response must include the following steps, each composed of three types of actions: **"Inner Thinking"**, **"Final Conclusion"**, and **"Verification"**:

1. **Inner Thinking**: Break down the reasoning process into multiple concise steps. Each step should start with a brief title to clarify its purpose.
2. **Final Conclusion**: Summarize the correct reasoning from all previous 'Inner Thinking' steps and provide the final X-ray diagnosis report. No title is needed for this section.
3. **Verification**: Verify the accuracy of the "Final Conclusion". If it holds, conclude the process. Otherwise, return to "Inner Thinking" for further refinement.

</response requirements>

<question> represents the task to generate a medical report based on the X-ray image, and <previous reasoning> contains your prior reasoning. Your task is to continue from the current 'Verification' step. I have manually reviewed the reasoning and determined that the **Final Conclusion** is false. Your 'Verification' results must align with mine. Proceed to refine the reasoning by making precise **corrections** to address prior flaws in your analysis and construct a new Final Conclusion.

### Output Format
Strictly follow the JSON structure below. You do not need to repeat your previous reasoning. Begin directly from the next 'Verification' stage.

```json
{{
"CoT": [
    {{"action": "Verification", "content": "..."}},
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""

gen_prompt_w_label = """<question>
Generate a corresponding medical report based on this X-ray image.
</question>

<previous reasoning>
{}  
</previous reasoning>

<response requirements>
Please refer to the reference report I provided and generate an appropriate thought process. Your response must include the following steps, each composed of three types of actions: **"Inner Thinking"**, **"Final Conclusion"**, and **"Verification"**:

1. **Inner Thinking**: Break down the reasoning process into multiple concise steps. Each step should start with a brief title to clarify its purpose.
2. **Final Conclusion**: Summarize the correct reasoning from all previous 'Inner Thinking' steps and provide the final X-ray diagnosis report. No title is needed for this section.
3. **Verification**: Verify the accuracy of the "Final Conclusion". If it holds, conclude the process. Otherwise, return to "Inner Thinking" for further refinement.

</response requirements>

<question> represents the task to generate a medical report based on the X-ray image, and <previous reasoning> contains your prior reasoning. Your task is to continue from the current 'Verification' step. Now, I'll tell you that the correct diagnosis is "{}",please reorganize your thought process based on the summary reference report to generate a final impression that matches the reference report. Your 'Verification' requires careful consideration, and if incorrect, you need to provide new Inner Thinking steps and a new Final Conclusion to ensure the final diagnosis aligns with the correct one.

### Output Format
Strictly follow the JSON structure below. You do not need to repeat your previous reasoning. Begin directly from the next 'Verification' stage.

```json
{{
"CoT": [
    {{"action": "Verification", "content": "..."}},
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""

reformat_to_complex_cot_prompt = """<Thought Process>
{}
</Thought Process>

<Question>
Generate a corresponding medical report based on this X-ray image.
</Question>

The <Thought Process> above reflects the model's reasoning based on the <Question>. Your task is to rewrite the <Thought Process> to resemble a more human-like, intuitive natural thinking process for medical diagnosis. The new version should:

1. Be presented as step-by-step reasoning, with each thought on a new line separated by a line break.
2. Avoid structured titles or formatting, focusing on natural transitions. Use casual and natural language for transitions or validations, such as "hmm," "oh," "also," or "wait."
3. Expand the content, making the reasoning richer, more detailed, and logically clear while still being conversational and intuitive, as if a doctor is explaining their thought process.

Return directly the revised natural thinking in JSON format as follows:
```json
{{
  "NaturalReasoning": "..."
}}
```"""

# get_final_response_prompt = """<Internal Thinking>
# {}
# </Internal Thinking>

# <Question>
# Generate a corresponding medical report based on this X-ray image.
# </Question>

# The <Internal Thinking> represents your internal thoughts about the <Question>. Based on this, generate a rich and high-quality final medical report for the user. Ensure your final report closely follows the <Question> and is presented in a professional medical tone. Output only your final report, without any additional content."""
# get_final_response_prompt = """<Internal Thinking>
# {}
# </Internal Thinking>

# <reference_report>
# {}
# </reference_report>

# <Question>
# Generate a corresponding medical report based on this X-ray image.
# </Question>

# Your task is to generate a medical report that strictly follows these rules:
# 1. Structure the report into: **Findings** and **Impression** sections.
# 2. In Findings, describe abnormalities using terms from the reference report.
# 3. Please strictly copy the content from <reference_report></reference_report> verbatim into the **Impression**! Do not add any extra words!.
# 4. Use professional medical language but avoid markdown formatting.

# Output only the final report in this exact format:
# **Findings**
# - [Detailed observations]

# **Impression**
# - [copy from reference report]"""
get_final_response_prompt = """<Internal Thinking>
{}
</Internal Thinking>

<reference_report>
{}
</reference_report>

<Question>
Generate a corresponding medical report based on this X-ray image.
</Question>

Your task is to generate a medical report that strictly follows these rules:
1. Structure the report into: **Findings** and **Impression** sections.
2. In Findings, describe abnormalities using terms from the reference report.
3. Please strictly copy the content from <reference_report></reference_report> verbatim into the **Impression**! Do not add any extra words!.
4. Use professional medical language but avoid markdown formatting.

Output only the final report in this exact format:
Findings : [Detailed observations] Impression : [copy from reference report]"""
# search strategies
search_strategies = [('Backtracking',gen_prompt_rethink_Backtracking),('Exploring New Paths',gen_prompt_rethink_Exploring_New_Path),('Verification',gen_prompt_rethink_Verification),('Correction',gen_prompt_rethink_Correction)]

def extract_bracket_content(text):
        # Extract content between the first '{' and the last '}'
        match = re.search(r'\{.*\}', text, re.DOTALL)
        return match.group(0) if match else None
# 解析 GPT 响应
def parse_gpt_response(response):
    try:
        if '{' != response[0]:
            response = extract_bracket_content(response)
        da = json.loads(response.replace('\n', ''))
        assert isinstance(da["CoT"], list), "CoT should be a list"
        assert da['CoT'][-3]['action'] == 'Inner Thinking', 'Inner Thinking should be the third last action'
        assert da['CoT'][-2]['action'] == 'Final Conclusion', 'Final Conclusion should be the second last action'
        assert da['CoT'][-1]['action'] == 'Verification', 'Verification should be the last action'
        return True, da
    except Exception as e:
        print(f"Parsing error: {e}")
        traceback.print_exc()
        return False, None

def parse_gpt_response_reformat(response):
    try:
        if '{' != response[0]:
            response = extract_bracket_content(response)
        da = json.loads(response.replace('\n',''))

        assert isinstance(da["NaturalReasoning"],str), "NaturalReasoning should be str"
        assert '\n' in da["NaturalReasoning"], "NaturalReasoning should have \\n"
        return True,da
    except Exception as e:
        print(e)
        traceback.print_exc()
        return False,None 


def get_stream_of_search(longcot):
    temp = '### {}\n{}\n'
    resstr = []
    for x in longcot:
        if 'title' in x:
            resstr.append(temp.format(x['title'],x['content']))
        else:
            resstr.append(temp.format(x['action'].replace('Final Conclusion','Conclusion'),x['content']))
    return '\n'.join(resstr).strip()
def main():
    # 参数解析，支持单张图像测试
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_url", type=str, default="", help="URL of the X-ray image.")
    parser.add_argument("--reference_report", type=str, default="", help="Reference medical report for verification.")
    parser.add_argument("--model_name", type=str, default="/data0/zhuoxu/yihong/code/Test_CoT_Med_MLLM/Qwen2.5-vl-32B-merger-lora-base", help="Name of the Qwen-vl model to use.")
    parser.add_argument("--api_key", type=str, default="qwen-abc123", help="DashScope API key.")
    parser.add_argument("--api_url", type=str, default="http://127.0.0.1:8000/v1", help="DashScope API URL.")
    parser.add_argument("--max_search_attempts", type=int, default=3, help="Maximum number of search attempts.")
    parser.add_argument("--max_search_depth", type=int, default=2, help="Maximum search depth.")
    parser.add_argument("--efficient_search", type=bool, default=True, help="Enable efficient search strategy.")
    parser.add_argument("--data_path", type=str, default="/data0/zhuoxu/yihong/code/Test_CoT_Med_MLLM/MIMIC-CXR-Randow-5000-2.json", help="Path to the input JSON data file.") #/data0/zhuoxu/yihong/code/Test_CoT_Med_MLLM/MIMIC-CXR-Randow-1W.json
    args = parser.parse_args()

    # 加载 训练集 JSON 文件
    with open(args.data_path, 'r') as f:
        json_data = json.load(f)
    # train_data = json_data['train']
    train_data = json_data
    # 生成 data 列表
    data = []
    process_id = 1
    # image_base_path = "/data0/zhuoxu/data/X-ray/iu_xray/images/"

    for item in train_data:
        # item_id = item['id']
        report = item['report']
        # print(report)
        full_img_path = item['image_path'] #os.path.join(image_base_path, img_path)
        # print(full_img_path)
        data_item = {
            "process_id": process_id,
            "image_url": full_img_path,
            "reference_report": report,
            # "id": item_id
        }
        data.append(data_item)
        process_id += 1
    # 打印验证信息
    print(f"Total data items: {len(data)}")
    if data:
        print(f"Testing with first data item: {json.dumps(data[0], indent=2)}")

    # 设置保存目录
    task_name = "multi_MIMIC_CXR_random_5000_2_xray_train_CoT_use_local_Qwen_VL_32B_Instruct_Lora_merger_base"
    save_dir = f"/data0/zhuoxu/yihong/code/Test_CoT_Med_MLLM/output_data_new/{task_name}"
    os.makedirs(save_dir, exist_ok=True)

    # 初始化 GPT 实例
    gpt_instance = GPT(model_name=args.model_name, api_key=args.api_key, api_url=args.api_url)

    # 定义验证函数
    def verify_gpt(conclusion, reference_report, d):
        query_text = verify_prompt.format(conclusion, reference_report)
        # query = [{"role": "user", "content": query_text}]
        base64_image = encode_image(d["image_url"])
        query = [{
            "role": "user",
            "content": [
                {"type": "text", "text": query_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{base64_image}"}} #mimicxr 之后query要修正成jpg
            ]
        }]
        response = gpt_instance.retry_call(query)
        d['Qwen_query_cot'].append(query)
        d['Qwen_response_cot'].append(response)
        if 'true' in response.lower():
            d['verify'].append(True)
            return True
        else:
            d['verify'].append(False)
            return False

    # 处理单个数据条目
    global wrongtime
    wrongtime = 0
    def write_piece_order_data(d):
        global wrongtime
        try:
            retry_time = 1
            d['verify'] = []
            d['Long_CoT'] = []
            d['Qwen_query_cot'] = []
            d['Qwen_response_cot'] = []
            d['response_struct'] = []
            d['response_type'] = []
            d['prior_fail_try'] = []

            save_path = os.path.join(save_dir, f"{d['process_id']}.json")
            
            # 初始推理
            base64_image = encode_image(d["image_url"])
            query = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": query_prompt_init.format(d['reference_report'])},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{base64_image}"}}
                ]
            }]
            d['Qwen_query_cot'].append(query)
            for ii in range(retry_time):
                response = gpt_instance.retry_call(query)
                if ii == 0:
                    d['Qwen_response_cot'].append(response)
                flag, struct = parse_gpt_response(response) #flag为True的话，相当于它回复的三段式格式是正确的
                if flag:
                    d['response_struct'].append(struct["CoT"])
                    d['Long_CoT'] = struct["CoT"]
                    d['response_type'].append('Init_CoT')
                    break
                else:
                    print(f'retrying Init_CoT', flush=True)
            if not flag: #相当于第一次没按要求回答 直接就抛出错误了
                raise Exception('init error')

            # 验证初始结论 看和参照答案是否大致相同
            verify_gpt(d['Long_CoT'][-2]['content'], d['reference_report'], d)

            # 再次验证和搜索策略
            for rethinking_try_time in range(args.max_search_attempts):
                if rethinking_try_time > 0:
                    # Archive the failed state
                    del d['prior_fail_try']
                    save_d['prior_fail_try'].append(d)
                    # Replace with a new state
                    d = save_d

                # Save the initial state
                save_d = copy.deepcopy(d)

                # Begin search
                for rethink_time in range(args.max_search_depth):
                    if d['verify'][-1]: #True 说明一致了
                        break
                    reasoning = json.dumps(d['Long_CoT'][:-1], ensure_ascii=False, indent=2)
                     # Search strategy
                    if rethink_time > 0:
                        strategy_name,strategy = random.choice(search_strategies)
                    else:
                        # exclude Backtracking
                        strategy_name,strategy = random.choice(search_strategies[1:])
                    base64_image = encode_image(d["image_url"])
                    query = [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": strategy.format(reasoning,d['reference_report'])},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{base64_image}"}}
                        ]
                    }]
                    d['Qwen_query_cot'].append(query)
                    
                    for ii in range(retry_time):
                        response = gpt_instance.retry_call(query)
                        flag, struct = parse_gpt_response(response)
                        if flag:
                            d['Qwen_response_cot'].append(response)
                            d['response_struct'].append(struct["CoT"])
                            d['Long_CoT'] = d['Long_CoT'][:-1] + struct["CoT"]
                            d['response_type'].append(f'Re_CoT_{strategy_name}')
                            break
                        else:
                            print(f'retrying strategy {strategy_name}', flush=True)
                    if not flag:
                        raise Exception('rethink error')
                    verify_gpt(d['Long_CoT'][-2]['content'], d['reference_report'], d) #再次验证下 True?还是False
                
                if d['verify'][-1]:
                    break
            # If it is still incorrect, generate a final Label_CoT round
            if not d['verify'][-1] and args.efficient_search:
                reasoning = json.dumps(d['Long_CoT'][:-1],ensure_ascii=False,indent=2)
                query_text = gen_prompt_w_label.format(reasoning,d['reference_report']) #同时给了参考报告进去
                base64_image = encode_image(d["image_url"])
                query = [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": query_text},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{base64_image}"}}
                        ]
                    }]
                d['Qwen_query_cot'].append(query)
                for ii in range(retry_time):
                    response = gpt_instance.retry_call(query)       
                    flag, struct = parse_gpt_response(response)
                    if flag:
                        d['Qwen_response_cot'].append(response)
                        d['response_struct'].append(struct["CoT"])
                        d['Long_CoT'] =  d['Long_CoT'][:-1] + struct["CoT"]
                        d['response_type'].append('Label_CoT')
                        # ignore verify
                        d['verify'].append(True)
                        break
                    else:
                        print(f'retrying Label_CoT',flush=True)
                if not flag:
                    raise Exception('label error') 
                
            if d['verify'][-1]:
                # Generate complex CoT and final response (Complex_CoT, response)
                sos = get_stream_of_search(d['Long_CoT'])
                query_text = reformat_to_complex_cot_prompt.format(sos) #使其更符合人类直觉，更自然地呈现医学诊断思维过程 可以使用诸如“嗯”、“哦”、“另外”或“等等”这样的词汇，使推理过程更贴近人类思维。
                base64_image = encode_image(d["image_url"])
                query = [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": query_text},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{base64_image}"}}
                        ]
                    }]
                d['Qwen_query_cot'].append(query)
                for ii in range(retry_time):
                    response = gpt_instance.retry_call(query)
                    flag, struct = parse_gpt_response_reformat(response)
                    if flag:
                        d['Qwen_response_cot'].append(response)
                        d["Complex_CoT"] = struct["NaturalReasoning"]
                        # get response
                        query_text = get_final_response_prompt.format(d['Complex_CoT'],d['reference_report']) #Output only your final report, without any additional content.
                        query = [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": query_text},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{base64_image}"}}
                            ]
                        }]
                        d['Qwen_query_cot'].append(query)
                        response = gpt_instance.retry_call(query)
                        d['Qwen_response_cot'].append(response)
                        d["Response"] = response
                        # d['Question'] = d['Open-ended Verifiable Question']
                        break

            # 保存结果
            with open(save_path, mode="w", encoding="utf-8") as fw:
                json.dump(d, fw, ensure_ascii=False, indent=2)
                wrongtime = 0

        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            wrongtime += 1
            if wrongtime > 20:  # 减少重试上限以便快速测试
                raise Exception('Too many errors')
        return 1

    # 处理单张图像
    with ThreadPoolExecutor(max_workers=5) as executor:  # 单线程测试
        list(tqdm(executor.map(write_piece_order_data, data), total=len(data), desc="Processing sample", unit="sample"))

    # # 合并并保存结果
    # def merge_saved_files(save_dir):
    #     _, _, filenames = [i for i in os.walk(save_dir)][0]
    #     json_files = [f for f in filenames if f.endswith('.json')]
    #     res = []
    #     for file_path in json_files:
    #         try:
    #             with open(os.path.join(save_dir, file_path), encoding="utf-8") as f:
    #                 da = json.loads(f.read())
    #                 res.append(da)
    #         except Exception as e:
    #             print(f"Error merging file {file_path}: {e}")
    #     return res
    def merge_saved_files(save_dir):
        _, _, filenames = next(os.walk(save_dir))  # 获取文件名列表
        json_files = sorted([f for f in filenames if f.endswith('.json')], key=lambda x: int(x.split(".")[0]))  # 按数字排序
        res = []
        
        for file_path in json_files:
            try:
                with open(os.path.join(save_dir, file_path), encoding="utf-8") as f:
                    da = json.load(f)
                    assert 'Complex_CoT' in da and 'Response' in da
                    filtered_data = {
                        "process_id": da.get("process_id"),
                        "image_url": da.get("image_url"),
                        "reference_report": da.get("reference_report"),
                        "Complex_CoT": da["Complex_CoT"],
                        "Response": da["Response"]
                    }
                    res.append(filtered_data)
            except Exception:
                continue
        return res
    final_data = merge_saved_files(save_dir)
    output_path = f"/data0/zhuoxu/yihong/code/Test_CoT_Med_MLLM/output_data_new/Latest_result/{task_name}_result.json"
    print(f"Processed {len(final_data)} items. Saving to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(final_data, file, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()

