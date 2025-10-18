# DiagCoT-main


![image](asset/overview.png)


## Content
<!-- Content包含各个子标题的预览和跳转： -->
+ [Requirements](#requirements)
+ [Stage 1](#stage-1)
+ [Stage 2](#stage-2)
+ [Stage 3](#stage-3)
+ [Citation](#citation)

## Overview
Developing artificial intelligence systems that can emulate radiologists’ reasoning is essential for reliable
deployment in clinical practice. This study introduces DiagCoT, a multi-stage fine-tuning framework that
equips general-purpose vision–language models (VLMs) with diagnostic reasoning abilities using only
free-text radiology reports. The framework integrates three components: (i) contrastive image–report
tuning to align medical images with domain-specific terminology, (ii) chain-of-thought supervision to
capture intermediate inferential steps, and (iii) reinforcement optimization with clinical reward signals
to enhance factual accuracy and linguistic fluency
## Requirements
<!-- 项目依赖环境及安装说明 -->
The experiments with the code of this project were conducted based on 2*NVIDIA A800 80GB PCIe graphics cards.
### Stage 1 environment
<!-- 第一阶段的环境要求 -->
```bash
conda create env -n LLaMA_Factory python=3.10
conda activate LLaMA_Factory
cd LLaMA-Factory-main/
pip install -r requirements.txt
```

### Stage 2 environment
<!-- 第二阶段的环境要求 -->
```bash
conda create env -n Qwen python=3.10
conda activate Qwen
cd CoT-Collection-Filtration/
pip install -r requirements.txt
```

### Stage 3  environment
<!-- 第三阶段的环境要求 -->
```bash
conda create env -n easyr1 python=3.10
conda activate easyr1
cd EasyR1-main/
pip install -r requirements.txt
```

## Stage 1
<!-- 第一阶段的详细内容 -->
### DiagCoT-Align-VLM
```bash
conda activate LLaMA_Factory
cd LLaMA-Factory-main
CUDA_VISIBLE_DEVICES=2,3 llamafactory-cli train DiagCoT-main/LLaMA-Factory-main/saves/train_Med_Qwen_2_VL_7B_Proj_stage1_warmup_only_F-I/training_args.yaml
```

### DiagCoT-Teacher-VLM
```bash
CUDA_VISIBLE_DEVICES=2,3 llamafactory-cli train DiagCoT-main/LLaMA-Factory-main/saves/tran_Qwen2.5_vl_32B_lora/training_args.yaml
```

## Stage 2
<!-- 第二阶段的详细内容 -->
### CoT Collection
```bash
conda activate LLaMA_Factory
llamafactory-cli export DiagCoT-main/LLaMA-Factory-main/saves/tran_Qwen2.5_vl_32B_lora/merge_config_report_lora_32B.yaml

conda activate Qwen
CUDA_VISIBLE_DEVICES=4 vllm serve DiagCoT-main/LLaMA-Factory-main/saves/tran_Qwen2.5_vl_32B_lora/mergerd_model --dtype auto --api-key qwen-abc123 --max_model_len=16000 --gpu_memory_utilization=0.98 --trust-remote-code 
cd CoT-Collection-Filtration/
python DiagCoT-main/CoT-Collection-Filtration/search_for_complex_reasoning_path-local-multi_image-MIMIC-CXR-vllm-32B-merger-lora.py
``` 

### CoT Filtration
```bash
conda activate Qwen
CUDA_VISIBLE_DEVICES=4 vllm serve Qwen2.5-VL-72B-Instruct-AWQ --dtype auto --api-key qwen-abc123 --max_model_len=16000 --gpu_memory_utilization=0.98 --trust-remote-code
cd CoT-Collection-Filtration/
python /data0/zhuoxu/yihong/code/DiagCoT-main/CoT-Collection-Filtration/extract_cot_data_use_72B_VL_model-report.py
``` 

### DiagCoT-CoT-VLM
```bash
conda activate LLaMA_Factory
CUDA_VISIBLE_DEVICES=2,3 llamafactory-cli train DiagCoT-main/LLaMA-Factory-main/saves/train_Stage2_use_32B_MIMIC_CXR_10000_1_train_merger_llm_1e-5_epoch2/training_args.yaml
``` 
## Stage 3
### DiagCoT-RFT-VLM
<!-- 第三阶段的详细内容 -->
```bash
cd EasyR1-main/examples
bash medical_report_2_alter_kl_beta.sh
``` 




## Evaluation of Report Task
We evaluate the DiagCoT-RFT-VLM on the MIMIC-CXR and IU-Xray datasets. The evaluation metrics include : BLEU{1-4}, ROUGE-L, Meteor, and CIDEr.
- MIMIC-CXR
```bash
python DiagCoT-main/LLaMA-Factory-main/eval_mimic_cxr_code/eval_med_qwen2_belu_rouge-MIMIC-CXR-RFT_step400.py
``` 

- IU-Xray
```bash
python DiagCoT-main/LLaMA-Factory-main/eval_iu_xray_code/eval_med_qwen2_belu_rouge_RFT.py
``` 

- Latest Result
```bash
python /data0/zhuoxu/yihong/code/DiagCoT-main/LLaMA-Factory-main/compute_eval_metrics_pycocoevalcap.py
``` 

## Evaluation of Grounding Task
- Grounding
```bash
python DiagCoT-main/LLaMA-Factory-main/eval_gounding_code/evaluate_grounding_cot_rft.py
``` 



## Model Weights Path

We have uploaded the final trained model weights and Dastset json to Quark Cloud Drive. The link is as follows: https://pan.quark.cn/s/9768b2105a3b?pwd=V6Be

## Todo
- [x] Release the code of Report (MIMIC-CXR and IU-Xray)
- [ ] Release the code of Classification 
- [ ] Release the code of Grounding 

## Acknowledgment
We sincerely thank the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and [EasyR1](https://github.com/hiyouga/EasyR1) projects for their excellent open-source work, which our SFT and GRPO codes are respectively based on.


## Citation
<!-- 项目引用方式及相关文献 -->
