#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2024
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import nltk
from typing import Dict, List
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
# from rouge_score import rouge_scorer
import spacy
# 确保nltk数据已下载
try:
    nltk.data.find('/data0/zhuoxu/yihong/code/tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# global RADIOLOGY_NLP
# RADIOLOGY_NLP = spacy.load("en_core_sci_lg")  # 医学专用模型

# 添加pycocoevalcap路径
import sys
sys.path.append('/data0/zhuoxu/yihong/code/R2Gen-main')
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
import shutil
# 检查Java是否可用
def is_java_available():
    return shutil.which('java') is not None

class COCOEvaluator:
    def __init__(self):
        self.tokenizer = PTBTokenizer()
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]
        
        # 只有在Java可用时才添加METEOR评估
        if is_java_available():
            from pycocoevalcap.meteor.meteor import Meteor
            self.scorers.append((Meteor(), "METEOR"))
            print("METEOR评估已启用")
        else:
            print("警告: Java不可用，METEOR评估将被跳过")
    
    def compute_scores(self, gts, res):
        """
        计算所有评估指标
        :param gts: 参考文本字典 {id: [{'caption': text}]}
        :param res: 预测文本字典 {id: [{'caption': text}]}
        :return: (评估结果字典, 每个图像的评估结果字典)
        """
        # 标记化
        print('标记化文本...')
        tokenized_gts = self.tokenizer.tokenize(gts)
        tokenized_res = self.tokenizer.tokenize(res)
        
        # 计算得分
        all_scores = {}
        all_scores_per_image = {}
        
        for scorer, method in self.scorers:
            print(f'计算 {scorer.method()} 得分...')
            score, scores_per_image = scorer.compute_score(tokenized_gts, tokenized_res)
            
            if type(method) == list:
                for m, s, spi in zip(method, score, scores_per_image):
                    all_scores[m] = s
                    print(f"{m}: {s:.4f}")
                    
                    # 保存每个图像的得分
                    for i, image_id in enumerate(gts.keys()):
                        if image_id not in all_scores_per_image:
                            all_scores_per_image[image_id] = {}
                        all_scores_per_image[image_id][m] = spi[i]
            else:
                all_scores[method] = score
                print(f"{method}: {score:.4f}")
                
                # 保存每个图像的得分
                for i, image_id in enumerate(gts.keys()):
                    if image_id not in all_scores_per_image:
                        all_scores_per_image[image_id] = {}
                    all_scores_per_image[image_id][method] = scores_per_image[i]
        
        return all_scores, all_scores_per_image


RADIOLOGY_ENT_WEIGHTS = {
    'PATHOLOGY': 1.5,               # 提高病理发现的权重
    'ANATOMY': 0.8,                 # 降低解剖结构的权重
    'IMAGING_FEATURE': 1.3,         # 提高影像特征的权重
    'BODY_FUNCTION': 0.7,           # 降低身体功能的权重
    'MEDICAL_DEVICE': 0.5,          # 降低医疗设备的权重
    'DIAGNOSIS': 1.8,               # 新增：诊断类别（最高权重）
    'NORMAL_FINDING': 1.0,          # 新增：正常发现
    'DIFFERENTIAL': 1.2,            # 新增：鉴别诊断
}

def format_reward(predict: str) -> float:
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, predict)
    return 1.0 if format_match else 0.0

# def calculate_entity_coverage(gen_text, ref_text):
#     """简化的医学实体覆盖计算，移除否定检测，只关注实体文本匹配"""
#     # 使用spaCy加载医学NLP模型
#     if 'RADIOLOGY_NLP' not in globals():
#         global RADIOLOGY_NLP
#         RADIOLOGY_NLP = spacy.load("en_core_sci_lg")  # 医学专用模型
    
#     # 文本预处理：规范化文本，便于更好的匹配
#     ref_text_clean = re.sub(r'\s+', ' ', ref_text.lower().strip())
#     gen_text_clean = re.sub(r'\s+', ' ', gen_text.lower().strip())
    
#     # 提取参考报告实体
#     ref_doc = RADIOLOGY_NLP(ref_text_clean)
#     ref_entities = {}
    
#     # 打印参考文本实体标签
#     entity_labels = set([ent.label_ for ent in ref_doc.ents])
#     if entity_labels:
#         print(f"参考文本实体标签: {entity_labels}")
    
#     # 提取参考文本中的实体
#     for ent in ref_doc.ents:
#         # 不管实体类型是否在RADIOLOGY_ENT_WEIGHTS中，都添加一个默认权重
#         weight = RADIOLOGY_ENT_WEIGHTS.get(ent.label_, 1.0)
#         ref_entities[ent.text.lower()] = max(ref_entities.get(ent.text.lower(), 0), weight)
    
#     # 增加关键词匹配作为补充
#     medical_terms = extract_medical_terms(ref_text_clean)
#     for term in medical_terms:
#         if term not in ref_entities:  # 避免覆盖已有实体
#             ref_entities[term] = 1.0  # 默认权重
    
#     # 如果仍没有找到参考实体，使用更宽松的文本匹配
#     if len(ref_entities) == 0:
#         print("未找到参考实体，使用文本片段...")
#         # 将文本分成短语，每个短语作为一个"实体"
#         for phrase in re.split(r'[.,;:]', ref_text_clean):
#             if len(phrase.strip()) > 5:  # 忽略太短的短语
#                 ref_entities[phrase.strip()] = 1.0
    
#     print(f"找到的参考实体数量: {len(ref_entities)}")
#     if len(ref_entities) > 0:
#         print(f"参考实体示例: {list(ref_entities.keys())[:3]}")
    
#     # 提取生成内容实体
#     gen_doc = RADIOLOGY_NLP(gen_text_clean)
#     matched_weights = []
#     matched_entities = set()  # 记录已匹配的实体，避免重复计算
    
#     # 1. 首先尝试精确实体匹配
#     for ent in gen_doc.ents:
#         entity_text = ent.text.lower()
#         if entity_text in ref_entities and entity_text not in matched_entities:
#             matched_weights.append(ref_entities[entity_text])
#             matched_entities.add(entity_text)
#             print(f"精确匹配到实体: {entity_text}, 权重: {ref_entities[entity_text]}")
    
#     # 2. 关键词匹配
#     medical_terms_gen = extract_medical_terms(gen_text_clean)
#     for term in medical_terms_gen:
#         if term in ref_entities and term not in matched_entities:
#             matched_weights.append(ref_entities[term])
#             matched_entities.add(term)
#             print(f"匹配到关键词: {term}, 权重: {ref_entities[term]}")
    
#     # 3. 部分匹配 - 检查生成文本是否包含参考实体的文本
#     for ref_entity_text, weight in ref_entities.items():
#         if ref_entity_text not in matched_entities:  # 只考虑尚未匹配的实体
#             # 检查生成文本是否包含该实体文本
#             if ref_entity_text in gen_text_clean:
#                 matched_weights.append(weight * 0.9)  # 部分匹配给予稍低的权重
#                 matched_entities.add(ref_entity_text)
#                 print(f"部分匹配到实体: {ref_entity_text}, 权重: {weight * 0.9}")
    
#     # 4. 模糊匹配 - 使用Levenshtein距离找到相似实体
#     try:
#         import Levenshtein
#         for gen_ent in gen_doc.ents:
#             gen_text = gen_ent.text.lower()
#             for ref_entity_text, weight in ref_entities.items():
#                 if ref_entity_text not in matched_entities:  # 只考虑尚未匹配的实体
#                     # 计算编辑距离，标准化为文本长度
#                     if len(ref_entity_text) > 3 and len(gen_text) > 3:  # 忽略太短的文本
#                         similarity = 1 - Levenshtein.distance(ref_entity_text, gen_text) / max(len(ref_entity_text), len(gen_text))
#                         if similarity > 0.7:  # 高相似度匹配
#                             matched_weights.append(weight * similarity)
#                             matched_entities.add(ref_entity_text)
#                             print(f"模糊匹配到实体: {ref_entity_text} ≈ {gen_text}, 相似度: {similarity:.2f}, 权重: {weight * similarity:.2f}")
#     except ImportError:
#         print("未安装Levenshtein库，跳过模糊匹配")
    
#     # 计算覆盖率（加权）
#     total_weight = sum(ref_entities.values()) if ref_entities else 1.0
    
#     # 移除惩罚，专注于匹配质量
#     coverage_score = sum(matched_weights) / total_weight if total_weight > 0 else 0.0
    
#     # 调整最终分数，增加基础分以避免过低
#     final_score = max(0.2, coverage_score)  # 确保最低分为0.2
    
#     print(f"实体覆盖分数: {coverage_score}, 最终分数: {final_score}")
    
#     return final_score

# def detect_negation(ent):
#     """改进的否定检测"""
#     try:
#         # 获取实体所在句子的所有token
#         sent_tokens = [t.lower_ for t in ent.sent]
        
#         # 检查实体前的否定词
#         if isinstance(ent.start_char, int) and hasattr(ent.sent, 'start_char') and isinstance(ent.sent.start_char, int):
#             # 计算实体在句子中的相对位置
#             relative_start = ent.start_char - ent.sent.start_char
#             # 获取实体前的token
#             tokens_before_entity = sent_tokens[:relative_start] if relative_start > 0 else []
#             # 检查最后3个token是否包含否定词
#             last_three_tokens = tokens_before_entity[-3:] if len(tokens_before_entity) >= 3 else tokens_before_entity
#             pre_negation = any(neg in last_three_tokens for neg in {'no', 'not', 'without', 'absence', 'negative'})
#         else:
#             pre_negation = False
        
#         # 检查常见否定短语
#         sent_text = ent.sent.text.lower() if hasattr(ent.sent, 'text') else ""
#         negation_phrases = [
#             'free of', 'absence of', 'ruled out', 'negative for', 
#             'no evidence of', 'no sign of', 'denies', 'without evidence of'
#         ]
#         phrase_negation = any(phrase in sent_text for phrase in negation_phrases)
        
#         return pre_negation or phrase_negation
#     except Exception as e:
#         print(f"否定检测出错: {e}")
#         return False

def calculate_length_adjustment(gen_text, ref_text):
    """动态长度调节器（仅下限版）"""
    ref_len = len(RADIOLOGY_NLP(ref_text))
    gen_len = len(RADIOLOGY_NLP(gen_text))
    
    # 自适应阈值（仅保留下限）
    lower = max(100, 0.6 * ref_len)  # 修改点1：删除upper相关代码
    
    if gen_len < lower:
        # 严格的下限惩罚机制
        return -0.5 * (1 - gen_len/lower)
    else:
        # 鼓励接近或超过参考长度的生成（删除原abs计算）
        return 1.0 - max(0, (ref_len - gen_len)/ref_len)  # 修改点2：仅惩罚短于参考的情况

def calculate_semantic_similarity(gen_report, ref_report):
    """使用TF-IDF和余弦相似度计算文本相似度，避免维度问题"""
    try:
        # 文本清理
        def clean_text(text):
            # 移除特殊字符并转为小写
            text = re.sub(r'[^\w\s.,;:!?()-]', ' ', text.lower())
            # 确保没有连续多个空格
            return re.sub(r'\s+', ' ', text).strip()
        
        gen_text = clean_text(gen_report)
        ref_text = clean_text(ref_report)
        
        if not gen_text or not ref_text:
            return 0.5
        
        # 使用scikit-learn的TF-IDF向量化
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # 初始化向量化器
        vectorizer = TfidfVectorizer()
        
        # 拟合并转换文本
        tfidf_matrix = vectorizer.fit_transform([gen_text, ref_text])
        
        # 计算余弦相似度
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return float(similarity)
    
    except Exception as e:
        print(f"TF-IDF语义相似度计算错误: {str(e)}")
        return 0.5  # 返回默认值




def extract_medical_terms(text: str) -> List[str]:
    """
    从文本中提取医学术语
    
    Args:
        text: 输入文本
        
    Returns:
        List[str]: 医学术语列表
    """
    # 常见的医学术语和发现的关键词列表
    common_findings = [
        "pneumonia", "effusion", "edema", "cardiomegaly", "atelectasis", 
        "pneumothorax", "consolidation", "pleural", "infiltrate", "opacity",
        "fracture", "nodule", "mass", "emphysema", "fibrosis", "calcification",
        "thickening", "enlarged", "fluid", "congestion", "collapse"
    ]
    
    # 提取文本中的医学术语
    found_terms = []
    for term in common_findings:
        if term in text.lower():
            found_terms.append(term)
    
    return found_terms


def compute_score(predicts: List[str], ground_truths: List[str], format_weight: float = 0.3) -> List[Dict[str, float]]:
    """
    计算生成报告的综合评分
    
    Args:
        predicts: 模型生成的报告列表
        ground_truths: 参考报告列表
        format_weight: 格式评分权重
        
    Returns:
        List[Dict[str, float]]: 评分结果列表
    """

    scores = []
    for predict, ground_truth in zip(predicts, ground_truths):
        # 首先检查生成文本是否包含answer标签
        gen_match = re.search(r"<answer>(.*?)</answer>", predict, re.DOTALL)
        if not gen_match:
            print(f"警告: 生成内容中未找到<answer>标签: {predict[:100]}...")
            # 返回一个包含所有必要键的字典，而不是单个浮点数
            scores.append({
                "overall": 0.1,  # 格式错误给予低分
                "format": 0.0,
                "nlg_metrics":0.0
                # "semantic": 0.0,
                # "length": 0.0,
                # "entity": 0.0,
            })
            continue
            
        predicts_answer = gen_match.group(1).strip()
        # 标准化处理
        ground_truth = ground_truth.replace('\n', ' ').strip()
        predicts_answer = predicts_answer.replace('\n', ' ').strip()
        
        # 计算格式评分
        # print("predicts_answer",predicts_answer)
        format_score = format_reward(predict)
        
        # 计算NLG指标奖励
        nlg_metrics_score = compute_nlg_metrics_reward(predicts_answer, ground_truth)

        scores.append({
            # "overall": 0.25*format_score + 0.25*semantic_score + 0.1*length_score + 0.2*entity_score + 0.2*nlg_metrics_score,
            "overall": 0.4*format_score + 0.6*nlg_metrics_score,
            "format": format_score,
            # "semantic": semantic_score,
            # "length": length_score,
            # "entity": entity_score,
            "nlg_metrics": nlg_metrics_score,
        })
    
    return scores 

def compute_nlg_metrics_reward(gen_text, ref_text):
    """计算NLG指标奖励"""
    # 准备评估数据格式
    gts = {'0': [{'caption': ref_text}]}
    res = {'0': [{'caption': gen_text}]}
    
    # 初始化评估器
    evaluator = COCOEvaluator()
    
    # 计算评估指标
    scores, _ = evaluator.compute_scores(gts, res)
    
    # 加权组合不同指标
    nlg_reward = (
        0.15 * scores.get('Bleu_1', 0) + 
        0.15 * scores.get('Bleu_2', 0) + 
        0.15 * scores.get('Bleu_3', 0) + 
        0.15 * scores.get('Bleu_4', 0) + 
        0.15 * scores.get('ROUGE_L', 0) + 
        0.15 * scores.get('CIDEr', 0) + 
        0.10 * scores.get('METEOR', 0)
    )
    
    return nlg_reward 