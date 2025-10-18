import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import json
import sys
# import torch
# from tqdm import tqdm
from collections import defaultdict
import nltk
# import numpy as np
import shutil

# # 添加pycocoevalcap路径
sys.path.append('/data0/zhuoxu/yihong/code/R2Gen-main')
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

try:
    nltk.data.find('/data0/zhuoxu/yihong/code/tokenizers/punkt')
except:
    nltk.download('punkt', quiet=True)

# 检查Java是否可用
def is_java_available():
    return shutil.which('java') is not None


"""
评估指标计算类  最终都是使用这个库进行计算
"""
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

def process_json_file(input_path, output_path):
    print(f"处理文件: {input_path}")
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # # 检查预测报告长度，找出过长的报告
    print("\n检查预测报告长度...")
    long_reports = []
    max_length_threshold = 1000  # 设置长度阈值，可以根据需要调整
    
    for i, case in enumerate(data['detailed_results']):
        predicted_report = case['predicted_report']
        report_length = len(predicted_report)
        
        if report_length > max_length_threshold:
            long_reports.append({
                'index': i,
                'subject_id':case['subject_id'],
                'study_id':case['study_id'],
                'length': report_length,
                'report_preview': predicted_report[:200] + '...' if len(predicted_report) > 200 else predicted_report
            })
    
    if long_reports:
        print(f"\n发现 {len(long_reports)} 条过长的预测报告（长度 > {max_length_threshold} 字符）:")
        for report_info in long_reports:
            print(f"  索引 {report_info['index']}: 长度 {report_info['length']} 字符")
            print(f"    预览: {report_info['report_preview']}")
            print(f"    subject_id: {report_info['subject_id']}")
            print(f"    study_id: {report_info['study_id']}")
            print()
        
        # 询问是否继续处理
        print("这些过长的报告可能导致METEOR评估出现问题。")
        print("建议先处理这些过长的报告，或者跳过METEOR评估。")
    else:
        print(f"所有预测报告长度都在 {max_length_threshold} 字符以内。")
    
    evaluator = COCOEvaluator()
    
    # 准备评估数据格式
    gts = {}  # 参考文本
    res = {}  # 预测文本
    
    for i, case in enumerate(data['detailed_results']):
        # 使用索引作为ID
        image_id = str(i)
        
        # 添加参考文本
        gts[image_id] = [{'caption': case['reference_report']}]
        
        # 添加预测文本
        res[image_id] = [{'caption': case['predicted_report']}]
    # 计算评估指标
    average_scores, scores_per_image = evaluator.compute_scores(gts, res)
    
    # 更新每个案例的指标
    for i, case in enumerate(data['detailed_results']):
        # 获取当前案例的指标
        image_id = str(i)
        case_metrics = scores_per_image.get(image_id, {})
        
        # 更新指标
        metrics = {}
        
        # 添加BLEU指标
        for bleu_key in ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']:
            if bleu_key in case_metrics:
                metrics[bleu_key] = case_metrics[bleu_key]
        
        # 添加ROUGE_L和CIDEr指标
        if 'ROUGE_L' in case_metrics:
            metrics['ROUGE_L'] = case_metrics['ROUGE_L']
        if 'CIDEr' in case_metrics:
            metrics['CIDEr'] = case_metrics['CIDEr']
        
        # 添加METEOR指标(如果可用)
        if 'METEOR' in case_metrics:
            metrics['METEOR'] = case_metrics['METEOR']
            
        case['metrics'] = metrics
    
    # 重新计算平均指标（确保准确性）
    recalculated_average = {}
    metrics_sum = defaultdict(float)
    metrics_count = defaultdict(int)
    
    for case in data['detailed_results']:
        if 'metrics' in case:
            for metric_name, metric_value in case['metrics'].items():
                metrics_sum[metric_name] += metric_value
                metrics_count[metric_name] += 1
    
    for metric_name in metrics_sum:
        if metrics_count[metric_name] > 0:
            recalculated_average[metric_name] = metrics_sum[metric_name] / metrics_count[metric_name]
    
    # 更新平均指标
    data['average_metrics'] = recalculated_average
    
    # 打印平均指标
    print("\n重新计算的平均指标:")
    for metric_name, metric_value in recalculated_average.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    # 保存结果
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"处理完成！结果已保存到: {output_path}")

if __name__ == "__main__":
    # # 输入和输出路径
    input_path = "/path/your_result.json"
    output_path = "/path/your_result_pycocoeval.json"
    
    # 处理文件
    process_json_file(input_path, output_path) 
 