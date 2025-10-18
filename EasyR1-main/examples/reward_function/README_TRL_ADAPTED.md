# TRL风格适配的医疗Grounding奖励函数

## 概述

`medical_grounding_reward_trl_adapted.py` 是一个将TRL库的奖励构造方式适配到VERL库的医疗grounding奖励函数。该函数融合了TRL库的先进奖励机制，旨在解决原始奖励函数中的数据不平衡问题。

## 主要特点

### 1. **双格式支持**
- **医疗Grounding格式**: `<box>(x1,y1),(x2,y2)</box>`
- **TRL标准格式**: `[{'Position': [x1, y1, x2, y2], 'Confidence': conf}, ...]`

### 2. **TRL风格的奖励计算**
- 采用TRL库的`compute_reward_iou_v2`算法
- 使用置信度加权的IoU计算
- 支持边界框去重和最优匹配

### 3. **数据不平衡优化**
- 降低无病变样本的奖励权重（0.3 vs 原来的0.9）
- 增加有病变检测的额外奖励（+0.1）
- 平衡格式奖励和准确率奖励的权重

### 4. **调试支持**
- 可通过环境变量开启调试模式
- 详细的日志记录功能
- 支持自定义日志路径

## 核心算法

### 边界框提取
```python
def extract_bounding_boxes_trl_style(text: str) -> List[Tuple[int, int, int, int]]:
    # 1. 尝试医疗grounding格式: <box>(x1,y1),(x2,y2)</box>
    # 2. 尝试TRL格式: [{'Position': [x1, y1, x2, y2], 'Confidence': conf}]
    # 3. 返回标准化的坐标列表
```

### IoU计算
```python
def sort_and_calculate_iou_trl_style(gt_boxes, pred_boxes, iou_threshold=0.5):
    # 1. 为预测框分配置信度
    # 2. 按置信度排序
    # 3. 使用贪心匹配算法
    # 4. 返回IoU和置信度对列表
```

### 奖励计算
```python
def compute_reward_iou_v2_trl_style(iou_results, len_gt):
    # 1. 累计所有IoU分数
    # 2. 根据真实框数量或预测框数量归一化
    # 3. 返回最终奖励分数
```

## 使用方法

### 1. **基本使用**
```bash
# 在训练脚本中指定奖励函数
worker.reward.reward_function=examples/reward_function/medical_grounding_reward_trl_adapted.py:compute_score_trl_adapted
```

### 2. **参数配置**
```python
# 可调整的参数
format_weight = 0.1  # 格式奖励权重
iou_threshold = 0.5  # IoU阈值
```

### 3. **调试模式**
```bash
# 开启调试模式
export DEBUG_MODE=true
export LOG_PATH=/path/to/debug.log
```

## 奖励分数说明

### 分数组成
- **overall**: 综合奖励分数 (0-1)
- **format_reward**: 格式奖励分数 (0-1)
- **accuracy_reward**: 准确率奖励分数 (0-1)

### 计算公式
```
overall = format_weight * format_reward + (1 - format_weight) * accuracy_reward
```

### 分数策略
| 情况 | 准确率奖励 | 说明 |
|------|------------|------|
| 完全匹配 | 0.8-1.0 | 基于IoU质量+额外奖励 |
| 部分匹配 | 0.1-0.7 | 基于IoU和匹配度 |
| 无病变正确 | 0.3 | 降低以平衡数据不平衡 |
| 假阳性 | 0.1 | 预测有病变但实际无 |
| 假阴性 | 0.05 | 预测无病变但实际有 |

## 与原始版本的对比

| 特性 | 原始版本 | TRL适配版本 |
|------|----------|-------------|
| 无病变奖励 | 0.9 | 0.3 |
| 有病变额外奖励 | 无 | +0.1 |
| 边界框匹配 | 匈牙利算法 | TRL贪心匹配 |
| 格式支持 | 单一格式 | 双格式支持 |
| 调试功能 | 基础 | 详细日志 |

## 预期效果

1. **减少无病变偏向**: 降低模型总是预测"无病变"的倾向
2. **提高检测精度**: 鼓励模型积极检测病变区域
3. **更好的格式兼容性**: 支持多种输出格式
4. **便于调试**: 详细的日志记录帮助分析模型行为

## 注意事项

1. **坐标系统**: 假设坐标范围为0-1000的归一化坐标
2. **置信度处理**: 当前版本对所有预测框使用固定置信度1.0
3. **内存使用**: 大批量处理时注意内存使用情况
4. **日志文件**: 调试模式下会产生大量日志，注意磁盘空间

## 扩展建议

1. **动态置信度**: 从模型输出中提取真实的置信度分数
2. **多阈值评估**: 支持多个IoU阈值的评估
3. **类别特定奖励**: 针对不同类型的病变给予不同权重
4. **在线调整**: 根据训练进度动态调整奖励权重 