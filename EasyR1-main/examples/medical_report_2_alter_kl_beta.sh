#!/bin/bash

set -x
export PYTHONPATH=/data0/zhuoxu/yihong/code/EasyR1-main:$PYTHONPATH
export PYTHONUNBUFFERED=1
export VLLM_ATTENTION_BACKEND=XFORMERS

# 增加Ray内存阈值，减少内存OOM问题
# export RAY_memory_usage_threshold=0.99
# export RAY_memory_monitor_refresh_ms=0
# export VLLM_USE_V1=1

# 设置路径
EASYR1_DIR=/data0/zhuoxu/yihong/code/EasyR1-main
DATA_DIR=${EASYR1_DIR}/examples/data/medical_report
WORK_DIR=/data0/zhuoxu/yihong/code

# # 首先转换数据集格式
# python3 ${EASYR1_DIR}/scripts/convert_medical_dataset.py \
#     --input /data0/zhuoxu/yihong/code/LLaMA-Factory-main/eval_data_dir/Med_mimic_cxr_only_ref_report_validate_only_F-I.json \
#     --output_dir ${DATA_DIR} \
#     --split val

# # 将JSON格式转换为Parquet格式
# python3 ${WORK_DIR}/convert_to_parquet.py \
#     --input ${DATA_DIR}/medical_report_val.json \
#     --output ${DATA_DIR}/medical_report_val.parquet

# 设置模型路径
# MODEL_PATH=/data0/zhuoxu/yihong/code/LLaMA-Factory-main/saves/Qwen2-VL-7B-Instruct/full/train_stage2_use_32B_MIMIC_CXR_5000_1 
MODEL_PATH=/data0/zhuoxu/yihong/code/LLaMA-Factory-main/saves/Qwen2-VL-7B-Instruct/full/train_Stage2_use_32B_MIMIC_CXR_10000_1_train_merger_llm_1e-5_epoch2

# 确保在EasyR1目录下执行
cd ${EASYR1_DIR}

# 启动GRPO训练，优化GPU使用
CUDA_VISIBLE_DEVICES=0,1 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=${DATA_DIR}/medical_report_val.parquet \
    data.val_files=${DATA_DIR}/medical_report_val.parquet \
    data.val_batch_size=2 \
    data.rollout_batch_size=4 \
    data.prompt_key=prompt \
    data.answer_key=answer \
    data.image_key=images \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.format_prompt=${EASYR1_DIR}/examples/format_prompt/medical_report_format.jinja \
    data.max_pixels=1000000 \
    data.min_pixels=10000 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.trust_remote_code=true \
    worker.reward.reward_function=examples/reward_function/medical_report.py:compute_score \
    worker.reward.reward_type=batch \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.gpu_memory_utilization=0.6 \
    worker.rollout.n=8 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.global_batch_size=1 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.rollout.temperature=0.9 \
    worker.actor.offload.offload_params=false \
    worker.actor.offload.offload_optimizer=false \
    worker.actor.model.freeze_vision_tower=true \
    trainer.experiment_name=medical_report_vl_grpo \
    trainer.n_gpus_per_node=2 \
    trainer.total_epochs=1 \
    trainer.val_before_train=false \
    trainer.val_freq=0 \
    trainer.save_freq=200 \
    trainer.save_checkpoint_path=${EASYR1_DIR}/checkpoints_report_kl_beta/easy_r1 \
    algorithm.adv_estimator=grpo \
    algorithm.kl_coef=5.0e-2
# trainer.save_freq=200 \ 
# worker.ref.fsdp.enable_cpu_offload=false \
# worker.actor.offload.offload_params=false \
# 训练完成后单独运行保存脚本
# CHECKPOINT_PATH=${EASYR1_DIR}/checkpoints/easy_r1/medical_report_vl_grpo/final
# mkdir -p ${CHECKPOINT_PATH}

# # 复制保存的actor模型到最终目录
# cp -r ${EASYR1_DIR}/checkpoints/easy_r1/medical_report_vl_grpo/global_step_* ${CHECKPOINT_PATH}/
# echo "模型已保存至: ${CHECKPOINT_PATH}"
#trainer.load_checkpoint_path=/data0/zhuoxu/yihong/code/EasyR1-main/checkpoints/easy_r1/global_step_400 \