#!/bin/bash

set -x
export PYTHONPATH=/data0/zhuoxu/yihong/code/EasyR1-main:$PYTHONPATH
export PYTHONUNBUFFERED=1
export VLLM_ATTENTION_BACKEND=XFORMERS

# 设置路径
EASYR1_DIR=/data0/zhuoxu/yihong/code/EasyR1-main
DATA_DIR=${EASYR1_DIR}/examples/data/medical_grouding_rsna
WORK_DIR=/data0/zhuoxu/yihong/code

# # 转换数据集格式
# python3 ${EASYR1_DIR}/scripts/convert_grounding_dataset.py \
#     --input ${DATA_DIR}/val_grounding.json \
#     --output_dir ${DATA_DIR}

# 设置模型路径 - 使用grounding模型 该模型为使用CoT数据（随机从训练集抽取5000条）在已经经过merger和LLM训练的模型上继续训练
MODEL_PATH=/data0/zhuoxu/yihong/code/EasyR1-main/Stage2_chxpert_lora_grounding_model/merged_model_grounding_new_w_think_aug_qwen2_vl_coord_use_cotdata_and_after_merger_llm_train 
# MODEL_PATH=/data0/zhuoxu/yihong/code/LLaMA-Factory-main/saves/Qwen2-VL-7B-Instruct/full/train_2025-07-13-11-02-40_grounding_5e-6_ep2 
# 确保在EasyR1目录下执行
cd ${EASYR1_DIR}

# 启动GRPO训练，优化GPU使用
CUDA_VISIBLE_DEVICES=0,1 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=${DATA_DIR}/grounding_train_augmented_Qwen2_vl.parquet \
    data.val_files=${DATA_DIR}/grounding_train_augmented_Qwen2_vl.parquet \
    data.rollout_batch_size=4 \
    data.prompt_key=prompt \
    data.answer_key=answer \
    data.image_key=images \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.format_prompt=${EASYR1_DIR}/examples/format_prompt/medical_grounding_format.jinja \
    data.max_pixels=2000000 \
    data.min_pixels=100000 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.trust_remote_code=true \
    worker.reward.reward_function=examples/reward_function/medical_grounding_reward.py:compute_score \
    worker.reward.reward_type=batch \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.gpu_memory_utilization=0.6 \
    worker.rollout.n=6 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.global_batch_size=1 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.rollout.temperature=0.9 \
    worker.actor.offload.offload_params=false \
    worker.actor.offload.offload_optimizer=false \
    worker.actor.model.freeze_vision_tower=true \
    trainer.experiment_name=medical_grounding_vl_grpo \
    trainer.n_gpus_per_node=2 \
    trainer.total_epochs=1 \
    trainer.val_before_train=false \
    trainer.val_freq=0 \
    trainer.save_freq=200 \
    trainer.save_checkpoint_path=${EASYR1_DIR}/checkpoints_grounding_3_use_cotdata_and_after_merger_llm_train_new/easy_r1 \
    algorithm.adv_estimator=grpo

# # 训练完成后保存模型
# CHECKPOINT_PATH=${EASYR1_DIR}/checkpoints/easy_r1/medical_grounding_vl_grpo/final
# mkdir -p ${CHECKPOINT_PATH}

# # 复制保存的actor模型到最终目录
# cp -r ${EASYR1_DIR}/checkpoints/easy_r1/medical_grounding_vl_grpo/global_step_* ${CHECKPOINT_PATH}/
# echo "模型已保存至: ${CHECKPOINT_PATH}" 