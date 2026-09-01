#!/bin/bash

# ========================================================================
# JMIR Revision: MLLM Training with Weighted Loss
# ========================================================================
# This script trains LLaVA-Med with sample-level weighted loss to handle
# class imbalance in the training data.
#
# Class weights (based on training set distribution):
#   KL-0: 2286 samples -> weight 0.506
#   KL-1: 1046 samples -> weight 1.105
#   KL-2: 1516 samples -> weight 0.762
#   KL-3: 757 samples  -> weight 1.527
#   KL-4: 173 samples  -> weight 6.680
# ========================================================================

# Activate llava environment (REQUIRED)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llava

CUDA_DEVICES='0'
CUDA='0'

# Change to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit

# Set up Python path to include LLaVA library
export PYTHONPATH=/root/autodl-tmp/LLaVA:$PYTHONPATH

# Disable wandb online sync (optional)
export WANDB_MODE=offline
export WANDB_DIR="your_save_dir/wandb"
export WANDB_CACHE_DIR="your_save_dir/wandb_cache"
export WANDB_CONFIG_DIR="your_save_dir/wandb_config"

echo "=========================================="
echo "JMIR Revision: MLLM Weighted Loss Training"
echo "=========================================="
echo "Working directory: $SCRIPT_DIR"
echo "Training script: train_weighted.py"
echo "Modified trainer: llava_trainer_weighted.py"
echo "=========================================="

# ========================================================================
# IMPORTANT: Configure these paths before running
# ========================================================================

# Model paths (CONFIGURED)
MODEL_PATH="your_base_model_path"
VISION_TOWER="your_vision_tower_path"

# Data paths (using full imbalanced dataset)
DATA_PATH="your_training_data_json"
IMAGE_FOLDER="your_image_folder"

# Output directory
OUTPUT_DIR="your_save_dir"

# ========================================================================
# Training Configuration
# ========================================================================

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES WANDB_MODE=offline deepspeed --include localhost:$CUDA \
    "$SCRIPT_DIR/train_weighted.py" \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --mm_projector_lr 2e-5 \
    --deepspeed "$SCRIPT_DIR/zero3.json" \
    --model_name_or_path "$MODEL_PATH" \
    --version v1 \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --vision_tower "$VISION_TOWER" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

echo "=========================================="
echo "Training completed!"
echo "Output saved to: $OUTPUT_DIR"
echo "=========================================="
