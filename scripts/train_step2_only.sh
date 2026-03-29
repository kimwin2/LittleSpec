#!/bin/bash
# ==============================================================================
# Step 2 Only: Train 0.9-bit Residual Model (skip Step 1)
# ==============================================================================
#
# Use this when Step 1 (draft model) already completed successfully
# but Step 2 failed or needs to be re-run.
#
# ==============================================================================

set -e

# ===========================
# USER CONFIGURATION
# ===========================

# Model
MODEL_ID="/group-volume/ym1012.kim/homepc/EAGLE/Llama-3.1-8B-Instruct"

# Dataset
DATASET="openhermes"
DATA_ROOT="./"

# Draft model from completed Step 1
DRAFT_MODEL_PATH="outputs/step1_draft_0.1bit/2026_03_28_15_10"

# Output directories
STEP1_SAVE_DIR="outputs/step1_draft_0.1bit"
STEP2_SAVE_DIR="outputs/step2_residual_0.9bit"

# Quantization
QUANT_FUNC="STEBinary"
QUANT_MOD="LittleBitLinear"
RESIDUAL="false"
KV_FACTOR=1.0
MIN_SPLIT_DIM=8

# Step-specific bit widths
STEP1_EFF_BIT=0.1
STEP2_EFF_BIT=0.9

# Training
NUM_EPOCHS=5
BATCH_SIZE=4
GRAD_ACC_STEPS=1
LR=4e-5
WARMUP_RATIO=0.03
L2L_LOSS_SCALE=1.0

# Training-time test
TRAIN_TIME_TEST="false"
TTT_STEPS=7
TTT_DECAY=0.8

# DeepSpeed
NUM_GPUS=4
DS_CONFIG="configs/zero3.json"

# Logging
STEP1_RUN_NAME="step1_draft_0.1bit"
STEP2_RUN_NAME="step2_residual_0.9bit"
REPORT="tensorboard"

# ===========================
# LAUNCH (Step 2 only)
# ===========================

echo "============================================================"
echo "Step 2 Only: Train 0.9-bit Residual Model"
echo "  Model:       ${MODEL_ID}"
echo "  Dataset:     ${DATASET}"
echo "  Draft model: ${DRAFT_MODEL_PATH}"
echo "  Output:      ${STEP2_SAVE_DIR}"
echo "  GPUs:        ${NUM_GPUS}"
echo "============================================================"

deepspeed --num_gpus=${NUM_GPUS} train_full_pipeline.py \
    --model_id ${MODEL_ID} \
    --dataset ${DATASET} \
    --data_root ${DATA_ROOT} \
    --step1_save_dir ${STEP1_SAVE_DIR} \
    --step2_save_dir ${STEP2_SAVE_DIR} \
    --step1_eff_bit ${STEP1_EFF_BIT} \
    --step2_eff_bit ${STEP2_EFF_BIT} \
    --quant_func ${QUANT_FUNC} \
    --quant_mod ${QUANT_MOD} \
    --residual ${RESIDUAL} \
    --kv_factor ${KV_FACTOR} \
    --min_split_dim ${MIN_SPLIT_DIM} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACC_STEPS} \
    --lr ${LR} \
    --warmup_ratio ${WARMUP_RATIO} \
    --l2l_loss_scale ${L2L_LOSS_SCALE} \
    --ds_config_path ${DS_CONFIG} \
    --step1_run_name ${STEP1_RUN_NAME} \
    --step2_run_name ${STEP2_RUN_NAME} \
    --report ${REPORT} \
    --train_time_test ${TRAIN_TIME_TEST} \
    --ttt_steps ${TTT_STEPS} \
    --ttt_decay ${TTT_DECAY} \
    --skip_step1 \
    --draft_model_path ${DRAFT_MODEL_PATH}

echo "============================================================"
echo "Step 2 Complete!"
echo "  Residual model: ${STEP2_SAVE_DIR}/"
echo "  Draft model:    ${DRAFT_MODEL_PATH}"
echo "============================================================"
