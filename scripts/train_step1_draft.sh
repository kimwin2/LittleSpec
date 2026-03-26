#!/bin/bash
# ==============================================================================
# Step 1: Train 0.1-bit Draft Model (Matryoshka Speculative Decoding)
# ==============================================================================
#
# This script trains the 0.1-bit draft model via QAT with knowledge distillation.
# The draft model will be used for fast token generation in speculative decoding.
#
# Configurable variables are at the top of this script.
# ==============================================================================

set -e

# ===========================
# USER CONFIGURATION
# ===========================

# Model
MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"

# Dataset: "openhermes" uses OpenHermes 2.5 with Llama 3.1 chat template
DATASET="openhermes"
DATA_ROOT="./"

# (Optional) Path to local ShareGPT .jsonl file
# If not set, will attempt to download from HuggingFace
SHAREGPT_PATH=""
# Example: SHAREGPT_PATH="/data/sharegpt_train.jsonl"

# Output
SAVE_DIR="outputs/step1_draft_0.1bit"

# Quantization
EFF_BIT=0.1
QUANT_FUNC="STEBinary"
QUANT_MOD="LittleBitLinear"
RESIDUAL="false"
KV_FACTOR=1.0
MIN_SPLIT_DIM=8

# Training
NUM_EPOCHS=3
BATCH_SIZE=4
GRAD_ACC_STEPS=1
LR=2e-5
WARMUP_RATIO=0.03
L2L_LOSS_SCALE=10.0

# Training-time test (EAGLE-style multi-step rollout)
# Set to "true" to train with multi-step prediction
TRAIN_TIME_TEST="false"
TTT_STEPS=7          # Number of rollout steps
TTT_DECAY=0.8        # Exponential decay for each step's loss weight

# DeepSpeed
NUM_GPUS=4
DS_CONFIG="configs/zero3.json"

# Logging
RUN_NAME="step1_draft_0.1bit"
REPORT="wandb"

# ===========================
# LAUNCH TRAINING
# ===========================

echo "============================================================"
echo "Step 1: Training 0.1-bit Draft Model"
echo "  Model: ${MODEL_ID}"
echo "  Dataset: ${DATASET}"
echo "  Eff Bit: ${EFF_BIT}"
echo "  GPUs: ${NUM_GPUS}"
echo "  Output: ${SAVE_DIR}"
echo "  Train-time test: ${TRAIN_TIME_TEST} (steps=${TTT_STEPS}, decay=${TTT_DECAY})"
echo "============================================================"

SHAREGPT_ARG=""
if [ -n "${SHAREGPT_PATH}" ]; then
    SHAREGPT_ARG="--sharegpt_path ${SHAREGPT_PATH}"
fi

deepspeed --num_gpus=${NUM_GPUS} train_step1_draft.py \
    --model_id ${MODEL_ID} \
    --dataset ${DATASET} \
    --data_root ${DATA_ROOT} \
    ${SHAREGPT_ARG} \
    --save_dir ${SAVE_DIR} \
    --eff_bit ${EFF_BIT} \
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
    --run_name ${RUN_NAME} \
    --report ${REPORT} \
    --train_time_test ${TRAIN_TIME_TEST} \
    --ttt_steps ${TTT_STEPS} \
    --ttt_decay ${TTT_DECAY}

echo "============================================================"
echo "Step 1 Complete!"
echo "Draft model saved to: ${SAVE_DIR}/"
echo "Next: Run scripts/train_step2_residual.sh"
echo "============================================================"
