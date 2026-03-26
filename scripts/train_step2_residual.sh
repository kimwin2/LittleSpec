#!/bin/bash
# ==============================================================================
# Step 2: Train 0.9-bit Residual Model (Matryoshka Speculative Decoding)
# ==============================================================================
#
# This script trains the 0.9-bit residual model using residual weights
# (W_original - W_0.1bit_approx) as initialization.
# The 0.1-bit draft model is frozen; only the residual parameters are trained.
#
# Prerequisites: Step 1 must be completed (0.1-bit draft model trained)
# ==============================================================================

set -e

# ===========================
# USER CONFIGURATION
# ===========================

# Model
MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"

# >>> IMPORTANT: Set this to the actual Step 1 output directory <<<
# Example: DRAFT_MODEL_PATH="outputs/step1_draft_0.1bit/2026_03_23_12_00"
DRAFT_MODEL_PATH="outputs/step1_draft_0.1bit/REPLACE_WITH_TIMESTAMP"

# Dataset
DATASET="openhermes"
DATA_ROOT="./"

# (Optional) Path to local ShareGPT .jsonl file
SHAREGPT_PATH=""

# Output
SAVE_DIR="outputs/step2_residual_0.9bit"

# Quantization
EFF_BIT=0.9
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

# DeepSpeed
NUM_GPUS=4
DS_CONFIG="configs/zero3.json"

# Logging
RUN_NAME="step2_residual_0.9bit"
REPORT="wandb"

# ===========================
# VALIDATE
# ===========================

if [ "${DRAFT_MODEL_PATH}" = "outputs/step1_draft_0.1bit/REPLACE_WITH_TIMESTAMP" ]; then
    echo "ERROR: Please set DRAFT_MODEL_PATH to the actual Step 1 output directory."
    echo "  Example: DRAFT_MODEL_PATH=\"outputs/step1_draft_0.1bit/2026_03_23_12_00\""
    exit 1
fi

if [ ! -d "${DRAFT_MODEL_PATH}" ]; then
    echo "ERROR: Draft model directory not found: ${DRAFT_MODEL_PATH}"
    echo "  Please run Step 1 first: scripts/train_step1_draft.sh"
    exit 1
fi

# ===========================
# LAUNCH TRAINING
# ===========================

echo "============================================================"
echo "Step 2: Training 0.9-bit Residual Model"
echo "  Base Model: ${MODEL_ID}"
echo "  Draft Model: ${DRAFT_MODEL_PATH}"
echo "  Dataset: ${DATASET}"
echo "  Eff Bit: ${EFF_BIT}"
echo "  GPUs: ${NUM_GPUS}"
echo "  Output: ${SAVE_DIR}"
echo "============================================================"

SHAREGPT_ARG=""
if [ -n "${SHAREGPT_PATH}" ]; then
    SHAREGPT_ARG="--sharegpt_path ${SHAREGPT_PATH}"
fi

deepspeed --num_gpus=${NUM_GPUS} train_step2_residual.py \
    --model_id ${MODEL_ID} \
    --draft_model_path ${DRAFT_MODEL_PATH} \
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
    --report ${REPORT}

echo "============================================================"
echo "Step 2 Complete!"
echo "Residual model saved to: ${SAVE_DIR}/"
echo "Draft model at: ${DRAFT_MODEL_PATH}"
echo ""
echo "Next: Run speculative decoding!"
echo "  scripts/run_speculative_decoding.sh"
echo "  scripts/eval_speculative.sh"
echo "============================================================"
