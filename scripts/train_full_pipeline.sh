#!/bin/bash
# ==============================================================================
# Full Pipeline: Train 0.1-bit Draft + 0.9-bit Residual (Matryoshka Style)
# ==============================================================================
#
# This script runs the full Matryoshka training pipeline in one go:
#   Step 1: Train 0.1-bit draft model via QAT (knowledge distillation)
#   Step 2: Train 0.9-bit residual model using draft model from Step 1
#
# The draft model checkpoint path is automatically passed from Step 1 to Step 2.
# No manual intervention required between steps.
#
# Flags:
#   --skip_step1            Skip Step 1 and use an existing draft model
#   --skip_step2            Only run Step 1
#   --draft_model_path      Path to existing draft model (for --skip_step1)
#
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
SHAREGPT_PATH=""

# Output directories
STEP1_SAVE_DIR="outputs/step1_draft_0.1bit"
STEP2_SAVE_DIR="outputs/step2_residual_0.9bit"

# Quantization (shared)
QUANT_FUNC="STEBinary"
QUANT_MOD="LittleBitLinear"
RESIDUAL="false"
KV_FACTOR=1.0
MIN_SPLIT_DIM=8

# Step-specific bit widths
STEP1_EFF_BIT=0.1
STEP2_EFF_BIT=0.9

# Training
NUM_EPOCHS=3
BATCH_SIZE=4
GRAD_ACC_STEPS=1
LR=2e-5
WARMUP_RATIO=0.03
L2L_LOSS_SCALE=10.0

# Training-time test (Step 1 only, EAGLE-style multi-step rollout)
TRAIN_TIME_TEST="false"
TTT_STEPS=7
TTT_DECAY=0.8

# DeepSpeed
NUM_GPUS=4
DS_CONFIG="configs/zero3.json"

# Logging
STEP1_RUN_NAME="step1_draft_0.1bit"
STEP2_RUN_NAME="step2_residual_0.9bit"
REPORT="wandb"

# Pipeline control (set to "true" to skip a step)
SKIP_STEP1="false"
SKIP_STEP2="false"

# (Required only if SKIP_STEP1="true")
# DRAFT_MODEL_PATH="outputs/step1_draft_0.1bit/<timestamp>"
DRAFT_MODEL_PATH=""

# ===========================
# LAUNCH PIPELINE
# ===========================

echo "============================================================"
echo "Full Pipeline: Matryoshka Speculative Decoding Training"
echo "  Model:    ${MODEL_ID}"
echo "  Dataset:  ${DATASET}"
echo "  Step 1:   ${STEP1_EFF_BIT}-bit draft  -> ${STEP1_SAVE_DIR}"
echo "  Step 2:   ${STEP2_EFF_BIT}-bit residual -> ${STEP2_SAVE_DIR}"
echo "  GPUs:     ${NUM_GPUS}"
echo "  Skip Step1: ${SKIP_STEP1}"
echo "  Skip Step2: ${SKIP_STEP2}"
echo "============================================================"

SHAREGPT_ARG=""
if [ -n "${SHAREGPT_PATH}" ]; then
    SHAREGPT_ARG="--sharegpt_path ${SHAREGPT_PATH}"
fi

SKIP_STEP1_ARG=""
if [ "${SKIP_STEP1}" = "true" ]; then
    SKIP_STEP1_ARG="--skip_step1"
fi

SKIP_STEP2_ARG=""
if [ "${SKIP_STEP2}" = "true" ]; then
    SKIP_STEP2_ARG="--skip_step2"
fi

DRAFT_MODEL_ARG=""
if [ -n "${DRAFT_MODEL_PATH}" ]; then
    DRAFT_MODEL_ARG="--draft_model_path ${DRAFT_MODEL_PATH}"
fi

deepspeed --num_gpus=${NUM_GPUS} train_full_pipeline.py \
    --model_id ${MODEL_ID} \
    --dataset ${DATASET} \
    --data_root ${DATA_ROOT} \
    ${SHAREGPT_ARG} \
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
    ${SKIP_STEP1_ARG} \
    ${SKIP_STEP2_ARG} \
    ${DRAFT_MODEL_ARG}

echo "============================================================"
echo "Full Pipeline Complete!"
echo "  Step 1 outputs: ${STEP1_SAVE_DIR}/"
echo "  Step 2 outputs: ${STEP2_SAVE_DIR}/"
echo ""
echo "Next: Run speculative decoding!"
echo "  scripts/run_speculative_decoding.sh"
echo "  scripts/eval_speculative.sh"
echo "============================================================"
