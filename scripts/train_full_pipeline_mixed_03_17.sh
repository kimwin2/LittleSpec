#!/bin/bash
# ==============================================================================
# Full Pipeline: Train 0.3-bit Draft + 1.7-bit Residual
#   Mixed Dataset: Regenerated OpenHermes 100k + WikiText2 + C4 (half shard 0)
# ==============================================================================
#
# KEY IMPROVEMENT: Uses target-model-regenerated responses instead of original
# GPT-4 responses from OpenHermes. This ensures the draft model learns the
# target model's token distribution, leading to higher acceptance rates in
# speculative decoding (following P-EAGLE / SpecBundle methodology).
#
# Pipeline:
#   Step 0: Regenerate responses using target model (one-time preprocessing)
#   Step 1: Train 0.3-bit draft model (QAT with KD)
#   Step 2: Train 1.7-bit residual model (Matryoshka)
#
# ==============================================================================

set -e

# ===========================
# USER CONFIGURATION
# ===========================

# Model
MODEL_ID="/group-volume/ym1012.kim/homepc/LittleSpec/Llama-3.1-8B-Instruct"

# Dataset: mixed_regen = Regenerated OpenHermes 100k + WikiText2 + C4 half
DATASET="mixed_regen_hermes_wiki_c4"
DATA_ROOT="./"
NUM_SAMPLES=100000   # OpenHermes prompt count for regeneration

# Regeneration settings
REGEN_OUTPUT_DIR="./data/regen_hermes"
REGEN_MAX_NEW_TOKENS=1024
REGEN_BATCH_SIZE=8

# Output directories
STEP1_SAVE_DIR="outputs/step1_draft_0.3bit_regen"
STEP2_SAVE_DIR="outputs/step2_residual_1.7bit_regen"

# Quantization (shared)
QUANT_FUNC="STEBinary"
QUANT_MOD="LittleBitLinear"
RESIDUAL="false"
KV_FACTOR=1.0
MIN_SPLIT_DIM=8

# Step-specific bit widths
STEP1_EFF_BIT=0.3
STEP2_EFF_BIT=1.7

# Training
NUM_EPOCHS=5
BATCH_SIZE=4
GRAD_ACC_STEPS=1
LR=4e-5
WARMUP_RATIO=0.03
L2L_LOSS_SCALE=1.0

# Training-time test (Step 1 only)
TRAIN_TIME_TEST="false"
TTT_STEPS=7
TTT_DECAY=0.8

# DeepSpeed
NUM_GPUS=8
DS_CONFIG="configs/zero3.json"

# Logging
STEP1_RUN_NAME="step1_draft_0.3bit_regen"
STEP2_RUN_NAME="step2_residual_1.7bit_regen"
REPORT="tensorboard"

# Pipeline control
SKIP_STEP0="false"   # Set to "true" to skip regeneration (if already done)
SKIP_STEP1="false"
SKIP_STEP2="false"
DRAFT_MODEL_PATH=""

# ===========================
# STEP 0: Regenerate Responses (one-time preprocessing)
# ===========================

if [ "${SKIP_STEP0}" != "true" ]; then
    echo "============================================================"
    echo "Step 0: Regenerating responses using target model (vLLM)"
    echo "  Model:       ${MODEL_ID}"
    echo "  Prompts:     ${NUM_SAMPLES} from OpenHermes"
    echo "  Output:      ${REGEN_OUTPUT_DIR}"
    echo "  Max tokens:  ${REGEN_MAX_NEW_TOKENS}"
    echo "  Backend:     vLLM (continuous batching)"
    echo "============================================================"

    # Check if already generated
    if [ -f "${REGEN_OUTPUT_DIR}/regen_conversations.jsonl" ]; then
        echo "Regenerated dataset already exists. Skipping Step 0."
        echo "  (Delete ${REGEN_OUTPUT_DIR} to force regeneration)"
    else
        python prepare_regen_dataset.py \
            --model_id ${MODEL_ID} \
            --output_dir ${REGEN_OUTPUT_DIR} \
            --data_root ${DATA_ROOT} \
            --num_samples ${NUM_SAMPLES} \
            --max_new_tokens ${REGEN_MAX_NEW_TOKENS} \
            --backend vllm \
            --tensor_parallel_size 1 \
            --gpu_memory_utilization 0.85
    fi
else
    echo "Skipping Step 0 (regeneration) -- using existing data"
fi

# ===========================
# LAUNCH TRAINING PIPELINE
# ===========================

echo ""
echo "============================================================"
echo "Full Pipeline: Matryoshka Training (Regenerated Dataset)"
echo "  Model:    ${MODEL_ID}"
echo "  Dataset:  ${DATASET}"
echo "    - Regenerated OpenHermes: ${NUM_SAMPLES} (target-model responses)"
echo "    - WikiText2:  full train split"
echo "    - C4:         first shard, 50%"
echo "  Step 1:   ${STEP1_EFF_BIT}-bit draft  -> ${STEP1_SAVE_DIR}"
echo "  Step 2:   ${STEP2_EFF_BIT}-bit residual -> ${STEP2_SAVE_DIR}"
echo "  GPUs:     ${NUM_GPUS}"
echo "============================================================"

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
    --num_samples ${NUM_SAMPLES} \
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
    --train_time_test ${TRAIN_TIME_TEST} \
    --ttt_steps ${TTT_STEPS} \
    --ttt_decay ${TTT_DECAY} \
    --ds_config_path ${DS_CONFIG} \
    --report ${REPORT} \
    --step1_run_name ${STEP1_RUN_NAME} \
    --step2_run_name ${STEP2_RUN_NAME} \
    ${SKIP_STEP1_ARG} \
    ${SKIP_STEP2_ARG} \
    ${DRAFT_MODEL_ARG}
