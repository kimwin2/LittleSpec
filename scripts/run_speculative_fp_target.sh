#!/bin/bash
# ==============================================================================
# Speculative Decoding: FP Target Mode (pre-Step2 benchmarking)
# ==============================================================================
#
# Draft model:  0.1-bit quantized (from Step 1)
# Target model: Original FP (full-precision) model
#
# Use this BEFORE Step 2 (residual training) to measure
# baseline speculative decoding performance.
# ==============================================================================

set -e

# ===========================
# USER CONFIGURATION
# ===========================

# Original FP model (used as both tokenizer source AND target model)
MODEL_ID="/group-volume/ym1012.kim/homepc/EAGLE/Llama-3.1-8B-Instruct"

# >>> Set to your trained Step 1 draft model <<<
DRAFT_MODEL_PATH="outputs/step1_draft_0.1bit/REPLACE_WITH_TIMESTAMP"

# Generation
PROMPT="Write a Python function to compute fibonacci numbers efficiently using memoization."
MAX_NEW_TOKENS=256
DRAFT_LENGTH=5
MODE="greedy"        # "greedy" or "sampling"
TEMPERATURE=1.0
COMPARE_BASELINE="true"

# Device
DEVICE="cuda"

# ===========================
# VALIDATE
# ===========================

if [[ "${DRAFT_MODEL_PATH}" == *"REPLACE_WITH_TIMESTAMP"* ]]; then
    echo "ERROR: Please set DRAFT_MODEL_PATH to actual trained model directory."
    exit 1
fi

if [ ! -d "${DRAFT_MODEL_PATH}" ]; then
    echo "ERROR: Draft model directory not found: ${DRAFT_MODEL_PATH}"
    exit 1
fi

# ===========================
# RUN
# ===========================

echo "============================================================"
echo "Speculative Decoding (FP Target Mode)"
echo "  Draft Model:  ${DRAFT_MODEL_PATH} (0.1-bit)"
echo "  Target Model: ${MODEL_ID} (FP)"
echo "  Draft Length:  ${DRAFT_LENGTH}"
echo "  Mode:         ${MODE}"
echo "============================================================"

python speculative_decoding.py \
    --base_model_id ${MODEL_ID} \
    --draft_model_path ${DRAFT_MODEL_PATH} \
    --target_mode fp \
    --prompt "${PROMPT}" \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --draft_length ${DRAFT_LENGTH} \
    --mode ${MODE} \
    --temperature ${TEMPERATURE} \
    --compare_baseline ${COMPARE_BASELINE} \
    --device ${DEVICE}
