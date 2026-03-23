#!/bin/bash
# ==============================================================================
# Evaluation: FP Target Mode (pre-Step2 benchmarking)
# ==============================================================================
#
# Draft:  0.1-bit quantized model
# Target: Original FP model
#
# Measure acceptance length, speedup, etc. before residual training.
# ==============================================================================

set -e

# ===========================
# USER CONFIGURATION
# ===========================

MODEL_ID="/group-volume/ym1012.kim/homepc/EAGLE/Llama-3.1-8B-Instruct"

# >>> Set to your trained Step 1 draft model <<<
DRAFT_MODEL_PATH="outputs/step1_draft_0.1bit/REPLACE_WITH_TIMESTAMP"

# Benchmark settings
BENCHMARK="mt_bench"  # "all", "mt_bench", "gsm8k", "humaneval", "summarization"
MAX_SAMPLES=20
MAX_NEW_TOKENS=128
DRAFT_LENGTHS="1,3,5,7"
MODE="greedy"

# Output
OUTPUT_FILE="eval_results/speculative_fp_target_eval.json"

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
echo "Evaluation: Speculative Decoding (FP Target)"
echo "  Draft:        ${DRAFT_MODEL_PATH} (0.1-bit)"
echo "  Target:       ${MODEL_ID} (FP)"
echo "  Benchmark:    ${BENCHMARK}"
echo "  Draft Lengths: ${DRAFT_LENGTHS}"
echo "============================================================"

mkdir -p $(dirname ${OUTPUT_FILE})

python eval_speculative.py \
    --base_model_id ${MODEL_ID} \
    --draft_model_path ${DRAFT_MODEL_PATH} \
    --target_mode fp \
    --benchmark ${BENCHMARK} \
    --max_samples ${MAX_SAMPLES} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --draft_lengths ${DRAFT_LENGTHS} \
    --mode ${MODE} \
    --output_file ${OUTPUT_FILE} \
    --device ${DEVICE}

echo "============================================================"
echo "Evaluation Complete! Results: ${OUTPUT_FILE}"
echo "============================================================"
