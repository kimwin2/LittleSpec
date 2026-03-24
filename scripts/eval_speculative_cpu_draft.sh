#!/bin/bash
# ==============================================================================
# Evaluation: CPU Kernel Draft + FP Target
# ==============================================================================
#
# Draft:  0.1-bit quantized model (CPU LittleBit kernel)
# Target: Original FP model (GPU)
#
# Measures TPS, acceptance length, speedup with CPU kernel draft.
# ==============================================================================

set -e

# ===========================
# USER CONFIGURATION
# ===========================

MODEL_ID="/group-volume/ym1012.kim/homepc/EAGLE/Llama-3.1-8B-Instruct"

# >>> Set to your CONVERTED runtime checkpoint <<<
DRAFT_MODEL_PATH="outputs/step1_draft_0.1bit/REPLACE_WITH_TIMESTAMP_runtime"

# Benchmark settings
BENCHMARK="mt_bench"
MAX_SAMPLES=20
MAX_NEW_TOKENS=128
DRAFT_LENGTHS="1,3,5,7"
MODE="greedy"

# Output
OUTPUT_FILE="eval_results/speculative_cpu_draft_eval.json"

# Device
DEVICE="cuda"
DRAFT_DEVICE="cpu_kernel"

# ===========================
# VALIDATE
# ===========================

if [[ "${DRAFT_MODEL_PATH}" == *"REPLACE_WITH_TIMESTAMP"* ]]; then
    echo "ERROR: Please set DRAFT_MODEL_PATH to actual converted runtime directory."
    exit 1
fi

if [ ! -d "${DRAFT_MODEL_PATH}" ]; then
    echo "ERROR: Draft model directory not found: ${DRAFT_MODEL_PATH}"
    exit 1
fi

# ===========================
# BUILD CPU EXTENSION (if needed)
# ===========================

echo "Building CPU extension..."
cd lb_kernels/littlebit_kernels_cpu
python setup.py build_ext --inplace 2>/dev/null || echo "CPU extension build skipped (may already be built)"
cd ../..

# ===========================
# RUN
# ===========================

echo "============================================================"
echo "Evaluation: Speculative Decoding (CPU Kernel Draft)"
echo "  Draft:        ${DRAFT_MODEL_PATH} (CPU kernel)"
echo "  Target:       ${MODEL_ID} (FP, GPU)"
echo "  Benchmark:    ${BENCHMARK}"
echo "  Draft Lengths: ${DRAFT_LENGTHS}"
echo "============================================================"

mkdir -p $(dirname ${OUTPUT_FILE})

python eval_speculative.py \
    --base_model_id ${MODEL_ID} \
    --draft_model_path ${DRAFT_MODEL_PATH} \
    --target_mode fp \
    --draft_device ${DRAFT_DEVICE} \
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
