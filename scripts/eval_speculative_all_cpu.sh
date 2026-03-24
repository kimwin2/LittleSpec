#!/bin/bash
# ==============================================================================
# Evaluation: ALL-CPU Speculative Decoding
# ==============================================================================
#
# Draft:  0.1-bit quantized model (CPU LittleBit kernel — FAST)
# Target: Original FP model (CPU PyTorch — SLOW)
#
# This is the correct scenario for CPU speculative decoding:
#   - Target model is slow on CPU (~500ms+/token for 8B FP)
#   - Draft model is fast on CPU (~10-50ms/token with LittleBit binary kernel)
#   - Draft is 10-50x faster → speculative decoding provides real speedup
# ==============================================================================

set -e

# ===========================
# USER CONFIGURATION
# ===========================

MODEL_ID="/group-volume/ym1012.kim/homepc/EAGLE/Llama-3.1-8B-Instruct"

# >>> Set to your CONVERTED runtime checkpoint <<<
DRAFT_MODEL_PATH="/group-volume/ym1012.kim/homepc/LittleSpec/outputs/step1_draft_0.1bit/2026_03_23_13_29_runtime"

# Benchmark settings (reduced for CPU — it's slow!)
BENCHMARK="mt_bench"
MAX_SAMPLES=5
MAX_NEW_TOKENS=64
DRAFT_LENGTHS="4"
MODE="greedy"

# Output
OUTPUT_FILE="eval_results/speculative_all_cpu_eval.json"

# ALL CPU
DEVICE="cpu"
DRAFT_DEVICE="cpu_kernel"

# ===========================
# VALIDATE
# ===========================

if [ ! -d "${DRAFT_MODEL_PATH}" ]; then
    echo "ERROR: Draft model directory not found: ${DRAFT_MODEL_PATH}"
    echo "Run convert_hf_to_runtime.py first!"
    exit 1
fi

# ===========================
# BUILD CPU EXTENSION (if needed)
# ===========================

echo "Building CPU extension..."
cd lb_kernels/littlebit_kernels_cpu
python setup.py build_ext --inplace 2>/dev/null || echo "CPU extension already built"
cd ../..

# ===========================
# RUN
# ===========================

echo "============================================================"
echo "Evaluation: ALL-CPU Speculative Decoding"
echo "  Draft:        ${DRAFT_MODEL_PATH}"
echo "                (CPU LittleBit kernel — binary ops, ~10-50x faster)"
echo "  Target:       ${MODEL_ID}"
echo "                (CPU PyTorch FP — slow baseline)"
echo "  Benchmark:    ${BENCHMARK} (${MAX_SAMPLES} samples)"
echo "  Draft Length:  K=${DRAFT_LENGTHS}"
echo "  Max Tokens:   ${MAX_NEW_TOKENS}"
echo ""
echo "  NOTE: This will be slow! CPU FP inference for 8B model."
echo "        Expected ~30-60 min for ${MAX_SAMPLES} samples."
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
