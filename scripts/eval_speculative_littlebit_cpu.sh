#!/bin/bash
# ==============================================================================
# Evaluation: ALL-CPU Speculative Decoding with LittleBit Target
# ==============================================================================
#
# Draft:   0.3-bit quantized model (CPU LittleBit C++ kernel)
# Target:  0.3+1.7-bit combined (CPU LittleBit C++ kernel — TWO models summed)
#
# Both draft and target run entirely on CPU using LittleBit binary kernels.
# Target uses KV cache with common-prefix detection for efficiency.
#
# This is the full LittleBit-on-CPU speculative decoding evaluation:
#   - No GPU required
#   - Draft model is ~10-50x faster than target (fewer bits)
#   - Target model accuracy = draft + residual logits
#   - Measures: TPS, acceptance rate, speedup vs autoregressive baseline
#
# Prerequisites:
#   1. Both models must be converted to runtime format:
#      python convert_hf_to_runtime.py --input_path <HF checkpoint> --output_path <runtime>
#   2. CPU extension must be built:
#      cd lb_kernels/littlebit_kernels_cpu && python setup.py build_ext --inplace
#
# ==============================================================================

set -e

# ===========================
# USER CONFIGURATION
# ===========================

# Base model (for tokenizer, embeddings, lm_head)
MODEL_ID="/group-volume/ym1012.kim/homepc/EAGLE/Llama-3.1-8B-Instruct"

# >>> RUNTIME-CONVERTED checkpoints <<<
DRAFT_RUNTIME_PATH="/group-volume/ym1012.kim/homepc/LittleSpec/outputs/step1_draft_0.3bit/2026_03_31_19_01_runtime"
RESIDUAL_RUNTIME_PATH="/group-volume/ym1012.kim/homepc/LittleSpec/outputs/step2_residual_1.7bit/2026_03_31_19_01_runtime"

# Draft model for cpu_kernel (same as draft runtime path)
DRAFT_MODEL_PATH="${DRAFT_RUNTIME_PATH}"

# Benchmark settings
BENCHMARK="mt_bench"
MAX_SAMPLES=10
MAX_NEW_TOKENS=64
DRAFT_LENGTHS="3,5,7"
MODE="greedy"

# Output
OUTPUT_FILE="eval_results/speculative_littlebit_cpu_eval.json"

# ALL CPU
DEVICE="cpu"
DRAFT_DEVICE="cpu_kernel"
TARGET_MODE="littlebit_cpu"

# ===========================
# VALIDATE
# ===========================

for DIR_PATH in "${DRAFT_RUNTIME_PATH}" "${RESIDUAL_RUNTIME_PATH}"; do
    if [ ! -d "${DIR_PATH}" ]; then
        echo "ERROR: Directory not found: ${DIR_PATH}"
        echo "Run convert_hf_to_runtime.py first!"
        exit 1
    fi
done

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
echo "Evaluation: ALL-CPU Speculative Decoding (LittleBit Target)"
echo "  Draft:        ${DRAFT_RUNTIME_PATH}"
echo "                (CPU LittleBit kernel — 0.3-bit)"
echo "  Target:       draft + residual combined"
echo "                ${RESIDUAL_RUNTIME_PATH}"
echo "                (CPU LittleBit kernel — 0.3+1.7 = 2.0-bit)"
echo "  Base Model:   ${MODEL_ID}"
echo "  Benchmark:    ${BENCHMARK} (${MAX_SAMPLES} samples)"
echo "  Draft Lengths: K=${DRAFT_LENGTHS}"
echo "  Max Tokens:   ${MAX_NEW_TOKENS}"
echo ""
echo "  NOTE: Both draft and target run on CPU."
echo "        Expected time depends on CPU performance."
echo "============================================================"

mkdir -p $(dirname ${OUTPUT_FILE})

python eval_speculative.py \
    --base_model_id ${MODEL_ID} \
    --draft_model_path ${DRAFT_MODEL_PATH} \
    --target_mode ${TARGET_MODE} \
    --draft_runtime_path ${DRAFT_RUNTIME_PATH} \
    --residual_runtime_path ${RESIDUAL_RUNTIME_PATH} \
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
