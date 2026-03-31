#!/bin/bash
# ==============================================================================
# MT-Bench Speculative Decoding Evaluation
# ==============================================================================
#
# Runs serial speculative decoding with:
#   - Draft model on CPU (C++ LittleBit kernel)
#   - Target model (Matryoshka 0.1+0.9 bit) on GPU
#
# Measures TPS, acceptance rate, mean acceptance length.
# Uses MT-Bench 80 questions (Turn 1 only).
# No LLM judge.
#
# ==============================================================================

set -e

# ===========================
# USER CONFIGURATION
# ===========================

# Base model (for tokenizer)
BASE_MODEL_ID="/group-volume/ym1012.kim/homepc/EAGLE/Llama-3.1-8B-Instruct"

# Draft model: runtime format (converted for CPU kernel)
# If you haven't converted yet, run:
#   python convert_hf_to_runtime.py \
#     --input_path outputs/step1_draft_0.1bit/2026_03_28_15_10 \
#     --output_path outputs/step1_draft_0.1bit/2026_03_28_15_10_runtime
DRAFT_MODEL_PATH="outputs/step1_draft_0.1bit/2026_03_28_15_10_runtime"

# Residual model (HF format, used on GPU)
RESIDUAL_MODEL_PATH="outputs/step2_residual_0.9bit/2026_03_29_13_24"

# Target mode: "matryoshka" = draft + residual combined
TARGET_MODE="matryoshka"

# Draft device: "cpu_kernel" for C++ LittleBit kernel
DRAFT_DEVICE="cpu_kernel"

# Speculative decoding params
DRAFT_LENGTH=5          # K: number of draft tokens per step
MAX_NEW_TOKENS=512
MAX_QUESTIONS=3         # Set to 80 for full eval, 3 for quick test

# Also run autoregressive baseline for speedup comparison?
RUN_BASELINE="true"

# Mode
MODE="greedy"

# Output
OUTPUT_DIR="eval_results/speculative_mt_bench"

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
echo "MT-Bench Speculative Decoding Evaluation"
echo "  Base Model:  ${BASE_MODEL_ID}"
echo "  Draft:       ${DRAFT_MODEL_PATH} (${DRAFT_DEVICE})"
echo "  Residual:    ${RESIDUAL_MODEL_PATH}"
echo "  Target Mode: ${TARGET_MODE}"
echo "  K:           ${DRAFT_LENGTH}"
echo "  Max Tokens:  ${MAX_NEW_TOKENS}"
echo "  Questions:   ${MAX_QUESTIONS}"
echo "  Baseline:    ${RUN_BASELINE}"
echo "============================================================"

mkdir -p "${OUTPUT_DIR}"

python eval_speculative_mt_bench.py \
    --base_model_id "${BASE_MODEL_ID}" \
    --draft_model_path "${DRAFT_MODEL_PATH}" \
    --residual_model_path "${RESIDUAL_MODEL_PATH}" \
    --target_mode "${TARGET_MODE}" \
    --draft_device "${DRAFT_DEVICE}" \
    --draft_length ${DRAFT_LENGTH} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --max_questions ${MAX_QUESTIONS} \
    --mode "${MODE}" \
    --run_baseline "${RUN_BASELINE}" \
    --output_dir "${OUTPUT_DIR}"

echo "============================================================"
echo "Evaluation Complete!"
echo "  Results: ${OUTPUT_DIR}/speculative_mt_bench_results.json"
echo "============================================================"
