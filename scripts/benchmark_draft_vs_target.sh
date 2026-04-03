#!/bin/bash
# ==============================================================================
# Benchmark: Draft vs Target model speed on CPU
# ==============================================================================
#
# Measures autoregressive generation speed (TPS) for:
#   - Draft model (0.3-bit) — CPUDraftModel with C++ kernel
#   - Target model (0.3+1.7-bit) — CPUTargetModel with C++ kernel
#
# Reports:
#   - ms/token, tokens/second for each model
#   - Draft/Target speed ratio (how much faster draft is)
#
# ==============================================================================

set -e

# ===========================
# USER CONFIGURATION
# ===========================

MODEL_ID="/group-volume/ym1012.kim/homepc/EAGLE/Llama-3.1-8B-Instruct"
DRAFT_RUNTIME_PATH="/group-volume/ym1012.kim/homepc/LittleSpec/outputs/step1_draft_0.3bit/2026_03_31_19_01_runtime"
RESIDUAL_RUNTIME_PATH="/group-volume/ym1012.kim/homepc/LittleSpec/outputs/step2_residual_1.7bit/2026_03_31_19_01_runtime"

NUM_TOKENS=64
WARMUP=5

# ===========================
# BUILD CPU EXTENSION
# ===========================

echo "Building CPU extension..."
cd lb_kernels/littlebit_kernels_cpu
python setup.py build_ext --inplace 2>/dev/null || echo "CPU extension already built"
cd ../..

# ===========================
# RUN
# ===========================

python benchmark_models_cpu.py \
    --base_model_id ${MODEL_ID} \
    --draft_runtime_path ${DRAFT_RUNTIME_PATH} \
    --residual_runtime_path ${RESIDUAL_RUNTIME_PATH} \
    --num_tokens ${NUM_TOKENS} \
    --warmup_tokens ${WARMUP}
