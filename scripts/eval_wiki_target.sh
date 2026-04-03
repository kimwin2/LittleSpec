#!/bin/bash
# ==============================================================================
# Wikitext2 PPL Evaluation for Target Model (0.3-bit draft + 1.7-bit residual)
# ==============================================================================
#
# Uses eval.py with LittleBitModel.from_pretrained to load the combined
# target model from the Step 2 output directory.
#
# The Step 2 checkpoint contains the full 2.0-bit model (draft + residual
# weights merged), so it can be loaded directly as a single model.
#
# ==============================================================================

set -e

# ===========================
# USER CONFIGURATION
# ===========================

# Step 2 residual model output (contains the combined 0.3+1.7 = 2.0-bit target model)
# Update the timestamp to match your actual training output
TARGET_MODEL_PATH="/group-volume/ym1012.kim/homepc/LittleSpec/outputs/step2_residual_1.7bit/2026_03_31_19_01"

# Evaluation settings
PPL_TASK="wikitext2"
ZEROSHOT_TASK=""       # Empty = skip zero-shot benchmarks (faster)
SEQLEN=2048

# ===========================
# VALIDATE
# ===========================

if [ ! -d "${TARGET_MODEL_PATH}" ]; then
    echo "ERROR: Target model directory not found: ${TARGET_MODEL_PATH}"
    echo "  Update TARGET_MODEL_PATH to point to your Step 2 output."
    exit 1
fi

if [ ! -f "${TARGET_MODEL_PATH}/littlebit_config.json" ]; then
    echo "ERROR: No littlebit_config.json found in ${TARGET_MODEL_PATH}"
    echo "  This doesn't look like a valid LittleBit checkpoint."
    exit 1
fi

echo "============================================================"
echo "Wikitext2 PPL Evaluation: Target Model (0.3 + 1.7 = 2.0-bit)"
echo "  Model:    ${TARGET_MODEL_PATH}"
echo "  PPL Task: ${PPL_TASK}"
echo "  SeqLen:   ${SEQLEN}"
echo "============================================================"
echo ""
echo "LittleBit config:"
cat "${TARGET_MODEL_PATH}/littlebit_config.json"
echo ""
echo "============================================================"

# ===========================
# RUN EVALUATION
# ===========================

python eval.py \
    --model_id "${TARGET_MODEL_PATH}" \
    --ppl_task "${PPL_TASK}" \
    --zeroshot_task "${ZEROSHOT_TASK}" \
    --seqlen ${SEQLEN} \
    --batch_size 1
