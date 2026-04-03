#!/bin/bash
# ==============================================================================
# Wikitext2 PPL Evaluation for Draft Model (0.3-bit)
# ==============================================================================
#
# Uses eval.py with LittleBitModel.from_pretrained to load the draft model
# from the Step 1 output directory.
#
# ==============================================================================

set -e

# ===========================
# USER CONFIGURATION
# ===========================

# Step 1 draft model output
DRAFT_MODEL_PATH="/group-volume/ym1012.kim/homepc/LittleSpec/outputs/step1_draft_0.3bit/2026_03_31_19_01"

# Evaluation settings
PPL_TASK="wikitext2"
ZEROSHOT_TASK=""       # Empty = skip zero-shot benchmarks (faster)
SEQLEN=2048

# ===========================
# VALIDATE
# ===========================

if [ ! -d "${DRAFT_MODEL_PATH}" ]; then
    echo "ERROR: Draft model directory not found: ${DRAFT_MODEL_PATH}"
    echo "  Update DRAFT_MODEL_PATH to point to your Step 1 output."
    exit 1
fi

echo "============================================================"
echo "Wikitext2 PPL Evaluation: Draft Model (0.3-bit)"
echo "  Model:    ${DRAFT_MODEL_PATH}"
echo "  PPL Task: ${PPL_TASK}"
echo "  SeqLen:   ${SEQLEN}"
echo "============================================================"

if [ -f "${DRAFT_MODEL_PATH}/littlebit_config.json" ]; then
    echo ""
    echo "LittleBit config:"
    cat "${DRAFT_MODEL_PATH}/littlebit_config.json"
    echo ""
fi

echo "============================================================"

# ===========================
# RUN EVALUATION
# ===========================

python eval.py \
    --model_id "${DRAFT_MODEL_PATH}" \
    --ppl_task "${PPL_TASK}" \
    --zeroshot_task "${ZEROSHOT_TASK}" \
    --seqlen ${SEQLEN} \
    --batch_size 1
