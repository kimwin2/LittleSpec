#!/bin/bash
# ==============================================================================
# Wikitext2 PPL Evaluation for Draft Model
# ==============================================================================

set -e

# ===========================
# USER CONFIGURATION
# ===========================

BASE_MODEL_ID="/group-volume/ym1012.kim/homepc/LittleSpec/Llama-3.1-8B-Instruct"

# Step 1 draft model output (0.1-bit, mixed dataset)
DRAFT_MODEL_PATH="/group-volume/ym1012.kim/homepc/LittleSpec/outputs/step1_draft_0.1bit_mixed/2026_04_05_02_58"

# Evaluation settings
PPL_TASK="wikitext2"
ZEROSHOT_TASK=""       # Empty = skip zero-shot benchmarks (faster)
SEQLEN=2048

# ===========================
# VALIDATE
# ===========================

if [ ! -d "${DRAFT_MODEL_PATH}" ]; then
    echo "ERROR: Draft model directory not found: ${DRAFT_MODEL_PATH}"
    exit 1
fi

echo "============================================================"
echo "Wikitext2 PPL Evaluation: Draft Model"
echo "  Model:    ${DRAFT_MODEL_PATH}"
echo "  Base:     ${BASE_MODEL_ID}"
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
    --base_model_id "${BASE_MODEL_ID}" \
    --ppl_task "${PPL_TASK}" \
    --zeroshot_task "${ZEROSHOT_TASK}" \
    --seqlen ${SEQLEN} \
    --batch_size 1
