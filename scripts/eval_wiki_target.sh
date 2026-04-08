#!/bin/bash
# ==============================================================================
# Wikitext2 PPL Evaluation for Target Model (draft + residual logits)
# ==============================================================================
#
# Target model = draft_logits + residual_logits
# Loads both models separately and sums their logits per-batch.
#
# Usage:
#   bash scripts/eval_wiki_target.sh
#
# ==============================================================================

set -e

# ===========================
# USER CONFIGURATION
# ===========================

BASE_MODEL_ID="/group-volume/ym1012.kim/homepc/LittleSpec/Llama-3.1-8B-Instruct"

# Draft model (0.1-bit, trained with mixed dataset)
DRAFT_MODEL_PATH="/group-volume/ym1012.kim/homepc/LittleSpec/outputs/step1_draft_0.1bit_mixed/2026_04_05_02_58"

# Residual model (1.9-bit, trained with mixed dataset)
RESIDUAL_MODEL_PATH="/group-volume/ym1012.kim/homepc/LittleSpec/outputs/step2_residual_1.9bit_mixed/2026_04_06_14_04"

# Evaluation settings
PPL_TASK="wikitext2"

# ===========================
# VALIDATE
# ===========================

if [ ! -d "${DRAFT_MODEL_PATH}" ]; then
    echo "ERROR: Draft model directory not found: ${DRAFT_MODEL_PATH}"
    exit 1
fi

if [ ! -d "${RESIDUAL_MODEL_PATH}" ]; then
    echo "ERROR: Residual model directory not found: ${RESIDUAL_MODEL_PATH}"
    exit 1
fi

echo "============================================================"
echo "Wikitext2 PPL Evaluation: Target Model (draft + residual)"
echo "  Draft:    ${DRAFT_MODEL_PATH}"
echo "  Residual: ${RESIDUAL_MODEL_PATH}"
echo "  PPL Task: ${PPL_TASK}"
echo "============================================================"

# ===========================
# RUN EVALUATION
# ===========================

python eval_wiki.py \
    --draft_model_path "${DRAFT_MODEL_PATH}" \
    --residual_model_path "${RESIDUAL_MODEL_PATH}" \
    --model_id "${BASE_MODEL_ID}" \
    --ppl_task "${PPL_TASK}" \
    --zeroshot_task ""
