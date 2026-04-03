#!/bin/bash
# ==============================================================================
# Wikitext2 PPL Evaluation for Draft Model
# ==============================================================================

set -e

BASE_MODEL_ID="/group-volume/ym1012.kim/homepc/EAGLE/Llama-3.1-8B-Instruct"
DRAFT_MODEL_PATH="/group-volume/ym1012.kim/homepc/LittleSpec/outputs/step1_draft_0.3bit/2026_03_31_19_01"

python eval_wiki.py \
    --draft_model_path "${DRAFT_MODEL_PATH}" \
    --model_id "${BASE_MODEL_ID}" \
    --ppl_task wikitext2 \
    --zeroshot_task ""
