#!/bin/bash
# ==============================================================================
# MT-Bench Quality Evaluation for LittleBit Models
# ==============================================================================
#
# Evaluates the generation quality (not speed) of:
#   1. Draft model (0.1-bit) — standalone
#   2. Target model (draft + 0.9-bit residual) — Matryoshka combined
#   3. (Optional) FP baseline — original full-precision
#
# Supports GPT-4 as judge for automatic scoring.
# ==============================================================================

set -e

# ===========================
# USER CONFIGURATION
# ===========================

# Model
BASE_MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"

# Trained model paths (UPDATE THESE)
DRAFT_MODEL_PATH="outputs/step1_draft_0.1bit/2026_03_28_15_10"
RESIDUAL_MODEL_PATH="outputs/step2_residual_0.9bit/2026_03_29_13_24"

# Which models to evaluate
EVAL_DRAFT="true"      # 0.1-bit draft model
EVAL_TARGET="true"     # draft + residual combined (1.0-bit)
EVAL_FP="false"        # FP baseline (slow, needs full model in memory)

# Generation
MAX_NEW_TOKENS=1024
TEMPERATURE=0.0        # 0.0 = greedy
MAX_QUESTIONS=80       # 80 = full MT-Bench

# GPT-4 Judging (requires OPENAI_API_KEY env var)
JUDGE="false"          # Set to "true" to enable GPT-4 scoring
JUDGE_MODEL="gpt-4o"

# Output
OUTPUT_DIR="eval_results/mt_bench"

# ===========================
# LAUNCH EVALUATION
# ===========================

echo "============================================================"
echo "MT-Bench Quality Evaluation"
echo "  Base Model: ${BASE_MODEL_ID}"
echo "  Draft:      ${DRAFT_MODEL_PATH}"
echo "  Residual:   ${RESIDUAL_MODEL_PATH}"
echo "  Eval Draft:  ${EVAL_DRAFT}"
echo "  Eval Target: ${EVAL_TARGET}"
echo "  Eval FP:     ${EVAL_FP}"
echo "  Questions:   ${MAX_QUESTIONS}"
echo "  Judge:       ${JUDGE} (${JUDGE_MODEL})"
echo "============================================================"

python eval_mt_bench.py \
    --base_model_id ${BASE_MODEL_ID} \
    --draft_model_path ${DRAFT_MODEL_PATH} \
    --residual_model_path ${RESIDUAL_MODEL_PATH} \
    --eval_draft ${EVAL_DRAFT} \
    --eval_target ${EVAL_TARGET} \
    --eval_fp ${EVAL_FP} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature ${TEMPERATURE} \
    --max_questions ${MAX_QUESTIONS} \
    --judge ${JUDGE} \
    --judge_model ${JUDGE_MODEL} \
    --output_dir ${OUTPUT_DIR}

echo "============================================================"
echo "MT-Bench Evaluation Complete!"
echo "  Results saved to: ${OUTPUT_DIR}/"
echo ""
echo "  To score with GPT-4 after generating answers:"
echo "    OPENAI_API_KEY=sk-... python eval_mt_bench.py --judge true ..."
echo "============================================================"
