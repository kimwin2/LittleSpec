#!/bin/bash
# ==============================================================================
# WikiText2 PPL Evaluation for Legacy (LittleBitITQSpec) Checkpoints
# ==============================================================================
#
# Loads the old-format checkpoint (single model with Matryoshka draft+residual)
# and measures both draft-only PPL and target (draft+residual) PPL.
#
# The old checkpoint uses LittleBitITQSpecLinear with resume_eff_bit, where
# the draft and residual paths are inside the same model.
#
# Usage:
#   bash scripts/eval_wiki_legacy.sh
#
# ==============================================================================

set -e

# ===========================
# USER CONFIGURATION
# ===========================

# Path to the legacy checkpoint (on server)
# This is the old checkpoint path from eval.sh reference:
MODEL_PATH="/home/gpu1/emtechllm/bs93.lee/LittleSpec/ckpts_lb_plus/llama2_7b/spec_llama2_7b_total1p0bit_resume0p1_SmoothSign_lr4e-5_bs4_wRes_LittleBitITQLinear/2026_02_20_00_03"

# Base model for tokenizer (adjust to match the model used for the checkpoint)
# For Llama-2-7b checkpoints:
BASE_MODEL_ID="meta-llama/Llama-2-7b-hf"
# For Llama-3.1-8B checkpoints, uncomment:
# BASE_MODEL_ID="/group-volume/ym1012.kim/homepc/LittleSpec/Llama-3.1-8B-Instruct"

# Quantization parameters (must match training config)
QUANT_FUNC="SmoothSign"
EFF_BIT=1.0              # Total effective bits (draft + residual)
RESUME_EFF_BIT=0.1       # Draft effective bits

# Evaluation settings
PPL_TASK="wikitext2"
SEQLEN=2048

# ===========================
# VALIDATE
# ===========================

if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Model directory not found: ${MODEL_PATH}"
    echo "Make sure you're running this on the server where the checkpoint exists."
    exit 1
fi

echo "============================================================"
echo "Legacy Checkpoint PPL Evaluation"
echo "  Checkpoint:     ${MODEL_PATH}"
echo "  Base Model:     ${BASE_MODEL_ID}"
echo "  quant_func:     ${QUANT_FUNC}"
echo "  eff_bit:        ${EFF_BIT}"
echo "  resume_eff_bit: ${RESUME_EFF_BIT}"
echo "  PPL Task:       ${PPL_TASK}"
echo "  SeqLen:         ${SEQLEN}"
echo "============================================================"

# ===========================
# RUN EVALUATION
# ===========================

python eval_wiki_legacy.py \
    --model_path "${MODEL_PATH}" \
    --base_model_id "${BASE_MODEL_ID}" \
    --quant_func "${QUANT_FUNC}" \
    --eff_bit ${EFF_BIT} \
    --resume_eff_bit ${RESUME_EFF_BIT} \
    --ppl_task "${PPL_TASK}" \
    --seqlen ${SEQLEN}
