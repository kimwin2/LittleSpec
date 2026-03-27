#!/bin/bash
# ==============================================================================
# Dataset Preparation: Download OpenHermes 2.5
# ==============================================================================
#
# Downloads and caches the OpenHermes 2.5 dataset from HuggingFace.
# Tokenization happens automatically at training time with the model's
# chat template (e.g. Llama 3.1 format).
#
# Run this ONCE before training.
# ==============================================================================

set -e

# ===========================
# USER CONFIGURATION
# ===========================

# Where to save dataset cache
OUTPUT_DIR="./data"

# Number of samples to use from OpenHermes 2.5 (~1M total)
NUM_SAMPLES=50000

# Random seed for sampling
SEED=42

# Dataset mode: "openhermes" (default) or "legacy" (wikitext2 + ShareGPT)
MODE="openhermes"

# (Legacy mode only) Path to local ShareGPT .jsonl/.json file
# SHAREGPT_PATH=""

# ===========================
# RUN
# ===========================

echo "============================================================"
echo "Dataset Preparation"
echo "  Mode: ${MODE}"
echo "  Output: ${OUTPUT_DIR}"
if [ "${MODE}" = "openhermes" ]; then
    echo "  Samples: ${NUM_SAMPLES}"
fi
echo "============================================================"

EXTRA_ARGS=""
if [ "${MODE}" = "legacy" ] && [ -n "${SHAREGPT_PATH:-}" ]; then
    EXTRA_ARGS="--sharegpt_path ${SHAREGPT_PATH}"
fi

python prepare_datasets.py \
    --output_dir ${OUTPUT_DIR} \
    --mode ${MODE} \
    --num_samples ${NUM_SAMPLES} \
    --seed ${SEED} \
    ${EXTRA_ARGS}

echo "============================================================"
echo "Done! Now run training with:"
echo "  --dataset openhermes"
echo "============================================================"
