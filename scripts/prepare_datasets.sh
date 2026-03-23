#!/bin/bash
# ==============================================================================
# Dataset Preparation: Download wikitext2 + ShareGPT (raw text)
# ==============================================================================
#
# Downloads raw text only (NO tokenization).
# Tokenization happens automatically at training time with the model's tokenizer.
# This means you can change models without re-downloading data.
#
# Run this ONCE before training.
# ==============================================================================

set -e

# ===========================
# USER CONFIGURATION
# ===========================

# Where to save raw text files
OUTPUT_DIR="./data"

# (Optional) Path to local ShareGPT .jsonl/.json file
# If not set, will download from HuggingFace automatically
SHAREGPT_PATH=""
# Example: SHAREGPT_PATH="/data/sharegpt_train.jsonl"

# ===========================
# RUN
# ===========================

echo "============================================================"
echo "Dataset Preparation (raw text download)"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================================"

SHAREGPT_ARG=""
if [ -n "${SHAREGPT_PATH}" ]; then
    SHAREGPT_ARG="--sharegpt_path ${SHAREGPT_PATH}"
fi

python prepare_datasets.py \
    --output_dir ${OUTPUT_DIR} \
    ${SHAREGPT_ARG}

echo "============================================================"
echo "Done! Now run training with:"
echo "  --sharegpt_path ${OUTPUT_DIR}/sharegpt_raw.txt"
echo "============================================================"
