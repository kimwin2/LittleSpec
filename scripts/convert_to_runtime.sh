#!/bin/bash
# ==============================================================================
# Convert HF LittleBit checkpoint to CPU runtime format
# ==============================================================================

set -e

# >>> Set to your trained Step 1 draft model <<<
INPUT_PATH="outputs/step1_draft_0.1bit/REPLACE_WITH_TIMESTAMP"
OUTPUT_PATH="${INPUT_PATH}_runtime"

# ==============================================================================
# VALIDATE
# ==============================================================================

if [[ "${INPUT_PATH}" == *"REPLACE_WITH_TIMESTAMP"* ]]; then
    echo "ERROR: Please set INPUT_PATH to actual trained model directory."
    exit 1
fi

# ==============================================================================
# RUN
# ==============================================================================

echo "============================================================"
echo "Converting HF checkpoint to CPU runtime format"
echo "  Input:  ${INPUT_PATH}"
echo "  Output: ${OUTPUT_PATH}"
echo "============================================================"

python convert_hf_to_runtime.py \
    --input_path ${INPUT_PATH} \
    --output_path ${OUTPUT_PATH}

echo "============================================================"
echo "Conversion Complete!"
echo "Runtime checkpoint: ${OUTPUT_PATH}"
echo "============================================================"
