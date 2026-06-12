#!/bin/bash
# Reproduce Image Forgery Screening paper experiments
# Usage: ./run_experiments.sh /path/to/CASIA2

set -e

DATA_DIR="${1:?Usage: $0 /path/to/CASIA2}"
OUTPUT_DIR="./results"
mkdir -p "$OUTPUT_DIR"

echo "=== FREUID-PRAI-2026 Experiment Reproduction ==="
echo "Data: $DATA_DIR"
echo ""

# Step 1: Extract features
echo "[1/3] Extracting features..."
python src/extract_features.py \
    --data-dir "$DATA_DIR" \
    --output "$OUTPUT_DIR/features.npy" \
    --workers 8

# Step 2: Run JPEG-only stratified CV
echo ""
echo "[2/3] Running JPEG-only stratified 5-fold CV..."
python src/evaluate.py \
    --features "$OUTPUT_DIR/features.npy" \
    --protocol jpeg-only

# Step 3: Run grouped CV (if splits available)
echo ""
echo "[3/3] Running source-aware grouped CV..."
if [ -f "splits/jpeg_group_folds.json" ]; then
    python src/evaluate.py \
        --features "$OUTPUT_DIR/features.npy" \
        --splits splits/jpeg_group_folds.json \
        --protocol grouped
else
    echo "  [SKIP] splits/jpeg_group_folds.json not found"
fi

echo ""
echo "=== Done ==="
