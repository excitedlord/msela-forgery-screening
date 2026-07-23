#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Reproduce ALL results from IEEE PRAI 2026 paper
# "Format-Controlled Multi-Scale JPEG Compression Response Analysis
#  for Image-Level Compression-History Forgery Screening"
#
# Usage:
#   ./scripts/reproduce_all.sh /path/to/CASIA2 /path/to/Columbia /path/to/CoMoFoD
#
# Arguments:
#   $1 — CASIA2 directory (must contain Au/ and Tp/ subdirectories)
#   $2 — Columbia ImSpliceDataset directory (optional; skip external if omitted)
#   $3 — CoMoFoD directory (optional; skip if omitted)
#
# Outputs saved to ./results/ and ./predictions/
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

CASIA_DIR="${1:?Usage: $0 /path/to/CASIA2 [/path/to/Columbia] [/path/to/CoMoFoD]}"
COLUMBIA_DIR="${2:-}"
COMOFOD_DIR="${3:-}"

RESULTS="./results"
PREDICTIONS="./predictions"
mkdir -p "$RESULTS" "$PREDICTIONS"

echo "═══════════════════════════════════════════════════════════════"
echo "  MS-ELA Forgery Screening — Full Reproduction Pipeline"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  CASIA2:   $CASIA_DIR"
echo "  Columbia: ${COLUMBIA_DIR:-[skipped]}"
echo "  CoMoFoD:  ${COMOFOD_DIR:-[skipped]}"
echo ""
echo "  Python:   $(python --version 2>&1)"
echo "  NumPy:    $(python -c 'import numpy; print(numpy.__version__)')"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Parse source groups and generate split files
# ─────────────────────────────────────────────────────────────────────────────
echo "━━━ [1/9] Parsing source groups ━━━"
python src/parse_casia_sources.py \
    --casia-dir "$CASIA_DIR" \
    --output splits/casia_source_groups.csv

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Extract features — JPEG-only (primary protocol)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "━━━ [2/9] Extracting JPEG-only features (9,501 images) ━━━"
python src/extract_features.py \
    --data-dir "$CASIA_DIR" \
    --output "$RESULTS/features_jpeg_only.npy" \
    --jpeg-only

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Extract features — All formats (for mixed/normalization)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "━━━ [3/9] Extracting all-format features (12,614 images) ━━━"
python src/extract_features.py \
    --data-dir "$CASIA_DIR" \
    --output "$RESULTS/features_all.npy"

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Main evaluation — JPEG-only stratified + grouped CV
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "━━━ [4/9] Main evaluation: JPEG-only stratified & grouped CV ━━━"
python src/evaluate.py \
    --features "$RESULTS/features_jpeg_only.npy" \
    --splits splits/casia_source_groups.csv \
    --protocol both \
    --output-dir "$PREDICTIONS" \
    --bootstrap

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Mixed-format evaluation (all 12,614 images)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "━━━ [5/9] Mixed-format evaluation (all 12,614) ━━━"
python src/evaluate.py \
    --features "$RESULTS/features_all.npy" \
    --protocol jpeg-only \
    --output-dir "$PREDICTIONS" \
    --bootstrap

# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Normalization protocols (q95, q85, q75)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "━━━ [6/9] Normalization evaluation (All→q95, q85, q75) ━━━"
python src/evaluate_normalization.py \
    --data-dir "$CASIA_DIR" \
    --qualities 95 85 75 \
    --output-dir "$RESULTS" \
    --bootstrap

# ─────────────────────────────────────────────────────────────────────────────
# Step 7: Feature ablation study
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "━━━ [7/9] Feature ablation study ━━━"
python src/run_ablations.py \
    --features "$RESULTS/features_jpeg_only.npy" \
    --output "$RESULTS/ablation_results.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Step 8: Robustness evaluation (perturbations)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "━━━ [8/9] Robustness evaluation ━━━"
python src/evaluate_robustness.py \
    --features "$RESULTS/features_jpeg_only.npy" \
    --data-dir "$CASIA_DIR" \
    --output-dir "$RESULTS"

# ─────────────────────────────────────────────────────────────────────────────
# Step 9: External dataset evaluation (Columbia + CoMoFoD)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "━━━ [9/9] External dataset evaluation ━━━"

EXT_ARGS="--casia-features $RESULTS/features_jpeg_only.npy --output-dir $PREDICTIONS"

if [ -n "$COLUMBIA_DIR" ]; then
    echo "  Extracting Columbia features..."
    python src/extract_features.py \
        --data-dir "$COLUMBIA_DIR" \
        --output "$RESULTS/features_columbia.npy" \
        --jpeg-only
    EXT_ARGS="$EXT_ARGS --columbia-features $RESULTS/features_columbia.npy"
fi

if [ -n "$COMOFOD_DIR" ]; then
    echo "  Extracting CoMoFoD features..."
    python src/extract_features.py \
        --data-dir "$COMOFOD_DIR" \
        --output "$RESULTS/features_comofod.npy"
    EXT_ARGS="$EXT_ARGS --comofod-features $RESULTS/features_comofod.npy"
fi

if [ -n "$COLUMBIA_DIR" ] || [ -n "$COMOFOD_DIR" ]; then
    eval python src/evaluate_external.py $EXT_ARGS
else
    echo "  [SKIPPED] No external dataset paths provided"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "━━━ Verifying reported results ━━━"
python scripts/verify_reported_results.py --predictions-dir "$PREDICTIONS"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Reproduction complete. Results in: $RESULTS/"
echo "  Predictions in: $PREDICTIONS/"
echo "═══════════════════════════════════════════════════════════════"
