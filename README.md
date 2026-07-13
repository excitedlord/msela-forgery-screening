# IEEE-PRAI-2026

**Format-Controlled Multi-Scale JPEG Compression Response Analysis for Image-Level Compression-History Forgery Screening**

Sujith K Mandala — IEEE PRAI 2026

## Overview

This repository contains code and reproducibility artifacts for the IEEE PRAI 2026 paper on format-controlled JPEG compression response analysis for image-level compression-history forgery screening.

Key results on CASIA v2.0 (JPEG-only, format-controlled):
- **AUC = 0.990** (stratified 5-fold CV)
- **AUC = 0.976** (source-component-aware grouped CV)
- 405 interpretable features, CPU-only, <0.4s/image

## Repository Structure

```
├── src/
│   ├── extract_features.py         # 405-feature extraction pipeline
│   ├── evaluate.py                 # Cross-validation + metrics
│   └── parse_casia_sources.py      # Source-group parsing (connected components)
├── splits/
│   ├── jpeg_group_folds.json       # Legacy group split indices
│   ├── casia_source_groups.csv     # All 12,614 images with parsed source IDs + component groups
│   ├── casia_jpeg_stratified_5fold.csv  # Stratified fold assignments (9,501 JPEG images)
│   └── casia_jpeg_grouped_5fold.csv     # Grouped fold assignments (source-component-aware)
├── predictions/
│   ├── casia_jpeg_oof_predictions.csv   # Out-of-fold predictions (stratified)
│   ├── casia_grouped_oof_predictions.csv # Out-of-fold predictions (grouped)
│   ├── columbia_predictions.csv         # External transfer predictions
│   └── comofod_predictions.csv          # Negative-control predictions
├── scripts/
│   ├── run_experiments.sh          # Reproduce all paper results
│   └── generate_repo_artifacts.py  # Generate splits + predictions from raw data
├── requirements.txt
└── README.md
```

## Source-Aware Grouping Protocol

The grouped split uses connected-component analysis on source identifiers parsed from CASIA v2.0 filenames:

1. Authentic images are assigned a source ID from the category-index pattern (e.g., `Au_ani_00018.jpg` → `ani00018`)
2. Tampered images have all recoverable source IDs parsed (e.g., `Tp_D_CND_M_N_ani00018_sec00096_00138.tif` → `ani00018`, `sec00096`)
3. IDs co-occurring in the same tampered filename are connected in a source graph
4. Connected components define GroupKFold partition groups

This ensures that images sharing any parsed donor or target identity are assigned to the same fold. See `src/parse_casia_sources.py` for implementation details.

## Requirements

- Python 3.10+
- See `requirements.txt`

## Usage

```bash
pip install -r requirements.txt

# Extract features
python src/extract_features.py --data-dir /path/to/CASIA2 --output features.npy

# Generate source groups
python src/parse_casia_sources.py --casia-dir /path/to/CASIA2 --output splits/casia_source_groups.csv

# Run evaluation
python src/evaluate.py --features features.npy --protocol jpeg-only

# Regenerate all artifacts (requires CASIA data + feature cache)
python scripts/generate_repo_artifacts.py --casia-dir /path/to/CASIA2 --cache-dir /path/to/cache
```

## Key Files for Reproducibility Verification

| File | Purpose |
|------|---------|
| `splits/casia_source_groups.csv` | All 12,614 images with parsed source IDs and component group assignments |
| `splits/casia_jpeg_stratified_5fold.csv` | Fold assignments for the primary stratified evaluation |
| `splits/casia_jpeg_grouped_5fold.csv` | Fold assignments for the source-component-aware evaluation |
| `predictions/casia_jpeg_oof_predictions.csv` | Per-image out-of-fold scores (stratified protocol) |
| `predictions/casia_grouped_oof_predictions.csv` | Per-image out-of-fold scores (grouped protocol) |
| `predictions/columbia_predictions.csv` | External transfer predictions on Columbia |
| `predictions/comofod_predictions.csv` | Negative-control predictions on CoMoFoD |

## Citation

```bibtex
@inproceedings{mandala2026prai,
  title={Format-Controlled Multi-Scale JPEG Compression Response Analysis for Image-Level Compression-History Forgery Screening},
  author={Mandala, Sujith K},
  booktitle={Proc. IEEE International Conference on Pattern Recognition and Artificial Intelligence (PRAI)},
  year={2026}
}
```

## License

MIT
