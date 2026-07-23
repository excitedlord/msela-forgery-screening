# MS-ELA Forgery Screening

**Format-Controlled Multi-Scale JPEG Compression Response Analysis for Image-Level Compression-History Forgery Screening**

Sujith K Mandala — IEEE PRAI 2026

## Overview

This repository contains the exact code, split files, and prediction outputs used to produce all results in the IEEE PRAI 2026 paper. The committed prediction files verify the headline internal and external metrics.

**Primary results on CASIA v2.0 (JPEG-only, format-controlled):**
- AUC = 0.991 (stratified 5-fold CV, 95% CI: 0.990–0.992)
- AUC = 0.974 (source-component-aware grouped 5-fold CV)
- 405 interpretable features, CPU-only, ~0.4s/image

## Repository Structure

```
├── src/
│   ├── extract_features.py          # 405-feature extraction (7 families)
│   ├── evaluate.py                  # Main CV: stratified + grouped
│   ├── evaluate_external.py         # Columbia + CoMoFoD transfer
│   ├── evaluate_normalization.py    # All→JPEG q95/q85/q75 protocols
│   ├── evaluate_robustness.py       # Perturbation robustness
│   ├── run_ablations.py             # Feature family ablation study
│   └── parse_casia_sources.py       # Source-group parsing (connected components)
├── scripts/
│   ├── reproduce_all.sh             # Reproduce ALL paper results end-to-end
│   └── verify_reported_results.py   # Assert metrics match paper tables
├── splits/
│   ├── casia_source_groups.csv      # All 12,614 images: source IDs + component groups
│   ├── casia_jpeg_stratified_5fold.csv   # Stratified fold assignments (9,501 JPEG)
│   └── casia_jpeg_grouped_5fold.csv      # Grouped fold assignments (source-aware)
├── predictions/
│   ├── casia_jpeg_oof_predictions.csv    # Out-of-fold predictions (stratified)
│   ├── casia_grouped_oof_predictions.csv # Out-of-fold predictions (grouped)
│   ├── columbia_predictions.csv          # External transfer predictions
│   └── comofod_predictions.csv           # Negative-control predictions
├── environment/
│   └── requirements-lock.txt        # Exact pinned versions for reproduction
├── requirements.txt                 # Minimum version constraints
└── README.md
```

## Reproduction

### Quick verification (from committed predictions)

```bash
python scripts/verify_reported_results.py --predictions-dir predictions/ --strict
```

This loads the committed CSV files and asserts that all reported metrics match within rounding tolerance.

### Full reproduction from raw data

```bash
# Install exact environment
pip install -r environment/requirements-lock.txt

# Run complete pipeline
./scripts/reproduce_all.sh /path/to/CASIA2 /path/to/Columbia /path/to/CoMoFoD
```

This reproduces: feature extraction, all CV protocols, normalization experiments, robustness evaluation, ablation study, external transfers, and metric verification.

## Feature Families (405 total)

| Family | Features | Description |
|--------|----------|-------------|
| Multi-scale ELA | 266 | 7 JPEG quality levels × (6 global + 32 spatial grid stats) |
| Cross-quality ratios | 16 | 4 pixel-wise ratio maps × (mean, std, p5, p95) |
| ELA entropy | 18 | 4×4 grid Shannon entropy on q=50 ELA map + std + range |
| FFT radial bands | 6 | 5 fractional-energy bands + high/low ratio |
| Edge density | 20 | Sobel magnitude: global stats (3) + 4×4 grid (16) + cross-cell std (1) |
| SRM noise | 27 | 3 kernels × 3 RGB channels × (mean, std, p95) |
| DCT/noise/color | 52 | Blockiness (8) + Laplacian grid (32) + color stats (12) |

## Evaluation Protocol

- **Classifier:** HistGradientBoostingClassifier (1000 max iterations, early stopping patience=50, 10% validation fraction, learning rate 0.03, max depth 8, min 20 samples/leaf, L2=1.0, 255 bins)
- **Class imbalance:** Inverse-frequency sample weighting
- **Threshold selection:** Per-fold F1-optimal threshold on validation predictions; median threshold applied globally for final metrics
- **Confidence intervals:** 2000-iteration stratified bootstrap

## Source-Aware Grouping Protocol

Connected-component analysis on source identifiers parsed from CASIA v2.0 filenames:

1. Authentic: source ID from category-index pattern (e.g., `Au_ani_00018.jpg` → `ani00018`)
2. Tampered: all recoverable source IDs parsed (e.g., `Tp_D_CND_M_N_ani00018_sec00096_00138.tif` → `ani00018`, `sec00096`)
3. IDs co-occurring in the same tampered filename are connected in a source graph
4. Connected components define GroupKFold partition groups (6,522 components)

This prevents images sharing any parsed donor or target identity from appearing in both train and test within the same fold.

## Environment

- Python 3.13.3
- NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.9.0, Pillow 12.2.0
- Hardware: Apple M3 Pro, 12-core, 36 GB unified memory
- All random seeds fixed at 42

## Citation

```bibtex
@inproceedings{mandala2026msela,
  title={Format-Controlled Multi-Scale {JPEG} Compression Response Analysis 
         for Image-Level Compression-History Forgery Screening},
  author={Mandala, Sujith K},
  booktitle={Proceedings of the IEEE International Conference on 
             Pattern Recognition and Artificial Intelligence (PRAI)},
  year={2026}
}
```

## License

This code is released for academic research purposes. See the paper for full methodology details.
