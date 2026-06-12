# IEEE-PRAI-2026

**Format-Controlled Multi-Scale JPEG Compression Response Analysis for Image-Level Forgery Screening**

Sujith K Mandala — IEEE PRAI 2026

## Overview

This repository contains code for reproducing the experiments in our IEEE PRAI 2026 paper on format-controlled JPEG compression response analysis for image forgery screening.

Key results on CASIA v2.0 (JPEG-only, format-controlled):
- **AUC = 0.990** (stratified 5-fold CV)
- **AUC = 0.976** (source-aware grouped CV)
- 405 interpretable features, CPU-only, <0.3s/image

## Repository Structure

```
├── src/
│   ├── extract_features.py     # 405-feature extraction pipeline
│   └── evaluate.py             # Cross-validation + metrics
├── splits/
│   └── jpeg_group_folds.json   # Source-aware group split indices
├── scripts/
│   └── run_experiments.sh      # Reproduce all paper results
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- See `requirements.txt`

## Usage

```bash
pip install -r requirements.txt
python src/extract_features.py --data-dir /path/to/CASIA2 --output features.npy
python src/evaluate.py --features features.npy --protocol jpeg-only
```

## Citation

```bibtex
@inproceedings{mandala2026,
  title={Format-Controlled Multi-Scale JPEG Compression Response Analysis for Image-Level Forgery Screening},
  author={Mandala, Sujith K},
  booktitle={Proc. IEEE International Conference on Pattern Recognition and Artificial Intelligence (PRAI)},
  year={2026}
}
```

## License

MIT
