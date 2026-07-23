"""
Normalization protocol evaluation for MS-ELA forgery screening (IEEE PRAI 2026).

Tests the effect of pre-converting all CASIA images to JPEG at various quality
levels before feature extraction, removing file-container and decoder-level
differences:
  - All → JPEG q95
  - All → JPEG q85
  - All → JPEG q75
"""

import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold

from extract_features import extract_all_features, TOTAL_FEATURES
from evaluate import (HGB_PARAMS, compute_sample_weights,
                      run_stratified_cv, bootstrap_auc_ci)


def normalize_and_extract(img_path: str, target_quality: int) -> np.ndarray:
    """Convert image to JPEG at target quality, then extract features."""
    img = Image.open(img_path).convert('RGB')

    # Re-save at target quality to normalize
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=target_quality)
    buf.seek(0)

    # Write to temp path for extract_all_features (which expects a path)
    # Instead, we directly use the buffer
    normalized = Image.open(buf).convert('RGB')

    # Extract features from the normalized image (inline)
    from extract_features import (
        extract_ela_features, extract_ratio_features,
        extract_entropy_features, extract_fft_features,
        extract_edge_features, extract_srm_features,
        extract_dct_noise_color_features
    )

    features = np.concatenate([
        extract_ela_features(normalized),
        extract_ratio_features(normalized),
        extract_entropy_features(normalized),
        extract_fft_features(normalized),
        extract_edge_features(normalized),
        extract_srm_features(normalized),
        extract_dct_noise_color_features(normalized),
    ])
    assert features.shape[0] == TOTAL_FEATURES
    return features.astype(np.float32)


def extract_normalized_dataset(image_paths: list, target_quality: int,
                               n_workers: int = None) -> np.ndarray:
    """Extract features from all images after normalizing to target JPEG quality."""
    from multiprocessing import Pool, cpu_count
    from functools import partial

    if n_workers is None:
        n_workers = min(cpu_count(), 12)

    func = partial(normalize_and_extract, target_quality=target_quality)
    with Pool(n_workers) as pool:
        results = pool.map(func, image_paths)

    return np.array(results, dtype=np.float32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate MS-ELA under JPEG normalization protocols')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to CASIA2 directory')
    parser.add_argument('--qualities', type=int, nargs='+', default=[95, 85, 75],
                        help='Target JPEG qualities for normalization')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--bootstrap', action='store_true')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    au_dir = data_dir / 'Au'
    tp_dir = data_dir / 'Tp'

    # Collect ALL images (mixed format) for normalization experiments
    valid_ext = {'.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    au_files = sorted([f for f in au_dir.iterdir()
                       if f.suffix.lower() in valid_ext
                       and not f.name.startswith('.')])
    tp_files = sorted([f for f in tp_dir.iterdir()
                       if f.suffix.lower() in valid_ext
                       and not f.name.startswith('.')
                       and f.name != 'Thumbs.db'])

    all_paths = [str(f) for f in au_files + tp_files]
    y = np.array([0] * len(au_files) + [1] * len(tp_files))
    filenames = [f.name for f in au_files + tp_files]

    print(f"Total images: {len(all_paths)} (Au: {len(au_files)}, Tp: {len(tp_files)})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for q in args.qualities:
        print(f"\n{'=' * 60}")
        print(f"Protocol: All → JPEG q{q} Normalization")
        print(f"{'=' * 60}")

        print(f"  Extracting features (normalized to q{q})...")
        X = extract_normalized_dataset(all_paths, q, n_workers=args.workers)
        print(f"  Features shape: {X.shape}")

        results = run_stratified_cv(X, y, filenames=filenames)

        if args.bootstrap:
            lo, hi = bootstrap_auc_ci(y, results['oof_scores'])
            print(f"  95% CI (bootstrap): [{lo:.3f}--{hi:.3f}]")

        # Save
        np.save(output_dir / f'features_normalized_q{q}.npy', X)
        if 'predictions_df' in results:
            results['predictions_df'].to_csv(
                output_dir / f'predictions_normalized_q{q}.csv', index=False)
