"""
Robustness evaluation for MS-ELA forgery screening (IEEE PRAI 2026).

Tests model degradation under post-processing perturbations applied to test
images at evaluation time:
  - Resize: 0.5x, 0.75x (downscaling destroys compression traces)
  - Gaussian blur: sigma=1, sigma=2
  - JPEG recompression: q85, q95
"""

import argparse
import io
from pathlib import Path
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from PIL import Image, ImageFilter
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold

from extract_features import extract_all_features, TOTAL_FEATURES
from evaluate import HGB_PARAMS, compute_sample_weights, select_threshold_on_validation


def perturb_image(img: Image.Image, perturbation: str) -> Image.Image:
    """Apply a perturbation to an image."""
    if perturbation == 'resize_050':
        w, h = img.size
        return img.resize((w // 2, h // 2), Image.BILINEAR)
    elif perturbation == 'resize_075':
        w, h = img.size
        return img.resize((int(w * 0.75), int(h * 0.75)), Image.BILINEAR)
    elif perturbation == 'blur_1':
        return img.filter(ImageFilter.GaussianBlur(radius=1))
    elif perturbation == 'blur_2':
        return img.filter(ImageFilter.GaussianBlur(radius=2))
    elif perturbation == 'jpeg_q85':
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=85)
        buf.seek(0)
        return Image.open(buf).convert('RGB')
    elif perturbation == 'jpeg_q95':
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=95)
        buf.seek(0)
        return Image.open(buf).convert('RGB')
    else:
        raise ValueError(f"Unknown perturbation: {perturbation}")


def extract_perturbed_features(args_tuple) -> np.ndarray:
    """Extract features from a perturbed image."""
    img_path, perturbation = args_tuple
    img = Image.open(img_path).convert('RGB')
    perturbed = perturb_image(img, perturbation)

    from extract_features import (
        compute_ela_cache,
        extract_ela_features, extract_ratio_features,
        extract_entropy_features, extract_fft_features,
        extract_edge_features, extract_srm_features,
        extract_dct_noise_color_features
    )

    residuals = compute_ela_cache(perturbed)
    features = np.concatenate([
        extract_ela_features(residuals),
        extract_ratio_features(residuals),
        extract_entropy_features(residuals),
        extract_fft_features(perturbed),
        extract_edge_features(perturbed),
        extract_srm_features(perturbed),
        extract_dct_noise_color_features(perturbed),
    ])
    assert features.shape[0] == TOTAL_FEATURES
    return features.astype(np.float32)


PERTURBATIONS = ['resize_050', 'resize_075', 'blur_1', 'blur_2', 'jpeg_q85', 'jpeg_q95']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate MS-ELA robustness under post-processing perturbations')
    parser.add_argument('--features', type=str, required=True,
                        help='Path to clean JPEG-only features .npy (for training)')
    parser.add_argument('--labels', type=str, default=None)
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to CASIA2 directory (for perturbed extraction)')
    parser.add_argument('--perturbations', nargs='+', default=PERTURBATIONS,
                        choices=PERTURBATIONS)
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--workers', type=int, default=None)
    args = parser.parse_args()

    # Load clean features for training
    X_clean = np.load(args.features)
    labels_path = args.labels or args.features.replace('.npy', '_labels.npy')
    y = np.load(labels_path)

    # Get image paths (JPEG only)
    data_dir = Path(args.data_dir)
    jpeg_ext = {'.jpg', '.jpeg'}
    au_files = sorted([f for f in (data_dir / 'Au').iterdir()
                       if f.suffix.lower() in jpeg_ext and not f.name.startswith('.')])
    tp_files = sorted([f for f in (data_dir / 'Tp').iterdir()
                       if f.suffix.lower() in jpeg_ext
                       and not f.name.startswith('.') and f.name != 'Thumbs.db'])
    all_paths = [str(f) for f in au_files + tp_files]

    assert len(all_paths) == len(y), \
        f"Path count ({len(all_paths)}) != label count ({len(y)})"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_workers = args.workers or min(cpu_count(), 12)

    # Evaluate each perturbation using proper OOF protocol:
    # train on clean fold, predict perturbed held-out fold only
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results_summary = []

    for pert in args.perturbations:
        print(f"\n{'=' * 60}")
        print(f"Perturbation: {pert}")
        print(f"{'=' * 60}")

        print(f"  Extracting perturbed features ({n_workers} workers)...")
        task_args = [(p, pert) for p in all_paths]
        with Pool(n_workers) as pool:
            X_pert = np.array(pool.map(extract_perturbed_features, task_args),
                              dtype=np.float32)

        # OOF: each sample predicted only by the model that never saw it
        oof_scores = np.full(len(y), np.nan)
        fold_thresholds = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_clean, y)):
            sw = compute_sample_weights(y[train_idx])
            model = HistGradientBoostingClassifier(**HGB_PARAMS)
            model.fit(X_clean[train_idx], y[train_idx], sample_weight=sw)

            # Predict perturbed version of the held-out fold only
            scores_val = model.predict_proba(X_pert[val_idx])[:, 1]
            oof_scores[val_idx] = scores_val

            t = select_threshold_on_validation(y[val_idx], scores_val)
            fold_thresholds.append(t)

        threshold = np.median(fold_thresholds)
        y_pred = (oof_scores >= threshold).astype(int)
        auc = roc_auc_score(y, oof_scores)
        f1 = f1_score(y, y_pred)

        print(f"  AUC = {auc:.4f}")
        print(f"  F1  = {f1:.4f}")

        results_summary.append({'perturbation': pert, 'auc': auc, 'f1': f1})
        np.save(output_dir / f'features_perturbed_{pert}.npy', X_pert)

    # Summary table
    print(f"\n{'=' * 60}")
    print("Robustness Summary")
    print(f"{'=' * 60}")
    print(f"  {'Perturbation':<15} {'AUC':<8} {'F1':<8}")
    print(f"  {'-' * 31}")
    for r in results_summary:
        print(f"  {r['perturbation']:<15} {r['auc']:<8.4f} {r['f1']:<8.4f}")
