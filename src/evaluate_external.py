"""
External dataset evaluation for MS-ELA forgery screening (IEEE PRAI 2026).

Evaluates transfer performance on:
  - Columbia Image Splicing Detection Dataset
  - CoMoFoD (negative control for copy-move)

Uses the CASIA-trained model (all 5 stratified folds) with the CASIA-selected
median threshold for F1 evaluation.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold

from evaluate import HGB_PARAMS, compute_sample_weights, select_threshold_on_validation


def train_full_model(X_train: np.ndarray, y_train: np.ndarray) -> list:
    """Train 5 stratified-fold models on CASIA for ensembled external prediction."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = []
    fold_thresholds = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        Xt, yt = X_train[train_idx], y_train[train_idx]
        Xv, yv = X_train[val_idx], y_train[val_idx]
        sw = compute_sample_weights(yt)

        model = HistGradientBoostingClassifier(**HGB_PARAMS)
        model.fit(Xt, yt, sample_weight=sw)
        models.append(model)

        scores_v = model.predict_proba(Xv)[:, 1]
        t = select_threshold_on_validation(yv, scores_v)
        fold_thresholds.append(t)

    median_threshold = np.median(fold_thresholds)
    return models, median_threshold


def evaluate_external(models: list, threshold: float,
                      X_ext: np.ndarray, y_ext: np.ndarray,
                      dataset_name: str, filenames: list = None) -> dict:
    """Evaluate ensemble of CASIA-trained models on an external dataset."""
    # Average predictions across 5 fold models
    scores = np.zeros(len(y_ext))
    for model in models:
        scores += model.predict_proba(X_ext)[:, 1]
    scores /= len(models)

    y_pred = (scores >= threshold).astype(int)

    auc = roc_auc_score(y_ext, scores) if len(np.unique(y_ext)) > 1 else float('nan')
    f1 = f1_score(y_ext, y_pred, zero_division=0)

    print(f"\n  {dataset_name}:")
    print(f"    AUC = {auc:.4f}")
    print(f"    F1  = {f1:.4f} (using CASIA threshold = {threshold:.4f})")
    print(f"    N   = {len(y_ext)} ({int(y_ext.sum())} positive)")

    results = {
        'dataset': dataset_name,
        'auc': auc,
        'f1': f1,
        'threshold': threshold,
        'scores': scores,
        'y_pred': y_pred,
    }

    if filenames is not None:
        df = pd.DataFrame({
            'filename': filenames,
            'label': ['authentic' if yi == 0 else 'tampered' for yi in y_ext],
            'y_true': y_ext,
            'y_score': scores,
            'y_pred': y_pred,
            'threshold': threshold,
            'protocol': f'external_{dataset_name.lower()}',
        })
        results['predictions_df'] = df

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate MS-ELA on external datasets (Columbia, CoMoFoD)')
    parser.add_argument('--casia-features', type=str, required=True,
                        help='Path to CASIA JPEG-only features .npy')
    parser.add_argument('--casia-labels', type=str, default=None,
                        help='Path to CASIA labels .npy')
    parser.add_argument('--columbia-features', type=str, default=None,
                        help='Path to Columbia features .npy')
    parser.add_argument('--columbia-labels', type=str, default=None,
                        help='Path to Columbia labels .npy')
    parser.add_argument('--comofod-features', type=str, default=None,
                        help='Path to CoMoFoD features .npy')
    parser.add_argument('--comofod-labels', type=str, default=None,
                        help='Path to CoMoFoD labels .npy')
    parser.add_argument('--output-dir', type=str, default='predictions',
                        help='Directory to save prediction CSVs')
    args = parser.parse_args()

    # Load CASIA training data
    X_casia = np.load(args.casia_features)
    casia_labels_path = args.casia_labels or args.casia_features.replace('.npy', '_labels.npy')
    y_casia = np.load(casia_labels_path)

    print(f"CASIA training data: {X_casia.shape}")
    print("Training 5-fold ensemble on CASIA...")
    models, threshold = train_full_model(X_casia, y_casia)
    print(f"CASIA median threshold: {threshold:.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Columbia evaluation
    if args.columbia_features:
        print("\n" + "=" * 60)
        print("External: Columbia Image Splicing Detection")
        print("=" * 60)
        X_col = np.load(args.columbia_features)
        col_labels_path = args.columbia_labels or args.columbia_features.replace('.npy', '_labels.npy')
        y_col = np.load(col_labels_path)

        col_filenames_path = args.columbia_features.replace('.npy', '_filenames.txt')
        col_filenames = None
        if Path(col_filenames_path).exists():
            with open(col_filenames_path) as f:
                col_filenames = [line.strip() for line in f]

        results = evaluate_external(models, threshold, X_col, y_col,
                                    'Columbia', col_filenames)
        if 'predictions_df' in results:
            out_path = output_dir / 'columbia_predictions.csv'
            results['predictions_df'].to_csv(out_path, index=False)
            print(f"    Saved: {out_path}")

    # CoMoFoD evaluation (negative control)
    if args.comofod_features:
        print("\n" + "=" * 60)
        print("External: CoMoFoD (Negative Control)")
        print("=" * 60)
        X_cmf = np.load(args.comofod_features)
        cmf_labels_path = args.comofod_labels or args.comofod_features.replace('.npy', '_labels.npy')
        y_cmf = np.load(cmf_labels_path)

        cmf_filenames_path = args.comofod_features.replace('.npy', '_filenames.txt')
        cmf_filenames = None
        if Path(cmf_filenames_path).exists():
            with open(cmf_filenames_path) as f:
                cmf_filenames = [line.strip() for line in f]

        results = evaluate_external(models, threshold, X_cmf, y_cmf,
                                    'CoMoFoD', cmf_filenames)
        if 'predictions_df' in results:
            out_path = output_dir / 'comofod_predictions.csv'
            results['predictions_df'].to_csv(out_path, index=False)
            print(f"    Saved: {out_path}")
