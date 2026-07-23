"""
Main evaluation script for MS-ELA forgery screening (IEEE PRAI 2026).

Implements the paper's evaluation protocol:
  - Inverse-frequency sample weighting
  - HistGradientBoosting with early stopping (patience 50, 10% validation)
  - Per-fold validation-selected F1 thresholds; median threshold applied globally
  - Stratified 5-fold and source-aware grouped 5-fold cross-validation
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, GroupKFold


# ═══════════════════════════════════════════════════════════════════════════════
# Model Configuration (matches paper Section III-H)
# ═══════════════════════════════════════════════════════════════════════════════

HGB_PARAMS = {
    'max_iter': 1000,
    'learning_rate': 0.03,
    'max_depth': 8,
    'min_samples_leaf': 20,
    'l2_regularization': 1.0,
    'max_bins': 255,
    'n_iter_no_change': 50,          # early stopping patience
    'validation_fraction': 0.1,       # 10% for early stopping
    'random_state': 42,
}


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """Compute inverse-frequency sample weights for class imbalance.

    w_i = N / (n_classes * n_i) where n_i is the count of class y_i.
    """
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    weights = np.zeros(n_samples)
    for cls, count in zip(classes, counts):
        weights[y == cls] = n_samples / (n_classes * count)
    return weights


def select_threshold_on_validation(y_val: np.ndarray, scores_val: np.ndarray,
                                   n_thresholds: int = 200) -> float:
    """Select F1-optimal threshold on validation fold predictions."""
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_f1 = 0.0
    best_t = 0.5
    for t in thresholds:
        f1 = f1_score(y_val, (scores_val >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-Validation Protocols
# ═══════════════════════════════════════════════════════════════════════════════

def run_stratified_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 5,
                      filenames: list = None) -> dict:
    """Run stratified 5-fold CV with validation-selected thresholds.

    Protocol:
        1. Train with inverse-frequency sample weights
        2. Per fold: select F1 threshold on that fold's validation predictions
        3. Apply median of per-fold thresholds as the global threshold
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_scores = np.full(len(y), np.nan)
    fold_thresholds = []
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        sample_weights = compute_sample_weights(y_train)

        model = HistGradientBoostingClassifier(**HGB_PARAMS)
        model.fit(X_train, y_train, sample_weight=sample_weights)

        scores = model.predict_proba(X_val)[:, 1]
        oof_scores[val_idx] = scores

        auc_fold = roc_auc_score(y_val, scores)
        fold_aucs.append(auc_fold)

        # Select threshold on this fold's validation data
        threshold = select_threshold_on_validation(y_val, scores)
        fold_thresholds.append(threshold)

        print(f"  Fold {fold + 1}: AUC = {auc_fold:.4f}, threshold = {threshold:.4f}")

    # Global metrics using median threshold
    median_threshold = np.median(fold_thresholds)
    y_pred = (oof_scores >= median_threshold).astype(int)

    overall_auc = roc_auc_score(y, oof_scores)
    overall_f1 = f1_score(y, y_pred)
    overall_acc = accuracy_score(y, y_pred)

    print(f"\n  Median threshold: {median_threshold:.6f}")
    print(f"  Overall AUC = {overall_auc:.4f}")
    print(f"  Overall F1  = {overall_f1:.4f}")
    print(f"  Overall Acc = {overall_acc:.4f}")

    # Build predictions DataFrame
    results = {
        'auc': overall_auc,
        'f1': overall_f1,
        'accuracy': overall_acc,
        'threshold': median_threshold,
        'fold_aucs': fold_aucs,
        'fold_thresholds': fold_thresholds,
        'oof_scores': oof_scores,
        'y_pred': y_pred,
    }

    if filenames is not None:
        fold_assignments = np.full(len(y), -1, dtype=int)
        for fold, (_, val_idx) in enumerate(skf.split(X, y)):
            fold_assignments[val_idx] = fold

        df = pd.DataFrame({
            'filename': filenames,
            'label': ['authentic' if yi == 0 else 'tampered' for yi in y],
            'fold': fold_assignments,
            'y_true': y,
            'y_score': oof_scores,
            'y_pred': y_pred,
            'threshold': median_threshold,
            'protocol': 'jpeg_only_stratified',
        })
        results['predictions_df'] = df

    return results


def run_grouped_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                   n_splits: int = 5, filenames: list = None) -> dict:
    """Run source-aware grouped 5-fold CV with validation-selected thresholds.

    Groups ensure images sharing parsed source identifiers never appear in
    both train and test within the same fold.
    """
    gkf = GroupKFold(n_splits=n_splits)
    oof_scores = np.full(len(y), np.nan)
    fold_thresholds = []
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        sample_weights = compute_sample_weights(y_train)

        model = HistGradientBoostingClassifier(**HGB_PARAMS)
        model.fit(X_train, y_train, sample_weight=sample_weights)

        scores = model.predict_proba(X_val)[:, 1]
        oof_scores[val_idx] = scores

        auc_fold = roc_auc_score(y_val, scores)
        fold_aucs.append(auc_fold)

        threshold = select_threshold_on_validation(y_val, scores)
        fold_thresholds.append(threshold)

        print(f"  Fold {fold + 1}: AUC = {auc_fold:.4f}, "
              f"threshold = {threshold:.4f}, n_val = {len(val_idx)}")

    median_threshold = np.median(fold_thresholds)
    y_pred = (oof_scores >= median_threshold).astype(int)

    overall_auc = roc_auc_score(y, oof_scores)
    overall_f1 = f1_score(y, y_pred)

    print(f"\n  Median threshold: {median_threshold:.6f}")
    print(f"  Overall AUC = {overall_auc:.4f}")
    print(f"  Overall F1  = {overall_f1:.4f}")

    results = {
        'auc': overall_auc,
        'f1': overall_f1,
        'threshold': median_threshold,
        'fold_aucs': fold_aucs,
        'fold_thresholds': fold_thresholds,
        'oof_scores': oof_scores,
        'y_pred': y_pred,
    }

    if filenames is not None:
        fold_assignments = np.full(len(y), -1, dtype=int)
        for fold, (_, val_idx) in enumerate(gkf.split(X, y, groups)):
            fold_assignments[val_idx] = fold

        df = pd.DataFrame({
            'filename': filenames,
            'label': ['authentic' if yi == 0 else 'tampered' for yi in y],
            'fold': fold_assignments,
            'y_true': y,
            'y_score': oof_scores,
            'y_pred': y_pred,
            'threshold': median_threshold,
            'protocol': 'grouped_source_aware',
        })
        results['predictions_df'] = df

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Bootstrap Confidence Intervals
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_auc_ci(y_true: np.ndarray, y_score: np.ndarray,
                     n_bootstrap: int = 2000, ci: float = 0.95,
                     seed: int = 42) -> tuple:
    """Compute bootstrap confidence interval for AUC."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    aucs = np.array(aucs)
    alpha = (1 - ci) / 2
    return np.percentile(aucs, 100 * alpha), np.percentile(aucs, 100 * (1 - alpha))


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate MS-ELA model with paper-specified protocol')
    parser.add_argument('--features', type=str, required=True,
                        help='Path to features .npy file')
    parser.add_argument('--labels', type=str, default=None,
                        help='Path to labels .npy file')
    parser.add_argument('--filenames', type=str, default=None,
                        help='Path to filenames .txt file')
    parser.add_argument('--splits', type=str, default=None,
                        help='Path to source groups CSV (for grouped CV)')
    parser.add_argument('--protocol', type=str, default='both',
                        choices=['jpeg-only', 'grouped', 'both'])
    parser.add_argument('--output-dir', type=str, default='predictions',
                        help='Directory to save prediction CSVs')
    parser.add_argument('--bootstrap', action='store_true',
                        help='Compute bootstrap 95%% CI for AUC')
    args = parser.parse_args()

    X = np.load(args.features)
    labels_path = args.labels or args.features.replace('.npy', '_labels.npy')
    y = np.load(labels_path)

    # Load filenames if available
    filenames = None
    filenames_path = args.filenames or args.features.replace('.npy', '_filenames.txt')
    if Path(filenames_path).exists():
        with open(filenames_path) as f:
            filenames = [line.strip() for line in f]

    print(f"Features: {X.shape}, Labels: {y.shape}")
    n_pos = int(y.sum())
    print(f"Class balance: {n_pos}/{len(y)} positive ({y.mean():.3f})")
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.protocol in ('jpeg-only', 'both'):
        print("=" * 60)
        print("Protocol: JPEG-only Stratified 5-Fold CV")
        print("=" * 60)
        results = run_stratified_cv(X, y, filenames=filenames)

        if args.bootstrap:
            lo, hi = bootstrap_auc_ci(y, results['oof_scores'])
            print(f"  95% CI (bootstrap): [{lo:.3f}--{hi:.3f}]")

        if 'predictions_df' in results:
            out_path = output_dir / 'casia_jpeg_oof_predictions.csv'
            results['predictions_df'].to_csv(out_path, index=False)
            print(f"  Saved: {out_path}")
        print()

    if args.protocol in ('grouped', 'both'):
        if args.splits is None:
            print("ERROR: --splits required for grouped protocol")
        else:
            print("=" * 60)
            print("Protocol: Source-Aware Grouped 5-Fold CV")
            print("=" * 60)

            splits_df = pd.read_csv(args.splits)
            # Align groups to feature order
            if filenames is not None:
                fn_to_group = dict(zip(splits_df['filename'],
                                       splits_df['component_group_id']))
                groups = np.array([fn_to_group.get(fn, fn) for fn in filenames])
            else:
                groups = splits_df['component_group_id'].values

            results = run_grouped_cv(X, y, groups, filenames=filenames)

            if args.bootstrap:
                lo, hi = bootstrap_auc_ci(y, results['oof_scores'])
                print(f"  95% CI (bootstrap): [{lo:.3f}--{hi:.3f}]")

            if 'predictions_df' in results:
                out_path = output_dir / 'casia_grouped_oof_predictions.csv'
                results['predictions_df'].to_csv(out_path, index=False)
                print(f"  Saved: {out_path}")
