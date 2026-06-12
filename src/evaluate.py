"""
Evaluation script for Image Forgery Screening.

Runs cross-validation protocols:
  - JPEG-only stratified 5-fold
  - Source-aware grouped 5-fold
  - All→JPEG q95 normalization
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold, GroupKFold


HGB_PARAMS = {
    'max_iter': 1000,
    'learning_rate': 0.03,
    'max_depth': 8,
    'min_samples_leaf': 20,
    'l2_regularization': 1.0,
    'random_state': 42,
}


def run_stratified_cv(X, y, n_splits=5):
    """Run stratified 5-fold CV."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        model = HistGradientBoostingClassifier(**HGB_PARAMS)
        model.fit(X[train_idx], y[train_idx])
        oof[val_idx] = model.predict_proba(X[val_idx])[:, 1]
        auc_fold = roc_auc_score(y[val_idx], oof[val_idx])
        print(f"  Fold {fold+1}: AUC = {auc_fold:.4f}")
    
    auc = roc_auc_score(y, oof)
    # Optimal threshold
    thresholds = np.linspace(0, 1, 200)
    f1s = [f1_score(y, (oof >= t).astype(int)) for t in thresholds]
    best_f1 = max(f1s)
    
    return auc, best_f1, oof


def run_grouped_cv(X, y, groups, n_splits=5):
    """Run source-aware grouped 5-fold CV."""
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(y))
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        model = HistGradientBoostingClassifier(**HGB_PARAMS)
        model.fit(X[train_idx], y[train_idx])
        oof[val_idx] = model.predict_proba(X[val_idx])[:, 1]
        auc_fold = roc_auc_score(y[val_idx], oof[val_idx])
        print(f"  Fold {fold+1}: AUC = {auc_fold:.4f}")
    
    auc = roc_auc_score(y, oof)
    thresholds = np.linspace(0, 1, 200)
    f1s = [f1_score(y, (oof >= t).astype(int)) for t in thresholds]
    best_f1 = max(f1s)
    
    return auc, best_f1, oof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the Screening model')
    parser.add_argument('--features', type=str, required=True,
                        help='Path to features .npy file')
    parser.add_argument('--labels', type=str, default=None,
                        help='Path to labels .npy file (default: inferred)')
    parser.add_argument('--splits', type=str, default=None,
                        help='Path to group splits JSON (for grouped CV)')
    parser.add_argument('--protocol', type=str, default='jpeg-only',
                        choices=['jpeg-only', 'grouped', 'both'])
    args = parser.parse_args()
    
    X = np.load(args.features)
    
    labels_path = args.labels or args.features.replace('.npy', '_labels.npy')
    y = np.load(labels_path)
    
    print(f"Features: {X.shape}, Labels: {y.shape}")
    print(f"Class balance: {y.sum()}/{len(y)} positive ({y.mean():.3f})")
    print()
    
    if args.protocol in ('jpeg-only', 'both'):
        print("=" * 60)
        print("Protocol: JPEG-only Stratified 5-Fold CV")
        print("=" * 60)
        auc, f1, oof = run_stratified_cv(X, y)
        print(f"\n  Overall AUC = {auc:.4f}, Best F1 = {f1:.4f}")
        print()
    
    if args.protocol in ('grouped', 'both'):
        if args.splits is None:
            print("ERROR: --splits required for grouped protocol")
        else:
            print("=" * 60)
            print("Protocol: Source-Aware Grouped 5-Fold CV")
            print("=" * 60)
            with open(args.splits) as f:
                groups = np.array(json.load(f)['groups'])
            auc, f1, oof = run_grouped_cv(X, y, groups)
            print(f"\n  Overall AUC = {auc:.4f}, Best F1 = {f1:.4f}")
