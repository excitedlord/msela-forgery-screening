"""
Feature ablation study for MS-ELA forgery screening (IEEE PRAI 2026).

Evaluates each feature family individually and in leave-one-out combinations
to measure marginal contribution and sufficiency:
  - Individual family performance
  - Full model minus one family
  - Ratio features alone (key finding: 0.985 AUC)
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier

from evaluate import HGB_PARAMS, compute_sample_weights, select_threshold_on_validation


# Feature family boundaries (cumulative indices)
FEATURE_FAMILIES = {
    'ELA (266)':         (0, 266),
    'Ratios (16)':       (266, 282),
    'Entropy (18)':      (282, 300),
    'FFT (6)':           (300, 306),
    'Edge (20)':         (306, 326),
    'SRM (27)':          (326, 353),
    'DCT/Noise/Color (52)': (353, 405),
}


def evaluate_subset(X: np.ndarray, y: np.ndarray, feature_mask: np.ndarray,
                    label: str) -> dict:
    """Run stratified 5-fold CV on a feature subset."""
    X_sub = X[:, feature_mask]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_scores = np.full(len(y), np.nan)
    fold_thresholds = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_sub, y)):
        sw = compute_sample_weights(y[train_idx])
        model = HistGradientBoostingClassifier(**HGB_PARAMS)
        model.fit(X_sub[train_idx], y[train_idx], sample_weight=sw)

        scores = model.predict_proba(X_sub[val_idx])[:, 1]
        oof_scores[val_idx] = scores
        t = select_threshold_on_validation(y[val_idx], scores)
        fold_thresholds.append(t)

    threshold = np.median(fold_thresholds)
    y_pred = (oof_scores >= threshold).astype(int)
    auc = roc_auc_score(y, oof_scores)
    f1 = f1_score(y, y_pred)

    return {'label': label, 'n_features': int(feature_mask.sum()),
            'auc': auc, 'f1': f1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Feature ablation study for MS-ELA')
    parser.add_argument('--features', type=str, required=True,
                        help='Path to full 405-dim features .npy')
    parser.add_argument('--labels', type=str, default=None)
    parser.add_argument('--output', type=str, default='results/ablation_results.csv')
    args = parser.parse_args()

    X = np.load(args.features)
    labels_path = args.labels or args.features.replace('.npy', '_labels.npy')
    y = np.load(labels_path)

    assert X.shape[1] == 405, f"Expected 405 features, got {X.shape[1]}"
    print(f"Features: {X.shape}, Labels: {y.shape}")
    print()

    results = []

    # Full model baseline
    print("=" * 60)
    print("Full Model (405 features)")
    print("=" * 60)
    full_mask = np.ones(405, dtype=bool)
    r = evaluate_subset(X, y, full_mask, 'All (405)')
    print(f"  AUC = {r['auc']:.4f}, F1 = {r['f1']:.4f}")
    results.append(r)

    # Individual families
    print(f"\n{'=' * 60}")
    print("Individual Feature Families")
    print(f"{'=' * 60}")
    for name, (start, end) in FEATURE_FAMILIES.items():
        mask = np.zeros(405, dtype=bool)
        mask[start:end] = True
        r = evaluate_subset(X, y, mask, name)
        print(f"  {name:<25} AUC = {r['auc']:.4f}, F1 = {r['f1']:.4f}")
        results.append(r)

    # Leave-one-out families
    print(f"\n{'=' * 60}")
    print("Leave-One-Out (Full minus one family)")
    print(f"{'=' * 60}")
    for name, (start, end) in FEATURE_FAMILIES.items():
        mask = np.ones(405, dtype=bool)
        mask[start:end] = False
        label = f"All - {name}"
        r = evaluate_subset(X, y, mask, label)
        print(f"  {label:<35} AUC = {r['auc']:.4f}, F1 = {r['f1']:.4f}")
        results.append(r)

    # Save results
    import pandas as pd
    df = pd.DataFrame(results)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")
