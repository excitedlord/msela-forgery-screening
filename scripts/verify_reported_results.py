"""
Verify that released prediction files reproduce the paper's reported metrics.

Loads committed CSV files and asserts that calculated metrics match Table III
and Table IV of the IEEE PRAI 2026 paper within rounding tolerance.

Usage:
    python scripts/verify_reported_results.py --predictions-dir predictions/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


def verify_metric(name: str, computed: float, expected: float,
                  tolerance: float = 0.0) -> bool:
    """Check a metric matches at 3-decimal rounding precision."""
    match = round(computed, 3) == expected
    status = "PASS" if match else "FAIL"
    print(f"  [{status}] {name}: computed={computed:.4f}, "
          f"expected={expected:.3f}, rounded={round(computed, 3):.3f}")
    return match


def main():
    parser = argparse.ArgumentParser(
        description='Verify reported results from prediction CSVs')
    parser.add_argument('--predictions-dir', type=str, default='predictions',
                        help='Directory containing prediction CSV files')
    parser.add_argument('--strict', action='store_true',
                        help='Exit with error code on any failure')
    args = parser.parse_args()

    pred_dir = Path(args.predictions_dir)
    all_pass = True

    # ═══════════════════════════════════════════════════════════════════════════
    # JPEG-only Stratified CV (Table III, row 1)
    # ═══════════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("JPEG-only Stratified 5-Fold CV")
    print("=" * 60)

    jpeg_csv = pred_dir / 'casia_jpeg_oof_predictions.csv'
    if jpeg_csv.exists():
        df = pd.read_csv(jpeg_csv)
        y_true = df['y_true'].values
        y_score = df['y_score'].values
        threshold = df['threshold'].iloc[0]
        y_pred = (y_score >= threshold).astype(int)

        auc = roc_auc_score(y_true, y_score)
        f1 = f1_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)

        all_pass &= verify_metric("JPEG-only AUC", auc, 0.991)
        all_pass &= verify_metric("JPEG-only F1", f1, 0.915)
        all_pass &= verify_metric("JPEG-only Accuracy", acc, 0.963)

        # Per-fold AUC stability
        fold_aucs = []
        for fold in sorted(df['fold'].unique()):
            fold_df = df[df['fold'] == fold]
            fold_auc = roc_auc_score(fold_df['y_true'], fold_df['y_score'])
            fold_aucs.append(fold_auc)
        print(f"  Per-fold AUCs: {', '.join(f'{a:.3f}' for a in fold_aucs)}")
        print(f"  Fold AUC std: {np.std(fold_aucs):.4f}")
    else:
        print(f"  [SKIP] {jpeg_csv} not found")
        all_pass = False

    # ═══════════════════════════════════════════════════════════════════════════
    # Grouped CV (Table III, row 2)
    # ═══════════════════════════════════════════════════════════════════════════
    print()
    print("=" * 60)
    print("Source-Aware Grouped 5-Fold CV")
    print("=" * 60)

    grouped_csv = pred_dir / 'casia_grouped_oof_predictions.csv'
    if grouped_csv.exists():
        df = pd.read_csv(grouped_csv)
        y_true = df['y_true'].values
        y_score = df['y_score'].values
        threshold = df['threshold'].iloc[0]
        y_pred = (y_score >= threshold).astype(int)

        auc = roc_auc_score(y_true, y_score)
        f1 = f1_score(y_true, y_pred)

        all_pass &= verify_metric("Grouped AUC", auc, 0.974)
        all_pass &= verify_metric("Grouped F1", f1, 0.816)
    else:
        print(f"  [SKIP] {grouped_csv} not found")
        all_pass = False

    # ═══════════════════════════════════════════════════════════════════════════
    # Columbia (Table IV)
    # ═══════════════════════════════════════════════════════════════════════════
    print()
    print("=" * 60)
    print("Columbia External Transfer")
    print("=" * 60)

    columbia_csv = pred_dir / 'columbia_predictions.csv'
    if columbia_csv.exists():
        df = pd.read_csv(columbia_csv)
        y_true = df['y_true'].values
        y_score = df['y_score'].values
        threshold = df['threshold'].iloc[0]
        y_pred = (y_score >= threshold).astype(int)

        auc = roc_auc_score(y_true, y_score)
        f1 = f1_score(y_true, y_pred)

        all_pass &= verify_metric("Columbia AUC", auc, 0.640)
        all_pass &= verify_metric("Columbia F1", f1, 0.614)
        print(f"  Threshold used: {threshold:.4f}")
    else:
        print(f"  [SKIP] {columbia_csv} not found")

    # ═══════════════════════════════════════════════════════════════════════════
    # CoMoFoD Negative Control (Table IV)
    # ═══════════════════════════════════════════════════════════════════════════
    print()
    print("=" * 60)
    print("CoMoFoD Negative Control")
    print("=" * 60)

    comofod_csv = pred_dir / 'comofod_predictions.csv'
    if comofod_csv.exists():
        df = pd.read_csv(comofod_csv)
        y_true = df['y_true'].values
        y_score = df['y_score'].values

        auc = roc_auc_score(y_true, y_score)
        all_pass &= verify_metric("CoMoFoD AUC", auc, 0.499)
    else:
        print(f"  [SKIP] {comofod_csv} not found")

    # ═══════════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════════
    print()
    print("=" * 60)
    if all_pass:
        print("ALL ASSERTIONS PASSED — results reproduce the paper.")
    else:
        print("SOME ASSERTIONS FAILED — check metrics above.")
    print("=" * 60)

    if args.strict and not all_pass:
        sys.exit(1)


if __name__ == '__main__':
    main()
