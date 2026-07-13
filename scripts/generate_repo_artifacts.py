"""
Generate all reproducibility artifacts for the PRAI paper repo.

Outputs:
  splits/casia_source_groups.csv
  splits/casia_jpeg_stratified_5fold.csv
  splits/casia_jpeg_grouped_5fold.csv
  predictions/casia_jpeg_oof_predictions.csv
  predictions/casia_grouped_oof_predictions.csv
  predictions/columbia_predictions.csv
  predictions/comofod_predictions.csv

Usage:
  python scripts/generate_repo_artifacts.py --casia-dir /path/to/CASIA2
         --cache-dir /path/to/cache
"""

import argparse
import os
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score, f1_score


SEED = 42


# ═══════════════════════════════════════════════════════════════════════════════
# Source group parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_casia_source_ids(filename):
    """Parse all recoverable CASIA-style source identifiers from a filename.
    
    Authentic: Au_ani_00018.jpg -> ['ani00018']
    Tampered:  Tp_D_CND_M_N_ani00018_sec00096_00138.tif -> ['ani00018', 'sec00096']
    """
    stem = Path(filename).stem
    
    if stem.startswith("Au_"):
        # Authentic: Au_<cat>_<idx>
        parts = stem.split("_")
        if len(parts) >= 3:
            return [parts[1] + parts[2]]
        return [stem]
    
    elif stem.startswith("Tp_"):
        # Tampered: find all <cat><idx> patterns like ani00018, sec00096
        # Pattern: 3 lowercase letters followed by 5 digits
        ids = re.findall(r'([a-z]{3}\d{5})', stem)
        # Remove trailing numeric ID (last match if it's just digits-only context)
        if ids:
            return list(dict.fromkeys(ids))  # deduplicate preserving order
        return [stem]
    
    return [stem]


def build_source_graph(filenames, labels):
    """Build connected components from parsed source IDs.
    
    Nodes = source IDs
    Edges = two IDs appear in the same tampered filename
    Components define groups for GroupKFold.
    """
    # Parse all source IDs
    file_ids = {}
    all_ids = set()
    for fn in filenames:
        ids = parse_casia_source_ids(fn)
        file_ids[fn] = ids
        all_ids.update(ids)
    
    # Build adjacency via union-find
    parent = {sid: sid for sid in all_ids}
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    
    # Connect IDs that co-occur in the same filename
    for fn, ids in file_ids.items():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                union(ids[i], ids[j])
    
    # Map each file to its component
    component_map = {}
    comp_counter = {}
    next_comp = 0
    
    file_components = {}
    for fn, ids in file_ids.items():
        root = find(ids[0])
        if root not in comp_counter:
            comp_counter[root] = next_comp
            next_comp += 1
        file_components[fn] = f"component_{comp_counter[root]:04d}"
    
    return file_ids, file_components


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--casia-dir", required=True, help="Path to CASIA2/ folder")
    parser.add_argument("--cache-dir", required=True, help="Path to feature cache folder")
    parser.add_argument("--output-dir", default=".", help="Repo root (default: current dir)")
    args = parser.parse_args()

    casia_dir = Path(args.casia_dir)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.output_dir)

    splits_dir = out_dir / "splits"
    preds_dir = out_dir / "predictions"
    splits_dir.mkdir(exist_ok=True)
    preds_dir.mkdir(exist_ok=True)

    # ── Load CASIA filenames ──
    au_dir = casia_dir / "Au"
    tp_dir = casia_dir / "Tp"
    
    au_files = sorted([f for f in os.listdir(au_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
                       and not f.startswith('.')])
    tp_files = sorted([f for f in os.listdir(tp_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
                       and not f.startswith('.') and f != 'Thumbs.db'])
    
    print(f"Authentic: {len(au_files)}, Tampered: {len(tp_files)}")
    
    all_files = au_files + tp_files
    labels = np.array([0] * len(au_files) + [1] * len(tp_files))
    extensions = [Path(f).suffix.lower().lstrip('.') for f in all_files]

    # ── Build source groups ──
    print("Parsing source groups...")
    file_ids, file_components = build_source_graph(all_files, labels)
    
    n_components = len(set(file_components.values()))
    print(f"Unique source components: {n_components}")

    # ── JPEG-only mask ──
    jpeg_mask = np.array([ext in ('jpg', 'jpeg') for ext in extensions])
    jpeg_indices = np.where(jpeg_mask)[0]
    print(f"JPEG-only subset: {jpeg_mask.sum()} images")

    # ── Source groups CSV ──
    print("Writing casia_source_groups.csv...")
    rows = []
    for i, fn in enumerate(all_files):
        ids = file_ids[fn]
        rows.append({
            'filename': fn,
            'label': 'authentic' if labels[i] == 0 else 'tampered',
            'extension': extensions[i],
            'parsed_source_ids': ';'.join(ids),
            'component_group_id': file_components[fn],
            'is_jpeg': jpeg_mask[i],
        })
    df_groups = pd.DataFrame(rows)
    df_groups.to_csv(splits_dir / "casia_source_groups.csv", index=False)

    # ── Load features ──
    print("Loading features...")
    data = np.load(cache_dir / "casia_features.npz")
    X_all = data['X']
    y_all = data['y']
    
    print(f"Feature matrix: {X_all.shape}, Labels: {y_all.shape}")
    
    # Filter to JPEG-only
    X_jpeg = X_all[jpeg_mask]
    y_jpeg = y_all[jpeg_mask] if len(y_all) == len(all_files) else labels[jpeg_mask]
    fnames_jpeg = [all_files[i] for i in jpeg_indices]
    
    # ── Classifier config ──
    clf_params = dict(
        max_iter=1000,
        learning_rate=0.03,
        max_depth=8,
        min_samples_leaf=20,
        l2_regularization=1.0,
        max_bins=255,
        early_stopping=True,
        n_iter_no_change=50,
        validation_fraction=0.1,
        random_state=SEED,
    )

    # ══════════════════════════════════════════════════════════════════════
    # 1. Stratified 5-fold on JPEG-only
    # ══════════════════════════════════════════════════════════════════════
    print("\n── Stratified 5-fold (JPEG-only) ──")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    strat_folds = np.full(len(jpeg_indices), -1, dtype=int)
    oof_scores_strat = np.full(len(jpeg_indices), np.nan)
    thresholds_strat = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_jpeg, y_jpeg)):
        clf = HistGradientBoostingClassifier(**clf_params)
        
        # Sample weights
        n_pos = y_jpeg[train_idx].sum()
        n_neg = len(train_idx) - n_pos
        w = np.where(y_jpeg[train_idx] == 1, len(train_idx) / (2 * n_pos),
                      len(train_idx) / (2 * n_neg))
        
        clf.fit(X_jpeg[train_idx], y_jpeg[train_idx], sample_weight=w)
        scores = clf.predict_proba(X_jpeg[test_idx])[:, 1]
        oof_scores_strat[test_idx] = scores
        strat_folds[test_idx] = fold_idx
        
        # Validation threshold
        from sklearn.metrics import precision_recall_curve
        prec, rec, threshs = precision_recall_curve(y_jpeg[test_idx], scores)
        f1s = 2 * prec * rec / (prec + rec + 1e-8)
        best_t = threshs[np.argmax(f1s)]
        thresholds_strat.append(best_t)
        
        auc = roc_auc_score(y_jpeg[test_idx], scores)
        print(f"  Fold {fold_idx+1}: AUC={auc:.4f}, threshold={best_t:.4f}")
    
    median_t_strat = np.median(thresholds_strat)
    overall_auc = roc_auc_score(y_jpeg, oof_scores_strat)
    preds_strat = (oof_scores_strat >= median_t_strat).astype(int)
    overall_f1 = f1_score(y_jpeg, preds_strat)
    print(f"  Overall: AUC={overall_auc:.4f}, F1={overall_f1:.4f}, threshold={median_t_strat:.4f}")

    # Write stratified fold CSV
    strat_rows = []
    for i in range(len(jpeg_indices)):
        strat_rows.append({
            'filename': fnames_jpeg[i],
            'label': 'authentic' if y_jpeg[i] == 0 else 'tampered',
            'fold': strat_folds[i],
        })
    pd.DataFrame(strat_rows).to_csv(splits_dir / "casia_jpeg_stratified_5fold.csv", index=False)

    # Write stratified OOF predictions
    oof_strat_rows = []
    for i in range(len(jpeg_indices)):
        oof_strat_rows.append({
            'filename': fnames_jpeg[i],
            'label': 'authentic' if y_jpeg[i] == 0 else 'tampered',
            'fold': strat_folds[i],
            'y_true': int(y_jpeg[i]),
            'y_score': round(float(oof_scores_strat[i]), 6),
            'y_pred': int(preds_strat[i]),
            'threshold': round(float(median_t_strat), 6),
            'protocol': 'jpeg_only_stratified',
        })
    pd.DataFrame(oof_strat_rows).to_csv(preds_dir / "casia_jpeg_oof_predictions.csv", index=False)

    # ══════════════════════════════════════════════════════════════════════
    # 2. Grouped 5-fold on JPEG-only
    # ══════════════════════════════════════════════════════════════════════
    print("\n── Grouped 5-fold (JPEG-only, source-component-aware) ──")
    
    groups_jpeg = np.array([file_components[fnames_jpeg[i]] for i in range(len(jpeg_indices))])
    # Map component strings to integers for GroupKFold
    unique_groups = list(set(groups_jpeg))
    group_to_int = {g: i for i, g in enumerate(unique_groups)}
    groups_int = np.array([group_to_int[g] for g in groups_jpeg])
    
    gkf = GroupKFold(n_splits=5)
    
    grouped_folds = np.full(len(jpeg_indices), -1, dtype=int)
    oof_scores_grouped = np.full(len(jpeg_indices), np.nan)
    thresholds_grouped = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_jpeg, y_jpeg, groups=groups_int)):
        clf = HistGradientBoostingClassifier(**clf_params)
        
        n_pos = y_jpeg[train_idx].sum()
        n_neg = len(train_idx) - n_pos
        w = np.where(y_jpeg[train_idx] == 1, len(train_idx) / (2 * n_pos),
                      len(train_idx) / (2 * n_neg))
        
        clf.fit(X_jpeg[train_idx], y_jpeg[train_idx], sample_weight=w)
        scores = clf.predict_proba(X_jpeg[test_idx])[:, 1]
        oof_scores_grouped[test_idx] = scores
        grouped_folds[test_idx] = fold_idx
        
        from sklearn.metrics import precision_recall_curve
        prec, rec, threshs = precision_recall_curve(y_jpeg[test_idx], scores)
        f1s = 2 * prec * rec / (prec + rec + 1e-8)
        best_t = threshs[np.argmax(f1s)]
        thresholds_grouped.append(best_t)
        
        auc = roc_auc_score(y_jpeg[test_idx], scores)
        print(f"  Fold {fold_idx+1}: AUC={auc:.4f}, n_test={len(test_idx)}, threshold={best_t:.4f}")
    
    median_t_grouped = np.median(thresholds_grouped)
    overall_auc_g = roc_auc_score(y_jpeg, oof_scores_grouped)
    preds_grouped = (oof_scores_grouped >= median_t_grouped).astype(int)
    overall_f1_g = f1_score(y_jpeg, preds_grouped)
    print(f"  Overall: AUC={overall_auc_g:.4f}, F1={overall_f1_g:.4f}, threshold={median_t_grouped:.4f}")

    # Write grouped fold CSV
    grouped_rows = []
    for i in range(len(jpeg_indices)):
        grouped_rows.append({
            'filename': fnames_jpeg[i],
            'label': 'authentic' if y_jpeg[i] == 0 else 'tampered',
            'fold': grouped_folds[i],
            'component_group_id': groups_jpeg[i],
        })
    pd.DataFrame(grouped_rows).to_csv(splits_dir / "casia_jpeg_grouped_5fold.csv", index=False)

    # Write grouped OOF predictions
    oof_grouped_rows = []
    for i in range(len(jpeg_indices)):
        oof_grouped_rows.append({
            'filename': fnames_jpeg[i],
            'label': 'authentic' if y_jpeg[i] == 0 else 'tampered',
            'fold': grouped_folds[i],
            'y_true': int(y_jpeg[i]),
            'y_score': round(float(oof_scores_grouped[i]), 6),
            'y_pred': int(preds_grouped[i]),
            'threshold': round(float(median_t_grouped), 6),
            'protocol': 'jpeg_only_grouped',
        })
    pd.DataFrame(oof_grouped_rows).to_csv(preds_dir / "casia_grouped_oof_predictions.csv", index=False)

    # ══════════════════════════════════════════════════════════════════════
    # 3. External predictions (Columbia)
    # ══════════════════════════════════════════════════════════════════════
    print("\n── Columbia predictions ──")
    col_features = np.load(cache_dir / "columbia_features.npy")
    col_labels = np.load(cache_dir / "columbia_labels.npy")
    
    # Train on full JPEG-only, predict Columbia
    clf_full = HistGradientBoostingClassifier(**clf_params)
    n_pos = y_jpeg.sum()
    n_neg = len(y_jpeg) - n_pos
    w_full = np.where(y_jpeg == 1, len(y_jpeg) / (2 * n_pos),
                       len(y_jpeg) / (2 * n_neg))
    clf_full.fit(X_jpeg, y_jpeg, sample_weight=w_full)
    
    col_scores = clf_full.predict_proba(col_features)[:, 1]
    col_auc = roc_auc_score(col_labels, col_scores)
    col_preds = (col_scores >= median_t_strat).astype(int)
    col_f1 = f1_score(col_labels, col_preds)
    print(f"  Columbia: AUC={col_auc:.4f}, F1={col_f1:.4f}")
    
    col_rows = []
    for i in range(len(col_labels)):
        col_rows.append({
            'sample_index': i,
            'y_true': int(col_labels[i]),
            'y_score': round(float(col_scores[i]), 6),
            'y_pred': int(col_preds[i]),
            'threshold': round(float(median_t_strat), 6),
            'protocol': 'columbia_transfer',
        })
    pd.DataFrame(col_rows).to_csv(preds_dir / "columbia_predictions.csv", index=False)

    # ══════════════════════════════════════════════════════════════════════
    # 4. External predictions (CoMoFoD)
    # ══════════════════════════════════════════════════════════════════════
    print("\n── CoMoFoD predictions ──")
    como_data = np.load(cache_dir / "comofod_features.npz")
    como_features = como_data['X']
    como_labels = como_data['y']
    
    como_scores = clf_full.predict_proba(como_features)[:, 1]
    como_auc = roc_auc_score(como_labels, como_scores)
    print(f"  CoMoFoD: AUC={como_auc:.4f}")
    
    como_rows = []
    for i in range(len(como_labels)):
        como_rows.append({
            'sample_index': i,
            'y_true': int(como_labels[i]),
            'y_score': round(float(como_scores[i]), 6),
            'threshold': round(float(median_t_strat), 6),
            'protocol': 'comofod_negative_control',
        })
    pd.DataFrame(como_rows).to_csv(preds_dir / "comofod_predictions.csv", index=False)

    # ══════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Generated artifacts:")
    print(f"  splits/casia_source_groups.csv          ({len(df_groups)} rows)")
    strat_df = pd.read_csv(splits_dir / "casia_jpeg_stratified_5fold.csv")
    print(f"  splits/casia_jpeg_stratified_5fold.csv  ({len(strat_df)} rows)")
    grouped_df = pd.read_csv(splits_dir / "casia_jpeg_grouped_5fold.csv")
    print(f"  splits/casia_jpeg_grouped_5fold.csv     ({len(grouped_df)} rows)")
    print(f"  predictions/casia_jpeg_oof_predictions.csv")
    print(f"  predictions/casia_grouped_oof_predictions.csv")
    print(f"  predictions/columbia_predictions.csv")
    print(f"  predictions/comofod_predictions.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
