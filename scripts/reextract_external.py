"""Re-extract features for Columbia and CoMoFoD with fixed entropy code."""
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from extract_features import extract_all_features

IMG_EXTS = {'.jpg', '.jpeg', '.bmp', '.png', '.tif', '.tiff'}
N_WORKERS = min(cpu_count(), 12)


def extract_columbia(columbia_dir, output_dir):
    columbia_dir = Path(columbia_dir)
    paths, labels = [], []
    for d in sorted(columbia_dir.iterdir()):
        if not d.is_dir():
            continue
        label = 1 if d.name.startswith('Sp') else 0
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in IMG_EXTS and not f.name.startswith('.'):
                paths.append(str(f))
                labels.append(label)

    n_auth = sum(l == 0 for l in labels)
    n_sp = sum(l == 1 for l in labels)
    print(f"Columbia: {n_auth} authentic, {n_sp} spliced, total={len(paths)}")

    with Pool(N_WORKERS) as pool:
        feats = list(pool.map(extract_all_features, paths))

    X = np.array(feats, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    filenames = [Path(p).name for p in paths]

    out = Path(output_dir)
    np.save(out / 'features_columbia.npy', X)
    np.save(out / 'features_columbia_labels.npy', y)
    with open(out / 'features_columbia_filenames.txt', 'w') as f:
        f.write('\n'.join(filenames))
    print(f"Saved Columbia: {X.shape}")


def extract_comofod(comofod_dir, output_dir):
    comofod_dir = Path(comofod_dir)
    paths, labels, filenames_list = [], [], []

    for f in sorted(comofod_dir.iterdir()):
        if f.suffix.lower() not in IMG_EXTS or f.name.startswith('.'):
            continue
        name = f.stem  # e.g. 001_F, 001_O, 001_F_JC1, 001_O_JC1, etc.
        # Files with _O_ or ending in _O → original (label=0)
        # Files with _F_ or ending in _F → forged (label=1)
        if '_O' in name:
            label = 0
        else:
            label = 1
        paths.append(str(f))
        labels.append(label)
        filenames_list.append(f.name)

    n_orig = sum(l == 0 for l in labels)
    n_forg = sum(l == 1 for l in labels)
    print(f"CoMoFoD: {n_orig} original, {n_forg} forged, total={len(paths)}")

    with Pool(N_WORKERS) as pool:
        feats = list(pool.map(extract_all_features, paths))

    X = np.array(feats, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    out = Path(output_dir)
    np.save(out / 'features_comofod.npy', X)
    np.save(out / 'features_comofod_labels.npy', y)
    with open(out / 'features_comofod_filenames.txt', 'w') as f:
        f.write('\n'.join(filenames_list))
    print(f"Saved CoMoFoD: {X.shape}")


if __name__ == '__main__':
    repo_root = Path(__file__).resolve().parent.parent
    base = repo_root.parent

    extract_columbia(
        base / 'columbia-data' / 'ImSpliceDataset',
        repo_root / 'results',
    )
    extract_comofod(
        base / 'comofod_small-data' / 'CoMoFoD_small_v2',
        repo_root / 'results',
    )
