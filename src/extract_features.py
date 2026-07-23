"""
Feature extraction pipeline for MS-ELA forgery screening (IEEE PRAI 2026).

Extracts 405 features across 7 families from JPEG images:
  Family 1: Multi-scale ELA (266 features) — 7 quality levels × 38 statistics
  Family 2: Cross-quality ELA ratios (16 features) — 4 pixel-wise ratio maps × 4 stats
  Family 3: ELA entropy (18 features) — 4×4 grid entropy on q=50 ELA map
  Family 4: FFT radial energy bands (6 features) — 5 bands + high/low ratio
  Family 5: Edge density (20 features) — Sobel magnitude + 4×4 grid
  Family 6: SRM steganalysis residuals (27 features) — 3 kernels × 3 RGB channels × 3 stats
  Family 7: DCT blockiness, noise, and color (52 features)
"""

import argparse
import io
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
from PIL import Image
from scipy.fft import fft2, fftshift
from scipy.signal import convolve2d
from scipy.stats import entropy as sp_entropy


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Family 1: Multi-Scale ELA (266 features)
# 7 quality levels × (6 global + 32 spatial grid) = 266
# ═══════════════════════════════════════════════════════════════════════════════

QUALITIES = [30, 50, 60, 75, 80, 90, 95]


def compute_ela(img: Image.Image, quality: int) -> np.ndarray:
    """Compute ELA residual map at a given JPEG quality.

    Returns channel-averaged residual: |I - recompress(I, q)|, averaged over RGB.
    """
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf)

    orig = np.array(img.convert('RGB'), dtype=np.float64)
    recomp = np.array(recompressed.convert('RGB'), dtype=np.float64)
    residual = np.abs(orig - recomp)
    return residual.mean(axis=2)  # Channel-averaged ELA map


def compute_ela_cache(img: Image.Image) -> dict:
    """Compute all seven ELA residual maps once and return as a dict.

    This ensures each quality level is computed exactly once (7 recompressions
    total), even when ELA maps are shared across feature families.
    """
    return {q: compute_ela(img, q) for q in QUALITIES}


def ela_statistics(residual: np.ndarray) -> np.ndarray:
    """Extract 38 statistics from one ELA residual map.

    Global statistics (6):
        mean, std, 75th percentile, 95th percentile, 99th percentile,
        high-residual fraction (pixels > mu + 2*sigma).

    Spatial grid statistics (32):
        4×4 grid; per cell: mean and std of residuals (16 means + 16 stds).
    """
    flat = residual.ravel()

    # --- 6 global statistics ---
    mu = flat.mean()
    sigma = flat.std()
    global_stats = [
        mu,
        sigma,
        np.percentile(flat, 75),
        np.percentile(flat, 95),
        np.percentile(flat, 99),
        (flat > mu + 2 * sigma).mean(),  # high-residual fraction
    ]

    # --- 32 spatial grid statistics (4×4 grid) ---
    h, w = residual.shape
    grid_h, grid_w = h // 4, w // 4
    grid_stats = []
    for row in range(4):
        for col in range(4):
            cell = residual[row * grid_h:(row + 1) * grid_h,
                            col * grid_w:(col + 1) * grid_w]
            grid_stats.append(cell.mean())
    for row in range(4):
        for col in range(4):
            cell = residual[row * grid_h:(row + 1) * grid_h,
                            col * grid_w:(col + 1) * grid_w]
            grid_stats.append(cell.std())

    return np.array(global_stats + grid_stats, dtype=np.float64)


def extract_ela_features(residuals: dict) -> np.ndarray:
    """Extract multi-scale ELA features: 7 qualities × 38 stats = 266."""
    features = []
    for q in QUALITIES:
        features.append(ela_statistics(residuals[q]))
    return np.concatenate(features)


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Family 2: Cross-Quality ELA Ratios (16 features)
# 4 pixel-wise ratio maps × (mean, std, p5, p95) = 16
# ═══════════════════════════════════════════════════════════════════════════════

RATIO_PAIRS = [(30, 90), (50, 90), (60, 80), (30, 60)]
EPSILON = 1e-6


def extract_ratio_features(residuals: dict) -> np.ndarray:
    """Extract 16 cross-quality ratio features from pixel-wise ratio maps.

    For each pair (qa, qb), constructs R(x,y) = E_qa(x,y) / (E_qb(x,y) + eps)
    and extracts mean, std, 5th percentile, 95th percentile.
    """
    features = []
    for qa, qb in RATIO_PAIRS:
        ratio_map = residuals[qa] / (residuals[qb] + EPSILON)
        flat = ratio_map.ravel()
        features.extend([
            flat.mean(),
            flat.std(),
            np.percentile(flat, 5),
            np.percentile(flat, 95),
        ])

    return np.array(features, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Family 3: ELA Entropy (18 features)
# 4×4 grid Shannon entropy on q=50 ELA map → 16 cells + std + range = 18
# ═══════════════════════════════════════════════════════════════════════════════

def extract_entropy_features(residuals: dict) -> np.ndarray:
    """Extract 18 ELA entropy features.

    Uses q=50 ELA map, normalized to [0,255] and quantized to integers.
    Divides into 4x4 grid; per cell computes Shannon entropy (32-bin histogram).
    Entropy uses normalized count probabilities: p = counts / counts.sum().
    Returns: 16 cell entropies + their std + their range = 18.
    """
    residual = residuals[50]

    # Normalize to [0, 255] and quantize
    r_min, r_max = residual.min(), residual.max()
    if r_max - r_min > 0:
        normalized = ((residual - r_min) / (r_max - r_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(residual, dtype=np.uint8)

    h, w = normalized.shape
    grid_h, grid_w = h // 4, w // 4

    cell_entropies = []
    for row in range(4):
        for col in range(4):
            cell = normalized[row * grid_h:(row + 1) * grid_h,
                              col * grid_w:(col + 1) * grid_w].ravel()
            counts, _ = np.histogram(cell, bins=32, range=(0, 256))
            p = counts / (counts.sum() + EPSILON)
            p_nz = p[p > 0]
            H = -np.sum(p_nz * np.log2(p_nz))
            cell_entropies.append(H)

    cell_entropies = np.array(cell_entropies)
    features = list(cell_entropies) + [
        cell_entropies.std(),
        cell_entropies.max() - cell_entropies.min(),
    ]

    return np.array(features, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Family 4: FFT Radial Energy Bands (6 features)
# 5 fractional-energy bands + high/low ratio = 6
# ═══════════════════════════════════════════════════════════════════════════════

FFT_SIZE = 256
FFT_BANDS = [(0, 16), (16, 32), (32, 64), (64, 96), (96, 128)]


def extract_fft_features(img: Image.Image) -> np.ndarray:
    """Extract 6 FFT radial energy band features.

    Resizes grayscale image to 256x256, computes 2D DFT magnitude spectrum,
    partitions into 5 concentric radial bands, and computes fractional energy
    per band plus a high-to-low ratio.
    """
    gray = np.array(img.convert('L').resize((FFT_SIZE, FFT_SIZE), Image.BILINEAR),
                    dtype=np.float64)

    f_transform = fftshift(fft2(gray))
    magnitude = np.abs(f_transform) ** 2  # Power spectrum

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    total_energy = magnitude.sum() + EPSILON

    band_energies = []
    for r_low, r_high in FFT_BANDS:
        mask = (R >= r_low) & (R < r_high)
        band_energies.append(magnitude[mask].sum() / total_energy)

    # High-to-low ratio: E_{r>=64} / E_{r<32}
    high_energy = band_energies[3] + band_energies[4]  # [64,96) + [96,128)
    low_energy = band_energies[0] + band_energies[1] + EPSILON  # [0,16) + [16,32)
    ratio = high_energy / low_energy

    features = band_energies + [ratio]
    return np.array(features, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Family 5: Edge Density (20 features)
# Global (mean, std, p95) + 4×4 grid mean density + cross-cell std = 20
# ═══════════════════════════════════════════════════════════════════════════════

def extract_edge_features(img: Image.Image) -> np.ndarray:
    """Extract 20 edge density features.

    Computes Sobel gradient magnitude on grayscale image normalized to [0,1].
    Extracts: global mean, std, 95th percentile (3);
              4x4 grid of mean edge density (16);
              cross-cell standard deviation (1).
    """
    gray = np.array(img.convert('L'), dtype=np.float64) / 255.0

    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

    gx = convolve2d(gray, sobel_x, mode='same', boundary='symm')
    gy = convolve2d(gray, sobel_y, mode='same', boundary='symm')
    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    # Global statistics (3)
    features = [
        magnitude.mean(),
        magnitude.std(),
        np.percentile(magnitude, 95),
    ]

    # 4×4 grid mean edge density (16)
    h, w = magnitude.shape
    grid_h, grid_w = h // 4, w // 4
    cell_means = []
    for row in range(4):
        for col in range(4):
            cell = magnitude[row * grid_h:(row + 1) * grid_h,
                             col * grid_w:(col + 1) * grid_w]
            cell_means.append(cell.mean())
    features.extend(cell_means)

    # Cross-cell standard deviation (1)
    features.append(np.std(cell_means))

    return np.array(features, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Family 6: SRM Steganalysis Residuals (27 features)
# 3 kernels × 3 RGB channels × (mean, std, p95) = 27
# ═══════════════════════════════════════════════════════════════════════════════

SRM_SIZE = 224

# Three 5×5 high-pass SRM kernels (Fridrich & Kodovsky, 2012)
SRM_KERNEL_1 = np.array([
    [-1,  2, -2,  2, -1],
    [ 2, -6,  8, -6,  2],
    [-2,  8, -12, 8, -2],
    [ 2, -6,  8, -6,  2],
    [-1,  2, -2,  2, -1],
], dtype=np.float64) / 12.0

SRM_KERNEL_2 = np.array([
    [ 0,  0,  0,  0,  0],
    [ 0, -1,  2, -1,  0],
    [ 0,  2, -4,  2,  0],
    [ 0, -1,  2, -1,  0],
    [ 0,  0,  0,  0,  0],
], dtype=np.float64) / 4.0

SRM_KERNEL_3 = np.array([
    [ 0,  0, -1,  0,  0],
    [ 0,  0,  2,  0,  0],
    [-1,  2, -4,  2, -1],
    [ 0,  0,  2,  0,  0],
    [ 0,  0, -1,  0,  0],
], dtype=np.float64) / 4.0

SRM_KERNELS = [SRM_KERNEL_1, SRM_KERNEL_2, SRM_KERNEL_3]


def extract_srm_features(img: Image.Image) -> np.ndarray:
    """Extract 27 SRM noise residual features.

    Resizes image to 224x224, applies three 5x5 high-pass SRM kernels to each
    RGB channel. For each of 9 filter-channel combinations, extracts: mean, std,
    and 95th percentile of absolute response.
    """
    rgb = np.array(img.convert('RGB').resize((SRM_SIZE, SRM_SIZE), Image.BILINEAR),
                   dtype=np.float64)

    features = []
    for kernel in SRM_KERNELS:
        for ch in range(3):
            response = convolve2d(rgb[:, :, ch], kernel, mode='same', boundary='symm')
            abs_response = np.abs(response)
            features.extend([
                abs_response.mean(),
                abs_response.std(),
                np.percentile(abs_response, 95),
            ])

    return np.array(features, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Family 7: DCT Blockiness, Noise, and Color (52 features)
# DCT blockiness (8) + Laplacian noise (32) + Color statistics (12) = 52
# ═══════════════════════════════════════════════════════════════════════════════

LAPLACIAN_KERNEL = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)


def extract_dct_noise_color_features(img: Image.Image) -> np.ndarray:
    """Extract 52 DCT blockiness, Laplacian noise, and color features.

    DCT blockiness (8):
        Boundary-to-interior intensity ratios at 8x8 block boundaries and
        their spatial variance across all blocks.

    Laplacian noise (32):
        Variance and mean absolute Laplacian in a 4x4 spatial grid.
        (16 variances + 16 mean-abs values)

    Color statistics (12):
        Inter-channel correlations (3): corr(R,G), corr(R,B), corr(G,B)
        Per-channel std, skewness, kurtosis (3 channels x 3 stats = 9)
    """
    gray = np.array(img.convert('L'), dtype=np.float64)
    rgb = np.array(img.convert('RGB'), dtype=np.float64)

    # --- DCT blockiness (8 features) ---
    h, w = gray.shape
    block_size = 8
    h_trim = (h // block_size) * block_size
    w_trim = (w // block_size) * block_size
    trimmed = gray[:h_trim, :w_trim]

    blocks = trimmed.reshape(h_trim // block_size, block_size,
                             w_trim // block_size, block_size)
    n_blocks_h = h_trim // block_size
    n_blocks_w = w_trim // block_size

    boundary_means = []
    interior_means = []
    for i in range(n_blocks_h):
        for j in range(n_blocks_w):
            block = blocks[i, :, j, :]
            boundary = np.concatenate([
                block[0, :], block[-1, :], block[1:-1, 0], block[1:-1, -1]
            ])
            interior = block[1:-1, 1:-1].ravel()
            if len(interior) > 0 and len(boundary) > 0:
                boundary_means.append(boundary.mean())
                interior_means.append(interior.mean())

    boundary_means = np.array(boundary_means)
    interior_means = np.array(interior_means)
    ratios = boundary_means / (interior_means + EPSILON)

    blockiness_features = [
        ratios.mean(),
        ratios.std(),
        np.percentile(ratios, 5),
        np.percentile(ratios, 25),
        np.percentile(ratios, 50),
        np.percentile(ratios, 75),
        np.percentile(ratios, 95),
        ratios.var(),
    ]

    # --- Laplacian noise (32 features) ---
    laplacian = convolve2d(gray, LAPLACIAN_KERNEL, mode='same', boundary='symm')

    grid_h, grid_w = h // 4, w // 4
    lap_features = []
    # 16 variances
    for row in range(4):
        for col in range(4):
            cell = laplacian[row * grid_h:(row + 1) * grid_h,
                             col * grid_w:(col + 1) * grid_w]
            lap_features.append(cell.var())
    # 16 mean absolute values
    for row in range(4):
        for col in range(4):
            cell = laplacian[row * grid_h:(row + 1) * grid_h,
                             col * grid_w:(col + 1) * grid_w]
            lap_features.append(np.abs(cell).mean())

    # --- Color statistics (12 features) ---
    r, g, b = rgb[:, :, 0].ravel(), rgb[:, :, 1].ravel(), rgb[:, :, 2].ravel()

    # Inter-channel correlations (3)
    color_features = [
        np.corrcoef(r, g)[0, 1],
        np.corrcoef(r, b)[0, 1],
        np.corrcoef(g, b)[0, 1],
    ]

    # Per-channel std, skewness, kurtosis (9)
    for ch_flat in [r, g, b]:
        mu = ch_flat.mean()
        sigma = ch_flat.std() + EPSILON
        standardized = (ch_flat - mu) / sigma
        color_features.extend([
            sigma,
            (standardized ** 3).mean(),  # skewness
            (standardized ** 4).mean(),  # kurtosis
        ])

    features = blockiness_features + lap_features + color_features
    assert len(features) == 52, f"Expected 52 features, got {len(features)}"

    return np.array(features, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# Full Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

TOTAL_FEATURES = 266 + 16 + 18 + 6 + 20 + 27 + 52  # = 405


def extract_all_features(img_path: str) -> np.ndarray:
    """Extract all 405 features from a single image."""
    img = Image.open(img_path).convert('RGB')

    # Compute all 7 ELA maps exactly once (7 recompressions total)
    residuals = compute_ela_cache(img)

    features = np.concatenate([
        extract_ela_features(residuals),              # 266
        extract_ratio_features(residuals),            # 16
        extract_entropy_features(residuals),          # 18
        extract_fft_features(img),                    # 6
        extract_edge_features(img),                   # 20
        extract_srm_features(img),                    # 27
        extract_dct_noise_color_features(img),        # 52
    ])

    assert features.shape[0] == TOTAL_FEATURES, \
        f"Expected {TOTAL_FEATURES} features, got {features.shape[0]}"
    return features.astype(np.float32)


def extract_dataset(image_paths: list, n_workers: int = None) -> np.ndarray:
    """Extract features for a list of image paths using multiprocessing."""
    if n_workers is None:
        n_workers = min(cpu_count(), 12)

    with Pool(n_workers) as pool:
        results = pool.map(extract_all_features, image_paths)

    return np.array(results, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract 405-dimensional MS-ELA features for forgery screening')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to CASIA2 directory (containing Au/ and Tp/)')
    parser.add_argument('--output', type=str, default='features.npy',
                        help='Output .npy file path')
    parser.add_argument('--jpeg-only', action='store_true', default=False,
                        help='Extract only from JPEG files (primary protocol)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    au_dir = data_dir / 'Au'
    tp_dir = data_dir / 'Tp'

    jpeg_ext = {'.jpg', '.jpeg'}
    all_ext = {'.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    valid_ext = jpeg_ext if args.jpeg_only else all_ext

    au_files = sorted([f for f in au_dir.iterdir()
                       if f.suffix.lower() in valid_ext
                       and not f.name.startswith('.')])
    tp_files = sorted([f for f in tp_dir.iterdir()
                       if f.suffix.lower() in valid_ext
                       and not f.name.startswith('.')
                       and f.name != 'Thumbs.db'])

    print(f"Authentic: {len(au_files)}, Tampered: {len(tp_files)}")
    print(f"Total: {len(au_files) + len(tp_files)} images")
    subset_label = "JPEG-only" if args.jpeg_only else "all formats"
    print(f"Subset: {subset_label}")

    all_paths = [str(f) for f in au_files + tp_files]
    labels = np.array([0] * len(au_files) + [1] * len(tp_files))

    n_workers = args.workers or min(cpu_count(), 12)
    print(f"Extracting with {n_workers} workers...")
    X = extract_dataset(all_paths, n_workers=n_workers)

    np.save(args.output, X)
    labels_path = args.output.replace('.npy', '_labels.npy')
    np.save(labels_path, labels)

    # Save filenames for alignment with splits
    filenames_path = args.output.replace('.npy', '_filenames.txt')
    with open(filenames_path, 'w') as f:
        for p in au_files + tp_files:
            f.write(p.name + '\n')

    print(f"Saved features: {args.output} {X.shape}")
    print(f"Saved labels: {labels_path}")
    print(f"Saved filenames: {filenames_path}")
