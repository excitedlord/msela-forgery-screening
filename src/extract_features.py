"""
Feature extraction pipeline for Image Forgery Screening.

Extracts 405 features across 7 families from JPEG images:
  - Multi-scale ELA (266 features): 7 quality levels × 38 statistics
  - Cross-quality ratios (16 features): 8 quality pairs × 2 stats
  - Block-entropy (18 features): 3 block sizes × 6 stats
  - FFT spectral (6 features): magnitude spectrum statistics
  - Edge density (20 features): 4 detectors × 5 scale stats
  - SRM noise (27 features): 3 filters × 9 stats
  - DCT/noise/color (52 features): quantization table, noise, color stats
"""

import argparse
import io
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
from PIL import Image
from scipy.fft import fft2, fftshift
from scipy.ndimage import uniform_filter
from scipy.stats import entropy as sp_entropy


# ═══════════════════════════════════════════════════════════════════════════════
# ELA Features (266)
# ═══════════════════════════════════════════════════════════════════════════════

QUALITIES = [30, 50, 60, 75, 80, 90, 95]
STATS_PER_QUALITY = 38  # mean, std, skew, kurt, percentiles, block stats, etc.


def compute_ela(img: Image.Image, quality: int) -> np.ndarray:
    """Compute ELA residual at a given JPEG quality."""
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf)
    
    orig = np.array(img.convert('RGB'), dtype=np.float32)
    recomp = np.array(recompressed.convert('RGB'), dtype=np.float32)
    residual = np.abs(orig - recomp)
    return residual.mean(axis=2)  # Average across channels


def block_stats(arr: np.ndarray, block_size: int = 8) -> np.ndarray:
    """Compute block-level statistics (mean, std) of a 2D array."""
    h, w = arr.shape
    h_trim = (h // block_size) * block_size
    w_trim = (w // block_size) * block_size
    blocks = arr[:h_trim, :w_trim].reshape(h_trim // block_size, block_size,
                                            w_trim // block_size, block_size)
    block_means = blocks.mean(axis=(1, 3)).ravel()
    return np.array([block_means.mean(), block_means.std(),
                     np.percentile(block_means, 5),
                     np.percentile(block_means, 95)])


def ela_stats(residual: np.ndarray) -> np.ndarray:
    """Compute 38 statistics from an ELA residual map."""
    flat = residual.ravel()
    stats = [
        flat.mean(), flat.std(),
        np.percentile(flat, 1), np.percentile(flat, 5),
        np.percentile(flat, 10), np.percentile(flat, 25),
        np.percentile(flat, 50), np.percentile(flat, 75),
        np.percentile(flat, 90), np.percentile(flat, 95),
        np.percentile(flat, 99),
    ]
    # Skewness and kurtosis
    m = flat.mean()
    s = flat.std() + 1e-10
    stats.append(((flat - m) / s).mean() ** 3)  # Approx skewness
    stats.append(((flat - m) / s).mean() ** 4)  # Approx kurtosis
    
    # Spatial statistics
    h, w = residual.shape
    # Quadrant means
    mid_h, mid_w = h // 2, w // 2
    stats.extend([
        residual[:mid_h, :mid_w].mean(),
        residual[:mid_h, mid_w:].mean(),
        residual[mid_h:, :mid_w].mean(),
        residual[mid_h:, mid_w:].mean(),
    ])
    # Row/col variation
    row_means = residual.mean(axis=1)
    col_means = residual.mean(axis=0)
    stats.extend([row_means.std(), col_means.std()])
    
    # Gradient stats
    grad_h = np.diff(residual, axis=0)
    grad_w = np.diff(residual, axis=1)
    stats.extend([np.abs(grad_h).mean(), np.abs(grad_w).mean()])
    
    # Block stats
    bs = block_stats(residual, 8)
    stats.extend(bs.tolist())
    bs16 = block_stats(residual, 16)
    stats.extend(bs16.tolist())
    
    # Entropy of histogram
    hist, _ = np.histogram(flat, bins=64, density=True)
    stats.append(sp_entropy(hist + 1e-10))
    
    # High-residual fraction
    stats.append((flat > flat.mean() + 2 * flat.std()).mean())
    
    # Pad or truncate to exactly 38
    stats = stats[:STATS_PER_QUALITY]
    while len(stats) < STATS_PER_QUALITY:
        stats.append(0.0)
    
    return np.array(stats, dtype=np.float32)


def extract_ela_features(img: Image.Image) -> np.ndarray:
    """Extract multi-scale ELA features: 7 qualities × 38 stats = 266."""
    features = []
    for q in QUALITIES:
        residual = compute_ela(img, q)
        features.append(ela_stats(residual))
    return np.concatenate(features)


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-Quality Ratio Features (16)
# ═══════════════════════════════════════════════════════════════════════════════

RATIO_PAIRS = [(30, 90), (30, 75), (50, 90), (50, 95),
               (60, 90), (60, 95), (75, 95), (80, 95)]


def extract_ratio_features(img: Image.Image) -> np.ndarray:
    """Extract 16 cross-quality ratio features."""
    # Cache ELA residuals
    residuals = {}
    for q in set(q for pair in RATIO_PAIRS for q in pair):
        residuals[q] = compute_ela(img, q)
    
    features = []
    for qa, qb in RATIO_PAIRS:
        ratio = (residuals[qa].mean() + 1e-10) / (residuals[qb].mean() + 1e-10)
        ratio_std = (residuals[qa].std() + 1e-10) / (residuals[qb].std() + 1e-10)
        features.extend([ratio, ratio_std])
    
    return np.array(features, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Block Entropy Features (18)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_entropy_features(img: Image.Image) -> np.ndarray:
    """Extract 18 block-entropy features."""
    gray = np.array(img.convert('L'), dtype=np.float32)
    features = []
    
    for block_size in [8, 16, 32]:
        h, w = gray.shape
        h_trim = (h // block_size) * block_size
        w_trim = (w // block_size) * block_size
        trimmed = gray[:h_trim, :w_trim]
        blocks = trimmed.reshape(h_trim // block_size, block_size,
                                  w_trim // block_size, block_size)
        
        entropies = []
        for i in range(blocks.shape[0]):
            for j in range(blocks.shape[2]):
                block = blocks[i, :, j, :].ravel()
                hist, _ = np.histogram(block, bins=32, density=True)
                entropies.append(sp_entropy(hist + 1e-10))
        
        entropies = np.array(entropies)
        features.extend([
            entropies.mean(), entropies.std(),
            np.percentile(entropies, 10), np.percentile(entropies, 90),
            entropies.max() - entropies.min(),
            (entropies > entropies.mean() + entropies.std()).mean()
        ])
    
    return np.array(features, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# FFT Spectral Features (6)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_fft_features(img: Image.Image) -> np.ndarray:
    """Extract 6 FFT spectral features."""
    gray = np.array(img.convert('L'), dtype=np.float32)
    f_transform = fftshift(fft2(gray))
    magnitude = np.log1p(np.abs(f_transform))
    
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    
    # Radial energy distribution
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((X - cx)**2 + (Y - cy)**2)
    
    low = magnitude[R < min(h, w) * 0.1].mean()
    mid = magnitude[(R >= min(h, w) * 0.1) & (R < min(h, w) * 0.3)].mean()
    high = magnitude[R >= min(h, w) * 0.3].mean()
    
    return np.array([
        low, mid, high,
        low / (high + 1e-10),
        magnitude.mean(), magnitude.std()
    ], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Edge Density Features (20)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_edge_features(img: Image.Image) -> np.ndarray:
    """Extract 20 edge density features."""
    gray = np.array(img.convert('L'), dtype=np.float32)
    features = []
    
    # Sobel-like gradients at multiple scales
    for sigma in [1, 2, 4, 8]:
        smoothed = uniform_filter(gray, size=sigma)
        grad_h = np.diff(smoothed, axis=0)
        grad_w = np.diff(smoothed, axis=1)
        mag = np.sqrt(grad_h[:, :-1]**2 + grad_w[:-1, :]**2)
        
        threshold = mag.mean() + mag.std()
        edge_density = (mag > threshold).mean()
        
        features.extend([
            mag.mean(), mag.std(), edge_density,
            np.percentile(mag, 90), np.percentile(mag, 99)
        ])
    
    return np.array(features, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# SRM Noise Features (27)
# ═══════════════════════════════════════════════════════════════════════════════

# SRM filter kernels (simplified versions)
SRM_FILTERS = {
    'edge3x3': np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32) / 4,
    'square3x3': np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32) / 4,
    'square5x5_edge': np.array([
        [-1, 2, -2, 2, -1],
        [2, -6, 8, -6, 2],
        [-2, 8, -12, 8, -2],
        [2, -6, 8, -6, 2],
        [-1, 2, -2, 2, -1]
    ], dtype=np.float32) / 12,
}


def apply_srm_filter(gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply SRM filter via convolution."""
    from scipy.signal import convolve2d
    return convolve2d(gray, kernel, mode='same', boundary='symm')


def extract_srm_features(img: Image.Image) -> np.ndarray:
    """Extract 27 SRM noise residual features."""
    gray = np.array(img.convert('L'), dtype=np.float32)
    features = []
    
    for name, kernel in SRM_FILTERS.items():
        noise = apply_srm_filter(gray, kernel)
        flat = noise.ravel()
        features.extend([
            flat.mean(), flat.std(), np.abs(flat).mean(),
            np.percentile(np.abs(flat), 75),
            np.percentile(np.abs(flat), 95),
            np.percentile(np.abs(flat), 99),
            sp_entropy(np.histogram(flat, bins=64, density=True)[0] + 1e-10),
            (np.abs(flat) > 3 * flat.std()).mean(),
            np.abs(flat).max()
        ])
    
    return np.array(features, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# DCT / Noise / Color Features (52)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_dct_noise_color_features(img: Image.Image) -> np.ndarray:
    """Extract 52 DCT, noise estimation, and color features."""
    rgb = np.array(img.convert('RGB'), dtype=np.float32)
    gray = np.array(img.convert('L'), dtype=np.float32)
    features = []
    
    # Noise estimation via median absolute deviation
    from scipy.ndimage import median_filter
    noise_est = gray - median_filter(gray, size=3)
    features.extend([
        noise_est.std(),
        np.abs(noise_est).mean(),
        np.percentile(np.abs(noise_est), 95),
    ])
    
    # Color channel statistics
    for ch in range(3):
        channel = rgb[:, :, ch]
        features.extend([
            channel.mean(), channel.std(),
            np.percentile(channel, 5), np.percentile(channel, 95),
        ])
    
    # Inter-channel correlations
    r, g, b = rgb[:, :, 0].ravel(), rgb[:, :, 1].ravel(), rgb[:, :, 2].ravel()
    features.extend([
        np.corrcoef(r, g)[0, 1],
        np.corrcoef(r, b)[0, 1],
        np.corrcoef(g, b)[0, 1],
    ])
    
    # Color saturation stats
    hsv_s = np.sqrt(((rgb - rgb.mean(axis=2, keepdims=True))**2).sum(axis=2))
    features.extend([hsv_s.mean(), hsv_s.std()])
    
    # DCT-domain approximation: block variance
    h, w = gray.shape
    block_size = 8
    h_trim = (h // block_size) * block_size
    w_trim = (w // block_size) * block_size
    trimmed = gray[:h_trim, :w_trim]
    blocks = trimmed.reshape(h_trim // block_size, block_size,
                              w_trim // block_size, block_size)
    block_vars = blocks.var(axis=(1, 3)).ravel()
    features.extend([
        block_vars.mean(), block_vars.std(),
        np.percentile(block_vars, 10),
        np.percentile(block_vars, 50),
        np.percentile(block_vars, 90),
        (block_vars < 10).mean(),  # Smooth block ratio
    ])
    
    # Pad to exactly 52
    features = features[:52]
    while len(features) < 52:
        features.append(0.0)
    
    return np.array(features, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Full Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def extract_all_features(img_path: str) -> np.ndarray:
    """Extract all 405 features from a single image."""
    img = Image.open(img_path).convert('RGB')
    
    features = np.concatenate([
        extract_ela_features(img),        # 266
        extract_ratio_features(img),      # 16
        extract_entropy_features(img),    # 18
        extract_fft_features(img),        # 6
        extract_edge_features(img),       # 20
        extract_srm_features(img),        # 27
        extract_dct_noise_color_features(img),  # 52
    ])
    
    assert features.shape[0] == 405, f"Expected 405 features, got {features.shape[0]}"
    return features


def extract_dataset(image_paths: list, n_workers: int = None) -> np.ndarray:
    """Extract features for a list of image paths using multiprocessing."""
    if n_workers is None:
        n_workers = min(cpu_count(), 8)
    
    with Pool(n_workers) as pool:
        results = pool.map(extract_all_features, image_paths)
    
    return np.array(results, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Screening features')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to CASIA2 directory (containing Au/ and Tp/)')
    parser.add_argument('--output', type=str, default='features.npy',
                        help='Output .npy file path')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    au_dir = data_dir / 'Au'
    tp_dir = data_dir / 'Tp'
    
    # Collect JPEG-only files
    au_files = sorted([f for f in au_dir.iterdir()
                       if f.suffix.lower() in {'.jpg', '.jpeg'}])
    tp_files = sorted([f for f in tp_dir.iterdir()
                       if f.suffix.lower() in {'.jpg', '.jpeg'}])
    
    print(f"Authentic JPEG: {len(au_files)}, Tampered JPEG: {len(tp_files)}")
    
    all_paths = [str(f) for f in au_files + tp_files]
    labels = np.array([0] * len(au_files) + [1] * len(tp_files))
    
    print(f"Extracting {len(all_paths)} images with {args.workers or cpu_count()} workers...")
    X = extract_dataset(all_paths, n_workers=args.workers)
    
    np.save(args.output, X)
    np.save(args.output.replace('.npy', '_labels.npy'), labels)
    print(f"Saved: {args.output} ({X.shape})")
