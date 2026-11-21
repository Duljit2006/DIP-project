"""
src/metrics.py - Image Quality Evaluation Metrics (PSNR & SSIM)
----------------------------------------------------------------
Lead: Digital Image Processing & Evaluation Lead (HER Track)

This module provides standard digital image processing fidelity metrics:
1. PSNR (Peak Signal-to-Noise Ratio in dB)
2. SSIM (Structural Similarity Index from 0.0 to 1.0)
3. Automated baseline benchmark evaluation across benchmark datasets (Set5, Set14).
"""

import sys
import os
import glob
import cv2
import numpy as np

# Ensure project root is in sys.path so 'src' imports work from anywhere
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
try:
    from src.dct_utils import load_and_slice, apply_dct_and_truncate, reconstruct_from_dct
except ImportError:
    from dct_utils import load_and_slice, apply_dct_and_truncate, reconstruct_from_dct


def calculate_psnr(original, reconstructed):
    """
    Calculates Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).
    Formula: 10 * log10(MAX^2 / MSE)
    Higher is better. Typical range for lossy compression: 20 - 40 dB.

    Args:
        original (np.ndarray): Original image array in [0.0, 1.0].
        reconstructed (np.ndarray): Restored/reconstructed image in [0.0, 1.0].

    Returns:
        float: PSNR score in dB.
    """
    orig = np.clip(original, 0.0, 1.0).astype(np.float32)
    recon = np.clip(reconstructed, 0.0, 1.0).astype(np.float32)
    return float(peak_signal_noise_ratio(orig, recon, data_range=1.0))


def calculate_ssim(original, reconstructed):
    """
    Calculates Structural Similarity Index (SSIM).
    Measures luminance, contrast, and structural fidelity.
    Scale: 0.0 (completely dissimilar) to 1.0 (identical).

    Args:
        original (np.ndarray): Original image array in [0.0, 1.0].
        reconstructed (np.ndarray): Restored/reconstructed image in [0.0, 1.0].

    Returns:
        float: SSIM score between 0.0 and 1.0.
    """
    orig = np.clip(original, 0.0, 1.0).astype(np.float32)
    recon = np.clip(reconstructed, 0.0, 1.0).astype(np.float32)
    return float(structural_similarity(orig, recon, data_range=1.0))


def evaluate_baseline_folder(folder_path, cutoff=15):
    """
    Evaluates the classical DCT truncation baseline on all images in a folder.

    Args:
        folder_path (str): Path to directory with test images.
        cutoff (int): Preserved zigzag low frequencies (default: 15).

    Returns:
        results (list of dicts): Per-image evaluation metrics.
        avg_psnr (float): Average baseline PSNR across all images.
        avg_ssim (float): Average baseline SSIM across all images.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
    image_paths.sort()

    if not image_paths:
        raise ValueError(f"No valid images found in folder: {folder_path}")

    results = []
    for path in image_paths:
        fname = os.path.basename(path)
        blocks, shape = load_and_slice(path)
        X_trunc, _ = apply_dct_and_truncate(blocks, cutoff=cutoff)
        baseline_img = reconstruct_from_dct(X_trunc, shape)

        # Load clean cropped original image for comparison
        orig_full = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        h, w = shape
        orig_clean = orig_full[:h, :w]

        psnr_val = calculate_psnr(orig_clean, baseline_img)
        ssim_val = calculate_ssim(orig_clean, baseline_img)

        results.append({
            "filename": fname,
            "shape": shape,
            "psnr": psnr_val,
            "ssim": ssim_val,
        })

    avg_psnr = float(np.mean([r["psnr"] for r in results]))
    avg_ssim = float(np.mean([r["ssim"] for r in results]))

    return results, avg_psnr, avg_ssim


def print_evaluation_table(dataset_name, results, avg_psnr, avg_ssim, cutoff=15):
    """
    Prints a clean, nicely formatted ASCII table of evaluation scores and saves to results.
    """
    lines = []
    lines.append("=" * 68)
    lines.append(f"  BASELINE EVALUATION: {dataset_name} (Cutoff = {cutoff} frequencies)")
    lines.append("=" * 68)
    lines.append(f"{'Image Name':<24} | {'Resolution':<12} | {'PSNR (dB)':<11} | {'SSIM':<8}")
    lines.append("-" * 68)
    for item in results:
        res_str = f"{item['shape'][0]}x{item['shape'][1]}"
        lines.append(f"{item['filename']:<24} | {res_str:<12} | {item['psnr']:>6.2f} dB   | {item['ssim']:>6.4f}")
    lines.append("-" * 68)
    lines.append(f"{'AVERAGE BASELINE':<24} | {'-':<12} | {avg_psnr:>6.2f} dB   | {avg_ssim:>6.4f}")
    lines.append("=" * 68 + "\n")

    table_text = "\n".join(lines)
    print(table_text)
    return table_text


if __name__ == "__main__":
    print("\n[INFO] Running Classical DIP Metrics & Baseline Benchmarking...\n")
    all_reports = []

    # Evaluate Set5
    set5_dir = os.path.join("datasets", "Set5")
    if os.path.exists(set5_dir):
        res5, avg_p5, avg_s5 = evaluate_baseline_folder(set5_dir, cutoff=15)
        all_reports.append(print_evaluation_table("Set5 Benchmark (5 images)", res5, avg_p5, avg_s5, cutoff=15))

    # Evaluate Set14
    set14_dir = os.path.join("datasets", "Set14")
    if os.path.exists(set14_dir):
        res14, avg_p14, avg_s14 = evaluate_baseline_folder(set14_dir, cutoff=15)
        all_reports.append(print_evaluation_table("Set14 Benchmark (14 images)", res14, avg_p14, avg_s14, cutoff=15))

    # Save output report to results/
    os.makedirs("results", exist_ok=True)
    report_file = os.path.join("results", "baseline_scores.txt")
    with open(report_file, "w") as f:
        f.write("\n".join(all_reports))
    print(f"[SUCCESS] Baseline report saved to: {report_file}")
