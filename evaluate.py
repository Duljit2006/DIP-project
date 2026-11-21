"""
evaluate.py - Benchmark Evaluation and Visual Verification
------------------------------------------------------------
Lead: Digital Image Processing & Evaluation Lead (HER Track)

This script evaluates the trained FrequencyGNN model against the classical
JPEG-style baseline across benchmark datasets (Set14, Set14):
1. Loads trained weights from checkpoints/model.pth.
2. Runs batch inference on test images to predict missing high-frequency coefficients.
3. Reconstructs full images using 2D-IDCT.
4. Computes comparative metrics (PSNR gain, SSIM gain).
5. Generates side-by-side comparison figures and error heatmaps in results/.
"""

import os
import sys

# Prevent Python from writing bytecode cache (.pyc)
sys.dont_write_bytecode = True

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Immediately delete previous results and bytecode caches before execution
from src.cleanup import clean_all
clean_all(clear_results=True)

import argparse
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from src.device_utils import get_optimal_device
from src.dct_utils import load_and_slice, apply_dct_and_truncate, reconstruct_from_dct
from src.metrics import calculate_psnr, calculate_ssim
from src.dataset import create_zigzag_mask, get_dct_dataloader
from src.models import FrequencyGNN


def evaluate_image(model, image_path, cutoff=15, device="cpu"):
    """
    Evaluates a single image through both the Baseline and GNN reconstruction pipelines.

    Args:
        model (FrequencyGNN): Trained PyTorch model.
        image_path (str): Path to image file.
        cutoff (int): Preserved zigzag low frequencies (default: 15).
        device (str): Inference device ('cpu' or 'cuda').

    Returns:
        dict: Contains images and quantitative evaluation scores.
    """
    model.eval()

    # 1. Load image and slice into 8x8 blocks
    blocks, shape = load_and_slice(image_path)
    X_trunc, Y_full = apply_dct_and_truncate(blocks, cutoff=cutoff)

    # 2. Baseline Reconstruction (IDCT with missing frequencies set to 0.0)
    baseline_img = reconstruct_from_dct(X_trunc, shape)

    # 3. GNN Model Inference (Batched for speed)
    dataloader = get_dct_dataloader(X_trunc, Y_full, cutoff=cutoff, batch_size=128, shuffle=False)
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            preds = model(batch.x, batch.edge_index)  # Shape: (B * 64,)
            all_preds.append(preds.cpu().numpy().reshape(-1, 64))

    all_preds = np.concatenate(all_preds, axis=0)  # Shape: (num_blocks, 64)

    # 4. Synthesize Final Frequency Blocks:
    # Strictly preserve true known low frequencies; fill in missing high frequencies from GNN.
    missing_mask = create_zigzag_mask(cutoff=cutoff).numpy()  # True for missing positions
    X_gnn_restored = X_trunc.copy()
    X_gnn_restored[:, missing_mask] = all_preds[:, missing_mask]

    # 5. IDCT Reconstruction using GNN restored coefficients
    gnn_img = reconstruct_from_dct(X_gnn_restored, shape)

    # 6. Load ground-truth cropped original image
    orig_raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    h, w = shape
    original_img = orig_raw[:h, :w]

    # 7. Calculate Metrics
    base_psnr = calculate_psnr(original_img, baseline_img)
    base_ssim = calculate_ssim(original_img, baseline_img)
    gnn_psnr = calculate_psnr(original_img, gnn_img)
    gnn_ssim = calculate_ssim(original_img, gnn_img)

    return {
        "original": original_img,
        "baseline": baseline_img,
        "gnn": gnn_img,
        "baseline_psnr": base_psnr,
        "baseline_ssim": base_ssim,
        "gnn_psnr": gnn_psnr,
        "gnn_ssim": gnn_ssim,
        "psnr_gain": gnn_psnr - base_psnr,
        "ssim_gain": gnn_ssim - base_ssim,
        "shape": shape,
    }


def save_visual_comparison(res, image_name, save_dir="results"):
    """
    Saves a 3-panel comparison figure: [Original] vs [Baseline] vs [GNN Restored].
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(res["original"], cmap="gray")
    axes[0].set_title("Original (Ground Truth)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(res["baseline"], cmap="gray")
    axes[1].set_title(
        f"Baseline JPEG (Cutoff={15})\nPSNR: {res['baseline_psnr']:.2f} dB | SSIM: {res['baseline_ssim']:.4f}",
        fontsize=10,
    )
    axes[1].axis("off")

    gain_str = f"+{res['psnr_gain']:.2f} dB" if res['psnr_gain'] >= 0 else f"{res['psnr_gain']:.2f} dB"
    axes[2].imshow(res["gnn"], cmap="gray")
    axes[2].set_title(
        f"GNN Restored (Our Model)\nPSNR: {res['gnn_psnr']:.2f} dB ({gain_str}) | SSIM: {res['gnn_ssim']:.4f}",
        fontsize=10,
        color="darkgreen" if res['psnr_gain'] >= 0 else "black",
    )
    axes[2].axis("off")

    plt.suptitle(f"Image Reconstruction Comparison: {image_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(save_dir, f"{image_name}_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def save_error_heatmap(res, image_name, save_dir="results"):
    """
    Saves pixel error difference maps comparing Baseline Error vs GNN Error.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    err_base = np.abs(res["original"] - res["baseline"])
    err_gnn = np.abs(res["original"] - res["gnn"])
    vmax = max(err_base.max(), 0.15)

    im1 = axes[0].imshow(err_base, cmap="hot", vmin=0, vmax=vmax)
    axes[0].set_title("Baseline Reconstruction Error Map", fontsize=11)
    axes[0].axis("off")

    im2 = axes[1].imshow(err_gnn, cmap="hot", vmin=0, vmax=vmax)
    axes[1].set_title("GNN Reconstruction Error Map (Darker = Less Error)", fontsize=11)
    axes[1].axis("off")

    plt.colorbar(im2, ax=axes, fraction=0.025, pad=0.04)
    plt.suptitle(f"Error Heatmap Comparison: {image_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(save_dir, f"{image_name}_error_heatmap.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FrequencyGNN against classical baseline.")
    parser.add_argument("--dataset", type=str, default="Set14", choices=["Set14", "Set5", "all"],
                        help="Benchmark dataset to evaluate on ('Set14', 'Set5', or 'all'). Default: Set14.")
    parser.add_argument("--device", type=str, default="auto",
                        help="Hardware accelerator ('auto', 'cuda', 'mps', 'xpu', 'directml', 'cpu'). Default: auto.")
    parser.add_argument("--cutoff", type=int, default=15,
                        help="Preserved low-frequency coefficients (default: 15).")
    parser.add_argument("--save_dir", type=str, default="results",
                        help="Directory to save visual comparison plots and scores (default: results).")
    return parser.parse_args()


def evaluate_dataset(model, dataset_name, cutoff=15, device="cpu", save_dir="results"):
    images = sorted(glob.glob(os.path.join("datasets", dataset_name, "*.png")))
    if not images:
        print(f"[WARNING] No images found in datasets/{dataset_name}")
        return []

    print(f"Evaluating {len(images)} images from {dataset_name} Benchmark...\n")
    results = []
    lines = []
    lines.append("=" * 78)
    lines.append(f"  BENCHMARK EVALUATION RESULTS: {dataset_name} (Baseline vs GNN)")
    lines.append("=" * 78)
    lines.append(f"{'Image':<18} | {'Baseline PSNR':<14} | {'GNN PSNR':<14} | {'Gain (dB)':<10} | {'GNN SSIM':<8}")
    lines.append("-" * 78)

    for path in images:
        fname = os.path.basename(path)
        base_name = os.path.splitext(fname)[0]
        res = evaluate_image(model, path, cutoff=cutoff, device=device)
        results.append(res)

        save_visual_comparison(res, f"{dataset_name}_{base_name}", save_dir=save_dir)
        save_error_heatmap(res, f"{dataset_name}_{base_name}", save_dir=save_dir)

        gain_prefix = "+" if res['psnr_gain'] >= 0 else ""
        gain_str = f"{gain_prefix}{res['psnr_gain']:.2f} dB"
        line = f"{fname:<18} | {res['baseline_psnr']:>6.2f} dB       | {res['gnn_psnr']:>6.2f} dB       | {gain_str:>9}  | {res['gnn_ssim']:>6.4f}"
        lines.append(line)
        print(f"  [{dataset_name}] Processed {fname:<18} -> Baseline: {res['baseline_psnr']:.2f} dB | GNN: {res['gnn_psnr']:.2f} dB ({gain_str})")

    avg_base_psnr = float(np.mean([r["baseline_psnr"] for r in results]))
    avg_gnn_psnr = float(np.mean([r["gnn_psnr"] for r in results]))
    avg_gain = avg_gnn_psnr - avg_base_psnr
    avg_base_ssim = float(np.mean([r["baseline_ssim"] for r in results]))
    avg_gnn_ssim = float(np.mean([r["gnn_ssim"] for r in results]))

    lines.append("-" * 78)
    gain_prefix = "+" if avg_gain >= 0 else ""
    lines.append(f"{'AVERAGE':<18} | {avg_base_psnr:>6.2f} dB       | {avg_gnn_psnr:>6.2f} dB       | {gain_prefix}{avg_gain:>7.2f} dB  | {avg_gnn_ssim:>6.4f}")
    lines.append("=" * 78)

    report_text = "\n".join(lines)
    print("\n" + report_text + "\n")
    return report_text


def main():
    args = parse_args()

    # Intelligent device selection (Dedicated GPU -> Integrated GPU -> CPU)
    device, device_info = get_optimal_device(preference=args.device, verbose=True)

    # Check for trained checkpoint
    checkpoint_path = os.path.join("checkpoints", "model.pth")
    if not os.path.exists(checkpoint_path):
        print(f"\n[ERROR] Model checkpoint not found at: {checkpoint_path}")
        print("Please run 'python train.py' first to generate model weights.")
        return

    # Load model onto optimal device
    model = FrequencyGNN(in_channels=3, hidden_channels=128, num_layers=3, dropout=0.0).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded trained weights from: {checkpoint_path} onto {device}\n")

    os.makedirs(args.save_dir, exist_ok=True)
    all_reports = []

    if args.dataset in ("Set14", "all"):
        rep14 = evaluate_dataset(model, "Set14", cutoff=args.cutoff, device=device, save_dir=args.save_dir)
        if rep14:
            all_reports.append(rep14)

    if args.dataset in ("Set5", "all"):
        rep5 = evaluate_dataset(model, "Set5", cutoff=args.cutoff, device=device, save_dir=args.save_dir)
        if rep5:
            all_reports.append(rep5)

    # Save final report to results/
    final_report_path = os.path.join(args.save_dir, "final_evaluation_scores.txt")
    with open(final_report_path, "w") as f:
        f.write("\n\n".join(all_reports))
    print(f"[SUCCESS] Final evaluation score report saved to: {final_report_path}")
    print(f"[SUCCESS] Visual comparison plots and error heatmaps saved into: {args.save_dir}/\n")


if __name__ == "__main__":
    main()
