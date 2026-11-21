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
from src.models import FrequencyGNN, SpatialRefinerCNN


def evaluate_image(model, cnn_model=None, image_path="", cutoff=15, device="cpu"):
    """
    Evaluates a single image through Baseline, GNN, and Dual-Domain Hybrid pipelines.
    Supports full 3-channel RGB color images (via standard YCbCr luminance processing).

    Args:
        model (FrequencyGNN): Trained Stage 1 GNN model.
        cnn_model (SpatialRefinerCNN, optional): Trained Stage 2 Spatial Refiner CNN.
        image_path (str): Path to image file.
        cutoff (int): Preserved zigzag low frequencies (default: 15).
        device (str): Inference device ('cpu' or 'cuda').

    Returns:
        dict: Contains images and quantitative evaluation scores for all stages.
    """
    model.eval()

    # Check if image is color or grayscale
    raw_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if raw_bgr is None:
        raise ValueError(f"Could not load image at path: {image_path}")

    h, w, c = raw_bgr.shape
    h_crop, w_crop = (h // 8) * 8, (w // 8) * 8
    raw_bgr = raw_bgr[:h_crop, :w_crop]
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Convert to YCrCb color space (Standard JPEG pipeline)
    raw_ycrcb = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2YCrCb)
    Y_channel = raw_ycrcb[:, :, 0]
    Cr_channel = raw_ycrcb[:, :, 1]
    Cb_channel = raw_ycrcb[:, :, 2]

    # Slice Luminance (Y) channel into 8x8 blocks
    blocks_Y = []
    for r in range(0, h_crop, 8):
        for c in range(0, w_crop, 8):
            blocks_Y.append(Y_channel[r:r+8, c:c+8])
    blocks_Y = np.array(blocks_Y, dtype=np.float32)

    # 1. 2D-DCT & Truncate on Y channel
    X_trunc, Y_full = apply_dct_and_truncate(blocks_Y, cutoff=cutoff)

    # 2. Baseline Reconstruction (IDCT with zero-filled missing high frequencies)
    baseline_Y = reconstruct_from_dct(X_trunc, (h_crop, w_crop))

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
    missing_mask = create_zigzag_mask(cutoff=cutoff).numpy()
    X_gnn_restored = X_trunc.copy()
    X_gnn_restored[:, missing_mask] = all_preds[:, missing_mask]

    # 5. IDCT Reconstruction using GNN restored coefficients (Stage 1)
    gnn_Y = reconstruct_from_dct(X_gnn_restored, (h_crop, w_crop))

    # 6. Reconstruct Full 3-Channel RGB Images
    baseline_ycrcb = np.stack([np.clip(baseline_Y, 0.0, 1.0), Cr_channel, Cb_channel], axis=-1)
    baseline_rgb = np.clip(cv2.cvtColor(baseline_ycrcb, cv2.COLOR_YCrCb2RGB), 0.0, 1.0)

    gnn_ycrcb = np.stack([np.clip(gnn_Y, 0.0, 1.0), Cr_channel, Cb_channel], axis=-1)
    gnn_rgb = np.clip(cv2.cvtColor(gnn_ycrcb, cv2.COLOR_YCrCb2RGB), 0.0, 1.0)

    original_rgb = raw_rgb

    # 7. Metrics for Baseline & Stage 1 GNN
    base_psnr = calculate_psnr(Y_channel, baseline_Y)
    base_ssim = calculate_ssim(Y_channel, baseline_Y)
    gnn_psnr = calculate_psnr(Y_channel, gnn_Y)
    gnn_ssim = calculate_ssim(Y_channel, gnn_Y)

    # 8. Stage 2: Spatial Refiner CNN (Dual-Domain Hybrid)
    hybrid_rgb = None
    hybrid_psnr = None
    hybrid_ssim = None
    hybrid_gain = None

    if cnn_model is not None:
        cnn_model.eval()
        with torch.no_grad():
            gnn_Y_tensor = torch.from_numpy(gnn_Y).float().unsqueeze(0).unsqueeze(0).to(device)
            refined_tensor = cnn_model(gnn_Y_tensor)
            hybrid_Y = np.clip(refined_tensor.squeeze().cpu().numpy(), 0.0, 1.0)

        hybrid_ycrcb = np.stack([hybrid_Y, Cr_channel, Cb_channel], axis=-1)
        hybrid_rgb = np.clip(cv2.cvtColor(hybrid_ycrcb, cv2.COLOR_YCrCb2RGB), 0.0, 1.0)
        hybrid_psnr = calculate_psnr(Y_channel, hybrid_Y)
        hybrid_ssim = calculate_ssim(Y_channel, hybrid_Y)
        hybrid_gain = hybrid_psnr - base_psnr

    return {
        "original": original_rgb,
        "baseline": baseline_rgb,
        "gnn": gnn_rgb,
        "hybrid": hybrid_rgb,
        "is_color": True,
        "baseline_psnr": base_psnr,
        "baseline_ssim": base_ssim,
        "gnn_psnr": gnn_psnr,
        "gnn_ssim": gnn_ssim,
        "psnr_gain": gnn_psnr - base_psnr,
        "ssim_gain": gnn_ssim - base_ssim,
        "hybrid_psnr": hybrid_psnr,
        "hybrid_ssim": hybrid_ssim,
        "hybrid_gain": hybrid_gain,
        "shape": (h_crop, w_crop),
    }


def save_visual_comparison(res, image_name, save_dir="results"):
    """
    Saves multi-panel comparison figure in vibrant FULL COLOR:
    [Original Color] vs [Baseline Color] vs [GNN Restored Color] vs [Hybrid Restored Color].
    """
    os.makedirs(save_dir, exist_ok=True)
    has_hybrid = res.get("hybrid") is not None
    n_cols = 4 if has_hybrid else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    is_color = res.get("is_color", False)
    cmap_arg = None if is_color else "gray"

    # Panel 1: Original
    axes[0].imshow(res["original"], cmap=cmap_arg)
    axes[0].set_title("Original (Ground Truth)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Baseline
    axes[1].imshow(res["baseline"], cmap=cmap_arg)
    axes[1].set_title(
        f"Baseline JPEG (Cutoff={15})\nPSNR: {res['baseline_psnr']:.2f} dB | SSIM: {res['baseline_ssim']:.4f}",
        fontsize=10,
    )
    axes[1].axis("off")

    # Panel 3: GNN Only
    gain_str = f"+{res['psnr_gain']:.2f} dB" if res['psnr_gain'] >= 0 else f"{res['psnr_gain']:.2f} dB"
    axes[2].imshow(res["gnn"], cmap=cmap_arg)
    axes[2].set_title(
        f"Stage 1: Frequency GNN\nPSNR: {res['gnn_psnr']:.2f} dB ({gain_str}) | SSIM: {res['gnn_ssim']:.4f}",
        fontsize=10,
        color="darkgreen" if res['psnr_gain'] >= 0 else "black",
    )
    axes[2].axis("off")

    # Panel 4: Dual-Domain Hybrid (if available)
    if has_hybrid:
        h_gain = res['hybrid_gain']
        h_gain_str = f"+{h_gain:.2f} dB" if h_gain >= 0 else f"{h_gain:.2f} dB"
        axes[3].imshow(res["hybrid"], cmap=cmap_arg)
        axes[3].set_title(
            f"Stage 2: Hybrid (GNN + CNN)\nPSNR: {res['hybrid_psnr']:.2f} dB ({h_gain_str}) | SSIM: {res['hybrid_ssim']:.4f}",
            fontsize=10,
            fontweight="bold",
            color="darkblue" if h_gain >= 0 else "black",
        )
        axes[3].axis("off")

    color_label = "Color RGB" if is_color else "Grayscale"
    plt.suptitle(f"Reconstruction Comparison ({color_label}): {image_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(save_dir, f"{image_name}_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def save_error_heatmap(res, image_name, save_dir="results"):
    """
    Saves pixel error difference maps comparing Baseline Error vs GNN Error vs Hybrid Error.
    """
    os.makedirs(save_dir, exist_ok=True)
    has_hybrid = res.get("hybrid") is not None
    n_cols = 3 if has_hybrid else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))

    is_color = res.get("is_color", False)
    if is_color:
        err_base = np.mean(np.abs(res["original"] - res["baseline"]), axis=-1)
        err_gnn = np.mean(np.abs(res["original"] - res["gnn"]), axis=-1)
        if has_hybrid:
            err_hybrid = np.mean(np.abs(res["original"] - res["hybrid"]), axis=-1)
    else:
        err_base = np.abs(res["original"] - res["baseline"])
        err_gnn = np.abs(res["original"] - res["gnn"])
        if has_hybrid:
            err_hybrid = np.abs(res["original"] - res["hybrid"])

    vmax = max(err_base.max(), 0.15)

    im1 = axes[0].imshow(err_base, cmap="hot", vmin=0, vmax=vmax)
    axes[0].set_title("Baseline Error Map", fontsize=11)
    axes[0].axis("off")

    im2 = axes[1].imshow(err_gnn, cmap="hot", vmin=0, vmax=vmax)
    axes[1].set_title("Stage 1 GNN Error Map", fontsize=11)
    axes[1].axis("off")

    if has_hybrid:
        im3 = axes[2].imshow(err_hybrid, cmap="hot", vmin=0, vmax=vmax)
        axes[2].set_title("Stage 2 Hybrid Error Map (Darkest = Least Error)", fontsize=11, fontweight="bold")
        axes[2].axis("off")
        plt.colorbar(im3, ax=axes, fraction=0.02, pad=0.03)
    else:
        plt.colorbar(im2, ax=axes, fraction=0.025, pad=0.04)

    plt.suptitle(f"Error Heatmap Comparison: {image_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(save_dir, f"{image_name}_error_heatmap.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FrequencyGNN & Dual-Domain Hybrid Model.")
    parser.add_argument("--dataset", type=str, default="Set14", choices=["Set14", "Set5", "all"],
                        help="Benchmark dataset to evaluate on ('Set14', 'Set5', or 'all'). Default: Set14.")
    parser.add_argument("--device", type=str, default="auto",
                        help="Hardware accelerator ('auto', 'cuda', 'mps', 'xpu', 'directml', 'cpu'). Default: auto.")
    parser.add_argument("--cutoff", type=int, default=15,
                        help="Preserved low-frequency coefficients (default: 15).")
    parser.add_argument("--save_dir", type=str, default="results",
                        help="Directory to save visual comparison plots and scores (default: results).")
    return parser.parse_args()


def evaluate_dataset(model, cnn_model, dataset_name, cutoff=15, device="cpu", save_dir="results"):
    images = sorted(glob.glob(os.path.join("datasets", dataset_name, "*.png")))
    if not images:
        print(f"[WARNING] No images found in datasets/{dataset_name}")
        return []

    has_hybrid = cnn_model is not None
    print(f"Evaluating {len(images)} images from {dataset_name} Benchmark (Hybrid Active: {has_hybrid})...\n")
    results = []
    lines = []
    lines.append("=" * 96)
    lines.append(f"  BENCHMARK EVALUATION RESULTS: {dataset_name} (Baseline vs GNN vs Dual-Domain Hybrid)")
    lines.append("=" * 96)

    if has_hybrid:
        lines.append(f"{'Image':<18} | {'Baseline PSNR':<14} | {'GNN PSNR':<13} | {'GNN Gain':<10} | {'Hybrid PSNR':<13} | {'Hybrid Gain':<11} | {'Hybrid SSIM':<8}")
        lines.append("-" * 96)
    else:
        lines.append(f"{'Image':<18} | {'Baseline PSNR':<14} | {'GNN PSNR':<14} | {'Gain (dB)':<10} | {'GNN SSIM':<8}")
        lines.append("-" * 78)

    for path in images:
        fname = os.path.basename(path)
        base_name = os.path.splitext(fname)[0]
        res = evaluate_image(model, cnn_model=cnn_model, image_path=path, cutoff=cutoff, device=device)
        results.append(res)

        save_visual_comparison(res, f"{dataset_name}_{base_name}", save_dir=save_dir)
        save_error_heatmap(res, f"{dataset_name}_{base_name}", save_dir=save_dir)

        gain_prefix = "+" if res['psnr_gain'] >= 0 else ""
        gain_str = f"{gain_prefix}{res['psnr_gain']:.2f} dB"

        if has_hybrid:
            h_gain = res['hybrid_gain']
            h_gain_prefix = "+" if h_gain >= 0 else ""
            h_gain_str = f"{h_gain_prefix}{h_gain:.2f} dB"
            line = f"{fname:<18} | {res['baseline_psnr']:>6.2f} dB       | {res['gnn_psnr']:>6.2f} dB     | {gain_str:>9} | {res['hybrid_psnr']:>6.2f} dB     | {h_gain_str:>10}  | {res['hybrid_ssim']:>6.4f}"
            print(f"  [{dataset_name}] {fname:<18} -> Base: {res['baseline_psnr']:.2f} dB | GNN: {res['gnn_psnr']:.2f} dB ({gain_str}) | Hybrid: {res['hybrid_psnr']:.2f} dB ({h_gain_str})")
        else:
            line = f"{fname:<18} | {res['baseline_psnr']:>6.2f} dB       | {res['gnn_psnr']:>6.2f} dB       | {gain_str:>9}  | {res['gnn_ssim']:>6.4f}"
            print(f"  [{dataset_name}] {fname:<18} -> Base: {res['baseline_psnr']:.2f} dB | GNN: {res['gnn_psnr']:.2f} dB ({gain_str})")

        lines.append(line)

    avg_base_psnr = float(np.mean([r["baseline_psnr"] for r in results]))
    avg_gnn_psnr = float(np.mean([r["gnn_psnr"] for r in results]))
    avg_gnn_gain = avg_gnn_psnr - avg_base_psnr

    if has_hybrid:
        avg_hybrid_psnr = float(np.mean([r["hybrid_psnr"] for r in results]))
        avg_hybrid_gain = avg_hybrid_psnr - avg_base_psnr
        avg_hybrid_ssim = float(np.mean([r["hybrid_ssim"] for r in results]))

        lines.append("-" * 96)
        g_pref = "+" if avg_gnn_gain >= 0 else ""
        h_pref = "+" if avg_hybrid_gain >= 0 else ""
        lines.append(f"{'AVERAGE':<18} | {avg_base_psnr:>6.2f} dB       | {avg_gnn_psnr:>6.2f} dB     | {g_pref}{avg_gnn_gain:>7.2f} dB | {avg_hybrid_psnr:>6.2f} dB     | {h_pref}{avg_hybrid_gain:>8.2f} dB | {avg_hybrid_ssim:>6.4f}")
        lines.append("=" * 96)
    else:
        avg_gnn_ssim = float(np.mean([r["gnn_ssim"] for r in results]))
        lines.append("-" * 78)
        g_pref = "+" if avg_gnn_gain >= 0 else ""
        lines.append(f"{'AVERAGE':<18} | {avg_base_psnr:>6.2f} dB       | {avg_gnn_psnr:>6.2f} dB       | {g_pref}{avg_gnn_gain:>7.2f} dB  | {avg_gnn_ssim:>6.4f}")
        lines.append("=" * 78)

    report_text = "\n".join(lines)
    print("\n" + report_text + "\n")
    return report_text


def main():
    args = parse_args()

    # Intelligent device selection (Dedicated GPU -> Integrated GPU -> CPU)
    device, device_info = get_optimal_device(preference=args.device, verbose=True)

    # Check for trained GNN checkpoint
    checkpoint_path = os.path.join("checkpoints", "model.pth")
    if not os.path.exists(checkpoint_path):
        print(f"\n[ERROR] Model checkpoint not found at: {checkpoint_path}")
        print("Please run 'python train.py' first to generate model weights.")
        return

    # Load GNN model onto optimal device
    model = FrequencyGNN(in_channels=4, hidden_channels=128, num_layers=3, dropout=0.0).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded Stage 1 FrequencyGNN weights from: {checkpoint_path} onto {device}")

    # Check for Stage 2 Spatial Refiner CNN
    cnn_model = None
    spatial_path = os.path.join("checkpoints", "spatial_refiner.pth")
    hybrid_path = os.path.join("checkpoints", "hybrid_model.pth")

    if os.path.exists(spatial_path):
        cnn_model = SpatialRefinerCNN(in_channels=1, hidden_channels=64, num_layers=4).to(device)
        cnn_model.load_state_dict(torch.load(spatial_path, map_location=device))
        cnn_model.eval()
        print(f"Loaded Stage 2 SpatialRefinerCNN weights from: {spatial_path} onto {device}")
    elif os.path.exists(hybrid_path):
        hybrid_data = torch.load(hybrid_path, map_location=device)
        if "cnn_state_dict" in hybrid_data:
            cnn_model = SpatialRefinerCNN(in_channels=1, hidden_channels=64, num_layers=4).to(device)
            cnn_model.load_state_dict(hybrid_data["cnn_state_dict"])
            cnn_model.eval()
            print(f"Loaded Stage 2 SpatialRefinerCNN from hybrid checkpoint: {hybrid_path} onto {device}")

    print()
    os.makedirs(args.save_dir, exist_ok=True)
    all_reports = []

    if args.dataset in ("Set14", "all"):
        rep14 = evaluate_dataset(model, cnn_model, "Set14", cutoff=args.cutoff, device=device, save_dir=args.save_dir)
        if rep14:
            all_reports.append(rep14)

    if args.dataset in ("Set5", "all"):
        rep5 = evaluate_dataset(model, cnn_model, "Set5", cutoff=args.cutoff, device=device, save_dir=args.save_dir)
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
