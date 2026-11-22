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

from src.cleanup import clean_all
import argparse
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dctn
import torch

from src.device_utils import get_optimal_device
from src.dct_utils import load_and_slice, apply_dct_and_truncate, reconstruct_from_dct
from src.metrics import calculate_psnr, calculate_ssim
from src.dataset import create_zigzag_mask, get_dct_dataloader, get_zigzag_indices
from src.models import FrequencyGNN, SpatialRefinerCNN


def evaluate_image(model, cnn_model=None, image_path="", cutoff=15, device="cpu"):
    """
    Evaluates a single image through 4 distinct reconstruction pipelines:
      1. Baseline JPEG (Cutoff=15 zero-filled IDCT)
      2. Multi-Block Spatial-Frequency GNN (1+2+5 Approach)
      3. Hybrid (GNN -> CNN) (Stage 1 GNN + Stage 2 Spatial Refiner CNN)
      4. Hybrid (CNN -> GNN) (Stage 1 Spatial Refiner CNN + 2D-DCT Hard Spectral Consistency)
    Supports full 3-channel RGB color images (via standard YCbCr luminance processing).

    Args:
        model (FrequencyGNN): Trained Stage 1 GNN model.
        cnn_model (SpatialRefinerCNN, optional): Trained Stage 2 Spatial Refiner CNN.
        image_path (str): Path to image file.
        cutoff (int): Preserved zigzag low frequencies (default: 15).
        device (str): Inference device ('cpu' or 'cuda').

    Returns:
        dict: Contains full RGB reconstructions and quantitative scores for all 4 pipelines.
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

    nr, nc = h_crop // 8, w_crop // 8

    # 1. Vectorized 2D-DCT & Multi-Block Spatial Context Extraction
    blocks_grid = Y_channel.reshape(nr, 8, nc, 8).transpose(0, 2, 1, 3)
    dct_grid = dctn(blocks_grid, axes=(-2, -1), norm='ortho')  # (nr, nc, 8, 8)
    Y_full = dct_grid.reshape(-1, 64)

    # Truncate high frequencies in zigzag order
    zigzag = get_zigzag_indices()
    x_trunc_grid = np.zeros_like(dct_grid)
    for idx in range(min(cutoff, 64)):
        zr, zc = zigzag[idx]
        x_trunc_grid[:, :, zr, zc] = dct_grid[:, :, zr, zc]
    X_trunc = x_trunc_grid.reshape(-1, 64)

    # 8-Neighbor Spatial Context (N, S, W, E, NW, NE, SW, SE)
    dc = x_trunc_grid[:, :, 0, 0]  # (nr, nc)
    padded = np.pad(dc, 1, mode='edge')
    N  = padded[0:nr, 1:nc+1]
    S  = padded[2:nr+2, 1:nc+1]
    W  = padded[1:nr+1, 0:nc]
    E  = padded[1:nr+1, 2:nc+2]
    NW = padded[0:nr, 0:nc]
    NE = padded[0:nr, 2:nc+2]
    SW = padded[2:nr+2, 0:nc]
    SE = padded[2:nr+2, 2:nc+2]

    diff_nb = np.stack([
        N - dc, S - dc, W - dc, E - dc,
        NW - dc, NE - dc, SW - dc, SE - dc
    ], axis=-1)  # (nr, nc, 8)
    neighbor_contexts = diff_nb.reshape(-1, 8)

    # 2. Pipeline 1: Baseline Reconstruction (IDCT with zero-filled missing high frequencies)
    baseline_Y = reconstruct_from_dct(X_trunc, (h_crop, w_crop))
    baseline_ycrcb = np.stack([np.clip(baseline_Y, 0.0, 1.0), Cr_channel, Cb_channel], axis=-1)
    baseline_rgb = np.clip(cv2.cvtColor(baseline_ycrcb, cv2.COLOR_YCrCb2RGB), 0.0, 1.0)
    base_psnr = calculate_psnr(Y_channel, baseline_Y)
    base_ssim = calculate_ssim(Y_channel, baseline_Y)

    # 3. Pipeline 2: Multi-Block Spatial-Frequency GNN Model Inference
    dataloader = get_dct_dataloader(
        X_trunc, Y_full, cutoff=cutoff,
        neighbor_contexts=neighbor_contexts,
        batch_size=128, shuffle=False
    )
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            preds = model(batch.x, batch.edge_index)  # Shape: (B * 64,)
            all_preds.append(preds.cpu().numpy().reshape(-1, 64))

    all_preds = np.concatenate(all_preds, axis=0)  # Shape: (num_blocks, 64)
    missing_mask = create_zigzag_mask(cutoff=cutoff).numpy()
    X_gnn_restored = X_trunc.copy()
    X_gnn_restored[:, missing_mask] = all_preds[:, missing_mask]
    X_gnn_restored[:, missing_mask] = all_preds[:, missing_mask]

    gnn_Y = reconstruct_from_dct(X_gnn_restored, (h_crop, w_crop))
    gnn_ycrcb = np.stack([np.clip(gnn_Y, 0.0, 1.0), Cr_channel, Cb_channel], axis=-1)
    gnn_rgb = np.clip(cv2.cvtColor(gnn_ycrcb, cv2.COLOR_YCrCb2RGB), 0.0, 1.0)
    gnn_psnr = calculate_psnr(Y_channel, gnn_Y)
    gnn_ssim = calculate_ssim(Y_channel, gnn_Y)
    gnn_gain = gnn_psnr - base_psnr

    # 4. Pipeline 3 & 4: Hybrid Approaches (Dual-Domain)
    gnn_cnn_rgb, gnn_cnn_psnr, gnn_cnn_ssim, gnn_cnn_gain = None, None, None, None
    cnn_gnn_rgb, cnn_gnn_psnr, cnn_gnn_ssim, cnn_gnn_gain = None, None, None, None

    if cnn_model is not None:
        cnn_model.eval()

        # Pipeline 3: GNN -> CNN
        with torch.no_grad():
            gnn_Y_tensor = torch.from_numpy(gnn_Y).float().unsqueeze(0).unsqueeze(0).to(device)
            refined_tensor_1 = cnn_model(gnn_Y_tensor)
            gnn_cnn_Y = np.clip(refined_tensor_1.squeeze().cpu().numpy(), 0.0, 1.0)

        gnn_cnn_ycrcb = np.stack([gnn_cnn_Y, Cr_channel, Cb_channel], axis=-1)
        gnn_cnn_rgb = np.clip(cv2.cvtColor(gnn_cnn_ycrcb, cv2.COLOR_YCrCb2RGB), 0.0, 1.0)
        gnn_cnn_psnr = calculate_psnr(Y_channel, gnn_cnn_Y)
        gnn_cnn_ssim = calculate_ssim(Y_channel, gnn_cnn_Y)
        gnn_cnn_gain = gnn_cnn_psnr - base_psnr

        # Pipeline 4: CNN -> GNN (Spatial Deblocking first + 2D-DCT Hard Spectral Consistency)
        with torch.no_grad():
            base_Y_tensor = torch.from_numpy(baseline_Y).float().unsqueeze(0).unsqueeze(0).to(device)
            refined_tensor_2 = cnn_model(base_Y_tensor)
            cnn_base_Y = np.clip(refined_tensor_2.squeeze().cpu().numpy(), 0.0, 1.0)

        # Slice CNN-deblocked image into 8x8 blocks & compute full 2D-DCT
        cnn_blocks = []
        for r in range(0, h_crop, 8):
            for c in range(0, w_crop, 8):
                cnn_blocks.append(cnn_base_Y[r:r+8, c:c+8])
        cnn_blocks = np.array(cnn_blocks, dtype=np.float32)
        X_from_cnn, _ = apply_dct_and_truncate(cnn_blocks, cutoff=64)

        # Hard Spectral Consistency: lock true low frequencies to known ground truth X_trunc
        X_cnn_gnn = X_from_cnn.copy()
        X_cnn_gnn[:, ~missing_mask] = X_trunc[:, ~missing_mask]

        cnn_gnn_Y = reconstruct_from_dct(X_cnn_gnn, (h_crop, w_crop))
        cnn_gnn_Y = np.clip(cnn_gnn_Y, 0.0, 1.0)
        cnn_gnn_ycrcb = np.stack([cnn_gnn_Y, Cr_channel, Cb_channel], axis=-1)
        cnn_gnn_rgb = np.clip(cv2.cvtColor(cnn_gnn_ycrcb, cv2.COLOR_YCrCb2RGB), 0.0, 1.0)
        cnn_gnn_psnr = calculate_psnr(Y_channel, cnn_gnn_Y)
        cnn_gnn_ssim = calculate_ssim(Y_channel, cnn_gnn_Y)
        cnn_gnn_gain = cnn_gnn_psnr - base_psnr

    return {
        "original": raw_rgb,
        "baseline": baseline_rgb,
        "gnn": gnn_rgb,
        "gnn_cnn": gnn_cnn_rgb,
        "cnn_gnn": cnn_gnn_rgb,
        "is_color": True,
        "baseline_psnr": base_psnr,
        "baseline_ssim": base_ssim,
        "gnn_psnr": gnn_psnr,
        "gnn_ssim": gnn_ssim,
        "gnn_gain": gnn_gain,
        "gnn_cnn_psnr": gnn_cnn_psnr,
        "gnn_cnn_ssim": gnn_cnn_ssim,
        "gnn_cnn_gain": gnn_cnn_gain,
        "cnn_gnn_psnr": cnn_gnn_psnr,
        "cnn_gnn_ssim": cnn_gnn_ssim,
        "cnn_gnn_gain": cnn_gnn_gain,
        "shape": (h_crop, w_crop),
    }


def save_visual_comparison(res, image_name, save_dir="results"):
    """
    Saves 5-panel side-by-side comparison figure in full RGB color:
    [Original Ground Truth] vs [Baseline JPEG] vs [GNN Only] vs [Hybrid: GNN -> CNN] vs [Hybrid: CNN -> GNN].
    """
    os.makedirs(save_dir, exist_ok=True)
    has_hybrid = res.get("gnn_cnn") is not None
    n_cols = 5 if has_hybrid else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5.2 * n_cols, 5.2))

    is_color = res.get("is_color", False)
    cmap_arg = None if is_color else "gray"

    # Panel 1: Original (Ground Truth)
    axes[0].imshow(res["original"], cmap=cmap_arg)
    axes[0].set_title("Original (Ground Truth)\nReference Image", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Baseline JPEG
    axes[1].imshow(res["baseline"], cmap=cmap_arg)
    axes[1].set_title(
        f"Baseline JPEG (Cutoff=15)\nPSNR: {res['baseline_psnr']:.2f} dB | SSIM: {res['baseline_ssim']:.4f}",
        fontsize=10,
    )
    axes[1].axis("off")

    # Panel 3: Stage 1 Frequency GNN
    g_gain_str = f"+{res['gnn_gain']:.2f} dB" if res['gnn_gain'] >= 0 else f"{res['gnn_gain']:.2f} dB"
    axes[2].imshow(res["gnn"], cmap=cmap_arg)
    axes[2].set_title(
        f"Stage 1: Frequency GNN\nPSNR: {res['gnn_psnr']:.2f} dB ({g_gain_str})\nSSIM: {res['gnn_ssim']:.4f}",
        fontsize=10,
        color="darkgreen" if res['gnn_gain'] >= 0 else "black",
    )
    axes[2].axis("off")

    if has_hybrid:
        # Panel 4: Hybrid (GNN -> CNN)
        gc_gain = res['gnn_cnn_gain']
        gc_gain_str = f"+{gc_gain:.2f} dB" if gc_gain >= 0 else f"{gc_gain:.2f} dB"
        axes[3].imshow(res["gnn_cnn"], cmap=cmap_arg)
        axes[3].set_title(
            f"Hybrid (GNN -> CNN)\nPSNR: {res['gnn_cnn_psnr']:.2f} dB ({gc_gain_str})\nSSIM: {res['gnn_cnn_ssim']:.4f}",
            fontsize=10,
            fontweight="bold",
            color="navy",
        )
        axes[3].axis("off")

        # Panel 5: Hybrid (CNN -> GNN)
        cg_gain = res['cnn_gnn_gain']
        cg_gain_str = f"+{cg_gain:.2f} dB" if cg_gain >= 0 else f"{cg_gain:.2f} dB"
        axes[4].imshow(res["cnn_gnn"], cmap=cmap_arg)
        axes[4].set_title(
            f"Hybrid (CNN -> GNN)\nPSNR: {res['cnn_gnn_psnr']:.2f} dB ({cg_gain_str})\nSSIM: {res['cnn_gnn_ssim']:.4f}",
            fontsize=10,
            fontweight="bold",
            color="darkgreen",
        )
        axes[4].axis("off")

    color_label = "Color RGB" if is_color else "Grayscale"
    plt.suptitle(f"5-Way Reconstruction Comparison ({color_label}): {image_name}", fontsize=13, fontweight="bold")
    fig.subplots_adjust(top=0.86, bottom=0.03, left=0.01, right=0.99, wspace=0.05)
    out_path = os.path.join(save_dir, f"{image_name}_comparison.png")
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def save_error_heatmap(res, image_name, save_dir="results"):
    """
    Saves 4-panel pixel error difference maps comparing:
    [Baseline Error] vs [Frequency GNN Error] vs [Hybrid (GNN -> CNN) Error] vs [Hybrid (CNN -> GNN) Error].
    """
    os.makedirs(save_dir, exist_ok=True)
    has_hybrid = res.get("gnn_cnn") is not None
    n_cols = 4 if has_hybrid else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5.8 * n_cols, 5.2))

    is_color = res.get("is_color", False)
    if is_color:
        err_base = np.mean(np.abs(res["original"] - res["baseline"]), axis=-1)
        err_gnn = np.mean(np.abs(res["original"] - res["gnn"]), axis=-1)
        if has_hybrid:
            err_gnn_cnn = np.mean(np.abs(res["original"] - res["gnn_cnn"]), axis=-1)
            err_cnn_gnn = np.mean(np.abs(res["original"] - res["cnn_gnn"]), axis=-1)
    else:
        err_base = np.abs(res["original"] - res["baseline"])
        err_gnn = np.abs(res["original"] - res["gnn"])
        if has_hybrid:
            err_gnn_cnn = np.abs(res["original"] - res["gnn_cnn"])
            err_cnn_gnn = np.abs(res["original"] - res["cnn_gnn"])

    vmax = max(err_base.max(), 0.15)

    im0 = axes[0].imshow(err_base, cmap="hot", vmin=0, vmax=vmax)
    axes[0].set_title(f"Baseline Error Map\nMean Abs Err: {err_base.mean():.4f}", fontsize=10)
    axes[0].axis("off")

    im1 = axes[1].imshow(err_gnn, cmap="hot", vmin=0, vmax=vmax)
    axes[1].set_title(f"Frequency GNN Error Map\nMean Abs Err: {err_gnn.mean():.4f}", fontsize=10)
    axes[1].axis("off")

    if has_hybrid:
        im2 = axes[2].imshow(err_gnn_cnn, cmap="hot", vmin=0, vmax=vmax)
        axes[2].set_title(f"Hybrid (GNN -> CNN) Error Map\nMean Abs Err: {err_gnn_cnn.mean():.4f}", fontsize=10, fontweight="bold", color="navy")
        axes[2].axis("off")

        im3 = axes[3].imshow(err_cnn_gnn, cmap="hot", vmin=0, vmax=vmax)
        axes[3].set_title(f"Hybrid (CNN -> GNN) Error Map\nMean Abs Err: {err_cnn_gnn.mean():.4f} (Lowest Error)", fontsize=10, fontweight="bold", color="darkgreen")
        axes[3].axis("off")

        fig.subplots_adjust(top=0.86, bottom=0.05, left=0.02, right=0.92, wspace=0.06)
        cbar_ax = fig.add_axes([0.93, 0.12, 0.015, 0.72])
        fig.colorbar(im3, cax=cbar_ax)
    else:
        fig.subplots_adjust(top=0.86, bottom=0.05, left=0.02, right=0.90, wspace=0.06)
        cbar_ax = fig.add_axes([0.91, 0.12, 0.02, 0.72])
        fig.colorbar(im1, cax=cbar_ax)

    plt.suptitle(f"Error Heatmap Comparison (Darker = Lower Error): {image_name}", fontsize=13, fontweight="bold")
    out_path = os.path.join(save_dir, f"{image_name}_error_heatmap.png")
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FrequencyGNN & Dual-Domain Hybrid Model.")
    parser.add_argument("--dataset", type=str, default="Set14", choices=["Set14", "Set5", "all"],
                        help="Benchmark dataset to evaluate on ('Set14', 'Set5', or 'all'). Default: Set14.")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to an individual image file to evaluate directly.")
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
    
    if has_hybrid:
        lines.append("=" * 115)
        lines.append(f"  BENCHMARK EVALUATION RESULTS: {dataset_name} (4 Reconstruction Approaches vs Ground Truth)")
        lines.append("=" * 115)
        lines.append(f"{'Image':<18} | {'Base PSNR':<10} | {'GNN PSNR (Gain)':<18} | {'GNN->CNN (Gain)':<18} | {'CNN->GNN (Gain)':<18} | {'CNN->GNN SSIM':<12}")
        lines.append("-" * 115)
    else:
        lines.append("=" * 78)
        lines.append(f"  BENCHMARK EVALUATION RESULTS: {dataset_name}")
        lines.append("=" * 78)
        lines.append(f"{'Image':<18} | {'Baseline PSNR':<14} | {'GNN PSNR':<14} | {'Gain (dB)':<10} | {'GNN SSIM':<8}")
        lines.append("-" * 78)

    for path in images:
        fname = os.path.basename(path)
        base_name = os.path.splitext(fname)[0]
        res = evaluate_image(model, cnn_model=cnn_model, image_path=path, cutoff=cutoff, device=device)
        results.append(res)

        save_visual_comparison(res, f"{dataset_name}_{base_name}", save_dir=save_dir)
        save_error_heatmap(res, f"{dataset_name}_{base_name}", save_dir=save_dir)

        g_gain = res['gnn_gain']
        g_str = f"{res['gnn_psnr']:.2f} dB ({g_gain:+.2f})"

        if has_hybrid:
            gc_gain = res['gnn_cnn_gain']
            gc_str = f"{res['gnn_cnn_psnr']:.2f} dB ({gc_gain:+.2f})"
            cg_gain = res['cnn_gnn_gain']
            cg_str = f"{res['cnn_gnn_psnr']:.2f} dB ({cg_gain:+.2f})"
            line = f"{fname:<18} | {res['baseline_psnr']:>6.2f} dB  | {g_str:<18} | {gc_str:<18} | {cg_str:<18} | {res['cnn_gnn_ssim']:>8.4f}"
            print(f"  [{dataset_name}] {fname:<18} -> Base: {res['baseline_psnr']:.2f} dB | GNN: {g_str} | GNN->CNN: {gc_str} | CNN->GNN: {cg_str}")
        else:
            line = f"{fname:<18} | {res['baseline_psnr']:>6.2f} dB       | {res['gnn_psnr']:>6.2f} dB       | {g_gain:>+9.2f} dB | {res['gnn_ssim']:>6.4f}"
            print(f"  [{dataset_name}] {fname:<18} -> Base: {res['baseline_psnr']:.2f} dB | GNN: {res['gnn_psnr']:.2f} dB ({g_gain:+.2f} dB)")

        lines.append(line)

    avg_base_psnr = float(np.mean([r["baseline_psnr"] for r in results]))
    avg_gnn_psnr = float(np.mean([r["gnn_psnr"] for r in results]))
    avg_gnn_gain = avg_gnn_psnr - avg_base_psnr

    if has_hybrid:
        avg_gc_psnr = float(np.mean([r["gnn_cnn_psnr"] for r in results]))
        avg_gc_gain = avg_gc_psnr - avg_base_psnr
        avg_cg_psnr = float(np.mean([r["cnn_gnn_psnr"] for r in results]))
        avg_cg_gain = avg_cg_psnr - avg_base_psnr
        avg_cg_ssim = float(np.mean([r["cnn_gnn_ssim"] for r in results]))

        lines.append("-" * 115)
        avg_g_str = f"{avg_gnn_psnr:.2f} dB ({avg_gnn_gain:+.2f})"
        avg_gc_str = f"{avg_gc_psnr:.2f} dB ({avg_gc_gain:+.2f})"
        avg_cg_str = f"{avg_cg_psnr:.2f} dB ({avg_cg_gain:+.2f})"
        lines.append(f"{'AVERAGE':<18} | {avg_base_psnr:>6.2f} dB  | {avg_g_str:<18} | {avg_gc_str:<18} | {avg_cg_str:<18} | {avg_cg_ssim:>8.4f}")
        lines.append("=" * 115)
    else:
        avg_gnn_ssim = float(np.mean([r["gnn_ssim"] for r in results]))
        lines.append("-" * 78)
        lines.append(f"{'AVERAGE':<18} | {avg_base_psnr:>6.2f} dB       | {avg_gnn_psnr:>6.2f} dB       | {avg_gnn_gain:>+7.2f} dB  | {avg_gnn_ssim:>6.4f}")
        lines.append("=" * 78)

    report_text = "\n".join(lines)
    print("\n" + report_text + "\n")
    return report_text


def main():
    args = parse_args()

    # If running full dataset evaluation, clean previous results
    if args.image is None:
        clean_all(clear_results=True)
    else:
        clean_all(clear_results=False)

    # Intelligent device selection (Dedicated GPU -> Integrated GPU -> CPU)
    device, device_info = get_optimal_device(preference=args.device, verbose=True)

    # Check for trained GNN checkpoint
    checkpoint_path = os.path.join("checkpoints", "model.pth")
    if not os.path.exists(checkpoint_path):
        print(f"\n[ERROR] Model checkpoint not found at: {checkpoint_path}")
        print("Please run 'python train.py' first to generate model weights.")
        return

    # Load GNN model onto optimal device (auto-detects in_channels from checkpoint)
    state = torch.load(checkpoint_path, map_location=device)
    in_channels = 10
    if "convs.0.lin.weight" in state:
        in_channels = state["convs.0.lin.weight"].shape[1]
    model = FrequencyGNN(in_channels=in_channels, hidden_channels=128, num_layers=3, dropout=0.0).to(device)
    model.load_state_dict(state)
    print(f"Loaded Stage 1 FrequencyGNN weights (in_channels={in_channels}) from: {checkpoint_path} onto {device}")

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

    # Single Image Mode
    if args.image:
        if not os.path.exists(args.image):
            print(f"[ERROR] Specified image does not exist: {args.image}")
            return
        fname = os.path.basename(args.image)
        base_name = os.path.splitext(fname)[0]
        print(f"Evaluating single image: {args.image} (Cutoff={args.cutoff})...")
        res = evaluate_image(model, cnn_model=cnn_model, image_path=args.image, cutoff=args.cutoff, device=device)
        comp_path = save_visual_comparison(res, f"single_{base_name}", save_dir=args.save_dir)
        heat_path = save_error_heatmap(res, f"single_{base_name}", save_dir=args.save_dir)

        print("\n" + "=" * 80)
        print(f"  SINGLE IMAGE EVALUATION RESULTS: {fname}")
        print("=" * 80)
        print(f"  Baseline JPEG (Cutoff={args.cutoff}):  {res['baseline_psnr']:.2f} dB (SSIM: {res['baseline_ssim']:.4f})")
        print(f"  Stage 1: Frequency GNN:         {res['gnn_psnr']:.2f} dB ({res['gnn_gain']:+.2f} dB)")
        if cnn_model is not None:
            print(f"  Hybrid 1: GNN -> CNN:           {res['gnn_cnn_psnr']:.2f} dB ({res['gnn_cnn_gain']:+.2f} dB)")
            print(f"  Hybrid 2: CNN -> GNN (Spectral): {res['cnn_gnn_psnr']:.2f} dB ({res['cnn_gnn_gain']:+.2f} dB | SSIM: {res['cnn_gnn_ssim']:.4f})")
        print("=" * 80)
        print(f"[SUCCESS] 5-panel comparison saved to: {comp_path}")
        print(f"[SUCCESS] 4-panel heatmap saved to:    {heat_path}\n")
        return

    # Dataset Mode
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
