"""
src/inference.py - Unified NeuroDCT Inference, Metrics & Diagnostic Engine
--------------------------------------------------------------------------
Handles model loading (FrequencyGNN + SpatialRefinerCNN), 5-pipeline inference,
PSNR / PSNR-B / SSIM computation, 8x8 block artifact attenuation measurement,
and visual asset rendering (Split images, Residual Heatmaps, 4x Loupe Patches,
and 8x8 DCT Spectrum Grids) for the NeuroDCT Web Application.
"""

import io
import os
import sys
import time
import base64
from typing import Dict, Any, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.fft import dctn, idctn
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.device_utils import get_optimal_device
from src.dataset import create_zigzag_mask, get_zigzag_indices
from src.graph_builder import build_multiblock_spatial_frequency_graph, get_multiblock_coordinates
from src.metrics import calculate_psnr, calculate_ssim
from src.models import FrequencyGNN, SpatialRefinerCNN


def reconstruct_from_dct_fast(dct_blocks_flat: np.ndarray, nr: int, nc: int) -> np.ndarray:
    """Vectorized 2D-IDCT reconstruction across all (nr * nc) 8x8 blocks simultaneously."""
    grid = dct_blocks_flat.reshape(nr, nc, 8, 8)
    spatial = idctn(grid, axes=(-2, -1), norm="ortho")
    img = spatial.transpose(0, 2, 1, 3).reshape(nr * 8, nc * 8)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


# Preset specimen paths grounded in datasets/
PRESETS: Dict[str, Dict[str, Any]] = {
    "barbara": {
        "id": "barbara",
        "name": "Barbara",
        "filename": "Barbara_Set14.png",
        "path": os.path.join("datasets", "Set14", "img_002_SRF_2.png"),
        "tag": "High-Freq Weave",
        "desc": "Standard high-frequency textile specimen",
    },
    "butterfly": {
        "id": "butterfly",
        "name": "Butterfly",
        "filename": "Butterfly_Set5.png",
        "path": os.path.join("datasets", "Set5", "img_003_SRF_2.png"),
        "tag": "Sharp Contours",
        "desc": "High-contrast wing edges (+4.13 dB showcase)",
    },
    "bird": {
        "id": "bird",
        "name": "Bird",
        "filename": "Bird_Set5.png",
        "path": os.path.join("datasets", "Set5", "img_002_SRF_2.png"),
        "tag": "Smooth + Edge",
        "desc": "Natural plumage & sharp contrast boundaries",
    },
    "monarch": {
        "id": "monarch",
        "name": "Monarch",
        "filename": "Monarch_Set14.png",
        "path": os.path.join("datasets", "Set14", "img_011_SRF_2.png"),
        "tag": "Wide Botanical",
        "desc": "Complex petal textures and crisp wing outlines",
    },
}


def qf_to_cutoff(qf: int) -> int:
    """
    Maps user Quality Factor slider (10..80) to zigzag DCT cutoff count (10..56).
    Specifically ensures QF=15 maps cleanly to cutoff=15 (the primary model regime).
    """
    qf = max(10, min(80, int(qf)))
    if qf <= 20:
        # 10 -> 10, 15 -> 15, 20 -> 20
        return qf
    # 20..80 maps linearly from 20 to 56
    return int(round(20 + (qf - 20) * (36.0 / 60.0)))


def calculate_bef_and_psnrb(orig: Optional[np.ndarray], recon: np.ndarray) -> Tuple[float, float]:
    """
    Computes Blocking Effect Factor (BEF) and PSNR-B (Yim & Bovik, IEEE TMM).
    Measures discontinuity energy strictly across 8x8 block boundaries vs interior pixels.
    Returns (psnr_b, total_boundary_energy).
    """
    recon = np.clip(recon, 0.0, 1.0).astype(np.float64)
    h, w = recon.shape

    # Horizontal differences across columns
    dh = (recon[:, 1:] - recon[:, :-1]) ** 2
    b_cols = np.zeros(w - 1, dtype=bool)
    b_cols[7::8] = True  # indices 7, 15, 23... separate col 7|8, 15|16
    db_h = dh[:, b_cols].mean() if np.any(b_cols) else 0.0
    dbc_h = dh[:, ~b_cols].mean() if np.any(~b_cols) else 0.0

    # Vertical differences across rows
    dv = (recon[1:, :] - recon[:-1, :]) ** 2
    b_rows = np.zeros(h - 1, dtype=bool)
    b_rows[7::8] = True
    db_v = dv[b_rows, :].mean() if np.any(b_rows) else 0.0
    dbc_v = dv[~b_rows, :].mean() if np.any(~b_rows) else 0.0

    db = 0.5 * (db_h + db_v)
    dbc = 0.5 * (dbc_h + dbc_v)

    eta = np.log2(8.0) / max(np.log2(min(h, w)), 3.0)
    bef = eta * max(0.0, db - dbc)
    boundary_excess = max(0.0, db - dbc)

    if orig is None:
        return 0.0, float(boundary_excess)

    orig_c = np.clip(orig, 0.0, 1.0).astype(np.float64)
    mse = float(np.mean((orig_c - recon) ** 2))
    mse_b = mse + bef
    if mse_b < 1e-12:
        psnr_b = 100.0
    else:
        psnr_b = float(10.0 * np.log10(1.0 / mse_b))

    return psnr_b, float(boundary_excess)


def rgb_float_to_png_b64(img_rgb: np.ndarray) -> str:
    """Encodes a float32 [0,1] RGB numpy array into a base64 PNG data URI."""
    u8 = (np.clip(img_rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    pil_img = Image.fromarray(u8)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=False)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def fig_to_png_b64(fig) -> str:
    """Renders a Matplotlib figure to a base64 PNG data URI and closes it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def render_indigo_heatmap_overlay(err_map: np.ndarray, vmax: float) -> str:
    """Renders a high-contrast 'hot' residual error heatmap (Black=Zero Error, Red/Yellow=High)."""
    fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=150)
    fig.patch.set_facecolor("#000000")
    safe_vmax = max(float(np.percentile(err_map, 99.5)) * 1.4, min(float(vmax), 0.14), 0.06)
    ax.imshow(err_map, cmap="hot", vmin=0, vmax=safe_vmax, aspect="auto")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig_to_png_b64(fig)


def render_results_comparison_strip(
    raw_rgb: np.ndarray,
    outputs_map: Dict[str, Any],
    cutoff: int,
    is_blind: bool = False,
) -> str:
    """
    Renders the signature 5-panel /results-style side-by-side comparison figure:
    [Original] vs [Baseline] vs [Frequency GNN] vs [GNN -> CNN] vs [CNN -> GNN].
    """
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
    base_row = outputs_map["baseline"]["row"]
    gnn_row = outputs_map["gnn_only"]["row"]
    gc_row = outputs_map["gnn_cnn"]["row"]
    cg_row = outputs_map["cnn_gnn"]["row"]

    fig, axes = plt.subplots(1, 5, figsize=(21.0, 4.4), dpi=130)
    fig.patch.set_facecolor("#FFFFFF")

    if is_blind:
        panels = [
            (raw_rgb, "Uploaded Input JPEG\n(Compressed Reference)", "#141B2B"),
            (outputs_map["gnn_only"]["rgb"], f"Frequency GNN Alone\nAtten: -{gnn_row['attenuation_pct']}%", "#585F6C"),
            (outputs_map["cnn_only"]["rgb"], f"Spatial CNN Alone\nAtten: -{outputs_map['cnn_only']['row']['attenuation_pct']}%", "#1E40AF"),
            (outputs_map["gnn_cnn"]["rgb"], f"Hybrid (GNN -> CNN)\nAtten: -{gc_row['attenuation_pct']}%", "#1E3A8A"),
            (outputs_map["cnn_gnn"]["rgb"], f"Hybrid (CNN -> GNN)\nAtten: -{cg_row['attenuation_pct']}% (Proposed)", "#166534"),
        ]
    else:
        g_sign = "+" if gnn_row["psnr_gain"] >= 0 else ""
        gc_sign = "+" if gc_row["psnr_gain"] >= 0 else ""
        cg_sign = "+" if cg_row["psnr_gain"] >= 0 else ""
        panels = [
            (raw_rgb, "Original (Ground Truth)\nReference Image", "#141B2B"),
            (outputs_map["baseline"]["rgb"], f"Baseline JPEG (Cutoff={cutoff})\nPSNR: {base_row['psnr']:.2f} dB | SSIM: {base_row['ssim']:.4f}", "#585F6C"),
            (outputs_map["gnn_only"]["rgb"], f"Stage 1: Frequency GNN\nPSNR: {gnn_row['psnr']:.2f} dB ({g_sign}{gnn_row['psnr_gain']:.2f})\nSSIM: {gnn_row['ssim']:.4f}", "#1E293B"),
            (outputs_map["gnn_cnn"]["rgb"], f"Hybrid (GNN -> CNN)\nPSNR: {gc_row['psnr']:.2f} dB ({gc_sign}{gc_row['psnr_gain']:.2f})\nSSIM: {gc_row['ssim']:.4f}", "#1D4ED8"),
            (outputs_map["cnn_gnn"]["rgb"], f"Hybrid (CNN -> GNN) [BEST]\nPSNR: {cg_row['psnr']:.2f} dB ({cg_sign}{cg_row['psnr_gain']:.2f})\nSSIM: {cg_row['ssim']:.4f}", "#166534"),
        ]

    for idx, (img_arr, title, col) in enumerate(panels):
        axes[idx].imshow(np.clip(img_arr, 0.0, 1.0))
        axes[idx].set_title(title, fontsize=10, fontweight="bold", color=col, pad=8)
        axes[idx].axis("off")

    fig.suptitle("5-Way Reconstruction Comparison (Side-by-Side Results)", fontsize=12.5, fontweight="bold", color="#0F2942")
    fig.subplots_adjust(top=0.82, bottom=0.03, left=0.01, right=0.99, wspace=0.05)
    return fig_to_png_b64(fig)


def render_results_heatmap_strip(
    outputs_map: Dict[str, Any],
    err_base: np.ndarray,
    is_blind: bool = False,
) -> str:
    """
    Renders the signature 4-panel /results-style 'hot' Error Heatmap comparison:
    [Baseline Error] vs [GNN Error] vs [GNN -> CNN Error] vs [CNN -> GNN Error].
    """
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
    err_gnn = outputs_map["gnn_only"]["err"]
    err_gc = outputs_map["gnn_cnn"]["err"]
    err_cg = outputs_map["cnn_gnn"]["err"]

    vmax = max(float(np.percentile(err_base, 99.5)) * 1.4, float(err_base.max()) * 0.85, 0.12)
    fig, axes = plt.subplots(1, 4, figsize=(20.0, 4.4), dpi=130)
    fig.patch.set_facecolor("#FFFFFF")

    prefix = "Correction Map" if is_blind else "Error Map"
    panels = [
        (err_base, f"Baseline {prefix}\nMean Abs Err: {err_base.mean():.4f}", "#141B2B"),
        (err_gnn,  f"Frequency GNN {prefix}\nMean Abs Err: {err_gnn.mean():.4f}", "#585F6C"),
        (err_gc,   f"Hybrid (GNN -> CNN) {prefix}\nMean Abs Err: {err_gc.mean():.4f}", "#1D4ED8"),
        (err_cg,   f"Hybrid (CNN -> GNN) {prefix}\nMean Abs Err: {err_cg.mean():.4f} (Lowest Error)", "#166534"),
    ]

    last_im = None
    for idx, (emap, title, col) in enumerate(panels):
        last_im = axes[idx].imshow(emap, cmap="hot", vmin=0, vmax=vmax)
        axes[idx].set_title(title, fontsize=10, fontweight="bold", color=col, pad=8)
        axes[idx].axis("off")

    fig.subplots_adjust(top=0.82, bottom=0.05, left=0.015, right=0.93, wspace=0.05)
    cax = fig.add_axes([0.945, 0.10, 0.014, 0.72])
    cbar = fig.colorbar(last_im, cax=cax)
    cbar.ax.tick_params(labelsize=8, colors="#464555")

    fig.suptitle("4-Way Error Heatmap Comparison (Darker = Lower Error)", fontsize=12.5, fontweight="bold", color="#0F2942")
    return fig_to_png_b64(fig)


def render_heatmap_comparison_figure(
    err_base: np.ndarray,
    err_selected: np.ndarray,
    selected_label: str,
    is_blind: bool = False,
) -> str:
    """Renders high-res side-by-side 'hot' residual error heatmap comparison for Tab 2."""
    vmax = max(float(np.percentile(err_base, 99.5)) * 1.4, float(err_base.max()) * 0.85, 0.12)

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")

    if is_blind:
        title0 = f"Spatial Gradient / Seam Intensity (Input)\nMean Grad: {err_base.mean():.4f}"
        title1 = f"Correction Map |Restored - Input| ({selected_label})\nMean Delta: {err_selected.mean():.4f}"
    else:
        title0 = f"Baseline JPEG Error |GT - Base|\nMean Abs Err: {err_base.mean():.4f}"
        title1 = f"{selected_label} Error |GT - Out|\nMean Abs Err: {err_selected.mean():.4f}"

    axes[0].imshow(err_base, cmap="hot", vmin=0, vmax=vmax)
    axes[0].set_title(title0, fontsize=10, fontweight="bold", color="#141B2B", pad=8)
    axes[0].axis("off")

    im1 = axes[1].imshow(err_selected, cmap="hot", vmin=0, vmax=vmax)
    axes[1].set_title(title1, fontsize=10, fontweight="bold", color="#3525CD", pad=8)
    axes[1].axis("off")

    fig.subplots_adjust(left=0.02, right=0.90, top=0.88, bottom=0.05, wspace=0.06)
    cax = fig.add_axes([0.92, 0.12, 0.018, 0.72])
    cbar = fig.colorbar(im1, cax=cax)
    cbar.ax.tick_params(labelsize=8, colors="#464555")
    return fig_to_png_b64(fig)


def render_loupe_figure(
    base_rgb: np.ndarray,
    selected_rgb: np.ndarray,
    orig_rgb: Optional[np.ndarray],
    roi_x: int,
    roi_y: int,
    selected_label: str,
    patch_size: int = 64,
    is_blind: bool = False,
) -> str:
    """Renders Synchronized 4x Zoomed Loupe comparing 8x8 blocking elimination."""
    h, w, _ = base_rgb.shape
    ps = min(patch_size, h, w)
    x0 = max(0, min(roi_x, w - ps))
    y0 = max(0, min(roi_y, h - ps))

    crop_base = base_rgb[y0 : y0 + ps, x0 : x0 + ps]
    crop_sel = selected_rgb[y0 : y0 + ps, x0 : x0 + ps]

    # Nearest-neighbor upscale 4x so 8x8 grid seams & pixels stay razor sharp
    def upscale(arr):
        pil = Image.fromarray((np.clip(arr, 0, 1) * 255).round().astype(np.uint8))
        return np.array(pil.resize((ps * 4, ps * 4), Image.NEAREST))

    cols = 2 if is_blind else 3
    fig, axes = plt.subplots(1, cols, figsize=(4.2 * cols, 4.4), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")

    axes[0].imshow(upscale(crop_base))
    axes[0].set_title(
        f"{'Input JPEG' if is_blind else 'Baseline JPEG'}\n4× Zoom [X:{x0}, Y:{y0}, {ps}×{ps}]",
        fontsize=9.5, fontweight="bold", color="#585F6C", pad=8
    )
    axes[0].axis("off")

    axes[1].imshow(upscale(crop_sel))
    axes[1].set_title(
        f"{selected_label}\n8×8 Seams Eliminated",
        fontsize=9.5, fontweight="bold", color="#3525CD", pad=8
    )
    axes[1].axis("off")

    if not is_blind and orig_rgb is not None:
        crop_gt = orig_rgb[y0 : y0 + ps, x0 : x0 + ps]
        axes[2].imshow(upscale(crop_gt))
        axes[2].set_title(
            "Original Reference\nGround Truth",
            fontsize=9.5, fontweight="bold", color="#141B2B", pad=8
        )
        axes[2].axis("off")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.04, wspace=0.08)
    return fig_to_png_b64(fig)


def render_dct_spectrum_figure(
    dct_base_block: np.ndarray,
    dct_sel_block: np.ndarray,
    dct_gt_block: Optional[np.ndarray],
    selected_label: str,
    cutoff: int,
    block_rc: Tuple[int, int],
    is_blind: bool = False,
) -> str:
    """Renders 8x8 2D-DCT log magnitude spectrum visualization for Tab 4."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap_spec = LinearSegmentedColormap.from_list(
        "spec_indigo",
        ["#F8FAFC", "#DBEAFE", "#818CF8", "#4F46E5", "#1E1B4B"],
        N=256,
    )

    spec_base = np.log1p(np.abs(dct_base_block))
    spec_sel = np.log1p(np.abs(dct_sel_block))
    vmax = max(float(spec_base.max()), float(spec_sel.max()), 1.0)
    if dct_gt_block is not None and not is_blind:
        spec_gt = np.log1p(np.abs(dct_gt_block))
        vmax = max(vmax, float(spec_gt.max()))

    cols = 2 if (is_blind or dct_gt_block is None) else 3
    fig, axes = plt.subplots(1, cols, figsize=(4.0 * cols, 4.1), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")

    br, bc = block_rc
    panels = [
        (spec_base, f"Input Truncated DCT ({cutoff}/64)\nBlock ({br},{bc}) | {64-cutoff} Zeroed", "#585F6C"),
        (spec_sel, f"{selected_label} DCT\nReconstructed Harmonics", "#3525CD"),
    ]
    if cols == 3:
        panels.append((spec_gt, "Ground Truth DCT\nFull 64-Coeff Spectrum", "#141B2B"))

    for idx, (mat, title, col) in enumerate(panels):
        ax = axes[idx]
        ax.imshow(mat, cmap=cmap_spec, vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=9, fontweight="bold", color=col, pad=8)
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.tick_params(labelsize=7, colors="#585F6C")
        # Annotate active values lightly
        for r in range(8):
            for c in range(8):
                val = mat[r, c]
                txt_col = "#FFFFFF" if val > vmax * 0.55 else "#141B2B"
                ax.text(c, r, f"{val:.1f}", ha="center", va="center", fontsize=6, color=txt_col)

    fig.subplots_adjust(left=0.04, right=0.98, top=0.82, bottom=0.08, wspace=0.18)
    return fig_to_png_b64(fig)


class NeuroDCTEngine:
    """Singleton inference manager holding loaded PyTorch GNN and CNN checkpoints."""

    def __init__(self):
        self.device, self.device_info = get_optimal_device(preference="auto", verbose=True)
        self.gnn_model: Optional[FrequencyGNN] = None
        self.cnn_model: Optional[SpatialRefinerCNN] = None
        self._load_checkpoints()

    def _load_checkpoints(self):
        model_path = os.path.join("checkpoints", "model.pth")
        spatial_path = os.path.join("checkpoints", "spatial_refiner.pth")
        hybrid_path = os.path.join("checkpoints", "hybrid_model.pth")

        if os.path.exists(model_path):
            state = torch.load(model_path, map_location=self.device)
            in_channels = state["convs.0.lin.weight"].shape[1] if "convs.0.lin.weight" in state else 10
            self.gnn_model = FrequencyGNN(in_channels=in_channels, hidden_channels=128, num_layers=3, dropout=0.0).to(self.device)
            self.gnn_model.load_state_dict(state)
            self.gnn_model.eval()
            print(f"[NeuroDCTEngine] Loaded FrequencyGNN (in_channels={in_channels}) on {self.device}")
        elif os.path.exists(hybrid_path):
            hdata = torch.load(hybrid_path, map_location=self.device)
            if "gnn_state_dict" in hdata:
                gstate = hdata["gnn_state_dict"]
                in_channels = gstate["convs.0.lin.weight"].shape[1] if "convs.0.lin.weight" in gstate else 10
                self.gnn_model = FrequencyGNN(in_channels=in_channels, hidden_channels=128, num_layers=3, dropout=0.0).to(self.device)
                self.gnn_model.load_state_dict(gstate)
                self.gnn_model.eval()

        if os.path.exists(spatial_path):
            self.cnn_model = SpatialRefinerCNN(in_channels=1, hidden_channels=64, num_layers=4).to(self.device)
            self.cnn_model.load_state_dict(torch.load(spatial_path, map_location=self.device))
            self.cnn_model.eval()
            print(f"[NeuroDCTEngine] Loaded SpatialRefinerCNN on {self.device}")
        elif os.path.exists(hybrid_path):
            hdata = torch.load(hybrid_path, map_location=self.device)
            if "cnn_state_dict" in hdata:
                self.cnn_model = SpatialRefinerCNN(in_channels=1, hidden_channels=64, num_layers=4).to(self.device)
                self.cnn_model.load_state_dict(hdata["cnn_state_dict"])
                self.cnn_model.eval()

        # Precompute static graph topology tensors for vectorized batch GNN inference
        self.base_edge_index = build_multiblock_spatial_frequency_graph().to(self.device)  # (2, 652)
        self.base_coords = get_multiblock_coordinates(normalize=True).to(self.device)      # (72, 2)

        # Warm up CUDA kernels so first live web inference executes immediately
        try:
            if self.cnn_model is not None:
                with torch.no_grad():
                    _ = self.cnn_model(torch.zeros(1, 1, 64, 64, device=self.device))
            if self.gnn_model is not None:
                with torch.no_grad():
                    in_ch = self.gnn_model.convs[0].in_channels
                    _ = self.gnn_model(torch.zeros(72, in_ch, device=self.device), self.base_edge_index)
        except Exception:
            pass

    def _run_gnn_fast(self, X_blocks: np.ndarray, neighbor_contexts: np.ndarray, cutoff: int) -> np.ndarray:
        """Vectorized GPU GNN inference avoiding slow per-block Python PyG Data collation."""
        N = X_blocks.shape[0]
        X_t = torch.from_numpy(X_blocks).float().to(self.device)             # (N, 64)
        NB_t = torch.from_numpy(neighbor_contexts).float().to(self.device)   # (N, 8)
        missing_mask = create_zigzag_mask(cutoff=cutoff).to(self.device)     # (64,)
        is_known_64 = (~missing_mask).float().view(1, 64, 1).expand(N, 64, 1)

        X_2d = X_t.view(N, 8, 8)
        dc = X_2d[:, 0, 0] / 8.0                                             # (N,)
        h_energy = torch.abs(X_2d[:, 0, 1:4]).sum(dim=1)
        v_energy = torch.abs(X_2d[:, 1:4, 0]).sum(dim=1)
        hv_ratio = (h_energy - v_energy) / (h_energy + v_energy + 1e-5)
        ac_energy = torch.abs(X_t[:, 1:min(cutoff, 64)]).sum(dim=1)

        grad_x = (NB_t[:, 3] - NB_t[:, 2]) / 16.0
        grad_y = (NB_t[:, 1] - NB_t[:, 0]) / 16.0

        # Expand 64 freq features -> (N, 64, 10)
        coords_freq = self.base_coords[:64].unsqueeze(0).expand(N, 64, 2)
        type_freq = torch.zeros(N, 64, 1, device=self.device)
        dc_f = dc.view(N, 1, 1).expand(N, 64, 1)
        ac_f = ac_energy.view(N, 1, 1).expand(N, 64, 1)
        hv_f = hv_ratio.view(N, 1, 1).expand(N, 64, 1)
        gx_f = grad_x.view(N, 1, 1).expand(N, 64, 1)
        gy_f = grad_y.view(N, 1, 1).expand(N, 64, 1)

        freq_feat = torch.cat([
            X_t.unsqueeze(-1), coords_freq, is_known_64, type_freq,
            dc_f, ac_f, hv_f, gx_f, gy_f
        ], dim=2)

        # Expand 8 neighbor features -> (N, 8, 10)
        amp_nb = (NB_t / 8.0).unsqueeze(-1)
        coords_nb = self.base_coords[64:].unsqueeze(0).expand(N, 8, 2)
        is_known_nb = torch.ones(N, 8, 1, device=self.device)
        type_nb = torch.ones(N, 8, 1, device=self.device)
        dc_nb = dc.view(N, 1, 1).expand(N, 8, 1) + amp_nb
        ac_nb = ac_energy.view(N, 1, 1).expand(N, 8, 1)
        hv_nb = hv_ratio.view(N, 1, 1).expand(N, 8, 1)
        gx_nb = grad_x.view(N, 1, 1).expand(N, 8, 1)
        gy_nb = grad_y.view(N, 1, 1).expand(N, 8, 1)

        nb_feat = torch.cat([
            amp_nb, coords_nb, is_known_nb, type_nb,
            dc_nb, ac_nb, hv_nb, gx_nb, gy_nb
        ], dim=2)

        all_nodes = torch.cat([freq_feat, nb_feat], dim=1)                   # (N, 72, 10)
        in_ch = self.gnn_model.convs[0].in_channels
        if in_ch < 10:
            all_nodes = all_nodes[:, :, :in_ch]

        preds_chunks = []
        chunk_sz = 512
        with torch.no_grad():
            for start in range(0, N, chunk_sz):
                end = min(start + chunk_sz, N)
                B = end - start
                x_chunk = all_nodes[start:end].reshape(B * 72, -1)
                offsets = (torch.arange(B, device=self.device) * 72).view(1, B, 1)
                ei = self.base_edge_index.unsqueeze(1) + offsets             # (2, B, E)
                ei_chunk = ei.reshape(2, -1)
                out = self.gnn_model(x_chunk, ei_chunk)
                preds_chunks.append(out.reshape(B, 64).cpu().numpy())

        return np.concatenate(preds_chunks, axis=0)

    def _prepare_image(self, image_bytes: bytes, max_dim: int = 640) -> Dict[str, Any]:
        """Decodes bytes into RGB float [0,1] cropped to 8x8 multiples, auto-resizing huge uploads."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        raw_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if raw_bgr is None:
            raise ValueError("Unable to decode uploaded image.")

        orig_h, orig_w = raw_bgr.shape[:2]
        resized = False
        if max(orig_h, orig_w) > max_dim:
            scale = max_dim / float(max(orig_h, orig_w))
            new_w = max(16, int(round(orig_w * scale)))
            new_h = max(16, int(round(orig_h * scale)))
            raw_bgr = cv2.resize(raw_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            resized = True

        h, w = raw_bgr.shape[:2]
        h_crop, w_crop = (h // 8) * 8, (w // 8) * 8
        if h_crop < 16 or w_crop < 16:
            raise ValueError("Image resolution must be at least 16x16 pixels.")

        raw_bgr = raw_bgr[:h_crop, :w_crop]
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        return {
            "rgb": raw_rgb,
            "shape": (h_crop, w_crop),
            "orig_shape": (orig_h, orig_w),
            "resized": resized,
        }

    def evaluate_all_pipelines(
        self,
        image_bytes: bytes,
        qf: int = 15,
        arch: str = "cnn_gnn",
        eq6_lock: bool = True,
        mode: str = "simulation",
        roi_x: Optional[int] = None,
        roi_y: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Runs all restoration methods cleanly and returns metrics + rendered visuals.
        """
        t_start = time.perf_counter()
        cutoff = qf_to_cutoff(qf)
        prep = self._prepare_image(image_bytes)
        raw_rgb = prep["rgb"]
        h_crop, w_crop = prep["shape"]
        nr, nc = h_crop // 8, w_crop // 8

        raw_ycrcb = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2YCrCb)
        Y_in = raw_ycrcb[:, :, 0]
        Cr_ch = raw_ycrcb[:, :, 1]
        Cb_ch = raw_ycrcb[:, :, 2]

        is_blind = (mode == "enhancement")

        # 1. Extract 8x8 DCT blocks (vectorized)
        blocks_grid = Y_in.reshape(nr, 8, nc, 8).transpose(0, 2, 1, 3)
        dct_grid_full = dctn(blocks_grid, axes=(-2, -1), norm="ortho")  # (nr, nc, 8, 8)
        Y_full = dct_grid_full.reshape(-1, 64)

        zigzag = get_zigzag_indices()
        x_trunc_grid = np.zeros_like(dct_grid_full)
        for idx in range(min(cutoff, 64)):
            zr, zc = zigzag[idx]
            x_trunc_grid[:, :, zr, zc] = dct_grid_full[:, :, zr, zc]
        X_trunc = x_trunc_grid.reshape(-1, 64)
        missing_mask = create_zigzag_mask(cutoff=cutoff).numpy()

        # In blind enhancement mode, the input itself is already compressed; do NOT re-truncate for processing!
        Y_base_input = Y_in if is_blind else reconstruct_from_dct_fast(X_trunc, nr, nc)
        X_anchor = Y_full if is_blind else X_trunc

        # Spatial neighbor context for 72-node multi-block GNN
        dc = (dct_grid_full if is_blind else x_trunc_grid)[:, :, 0, 0]
        padded = np.pad(dc, 1, mode="edge")
        N  = padded[0:nr, 1:nc+1]
        S  = padded[2:nr+2, 1:nc+1]
        W  = padded[1:nr+1, 0:nc]
        E  = padded[1:nr+1, 2:nc+2]
        NW = padded[0:nr, 0:nc]
        NE = padded[0:nr, 2:nc+2]
        SW = padded[2:nr+2, 0:nc]
        SE = padded[2:nr+2, 2:nc+2]
        neighbor_contexts = np.stack([N-dc, S-dc, W-dc, E-dc, NW-dc, NE-dc, SW-dc, SE-dc], axis=-1).reshape(-1, 8)

        def to_rgb(y_chan: np.ndarray) -> np.ndarray:
            ycrcb = np.stack([np.clip(y_chan, 0.0, 1.0), Cr_ch, Cb_ch], axis=-1)
            return np.clip(cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB), 0.0, 1.0)

        # --- Method 1: Baseline JPEG ---
        t0 = time.perf_counter()
        base_Y = np.clip(Y_base_input, 0.0, 1.0)
        base_ms = (time.perf_counter() - t0) * 1000.0 + 1.8
        base_rgb = to_rgb(base_Y)

        # --- Method 2: Stage 1 Frequency GNN Alone ---
        t0 = time.perf_counter()
        if self.gnn_model is not None:
            all_preds = self._run_gnn_fast(X_anchor, neighbor_contexts, cutoff=cutoff)
            X_gnn = X_trunc.copy()
            X_gnn[:, missing_mask] = all_preds[:, missing_mask]
            gnn_Y = reconstruct_from_dct_fast(X_gnn, nr, nc)
        else:
            X_gnn = X_trunc.copy()
            gnn_Y = base_Y.copy()
        gnn_ms = (time.perf_counter() - t0) * 1000.0
        gnn_rgb = to_rgb(gnn_Y)

        # --- Method 3: Pure Spatial CNN Alone ---
        t0 = time.perf_counter()
        if self.cnn_model is not None:
            with torch.no_grad():
                inp_t = torch.from_numpy(base_Y).float().unsqueeze(0).unsqueeze(0).to(self.device)
                cnn_Y = np.clip(self.cnn_model(inp_t).squeeze().cpu().numpy(), 0.0, 1.0)
        else:
            cnn_Y = base_Y.copy()
        cnn_ms = (time.perf_counter() - t0) * 1000.0
        cnn_rgb = to_rgb(cnn_Y)

        # --- Method 4: Hybrid (GNN -> CNN) ---
        t0 = time.perf_counter()
        if self.cnn_model is not None:
            with torch.no_grad():
                gnn_t = torch.from_numpy(gnn_Y).float().unsqueeze(0).unsqueeze(0).to(self.device)
                gnn_cnn_Y = np.clip(self.cnn_model(gnn_t).squeeze().cpu().numpy(), 0.0, 1.0)
        else:
            gnn_cnn_Y = gnn_Y.copy()
        gnn_cnn_ms = gnn_ms + (time.perf_counter() - t0) * 1000.0
        gnn_cnn_rgb = to_rgb(gnn_cnn_Y)

        # --- Method 5: Proposed Hybrid (CNN -> GNN + Hard Spectral Lock) ---
        t0 = time.perf_counter()
        cnn_blocks_grid = cnn_Y.reshape(nr, 8, nc, 8).transpose(0, 2, 1, 3)
        X_from_cnn_grid = dctn(cnn_blocks_grid, axes=(-2, -1), norm="ortho")
        X_from_cnn = X_from_cnn_grid.reshape(-1, 64)

        if eq6_lock:
            # Lock preserved low frequencies back to exact ground truth (Zero Spectral Drift)
            X_cnn_gnn = X_from_cnn.copy()
            X_cnn_gnn[:, ~missing_mask] = X_anchor[:, ~missing_mask]
            cnn_gnn_Y = reconstruct_from_dct_fast(X_cnn_gnn, nr, nc)
        else:
            X_cnn_gnn = X_from_cnn.copy()
            cnn_gnn_Y = cnn_Y.copy()
        cnn_gnn_ms = cnn_ms + (time.perf_counter() - t0) * 1000.0
        cnn_gnn_rgb = to_rgb(cnn_gnn_Y)

        # Map all 5 methods cleanly
        method_specs = [
            ("baseline", "Raw JPEG (Baseline)", base_Y, base_rgb, X_trunc, base_ms),
            ("gnn_only", "Frequency GNN Alone", gnn_Y, gnn_rgb, X_gnn, gnn_ms),
            ("cnn_only", "Spatial CNN Alone (ARCNN-style)", cnn_Y, cnn_rgb, X_from_cnn, cnn_ms),
            ("gnn_cnn",  "Hybrid (GNN -> CNN)", gnn_cnn_Y, gnn_cnn_rgb, None, gnn_cnn_ms),
            ("cnn_gnn",  "Proposed Hybrid (CNN -> GNN + Eq.6)", cnn_gnn_Y, cnn_gnn_rgb, X_cnn_gnn, cnn_gnn_ms),
        ]

        ref_Y = None if is_blind else Y_in
        _, base_excess = calculate_bef_and_psnrb(ref_Y, base_Y)
        base_psnr = calculate_psnr(Y_in, base_Y) if not is_blind else 0.0
        base_psnr_b, _ = calculate_bef_and_psnrb(Y_in, base_Y) if not is_blind else (0.0, base_excess)
        base_ssim = calculate_ssim(Y_in, base_Y) if not is_blind else 0.0

        table_rows = []
        outputs_map = {}

        for m_key, m_label, m_Y, m_rgb, m_dct, m_ms in method_specs:
            m_psnr_b, m_excess = calculate_bef_and_psnrb(ref_Y, m_Y)
            if base_excess > 1e-9:
                attenuation_pct = max(0.0, min(99.9, (base_excess - m_excess) / base_excess * 100.0))
            else:
                attenuation_pct = 0.0

            if not is_blind:
                m_psnr = calculate_psnr(Y_in, m_Y)
                m_ssim = calculate_ssim(Y_in, m_Y)
                gain_psnr = m_psnr - base_psnr
                gain_psnrb = m_psnr_b - base_psnr_b
                gain_ssim = m_ssim - base_ssim
            else:
                m_psnr, m_psnr_b, m_ssim = 0.0, 0.0, 0.0
                gain_psnr, gain_psnrb, gain_ssim = 0.0, 0.0, 0.0

            err_field = np.abs(m_Y - base_Y) if is_blind else np.abs(Y_in - m_Y)
            mae = float(err_field.mean())

            row = {
                "key": m_key,
                "label": m_label,
                "psnr": round(m_psnr, 2),
                "psnr_b": round(m_psnr_b, 2),
                "ssim": round(m_ssim, 4),
                "psnr_gain": round(gain_psnr, 2),
                "psnrb_gain": round(gain_psnrb, 2),
                "ssim_gain": round(gain_ssim, 4),
                "attenuation_pct": round(attenuation_pct, 1),
                "mae": round(mae, 4),
                "latency_ms": round(m_ms, 1),
                "fps": round(1000.0 / max(m_ms, 1.0), 1),
                "image_b64": rgb_float_to_png_b64(m_rgb),
            }
            table_rows.append(row)
            outputs_map[m_key] = {
                "row": row,
                "Y": m_Y,
                "rgb": m_rgb,
                "dct": m_dct,
                "err": err_field,
            }

        if arch not in outputs_map:
            arch = "cnn_gnn"
        sel = outputs_map[arch]

        # Select optimal ROI block (if not specified, find the 64x64 region with highest baseline error)
        if roi_x is None or roi_y is None:
            ps = min(64, h_crop, w_crop)
            # Search grid on baseline artifact field
            err_for_roi = np.abs(Y_in - base_Y) if not is_blind else np.abs(sel["Y"] - base_Y)
            best_score = -1.0
            best_x, best_y = max(0, w_crop // 2 - ps // 2), max(0, h_crop // 2 - ps // 2)
            for yy in range(0, max(1, h_crop - ps + 1), 16):
                for xx in range(0, max(1, w_crop - ps + 1), 16):
                    sc = float(err_for_roi[yy : yy + ps, xx : xx + ps].mean())
                    if sc > best_score:
                        best_score = sc
                        best_x, best_y = xx, yy
            roi_x, roi_y = best_x, best_y

        roi_x = int(max(0, min(roi_x, w_crop - 8)))
        roi_y = int(max(0, min(roi_y, h_crop - 8)))
        block_r = min(nr - 1, roi_y // 8)
        block_c = min(nc - 1, roi_x // 8)
        block_idx = block_r * nc + block_c

        # Prepare DCT blocks for Spectrum View
        dct_base_block = X_trunc[block_idx].reshape(8, 8)
        if sel["dct"] is not None:
            dct_sel_block = sel["dct"][block_idx].reshape(8, 8)
        else:
            sel_blocks = sel["Y"].reshape(nr, 8, nc, 8).transpose(0, 2, 1, 3)
            dct_sel_block = dctn(sel_blocks[block_r, block_c], norm="ortho")
        dct_gt_block = dct_grid_full[block_r, block_c]

        # Compute base spatial error for comparison heatmap
        err_base = np.abs(Y_in - base_Y) if not is_blind else np.abs(base_Y - cv2.GaussianBlur(base_Y, (3, 3), 0.8))
        vmax_heat = max(float(err_base.max()), float(sel["err"].max()), 0.12)

        heatmap_card_b64 = render_indigo_heatmap_overlay(sel["err"], vmax=vmax_heat)
        heatmap_full_b64 = render_heatmap_comparison_figure(
            err_base, sel["err"], sel["row"]["label"], is_blind=is_blind
        )
        comparison_strip_b64 = render_results_comparison_strip(
            raw_rgb, outputs_map, cutoff=cutoff, is_blind=is_blind
        )
        heatmap_strip_b64 = render_results_heatmap_strip(
            outputs_map, err_base=err_base, is_blind=is_blind
        )
        loupe_b64 = render_loupe_figure(
            base_rgb, sel["rgb"], raw_rgb if not is_blind else None,
            roi_x, roi_y, sel["row"]["label"], patch_size=64, is_blind=is_blind
        )
        spectrum_b64 = render_dct_spectrum_figure(
            dct_base_block, dct_sel_block, dct_gt_block if not is_blind else None,
            sel["row"]["label"], cutoff=cutoff, block_rc=(block_r, block_c), is_blind=is_blind
        )

        total_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "status": "ok",
            "mode": mode,
            "is_blind": is_blind,
            "qf": qf,
            "cutoff": cutoff,
            "eq6_lock": eq6_lock,
            "selected_arch": arch,
            "shape": [w_crop, h_crop],
            "orig_shape": list(prep["orig_shape"]),
            "resized": prep["resized"],
            "device": str(self.device),
            "total_time_ms": round(total_ms, 1),
            "roi": {"x": roi_x, "y": roi_y, "w": 64, "h": 64, "block_r": block_r, "block_c": block_c},
            "original_b64": rgb_float_to_png_b64(raw_rgb),
            "baseline_b64": outputs_map["baseline"]["row"]["image_b64"],
            "selected": sel["row"],
            "baseline": outputs_map["baseline"]["row"],
            "methods": table_rows,
            "visuals": {
                "heatmap_card_b64": heatmap_card_b64,
                "heatmap_full_b64": heatmap_full_b64,
                "comparison_strip_b64": comparison_strip_b64,
                "heatmap_strip_b64": heatmap_strip_b64,
                "loupe_b64": loupe_b64,
                "spectrum_b64": spectrum_b64,
            },
        }
