"""
scripts/build_report.py - Generates MS Word-ready Project Report in report/
-------------------------------------------------------------------------
Saves all figures, diagrams, equations, and visual comparisons as standalone
PNG files inside report/images/ and links them via relative paths with exact
pixel width/height attributes so Microsoft Word renders every image cleanly.
"""

import os
import sys
import shutil
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.inference import NeuroDCTEngine, PRESETS


REPORT_DIR = "report"
IMG_DIR = os.path.join(REPORT_DIR, "images")


def save_fig_png(fig, filename: str, max_display_w: int = 620, dpi: int = 220) -> tuple:
    """Saves matplotlib figure to report/images/<filename> and returns (rel_path, disp_w, disp_h)."""
    out_path = os.path.join(IMG_DIR, filename)
    fig.savefig(out_path, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.08, facecolor="#FFFFFF")
    plt.close(fig)
    im = Image.open(out_path)
    w, h = im.size
    if w > max_display_w:
        scale = max_display_w / float(w)
        disp_w = max_display_w
        disp_h = int(round(h * scale))
    else:
        disp_w, disp_h = w, h
    return f"images/{filename}", disp_w, disp_h


def copy_and_resize_asset(src_path: str, filename: str, max_display_w: int = 620) -> tuple:
    """Copies an image file into report/images/<filename> cleanly and returns (rel_path, disp_w, disp_h)."""
    out_path = os.path.join(IMG_DIR, filename)
    im = Image.open(src_path).convert("RGB")
    if im.width > 1400:
        ratio = 1400.0 / float(im.width)
        im = im.resize((1400, int(round(im.height * ratio))), Image.LANCZOS)
    im.save(out_path, format="PNG", optimize=True)

    scale = min(1.0, max_display_w / float(im.width))
    disp_w = int(round(im.width * scale))
    disp_h = int(round(im.height * scale))
    return f"images/{filename}", disp_w, disp_h


def save_b64_png(b64_uri: str, filename: str, max_display_w: int = 620) -> tuple:
    """Decodes base64 PNG URI and saves directly into report/images/<filename>."""
    import base64
    raw = base64.b64decode(b64_uri.split(",", 1)[1])
    out_path = os.path.join(IMG_DIR, filename)
    with open(out_path, "wb") as f:
        f.write(raw)
    im = Image.open(out_path)
    scale = min(1.0, max_display_w / float(im.width))
    return f"images/{filename}", int(round(im.width * scale)), int(round(im.height * scale))


def save_equation_png(tex: str, filename: str, fontsize: int = 12, max_display_w: int = 520, max_display_h: int = 58) -> tuple:
    """Renders crisp textbook LaTeX math to report/images/<filename> with Word-safe dimensions."""
    plt.rcParams["mathtext.fontset"] = "cm"
    fig = plt.figure(figsize=(7.5, 0.9), dpi=250)
    fig.patch.set_facecolor("#FFFFFF")
    fig.text(0.5, 0.5, tex, fontsize=fontsize, ha="center", va="center", color="#111827")
    out_path = os.path.join(IMG_DIR, filename)
    fig.savefig(out_path, format="png", dpi=250, bbox_inches="tight", pad_inches=0.08, facecolor="#FFFFFF")
    plt.close(fig)

    im = Image.open(out_path)
    scale_w = max_display_w / float(im.width)
    scale_h = max_display_h / float(im.height)
    scale = min(1.0, scale_w, scale_h)
    return f"images/{filename}", int(round(im.width * scale)), int(round(im.height * scale))


def render_set5_chart() -> tuple:
    plt.rcParams["font.sans-serif"] = ["Times New Roman", "DejaVu Serif", "serif"]
    labels = ["Baby", "Bird", "Butterfly", "Head", "Woman", "AVERAGE"]
    base = [36.41, 36.14, 26.40, 33.73, 31.09, 32.75]
    gnn = [36.41, 36.15, 26.40, 33.73, 31.09, 32.76]
    gnn_cnn = [36.81, 38.52, 30.17, 34.00, 32.65, 34.43]
    cnn_gnn = [37.03, 38.93, 30.53, 34.12, 32.89, 34.70]

    x = np.arange(len(labels))
    w = 0.20

    fig, ax = plt.subplots(figsize=(8.6, 3.6), dpi=220)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    ax.bar(x - 1.5*w, base, w, label="Baseline JPEG (Cutoff=15)", color="#E2E8F0", edgecolor="#64748B", linewidth=0.6)
    ax.bar(x - 0.5*w, gnn, w, label="Pure Frequency GNN", color="#94A3B8", edgecolor="#475569", linewidth=0.6)
    ax.bar(x + 0.5*w, gnn_cnn, w, label="Hybrid (GNN -> CNN)", color="#64748B", edgecolor="#334155", linewidth=0.6)
    bars_best = ax.bar(x + 1.5*w, cnn_gnn, w, label="Proposed (CNN -> GNN + Eq. 6)", color="#1E3A5F", edgecolor="#0F172A", linewidth=0.6)

    for b in bars_best:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 0.25, f"{h:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#1E3A5F")

    ax.set_ylabel("PSNR (dB)", fontsize=10, fontweight="bold", color="#111827")
    ax.set_title("Set5 Quantitative Reconstruction Comparison (Higher is Better)", fontsize=11, fontweight="bold", color="#1E3A5F", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5, color="#111827")
    ax.set_ylim(24, 41.5)
    ax.grid(axis="y", linestyle="--", alpha=0.4, color="#CBD5E1")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=8, ncol=2, loc="upper right")

    return save_fig_png(fig, "fig5_1_set5_chart.png", max_display_w=620)


def img_tag(info: tuple, alt: str, border: bool = True) -> str:
    src, w, h = info
    style = 'border: 1px solid #D1D5DB;' if border else 'border: none;'
    return f'<img src="{src}" width="{w}" height="{h}" alt="{alt}" style="{style}" />'


def build_report():
    os.makedirs(IMG_DIR, exist_ok=True)

    print("Saving equations to report/images/ ...")
    eq_dct = save_equation_png(
        r"$F(u, v) = \alpha(u)\alpha(v)\sum_{x=0}^{7}\sum_{y=0}^{7} f(x, y)\cos\!\left[\frac{(2x+1)u\pi}{16}\right]\cos\!\left[\frac{(2y+1)v\pi}{16}\right]$",
        "eq1_dct.png",
    )
    eq_loss = save_equation_png(
        r"$\mathcal{L}_{\mathrm{freq}} = \frac{1}{|M|}\sum_{k \in M} w_k \left|\hat{Y}_k - Y_k\right|, \quad w_k = \frac{(k - c + 1)^{-0.5}}{\frac{1}{|M|}\sum_{j \in M}(j - c + 1)^{-0.5}}$",
        "eq3_loss.png",
    )
    eq_lock = save_equation_png(
        r"$X_{\mathrm{final}}[i] = X_{\mathrm{true}}[i] \;\; (0 \leq i < 15), \qquad X_{\mathrm{final}}[i] = \mathcal{F}_{\mathrm{DCT}}\!\left(\mathrm{CNN}(I_{\mathrm{base}})\right)[i] \;\; (15 \leq i < 64)$",
        "eq6_lock.png",
    )
    eq_psnr = save_equation_png(
        r"$\mathrm{PSNR} = 10 \log_{10}\!\left(\frac{\mathrm{MAX}^2}{\mathrm{MSE}}\right), \qquad \mathrm{PSNR\mathrm{-}B} = 10 \log_{10}\!\left(\frac{\mathrm{MAX}^2}{\mathrm{MSE} + \mathrm{BEF}}\right)$",
        "eq4_psnr.png",
    )
    eq_ssim = save_equation_png(
        r"$\mathrm{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$",
        "eq5_ssim.png",
    )

    chart_set5 = render_set5_chart()

    print("Copying technical diagrams to report/images/ ...")
    img_jpeg_flaw    = copy_and_resize_asset("assets/diagram_jpeg_flaw.png", "fig1_1_jpeg_flaw.png")
    img_gnn_pipe     = copy_and_resize_asset("assets/diagram_gnn_pipeline.png", "fig3_1_gnn_pipe.png")
    img_graph_topo   = copy_and_resize_asset("assets/diagram_graph_topology.png", "fig3_2_graph_topo.png")
    img_gnn_cnn_pipe = copy_and_resize_asset("assets/diagram_gnn_cnn_pipeline.png", "fig3_3_gnn_cnn_pipe.png")
    img_cnn_gnn_pipe = copy_and_resize_asset("assets/diagram_cnn_gnn_pipeline.png", "fig3_4_cnn_gnn_pipe.png")
    img_zoom_seam    = copy_and_resize_asset("assets/zoomed_seam_patch.png", "fig5_6_zoom_seam.png")

    print("Copying newly evaluated Set5 Butterfly & Set14 Barbara results to report/images/ ...")
    img_bf_comp   = copy_and_resize_asset("results/Set5_img_003_SRF_2_comparison.png", "fig5_2_butterfly_comp.png")
    img_bf_heat   = copy_and_resize_asset("results/Set5_img_003_SRF_2_error_heatmap.png", "fig5_3_butterfly_heat.png")
    img_barb_comp = copy_and_resize_asset("results/Set14_img_002_SRF_2_comparison.png", "fig5_4_barbara_comp.png")
    img_barb_heat = copy_and_resize_asset("results/Set14_img_002_SRF_2_error_heatmap.png", "fig5_5_barbara_heat.png")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Project Report — Dual-Domain JPEG Artifact Restoration (NeuroDCT)</title>
  <style>
    @page {{
      size: A4;
      margin: 0.85in 0.85in 0.85in 0.85in;
    }}

    body {{
      font-family: "Times New Roman", Cambria, Georgia, serif;
      font-size: 11.5pt;
      line-height: 1.55;
      color: #111827;
      background-color: #FFFFFF;
      margin: 0 auto;
      padding: 30px 40px;
      max-width: 820px;
    }}

    h1, h2, h3, h4 {{
      font-family: "Calibri", "Segoe UI", Arial, sans-serif;
      color: #1E3A5F;
      margin-top: 22pt;
      margin-bottom: 8pt;
      page-break-after: avoid;
    }}

    h1.chapter-title {{
      font-size: 17pt;
      border-bottom: 1.5pt solid #1E3A5F;
      padding-bottom: 4pt;
      margin-top: 24pt;
    }}

    h2 {{
      font-size: 13pt;
      color: #1F2937;
      margin-top: 16pt;
    }}

    p {{
      text-align: justify;
      margin-top: 0;
      margin-bottom: 9pt;
    }}

    ul, ol {{
      margin-top: 4pt;
      margin-bottom: 10pt;
      padding-left: 22pt;
    }}

    li {{
      margin-bottom: 4pt;
      text-align: justify;
    }}

    code {{
      font-family: "Consolas", "Courier New", monospace;
      font-size: 9.5pt;
      background-color: #F3F4F6;
      padding: 1pt 3pt;
      color: #111827;
    }}

    .page-break {{
      page-break-before: always;
      mso-break-type: page-break;
    }}

    .cover-page {{
      text-align: center;
      padding-top: 60pt;
      padding-bottom: 40pt;
    }}
    .cover-label {{
      font-family: "Calibri", Arial, sans-serif;
      font-size: 11pt;
      font-weight: bold;
      letter-spacing: 1.5pt;
      text-transform: uppercase;
      color: #4B5563;
      margin-bottom: 18pt;
    }}
    .cover-title {{
      font-family: "Calibri", Arial, sans-serif;
      font-size: 24pt;
      font-weight: bold;
      color: #1E3A5F;
      line-height: 1.25;
      margin-bottom: 14pt;
    }}
    .cover-subtitle {{
      font-size: 13pt;
      font-style: italic;
      color: #374151;
      margin-bottom: 36pt;
    }}
    .cover-box {{
      border: 1pt solid #9CA3AF;
      background-color: #F9FAFB;
      padding: 16pt 24pt;
      margin: 20pt auto;
      width: 85%;
      text-align: left;
    }}

    table.academic-table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 10pt;
      margin-bottom: 14pt;
      font-size: 10pt;
      page-break-inside: avoid;
    }}
    table.academic-table th {{
      background-color: #1E3A5F;
      color: #FFFFFF;
      font-family: "Calibri", Arial, sans-serif;
      font-weight: bold;
      text-align: left;
      padding: 6pt 8pt;
      border: 0.75pt solid #1E3A5F;
    }}
    table.academic-table td {{
      padding: 5.5pt 8pt;
      border: 0.75pt solid #D1D5DB;
      color: #111827;
    }}
    table.academic-table tr.alt-row td {{
      background-color: #F9FAFB;
    }}
    table.academic-table tr.highlight-row td {{
      background-color: #EEF2F7;
      font-weight: bold;
      color: #1E3A5F;
    }}

    .eq-box {{
      width: 100%;
      margin: 10pt 0;
      border: 0.5pt solid #E5E7EB;
      background-color: #FAFAFA;
      padding: 6pt 10pt;
      page-break-inside: avoid;
    }}
    .eq-table {{
      width: 100%;
      border-collapse: collapse;
    }}
    .eq-num {{
      width: 55pt;
      text-align: right;
      font-family: "Times New Roman", serif;
      font-size: 10.5pt;
      color: #4B5563;
      vertical-align: middle;
    }}

    .figure-block {{
      text-align: center;
      margin: 14pt 0;
      page-break-inside: avoid;
    }}
    .figure-caption {{
      font-family: "Calibri", Arial, sans-serif;
      font-size: 9.5pt;
      color: #4B5563;
      margin-top: 5pt;
      font-style: italic;
    }}

    .note-box {{
      border-left: 3pt solid #1E3A5F;
      background-color: #F8FAFC;
      padding: 8pt 12pt;
      margin: 10pt 0;
      font-size: 10.5pt;
      page-break-inside: avoid;
    }}
  </style>
</head>
<body>

  <!-- ==================== COVER PAGE ==================== -->
  <div class="cover-page">
    <div class="cover-label">Digital Image Processing — Project Assignment Report</div>
    <div class="cover-title">
      Dual-Domain JPEG Artifact Restoration:<br/>
      Synergizing Spatial Convolutional Networks and Frequency Graph Neural Networks
    </div>
    <div class="cover-subtitle">
      Eliminating Dirichlet Boundary Blocking and Quantization Spectral Loss via Hard Spectral Consistency
    </div>

    <div class="cover-box">
      <table style="width:100%; border-collapse:collapse; font-size:11pt;">
        <tr>
          <td style="padding:4pt 0; width:35%; font-weight:bold; color:#1E3A5F;">Project System:</td>
          <td style="padding:4pt 0;">NeuroDCT Artifact Reduction Lab (v3.4)</td>
        </tr>
        <tr>
          <td style="padding:4pt 0; font-weight:bold; color:#1E3A5F;">Submitted By:</td>
          <td style="padding:4pt 0;">[Your Name &amp; Roll Number]<br/>[Teammate Name &amp; Roll Number]</td>
        </tr>
        <tr>
          <td style="padding:4pt 0; font-weight:bold; color:#1E3A5F;">Submitted To:</td>
          <td style="padding:4pt 0;">[Faculty / Research Scholar Name]</td>
        </tr>
        <tr>
          <td style="padding:4pt 0; font-weight:bold; color:#1E3A5F;">Department / Institute:</td>
          <td style="padding:4pt 0;">Gauhati University (27 November 2026)</td>
        </tr>
        <tr>
          <td style="padding:4pt 0; font-weight:bold; color:#1E3A5F;">Repository:</td>
          <td style="padding:4pt 0;"><code>https://github.com/Duljit2006/DIP-project</code></td>
        </tr>
      </table>
    </div>
  </div>

  <!-- ==================== ABSTRACT ==================== -->
  <div class="page-break"></div>
  <h1 class="chapter-title">Abstract</h1>
  <p>
    Despite the availability of lossless image formats such as PNG and lossless WebP, lossy JPEG remains the predominant standard for web transmission, mobile messaging, and embedded imaging due to strict bandwidth and storage constraints. At low bitrates (high compression), standard JPEG compression partitions images into independent 8&times;8 blocks, transforms them via the 2D Discrete Cosine Transform (2D-DCT), and discards high-frequency alternating-current (AC) coefficients. This quantization yields two severe defects: blurred fine details (spectral loss) and visible checkerboard discontinuities across 8&times;8 borders (Dirichlet boundary blocking artifacts).
  </p>
  <p>
    In this project, we systematically investigate three neural restoration paradigms across both frequency and spatial domains. First, we demonstrate why a pure <b>Frequency Graph Neural Network (FrequencyGNN)</b> operating on intra-block harmonic edges fails to reduce blocking (+0.00 dB to +0.01 dB PSNR gain on Set5) due to inter-block border blindness. Second, we show that ordering the pipeline as <b>GNN &rarr; CNN</b> achieves significant seam removal (+1.68 dB average gain) but introduces <i>spectral drift</i>—where spatial convolutions inadvertently corrupt the preserved low-frequency ground-truth coefficients. Finally, we introduce our proposed <b>Dual-Domain Hybrid Architecture (CNN &rarr; GNN)</b> augmented with <b>Hard Spectral Consistency (Eq. 6)</b>. By eliminating spatial grid seams first via a residual CNN, re-projecting into the 2D-DCT domain, refining harmonic octaves, and clamping the known low frequencies back to exact ground truth, our final BSDS500-trained model (30 GNN + 70 CNN epochs) achieves an average improvement of <b>+1.95 dB</b> across Set5 (up to <b>+4.13 dB</b> on Butterfly, SSIM 0.9499) alongside <b>+1.47 dB</b> average improvement across Set14 (SSIM 0.9064). Furthermore, we deploy the complete inference system in an interactive, real-time web application (<i>NeuroDCT Lab</i>) running on GPU acceleration.
  </p>

  <!-- ==================== TABLE OF CONTENTS ==================== -->
  <h1 class="chapter-title">Table of Contents</h1>
  <table class="academic-table">
    <thead>
      <tr><th>Section</th><th>Topic Covered</th></tr>
    </thead>
    <tbody>
      <tr><td><b>Chapter 1</b></td><td>Introduction, Bandwidth Dilemma &amp; Mathematics of JPEG</td></tr>
      <tr class="alt-row"><td><b>Chapter 2</b></td><td>Literature Review &amp; Identified Architectural Gaps</td></tr>
      <tr><td><b>Chapter 3</b></td><td>Methodologies Investigated (Pure GNN, GNN&rarr;CNN, Proposed CNN&rarr;GNN)</td></tr>
      <tr class="alt-row"><td><b>Chapter 4</b></td><td>Dataset Preparation, Loss Formulation &amp; Training Pipeline</td></tr>
      <tr><td><b>Chapter 5</b></td><td>Experimental Results, Benchmarks (Set5 &amp; Set14) &amp; Visual Proof</td></tr>
      <tr class="alt-row"><td><b>Chapter 6</b></td><td>NeuroDCT Interactive Web System &amp; Deployment</td></tr>
      <tr><td><b>Chapter 7</b></td><td>Conclusion &amp; Future Scope</td></tr>
    </tbody>
  </table>

  <!-- ==================== CHAPTER 1 ==================== -->
  <div class="page-break"></div>
  <h1 class="chapter-title">Chapter 1: Introduction &amp; Problem Statement</h1>

  <h2>1.1 Why JPEG Remains Dominant Over Lossless Formats</h2>
  <p>
    Modern lossless image formats such as PNG, TIFF, and lossless WebP preserve pixel values with zero distortion. However, uncompressed or losslessly compressed high-resolution images occupy 5&times; to 10&times; greater file sizes than lossy compressed equivalents. Across global mobile networks, social messaging platforms, web browsers, and low-bandwidth remote sensing links, transmitting multi-megabyte lossless files creates prohibitive latency and storage overhead. Consequently, JPEG (ISO/IEC 10918-1) continues to power the vast majority of digital imagery. The central Digital Image Processing challenge is therefore: <i>can we retain JPEG's extreme compression efficiency while mathematically reconstructing lost structural fidelity at the receiver?</i>
  </p>

  <h2>1.2 Mathematical Working of JPEG Compression</h2>
  <p>
    To exploit spatial redundancy and human visual insensitivity to high spatial frequencies, the JPEG encoder converts an RGB image to the YCbCr color space (separating luminance <i>Y</i> from chrominance <i>Cb</i>, <i>Cr</i>) and partitions the luminance plane into non-overlapping 8&times;8 pixel blocks. Each block <i>f</i>(<i>x</i>, <i>y</i>) is transformed into 64 orthogonal spatial frequency coefficients via the Type-II Orthonormal 2D Discrete Cosine Transform (2D-DCT):
  </p>
  <div class="eq-box">
    <table class="eq-table"><tr>
      <td style="text-align:center;">{img_tag(eq_dct, "2D-DCT Equation", border=False)}</td>
      <td class="eq-num">(Eq. 1)</td>
    </tr></table>
  </div>
  <p>
    where &alpha;(0) = 1/&radic;8 and &alpha;(<i>k</i>) = &radic;(2/8) for <i>k</i> &gt; 0. The top-left coefficient <i>F</i>(0,0) represents the block's average intensity (DC component), while coefficients toward the bottom-right represent progressively higher spatial frequencies (AC components). During quantization and zigzag truncation (e.g., keeping only the first <i>c</i> = 15 coefficients out of 64), the remaining 49 high-frequency coefficients are quantized to zero to save bits.
  </p>

  <h2>1.3 Inherent Mathematical Flaws of Truncated JPEG</h2>
  <p>When the decoder applies the 2D Inverse DCT (2D-IDCT) independently to each 8&times;8 block, two distinct mathematical artifacts emerge:</p>
  <ol>
    <li><b>Spectral Information Loss (Blurring &amp; Ringing):</b> Zeroing out 49 high-frequency AC coefficients discards fine textures, hair strands, fabric weaves, and sharp contour transitions.</li>
    <li><b>Dirichlet Boundary Discontinuities (8&times;8 Block Seams):</b> Because every 8&times;8 transform operates independently with zero boundary continuity constraints against neighboring blocks, pixel intensities experience artificial step-jumps across block boundaries—producing visible checkerboard grids.</li>
  </ol>

  <div class="figure-block">
    {img_tag(img_jpeg_flaw, "8x8 DCT Truncation and Blocking Defects")}
    <div class="figure-caption">Figure 1.1: Visualization of 8&times;8 zigzag DCT truncation (15 preserved low frequencies vs. 49 zeroed high frequencies) and the resulting spatial and spectral defects.</div>
  </div>

  <!-- ==================== CHAPTER 2 ==================== -->
  <h1 class="chapter-title">Chapter 2: Literature Review</h1>
  <p>
    Existing JPEG artifact reduction and visual graph architectures motivate our dual-domain design. Below we review the classical foundations alongside two primary modern architectures—<b>CARD (Sun et al., 2020)</b> and <b>Vision GNN / ViG (Han et al., NeurIPS 2022)</b>—that directly inspired our 8&times;8 DCT graph formulation:
  </p>

  <h2>2.1 Key Foundational Papers</h2>
  <ul>
    <li>
      <b>CARD — Compression Artifact Reduction with Dual-Domain Prior (Sun et al., 2020):</b> Extracts overlapping 8&times;8 patches via fixed CNN filters, predicts missing DCT coefficients using 20 Wide Activation Residual blocks, and enforces a quantization constraint prior before 2D-IDCT reconstruction. While highly effective at quantization clipping and overlapping patch prediction, it relies purely on dense convolutions rather than explicit spectral harmonic graph relationships.
    </li>
    <li>
      <b>Vision GNN / ViG (Han et al., NeurIPS 2022):</b> Represents image patches as topological graph nodes connected by semantic similarity to capture non-local spatial dependencies across the visual field. Although designed for high-level recognition rather than JPEG frequency bounds (lacking 8&times;8 DCT alignment and frequency locking), its graph message-passing paradigm directly motivated our 8&times;8 DCT-node harmonic graph formulation.
    </li>
  </ul>

  <h2>2.2 Comparative Summary of Existing Approaches</h2>
  <table class="academic-table">
    <thead>
      <tr>
        <th style="width:24%;">Method / Reference</th>
        <th style="width:36%;">Key Mechanism &amp; Strengths</th>
        <th style="width:40%;">Identified Architectural Limitation</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><b>Classical Spatial Filters</b><br/>(Bilateral, SA-DCT)</td>
        <td>Local spatial smoothing &amp; adaptive transform thresholding.</td>
        <td>Over-smooths genuine edges alongside block borders; unable to synthesize missing high-frequency harmonics.</td>
      </tr>
      <tr class="alt-row">
        <td><b>ARCNN &amp; DnCNN</b><br/>(Dong et al., 2015; Zhang et al., 2017)</td>
        <td>Pure pixel-domain CNN residual deblocking across seams.</td>
        <td>Lacks DCT spectral constraints—causing <i>Spectral Drift</i> that distorts original low-frequency brightness and hue.</td>
      </tr>
      <tr>
        <td><b>CARD</b><br/>(Sun et al., 2020)</td>
        <td>DCT prediction, overlapping 8&times;8 patches, wide residual blocks &amp; quantization clipping prior.</td>
        <td>Lacks explicit graph harmonic edges connecting fundamental frequencies to octave overtones.</td>
      </tr>
      <tr class="alt-row">
        <td><b>Vision GNN (ViG)</b><br/>(Han et al., NeurIPS 2022)</td>
        <td>Patch graph nodes with non-local edge message passing.</td>
        <td>Operates purely on semantic spatial patches; lacks 8&times;8 block alignment and DCT low-frequency spectral locking.</td>
      </tr>
    </tbody>
  </table>
  <div class="note-box">
    <b>Identified Research Gap:</b> Combining the spatial boundary continuity of residual CNNs, the non-local/harmonic message passing of GNNs (inspired by ViG), and strict DCT quantization priors (inspired by CARD) yields a unified <b>Dual-Domain Hybrid Architecture</b>.
  </div>

  <!-- ==================== CHAPTER 3 ==================== -->
  <div class="page-break"></div>
  <h1 class="chapter-title">Chapter 3: Methodologies Investigated</h1>
  <p>
    To isolate the strengths and weaknesses of frequency-domain graphs and spatial convolutions, we designed and benchmarked three progressive architectures.
  </p>

  <h2>3.1 Methodology 1: Pure Frequency GNN (<code>FrequencyGNN</code>)</h2>
  <p>
    In our first architecture, each 8&times;8 DCT block is modeled as a relational graph. To supply multi-block boundary context, we construct a <b>72-Node Spatial-Frequency Graph</b> containing 64 intra-block DCT nodes and 8 spatial neighbor super-nodes (N, S, W, E, NW, NE, SW, SE), interconnected by 652 bidirectional edges capturing 8-neighborhood spectral adjacency and harmonic octave relationships ((<i>u</i>,<i>v</i>) &harr; (2<i>u</i>,<i>v</i>), (<i>u</i>,2<i>v</i>), (2<i>u</i>,2<i>v</i>)).
  </p>
  <p>
    Each node carries a 10-channel feature vector: <code>[amplitude, u_norm, v_norm, is_known, node_type, block_dc, ac_energy, hv_ratio, grad_x, grad_y]</code>. The network uses 3 Graph Convolutional (<code>GCNConv</code>) layers (<code>hidden_channels=128</code>) with <code>LayerNorm</code>, <code>LeakyReLU(0.2)</code>, and residual skip connections.
  </p>

  <div class="figure-block">
    {img_tag(img_gnn_pipe, "Methodology 1 Pipeline")}
    <div class="figure-caption">Figure 3.1: Methodology 1 pipeline (Pure Frequency GNN) and diagnosis of inter-block border blindness.</div>
  </div>
  <div class="figure-block">
    {img_tag(img_graph_topo, "72-Node Spatial-Frequency Graph")}
    <div class="figure-caption">Figure 3.2: 72-node multi-block spatial-frequency graph topology (64 frequency nodes + 8 spatial DC context nodes, 652 edges).</div>
  </div>
  <p>
    <b>Result &amp; Diagnosis:</b> Pure GNN alone achieved only <b>+0.01 dB</b> improvement on Set5 (32.75 dB &rarr; 32.76 dB). Because message passing operates inside discrete 8&times;8 frequency windows, it lacks continuous pixel-level filters capable of spanning block boundaries.
  </p>

  <div class="page-break"></div>
  <h2>3.2 Methodology 2: Hybrid <code>GNN &rarr; CNN</code> (Frequency First, Spatial Second)</h2>
  <p>
    To eliminate block boundaries, Stage 1 uses the Frequency GNN to estimate missing high-frequency coefficients, applies 2D-IDCT to obtain an intermediate pixel image, and passes it into Stage 2—a 4-layer <b>Spatial Refiner CNN</b> (<code>SpatialRefinerCNN</code>, 64 channels, 3&times;3 kernels, <code>PReLU</code> activations, global residual connection).
  </p>
  <div class="figure-block">
    {img_tag(img_gnn_cnn_pipe, "Methodology 2 Pipeline")}
    <div class="figure-caption">Figure 3.3: Methodology 2 (GNN &rarr; CNN) workflow highlighting the Spectral Drift limitation.</div>
  </div>
  <p>
    <b>Result &amp; Dilemma:</b> Average Set5 PSNR jumped by <b>+1.68 dB</b> (32.75 dB &rarr; 34.43 dB), confirming that CNN convolutions eliminate blocking seams. However, placing the spatial CNN at the end introduced <b>Spectral Drift</b>: while smoothing seams, the 3&times;3 filters altered all pixel values indiscriminately, inadvertently corrupting the 15 low-frequency DCT coefficients that were already 100% accurate in the original compressed file.
  </p>

  <h2>3.3 Methodology 3: Proposed Hybrid <code>CNN &rarr; GNN</code> + Hard Spectral Lock (Best)</h2>
  <p>
    Reversing the execution order resolves both flaws simultaneously through a dual-domain handshake:
  </p>
  <ol>
    <li><b>1st IDCT &amp; Spatial Deblocking CNN:</b> Standard JPEG decompression (1st IDCT) renders pixels; the Spatial CNN smooths 8&times;8 grid boundaries and establishes continuous macro edges.</li>
    <li><b>2nd DCT Re-Slicing:</b> The deblocked image is sliced back into 8&times;8 DCT blocks, activating high-frequency components predicted by the CNN.</li>
    <li><b>Hard Spectral Consistency (Eq. 6 Anchor) &amp; GNN Harmonic Physics:</b> We clamp the first <i>c</i> = 15 known low-frequency coefficients back to exact ground truth while retaining only the refined high frequencies above cutoff 15:</li>
  </ol>
  <div class="eq-box">
    <table class="eq-table"><tr>
      <td style="text-align:center;">{img_tag(eq_lock, "Hard Spectral Lock Equation", border=False)}</td>
      <td class="eq-num">(Eq. 6)</td>
    </tr></table>
  </div>
  <ol start="4">
    <li><b>2nd IDCT Synthesis:</b> Converts the spectrally anchored, edge-refined coefficients into the final pixel image.</li>
  </ol>

  <div class="figure-block">
    {img_tag(img_cnn_gnn_pipe, "Methodology 3 Proposed Pipeline")}
    <div class="figure-caption">Figure 3.4: Proposed Methodology 3 (CNN &rarr; GNN) dual-domain pipeline with Hard Spectral Consistency Lock.</div>
  </div>

  <!-- ==================== CHAPTER 4 ==================== -->
  <div class="page-break"></div>
  <h1 class="chapter-title">Chapter 4: Implementation &amp; Training Pipeline</h1>

  <h2>4.1 Dataset Engineering &amp; Model Capacity Alignment</h2>
  <p>
    Training was conducted using the <b>BSDS500</b> dataset (500 natural images partitioned across train, validation, and test splits, generating ~1.2 million 8&times;8 luminance blocks). An important experimental finding during scaling tests was that training a compact 4-layer CNN (~75,000 parameters) on raw random crops from massive 2K DSLR datasets (such as DIV2K + Flickr2K) led to <i>underfitting</i> and <i>bokeh dilution</i> (capping PSNR gains around +0.9 dB), because over half of random 2K camera crops fall on out-of-focus smooth backgrounds. By contrast, dense, edge-rich natural scenes in BSDS500 align tightly with our compact model capacity. Benchmark testing is conducted strictly on unseen standard datasets: <b>Set5</b> and <b>Set14</b>.
  </p>

  <h2>4.2 Loss Functions &amp; Training Mechanics</h2>
  <ul>
    <li><b>Stage 1 (Frequency GNN):</b> Trained using <b>Frequency-Weighted L1 Loss</b>, which weights errors on primary missing mid-frequency harmonics (positions 15&ndash;30 in zigzag order) higher than ultra-high-frequency noise using an inverse power decay (decay power = 0.5):</li>
  </ul>
  <div class="eq-box">
    <table class="eq-table"><tr>
      <td style="text-align:center;">{img_tag(eq_loss, "Frequency Weighted L1 Loss", border=False)}</td>
      <td class="eq-num">(Eq. 3)</td>
    </tr></table>
  </div>
  <ul>
    <li><b>Stage 2 (Spatial Refiner CNN):</b> Trained using Spatial L1 Loss (<code>nn.L1Loss</code>) to preserve sharp edges without the over-smoothing penalty associated with L2/MSE loss.</li>
    <li><b>Optimization:</b> Adam optimizer (LR = 10<sup>-3</sup> for GNN, 5&times;10<sup>-4</sup> for CNN), StepLR decay, and GPU acceleration via PyTorch + PyTorch Geometric.</li>
  </ul>

  <h2>4.3 Quantitative Evaluation Metrics</h2>
  <p>
    In addition to standard <b>PSNR</b> and <b>SSIM</b>, we evaluate <b>PSNR-B</b> (Yim &amp; Bovik), which explicitly augments MSE with the Blocking Effect Factor (BEF) across 8&times;8 boundaries to mathematically verify grid seam elimination:
  </p>
  <div class="eq-box">
    <table class="eq-table"><tr>
      <td style="text-align:center;">{img_tag(eq_psnr, "PSNR and PSNR-B Equations", border=False)}</td>
      <td class="eq-num">(Eq. 4)</td>
    </tr></table>
  </div>
  <div class="eq-box">
    <table class="eq-table"><tr>
      <td style="text-align:center;">{img_tag(eq_ssim, "SSIM Equation", border=False)}</td>
      <td class="eq-num">(Eq. 5)</td>
    </tr></table>
  </div>

  <!-- ==================== CHAPTER 5 ==================== -->
  <div class="page-break"></div>
  <h1 class="chapter-title">Chapter 5: Experimental Results &amp; Analysis</h1>

  <h2>5.1 Set5 Quantitative Benchmark Comparison (Cutoff = 15)</h2>
  <table class="academic-table">
    <thead>
      <tr>
        <th>Set5 Test Image</th>
        <th>Baseline JPEG</th>
        <th>Pure GNN</th>
        <th>Hybrid (GNN&rarr;CNN)</th>
        <th>Proposed (CNN&rarr;GNN)</th>
        <th>CNN&rarr;GNN SSIM</th>
      </tr>
    </thead>
    <tbody>
      <tr><td><code>img_001</code> (Baby)</td><td>36.41 dB</td><td>36.41 dB (+0.00)</td><td>36.81 dB (+0.40)</td><td><b>37.03 dB (+0.62)</b></td><td>0.9671</td></tr>
      <tr class="alt-row"><td><code>img_002</code> (Bird)</td><td>36.14 dB</td><td>36.15 dB (+0.01)</td><td>38.52 dB (+2.38)</td><td><b>38.93 dB (+2.80)</b></td><td>0.9852</td></tr>
      <tr><td><code>img_003</code> (Butterfly)</td><td>26.40 dB</td><td>26.40 dB (+0.00)</td><td>30.17 dB (+3.77)</td><td><b>30.53 dB (+4.13)</b></td><td>0.9541</td></tr>
      <tr class="alt-row"><td><code>img_004</code> (Head)</td><td>33.73 dB</td><td>33.73 dB (+0.00)</td><td>34.00 dB (+0.27)</td><td><b>34.12 dB (+0.39)</b></td><td>0.8764</td></tr>
      <tr><td><code>img_005</code> (Woman)</td><td>31.09 dB</td><td>31.09 dB (+0.00)</td><td>32.65 dB (+1.56)</td><td><b>32.89 dB (+1.80)</b></td><td>0.9670</td></tr>
      <tr class="highlight-row"><td>AVERAGE (Set5)</td><td>32.75 dB</td><td>32.76 dB (+0.00)</td><td>34.43 dB (+1.68)</td><td>34.70 dB (+1.95 dB)</td><td>0.9499</td></tr>
    </tbody>
  </table>

  <div class="figure-block">
    {img_tag(chart_set5, "Set5 PSNR Quantitative Comparison Chart")}
    <div class="figure-caption">Figure 5.1: PSNR comparison across Set5 benchmark images demonstrating progressive gains from Pure GNN to Proposed CNN&rarr;GNN.</div>
  </div>

  <h2>5.2 Set14 Cross-Dataset Benchmark Summary (14 Images, Cutoff = 15)</h2>
  <p>
    Evaluated across 14 diverse high-detail natural images in <b>Set14</b>, our newly trained BSDS500 model (30 GNN + 70 CNN epochs) achieved an overall average gain of <b>+1.47 dB</b> (29.57 dB &rarr; <b>31.04 dB</b>, SSIM <b>0.9064</b>), peaking at <b>+3.67 dB</b> on Monarch (<code>img_011</code>: 35.56 dB) and <b>+3.57 dB</b> on Zebra/PPT patterns (<code>img_013</code>: 31.46 dB).
  </p>
  <table class="academic-table">
    <thead>
      <tr>
        <th>Set14 Image Subset / Highlight</th>
        <th>Baseline JPEG</th>
        <th>Pure GNN</th>
        <th>Hybrid (GNN&rarr;CNN)</th>
        <th>Proposed (CNN&rarr;GNN)</th>
        <th>CNN&rarr;GNN SSIM</th>
      </tr>
    </thead>
    <tbody>
      <tr><td><code>img_002</code> (Barbara)</td><td>26.97 dB</td><td>26.97 dB (+0.00)</td><td>27.22 dB (+0.26)</td><td><b>27.29 dB (+0.33)</b></td><td>0.8641</td></tr>
      <tr class="alt-row"><td><code>img_007</code> (Foreman/Building)</td><td>29.47 dB</td><td>29.47 dB (+0.00)</td><td>31.37 dB (+1.90)</td><td><b>31.58 dB (+2.11)</b></td><td>0.9318</td></tr>
      <tr><td><code>img_011</code> (Monarch)</td><td>31.89 dB</td><td>31.89 dB (+0.00)</td><td>35.25 dB (+3.36)</td><td><b>35.56 dB (+3.67)</b></td><td>0.9719</td></tr>
      <tr class="alt-row"><td><code>img_013</code> (High-Freq Stripes)</td><td>27.89 dB</td><td>27.89 dB (+0.00)</td><td>30.67 dB (+2.78)</td><td><b>31.46 dB (+3.57)</b></td><td>0.9788</td></tr>
      <tr><td><code>img_014</code> (Zebra/Natural)</td><td>30.46 dB</td><td>30.47 dB (+0.00)</td><td>31.61 dB (+1.15)</td><td><b>32.09 dB (+1.63)</b></td><td>0.9446</td></tr>
      <tr class="highlight-row"><td>AVERAGE (All 14 Set14 Images)</td><td>29.57 dB</td><td>29.57 dB (+0.00)</td><td>30.80 dB (+1.24)</td><td>31.04 dB (+1.47 dB)</td><td>0.9064</td></tr>
    </tbody>
  </table>

  <h2>5.3 Visual Verification: 5-Way Reconstruction, Heatmaps &amp; 4&times; Seam Loupe</h2>
  <div class="figure-block">
    {img_tag(img_bf_comp, "Set5 Butterfly 5-Way Comparison")}
    <div class="figure-caption">Figure 5.2: 5-Way Visual Reconstruction on Set5 Butterfly showing 30.53 dB (+4.13 dB) recovery.</div>
  </div>
  <div class="figure-block">
    {img_tag(img_bf_heat, "Set5 Butterfly Error Heatmap")}
    <div class="figure-caption">Figure 5.3: 4-Panel Residual Error Heatmap comparison on Set5 Butterfly (darker regions indicate lower reconstruction error).</div>
  </div>

  <div class="page-break"></div>
  <div class="figure-block">
    {img_tag(img_barb_comp, "Set14 Barbara 5-Way Comparison")}
    <div class="figure-caption">Figure 5.4: 5-Way Visual Reconstruction on Set14 Barbara showing preservation of diagonal textile patterns.</div>
  </div>
  <div class="figure-block">
    {img_tag(img_barb_heat, "Set14 Barbara Error Heatmap")}
    <div class="figure-caption">Figure 5.5: 4-Panel Residual Error Heatmap on Set14 Barbara illustrating seam suppression in CNN&rarr;GNN.</div>
  </div>
  <div class="figure-block">
    {img_tag(img_zoom_seam, "Zoomed 4x Seam Elimination Patch")}
    <div class="figure-caption">Figure 5.6: Magnified 64&times;64 patch comparison proving elimination of 8&times;8 Dirichlet checkerboard seams against uncompressed Ground Truth.</div>
  </div>

  <!-- ==================== CHAPTER 6 ==================== -->
  <div class="page-break"></div>
  <h1 class="chapter-title">Chapter 6: Interactive Web System &amp; Deployment (NeuroDCT Lab)</h1>
  <p>
    To make our research accessible for live evaluation and mobile testing by reviewers, we engineered a full-stack web laboratory application (<b>NeuroDCT Research System</b>):
  </p>
  <ul>
    <li><b>Backend Inference API (<code>server.py</code> &amp; <code>src/inference.py</code>):</b> Built with FastAPI and Uvicorn, executing vectorized 2D-IDCT and batched PyTorch GPU graph inference in ~8&ndash;12 ms on an NVIDIA RTX 2050 (or Google Colab T4 GPU).</li>
    <li><b>Responsive 3-Stage Interface (<code>web/index.html</code>):</b> Designed in a clinical Swiss laboratory theme across three structured stages:
      <ol>
        <li><i>Stage 01 — Input Selection:</i> Drag-and-drop image upload, benchmark specimen presets, and dual operating modes (<b>Simulation Evaluation Mode</b> with ground-truth PSNR/heatmaps vs. <b>Blind Enhancement Mode</b> with correction maps for already compressed photos).</li>
        <li><i>Stage 02 — Configure Pipeline:</i> Interactive clickable 8&times;8 DCT grid ROI selector, Quality Factor / Cutoff slider, 4 architecture selectors, and a live Eq. 6 Hard Spectral Consistency lock switch.</li>
        <li><i>Stage 03 — Evaluation &amp; Results:</i> Synchronized split-slider comparison, 5-way visual comparison strip, 4-way "hot" residual error heatmaps, 4&times; zoomed loupe patches, 8&times;8 log-magnitude DCT spectrum plots, live PSNR/PSNR-B/SSIM cards, and one-click JSON/PNG exports.</li>
      </ol>
    </li>
    <li><b>Mobile Judge Access:</b> Built-in QR code generation paired with Cloudflare Tunnel (<code>cloudflared</code>) enables instant testing on smartphones alongside zero-lag offline execution on the presenter's laptop.</li>
  </ul>

  <!-- ==================== CHAPTER 7 ==================== -->
  <h1 class="chapter-title">Chapter 7: Conclusion &amp; Future Scope</h1>

  <h2>7.1 Summary of Contributions</h2>
  <ol>
    <li><b>Diagnosed Pure Frequency GNN Failure:</b> Proved empirically (+0.01 dB gain) that intra-block frequency graphs lack continuous boundary visibility across 8&times;8 borders.</li>
    <li><b>Identified Spectral Drift in <code>GNN &rarr; CNN</code>:</b> Showed that post-processing spatial convolutions inadvertently alter true preserved low-frequency coefficients (+1.68 dB avg Set5 gain).</li>
    <li><b>Engineered Dual-Domain <code>CNN &rarr; GNN</code> + Hard Spectral Lock:</b> Eliminated inter-block grid seams spatially before locking true low-frequency anchors and refining harmonic octaves—delivering up to <b>+4.13 dB</b> improvement on Set5 Butterfly and deploying a real-time interactive lab web application.</li>
  </ol>

  <h2>7.2 Future Scope</h2>
  <ul>
    <li>Extending the dual-domain harmonic lock to multi-scale attention architectures and adaptive ISO/IEC quantization tables.</li>
    <li>Real-time temporal video deblocking across H.264/HEVC intra-coded frames.</li>
  </ul>

  <!-- ==================== REFERENCES ==================== -->
  <h1 class="chapter-title">References</h1>
  <ol style="font-size:10.5pt;">
    <li>Sun, Y., et al. (2020). <i>CARD: Dual-Domain Compression Artifact Reduction via Wide Activation Residual Networks and Quantization Priors</i>.</li>
    <li>Han, K., Wang, Y., Guo, J., Tang, Y., &amp; Wu, E. (2022). <i>Vision GNN: An Image is Worth Graph of Nodes (ViG)</i>. Advances in Neural Information Processing Systems (NeurIPS).</li>
    <li>Dong, C., Deng, Y., Loy, C. C., &amp; Tang, X. (2015). <i>Compression Artifacts Reduction by a Deep Convolutional Network (ARCNN)</i>. IEEE ICCV.</li>
    <li>Zhang, K., Zuo, W., Chen, Y., Meng, D., &amp; Zhang, L. (2017). <i>Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising (DnCNN)</i>. IEEE Transactions on Image Processing.</li>
    <li>Yim, C., &amp; Bovik, A. C. (2011). <i>Quality Assessment of Deblocked Images (PSNR-B)</i>. IEEE Transactions on Multimedia.</li>
    <li>ISO/IEC 10918-1 (ITU-T Rec. T.81). <i>Information Technology — Digital Compression and Coding of Continuous-Tone Still Images (JPEG)</i>.</li>
    <li>Fey, M., &amp; Lenssen, J. E. (2019). <i>Fast Graph Representation Learning with PyTorch Geometric</i>. ICLR Workshop on Representation Learning on Graphs and Manifolds.</li>
  </ol>

</body>
</html>
"""

    out_html = os.path.join(REPORT_DIR, "project_report.html")
    try:
        with open(out_html, "w", encoding="utf-8") as f:
            f.write(html)
    except PermissionError:
        out_html = os.path.join(REPORT_DIR, "project_report_updated.html")
        with open(out_html, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"[INFO] project_report.html is open in Word; saved updated HTML to: {out_html}")

    # Remove old root project_report.html if present so there is no broken duplicate
    if os.path.exists("project_report.html"):
        os.remove("project_report.html")

    print(f"[SUCCESS] Word-compatible report folder created at: {out_html}")
    print(f"[SUCCESS] Total images linked inside {IMG_DIR}: {len(os.listdir(IMG_DIR))}")


if __name__ == "__main__":
    build_report()
