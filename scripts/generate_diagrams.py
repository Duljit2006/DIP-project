"""
scripts/generate_diagrams.py - Clean Academic Diagrams (Navy & Slate Monochrome)
-------------------------------------------------------------------------------
Zero overlapping boxes, mathematically consistent uniform arrow gaps & heads,
zero emoji glyphs, and a restrained professional print palette (#1E3A5F Navy + Slate).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.makedirs("assets", exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Calibri", "Segoe UI", "DejaVu Sans", "Arial"]
plt.rcParams["font.family"] = "sans-serif"

# Restrained Professional Print Palette (Only Navy, Charcoal, and Neutral Slates)
C_NAVY        = "#1E3A5F"
C_DARK_TEXT   = "#111827"
C_MUTED_TEXT  = "#4B5563"
C_SLATE_EDGE  = "#64748B"
C_BOX_GRAY    = "#F8FAFC"
C_BOX_NAVY_BG = "#EFF6FF"
C_CELL_KEPT   = "#DBEAFE"
C_CELL_ZERO   = "#F1F5F9"
C_GRID_LINE   = "#CBD5E1"


def draw_pipeline_row(ax, steps, y_center=2.55, box_w=1.45, box_h=1.15, gap=0.48):
    """
    Draws a perfectly spaced horizontal row of boxes with identical gap widths
    and identical arrow head geometries (pad=0.0 prevents box bloat overlap).
    """
    n = len(steps)
    total_w = n * box_w + (n - 1) * gap
    start_x = (10.8 - total_w) / 2.0

    for i, (title, sub, is_highlight) in enumerate(steps):
        bx = start_x + i * (box_w + gap)
        by = y_center - box_h / 2.0
        bg = C_BOX_NAVY_BG if is_highlight else C_BOX_GRAY
        ec = C_NAVY if is_highlight else C_SLATE_EDGE
        lw = 1.9 if is_highlight else 1.4

        box = patches.FancyBboxPatch(
            (bx, by), box_w, box_h,
            boxstyle="round,pad=0.0,rounding_size=0.10",
            facecolor=bg, edgecolor=ec, linewidth=lw, zorder=4
        )
        ax.add_patch(box)

        # Title + Subtitle cleanly centered inside box
        ax.text(bx + box_w / 2.0, y_center + 0.16, title,
                ha="center", va="center", fontsize=8.5, fontweight="bold",
                color=C_NAVY if is_highlight else C_DARK_TEXT, zorder=5)
        if sub:
            ax.text(bx + box_w / 2.0, y_center - 0.24, sub,
                    ha="center", va="center", fontsize=7.5,
                    color=C_MUTED_TEXT, zorder=5)

        # Consistent Arrow to next box
        if i < n - 1:
            ax_start = bx + box_w + 0.05
            ax_end   = bx + box_w + gap - 0.05
            ax.annotate(
                "", xy=(ax_end, y_center), xytext=(ax_start, y_center),
                arrowprops=dict(
                    arrowstyle="-|>,head_width=0.22,head_length=0.26",
                    color=C_NAVY, lw=1.8, shrinkA=0, shrinkB=0
                ),
                zorder=6
            )

    return start_x, total_w


def draw_note_banner(ax, x_left, width, y_bottom, height, heading, body, is_highlight=False):
    bg = C_BOX_NAVY_BG if is_highlight else C_BOX_GRAY
    ec = C_NAVY if is_highlight else C_SLATE_EDGE
    banner = patches.FancyBboxPatch(
        (x_left, y_bottom), width, height,
        boxstyle="round,pad=0.0,rounding_size=0.10",
        facecolor=bg, edgecolor=ec, linewidth=1.4, zorder=4
    )
    ax.add_patch(banner)
    ax.text(x_left + width / 2.0, y_bottom + height - 0.28, heading,
            ha="center", va="center", fontsize=9.2, fontweight="bold", color=C_NAVY, zorder=5)
    ax.text(x_left + width / 2.0, y_bottom + (height - 0.38) / 2.0, body,
            ha="center", va="center", fontsize=8.0, color=C_DARK_TEXT, linespacing=1.35, zorder=5)


# ==============================================================================
# Diagram 1: JPEG Flaw & Truncation Mechanism (Monochrome Navy + Gray)
# ==============================================================================
def create_diagram_jpeg_flaw():
    fig, ax = plt.subplots(figsize=(10.2, 4.1), dpi=250)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    ax.text(2.15, 3.85, "8×8 DCT Spectrum (64 Coefficients)", fontsize=10.5, fontweight="bold", color=C_NAVY, ha="center")

    gx, gy = 0.65, 0.65
    cw, ch = 0.36, 0.36
    for r in range(8):
        for c in range(8):
            is_kept = (r + c <= 4)
            fc = C_CELL_KEPT if is_kept else C_CELL_ZERO
            ec = C_NAVY if is_kept else C_GRID_LINE
            rect = patches.Rectangle((gx + c * cw, gy + (7 - r) * ch), cw, ch,
                                     facecolor=fc, edgecolor=ec, linewidth=1.0)
            ax.add_patch(rect)
            cx = gx + c * cw + cw / 2.0
            cy = gy + (7 - r) * ch + ch / 2.0
            if r == 0 and c == 0:
                ax.text(cx, cy, "DC", ha="center", va="center", fontsize=7.5, fontweight="bold", color=C_NAVY)
            elif is_kept:
                ax.text(cx, cy, "AC", ha="center", va="center", fontsize=6.5, color=C_NAVY)
            else:
                ax.text(cx, cy, "0", ha="center", va="center", fontsize=6.5, color="#94A3B8")

    # Legend under matrix
    ax.add_patch(patches.Rectangle((0.65, 0.20), 0.25, 0.18, facecolor=C_CELL_KEPT, edgecolor=C_NAVY))
    ax.text(0.98, 0.29, "15 Preserved Low Freqs", fontsize=7.8, color=C_DARK_TEXT, va="center")
    ax.add_patch(patches.Rectangle((2.50, 0.20), 0.25, 0.18, facecolor=C_CELL_ZERO, edgecolor=C_GRID_LINE))
    ax.text(2.83, 0.29, "49 Zeroed High Freqs", fontsize=7.8, color=C_MUTED_TEXT, va="center")

    # Consistent Arrow
    ax.annotate("", xy=(4.65, 2.05), xytext=(3.75, 2.05),
                arrowprops=dict(arrowstyle="-|>,head_width=0.24,head_length=0.28", color=C_NAVY, lw=2.0, shrinkA=0, shrinkB=0))
    ax.text(4.20, 2.38, "2D-IDCT\nSynthesis", fontsize=8.5, ha="center", color=C_NAVY, fontweight="bold")

    # Two clean Defect Boxes (Navy + Neutral Slate)
    d1 = patches.FancyBboxPatch((4.85, 2.15), 4.85, 1.35, boxstyle="round,pad=0.0,rounding_size=0.10",
                                facecolor=C_BOX_GRAY, edgecolor=C_NAVY, linewidth=1.4)
    ax.add_patch(d1)
    ax.text(5.10, 3.18, "Defect 1: Spectral Information Loss (Blur)", fontsize=9.5, fontweight="bold", color=C_NAVY)
    ax.text(5.10, 2.58, "• Discarding 49 high AC frequencies removes fine texture,\n  sharp outlines, and high-contrast edge details.",
            fontsize=8.2, color=C_DARK_TEXT, linespacing=1.35)

    d2 = patches.FancyBboxPatch((4.85, 0.55), 4.85, 1.35, boxstyle="round,pad=0.0,rounding_size=0.10",
                                facecolor=C_BOX_NAVY_BG, edgecolor=C_NAVY, linewidth=1.4)
    ax.add_patch(d2)
    ax.text(5.10, 1.58, "Defect 2: Dirichlet Boundary Seams (8×8 Blocking)", fontsize=9.5, fontweight="bold", color=C_NAVY)
    ax.text(5.10, 0.98, "• Independent 8×8 block transforms lack border continuity,\n  creating artificial step-jumps (checkerboard grid lines).",
            fontsize=8.2, color=C_DARK_TEXT, linespacing=1.35)

    ax.set_xlim(0.2, 10.0)
    ax.set_ylim(0.0, 4.15)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("assets/diagram_jpeg_flaw.png", dpi=250, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close()


# ==============================================================================
# Diagram 2: Methodology 1 (Pure Frequency GNN)
# ==============================================================================
def create_diagram_gnn_pipeline():
    fig, ax = plt.subplots(figsize=(10.8, 3.6), dpi=250)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    steps = [
        ("Input Image", "(Spatial Pixels)", False),
        ("2D-DCT", "(8×8 Blocks)", False),
        ("JPEG Truncation", "(15 Low / 49 Zero)", False),
        ("Frequency GNN", "(Harmonic Graph)", True),
        ("2D-IDCT", "(Reconstruct Pixels)", False),
    ]
    sx, tw = draw_pipeline_row(ax, steps, y_center=2.55, box_w=1.55, box_h=1.12, gap=0.48)
    draw_note_banner(
        ax, sx, tw, 0.30, 1.15,
        "DIAGNOSIS: Why Pure GNN Achieved Only +0.00 to +0.01 dB Gain",
        "Each 8×8 frequency graph operates strictly inside its own isolated transform block.\n"
        "Without continuous spatial filters spanning pixel borders, the GNN cannot eliminate 8×8 grid seams.",
        is_highlight=False
    )

    ax.set_xlim(0.0, 10.8)
    ax.set_ylim(0.1, 3.4)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("assets/diagram_gnn_pipeline.png", dpi=250, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close()


# ==============================================================================
# Diagram 3: Methodology 2 (Hybrid GNN -> CNN)
# ==============================================================================
def create_diagram_gnn_cnn_pipeline():
    fig, ax = plt.subplots(figsize=(10.8, 3.6), dpi=250)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    steps = [
        ("Truncated DCT", "(15 Low Freqs)", False),
        ("STAGE 1: GNN", "(Freq Domain)", True),
        ("1st IDCT", "(To Pixels)", False),
        ("STAGE 2: CNN", "(Spatial Deblock)", True),
        ("Restored Output", "(+1.68 dB Set5)", False),
    ]
    sx, tw = draw_pipeline_row(ax, steps, y_center=2.55, box_w=1.55, box_h=1.12, gap=0.48)
    draw_note_banner(
        ax, sx, tw, 0.30, 1.15,
        "LIMITATION: Spatial Post-Processing Causes 'Spectral Drift'",
        "Putting the Spatial CNN at the end removes grid seams, but 3×3 convolutions alter ALL pixel intensities\n"
        "indiscriminately—corrupting the 15 fundamental low-frequency DCT coefficients that were already exact.",
        is_highlight=False
    )

    ax.set_xlim(0.0, 10.8)
    ax.set_ylim(0.1, 3.4)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("assets/diagram_gnn_cnn_pipeline.png", dpi=250, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close()


# ==============================================================================
# Diagram 4: Methodology 3 (Proposed CNN -> GNN + Eq. 6)
# ==============================================================================
def create_diagram_cnn_gnn_pipeline():
    fig, ax = plt.subplots(figsize=(10.8, 3.85), dpi=250)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    steps = [
        ("1. Truncated JPEG", "(15 Low Freqs)", False),
        ("1st IDCT", "(To Pixels)", False),
        ("2. Spatial CNN", "(Deblock Seams)", True),
        ("2nd DCT", "(Re-Slice 8×8)", False),
        ("3. GNN + Eq. 6", "(Lock Low Freqs)", True),
        ("2nd IDCT", "(Final Output)", False),
    ]
    sx, tw = draw_pipeline_row(ax, steps, y_center=2.80, box_w=1.35, box_h=1.12, gap=0.40)
    draw_note_banner(
        ax, sx, tw, 0.28, 1.45,
        "BREAKTHROUGH: Spatial First, Spectral Second (+1.95 dB Avg / +4.13 dB Set5 Butterfly)",
        "1. Spatial CNN eliminates 8×8 boundary seams first and establishes continuous visual edges.\n"
        "2. 2nd DCT translates smoothed pixels to frequency space; GNN refines harmonic octaves (u <-> 2u).\n"
        "3. HARD SPECTRAL LOCK (Eq. 6): Clamps true 15 low frequencies back to ground truth (Zero Spectral Drift).",
        is_highlight=True
    )

    ax.set_xlim(0.0, 10.8)
    ax.set_ylim(0.1, 3.6)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("assets/diagram_cnn_gnn_pipeline.png", dpi=250, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close()


# ==============================================================================
# Diagram 5: 72-Node Spatial-Frequency Graph Topology (Monochrome Navy + Gray)
# ==============================================================================
def create_diagram_graph_topology():
    fig, ax = plt.subplots(figsize=(10.4, 4.0), dpi=250)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    ax.text(2.5, 3.65, "Center Block: 64 Frequency Nodes", fontsize=10.2, fontweight="bold", color=C_NAVY, ha="center")

    coords = []
    for r in range(4):
        for c in range(4):
            x = 1.2 + c * 0.85
            y = 1.0 + (3 - r) * 0.65
            coords.append((x, y))
            color = C_NAVY if (r + c <= 2) else "#94A3B8"
            circle = plt.Circle((x, y), 0.16, facecolor=color, edgecolor="#FFFFFF", linewidth=1.4, zorder=5)
            ax.add_patch(circle)

    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if np.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1]) < 1.0:
                ax.plot([coords[i][0], coords[j][0]], [coords[i][1], coords[j][1]],
                        color="#93C5FD", linewidth=1.2, zorder=3)

    ax.text(7.4, 3.65, "8 Spatial Neighbor Super-Nodes", fontsize=10.2, fontweight="bold", color=C_NAVY, ha="center")

    nb_names = ["NW", "N", "NE", "W", "E", "SW", "S", "SE"]
    nb_offsets = [(-1, 1), (0, 1), (1, 1), (-1, 0), (1, 0), (-1, -1), (0, -1), (1, -1)]
    cx, cy = 7.4, 2.0

    for name, (dx, dy) in zip(nb_names, nb_offsets):
        nx = cx + dx * 1.45
        ny = cy + dy * 0.90
        rect = patches.FancyBboxPatch((nx - 0.34, ny - 0.24), 0.68, 0.48,
                                      boxstyle="round,pad=0.0,rounding_size=0.08",
                                      facecolor=C_BOX_NAVY_BG, edgecolor=C_NAVY, linewidth=1.3, zorder=5)
        ax.add_patch(rect)
        ax.text(nx, ny, name, ha="center", va="center", fontsize=8.2, fontweight="bold", color=C_NAVY, zorder=6)
        ax.plot([cx, nx], [cy, ny], color=C_SLATE_EDGE, linestyle="--", linewidth=1.2, zorder=3)

    center_hub = patches.FancyBboxPatch((cx - 0.52, cy - 0.34), 1.04, 0.68,
                                        boxstyle="round,pad=0.0,rounding_size=0.10",
                                        facecolor=C_NAVY, edgecolor="#FFFFFF", linewidth=1.8, zorder=7)
    ax.add_patch(center_hub)
    ax.text(cx, cy, "Center\nBlock", ha="center", va="center", fontsize=8.2, fontweight="bold", color="#FFFFFF", zorder=8)

    ax.text(5.2, 0.22, "Total: 72 Nodes (64 Frequency + 8 Spatial Context) | 652 Harmonic & Spatial Edges",
            fontsize=9.2, fontweight="bold", color=C_NAVY, ha="center")

    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 4.0)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("assets/diagram_graph_topology.png", dpi=250, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close()


if __name__ == "__main__":
    create_diagram_jpeg_flaw()
    create_diagram_gnn_pipeline()
    create_diagram_gnn_cnn_pipeline()
    create_diagram_cnn_gnn_pipeline()
    create_diagram_graph_topology()
    print("[SUCCESS] All 5 monochrome academic diagrams regenerated with zero overlap & uniform arrows!")
