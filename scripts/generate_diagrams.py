import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

os.makedirs("assets", exist_ok=True)

# Set high DPI and professional light blue styling
plt.rcParams['font.sans-serif'] = 'Segoe UI', 'DejaVu Sans', 'Arial'
plt.rcParams['font.family'] = 'sans-serif'

# Colors
C_DARK_BLUE = "#0F2942"
C_PRIMARY_BLUE = "#1D4ED8"
C_LIGHT_BLUE = "#EFF6FF"
C_ACCENT_BLUE = "#3B82F6"
C_BORDER_BLUE = "#93C5FD"
C_GRAY_BG = "#F8FAFC"
C_GRAY_TEXT = "#64748B"
C_SUCCESS = "#10B981"
C_WARNING = "#F59E0B"
C_ALERT = "#EF4444"

# ==============================================================================
# Diagram 1: JPEG Flaw & Truncation Mechanism
# ==============================================================================
def create_diagram_jpeg_flaw():
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=200)
    ax.set_facecolor("#FFFFFF")
    fig.patch.set_facecolor("#FFFFFF")
    
    # 1. 8x8 DCT Frequency Matrix Grid
    ax.text(2.0, 3.8, "8×8 DCT Spectrum (64 Coefficients)", fontsize=11, fontweight='bold', color=C_DARK_BLUE, ha='center')
    
    grid_x, grid_y = 0.5, 0.4
    cell_w, cell_h = 0.35, 0.35
    
    for r in range(8):
        for c in range(8):
            # Zigzag order approximation: r + c <= 4 gives ~15 coefficients
            is_kept = (r + c <= 4)
            color = "#DBEAFE" if is_kept else "#F1F5F9"
            edge = C_PRIMARY_BLUE if is_kept else "#CBD5E1"
            rect = patches.Rectangle((grid_x + c * cell_w, grid_y + (7 - r) * cell_h), cell_w, cell_h,
                                     facecolor=color, edgecolor=edge, linewidth=1.2)
            ax.add_patch(rect)
            
            if r == 0 and c == 0:
                ax.text(grid_x + c * cell_w + cell_w/2, grid_y + (7 - r) * cell_h + cell_h/2, "DC",
                        ha='center', va='center', fontsize=8, fontweight='bold', color=C_DARK_BLUE)
            elif is_kept:
                ax.text(grid_x + c * cell_w + cell_w/2, grid_y + (7 - r) * cell_h + cell_h/2, "AC",
                        ha='center', va='center', fontsize=7, color=C_PRIMARY_BLUE)
            else:
                ax.text(grid_x + c * cell_w + cell_w/2, grid_y + (7 - r) * cell_h + cell_h/2, "0",
                        ha='center', va='center', fontsize=7, color="#94A3B8")

    # Legend for grid
    rect_kept = patches.Rectangle((0.5, 0.05), 0.3, 0.2, facecolor="#DBEAFE", edgecolor=C_PRIMARY_BLUE)
    ax.add_patch(rect_kept)
    ax.text(0.9, 0.12, "15 Preserved Low Frequencies (DC + 14 ACs)", fontsize=8, color=C_DARK_BLUE, va='center')
    
    rect_zero = patches.Rectangle((2.3, 0.05), 0.3, 0.2, facecolor="#F1F5F9", edgecolor="#CBD5E1")
    ax.add_patch(rect_zero)
    ax.text(2.7, 0.12, "49 Truncated High Frequencies (Zeroed Out)", fontsize=8, color=C_GRAY_TEXT, va='center')

    # Arrow to Consequences
    arrow = patches.FancyArrowPatch((3.7, 2.0), (4.5, 2.0), arrowstyle='->', mutation_scale=20,
                                   color=C_PRIMARY_BLUE, linewidth=2.5)
    ax.add_patch(arrow)
    ax.text(4.1, 2.3, "2D-IDCT\nSynthesis", fontsize=8.5, ha='center', color=C_PRIMARY_BLUE, fontweight='bold')

    # Consequence 1: Blurry Edges
    c1 = patches.FancyBboxPatch((4.7, 2.2), 4.8, 1.4, boxstyle="round,pad=0.1,rounding_size=0.15",
                               facecolor="#FEF2F2", edgecolor="#FCA5A5", linewidth=1.5)
    ax.add_patch(c1)
    ax.text(4.9, 3.2, "Defect 1: Spectral Loss (Blurry Edges)", fontsize=10, fontweight='bold', color="#991B1B")
    ax.text(4.9, 2.65, "• Discarding 49 high AC frequencies eliminates fine texture,\n  sharp outlines, and high-contrast hair/eyelash detail.",
            fontsize=8.5, color="#7F1D1D")

    # Consequence 2: Grid Seams
    c2 = patches.FancyBboxPatch((4.7, 0.4), 4.8, 1.5, boxstyle="round,pad=0.1,rounding_size=0.15",
                               facecolor="#FFFBEB", edgecolor="#FCD34D", linewidth=1.5)
    ax.add_patch(c2)
    ax.text(4.9, 1.5, "Defect 2: Dirichlet Boundary Seams (Blocking)", fontsize=10, fontweight='bold', color="#92400E")
    ax.text(4.9, 0.85, "• Independent 8×8 block transforms cause step-discontinuities\n  at block boundaries, forming an ugly checkerboard grid.",
            fontsize=8.5, color="#78350F")

    ax.set_xlim(0, 9.8)
    ax.set_ylim(0, 4.2)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("assets/diagram_jpeg_flaw.png", dpi=250, bbox_inches='tight')
    plt.close()

# ==============================================================================
# Diagram 2: Methodology 1 (GNN Alone) & Border Blindness
# ==============================================================================
def create_diagram_gnn_pipeline():
    fig, ax = plt.subplots(figsize=(10, 4.0), dpi=200)
    ax.set_facecolor("#FFFFFF")
    fig.patch.set_facecolor("#FFFFFF")

    # Boxes for pipeline
    steps = [
        ("Input Image\n(Pixels)", 0.6, 2.0, "#EFF6FF", C_PRIMARY_BLUE),
        ("2D-DCT\nTransform", 2.4, 2.0, "#F8FAFC", "#64748B"),
        ("JPEG Truncation\n(15 Low / 49 Zero)", 4.2, 2.0, "#FEF2F2", "#EF4444"),
        ("Frequency GNN\n(Harmonic Graphs)", 6.2, 2.0, "#EFF6FF", C_PRIMARY_BLUE),
        ("2D-IDCT\nReconstruct", 8.2, 2.0, "#F8FAFC", "#64748B"),
    ]

    for title, x, y, bg, border in steps:
        box = patches.FancyBboxPatch((x-0.75, y-0.6), 1.5, 1.2, boxstyle="round,pad=0.1,rounding_size=0.12",
                                    facecolor=bg, edgecolor=border, linewidth=1.8)
        ax.add_patch(box)
        ax.text(x, y, title, ha='center', va='center', fontsize=8.5, fontweight='bold', color=C_DARK_BLUE)

    # Arrows between boxes
    for i in range(len(steps) - 1):
        x1 = steps[i][1] + 0.75
        x2 = steps[i+1][1] - 0.75
        arrow = patches.FancyArrowPatch((x1, 2.0), (x2, 2.0), arrowstyle='->', mutation_scale=15,
                                       color=C_PRIMARY_BLUE, linewidth=2)
        ax.add_patch(arrow)

    # Diagnosis Alert Box at Bottom
    alert = patches.FancyBboxPatch((0.6, 0.2), 8.4, 0.9, boxstyle="round,pad=0.1,rounding_size=0.12",
                                  facecolor="#FEF2F2", edgecolor="#F87171", linewidth=1.5)
    ax.add_patch(alert)
    ax.text(4.8, 0.75, "❌ Why GNN Alone Flatlined (+0.01 dB gain):", fontsize=9.5, fontweight='bold', color="#991B1B", ha='center')
    ax.text(4.8, 0.42, "Each 8×8 frequency graph lives strictly inside its own block. It has zero spatial connections to neighboring pixels,\nmaking it physically impossible for the GNN to detect or smooth inter-block boundary seams.",
            fontsize=8, color="#7F1D1D", ha='center')

    ax.set_xlim(-0.2, 9.8)
    ax.set_ylim(0, 3.4)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("assets/diagram_gnn_pipeline.png", dpi=250, bbox_inches='tight')
    plt.close()

# ==============================================================================
# Diagram 3: Methodology 2 (GNN -> CNN)
# ==============================================================================
def create_diagram_gnn_cnn_pipeline():
    fig, ax = plt.subplots(figsize=(10, 4.0), dpi=200)
    ax.set_facecolor("#FFFFFF")
    fig.patch.set_facecolor("#FFFFFF")

    # Workflow
    steps = [
        ("JPEG Truncated\nSpectrum", 0.8, 2.2, "#FEF2F2", "#EF4444"),
        ("STAGE 1: GNN\n(Frequency Domain)", 2.8, 2.2, "#EFF6FF", C_PRIMARY_BLUE),
        ("2D-IDCT\nTransform", 4.8, 2.2, "#F8FAFC", "#64748B"),
        ("STAGE 2: CNN\n(Spatial Deblocking)", 6.8, 2.2, "#EFF6FF", C_PRIMARY_BLUE),
        ("Restored Output\n(+1.58 dB)", 8.8, 2.2, "#ECFDF5", C_SUCCESS),
    ]

    for title, x, y, bg, border in steps:
        box = patches.FancyBboxPatch((x-0.8, y-0.6), 1.6, 1.2, boxstyle="round,pad=0.1,rounding_size=0.12",
                                    facecolor=bg, edgecolor=border, linewidth=1.8)
        ax.add_patch(box)
        ax.text(x, y, title, ha='center', va='center', fontsize=8.5, fontweight='bold', color=C_DARK_BLUE)

    for i in range(len(steps) - 1):
        x1 = steps[i][1] + 0.8
        x2 = steps[i+1][1] - 0.8
        arrow = patches.FancyArrowPatch((x1, 2.2), (x2, 2.2), arrowstyle='->', mutation_scale=15,
                                       color=C_PRIMARY_BLUE, linewidth=2)
        ax.add_patch(arrow)

    # Explanation of Spectral Drift
    warning = patches.FancyBboxPatch((0.8, 0.2), 8.8, 1.0, boxstyle="round,pad=0.1,rounding_size=0.12",
                                    facecolor="#FFFBEB", edgecolor="#FCD34D", linewidth=1.5)
    ax.add_patch(warning)
    ax.text(5.2, 0.85, "⚠️ The Limitation: 'Spectral Drift'", fontsize=9.5, fontweight='bold', color="#92400E", ha='center')
    ax.text(5.2, 0.50, "GNN fixes harmonics first, but when the spatial CNN deblocks seams at the end, it alters the fundamental low frequencies\nthat were originally 100% correct! This causes subtle color blurring and limits maximum PSNR.",
            fontsize=8, color="#78350F", ha='center')

    ax.set_xlim(-0.2, 10.4)
    ax.set_ylim(0, 3.4)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("assets/diagram_gnn_cnn_pipeline.png", dpi=250, bbox_inches='tight')
    plt.close()

# ==============================================================================
# Diagram 4: Methodology 3 (CNN -> GNN with Hard Spectral Lock) - WINNER
# ==============================================================================
def create_diagram_cnn_gnn_pipeline():
    fig, ax = plt.subplots(figsize=(10, 4.4), dpi=200)
    ax.set_facecolor("#FFFFFF")
    fig.patch.set_facecolor("#FFFFFF")

    # Pipeline nodes across two rows or sequence
    steps = [
        ("1. Truncated\nJPEG File", 0.7, 2.6, "#FEF2F2", "#EF4444"),
        ("1st IDCT\nDecompress", 2.2, 2.6, "#F8FAFC", "#64748B"),
        ("2. Spatial CNN\n(Macro Deblocking)", 3.9, 2.6, "#EFF6FF", C_PRIMARY_BLUE),
        ("2nd DCT\nRe-slice", 5.6, 2.6, "#F8FAFC", "#64748B"),
        ("3. GNN Refiner &\nHARD SPECTRAL LOCK", 7.4, 2.6, "#ECFDF5", C_SUCCESS),
        ("2nd IDCT\nFinal Image", 9.1, 2.6, "#EFF6FF", C_PRIMARY_BLUE),
    ]

    for title, x, y, bg, border in steps:
        box = patches.FancyBboxPatch((x-0.65, y-0.6), 1.3, 1.2, boxstyle="round,pad=0.1,rounding_size=0.12",
                                    facecolor=bg, edgecolor=border, linewidth=1.8)
        ax.add_patch(box)
        ax.text(x, y, title, ha='center', va='center', fontsize=8, fontweight='bold', color=C_DARK_BLUE)

    for i in range(len(steps) - 1):
        x1 = steps[i][1] + 0.65
        x2 = steps[i+1][1] - 0.65
        arrow = patches.FancyArrowPatch((x1, 2.6), (x2, 2.6), arrowstyle='->', mutation_scale=14,
                                       color=C_PRIMARY_BLUE, linewidth=2)
        ax.add_patch(arrow)

    # Breakthrough Callout Box
    success = patches.FancyBboxPatch((0.7, 0.2), 9.1, 1.25, boxstyle="round,pad=0.1,rounding_size=0.14",
                                    facecolor="#F0FDF4", edgecolor="#86EFAC", linewidth=2.0)
    ax.add_patch(success)
    ax.text(5.25, 1.15, "🏆 The Breakthrough: Spatial First, Spectral Second (Up to +4.13 dB gain!)",
            fontsize=10, fontweight='bold', color="#166534", ha='center')
    ax.text(5.25, 0.70, "1. CNN eliminates 8×8 block seams in the pixel domain and creates continuous edges.\n"
                        "2. 2nd DCT converts CNN image to frequencies -> GNN checks harmonic octave chains (u <-> 2u).\n"
                        "3. HARD SPECTRAL LOCK: The 15 true low-frequencies are clamped back to 100% exact truth (Zero Spectral Drift)!",
            fontsize=8.2, color="#14532D", ha='center')

    ax.set_xlim(-0.2, 10.6)
    ax.set_ylim(0, 3.8)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("assets/diagram_cnn_gnn_pipeline.png", dpi=250, bbox_inches='tight')
    plt.close()

# ==============================================================================
# Diagram 5: 72-Node Spatial-Frequency Graph Topology
# ==============================================================================
def create_diagram_graph_topology():
    fig, ax = plt.subplots(figsize=(10, 4.0), dpi=200)
    ax.set_facecolor("#FFFFFF")
    fig.patch.set_facecolor("#FFFFFF")

    # Center: 8x8 frequency block
    ax.text(2.5, 3.6, "Center Block: 64 Frequency Nodes", fontsize=10.5, fontweight='bold', color=C_DARK_BLUE, ha='center')
    
    # Draw small network nodes
    np.random.seed(42)
    coords = []
    for r in range(4):
        for c in range(4):
            x = 1.2 + c * 0.85
            y = 1.0 + (3 - r) * 0.65
            coords.append((x, y))
            color = C_PRIMARY_BLUE if (r + c <= 2) else "#94A3B8"
            circle = plt.Circle((x, y), 0.16, facecolor=color, edgecolor="#FFFFFF", linewidth=1.5, zorder=5)
            ax.add_patch(circle)

    # Connect harmonic octave edges (u <-> 2u)
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dist = np.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            if dist < 1.0:
                ax.plot([coords[i][0], coords[j][0]], [coords[i][1], coords[j][1]],
                        color="#BFDBFE", linewidth=1.2, zorder=3)

    # 8 Spatial Neighbor Nodes
    ax.text(7.2, 3.6, "8 Spatial Neighbor Super-Nodes", fontsize=10.5, fontweight='bold', color=C_DARK_BLUE, ha='center')
    
    nb_names = ["NW", "N", "NE", "W", "E", "SW", "S", "SE"]
    nb_offsets = [(-1, 1), (0, 1), (1, 1), (-1, 0), (1, 0), (-1, -1), (0, -1), (1, -1)]
    cx, cy = 7.2, 2.0
    
    for name, (dx, dy) in zip(nb_names, nb_offsets):
        nx = cx + dx * 1.3
        ny = cy + dy * 0.9
        rect = patches.FancyBboxPatch((nx-0.35, ny-0.25), 0.7, 0.5, boxstyle="round,pad=0.05,rounding_size=0.1",
                                     facecolor="#EFF6FF", edgecolor=C_PRIMARY_BLUE, linewidth=1.5, zorder=5)
        ax.add_patch(rect)
        ax.text(nx, ny, name, ha='center', va='center', fontsize=8, fontweight='bold', color=C_PRIMARY_BLUE, zorder=6)
        
        # Connect to center
        ax.plot([cx, nx], [cy, ny], color="#93C5FD", linestyle='--', linewidth=1.4, zorder=3)

    center_hub = patches.FancyBboxPatch((cx-0.5, cy-0.35), 1.0, 0.7, boxstyle="round,pad=0.05,rounding_size=0.1",
                                       facecolor=C_PRIMARY_BLUE, edgecolor="#FFFFFF", linewidth=2, zorder=7)
    ax.add_patch(center_hub)
    ax.text(cx, cy, "Center\nBlock", ha='center', va='center', fontsize=8, fontweight='bold', color="#FFFFFF", zorder=8)

    ax.text(5.0, 0.2, "Total: 72 Nodes (64 Frequency + 8 Spatial Context) | 652 Harmonic & Spatial Edges",
            fontsize=9.5, fontweight='bold', color=C_DARK_BLUE, ha='center')

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.0)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("assets/diagram_graph_topology.png", dpi=250, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    create_diagram_jpeg_flaw()
    create_diagram_gnn_pipeline()
    create_diagram_gnn_cnn_pipeline()
    create_diagram_cnn_gnn_pipeline()
    create_diagram_graph_topology()
    print("All technical diagram graphics created successfully in assets/!")
