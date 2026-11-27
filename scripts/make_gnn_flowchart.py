import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.makedirs("assets", exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]

def generate_gnn_vertical_flowchart(out_path="assets/ppt_gnn_only_flowchart.png"):
    fig, ax = plt.subplots(figsize=(6.4, 8.8), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # Palette matching the light-blue PPT theme
    NAVY = "#0F2942"
    BLUE = "#1D4ED8"
    BOX_BG = "#F8FAFC"
    BOX_BORDER = "#93C5FD"
    HIGHLIGHT_BG = "#EFF6FF"
    HIGHLIGHT_BORDER = "#1D4ED8"
    PILL_BG = "#E0E7FF"
    OUT_BG = "#FEF2F2"
    OUT_BORDER = "#F87171"
    OUT_TEXT = "#991B1B"

    cx = 3.2
    box_w = 5.2

    # 1. Top Pill: [Original Image]
    top_pill = patches.FancyBboxPatch(
        (cx - 1.6, 8.15), 3.2, 0.45,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        facecolor=PILL_BG, edgecolor=BLUE, linewidth=1.6
    )
    ax.add_patch(top_pill)
    ax.text(cx, 8.375, "Original Image", ha="center", va="center", fontsize=11.5, fontweight="bold", color=NAVY)

    # Blocks list: (y_center, height, title, bullets, bg, border, title_color)
    blocks = [
        (7.10, 0.90, "2D-DCT Transform (8×8 Blocks)",
         "• Converts pixels to 64 DCT coefficients",
         BOX_BG, BOX_BORDER, NAVY),

        (5.55, 1.05, "JPEG Truncation (Compression)",
         "• Keeps 15 Low Freqs (DC + 14 ACs)\n• Zeros out 49 High Freqs",
         "#FFFBEB", "#F59E0B", "#92400E"),

        (3.90, 1.05, "Frequency GNN (Intra-Block)",
         "• Tries to predict missing 49 ACs\n  from known 15 low frequencies",
         HIGHLIGHT_BG, HIGHLIGHT_BORDER, BLUE),

        (2.30, 0.90, "2D-Inverse DCT (IDCT) Transform",
         "• Converts frequencies back to pixels",
         BOX_BG, BOX_BORDER, NAVY),
    ]

    # Arrow from Top Pill to Block 1
    ax.annotate("", xy=(cx, 7.58), xytext=(cx, 8.12),
                arrowprops=dict(arrowstyle="-|>,head_width=0.25,head_length=0.3", color=BLUE, lw=2.0))

    for idx, (yc, bh, title, bullets, bg, border, tcol) in enumerate(blocks):
        rect = patches.FancyBboxPatch(
            (cx - box_w / 2, yc - bh / 2), box_w, bh,
            boxstyle="round,pad=0.05,rounding_size=0.14",
            facecolor=bg, edgecolor=border, linewidth=2.0
        )
        ax.add_patch(rect)

        ax.text(cx - box_w / 2 + 0.25, yc + bh / 2 - 0.28, title,
                ha="left", va="center", fontsize=11.5, fontweight="bold", color=tcol)
        ax.text(cx - box_w / 2 + 0.25, yc - 0.14, bullets,
                ha="left", va="center", fontsize=9.8, color="#1E293B", linespacing=1.4)

        # Draw arrow to next stage
        next_y_top = blocks[idx + 1][0] + blocks[idx + 1][1] / 2 + 0.05 if idx < len(blocks) - 1 else 1.25
        ax.annotate("", xy=(cx, next_y_top), xytext=(cx, yc - bh / 2 - 0.05),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.25,head_length=0.3", color=BLUE, lw=2.0))

    # Bottom Outcome Pill: [Restored Output Image] (+0.01 dB gain)
    bot_pill = patches.FancyBboxPatch(
        (cx - 2.35, 0.68), 4.7, 0.52,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        facecolor=OUT_BG, edgecolor=OUT_BORDER, linewidth=1.8
    )
    ax.add_patch(bot_pill)
    ax.text(cx, 0.94, "Restored Output Image  (+0.01 dB Gain)",
            ha="center", va="center", fontsize=11, fontweight="bold", color=OUT_TEXT)

    ax.set_xlim(0.2, 6.2)
    ax.set_ylim(0.45, 8.85)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close()
    print(f"[SUCCESS] Generated crisp GNN-only PPT flowchart at: {out_path}")


if __name__ == "__main__":
    generate_gnn_vertical_flowchart()
