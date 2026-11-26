import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

os.makedirs("assets", exist_ok=True)

# Load comparison image
comp = Image.open('results/Set14_img_001_SRF_2_comparison.png')
w, h = comp.size
panel_w = w // 5

# Extract panels
# Panel 0: Ground Truth
# Panel 1: Baseline JPEG
# Panel 2: GNN Alone
# Panel 3: GNN -> CNN
# Panel 4: CNN -> GNN
p_gt = comp.crop((0 * panel_w, 0, 1 * panel_w, h))
p_base = comp.crop((1 * panel_w, 0, 2 * panel_w, h))
p_ours = comp.crop((4 * panel_w, 0, 5 * panel_w, h))

# Pick a high-detail patch (e.g., center-ish region)
cx, cy = panel_w // 2, h // 2
crop_sz = 140
box = (cx - crop_sz//2, cy - crop_sz//2, cx + crop_sz//2, cy + crop_sz//2)

crop_gt = p_gt.crop(box).resize((320, 320), Image.NEAREST)
crop_base = p_base.crop(box).resize((320, 320), Image.NEAREST)
crop_ours = p_ours.crop(box).resize((320, 320), Image.NEAREST)

# Create 3-way zoomed figure
fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), dpi=200)
fig.patch.set_facecolor("#FFFFFF")

axes[0].imshow(crop_base)
axes[0].set_title("1. Baseline JPEG (Cutoff 15)\nSevere 8×8 Grid Checkerboard", fontsize=9, fontweight='bold', color="#991B1B")
axes[0].axis('off')

axes[1].imshow(crop_ours)
axes[1].set_title("2. CNN ➔ GNN (Proposed)\nSeams Eliminated & Edges Sharp", fontsize=9, fontweight='bold', color="#166534")
axes[1].axis('off')

axes[2].imshow(crop_gt)
axes[2].set_title("3. Ground Truth Reference\nOriginal Uncompressed", fontsize=9, fontweight='bold', color="#0F2942")
axes[2].axis('off')

plt.tight_layout()
plt.savefig("assets/zoomed_seam_patch.png", dpi=250, bbox_inches='tight')
plt.close()
print("Saved assets/zoomed_seam_patch.png")
