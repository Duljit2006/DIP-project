import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

os.makedirs("assets", exist_ok=True)

# Load newly evaluated Set5 Butterfly 5-way comparison strip
comp = Image.open('results/Set5_img_003_SRF_2_comparison.png')
w, h = comp.size
panel_w = w // 5

p_gt = comp.crop((0 * panel_w, 0, 1 * panel_w, h))
p_base = comp.crop((1 * panel_w, 0, 2 * panel_w, h))
p_ours = comp.crop((4 * panel_w, 0, 5 * panel_w, h))

# Crop tight butterfly wing area inside panel (skipping white plot margins & titles)
cx, cy = panel_w // 2, int(h * 0.56)
crop_sz = 180
box = (cx - crop_sz//2, cy - crop_sz//2, cx + crop_sz//2, cy + crop_sz//2)

crop_gt = p_gt.crop(box).resize((320, 320), Image.NEAREST)
crop_base = p_base.crop(box).resize((320, 320), Image.NEAREST)
crop_ours = p_ours.crop(box).resize((320, 320), Image.NEAREST)

# Create 3-way zoomed figure in restrained Navy + Charcoal palette
fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), dpi=220)
fig.patch.set_facecolor("#FFFFFF")

axes[0].imshow(crop_base)
axes[0].set_title("1. Baseline JPEG (Cutoff 15)\nVisible 8×8 Checkerboard Seams", fontsize=9.5, fontweight='bold', color="#4B5563")
axes[0].axis('off')

axes[1].imshow(crop_ours)
axes[1].set_title("2. Proposed CNN ➔ GNN (+4.13 dB)\nSeams Eliminated & Sharp Contours", fontsize=9.5, fontweight='bold', color="#1E3A5F")
axes[1].axis('off')

axes[2].imshow(crop_gt)
axes[2].set_title("3. Ground Truth Reference\nOriginal Uncompressed", fontsize=9.5, fontweight='bold', color="#111827")
axes[2].axis('off')

plt.tight_layout()
plt.savefig("assets/zoomed_seam_patch.png", dpi=250, bbox_inches='tight')
plt.close()
print("Saved assets/zoomed_seam_patch.png")
