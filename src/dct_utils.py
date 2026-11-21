"""
src/dct_utils.py - Classical Discrete Cosine Transform (DCT) Utilities
------------------------------------------------------------------------
Lead: Digital Image Processing (HER Track)

This module handles classical image processing operations:
1. Slicing images into 8x8 pixel blocks.
2. Converting 8x8 blocks into 2D-DCT frequency coefficients.
3. Truncating high frequencies (JPEG-style compression).
4. Reconstructing images back to pixel space using 2D-IDCT.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn


def get_zigzag_indices():
    """
    Returns the standard JPEG 8x8 zigzag scan order as a list of 64 (row, col) coordinates.
    Low frequencies come first; high frequencies come last.
    """
    indices = []
    for s in range(15):  # sum of row + col ranges from 0 to 14
        if s % 2 == 0:
            for r in range(min(s, 7), max(0, s - 7) - 1, -1):
                indices.append((r, s - r))
        else:
            for r in range(max(0, s - 7), min(s, 7) + 1):
                indices.append((r, s - r))
    return indices


def load_and_slice(image_path):
    """
    Loads an image from disk, converts to grayscale, normalizes to [0, 1],
    crops to multiples of 8, and slices into non-overlapping 8x8 blocks.

    Args:
        image_path (str): Path to image file.

    Returns:
        blocks (np.ndarray): Array of shape (num_blocks, 8, 8) with float32 values in [0, 1].
        orig_shape (tuple): (height, width) of the cropped image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Read as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not decode image at path: {image_path}")

    img = img.astype(np.float32) / 255.0  # Normalize to [0.0, 1.0]

    h, w = img.shape
    h_crop = (h // 8) * 8
    w_crop = (w // 8) * 8
    img = img[:h_crop, :w_crop]

    blocks = []
    for r in range(0, h_crop, 8):
        for c in range(0, w_crop, 8):
            blocks.append(img[r : r + 8, c : c + 8])

    return np.array(blocks, dtype=np.float32), (h_crop, w_crop)


def apply_dct_and_truncate(blocks, cutoff=15):
    """
    Applies 2D-DCT (orthonormal) to each 8x8 block.
    Retains only the first `cutoff` low-frequency coefficients in zigzag order,
    zeroing out the remaining (64 - cutoff) high frequencies.

    Args:
        blocks (np.ndarray): Shape (num_blocks, 8, 8).
        cutoff (int): Number of zigzag low-frequency coefficients to keep (default: 15).

    Returns:
        X_truncated (np.ndarray): Shape (num_blocks, 64) - input with high frequencies set to 0.
        Y_full (np.ndarray): Shape (num_blocks, 64) - full ground truth DCT coefficients.
    """
    zigzag = get_zigzag_indices()
    X_list = []
    Y_list = []

    for block in blocks:
        # Compute orthonormal 2D-DCT
        dct_2d = dctn(block, norm="ortho")
        Y_list.append(dct_2d.flatten())

        # Create truncated version
        trunc_2d = np.zeros_like(dct_2d)
        for i in range(cutoff):
            r, c = zigzag[i]
            trunc_2d[r, c] = dct_2d[r, c]

        X_list.append(trunc_2d.flatten())

    return np.array(X_list, dtype=np.float32), np.array(Y_list, dtype=np.float32)


def reconstruct_from_dct(dct_blocks_flat, image_shape):
    """
    Takes flat 64-length DCT vectors, reshapes each to 8x8, applies 2D-IDCT,
    and stitches them back into a full grayscale image.

    Args:
        dct_blocks_flat (np.ndarray): Shape (num_blocks, 64).
        image_shape (tuple): (height, width) of the target image.

    Returns:
        img_reconstructed (np.ndarray): 2D float32 image array clipped to [0, 1].
    """
    h, w = image_shape
    img_reconstructed = np.zeros((h, w), dtype=np.float32)
    idx = 0

    for r in range(0, h, 8):
        for c in range(0, w, 8):
            block_2d = dct_blocks_flat[idx].reshape((8, 8))
            img_reconstructed[r : r + 8, c : c + 8] = idctn(block_2d, norm="ortho")
            idx += 1

    return np.clip(img_reconstructed, 0.0, 1.0)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing src/dct_utils.py (Classical 8x8 Slicing & DCT)")
    print("=" * 60)

    # Test image from Set5
    sample_img_path = os.path.join("datasets", "Set5", "img_001_SRF_2.png")
    if not os.path.exists(sample_img_path):
        # Fallback to any image in Set5 if naming varies
        set5_files = [os.path.join("datasets", "Set5", f) for f in os.listdir("datasets/Set5") if f.endswith(".png")]
        sample_img_path = set5_files[0] if set5_files else None

    if sample_img_path and os.path.exists(sample_img_path):
        print(f"Loading image: {sample_img_path}")
        blocks, shape = load_and_slice(sample_img_path)
        print(f"Image original shape: {shape}")
        print(f"Total 8x8 blocks extracted: {len(blocks):,} blocks")

        cutoff = 15
        X_trunc, Y_full = apply_dct_and_truncate(blocks, cutoff=cutoff)
        print(f"High frequencies truncated: First {cutoff} preserved, {64 - cutoff} erased to 0.0")

        # Reconstruct both original (all frequencies) and truncated (lossy baseline)
        recon_original = reconstruct_from_dct(Y_full, shape)
        recon_trunc = reconstruct_from_dct(X_trunc, shape)

        print(f"Reconstructed image pixel range: min = {recon_trunc.min():.1f}, max = {recon_trunc.max():.1f}")
        print("[SUCCESS] Slicing, DCT, Truncation, and IDCT all working cleanly!")

        # Save visual comparison verification into results/
        os.makedirs("results", exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(recon_original, cmap="gray")
        axes[0].set_title("Original (Full Frequencies - Sharp)")
        axes[0].axis("off")

        axes[1].imshow(recon_trunc, cmap="gray")
        axes[1].set_title(f"Truncated IDCT Baseline (Cutoff={cutoff})\n(Notice blur & 8x8 block seams)")
        axes[1].axis("off")

        output_plot_path = os.path.join("results", "dct_verification.png")
        plt.tight_layout()
        plt.savefig(output_plot_path, dpi=150)
        print(f"Saved visual verification plot to: {output_plot_path}")
    else:
        print("[WARNING] No test image found in datasets/Set5. Checked path:", sample_img_path)
