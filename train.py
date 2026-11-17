"""
Training Orchestrator for FrequencyGNN.

This script trains the Graph Neural Network to predict missing high-frequency
DCT coefficients from preserved low-frequency coefficients.

Workflow:
    1. Loads images from a training folder (e.g., datasets/BSDS500/train or Set5).
    2. Slices images into 8x8 blocks and converts them into 2D-DCT coefficients.
    3. Masks high frequencies in zigzag order (simulating JPEG compression).
    4. Packages the blocks into PyG graph batches using DCTGraphDataset.
    5. Trains FrequencyGNN using MSE loss strictly on missing high frequencies.
    6. Saves the best model checkpoint to checkpoints/model.pth.
"""

import os
import sys
import glob
import time
import argparse
import numpy as np
import cv2
from scipy.fft import dctn
import torch
import torch.nn as nn

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.dataset import get_zigzag_indices, get_dct_dataloader
from src.models import FrequencyGNN


# =============================================================================
# Image Slicing and 2D-DCT Processing
# =============================================================================

def process_single_image(image_path: str, cutoff: int = 15):
    """
    Reads an image, crops it to multiples of 8, slices into 8x8 blocks,
    applies 2D-DCT, and creates:
        - X_blocks: Truncated DCT blocks (high frequencies set to 0.0)
        - Y_blocks: Full ground truth DCT blocks

    Returns:
        X_blocks, Y_blocks as numpy arrays of shape (num_blocks, 64)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None

    h, w = img.shape
    h_crop, w_crop = (h // 8) * 8, (w // 8) * 8
    if h_crop == 0 or w_crop == 0:
        return None, None

    img = img[:h_crop, :w_crop].astype(np.float32) / 255.0

    zigzag = get_zigzag_indices()
    X_list, Y_list = [], []

    for r in range(0, h_crop, 8):
        for c in range(0, w_crop, 8):
            block = img[r:r+8, c:c+8]
            dct_block = dctn(block, norm='ortho')
            y_full = dct_block.flatten()

            # Create truncated block X (keep first cutoff in zigzag, rest zeroed)
            x_trunc_2d = np.zeros_like(dct_block)
            for idx in range(min(cutoff, 64)):
                zr, zc = zigzag[idx]
                x_trunc_2d[zr, zc] = dct_block[zr, zc]

            X_list.append(x_trunc_2d.flatten())
            Y_list.append(y_full)

    return np.array(X_list, dtype=np.float32), np.array(Y_list, dtype=np.float32)


def load_dataset_blocks(folder_path: str, cutoff: int = 15, max_images: int = None):
    """
    Scans a folder for images (.jpg, .png, .bmp), extracts all 8x8 DCT blocks,
    and returns combined X (truncated) and Y (ground truth) arrays.
    """
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_paths = []
    for p in patterns:
        image_paths.extend(glob.glob(os.path.join(folder_path, p)))
        image_paths.extend(glob.glob(os.path.join(folder_path, "**", p), recursive=True))

    image_paths = sorted(list(set(image_paths)))
    if max_images is not None and max_images > 0:
        image_paths = image_paths[:max_images]

    if not image_paths:
        raise ValueError(f"No images found in folder: {folder_path}")

    print(f"Loading {len(image_paths)} image(s) from {folder_path}...")
    all_X, all_Y = [], []

    for idx, path in enumerate(image_paths):
        X, Y = process_single_image(path, cutoff=cutoff)
        if X is not None and len(X) > 0:
            all_X.append(X)
            all_Y.append(Y)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(image_paths):
            print(f"  Processed [{idx + 1}/{len(image_paths)}] images...")

    all_X = np.concatenate(all_X, axis=0)
    all_Y = np.concatenate(all_Y, axis=0)
    print(f"Total 8x8 DCT blocks prepared: {len(all_X):,} blocks.")
    return all_X, all_Y


# =============================================================================
# Main Training Function
# =============================================================================

def train(args):
    # Determine device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # 1. Load Data
    X_train, Y_train = load_dataset_blocks(
        folder_path=args.train_folder,
        cutoff=args.cutoff,
        max_images=args.max_images
    )

    dataloader = get_dct_dataloader(
        X_blocks=X_train,
        Y_blocks=Y_train,
        cutoff=args.cutoff,
        batch_size=args.batch_size,
        shuffle=True
    )

    # 2. Instantiate Model, Loss, Optimizer, Scheduler
    model = FrequencyGNN(
        in_channels=3,
        hidden_channels=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)

    print("\n" + "=" * 60)
    print("Starting Training:")
    print(f"  Total Epochs:      {args.epochs}")
    print(f"  Batch Size:        {args.batch_size} blocks")
    print(f"  Learning Rate:     {args.lr}")
    print(f"  Cutoff Frequencies:{args.cutoff} (Predicting {64 - args.cutoff} missing ACs)")
    print(f"  Total Parameters:  {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("=" * 60 + "\n")

    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float("inf")
    start_time = time.time()

    # 3. Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass: predict DCT amplitudes for all nodes in the batch
            preds = model(batch.x, batch.edge_index)

            # Grade strictly on missing high-frequency nodes (missing_mask == True)
            loss = criterion(preds[batch.missing_mask], batch.y[batch.missing_mask])

            # Backpropagation
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        scheduler.step()
        epoch_loss = running_loss / max(num_batches, 1)

        # Check for best model
        is_best = epoch_loss < best_loss
        if is_best:
            best_loss = epoch_loss
            best_path = os.path.join(args.save_dir, "model.pth")
            torch.save(model.state_dict(), best_path)

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start_time
        star = " *" if is_best else ""
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] - Loss: {epoch_loss:.6f} - LR: {current_lr:.1e} - Time: {elapsed:.1f}s{star}")

    print("\n" + "=" * 60)
    print(f"[SUCCESS] Training complete in {time.time() - start_time:.1f}s!")
    print(f"Best Loss: {best_loss:.6f}")
    print(f"Saved weights to: {os.path.join(args.save_dir, 'model.pth')}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FrequencyGNN on DCT blocks.")
    parser.add_argument("--train_folder", type=str, default="datasets/Set5",
                        help="Folder containing training images (e.g., datasets/BSDS500/train or datasets/Set5).")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Directory where model weights will be saved.")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs (default: 5 for quick check, 30 for full training).")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Mini-batch size in terms of 8x8 blocks (default: 64).")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate (default: 0.001).")
    parser.add_argument("--hidden", type=int, default=128,
                        help="Hidden dimension size of GNN layers (default: 128).")
    parser.add_argument("--layers", type=int, default=3,
                        help="Number of GNN layers (default: 3).")
    parser.add_argument("--cutoff", type=int, default=15,
                        help="Number of preserved low-frequency coefficients in zigzag order (default: 15).")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout rate (default: 0.0).")
    parser.add_argument("--step_size", type=int, default=15,
                        help="Epoch interval for StepLR learning rate decay (default: 15).")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Optional cap on number of images to use (useful for fast debugging).")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device to train on ('cpu' or 'cuda').")

    args = parser.parse_args()
    train(args)
