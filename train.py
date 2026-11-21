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

# Prevent Python from writing bytecode cache (.pyc)
sys.dont_write_bytecode = True

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Immediately delete previous results and bytecode caches before execution
from src.cleanup import clean_all
clean_all(clear_results=True)

import glob
import time
import argparse
import numpy as np
import cv2
from scipy.fft import dctn
import torch
import torch.nn as nn

from src.device_utils import get_optimal_device
from src.dataset import get_zigzag_indices, get_dct_dataloader, create_zigzag_mask
from src.dct_utils import load_and_slice, apply_dct_and_truncate, reconstruct_from_dct
from src.models import FrequencyGNN, SpatialRefinerCNN, DualDomainHybridModel


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
    and returns:
        valid_paths: list of paths to loaded images
        all_X: array of truncated blocks (num_blocks, 64)
        all_Y: array of ground truth blocks (num_blocks, 64)
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
    valid_paths = []
    all_X, all_Y = [], []

    for idx, path in enumerate(image_paths):
        X, Y = process_single_image(path, cutoff=cutoff)
        if X is not None and len(X) > 0:
            valid_paths.append(path)
            all_X.append(X)
            all_Y.append(Y)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(image_paths):
            print(f"  Processed [{idx + 1}/{len(image_paths)}] images...")

    all_X = np.concatenate(all_X, axis=0)
    all_Y = np.concatenate(all_Y, axis=0)
    print(f"Total 8x8 DCT blocks prepared: {len(all_X):,} blocks from {len(valid_paths)} images.")
    return valid_paths, all_X, all_Y


# =============================================================================
# Stage 2: Spatial Domain Refiner Training
# =============================================================================

def train_spatial_refiner(gnn_model, image_paths, cutoff=15, epochs=15, lr=5e-4, device="cpu", save_dir="checkpoints"):
    """
    Stage 2: Trains SpatialRefinerCNN on full images reconstructed from GNN predictions
    to eliminate inter-block boundary seams (JPEG blocking artifacts).
    """
    print("\n" + "=" * 60)
    print("STAGE 2: Training Spatial Refiner CNN (Inter-Block Deblocking)")
    print("=" * 60)

    gnn_model.eval()
    spatial_cnn = SpatialRefinerCNN(in_channels=1, hidden_channels=64, num_layers=4).to(device)
    optimizer = torch.optim.Adam(spatial_cnn.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 2, 1), gamma=0.5)
    criterion = nn.L1Loss()

    # Pre-generate GNN-restored images and clean targets
    print(f"Synthesizing intermediate GNN restorations for {len(image_paths)} training images...")
    dataset_pairs = []
    missing_mask = create_zigzag_mask(cutoff=cutoff).numpy()

    with torch.no_grad():
        for path in image_paths:
            blocks, shape = load_and_slice(path)
            if blocks is None or len(blocks) == 0:
                continue
            X_trunc, Y_full = apply_dct_and_truncate(blocks, cutoff=cutoff)
            dl = get_dct_dataloader(X_trunc, Y_full, cutoff=cutoff, batch_size=128, shuffle=False)
            preds = []
            for b in dl:
                b = b.to(device)
                p = gnn_model(b.x, b.edge_index)
                preds.append(p.cpu().numpy().reshape(-1, 64))
            preds = np.concatenate(preds, axis=0)
            X_restored = X_trunc.copy()
            X_restored[:, missing_mask] = preds[:, missing_mask]
            gnn_img = reconstruct_from_dct(X_restored, shape)

            clean_raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            clean_img = clean_raw[:shape[0], :shape[1]]

            gnn_tensor = torch.from_numpy(gnn_img).float().unsqueeze(0).unsqueeze(0).to(device)
            clean_tensor = torch.from_numpy(clean_img).float().unsqueeze(0).unsqueeze(0).to(device)
            dataset_pairs.append((gnn_tensor, clean_tensor))

    print(f"Prepared {len(dataset_pairs)} spatial image pairs for Stage 2.")
    print(f"Training SpatialRefinerCNN for {epochs} epochs...")

    best_loss = float("inf")
    start_time = time.time()
    for ep in range(1, epochs + 1):
        spatial_cnn.train()
        ep_loss = 0.0
        for gnn_tensor, clean_tensor in dataset_pairs:
            optimizer.zero_grad()
            refined = spatial_cnn(gnn_tensor)
            loss = criterion(refined, clean_tensor)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()

        scheduler.step()
        avg_loss = ep_loss / max(len(dataset_pairs), 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            cnn_save_path = os.path.join(save_dir, "spatial_refiner.pth")
            torch.save(spatial_cnn.state_dict(), cnn_save_path)
        print(f"  Stage 2 Epoch [{ep:02d}/{epochs:02d}] - L1 Loss: {avg_loss:.6f}")

    print(f"[SUCCESS] Spatial Refiner trained in {time.time() - start_time:.1f}s (Best L1 Loss: {best_loss:.6f})")

    # Save combined hybrid checkpoint
    hybrid_path = os.path.join(save_dir, "hybrid_model.pth")
    torch.save({
        "gnn_state_dict": gnn_model.state_dict(),
        "cnn_state_dict": spatial_cnn.state_dict(),
        "cutoff": cutoff,
    }, hybrid_path)
    print(f"Saved Dual-Domain Hybrid model to: {hybrid_path}\n")
    return spatial_cnn


# =============================================================================
# Main Training Function
# =============================================================================

def train(args):
    # Intelligent device selection (Dedicated GPU -> Integrated GPU -> CPU)
    device, device_info = get_optimal_device(preference=args.device, verbose=True)

    # 1. Load Data
    valid_paths, X_train, Y_train = load_dataset_blocks(
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

    # 2. Instantiate Model, Loss, Optimizer, Scheduler (STAGE 1: Frequency GNN)
    model = FrequencyGNN(
        in_channels=4,
        hidden_channels=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout
    ).to(device)

    # L1 Loss ensures strong, constant gradients to predict non-zero high-frequency textures
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)

    print("\n" + "=" * 60)
    print("STAGE 1: Training Frequency GNN (Intra-Block Harmonics)")
    print(f"  Total Epochs:      {args.epochs}")
    print(f"  Batch Size:        {args.batch_size} blocks")
    print(f"  Learning Rate:     {args.lr}")
    print(f"  Cutoff Frequencies:{args.cutoff} (Predicting {64 - args.cutoff} missing ACs)")
    print(f"  Loss Function:     L1 Loss (Mean Absolute Error)")
    print(f"  Total Parameters:  {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("=" * 60 + "\n")

    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float("inf")
    start_time = time.time()

    # 3. Training Loop for GNN
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass: predict DCT amplitudes for all nodes in the batch
            preds = model(batch.x, batch.edge_index)

            # Grade strictly on missing high-frequency nodes (missing_mask == True) with L1 loss
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
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] - L1 Loss: {epoch_loss:.6f} - LR: {current_lr:.1e} - Time: {elapsed:.1f}s{star}")

    print("\n" + "=" * 60)
    print(f"[SUCCESS] Stage 1 GNN Training complete in {time.time() - start_time:.1f}s!")
    print(f"Best GNN L1 Loss: {best_loss:.6f}")
    print(f"Saved weights to: {os.path.join(args.save_dir, 'model.pth')}")
    print("=" * 60)

    # Load best GNN weights for Stage 2
    model.load_state_dict(torch.load(os.path.join(args.save_dir, "model.pth"), map_location=device))

    # 4. Optional Stage 2: Spatial Refiner CNN (Dual-Domain Hybrid)
    if not args.no_hybrid:
        train_spatial_refiner(
            gnn_model=model,
            image_paths=valid_paths,
            cutoff=args.cutoff,
            epochs=args.cnn_epochs,
            lr=5e-4,
            device=device,
            save_dir=args.save_dir
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Dual-Domain Hybrid Model (FrequencyGNN + SpatialRefinerCNN).")
    parser.add_argument("--train_folder", type=str, default="datasets/Set5",
                        help="Folder containing training images (e.g., datasets/BSDS500/train or datasets/Set5).")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Directory where model weights will be saved.")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of Stage 1 GNN training epochs (default: 30).")
    parser.add_argument("--cnn_epochs", type=int, default=15,
                        help="Number of Stage 2 Spatial CNN training epochs (default: 15).")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Mini-batch size in terms of 8x8 blocks (default: 128 for GPU).")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate for GNN (default: 0.001).")
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
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to train on ('auto' for priority: Dedicated GPU > Integrated GPU > CPU, or 'cuda', 'mps', 'xpu', 'directml', 'cpu').")
    parser.add_argument("--no_hybrid", action="store_true",
                        help="Train only the Stage 1 Frequency GNN without the Stage 2 Spatial Refiner.")

    args = parser.parse_args()
    train(args)
