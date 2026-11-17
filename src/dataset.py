"""
PyTorch Geometric Dataset and DataLoader for DCT Blocks.

This module defines DCTGraphDataset, which packages truncated 8x8 DCT blocks
into PyTorch Geometric graph Data objects. Each graph represents an 8x8 block
with 64 nodes, its 500 edges from graph_builder, and masks identifying which
nodes are the missing high frequencies that the model needs to predict.
"""

import os
import sys
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

# Ensure src modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.graph_builder import build_frequency_graph, get_frequency_coordinates


def get_zigzag_indices():
    """
    Returns standard JPEG zigzag scan order for an 8x8 block as a list of (row, col) tuples.
    Zigzag scan groups frequencies from lowest (top-left) to highest (bottom-right).
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


def create_zigzag_mask(cutoff: int = 15) -> torch.Tensor:
    """
    Creates a boolean mask of shape (64,) where:
        False = Preserved Low Frequencies (first `cutoff` in zigzag order)
        True  = Missing High Frequencies (to be predicted by GNN)
    """
    zigzag = get_zigzag_indices()
    mask_2d = np.ones((8, 8), dtype=bool)  # True = missing
    
    # First `cutoff` positions in zigzag order are preserved
    for idx in range(min(cutoff, 64)):
        r, c = zigzag[idx]
        mask_2d[r, c] = False  # False = known / not missing
        
    return torch.tensor(mask_2d.flatten(), dtype=torch.bool)


class DCTGraphDataset(Dataset):
    """
    PyTorch Geometric Dataset for 8x8 DCT Frequency Blocks.
    
    Each item in the dataset is a torch_geometric.data.Data object with:
        - x: (64, 3) Node features [amplitude, u_coord, v_coord]
        - edge_index: (2, 500) Shared frequency graph connectivity
        - y: (64,) Ground truth untouched DCT coefficients
        - missing_mask: (64,) Boolean tensor (True = missing high-frequency node)
    """
    def __init__(self, X_blocks, Y_blocks, cutoff: int = 15):
        """
        Args:
            X_blocks: Array or Tensor of shape (N, 64) with truncated DCT values
                      (high frequencies zeroed out).
            Y_blocks: Array or Tensor of shape (N, 64) with full ground truth DCT values.
            cutoff: Number of low-frequency coefficients preserved in zigzag scan.
        """
        super().__init__()
        
        # Convert inputs to float32 tensors
        if isinstance(X_blocks, np.ndarray):
            self.X = torch.from_numpy(X_blocks).float()
        else:
            self.X = X_blocks.float()
            
        if isinstance(Y_blocks, np.ndarray):
            self.Y = torch.from_numpy(Y_blocks).float()
        else:
            self.Y = Y_blocks.float()
            
        assert len(self.X) == len(self.Y), "X and Y must have the exact same number of blocks!"
        assert self.X.shape[1] == 64 and self.Y.shape[1] == 64, "Blocks must contain 64 coefficients!"

        self.cutoff = cutoff
        
        # Precompute static graph topology shared across all 8x8 blocks
        self.edge_index = build_frequency_graph()                  # Shape: (2, 500)
        self.coords = get_frequency_coordinates(normalize=True)    # Shape: (64, 2)
        self.missing_mask = create_zigzag_mask(cutoff=cutoff)      # Shape: (64,)

    def len(self) -> int:
        return len(self.X)

    def get(self, idx: int) -> Data:
        """
        Constructs and returns a single Graph Data object for block at index `idx`.
        """
        # Node amplitude values: shape (64, 1)
        amp = self.X[idx].unsqueeze(1)
        
        # Node features: concatenate [amplitude, u_norm, v_norm] -> shape (64, 3)
        node_features = torch.cat([amp, self.coords], dim=1)
        
        # Ground truth target: shape (64,)
        target = self.Y[idx]

        return Data(
            x=node_features,
            edge_index=self.edge_index,
            y=target,
            missing_mask=self.missing_mask
        )


def get_dct_dataloader(X_blocks, Y_blocks, cutoff: int = 15, batch_size: int = 64, shuffle: bool = True):
    """
    Convenience helper function to create a PyTorch Geometric DataLoader.
    
    Args:
        X_blocks: Truncated DCT blocks (N, 64)
        Y_blocks: Ground-truth DCT blocks (N, 64)
        cutoff: Zigzag cutoff count
        batch_size: Number of 8x8 blocks per mini-batch
        shuffle: Whether to shuffle during training
        
    Returns:
        torch_geometric.loader.DataLoader
    """
    dataset = DCTGraphDataset(X_blocks, Y_blocks, cutoff=cutoff)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing src/dataset.py (DCTGraphDataset & DataLoader)")
    print("=" * 60)

    # 1. Test Zigzag Mask Creation
    cutoff = 15
    mask = create_zigzag_mask(cutoff=cutoff)
    num_missing = mask.sum().item()
    num_preserved = (~mask).sum().item()
    print(f"Zigzag cutoff:       {cutoff}")
    print(f"Preserved low nodes: {num_preserved} / 64")
    print(f"Missing high nodes:  {num_missing} / 64 (to be predicted)")
    assert num_preserved == cutoff, f"Preserved nodes ({num_preserved}) must equal cutoff ({cutoff})!"
    assert num_missing == 64 - cutoff, f"Missing nodes ({num_missing}) must equal 64 - cutoff!"

    # 2. Simulate 100 blocks of synthetic data
    print("\n[Simulating 100 DCT blocks...]")
    num_test_blocks = 100
    
    # Ground truth: random frequency amplitudes
    Y_synthetic = np.random.randn(num_test_blocks, 64).astype(np.float32) * 50.0
    
    # Input X: zero out high frequencies using the mask
    X_synthetic = Y_synthetic.copy()
    X_synthetic[:, mask.numpy()] = 0.0

    # 3. Create Dataset
    dataset = DCTGraphDataset(X_synthetic, Y_synthetic, cutoff=cutoff)
    print(f"Dataset length:      {len(dataset)} blocks")

    # Inspect a single item
    sample_data = dataset[0]
    print(f"Single graph x:      {sample_data.x.shape} (Expected: torch.Size([64, 3]))")
    print(f"Single graph edges:  {sample_data.edge_index.shape} (Expected: torch.Size([2, 500]))")
    print(f"Single graph y:      {sample_data.y.shape} (Expected: torch.Size([64]))")
    print(f"Missing mask count:  {sample_data.missing_mask.sum().item()} missing nodes")

    # 4. Test Mini-Batching with DataLoader
    batch_size = 16
    loader = get_dct_dataloader(X_synthetic, Y_synthetic, cutoff=cutoff, batch_size=batch_size, shuffle=True)
    first_batch = next(iter(loader))

    print(f"\n[Batching Test (batch_size = {batch_size})]")
    print(f"Batch x shape:       {first_batch.x.shape} (Expected: torch.Size([{batch_size * 64}, 3]))")
    print(f"Batch edge_index:    {first_batch.edge_index.shape} (Expected: torch.Size([2, {batch_size * 500}]))")
    print(f"Batch y shape:       {first_batch.y.shape} (Expected: torch.Size([{batch_size * 64}]))")
    print(f"Batch graphs count:  {first_batch.num_graphs} (Expected: {batch_size})")

    assert first_batch.x.shape == torch.Size([batch_size * 64, 3])
    assert first_batch.edge_index.shape == torch.Size([2, batch_size * 500])
    assert first_batch.num_graphs == batch_size

    print("\n[SUCCESS] DCTGraphDataset and DataLoader successfully validated!")
