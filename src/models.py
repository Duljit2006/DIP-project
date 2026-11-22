"""
GNN Architecture for DCT High-Frequency Coefficient Prediction.

This module defines the FrequencyGNN model using Graph Convolutional Networks (GCN).
The model receives the 64 frequency nodes in an 8x8 DCT block and predicts the
missing high-frequency values using message passing along harmonic and spatial edges.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class FrequencyGNN(nn.Module):
    """
    Graph Neural Network that predicts missing high-frequency DCT coefficients.
    
    Architecture:
        1. Node Feature Projection:
           Takes 4 input features (known DCT amplitude, normalized [u, v] coordinates,
           and is_known binary mask flag) and projects them into hidden space.
        2. Message Passing Layers:
           Multiple GCNConv layers pass information between connected frequency
           nodes across spatial neighbor edges and harmonic octave edges.
        3. Residual / Skip Connections:
           Helps maintain gradient stability and retain low-frequency information.
        4. Linear Regression Head:
           Projects the final hidden representation of each node down to a single
           scalar: the predicted DCT coefficient amplitude.
    """
    def __init__(self, in_channels: int = 4, hidden_channels: int = 128, num_layers: int = 3, dropout: float = 0.0):
        """
        Args:
            in_channels: Input feature dimension per node (default: 4 -> [amp, u, v, is_known]).
            hidden_channels: Hidden feature dimension for each GCN layer.
            num_layers: Total number of message-passing GCN layers.
            dropout: Dropout probability between layers (default 0.0 for regression).
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First GCN Layer: in_channels -> hidden_channels
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.norms.append(nn.LayerNorm(hidden_channels))
        
        # Intermediate GCN Layers: hidden_channels -> hidden_channels
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.norms.append(nn.LayerNorm(hidden_channels))
            
        self.act = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        # Final output projection head: hidden_channels -> 1 scalar DCT amplitude
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_channels // 2, 1)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, return_freq_only: bool = True) -> torch.Tensor:
        """
        Forward computation of the FrequencyGNN.
        
        Args:
            x: Node features of shape (num_nodes, in_channels).
               For a single block: (64, in_channels) or (72, in_channels).
               For a batch of B blocks: (B * 64, in_channels) or (B * 72, in_channels).
            edge_index: Graph connectivity tensor of shape (2, num_edges).
            return_freq_only: If True and input is 72-node multi-block, returns only
                              the 64 frequency node predictions of shape (B * 64,).
            
        Returns:
            torch.Tensor of shape (B * 64,) representing predicted DCT amplitudes.
        """
        h = x
        for i in range(self.num_layers):
            # Message passing across edges
            h_next = self.convs[i](h, edge_index)
            h_next = self.norms[i](h_next)
            h_next = self.act(h_next)
            h_next = self.dropout(h_next)
            
            # Residual connection if dimensions match
            if h.shape == h_next.shape:
                h = h + h_next
            else:
                h = h_next

        # Predict scalar DCT coefficient for each node
        out = self.head(h).squeeze(-1)  # Shape: (num_nodes,)

        # If 72-node multi-block graph is used and return_freq_only is True, slice first 64 nodes per block
        if return_freq_only and (out.shape[0] % 72 == 0):
            return out.view(-1, 72)[:, :64].reshape(-1)

        return out


class FrequencyWeightedL1Loss(nn.Module):
    """
    Component 2: Frequency-Weighted L1 Loss.
    
    Assigns higher penalty weights to the first missing mid-frequency harmonics (positions 15..30),
    which carry 90%+ of the visible edge structure and contrast energy, and decaying weights
    to high-frequency noise coefficients.
    """
    def __init__(self, cutoff: int = 15, decay_power: float = 0.5):
        super().__init__()
        self.cutoff = cutoff
        ranks = torch.arange(1, 65 - cutoff, dtype=torch.float32)
        raw_weights = 1.0 / (ranks ** decay_power)
        # Normalize weights so mean is 1.0 (maintaining loss scale)
        self.register_buffer("weights", raw_weights / raw_weights.mean())

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: Predicted DCT coefficients of shape (B * 64,) or (B, 64)
            targets: Ground truth DCT coefficients of shape (B * 64,) or (B, 64)
            missing_mask: Boolean tensor of shape (64,) or (B * 64,) where True = missing
        """
        if missing_mask.shape[0] != 64:
            mask_64 = missing_mask[:64]
        else:
            mask_64 = missing_mask

        p_missing = preds.reshape(-1, 64)[:, mask_64]    # (B, 49)
        t_missing = targets.reshape(-1, 64)[:, mask_64]  # (B, 49)
        err = torch.abs(p_missing - t_missing)           # (B, 49)
        weighted_err = err * self.weights.unsqueeze(0)   # (B, 49)
        return weighted_err.mean()


class SpatialRefinerCNN(nn.Module):
    """
    Stage 2: Spatial Domain Deblocking & Boundary Refinement Network.
    
    A multi-scale residual CNN that operates on the full reconstructed H x W image
    to eliminate 8x8 block boundary seams (JPEG grid artifacts) and enforce spatial
    continuity across neighboring frequency blocks.
    """
    def __init__(self, in_channels: int = 1, hidden_channels: int = 64, num_layers: int = 4):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1))
        layers.append(nn.PReLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1))
            layers.append(nn.PReLU())
        layers.append(nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Spatial image tensor of shape (B, 1, H, W) or (B, C, H, W).
        Returns:
            Refined spatial image tensor of shape (B, 1, H, W).
        """
        return x + self.net(x)


class DualDomainHybridModel(nn.Module):
    """
    Unified Dual-Domain Restoration Model combining:
      1. FrequencyGNN: Intra-block harmonic DCT high-frequency restoration.
      2. SpatialRefinerCNN: Inter-block boundary deblocking and spatial refinement.

    Supports two directional configurations:
      - GNN -> CNN: Frequency GNN restores missing ACs, then Spatial CNN deblocks.
      - CNN -> GNN: Spatial CNN deblocks first, then 2D-DCT projects to frequency domain
                    with Hard Spectral Consistency (locking true low frequencies).
    """
    def __init__(self, gnn: FrequencyGNN, cnn: SpatialRefinerCNN):
        super().__init__()
        self.gnn = gnn
        self.cnn = cnn

    def forward_gnn_then_cnn(self, gnn_Y: torch.Tensor) -> torch.Tensor:
        """Applies spatial refinement CNN on top of GNN reconstructed luminance."""
        return self.cnn(gnn_Y)

    def forward_cnn_then_spectral(self, base_Y: torch.Tensor) -> torch.Tensor:
        """Applies spatial refinement CNN on baseline JPEG luminance."""
        return self.cnn(base_Y)


if __name__ == "__main__":
    import sys
    import os
    
    # Add project root to sys.path to allow running directly
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.graph_builder import build_frequency_graph, get_frequency_coordinates

    print("=" * 60)
    print("Testing src/models.py (FrequencyGNN & SpatialRefinerCNN)")
    print("=" * 60)

    # 1. Instantiate GNN
    model = FrequencyGNN(in_channels=4, hidden_channels=128, num_layers=3)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nFrequencyGNN Trainable parameters: {total_params:,}")

    # 2. Test single block inference
    print("\n[Test 1: GNN Single Block (64 nodes)]")
    edge_index = build_frequency_graph()
    coords = get_frequency_coordinates(normalize=True)  # (64, 2)
    is_known = torch.ones(64, 1)
    is_known[15:] = 0.0
    amplitudes = torch.zeros(64, 1)
    amplitudes[:15] = torch.randn(15, 1) * 2.0
    
    # Combine values + coordinates + mask flag: (64, 4)
    node_features = torch.cat([amplitudes, coords, is_known], dim=1)
    output = model(node_features, edge_index)
    print(f"Input features shape:  {node_features.shape}")
    print(f"Output predictions:    {output.shape} (Expected: torch.Size([64]))")
    assert output.shape == torch.Size([64]), "Output shape must be (64,)!"

    # 3. Test SpatialRefinerCNN
    print("\n[Test 2: SpatialRefinerCNN on Full Image (1, 1, 128, 128)]")
    cnn = SpatialRefinerCNN(in_channels=1, hidden_channels=64, num_layers=4)
    cnn_params = sum(p.numel() for p in cnn.parameters() if p.requires_grad)
    print(f"SpatialRefinerCNN Trainable parameters: {cnn_params:,}")
    dummy_img = torch.rand(1, 1, 128, 128)
    cnn_out = cnn(dummy_img)
    print(f"CNN Input shape:  {dummy_img.shape}")
    print(f"CNN Output shape: {cnn_out.shape}")
    assert cnn_out.shape == dummy_img.shape, "CNN output shape must match input shape!"

    print("\n[SUCCESS] Both FrequencyGNN and SpatialRefinerCNN validated successfully!")
