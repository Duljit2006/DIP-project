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
           Takes input features (known DCT value + 2D normalized coordinates [u, v])
           and projects them into a high-dimensional feature space.
        2. Message Passing Layers:
           Multiple GCNConv layers pass information between connected frequency
           nodes across spatial neighbor edges and harmonic octave edges.
        3. Residual / Skip Connections:
           Helps maintain gradient stability and retain low-frequency information.
        4. Linear Regression Head:
           Projects the final hidden representation of each node down to a single
           scalar: the predicted DCT coefficient amplitude.
    """
    def __init__(self, in_channels: int = 3, hidden_channels: int = 128, num_layers: int = 3, dropout: float = 0.0):
        """
        Args:
            in_channels: Input feature dimension per node. Default is 3
                         (1 for frequency amplitude, 2 for normalized [u, v] coordinates).
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
            
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        # Final output projection head: hidden_channels -> 1 scalar DCT amplitude
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward computation of the FrequencyGNN.
        
        Args:
            x: Node features of shape (num_nodes, in_channels).
               For a single block: (64, 3).
               For a batch of B blocks: (B * 64, 3).
            edge_index: Graph connectivity tensor of shape (2, num_edges).
            
        Returns:
            torch.Tensor of shape (num_nodes,) representing predicted DCT amplitudes.
        """
        h = x
        for i in range(self.num_layers):
            # Message passing across edges
            h_next = self.convs[i](h, edge_index)
            h_next = self.norms[i](h_next)
            h_next = self.relu(h_next)
            h_next = self.dropout(h_next)
            
            # Residual connection if dimensions match
            if h.shape == h_next.shape:
                h = h + h_next
            else:
                h = h_next

        # Predict scalar DCT coefficient for each node
        out = self.head(h)          # Shape: (num_nodes, 1)
        return out.squeeze(-1)      # Shape: (num_nodes,)


if __name__ == "__main__":
    import sys
    import os
    
    # Add project root to sys.path to allow running directly
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.graph_builder import build_frequency_graph, get_frequency_coordinates

    print("=" * 60)
    print("Testing src/models.py (FrequencyGNN)")
    print("=" * 60)

    # 1. Instantiate model
    model = FrequencyGNN(in_channels=3, hidden_channels=128, num_layers=3)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model architecture:\n{model}")
    print(f"\nTotal trainable parameters: {total_params:,}")

    # 2. Test single block inference
    print("\n[Test 1: Single Block (64 nodes)]")
    edge_index = build_frequency_graph()
    coords = get_frequency_coordinates(normalize=True)  # (64, 2)
    
    # Simulate a truncated block where low frequencies (first 15) are known, rest are 0.0
    amplitudes = torch.zeros(64, 1)
    amplitudes[:15] = torch.randn(15, 1) * 100.0  # Simulated low frequency DCT values
    
    # Combine values + coordinates into node features: (64, 3)
    node_features = torch.cat([amplitudes, coords], dim=1)
    
    # Forward pass
    output = model(node_features, edge_index)
    print(f"Input node features shape:  {node_features.shape}")
    print(f"Graph edge index shape:     {edge_index.shape}")
    print(f"Output predictions shape:   {output.shape} (Expected: torch.Size([64]))")
    assert output.shape == torch.Size([64]), "Output shape must be (64,) for a single block!"

    # 3. Test batched inference (e.g. 4 blocks = 256 nodes)
    print("\n[Test 2: Batched Blocks (4 blocks = 256 nodes)]")
    from torch_geometric.data import Data, Batch
    
    data_list = []
    for _ in range(4):
        data_list.append(Data(x=node_features, edge_index=edge_index))
    
    batch = Batch.from_data_list(data_list)
    batch_output = model(batch.x, batch.edge_index)
    print(f"Batch input node features:  {batch.x.shape} (Expected: torch.Size([256, 3]))")
    print(f"Batch edge index:           {batch.edge_index.shape}")
    print(f"Batch output predictions:   {batch_output.shape} (Expected: torch.Size([256]))")
    assert batch_output.shape == torch.Size([256]), "Batch output must match total node count!"

    print("\n[SUCCESS] FrequencyGNN successfully validated for single and batched inputs!")
