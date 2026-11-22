"""
Frequency Graph Builder for 8x8 DCT Blocks.

This module constructs the relational graph representing the 64 frequency
coefficients in an 8x8 Discrete Cosine Transform (DCT) block.

Nodes:
    - 64 nodes (indices 0 to 63), where index = u * 8 + v.
    - (u, v) represents the horizontal and vertical spatial frequency coordinates.

Edges:
    1. Spatial Adjacency: Connects neighboring frequency coordinates (horizontal,
       vertical, and diagonal) reflecting smooth transitions in frequency space.
    2. Harmonic Relationships: Connects fundamental low frequencies (u, v) to
       higher harmonics like (2u, 2v), (2u, v), (u, 2v) within the 8x8 grid.
    3. Bidirectional: All edges are bidirectional to enable 2-way message passing.
"""

import torch


def coord_to_idx(u: int, v: int) -> int:
    """Converts 2D frequency coordinate (u, v) to flat node index (0 to 63)."""
    return u * 8 + v


def idx_to_coord(idx: int) -> tuple:
    """Converts flat node index (0 to 63) to 2D frequency coordinate (u, v)."""
    return idx // 8, idx % 8


def get_frequency_coordinates(normalize: bool = True) -> torch.Tensor:
    """
    Returns 2D (u, v) spatial frequency coordinates for all 64 nodes.
    
    Args:
        normalize: If True, coordinates are divided by 7.0 to scale into [0, 1].
        
    Returns:
        torch.Tensor of shape (64, 2)
    """
    coords = []
    for u in range(8):
        for v in range(8):
            coords.append([u, v])
    coords_tensor = torch.tensor(coords, dtype=torch.float32)
    if normalize:
        coords_tensor = coords_tensor / 7.0
    return coords_tensor


def build_frequency_graph() -> torch.Tensor:
    """
    Constructs the PyTorch Geometric edge_index for an 8x8 DCT block.
    
    Connects:
        - Local frequency neighbors (8-neighborhood)
        - Harmonic frequency pairs ((u, v) <-> (2u, 2v), (2u, v), (u, 2v))
        
    Returns:
        edge_index: LongTensor of shape (2, num_edges), unique, bidirectional.
    """
    edges = set()

    for u in range(8):
        for v in range(8):
            src = coord_to_idx(u, v)

            # -----------------------------------------------------------------
            # 1. Local Frequency Neighbors (8-neighborhood)
            # -----------------------------------------------------------------
            neighbor_deltas = [
                (0, 1),   # Right
                (1, 0),   # Down
                (1, 1),   # Down-Right diagonal
                (1, -1),  # Down-Left diagonal
            ]
            for du, dv in neighbor_deltas:
                nu, nv = u + du, v + dv
                if 0 <= nu < 8 and 0 <= nv < 8:
                    dst = coord_to_idx(nu, nv)
                    edges.add((src, dst))
                    edges.add((dst, src))  # Bidirectional

            # -----------------------------------------------------------------
            # 2. Harmonic Frequency Relationships
            # Connect fundamental frequencies (u, v) to octave harmonics:
            # - (2u, 2v): Diagonal octave harmonic
            # - (2u, v) : Vertical harmonic
            # - (u, 2v) : Horizontal harmonic
            # -----------------------------------------------------------------
            harmonics = []
            if u > 0 and 2 * u < 8:
                harmonics.append((2 * u, v))
            if v > 0 and 2 * v < 8:
                harmonics.append((u, 2 * v))
            if u > 0 and v > 0 and 2 * u < 8 and 2 * v < 8:
                harmonics.append((2 * u, 2 * v))

            for hu, hv in harmonics:
                dst = coord_to_idx(hu, hv)
                if dst != src:
                    edges.add((src, dst))
                    edges.add((dst, src))  # Bidirectional

    # Convert unique edges to tensor of shape (2, num_edges)
    edge_list = sorted(list(edges))
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    return edge_index


def build_multiblock_spatial_frequency_graph() -> torch.Tensor:
    """
    Constructs the 72-node Multi-Block Spatial-Frequency Graph:
      - Nodes 0..63: 64 intra-block DCT frequency coefficient nodes
      - Nodes 64..71: 8 inter-block spatial boundary neighbor nodes:
          64: North (Top)
          65: South (Bottom)
          66: West (Left)
          67: East (Right)
          68: North-West
          69: North-East
          70: South-West
          71: South-East

    Edges:
      1. Intra-block: All 500 local harmonic and spatial frequency edges.
      2. Inter-block Cross-domain:
         - N (64) & S (65) connect to vertical variation frequency nodes (u, 0) and (u, 1) for u=1..7.
         - W (66) & E (67) connect to horizontal variation frequency nodes (0, v) and (1, v) for v=1..7.
         - Diagonals (68..71) connect to diagonal frequency nodes (d, d) for d=1..5.

    Returns:
        edge_index: LongTensor of shape (2, 652), bidirectional.
    """
    base_edges = build_frequency_graph()
    edges = set()
    for i in range(base_edges.shape[1]):
        edges.add((base_edges[0, i].item(), base_edges[1, i].item()))

    # Connect N (64) and S (65) to vertical frequency nodes (u, 0) and (u, 1)
    for u in range(1, 8):
        for v in [0, 1]:
            dst = coord_to_idx(u, v)
            edges.add((64, dst))
            edges.add((dst, 64))
            edges.add((65, dst))
            edges.add((dst, 65))

    # Connect W (66) and E (67) to horizontal frequency nodes (0, v) and (1, v)
    for v in range(1, 8):
        for u in [0, 1]:
            dst = coord_to_idx(u, v)
            edges.add((66, dst))
            edges.add((dst, 66))
            edges.add((67, dst))
            edges.add((dst, 67))

    # Connect Diagonals to diagonal frequency nodes (d, d)
    for d in range(1, 6):
        dst = coord_to_idx(d, d)
        for nid in [68, 69, 70, 71]:
            edges.add((nid, dst))
            edges.add((dst, nid))

    edge_list = sorted(list(edges))
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index


def get_multiblock_coordinates(normalize: bool = True) -> torch.Tensor:
    """
    Returns 2D normalized (u, v) coordinates for all 72 nodes.
    - Nodes 0..63: [u / 7.0, v / 7.0]
    - Nodes 64..71: Normalized directional offsets of the 8 neighbors:
        N  (-1,  0) -> [-0.5,  0.0]
        S  ( 1,  0) -> [ 0.5,  0.0]
        W  ( 0, -1) -> [ 0.0, -0.5]
        E  ( 0,  1) -> [ 0.0,  0.5]
        NW (-1, -1) -> [-0.5, -0.5]
        NE (-1,  1) -> [-0.5,  0.5]
        SW ( 1, -1) -> [ 0.5, -0.5]
        SE ( 1,  1) -> [ 0.5,  0.5]
    """
    freq_coords = get_frequency_coordinates(normalize=normalize)  # (64, 2)
    neighbor_offsets = torch.tensor([
        [-0.5,  0.0],  # 64: N
        [ 0.5,  0.0],  # 65: S
        [ 0.0, -0.5],  # 66: W
        [ 0.0,  0.5],  # 67: E
        [-0.5, -0.5],  # 68: NW
        [-0.5,  0.5],  # 69: NE
        [ 0.5, -0.5],  # 70: SW
        [ 0.5,  0.5],  # 71: SE
    ], dtype=torch.float32)
    return torch.cat([freq_coords, neighbor_offsets], dim=0)  # (72, 2)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing src/graph_builder.py")
    print("=" * 60)

    edge_index = build_frequency_graph()
    coords = get_frequency_coordinates()

    print(f"Edge index shape:     {edge_index.shape}")
    print(f"Total directed edges: {edge_index.shape[1]}")
    print(f"Node count:           64 (Nodes 0 to 63)")
    print(f"Coordinates shape:    {coords.shape}")
    print(f"Min node index:       {edge_index.min().item()}")
    print(f"Max node index:       {edge_index.max().item()}")

    # Verify all 64 nodes have at least one connection
    unique_nodes = torch.unique(edge_index)
    print(f"Connected nodes:      {len(unique_nodes)} / 64")
    assert len(unique_nodes) == 64, "All 64 nodes must be connected!"

    # Show sample harmonic edge
    u, v = 1, 2
    src = coord_to_idx(u, v)
    dst = coord_to_idx(2 * u, 2 * v)
    is_harmonic_connected = ((edge_index[0] == src) & (edge_index[1] == dst)).any().item()
    print(f"Sample Harmonic edge (1,2) -> (2,4): Connected = {is_harmonic_connected}")

    print("\n[SUCCESS] Graph Builder successfully validated!")
