# 📂 Project Files Documentation & Explanations

This document keeps a record of each file in our project, explaining what it does, the problem it solves, how it works step-by-step, and what outputs it produces.

---

## 1. `src/graph_builder.py`

### In Simple Words
`src/graph_builder.py` builds the roadmap (the "network of roads") that the Graph Neural Network (GNN) uses to communicate and pass clues between frequency coefficients.

---

### The Problem It Solves
When an $8 \times 8$ image tile is converted into DCT (Discrete Cosine Transform), you get **64 numbers** arranged in an $8 \times 8$ table:

```text
[ Low Frequencies (Known) ] ──────► [ High Frequencies (Erased / 0) ]
        (Top-Left)                                (Bottom-Right)
```

JPEG compression **erased the bottom-right numbers**. 

A Graph Neural Network (GNN) cannot guess those numbers out of thin air. It needs **lines (edges)** connecting the known numbers to the missing numbers so clues can travel across. 

**`src/graph_builder.py` creates those lines!**

---

### What It Does Step-by-Step

1. **Turns the 64 positions into 64 "Nodes" (Numbered 0 to 63):**
   - It numbers every position in the $8 \times 8$ frequency grid from left to right, top to bottom.
   - Position `(0, 0)` is Node `0` (the DC brightness).
   - Position `(7, 7)` is Node `63` (the highest frequency).

2. **Draws "Neighbor Roads" (Spatial Edges):**
   - Frequencies that sit right next to each other in the grid are related.
   - It connects every node to its 8 surrounding neighbors (up, down, left, right, diagonals).
   - *Why?* If frequency $(1, 1)$ has a certain pattern, $(1, 2)$ right next to it will have a very similar pattern.

3. **Draws "Harmonic Roads" (The Secret Weapon):**
   - In audio and image math, a low-frequency wave naturally creates **harmonics** at double the frequency:
     - $(1, 1) \longrightarrow (2, 2)$
     - $(1, 2) \longrightarrow (2, 4)$
     - $(2, 1) \longrightarrow (4, 2)$
   - This file explicitly draws a direct line connecting the **known low-frequency nodes** straight to their **missing high-frequency harmonic nodes**.
   - *Why?* This gives the GNN a direct highway to say: *"I know low frequency $(1, 2)$ has a value of 45.0, so its harmonic at $(2, 4)$ should be approximately 12.0!"*

4. **Makes all roads two-way (Bidirectional):**
   - Every connection goes both ways so messages can flow back and forth during neural network training.

---

### What It Outputs
It outputs a PyTorch tensor called **`edge_index`** of size `[2, 500]`:
- **500 total one-way roads** connecting the 64 nodes.
- When your GNN runs, PyTorch Geometric reads this `edge_index` to know exactly which nodes to send messages between.

---

### How to Test It
Run this command from your terminal:
```powershell
.venv\Scripts\python src/graph_builder.py
```
**Expected Output:**
```text
============================================================
Testing src/graph_builder.py
============================================================
Edge index shape:     torch.Size([2, 500])
Total directed edges: 500
Node count:           64 (Nodes 0 to 63)
Coordinates shape:    torch.Size([64, 2])
Min node index:       0
Max node index:       63
Connected nodes:      64 / 64
Sample Harmonic edge (1,2) -> (2,4): Connected = True

[SUCCESS] Graph Builder successfully validated!
```

---

*(More files will be added here as we implement them!)*
