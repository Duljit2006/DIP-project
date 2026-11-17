# 📂 Project Files Documentation & Explanations

This document keeps a record of each file in our project, explaining what it does, the problem it solves, how it works step-by-step, and what outputs it produces.

---

## 🔍 Visualizing the Core Problem: An $8 \times 8$ Pixel Example

To easily visualize how an image block is transformed, degraded, and then restored by our GNN, let us walk through a concrete $8 \times 8$ block with actual numbers step-by-step.

### Step 1: The Raw Image Block (Pixel Intensities)
In grayscale, every pixel is a number from **0 (pure black)** to **255 (pure white)**.

Imagine our camera zoomed into an $8 \times 8$ patch showing a **soft horizontal edge** (the top half is darker gray, the bottom half is brighter gray):

```text
Row 0: [  50,  52,  49,  51,  50,  52,  51,  49 ]   <-- Dark gray (~50)
Row 1: [  51,  50,  52,  50,  49,  51,  50,  52 ]
Row 2: [  52,  51,  50,  52,  51,  49,  52,  50 ]
Row 3: [  50,  49,  51,  50,  52,  51,  50,  49 ]
Row 4: [ 180, 182, 179, 181, 180, 182, 181, 179 ]   <-- Bright gray (~180)
Row 5: [ 179, 181, 180, 182, 181, 179, 180, 182 ]
Row 6: [ 181, 180, 182, 179, 180, 181, 182, 180 ]
Row 7: [ 180, 182, 181, 180, 179, 181, 180, 182 ]
```

All 64 numbers look important right now. There is no simple way to drop numbers without ruining the image.

---

### Step 2: Run 2D-DCT (The "Frequency Transformation")
DCT performs a mathematical change of perspective. Instead of describing *individual pixels*, it asks:
> *"How much of this block is flat brightness, how much is a broad wave, and how much is tiny rapid ripples?"*

When we run 2D-DCT on those 64 pixel values, here is the resulting **$8 \times 8$ DCT Matrix**:

```text
             Column 0   Column 1   Column 2  ...  Column 7
Row 0 (Top): [  920,        2,        -1,    ...     0  ]  <-- Top-Left: Low frequencies
Row 1:       [ -360,        0,         1,    ...     0  ]
Row 2:       [   15,       -2,         0,    ...     0  ]
Row 3:       [  -70,        1,        -1,    ...     0  ]
...          ...
Row 7:       [    1,        0,         0,    ...     0  ]  <-- Bottom-Right: High frequencies
```

Notice what happened:
1. **Top-Left Corner (`Row 0, Col 0` = 920):**  
   This is the **DC Component**. It is the average brightness of all 64 pixels combined! It is a large number (`920`) because it holds the core light energy of the block.
2. **First Column (`Row 1, Col 0` = -360):**  
   Notice this large negative number. It captures the **sudden horizontal jump** from dark (50) to bright (180).
3. **Bottom-Right Corner:**  
   Look at all the tiny numbers: `-1, 0, 1, 2, 0`.  
   These are the **High-Frequency AC Components** (the tiny imperceptible noise variations between neighbor pixels).

---

### Step 3: JPEG Compression / Truncation (Information Loss)
To shrink file size, we **zero-out the high frequencies** (everything in the bottom-right):

```text
[  920,    2,   -1,    0,   0,   0,   0,   0 ]
[ -360,    0,    1,    0,   0,   0,   0,   0 ]
[   15,   -2,    0,    0,   0,   0,   0,   0 ]  <-- Top-left is KEPT (Low frequencies)
[  -70,    1,    0,    0,   0,   0,   0,   0 ]
------------------------------------------------
[    0,    0,    0,    0,   0,   0,   0,   0 ]
[    0,    0,    0,    0,   0,   0,   0,   0 ]  <-- Bottom-right is ERASED to 0 (High frequencies)
[    0,    0,    0,    0,   0,   0,   0,   0 ]
[    0,    0,    0,    0,   0,   0,   0,   0 ]
```

- We saved massive file size because long strings of `0` compress to almost nothing.
- **The Problem:** When you run Inverse DCT (IDCT) on this truncated block, the edge looks slightly blurry, and the boundaries between this block and its neighbors create a faint square seam (blocking artifacts).

---

### Step 4: Where Our GNN Enters the Story
Here is how our project fixes that block:

1. **Turn the 64 positions into 64 Nodes:**
   - Node 0 = `920` (DC - known)
   - Node 8 = `-360` (known)
   - Node 63 = `0.0` (erased, target to predict)
2. **Our `src/graph_builder.py` roads connect them:**
   - Road connects Node 8 (`-360`) to its harmonic Node 24 (`-70`).
3. **The GNN "Thinks":**
   - The GNN inspects Node 0 (`920`) and Node 8 (`-360`).
   - It sends messages across the harmonic edges:  
     *"I see a strong step transition of -360 in the low frequencies! Therefore, the missing high-frequency ripple at Node 32 wasn't 0.0, it should be approximately -6.5!"*
4. **Reconstruction:**
   - We replace the zeros with the GNN's predicted numbers.
   - We run **IDCT** back to pixels.
   - **Result:** The edge becomes crisp and the block boundary seam disappears!

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

## 2. `src/models.py`

### In Simple Words
`src/models.py` defines the **"Brain"** of our project: the Graph Neural Network (GNN) model named **`FrequencyGNN`**. It takes the graph connections built by `graph_builder.py` and actually executes the thinking and clue-passing to predict the missing high-frequency values.

---

### The Problem It Solves
`graph_builder.py` gave us the **roads (edges)** between frequencies, but roads alone do nothing without a vehicle to transport information. 

`FrequencyGNN` is the engine that travels along those roads:
- It looks at the known numbers (low frequencies at top-left).
- It transmits their clues along the spatial and harmonic edges.
- It outputs a predicted scalar number for the missing high-frequency spots that were erased during compression.

---

### What It Does Step-by-Step

1. **Input Features per Node (`in_channels = 3`):**
   Every single node (from 0 to 63) is given 3 numbers:
   - **Number 1:** Its current frequency amplitude (real number if known, `0.0` if erased).
   - **Number 2 & 3:** Its normalized $(u, v)$ frequency coordinate on the $8 \times 8$ grid (`u / 7.0`, `v / 7.0`), acting like GPS coordinates so the node knows where it sits.

2. **The "Thinking" Message-Passing Layers (`GCNConv`):**
   - The model has **3 Graph Convolutional Layers** with 128 hidden neurons each.
   - In every layer, each node collects clues from all neighboring nodes connected by the 500 edges, adds them up, applies normalization (`LayerNorm`), and activates them (`ReLU`).
   - It includes **residual skip connections** ($h = h + h_{\text{next}}$) so low-frequency signals are never lost or over-smoothed.

3. **The Prediction Head (`Linear Head`):**
   - After the nodes finish exchanging messages across all layers, a final small linear neural network maps each node's 128 internal thoughts into a single final guess: its predicted DCT amplitude.

---

### What It Takes In and What It Outputs

- **Input:**
  - `x`: Node features tensor of shape `(64, 3)` for a single block (or `(B * 64, 3)` for a batch of $B$ blocks).
  - `edge_index`: Road connection tensor of shape `(2, 500)` from `graph_builder.py`.
- **Output:**
  - A 1D tensor of shape `(64,)` for a single block (or `(B * 64,)` for a batch), where each number is the model's predicted DCT amplitude for that frequency node.

---

### Model Stats
- **Total Trainable Parameters:** **42,625** (Feather-light! Only takes ~170 KB of storage and trains in milliseconds).
- **GPU & CPU Compatible:** Runs seamlessly on both your laptop CPU and free cloud GPUs (Google Colab).

---

### How to Test It
Run this command from your terminal:
```powershell
.venv\Scripts\python src/models.py
```
**Expected Output:**
```text
============================================================
Testing src/models.py (FrequencyGNN)
============================================================
Model architecture:
FrequencyGNN(
  (convs): ModuleList(
    (0): GCNConv(3, 128)
    (1-2): 2 x GCNConv(128, 128)
  )
  (norms): ModuleList(
    (0-2): 3 x LayerNorm((128,), eps=1e-05, elementwise_affine=True, bias=True)
  )
  (relu): ReLU()
  (dropout): Identity()
  (head): Sequential(
    (0): Linear(in_features=128, out_features=64, bias=True)
    (1): ReLU()
    (2): Linear(in_features=64, out_features=1, bias=True)
  )
)

Total trainable parameters: 42,625

[Test 1: Single Block (64 nodes)]
Input node features shape:  torch.Size([64, 3])
Graph edge index shape:     torch.Size([2, 500])
Output predictions shape:   torch.Size([64]) (Expected: torch.Size([64]))

[Test 2: Batched Blocks (4 blocks = 256 nodes)]
Batch input node features:  torch.Size([256, 3]) (Expected: torch.Size([256, 3]))
Batch edge index:           torch.Size([2, 2000])
Batch output predictions:   torch.Size([256]) (Expected: torch.Size([256]))

[SUCCESS] FrequencyGNN successfully validated for single and batched inputs!
```

---

## 3. `src/dataset.py`

### In Simple Words
`src/dataset.py` is the **"Delivery Truck" / Data Pipeline** of our project. It takes thousands of $8 \times 8$ DCT blocks, attaches the 500 graph roads from `graph_builder.py` to each one, marks which numbers are missing, and packages them into clean mini-batches so the GPU/CPU can learn quickly.

---

### The Problem It Solves
Raw images and DCT blocks are just tables of numbers. But our `FrequencyGNN` model expects **PyTorch Geometric Graph Data objects** (`Data(x=..., edge_index=..., y=..., missing_mask=...)`).

`src/dataset.py` bridges that gap:
1. It automatically turns flat 64-number blocks into 64 graph nodes.
2. It gives each node its GPS location: normalized $(u, v)$ coordinates.
3. It creates a **`missing_mask`** so the training loss knows *only* to grade the erased numbers, never penalizing the model for the already-known low frequencies!
4. It stacks multiple blocks together into **mini-batches** (e.g. 64 blocks at a time) for fast, parallel GPU training.

---

### Key Components Step-by-Step

1. **`get_zigzag_indices()` & `create_zigzag_mask(cutoff)`:**
   - In JPEG compression, frequencies are scanned in a diagonal **zigzag order** from top-left (lowest) to bottom-right (highest).
   - This function creates a boolean mask where the first `cutoff` numbers (e.g., 15) are `False` (known/kept), and the remaining 49 numbers are `True` (missing/erased to be predicted).

2. **`DCTGraphDataset` Class:**
   - Inherits from PyTorch Geometric's `Dataset`.
   - Attaches `self.edge_index` (the 500 roads from `graph_builder.py`) to every block.
   - For every block, outputs a `Data` object containing:
     - `x`: Node features of shape `(64, 3)` $\rightarrow$ `[amplitude, u_coord, v_coord]`.
     - `edge_index`: Shape `(2, 500)`.
     - `y`: Untouched ground-truth DCT values (the answer key).
     - `missing_mask`: 49 missing node flags.

3. **`get_dct_dataloader(...)` Helper:**
   - Bundles the dataset into a PyG `DataLoader` with parallel batching and shuffling.

---

### How to Test It
Run this command from your terminal:
```powershell
.venv\Scripts\python src/dataset.py
```
**Expected Output:**
```text
============================================================
Testing src/dataset.py (DCTGraphDataset & DataLoader)
============================================================
Zigzag cutoff:       15
Preserved low nodes: 15 / 64
Missing high nodes:  49 / 64 (to be predicted)

[Simulating 100 DCT blocks...]
Dataset length:      100 blocks
Single graph x:      torch.Size([64, 3]) (Expected: torch.Size([64, 3]))
Single graph edges:  torch.Size([2, 500]) (Expected: torch.Size([2, 500]))
Single graph y:      torch.Size([64]) (Expected: torch.Size([64]))
Missing mask count:  49 missing nodes

[Batching Test (batch_size = 16)]
Batch x shape:       torch.Size([1024, 3]) (Expected: torch.Size([1024, 3]))
Batch edge_index:    torch.Size([2, 8000]) (Expected: torch.Size([2, 8000]))
Batch y shape:       torch.Size([1024]) (Expected: torch.Size([1024]))
Batch graphs count:  16 (Expected: 16)

[SUCCESS] DCTGraphDataset and DataLoader successfully validated!
```

---

*(More files will be added here as we implement them!)*
