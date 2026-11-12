# 🎨 Digital Image Processing Project: DCT & GNN Image Reconstruction

Welcome to our Digital Image Processing (DIP) project repository!  
This guide is written to help every team member get set up quickly, comfortably, and with zero confusion.

---

## 🧭 Table of Contents
1. [What This Project Is About](#-what-this-project-is-about)
2. [First Steps on Your Computer](#1-first-steps-on-your-computer-before-touching-git)
3. [How to Get the Project (Cloning)](#2-how-to-get-the-project-onto-your-computer-cloning)
4. [Setting Up Your Python Environment](#3-installing-the-required-libraries)
5. [How to Work and Save Your Code (New Branch)](#4-how-to-work-and-save-your-changes-safely)
6. [How to Get the Latest Code from Main](#5-how-to-get-the-latest-updates-from-the-team)
7. [Team Roles and Responsibilities](#6-team-roles-and-who-does-what)
8. [How to Run Your Work and See Results](#7-how-to-run-your-files-and-see-results)

---

## 📖 What This Project Is About
When JPEG compresses photos, it cuts them into $8 \times 8$ pixel squares, converts them to frequencies using **Discrete Cosine Transform (DCT)**, and removes fine details (high frequencies) to save file space. This causes reconstructed images to look blurry and full of grid lines.

**Our goal:** We build a lightweight **Graph Neural Network (GNN)** to predict those missing fine details from the preserved low frequencies, restoring cleaner, sharper images without grid seams!

---

## 1. First Steps on Your Computer (Before Touching Git)

Before working with the project repository, make sure your computer has the basic tools ready:

1. **Install Python:**  
   Make sure Python (version 3.9, 3.10, or 3.11) is installed. You can check by opening your terminal or Command Prompt and typing:
   ```bash
   python --version
   ```
2. **Install Git:**  
   Download and install Git from [git-scm.com](https://git-scm.com/) if you haven't already.
3. **Open Antigravity / Your Code Editor:**  
   Open Antigravity or VS Code. Whenever you need to run a command, open the built-in terminal (`Terminal -> New Terminal` or press ``Ctrl + ` ``).

> 💡 **Tip:** If you prefer using prompts in Antigravity rather than typing commands manually, look for the **🤖 Antigravity Prompt** box in each section below!

---

## 2. How to Get the Project onto Your Computer (Cloning)

"Cloning" simply means downloading a complete, linked copy of this project folder onto your laptop.

### Option A: Using the Terminal
1. Open your terminal in the folder where you want to keep your project (e.g. `C:\Users\YourName\Documents`).
2. Run this command:
   ```bash
   git clone https://github.com/Duljit2006/DIP-project.git
   ```
3. Move into the newly downloaded folder:
   ```bash
   cd DIP-project
   ```

### Option B: Using an Antigravity Prompt
> 🤖 **Prompt to Antigravity:**  
> *"Please clone the repository https://github.com/Duljit2006/DIP-project.git into my current folder and open the project directory for me."*

---

## 3. Installing the Required Libraries

All the tools and libraries we need are listed in `requirements.txt`. Installing them takes just one command.

### Option A: Using the Terminal
Inside the `DIP-project` folder, run:
```bash
pip install -r requirements.txt
```

*(Note: If you are working on the classical image processing and metrics files, standard packages like `numpy`, `scipy`, `opencv-python`, and `scikit-image` are all you need. You do not need any special GPU software on your laptop!)*

### Option B: Using an Antigravity Prompt
> 🤖 **Prompt to Antigravity:**  
> *"Please install all the libraries listed in requirements.txt in my environment."*

---

## 4. How to Work and Save Your Changes Safely

To make sure team members don't accidentally overwrite each other's code, we work in our own separate workspace called a **branch**.

### Step 4.1: Create Your Own Branch
Before editing code, create a new branch with your name or task name:

* **Terminal Command:**
  ```bash
  git checkout -b my-feature-branch
  ```
  *(Example: `git checkout -b feature-dct-metrics`)*

* 🤖 **Prompt to Antigravity:**  
  *"Please create and switch to a new git branch called feature-dct-metrics."*

---

### Step 4.2: Save and Upload Your Work to GitHub
When you finish editing your files (e.g. `src/dct_utils.py` or `src/metrics.py`) and want to save them to GitHub:

* **Terminal Commands:**
  ```bash
  # 1. Stage all your changed files
  git add .

  # 2. Add a short message describing what you did
  git commit -m "Add 8x8 DCT slicing and baseline PSNR calculation"

  # 3. Push your branch to GitHub
  git push -u origin my-feature-branch
  ```

* 🤖 **Prompt to Antigravity:**  
  *"Please stage all my recent file changes, commit them with the message 'Added DCT slicing functions', and push my branch to GitHub."*

---

## 5. How to Get the Latest Updates from the Team

When your teammate updates code on the main branch (e.g., adds new model weights or new helper scripts), here is how to bring those updates into your laptop smoothly:

### Option A: Using the Terminal
```bash
# 1. Switch to the main branch
git checkout main

# 2. Download the newest files from GitHub
git pull origin main

# 3. Switch back to your own branch and incorporate the updates
git checkout my-feature-branch
git merge main
```

### Option B: Using an Antigravity Prompt
> 🤖 **Prompt to Antigravity:**  
> *"Please fetch the latest updates from the main branch on GitHub and update my local branch with the new code."*

---

## 6. Team Roles and Who Does What

To keep our project organized, our workflow is split into two complementary tracks:

```
┌───────────────────────────────────────────────┐
│  TRACK A: Classical DIP & Evaluation Lead     │
│  (Runs on normal laptop CPU - No GPU needed)  │
├───────────────────────────────────────────────┤
│ • src/dct_utils.py: Slice 8x8 blocks, 2D-DCT  │
│ • src/metrics.py: Calculate PSNR & SSIM       │
│ • evaluate.py: Run tests & make visual plots  │
└──────────────────────┬────────────────────────┘
                       │ Passes data & baseline
                       ▼
┌───────────────────────────────────────────────┐
│  TRACK B: GNN Modeling & Training Lead        │
│  (Trained on Cloud GPU / Google Colab)        │
├───────────────────────────────────────────────┤
│ • src/graph_builder.py: Build frequency graph │
│ • src/dataset.py: PyTorch Geometric loader    │
│ • src/models.py: GNN architecture             │
│ • train.py: Training loop on BSDS500 dataset  │
└───────────────────────────────────────────────┘
```

---

## 7. How to Run Your Files and See Results

You will see your outputs in three simple ways:

### 1. Classical 8×8 Slicing (`src/dct_utils.py`)
* **Terminal Command:**
  ```bash
  python src/dct_utils.py
  ```
* 🤖 **Antigravity Prompt:** *"Run src/dct_utils.py and show me the block shapes and reconstruction test."*
* **What you will see:**  
  The terminal will confirm that images are successfully chopped into 4,096 blocks of $8 \times 8$. An interactive window will pop up showing the clear original image next to the blurry, blocky truncated image.

---

### 2. Baseline Score Check (`src/metrics.py`)
* **Terminal Command:**
  ```bash
  python src/metrics.py
  ```
* 🤖 **Antigravity Prompt:** *"Run src/metrics.py to calculate the baseline PSNR and SSIM on Set5 images."*
* **What you will see:**  
  The terminal will print clear baseline numbers in decibels (e.g., `Average Baseline PSNR: 25.34 dB`). Note these down for our report!

---

### 3. Final Model Evaluation & Comparison (`evaluate.py`)
Once the trained `model.pth` file is placed in the `checkpoints/` folder:
* **Terminal Command:**
  ```bash
  python evaluate.py
  ```
* 🤖 **Antigravity Prompt:** *"Run evaluate.py to test the trained GNN against the baseline and save visual comparison plots."*
* **What you will see:**  
  1. **Terminal:** Prints the score gain (e.g., `GNN PSNR: 28.75 dB (+2.33 dB gain!)`).
  2. **Screen Pop-up:** An interactive 3-panel comparison window (`Original` vs `Blurry Baseline` vs `Our AI Restored`) opens automatically.
  3. **Folder:** Ready-to-use comparison pictures and error heatmaps are saved inside the `results/` folder for our presentation slides!

---

## 📚 Visual Guides & References

If you want to read more about the project background or inspect diagrams, open any of the HTML guides in your web browser:
* `GUIDE/DIP_Implementation_Plan.html` — Complete step-by-step master plan with starter code.
* `GUIDE/DIP_Results_Guide.html` — Visual guide showing exact screenshots of what the outputs look like.
* `GUIDE/DIP_Model_Training_Guide.html` — Technical deep-dive on training math and pipelines.
* `GUIDE/DIP_Project_Guide.html` — Beginner-friendly introduction to DCT and neural networks.

---
*Happy coding! If anything ever feels confusing or an error pops up, simply ask Antigravity or ask your teammate!* 🚀
