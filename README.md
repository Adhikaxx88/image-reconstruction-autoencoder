# UAS No. 2 — Image Reconstruction with Convolutional Autoencoders

## Project Overview

This repository contains the solution for **Problem 2** of the Final Exam (UAS) for the Deep Learning course, by **Adhika Gunawan (2802438205)**.

The task is to build and evaluate **convolutional autoencoders** that reconstruct grayscale satellite/overhead images from two classes — **Oil Gas Field** and **Stadium** — and to improve reconstruction quality using a modified architecture and loss function. The notebook is structured in three parts:

- **2A** — EDA & preprocessing (per class)
- **2B** — Baseline convolutional autoencoder (per class)
- **2C** — Modified autoencoder with a custom MSE + SSIM loss (per class)

### Key Findings

| Class | Baseline SSIM | Modified SSIM | Improvement |
|---|---|---|---|
| Oil Gas Field | 0.3973 | 0.4965 | +0.0992 |
| Stadium | 0.6613 | 0.7449 | +0.0836 |

- `Oil Gas Field` is consistently harder to reconstruct than `Stadium` due to its more complex, bimodal pixel-intensity distribution (std ≈ 53.91 vs. 42.47).
- Adding `BatchNormalization`, a lower learning rate (0.0005), and a **combined MSE + SSIM loss** (`MSE + alpha * (1 - SSIM)`) improved structural similarity for both classes, at the cost of a larger apparent train/validation loss gap (partly explained by the different loss scale, not pure overfitting — SSIM is the metric that matters for comparison here).

## Dataset Description

- **Classes**: `oil_gas_field` and `stadium` — overhead/satellite grayscale imagery.
- Original `train`/`test` folders were merged per class and re-split (two-stage random split) into **80% train / 10% validation / 10% test**:
  - Oil Gas Field: 798 / 100 / 100
  - Stadium: 758 / 95 / 95
- Images loaded in grayscale, normalized to [0, 1], and paired as `(X, X)` since an autoencoder learns to reconstruct its own input.
- Pixel statistics: Oil Gas Field mean ≈ 151.48, std ≈ 53.91 (brighter, bimodal distribution, higher texture variance); Stadium mean ≈ 98.70, std ≈ 42.47 (darker, more uniform).
- Note: the raw image folders (`oil_gas_field/`, `stadium/`) are part of this working directory but are large binary image collections — see Project Structure for what is tracked in version control.

## Methodology & Implementation

1. **Baseline autoencoder (2B)**: encoder compresses 28×28 images to a 128-dimensional latent vector via `Conv2D` → `MaxPooling2D` → `Flatten` → `Dense(128)`; decoder mirrors this with `Dense` → `Reshape` → `UpSampling2D` → two `Conv2D` layers, sigmoid output. Compiled with Adam + MSE loss, trained with `EarlyStopping` (patience 10) and `ModelCheckpoint`.
   - Oil Gas Field: mild overfitting, early-stopped at epoch 37, test SSIM 0.3973.
   - Stadium: stronger generalization, early-stopped at epoch 60, test SSIM 0.6613.
2. **Modified autoencoder (2C)**:
   - `BatchNormalization` added after each `Conv2D` in both encoder and decoder for training stability.
   - Learning rate reduced to 0.0005 for finer convergence.
   - **Custom loss**: `MSE + alpha * (1 - SSIM)` — directly optimizes structural similarity, not just pixel-wise error. `alpha = 0.5` for Oil Gas Field, `alpha = 1.0` for Stadium (its more uniform structure tolerates a more aggressive SSIM weighting).
   - Callbacks: `EarlyStopping` (patience 10), `ModelCheckpoint`, `ReduceLROnPlateau` (factor 0.5, patience 4).
3. **Evaluation metric**: **Structural Similarity Index (SSIM)** between original and reconstructed images on the test set — chosen over raw MSE because it captures structural/perceptual similarity rather than pure pixel-wise error.

## Project Structure

```
UAS/no 2/
├── README.md              # This file
├── no2.ipynb                # Full pipeline: EDA, re-splitting, baseline & modified autoencoders, SSIM evaluation
├── models/                  # Trained model checkpoints
│   ├── baseline_ogf.keras     # Baseline autoencoder — Oil Gas Field
│   ├── baseline_std.keras     # Baseline autoencoder — Stadium
│   ├── modified_ogf.keras     # Modified (MSE+SSIM) autoencoder — Oil Gas Field
│   └── modified_std.keras     # Modified (MSE+SSIM) autoencoder — Stadium
├── oil_gas_field/            # Re-split image data (train/val/test) for the Oil Gas Field class
│   ├── train/ val/ test/
└── stadium/                   # Re-split image data (train/val/test) for the Stadium class
    ├── train/ val/ test/
```

## How to Run / Requirements

### Prerequisites

- Python 3.9+
- Jupyter Notebook / JupyterLab
- A GPU is recommended for faster convolutional autoencoder training

### Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-image tensorflow
```

### Run

1. Open `no2.ipynb` in Jupyter and run all cells top to bottom.
2. The notebook expects the `oil_gas_field/` and `stadium/` image folders (in this directory) to already be organized into `train/`, `val/`, and `test/` subfolders per class — the re-splitting cell in section 2A will (re)generate this structure from the original class-labeled data if needed.
3. Training will save the best checkpoints to `models/`, and the evaluation cells compute per-class SSIM on the test set and visualize original-vs-reconstructed image pairs.
4. To reuse a trained model without retraining:

   ```python
   from tensorflow.keras.models import load_model
   model = load_model('models/modified_ogf.keras')   # or any other checkpoint in models/
   ```
