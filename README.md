# LAN: Low-Light Image Enhancement on the LOL Dataset

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A PyTorch implementation of **LAN** (Layered enhancement Network), a lightweight
deep learning model that restores low-light images to normal illumination
using a dual-branch **reflectance + illumination** architecture, trained and
evaluated on the [LOL (LOw-Light) dataset](https://daooshee.github.io/BMVC2018website/).

<p align="center">
  <em>Input (low-light) → LAN → Enhanced output</em>
</p>

---

## Overview

Images captured in low-light conditions suffer from poor visibility, low
contrast, and noise. This project implements an end-to-end trainable network
that decomposes the enhancement problem into two cooperating branches,
inspired by Retinex theory:

- **Reflectance branch** — a multi-scale convolutional encoder
  (`ReflectanceEstimator`) that extracts structural / texture features robust
  to illumination changes.
- **Illumination branch** — an encoder + iterative residual refinement module
  (`IlluminationEnhancer`) that progressively estimates a corrected
  illumination map over a fixed number of refinement steps.

The two branches are fused with a `1x1` convolution and decoded by a
`SynthesisModule` into the final enhanced RGB image, with a sigmoid output
activation and full-resolution bilinear resizing so the model works on
arbitrary input sizes.

```
Low-light image (H, W, 3)
        │
        ├──► ReflectanceEstimator ─────► reflectance features (128-ch)
        │                                          │
        └──► IlluminationEnhancer ─────► illumination map (3-ch, resized)
                                                     │
                                        concat + 1x1 fusion conv
                                                     │
                                             SynthesisModule
                                                     │
                                     Enhanced image (H, W, 3), sigmoid
```

## Features

- Clean, modular PyTorch implementation (`model.py`, `dataset.py`, `train.py`, `infer.py`)
- CLI-driven training with configurable hyperparameters and optional validation split
- PSNR / SSIM evaluation utilities (`utils.py`)
- Single-image and batch inference script with optional side-by-side visualization
- Checkpointing (best model by validation PSNR + final model)
- Originally prototyped in Google Colab (`notebooks/LOL.ipynb`), refactored here
  into a reusable, script-based project

## Project Structure

```
lol-low-light-enhancement/
├── src/
│   ├── model.py       # LAN network architecture
│   ├── dataset.py      # LowLightDataset (paired low/normal images)
│   ├── train.py         # CLI training script
│   ├── infer.py          # CLI inference / visualization script
│   └── utils.py            # PSNR/SSIM metrics, checkpoint helpers
├── notebooks/
│   └── LOL.ipynb        # Original prototyping notebook (Colab)
├── requirements.txt
├── LICENSE
└── README.md
```

## Dataset

This project uses the **LOL dataset**: 500 low-light/normal-light image pairs
(485 for training, 15 for evaluation), commonly used as a benchmark for
low-light enhancement research.

- Paper: *Deep Retinex Decomposition for Low-Light Enhancement* (BMVC 2018)
- Download: https://daooshee.github.io/BMVC2018website/

Expected directory layout after downloading and extracting:

```
data/lol_dataset/
├── our485/
│   ├── low/     # 485 low-light training images
│   └── high/    # 485 corresponding normal-light images
└── eval15/
    ├── low/     # 15 low-light evaluation images
    └── high/    # 15 corresponding normal-light images
```

## Installation

```bash
git clone https://github.com/<your-username>/lol-low-light-enhancement.git
cd lol-low-light-enhancement
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

## Usage

### 1. Train

```bash
python src/train.py \
    --low_dir data/lol_dataset/our485/low \
    --high_dir data/lol_dataset/our485/high \
    --val_low_dir data/lol_dataset/eval15/low \
    --val_high_dir data/lol_dataset/eval15/high \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4 \
    --checkpoint_dir checkpoints
```

Key arguments (see `python src/train.py --help` for the full list):

| Argument         | Default | Description                          |
|------------------|---------|---------------------------------------|
| `--image_size`   | 256     | Resize resolution for training images |
| `--batch_size`   | 4       | Training batch size                   |
| `--epochs`       | 50      | Number of training epochs             |
| `--lr`           | 1e-4    | Adam learning rate                    |
| `--device`       | auto    | `cuda` if available, else `cpu`       |

### 2. Run inference

Single image:

```bash
python src/infer.py \
    --checkpoint checkpoints/lan_model.pth \
    --input data/lol_dataset/eval15/low/146.png \
    --output outputs/146_enhanced.png \
    --show
```

Whole folder:

```bash
python src/infer.py \
    --checkpoint checkpoints/lan_model.pth \
    --input data/lol_dataset/eval15/low \
    --output outputs/
```

### 3. Evaluate quantitatively

`src/train.py` reports validation PSNR each epoch when `--val_low_dir` /
`--val_high_dir` are supplied. `src/utils.py` exposes `compute_psnr` and
`compute_ssim` for standalone evaluation scripts.

## Results

> Fill in after training on your hardware — numbers depend on epochs, batch
> size, and image resolution used.

| Model | Dataset (eval15) | PSNR (dB) | SSIM |
|-------|-------------------|-----------|------|
| LAN (this repo) | LOL eval15 | 17.64 | 0.7068 |

## Future Work

- Add perceptual (VGG) and SSIM loss terms alongside MSE
- Data augmentation (random crop, flip, color jitter) for better generalization
- Export to ONNX / TorchScript for deployment
- Compare against baselines (Retinex-Net, Zero-DCE, EnlightenGAN)

## Author

**Sayan Dey**
M.Tech (Artificial Intelligence), IIT Bhubaneswar

## License

This project is licensed under the [MIT License](LICENSE).
