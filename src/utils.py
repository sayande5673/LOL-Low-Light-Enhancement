"""
utils.py
========
Small helper utilities: PSNR/SSIM metrics for evaluating enhancement
quality, and a checkpoint save/load helper.
"""

import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """(C, H, W) tensor in [0, 1] -> (H, W, C) numpy array in [0, 1]."""
    return t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR between two (C, H, W) tensors in [0, 1]."""
    pred_np = tensor_to_numpy(pred)
    target_np = tensor_to_numpy(target)
    return float(_psnr(target_np, pred_np, data_range=1.0))


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """SSIM between two (C, H, W) tensors in [0, 1]."""
    pred_np = tensor_to_numpy(pred)
    target_np = tensor_to_numpy(target)
    return float(_ssim(target_np, pred_np, data_range=1.0, channel_axis=2))


def save_checkpoint(model: torch.nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)


def load_checkpoint(model: torch.nn.Module, path: str, device: str = "cpu") -> torch.nn.Module:
    model.load_state_dict(torch.load(path, map_location=device))
    return model
