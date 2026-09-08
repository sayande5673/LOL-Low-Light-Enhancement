"""
dataset.py
==========
PyTorch Dataset for the LOL (LOw-Light) paired dataset.

Expected directory layout (default LOL dataset release):

    lol_dataset/
        our485/
            low/    -> 485 low-light training images
            high/   -> 485 corresponding normal-light images
        eval15/
            low/    -> 15 low-light evaluation images
            high/   -> 15 corresponding normal-light images

Download: https://daooshee.github.io/BMVC2018website/
"""

import os
import glob

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class LowLightDataset(Dataset):
    """Loads paired (low-light, normal-light) images for training/eval."""

    def __init__(self, low_light_dir: str, normal_light_dir: str, image_size: int = 256):
        self.low_paths = sorted(glob.glob(os.path.join(low_light_dir, "*")))
        self.normal_paths = sorted(glob.glob(os.path.join(normal_light_dir, "*")))

        if len(self.low_paths) == 0:
            raise FileNotFoundError(f"No images found in low_light_dir: {low_light_dir}")
        if len(self.low_paths) != len(self.normal_paths):
            raise ValueError(
                f"Mismatched pair counts: {len(self.low_paths)} low-light images vs "
                f"{len(self.normal_paths)} normal-light images"
            )

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.low_paths)

    def __getitem__(self, idx: int):
        low_img = Image.open(self.low_paths[idx]).convert("RGB")
        normal_img = Image.open(self.normal_paths[idx]).convert("RGB")

        low_img = self.transform(low_img)
        normal_img = self.transform(normal_img)

        return low_img, normal_img
