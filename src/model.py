"""
model.py
========
Network architecture for LAN (Layered Attention-free Network) —
a lightweight low-light image enhancement model with two parallel
branches (reflectance + illumination) that are fused and decoded
into the final enhanced image.

Architecture summary
---------------------
Input (low-light RGB image)
    ├── ReflectanceEstimator  -> coarse structural/reflectance features
    └── IlluminationEnhancer  -> iteratively refined illumination map
              │
              ▼
      concat + 1x1 fusion conv
              │
              ▼
        SynthesisModule -> enhanced RGB image (sigmoid output)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Simple 2-conv residual block with identity skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class FeatureBlock(nn.Module):
    """Conv -> ReLU -> ResidualBlock -> Conv -> ReLU feature extractor."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ReflectanceEstimator(nn.Module):
    """
    Encodes structural / reflectance information at three spatial
    resolutions via strided convolutions between FeatureBlocks.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        self.stage1 = FeatureBlock(in_channels, base_channels)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1)
        self.stage2 = FeatureBlock(base_channels * 2, base_channels * 2)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
        self.stage3 = FeatureBlock(base_channels * 4, base_channels * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.stage1(x)
        d1 = self.down1(s1)
        s2 = self.stage2(d1)
        d2 = self.down2(s2)
        reflectance = self.stage3(d2)
        return reflectance


class IlluminationEnhancer(nn.Module):
    """
    Encodes the image into a latent space and iteratively refines it
    with a shared residual "refine" block (a lightweight recurrent
    style refinement, inspired by curve/iterative enhancement methods),
    then decodes back to a 3-channel illumination map.
    """

    def __init__(self, in_channels: int = 3, latent_channels: int = 64, iterations: int = 8):
        super().__init__()
        self.iterations = iterations

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, latent_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_channels, latent_channels, 3, padding=1),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(latent_channels),
            nn.Conv2d(latent_channels, latent_channels, 3, padding=1),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_channels, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        for _ in range(self.iterations):
            delta = self.refine(z)
            z = z + delta
        illumination = self.decoder(z)
        return illumination


class SynthesisModule(nn.Module):
    """Fuses reflectance + illumination features into the final RGB output."""

    def __init__(self, in_channels: int = 128):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.block(x))


class LAN(nn.Module):
    """
    Full low-light enhancement network combining a reflectance branch
    and an illumination branch, fused and synthesized into the final
    enhanced image. Output is resized back to the input resolution so
    the model accepts arbitrary input sizes.
    """

    def __init__(self):
        super().__init__()
        self.reflectance = ReflectanceEstimator()
        self.illumination = IlluminationEnhancer()

        # Fusion input = 128 channels (reflectance, base*4 = 128) + 3 (illumination)
        self.fusion = nn.Conv2d(128 + 3, 128, 1)

        self.synthesis = SynthesisModule()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.reflectance(x)
        i = self.illumination(x)

        # Match spatial resolution before concatenation
        i = F.interpolate(i, size=r.shape[2:], mode="bilinear", align_corners=False)

        fused = torch.cat([r, i], dim=1)
        fused = self.fusion(fused)

        out = self.synthesis(fused)

        # Restore original input resolution
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return out


if __name__ == "__main__":
    # Quick sanity check: forward pass with a dummy tensor
    model = LAN()
    dummy = torch.randn(1, 3, 256, 256)
    output = model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Total parameters: {n_params:,}")
