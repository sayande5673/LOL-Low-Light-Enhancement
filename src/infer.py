"""
infer.py
========
Run a trained LAN model on a single image (or a folder of images) and
save the enhanced output(s). Optionally displays a side-by-side plot.

Examples
--------
Single image, save enhanced result:
    python src/infer.py --checkpoint checkpoints/lan_model.pth \
        --input data/lol_dataset/eval15/low/146.png \
        --output outputs/146_enhanced.png

Whole folder:
    python src/infer.py --checkpoint checkpoints/lan_model.pth \
        --input data/lol_dataset/eval15/low \
        --output outputs/
"""

import argparse
import os

import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from model import LAN
from utils import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a trained LAN model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a trained .pth checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Path to a single image or a directory of images")
    parser.add_argument("--output", type=str, required=True, help="Output file path (single image) or directory (folder input)")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--show", action="store_true", help="Display input vs. enhanced side by side with matplotlib")
    return parser.parse_args()


def enhance_image(model, image_path, transform, device):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    return image, output.squeeze(0).clamp(0, 1).cpu()


def main(args):
    model = LAN()
    model = load_checkpoint(model, args.checkpoint, device=args.device)
    model.to(args.device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
    )

    is_dir = os.path.isdir(args.input)
    image_paths = (
        [os.path.join(args.input, f) for f in sorted(os.listdir(args.input))]
        if is_dir
        else [args.input]
    )

    if is_dir:
        os.makedirs(args.output, exist_ok=True)

    for path in image_paths:
        original, enhanced = enhance_image(model, path, transform, args.device)

        out_path = (
            os.path.join(args.output, os.path.basename(path)) if is_dir else args.output
        )
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        save_image(enhanced, out_path)
        print(f"Saved: {out_path}")

        if args.show:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("Input")
            plt.imshow(original)
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.title("Enhanced")
            plt.imshow(enhanced.permute(1, 2, 0).numpy())
            plt.axis("off")
            plt.show()


if __name__ == "__main__":
    main(parse_args())
