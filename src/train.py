"""
train.py
========
CLI entry point to train the LAN low-light enhancement model on the
LOL dataset.

Example
-------
    python src/train.py \
        --low_dir data/lol_dataset/our485/low \
        --high_dir data/lol_dataset/our485/high \
        --epochs 50 \
        --batch_size 4 \
        --lr 1e-4 \
        --checkpoint_dir checkpoints
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LowLightDataset
from model import LAN
from utils import save_checkpoint, compute_psnr


def parse_args():
    parser = argparse.ArgumentParser(description="Train the LAN low-light enhancement model.")
    parser.add_argument("--low_dir", type=str, required=True, help="Path to low-light training images")
    parser.add_argument("--high_dir", type=str, required=True, help="Path to normal-light (ground truth) training images")
    parser.add_argument("--val_low_dir", type=str, default=None, help="Optional path to low-light validation images")
    parser.add_argument("--val_high_dir", type=str, default=None, help="Optional path to normal-light validation images")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint_name", type=str, default="lan_model.pth")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=2)
    return parser.parse_args()


def evaluate(model, val_loader, device):
    model.eval()
    total_psnr = 0.0
    n = 0
    with torch.no_grad():
        for low_img, gt_img in val_loader:
            low_img, gt_img = low_img.to(device), gt_img.to(device)
            enhanced = model(low_img)
            for i in range(enhanced.size(0)):
                total_psnr += compute_psnr(enhanced[i], gt_img[i])
                n += 1
    return total_psnr / max(n, 1)


def train(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_dataset = LowLightDataset(args.low_dir, args.high_dir, image_size=args.image_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = None
    if args.val_low_dir and args.val_high_dir:
        val_dataset = LowLightDataset(args.val_low_dir, args.val_high_dir, image_size=args.image_size)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = LAN().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_psnr = -1.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for low_img, gt_img in progress:
            low_img, gt_img = low_img.to(args.device), gt_img.to(args.device)

            optimizer.zero_grad()
            enhanced = model(low_img)
            loss = criterion(enhanced, gt_img)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        log_line = f"Epoch [{epoch + 1}/{args.epochs}] Loss: {avg_loss:.4f}"

        if val_loader is not None:
            val_psnr = evaluate(model, val_loader, args.device)
            log_line += f" | Val PSNR: {val_psnr:.2f} dB"
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_checkpoint(model, os.path.join(args.checkpoint_dir, "best_" + args.checkpoint_name))

        print(log_line)

    # Always save the final model
    save_checkpoint(model, os.path.join(args.checkpoint_dir, args.checkpoint_name))
    print(f"Training finished. Model saved to {os.path.join(args.checkpoint_dir, args.checkpoint_name)}")


if __name__ == "__main__":
    train(parse_args())
