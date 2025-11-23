# train.py
"""
Minimal training loop demonstrating how to train the ResNetTrajectory on EgoDataset.
This reproduces the simple MSE baseline from notebook (flatten future positions).
Note: l5kit typical pipelines use advanced collate, augmentations, target availability weighting, etc.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from data_loader import load_config, build_dataset
from model import ResNetTrajectory

def train_model(cfg_path: str,
                zarr_key: str = None,
                dataset_type: str = "ego",
                epochs: int = 5,
                batch_size: int = 8,
                lr: float = 1e-4,
                save_path: str = "saved_model.pth",
                device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    cfg = load_config(cfg_path)
    dataset, rasterizer, zarr_dataset = build_dataset(cfg, zarr_key, dataset_type)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=lambda x: x)

    future_num = int(cfg["model_params"].get("future_num_frames", 50))
    model = ResNetTrajectory(future_num_frames=future_num, pretrained=True)
    model.to(device)

    criterion = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        steps = 0
        for batch in dataloader:
            # batch is a list of dicts
            images = np.stack([b["image"] for b in batch])  # shape (B, C, H, W)
            targets = np.stack([b["target_positions"] for b in batch])  # (B, future, 2)
            images_t = torch.tensor(images, dtype=torch.float32).to(device)
            targets_t = torch.tensor(targets, dtype=torch.float32).view(images_t.size(0), -1).to(device)

            optimizer.zero_grad()
            outputs = model(images_t)
            loss = criterion(outputs, targets_t)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            steps += 1

        avg_loss = running_loss / max(1, steps)
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.6f}")

        # Save checkpoint each epoch
        torch.save(model.state_dict(), f"{save_path}.epoch{epoch+1}")

    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Model saved to {save_path}")
    return model
