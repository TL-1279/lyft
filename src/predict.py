# predict.py
"""
Simple prediction helper to run model inference on dataset.
"""

import torch
import numpy as np
from model import ResNetTrajectory

def predict(model, dataset, batch_size: int = 8, device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model.to(device)
    model.eval()

    preds = []
    n = len(dataset)
    i = 0
    with torch.no_grad():
        while i < n:
            batch_indices = list(range(i, min(i + batch_size, n)))
            batch = [dataset[idx] for idx in batch_indices]
            images = np.stack([b["image"] for b in batch])
            images_t = torch.tensor(images, dtype=torch.float32).to(device)
            outputs = model(images_t)
            preds.append(outputs.cpu().numpy())
            i += batch_size

    preds = np.vstack(preds)
    return preds
