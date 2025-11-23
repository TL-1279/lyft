# train.py
import torch # type: ignore
from torch.utils.data import DataLoader, TensorDataset # type: ignore
import numpy as np # type: ignore
from model import SimpleLSTM
from simulator import simulate_scene

def prepare_synth_loader(num_scenes=200, num_agents=8, history=10, future=50, batch_size=32):
    X = []
    Y = []
    for _ in range(num_scenes):
        hist, fut, _ = simulate_scene(num_agents=num_agents, history_steps=history, future_steps=future, seed=None)
        # hist shape: (num_agents, history, 2)
        # fut shape:  (num_agents, future, 2)
        # flatten agents across scenes
        X.append(hist)
        Y.append(fut)
    X = np.concatenate(X, axis=0).astype(np.float32)  # (num_scenes*num_agents, history, 2)
    Y = np.concatenate(Y, axis=0).astype(np.float32)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    return loader

def train(epochs=5, device=None, **kwargs):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    loader = prepare_synth_loader(**kwargs)
    model = SimpleLSTM(future_len=kwargs.get("future", 50)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    for ep in range(epochs):
        model.train()
        tot = 0.0
        n = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * xb.size(0)
            n += xb.size(0)
        print(f"Epoch {ep+1}/{epochs}  avg_loss={tot/n:.6f}")
    return model
