# predict.py
import torch # type: ignore
import numpy as np # type: ignore

def predict_from_model(model, history):
    """
    history: np.array (N, history, 2)
    returns: np.array (N, future, 2)
    """
    model.eval()
    import torch # type: ignore
    device = next(model.parameters()).device
    with torch.no_grad():
        x = torch.from_numpy(history.astype('float32')).to(device)
        out = model(x).cpu().numpy()
    return out
