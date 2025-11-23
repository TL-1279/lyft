# data_loader.py. đọc .zarr nếu có hoặc trả về synthetic
import os
import numpy as np # type: ignore
from simulator import simulate_scene

# Optional L5Kit imports guarded
try:
    from l5kit.data import ChunkedDataset, LocalDataManager # type: ignore
    from l5kit.dataset import EgoDataset, AgentDataset # type: ignore
    from l5kit.rasterization import build_rasterizer # type: ignore
    from l5kit.configs import load_config_data # type: ignore
    L5KIT_AVAILABLE = True
except Exception:
    L5KIT_AVAILABLE = False

def load_synthetic_scene(**kwargs):
    return simulate_scene(**kwargs)

def load_l5kit_scene(cfg_path, zarr_key='scenes/sample.zarr', scene_idx=0):
    if not L5KIT_AVAILABLE:
        raise RuntimeError("l5kit not installed in environment.")
    cfg = load_config_data(cfg_path)
    dm = LocalDataManager(None)
    dataset_path = dm.require(zarr_key)
    z = ChunkedDataset(dataset_path); z.open()
    rasterizer = build_rasterizer(cfg, dm)
    ego_ds = EgoDataset(cfg, z, rasterizer)
    # pick scene indices and read frames -> convert to history/future arrays
    indices = ego_ds.get_scene_indices(scene_idx)
    # For demo return first N indices as "frames"
    history = []
    future = []
    metas = []
    for idx in indices:
        d = ego_ds[idx]
        # d["target_positions"] is (future_len, 2) in meters (relative)
        centroid = d["centroid"][:2]
        hist = (d["history_positions"] - centroid) if "history_positions" in d else np.zeros((cfg["model_params"]["history_num_frames"],2))
        fut = (d["target_positions"] + centroid)
        history.append(hist)
        future.append(fut)
        metas.append({"index": idx})
    history = np.asarray(history)
    future = np.asarray(future)
    return history, future, metas
