# data_loader.py
"""
Utilities to load config and build l5kit datasets (EgoDataset / AgentDataset).
Wraps LocalDataManager + ChunkedDataset + rasterizer creation.
"""

from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.rasterization import build_rasterizer
from l5kit.dataset import EgoDataset, AgentDataset

def load_config(path: str):
    """
    Load a YAML/JSON config using l5kit helper.
    """
    cfg = load_config_data(path)
    return cfg

def build_dataset(cfg: dict, zarr_key: str = None, dataset_type: str = "ego"):
    """
    Build and return (dataset, rasterizer, zarr_dataset).

    - cfg: loaded config dict
    - zarr_key: relative path string e.g. "scenes/sample.zarr". If None uses cfg['val_data_loader']['key'] if present.
    - dataset_type: "ego" or "agent"
    """
    dm = LocalDataManager()
    if zarr_key is None:
        # try config fallback
        if "val_data_loader" in cfg and "key" in cfg["val_data_loader"]:
            zarr_key = cfg["val_data_loader"]["key"]
        else:
            raise ValueError("Provide zarr_key or set val_data_loader.key in config")

    dataset_path = dm.require(zarr_key)
    zarr_dataset = ChunkedDataset(dataset_path)
    zarr_dataset.open()

    rasterizer = build_rasterizer(cfg, dm)

    if dataset_type.lower() == "ego":
        dataset = EgoDataset(cfg, zarr_dataset, rasterizer)
    else:
        dataset = AgentDataset(cfg, zarr_dataset, rasterizer)

    return dataset, rasterizer, zarr_dataset
