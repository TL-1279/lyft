# utils/helpers.py
import os

def set_l5kit_data_folder(path: str):
    """
    Set L5Kit data folder environment variable used by LocalDataManager.
    path: absolute path to folder that contains scenes/, aerial_map/, semantic_map/, meta.json
    """
    if path is None:
        return
    os.environ["L5KIT_DATA_FOLDER"] = str(path)
