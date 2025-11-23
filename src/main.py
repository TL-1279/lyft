# main.py
"""
Entry point script to visualize / animate / train / predict with the project.
Usage examples in README.
"""

import argparse
from data_loader import load_config, build_dataset
from utils.helpers import set_l5kit_data_folder
from visualize import visualize_frame, animate_scene
from train import train_model

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--l5kit-data", type=str, default=None, help="Path to Lyft dataset folder (sets L5KIT_DATA_FOLDER).")
    p.add_argument("--config", type=str, default="config.yaml", help="Path to config yaml.")
    p.add_argument("--zarr-key", type=str, default=None, help="Relative zarr key (e.g. scenes/sample.zarr)")
    p.add_argument("--mode", type=str, choices=["visualize", "animate", "train"], default="visualize")
    p.add_argument("--scene-idx", type=int, default=0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--save-model", type=str, default="saved_model.pth")
    return p.parse_args()

def main():
    args = parse_args()
    if args.l5kit_data:
        set_l5kit_data_folder(args.l5kit_data)

    cfg = load_config(args.config)
    dataset, rasterizer, zarr_dataset = build_dataset(cfg, args.zarr_key, dataset_type="ego")

    if args.mode == "visualize":
        print("Visualizing index 0")
        visualize_frame(dataset, index=0, show=True)
    elif args.mode == "animate":
        out_file = f"scene_{args.scene_idx}.mp4"
        animate_scene(dataset, scene_idx=args.scene_idx, out_file=out_file, max_frames=300, interval=60)
    elif args.mode == "train":
        train_model(args.config, zarr_key=args.zarr_key, dataset_type="ego",
                    epochs=args.epochs, batch_size=args.batch_size, save_path=args.save_model)
    else:
        print("Unknown mode")

if __name__ == "__main__":
    main()
