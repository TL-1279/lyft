# main.py
import argparse
from data_loader import load_synthetic_scene, load_l5kit_scene
from train import train
from predict import predict_from_model
from visualize import plot_agents
from model import SimpleLSTM
import torch # type: ignore
import numpy as np # type: ignore

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["visualize", "animate", "train", "predict"], default="visualize")
    p.add_argument("--use-l5kit", action="store_true", help="Use l5kit dataset")
    p.add_argument("--l5kit-data", default=None, help="path to L5KIT_DATA_FOLDER or config")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--scene-idx", type=int, default=0)
    args = p.parse_args()

    if args.mode == "visualize":
        if args.use_l5kit:
            if args.l5kit_data is None:
                raise RuntimeError("Provide --l5kit-data when using --use-l5kit")
            # adapt load_l5kit_scene usage — for simplicity expect sample zarr key
            history, future, metas = load_l5kit_scene(args.l5kit_data, scene_idx=args.scene_idx)
            print("Loaded L5Kit scene arrays:", history.shape, future.shape)
            plot_agents(history[:8], future[:8])
        else:
            history, future, meta = load_synthetic_scene(num_agents=8, history_steps=10, future_steps=50, seed=42)
            plot_agents(history, future)
    elif args.mode == "train":
        model = train(epochs=args.epochs, num_scenes=100, num_agents=8, history=10, future=50)
        torch.save(model.state_dict(), "model.pth")
        print("Saved model.pth")
    elif args.mode == "predict":
        # quick demo: load model and predict on one synthetic scene
        history, future, meta = load_synthetic_scene(num_agents=8, history_steps=10, future_steps=50, seed=1)
        model = SimpleLSTM(future_len=50)
        try:
            model.load_state_dict(torch.load("model.pth"))
            print("Loaded model.pth")
        except Exception as e:
            print("Could not load model.pth — you can run --mode train first. Exception:", e)
        preds = predict_from_model(model, history)
        plot_agents(history, future, preds)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()
