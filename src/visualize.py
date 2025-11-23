# visualize.py
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
from matplotlib import animation # type: ignore
from PIL import Image # type: ignore

def plot_agents(history, future=None, preds=None, title=""):
    """
    history: (N, history, 2)
    future:  (N, future, 2) optional (ground truth)
    preds:   (N, future, 2) optional (pred)
    """
    plt.figure(figsize=(6,6))
    for i in range(history.shape[0]):
        h = history[i]
        plt.plot(h[:,0], h[:,1], '-o', alpha=0.6)
        if future is not None:
            f = future[i]
            plt.plot(f[:,0], f[:,1], '--', alpha=0.6)
        if preds is not None:
            p = preds[i]
            plt.plot(p[:,0], p[:,1], ':', alpha=0.9)
    plt.title(title)
    plt.axis('equal')
    plt.show()

def animate_scene_from_frames(frames, save_path=None, interval=60):
    """
    frames: list of images as numpy arrays (H,W,3)
    """
    fig, ax = plt.subplots()
    im = ax.imshow(frames[0])
    ax.axis('off')
    def animate(i):
        im.set_array(frames[i])
        return (im,)
    anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=interval, blit=True)
    if save_path is not None:
        anim.save(save_path, fps=30, dpi=150)
    return anim
