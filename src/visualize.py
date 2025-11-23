# visualize.py
"""
Visualization utilities: single-frame visualization and scene animation.
These functions use l5kit rasterizer + draw_trajectory.
"""

import matplotlib.pyplot as plt
from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR
from l5kit.geometry import transform_points
from matplotlib import animation
import numpy as np
import PIL

def visualize_frame(dataset, index: int = 0, show: bool = True, save_path: str = None):
    data = dataset[index]
    im = data["image"].transpose(1, 2, 0)   # (C,H,W) -> (H,W,C) with C channels
    im = dataset.rasterizer.to_rgb(im)
    # target_positions in world coords (relative displacements) + centroid -> world coordinates
    target_positions_pixels = transform_points(
        data["target_positions"] + data["centroid"][:2],
        data["world_to_image"]
    )
    draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)
    plt.imshow(im[::-1])  # flip for display consistency
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

def animate_scene(dataset, scene_idx: int, out_file: str = None, max_frames: int = None, interval: int = 60):
    """
    Animate all frames in a given scene (retrieved by dataset.get_scene_indices(scene_idx)).
    If out_file provided, saves to mp4 (requires ffmpeg). Otherwise returns matplotlib.animation.FuncAnimation.
    """
    indices = dataset.get_scene_indices(scene_idx)
    if max_frames:
        indices = indices[:max_frames]

    images = []
    for idx in indices:
        data = dataset[idx]
        im = data["image"].transpose(1, 2, 0)
        im = dataset.rasterizer.to_rgb(im)
        target_positions_pixels = transform_points(
            data["target_positions"] + data["centroid"][:2],
            data["world_to_image"]
        )
        draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)
        images.append(PIL.Image.fromarray(im[::-1]))

    fig, ax = plt.subplots()
    implot = ax.imshow(images[0])
    ax.axis("off")

    def animate(i):
        implot.set_data(images[i])
        return (implot,)

    def init():
        implot.set_data(images[0])
        return (implot,)

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(images), interval=interval, blit=True)

    if out_file:
        # save to mp4 (ffmpeg required)
        anim.save(out_file, writer="ffmpeg")
        print(f"Saved animation to {out_file}")
        plt.close(fig)
        return out_file
    else:
        return anim
