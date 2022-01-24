import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pickle
import torch
from typing import List, Tuple
import os


def save_tensors_as_image_pairs(
    sketch_tensors: torch.Tensor, image_tensors: torch.Tensor, path: Path
) -> None:
    sketches = (sketch_tensors.squeeze(1).cpu().numpy() + 1) / 2
    images = (image_tensors.permute(0, 2, 3, 1).cpu().numpy() + 1) / 2

    fig, ax = plt.subplots(2, len(sketches))
    for i, (sketch, image) in enumerate(zip(sketches, images)):
        ax[0, i].imshow(sketch, cmap="gray")
        ax[1, i].imshow(image)
        ax[0, i].axis("off")
        ax[1, i].axis("off")
    fig.set_size_inches(len(images) * 5, 2 * 5)
    plt.savefig(path)
    plt.close(fig)


def save_model(
    epoch,
    generator,
    g_optimizer,
    discriminator,
    d_optimizer,
    checkpoint_path: Path,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "generator_state_dict": generator.state_dict(),
            "g_optimizer_state_dict": g_optimizer.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "d_optimizer_state_dict": d_optimizer.state_dict(),
        },
        checkpoint_path,
    )


def load_model(
    generator, g_optimizer, discriminator, d_optimizer, checkpoint_path: Path
) -> int:
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    d_optimizer.load_state_dict(checkpoint["d_optimizer_state_dict"])
    return checkpoint["epoch"]


def ensure_dir(path: Path):
    if not os.path.exists(path):
        os.makedirs(path)
