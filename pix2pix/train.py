import argparse
import os
from tabnanny import check
from cv2 import DrawMatchesFlags_DRAW_OVER_OUTIMG
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from pathlib import Path
from typing import List, Tuple
import cv2
from torch.utils.tensorboard import SummaryWriter
from models import Discriminator, Generator
import torch.utils.data as data
import pandas as pd
from dataset import Pix2PixDataset
from utils import (
    save_tensors_as_image_pairs,
    load_model,
    save_model,
    ensure_dir,
)

writer = None


def train_loop(
    generator: Generator,
    discriminator: Discriminator,
    g_optimizer,
    d_optimizer,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    epoch,
    epochs: int,
    device: torch.device,
    data_root: Path,
    lambda_: int,
    checkpoint_path: Path,
    run_name: str,
):
    bce_loss = torch.nn.BCELoss()
    l1_loss = torch.nn.L1Loss()
    discriminator.train()
    generator.train()

    for epoch in range(epoch, epochs):
        for i, (inputs, targets) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Train discriminator
            discriminator.zero_grad()
            output = discriminator(torch.cat((inputs, targets), dim=1))
            y_ones = torch.full(
                output.shape, 1, dtype=torch.float, device=device
            )
            d_loss_real = bce_loss(output, y_ones)
            real_acc = (output > 0.5).type(torch.float32).mean()

            fake_images = generator(inputs)
            output = discriminator(
                torch.cat((inputs, fake_images.detach()), dim=1)
            )
            y_zeros = torch.full(
                output.shape, 0, dtype=torch.float, device=device
            )
            d_loss_fake = bce_loss(output, y_zeros)
            fake_acc = (output <= 0.5).type(torch.float32).mean()

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # Train generator
            generator.zero_grad()
            output = discriminator(torch.cat((inputs, fake_images), dim=1))
            y_ones = torch.full(
                output.shape, 1, dtype=torch.float, device=device
            )
            g_d_loss = bce_loss(output, y_ones)
            g_l1_loss = lambda_ * l1_loss(fake_images, targets)
            g_loss = g_d_loss + g_l1_loss
            g_loss.backward()
            g_optimizer.step()

            # Logs
            step = epoch * len(train_dataloader) + i
            writer.add_scalar("D/Loss", d_loss.mean().item(), global_step=step)
            writer.add_scalar(
                "D/Loss/Fake", d_loss_fake.mean().item(), global_step=step
            )
            writer.add_scalar(
                "D/Loss/Real", d_loss_real.mean().item(), global_step=step
            )
            writer.add_scalar("G/Loss", g_loss.mean().item(), global_step=step)
            writer.add_scalar(
                "G/Loss/D", g_d_loss.mean().item(), global_step=step
            )
            writer.add_scalar(
                "G/Loss/L1", g_l1_loss.mean().item(), global_step=step
            )
            writer.add_scalar("D/Acc/Fake", fake_acc, global_step=step)
            writer.add_scalar("D/Acc/Real", real_acc, global_step=step)

            if i % 100 == 0:
                with torch.no_grad():
                    inputs, _ = next(iter(valid_dataloader))
                    inputs = inputs.to(device)
                    output = generator(inputs)
                    save_tensors_as_image_pairs(
                        inputs,
                        output,
                        data_root
                        / "runs"
                        / run_name
                        / f"valid/{epoch:03d}_{i:04d}.png",
                    )
            if i % 1000 == 0:
                save_model(
                    epoch,
                    generator,
                    g_optimizer,
                    discriminator,
                    d_optimizer,
                    checkpoint_path,
                )
        save_model(
            epoch,
            generator,
            g_optimizer,
            discriminator,
            d_optimizer,
            checkpoint_path,
        )


def train(
    epochs: int,
    batch_size: int,
    workers: int,
    learning_rate: float,
    device: torch.device,
    data_root: Path,
    dataframe_name: str,
    image_size: int,
    lambda_: int,
    beta_1: float,
    beta_2: float,
    run_name: str,
) -> None:
    print("Starting training on device:", device)

    df = pd.read_csv(data_root / dataframe_name)

    train_rows = df["data set"] == "train"
    valid_rows = df["data set"] == "valid"
    valid_ids = range(0, 50, 5)
    train_dataset = Pix2PixDataset(
        f"{data_root}/" + df["sketch_filepaths"][train_rows],
        f"{data_root}/" + df["filepaths"][train_rows],
        image_size,
    )
    valid_dataset = Pix2PixDataset(
        list(
            f"{data_root}/"
            + df["sketch_filepaths"][valid_rows].iloc[valid_ids]
        ),
        list(f"{data_root}/" + df["filepaths"][valid_rows].iloc[valid_ids]),
        image_size,
    )

    save_tensors_as_image_pairs(
        torch.stack([valid_dataset[i][0] for i in range(len(valid_ids))]),
        torch.stack([valid_dataset[i][1] for i in range(len(valid_ids))]),
        data_root / "validation_birds_samples.png",
    )
    generator = Generator(1, 3).to(device)
    discriminator = Discriminator(3 + 1).to(device)

    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=learning_rate, betas=[beta_1, beta_2]
    )
    g_optimizer = torch.optim.Adam(
        generator.parameters(), lr=learning_rate, betas=[beta_1, beta_2]
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )

    epoch = 0
    checkpoint_path = data_root / "runs" / run_name / "checkpoint.pth"
    ensure_dir(data_root / "runs" / run_name / "valid")
    ensure_dir(checkpoint_path.parent)

    if checkpoint_path.exists():
        print("Loading a saved model")
        epoch = (
            load_model(
                generator,
                g_optimizer,
                discriminator,
                d_optimizer,
                checkpoint_path,
            )
            + 1
        )
        print(f"Continuing from epoch {epoch}")
    else:
        print("Starting a new run")

    train_loop(
        generator,
        discriminator,
        g_optimizer,
        d_optimizer,
        train_dataloader,
        valid_dataloader,
        epoch,
        epochs,
        device,
        data_root,
        lambda_,
        checkpoint_path,
        run_name,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, help="number of epochs", default=50
    )
    parser.add_argument(
        "--workers", type=int, help="number of data loading workers", default=2
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="batch size"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument(
        "--beta_1", type=float, default=0.5, help="beta 1 for Adam"
    )
    parser.add_argument(
        "--beta_2", type=float, default=0.999, help="beta 2 for Adam"
    )
    parser.add_argument(
        "--lambda",
        type=int,
        default=100,
        help="lambda coefficient for L1 loss",
        dest="lambda_",
    )
    parser.add_argument(
        "--cuda", default=True, action="store_true", help="enable cuda"
    )
    parser.add_argument(
        "--data_root", type=str, default="data", help="data directory path"
    )
    parser.add_argument(
        "--dataframe_name",
        type=str,
        default="sketch_birds.csv",
        help="name of dataframe with images paths",
    )
    parser.add_argument(
        "--run_name", type=str, help="experiment name", required=True
    )
    parser.add_argument(
        "--image_size", type=int, default=256, help="image size to resize to"
    )
    args = parser.parse_args()
    writer = SummaryWriter(comment=args.run_name)
    train(
        args.epochs,
        args.batch_size,
        args.workers,
        args.lr,
        torch.device("cuda") if args.cuda else torch.device("cpu"),
        Path(args.data_root),
        args.dataframe_name,
        args.image_size,
        args.lambda_,
        args.beta_1,
        args.beta_2,
        args.run_name,
    )
    writer.flush()
