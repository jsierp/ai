import argparse
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchsummary import summary
from discriminator import Discriminator
from generator import Generator
from pathlib import Path


IMAGE_SIZE = 28


def save_images(images: np.ndarray, path: Path) -> None:
    w = int(np.sqrt(len(images)))
    h = (len(images) + w - 1) // w
    for i, im in enumerate(images):
        plt.subplot(h, w, 1 + i)
        plt.axis("off")
        plt.imshow(im, cmap="gray_r")
    plt.savefig(path)


def start_new_epoch(logs: list) -> None:
    logs.append(
        {
            "real_acc": [],
            "fake_acc": [],
            "generator_faked_rate": [],
            "d_loss": [],
            "g_loss": [],
            "length": [],
        }
    )


def log_batch(
    log: dict,
    real_acc: float,
    fake_acc: float,
    generator_faked_rate: float,
    d_loss: float,
    g_loss: float,
    length: int,
) -> None:
    log["real_acc"].append(real_acc)
    log["fake_acc"].append(fake_acc)
    log["generator_faked_rate"].append(generator_faked_rate)
    log["d_loss"].append(d_loss)
    log["g_loss"].append(g_loss)
    log["length"].append(length)


def save_logs(logs: list, path: Path) -> None:
    with open(path, "wb") as file:
        pickle.dump(logs, file)


def main(
    epochs: int,
    batch_size: int,
    workers: int,
    latent_dim: int,
    learning_rate: float,
    device: torch.device,
    data_root: Path,
    preview_length: int,
) -> None:
    print("Device", device)

    if not os.path.exists(data_root):
        os.makedirs(data_root)
    if not os.path.exists(data_root / "epochs"):
        os.makedirs(data_root / "epochs")

    train_data = datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    save_images(
        [train_data[i][0][0] for i in range(preview_length)],
        data_root / "mnist_samples.png",
    )
    discriminator = Discriminator().to(device)
    print(summary(discriminator, input_size=(1, IMAGE_SIZE, IMAGE_SIZE)))
    generator = Generator().to(device)
    print(summary(generator, input_size=(latent_dim,)))

    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=learning_rate, betas=[0.5, 0.999]
    )
    g_optimizer = torch.optim.Adam(
        generator.parameters(), lr=learning_rate, betas=[0.5, 0.999]
    )
    criterion = torch.nn.BCELoss()
    fixed_points = torch.randn(preview_length, latent_dim, device=device)
    dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=workers
    )

    _ = discriminator.train()
    _ = generator.train()

    logs = []
    for epoch in range(epochs):
        start_new_epoch(logs)
        for i, (images, _) in enumerate(dataloader):
            # Train discriminator
            length = images.shape[0]
            X_real = images.to(device)
            y_real = torch.full((length,), 1, dtype=torch.float, device=device)

            discriminator.zero_grad()
            output = discriminator(X_real)
            d_loss = criterion(output, y_real)
            d_loss.backward()

            real_acc = int(((output > 0.5)).sum()) / length

            latent_points = torch.randn(length, latent_dim, device=device)
            X_fake = generator(latent_points)
            y_fake = torch.full((length,), 0, dtype=torch.float, device=device)

            output = discriminator(X_fake.detach())
            d_loss_fake = criterion(output, y_fake)
            d_err = d_loss_fake.mean().item() + d_loss.mean().item()

            d_loss_fake.backward()
            d_optimizer.step()

            fake_acc = int(((output <= 0.5)).sum()) / length

            # Train generator
            generator.zero_grad()
            latent_points = torch.randn(length, latent_dim, device=device)
            y_generator = torch.full((length,), 1, dtype=torch.float, device=device)
            output = discriminator(X_fake)
            g_loss = criterion(output, y_generator)
            g_err = g_loss.mean().item()

            g_loss.backward()
            g_optimizer.step()

            generator_faked_rate = int(((output > 0.5)).sum()) / length

            log_batch(
                logs[-1],
                real_acc,
                fake_acc,
                generator_faked_rate,
                float(d_loss),
                float(g_loss),
                length,
            )
            if i % 50 == 0:
                print(
                    f"Epoch {epoch+1}: {i+1}/{len(dataloader)} real_acc={real_acc:.3f} fake_acc={fake_acc:.3f} generator_faked_rate={generator_faked_rate:.3f} d_loss: {d_err}, g_loss: {g_err}"
                )
                X_fake = generator(fixed_points)
                save_images(
                    X_fake.detach().cpu().numpy().reshape((-1, IMAGE_SIZE, IMAGE_SIZE)),
                    data_root / "epochs" / f"{epoch}_{i}.png",
                )
        save_logs(logs, data_root / "logs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, help="number of epochs", default=50)
    parser.add_argument(
        "--workers", type=int, help="number of data loading workers", default=2
    )
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=100,
        help="number of dimensions in latent space",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="learning rate, default=0.0001"
    )
    parser.add_argument("--cuda", default=True, action="store_true", help="enable cuda")
    parser.add_argument(
        "--data_root", type=str, default="data", help="data directory path"
    )
    parser.add_argument(
        "--preview_length", type=int, default=25, help="number of samples in preview"
    )
    args = parser.parse_args()
    main(
        args.epochs,
        args.batch_size,
        args.workers,
        args.latent_dim,
        args.lr,
        torch.device("cuda") if args.cuda else torch.device("cpu"),
        Path(args.data_root),
        args.preview_length,
    )
