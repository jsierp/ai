import torch


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2),
        )
        self.linear = torch.nn.Linear(256 * 4 * 4, 1)

    def forward(self, x):
        return torch.sigmoid(
            self.linear(torch.flatten(self.main(x), start_dim=1))
        ).squeeze(1)
