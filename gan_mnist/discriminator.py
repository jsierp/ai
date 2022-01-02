import torch


class Discriminator(torch.nn.Module):
    def __init__(self, in_shape=(28, 28)):
        super(Discriminator, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.4),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.4),
        )
        self.linear = torch.nn.Linear(64 * 7 * 7, 1)

    def forward(self, x):
        return torch.sigmoid(
            self.linear(torch.flatten(self.main(x), start_dim=1))
        ).squeeze(1)
