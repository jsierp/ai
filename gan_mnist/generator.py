import torch


class Generator(torch.nn.Module):
    def __init__(self, in_length=100):
        super(Generator, self).__init__()
        self.linear = torch.nn.Linear(in_length, 256 * 4 * 4)
        self.lrelu = torch.nn.LeakyReLU(0.2)
        self.generator = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.2),
            torch.nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        x = self.lrelu(self.linear(x)).reshape((-1, 256, 4, 4))
        x = self.generator(x)
        return x
