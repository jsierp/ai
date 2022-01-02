import torch


class Generator(torch.nn.Module):
    def __init__(self, in_length=100):
        super(Generator, self).__init__()
        self.linear = torch.nn.Linear(in_length, 128 * 7 * 7)
        self.convtrans2d_1 = torch.nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2)
        self.convtrans2d_2 = torch.nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2)
        self.conv = torch.nn.Conv2d(128, 1, kernel_size=7)
        self.lrelu = torch.nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.lrelu(self.linear(x)).reshape((-1, 128, 7, 7))
        x = self.lrelu(self.convtrans2d_1(x))
        x = self.lrelu(self.convtrans2d_2(x))
        return torch.sigmoid(self.conv(x))
