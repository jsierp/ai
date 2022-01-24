import torch
import torch.nn as nn


class DownConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_ch, out_ch, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConvDropout(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConvDropout, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_ch, out_ch, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(out_ch),
            nn.Dropout(0.5),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Generator, self).__init__()

        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.down2 = DownConv(64, 128)
        self.down3 = DownConv(128, 256)
        self.down4 = DownConv(256, 512)
        self.down5 = DownConv(512, 512)
        self.down6 = DownConv(512, 512)
        self.down7 = DownConv(512, 512)
        self.down8 = DownConv(512, 512)
        self.up1 = UpConvDropout(512, 512)
        self.up2 = UpConvDropout(1024, 512)
        self.up3 = UpConvDropout(1024, 512)
        self.up4 = UpConv(1024, 512)
        self.up5 = UpConv(1024, 256)
        self.up6 = UpConv(512, 128)
        self.up7 = UpConv(256, 64)
        self.outconv = nn.Sequential(
            nn.ConvTranspose2d(
                128, out_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x7 = self.down7(x6)
        x8 = self.down8(x7)
        x = self.up1(x8)
        x = self.up2(torch.cat((x7, x), dim=1))
        x = self.up3(torch.cat((x6, x), dim=1))
        x = self.up4(torch.cat((x5, x), dim=1))
        x = self.up5(torch.cat((x4, x), dim=1))
        x = self.up6(torch.cat((x3, x), dim=1))
        x = self.up7(torch.cat((x2, x), dim=1))
        x = self.outconv(torch.cat((x1, x), dim=1))
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()

        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.down2 = DownConv(64, 128)
        self.down3 = DownConv(128, 256)
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        self.outconv = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.outconv(x)
        return x
