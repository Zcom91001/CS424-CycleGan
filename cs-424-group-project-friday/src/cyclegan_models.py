import torch
import torch.nn as nn


class ResnetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)

class ResnetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, ngf=64, n_blocks=6):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]

        c = ngf
        for _ in range(2):
            layers += [
                nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(c * 2),
                nn.ReLU(inplace=True),
            ]
            c *= 2

        for _ in range(n_blocks):
            layers.append(ResnetBlock(c))

        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(c, c // 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(c // 2),
                nn.ReLU(inplace=True),
            ]
            c //= 2

        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(c, out_channels, kernel_size=7, stride=1, padding=0),
            nn.Tanh(),
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.net(x)


def init_weights(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def build_cyclegan_models(ngf=64, ndf=64, n_blocks=6, device="gpu"):
    g_ab = ResnetGenerator(3, 3, ngf=ngf, n_blocks=n_blocks).to(device)
    g_ba = ResnetGenerator(3, 3, ngf=ngf, n_blocks=n_blocks).to(device)
    d_a = PatchDiscriminator(3, ndf=ndf).to(device)
    d_b = PatchDiscriminator(3, ndf=ndf).to(device)

    g_ab.apply(init_weights)
    g_ba.apply(init_weights)
    d_a.apply(init_weights)
    d_b.apply(init_weights)
    return g_ab, g_ba, d_a, d_b
