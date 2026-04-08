import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


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


def _apply_spectral_norm(module, enabled):
    return spectral_norm(module) if enabled else module


class NLayerPatchDiscriminator(nn.Module):
    """
    Configurable PatchGAN discriminator with optional spectral normalization.
    """

    def __init__(
        self,
        in_channels=3,
        ndf=64,
        n_layers=3,
        use_spectral_norm=False,
        use_instance_norm=True,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1")

        kw = 4
        padw = 1
        sequence = [
            _apply_spectral_norm(
                nn.Conv2d(in_channels, ndf, kernel_size=kw, stride=2, padding=padw),
                use_spectral_norm,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1
        for layer_idx in range(1, n_layers):
            prev_nf_mult = nf_mult
            nf_mult = min(2 ** layer_idx, 8)
            sequence.extend(
                [
                    _apply_spectral_norm(
                        nn.Conv2d(
                            ndf * prev_nf_mult,
                            ndf * nf_mult,
                            kernel_size=kw,
                            stride=2,
                            padding=padw,
                            bias=not use_instance_norm,
                        ),
                        use_spectral_norm,
                    ),
                ]
            )
            if use_instance_norm:
                sequence.append(nn.InstanceNorm2d(ndf * nf_mult))
            sequence.append(nn.LeakyReLU(0.2, inplace=True))

        prev_nf_mult = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence.extend(
            [
                _apply_spectral_norm(
                    nn.Conv2d(
                        ndf * prev_nf_mult,
                        ndf * nf_mult,
                        kernel_size=kw,
                        stride=1,
                        padding=padw,
                        bias=not use_instance_norm,
                    ),
                    use_spectral_norm,
                ),
            ]
        )
        if use_instance_norm:
            sequence.append(nn.InstanceNorm2d(ndf * nf_mult))
        sequence.extend(
            [
                nn.LeakyReLU(0.2, inplace=True),
                _apply_spectral_norm(
                    nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),
                    use_spectral_norm,
                ),
            ]
        )

        self.net = nn.Sequential(*sequence)

    def forward(self, x):
        return self.net(x)


class MultiScaleDiscriminator(nn.Module):
    """
    Run the same PatchGAN discriminator at multiple image scales.
    """

    def __init__(
        self,
        in_channels=3,
        ndf=64,
        num_scales=3,
        n_layers=3,
        use_spectral_norm=False,
        use_instance_norm=True,
    ):
        super().__init__()
        if num_scales < 1:
            raise ValueError("num_scales must be at least 1")

        self.discriminators = nn.ModuleList(
            [
                NLayerPatchDiscriminator(
                    in_channels=in_channels,
                    ndf=ndf,
                    n_layers=n_layers,
                    use_spectral_norm=use_spectral_norm,
                    use_instance_norm=use_instance_norm,
                )
                for _ in range(num_scales)
            ]
        )
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x):
        outputs = []
        current = x
        for idx, discriminator in enumerate(self.discriminators):
            outputs.append(discriminator(current))
            if idx < len(self.discriminators) - 1:
                current = self.downsample(current)
        return outputs


class PatchDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels=3,
        ndf=64,
        n_layers=3,
        use_spectral_norm=False,
        use_instance_norm=True,
    ):
        super().__init__()
        self.net = NLayerPatchDiscriminator(
            in_channels=in_channels,
            ndf=ndf,
            n_layers=n_layers,
            use_spectral_norm=use_spectral_norm,
            use_instance_norm=use_instance_norm,
        ).net

    def forward(self, x):
        return self.net(x)


def init_weights(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        weight = getattr(module, "weight_orig", module.weight)
        nn.init.normal_(weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def build_cyclegan_models(ngf=64, ndf=64, n_blocks=6, device="gpu", discriminator_cfg=None):
    discriminator_cfg = discriminator_cfg or {}
    if not isinstance(discriminator_cfg, dict):
        raise TypeError("model.discriminator must be a mapping when provided.")
    disc_type = str(discriminator_cfg.get("type", "patchgan")).strip().lower()
    disc_n_layers = int(discriminator_cfg.get("n_layers", 3))
    disc_use_spectral_norm = bool(discriminator_cfg.get("use_spectral_norm", False))
    disc_use_instance_norm = bool(discriminator_cfg.get("use_instance_norm", True))

    g_ab = ResnetGenerator(3, 3, ngf=ngf, n_blocks=n_blocks).to(device)
    g_ba = ResnetGenerator(3, 3, ngf=ngf, n_blocks=n_blocks).to(device)

    if disc_type == "patchgan":
        discriminator_factory = lambda: PatchDiscriminator(
            3,
            ndf=ndf,
            n_layers=disc_n_layers,
            use_spectral_norm=disc_use_spectral_norm,
            use_instance_norm=disc_use_instance_norm,
        )
    elif disc_type == "multiscale":
        disc_num_scales = int(discriminator_cfg.get("num_scales", 3))
        discriminator_factory = lambda: MultiScaleDiscriminator(
            in_channels=3,
            ndf=ndf,
            num_scales=disc_num_scales,
            n_layers=disc_n_layers,
            use_spectral_norm=disc_use_spectral_norm,
            use_instance_norm=disc_use_instance_norm,
        )
    else:
        raise ValueError(f"Unsupported discriminator type: {disc_type}")

    d_a = discriminator_factory().to(device)
    d_b = discriminator_factory().to(device)

    g_ab.apply(init_weights)
    g_ba.apply(init_weights)
    d_a.apply(init_weights)
    d_b.apply(init_weights)
    return g_ab, g_ba, d_a, d_b
