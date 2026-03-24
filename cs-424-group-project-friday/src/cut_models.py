import torch
import torch.nn as nn
import torch.nn.functional as F


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


class CutGenerator(nn.Module):
    """
    ResNet generator that can return intermediate features for PatchNCE.
    """

    def __init__(self, in_channels=3, out_channels=3, ngf=64, n_blocks=6):
        super().__init__()

        self.stem = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
        )

        self.resblocks = nn.ModuleList([ResnetBlock(ngf * 4) for _ in range(n_blocks)])

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(
                ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(
                ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, kernel_size=7, stride=1, padding=0),
            nn.Tanh(),
        )

    def forward(self, x, return_features=False):
        feats = []
        x = self.stem(x)
        feats.append(x)  # 0
        x = self.down1(x)
        feats.append(x)  # 1
        x = self.down2(x)
        feats.append(x)  # 2
        for rb in self.resblocks:
            x = rb(x)
        feats.append(x)  # 3
        x = self.up1(x)
        feats.append(x)  # 4
        x = self.up2(x)
        feats.append(x)  # 5
        out = self.head(x)
        if return_features:
            return out, feats
        return out


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


class PatchSampleMLP(nn.Module):
    """
    Feature projector for PatchNCE.
    """

    def __init__(self, in_channels_list, proj_dim=256):
        super().__init__()
        self.mlps = nn.ModuleList()
        for c in in_channels_list:
            self.mlps.append(
                nn.Sequential(
                    nn.Linear(c, proj_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(proj_dim, proj_dim),
                )
            )

    def _sample_patches(self, feat, num_patches, patch_ids=None):
        # feat: [B, C, H, W] -> [B, HW, C]
        b, c, h, w = feat.shape
        feat = feat.permute(0, 2, 3, 1).reshape(b, h * w, c)
        if patch_ids is None:
            ids = torch.randint(0, h * w, (num_patches,), device=feat.device)
        else:
            ids = patch_ids
        x = feat[:, ids, :]  # [B, P, C]
        return x, ids

    def forward(self, feats, num_patches=256, patch_ids=None):
        pooled = []
        ids_out = []
        for i, feat in enumerate(feats):
            x, ids = self._sample_patches(feat, num_patches=num_patches, patch_ids=None if patch_ids is None else patch_ids[i])
            b, p, c = x.shape
            x = x.reshape(b * p, c)
            x = self.mlps[i](x)
            x = F.normalize(x, dim=1)
            x = x.view(b, p, -1)
            pooled.append(x)
            ids_out.append(ids)
        return pooled, ids_out


class PatchNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, feat_q, feat_k):
        # feat_q, feat_k: [B, P, C]
        b, p, c = feat_q.shape
        q = feat_q.reshape(b * p, c)
        k = feat_k.reshape(b * p, c).detach()

        # Positive logits: dot product with matching patch.
        l_pos = torch.sum(q * k, dim=1, keepdim=True)  # [BP, 1]

        # Negative logits: full patch queue within minibatch.
        l_neg = torch.mm(q, k.t())  # [BP, BP]
        diag = torch.eye(l_neg.size(0), device=l_neg.device, dtype=torch.bool)
        l_neg.masked_fill_(diag, -10.0)

        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return self.ce(logits, targets)


def init_weights(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)
