# -*- coding: utf-8 -*-
"""
v5.4 - v5 with 500 epochs
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch_fidelity
import os
import numpy as np
import pandas as pd
import random

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")

set_seed(42)

"""
1. Data augmentation (modification allowed)
"""
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.1),
    transforms.RandomApply([
        transforms.RandomChoice([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.RandomCrop(64, padding=4, padding_mode='reflect'),
        ])
    ], p=0.8),
    transforms.ToTensor()])

"""
2. Data loading (modification forbidden, except for the batch_size and multiple workers in the dataloader (num_workers= ...))
"""
dataset = ImageFolder(root="C:\\Users\\dan_t\\OneDrive\\Desktop\\CS424\\VAE_generation\\VAE_generation", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

"""
3. Model architecture (modification allowed, except for "pretrained=False" and "latent_dim=32")
"""
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(ch)

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        out = self.bn2(self.conv2(out))
        return F.leaky_relu(out + x, 0.2)

class Encoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(Encoder, self).__init__()
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
        )
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=32, img_channels=3):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 512)

        def up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2),
            )

        self.up1  = up_block(512, 256)  # 1x1  -> 2x2
        self.res1 = ResBlock(256)
        self.up2  = up_block(256, 128)  # 2x2  -> 4x4
        self.res2 = ResBlock(128)
        self.up3  = up_block(128, 64)   # 4x4  -> 8x8
        self.res3 = ResBlock(64)
        self.up4  = up_block(64, 64)    # 8x8  -> 16x16
        self.res4 = ResBlock(64)
        self.up5  = up_block(64, 32)    # 16x16 -> 32x32
        self.res5 = ResBlock(32)
        self.up6  = nn.Sequential(      # 32x32 -> 64x64
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, img_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 512, 1, 1)
        x = self.res1(self.up1(x))
        x = self.res2(self.up2(x))
        x = self.res3(self.up3(x))
        x = self.res4(self.up4(x))
        x = self.res5(self.up5(x))
        return self.up6(x)

class VAE(nn.Module):
    def __init__(self, latent_dim=32, img_channels=3):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim, img_channels)

    def reparameterize(self, mu, logvar):
        logvar = logvar.clamp(-10, 10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def decode(self, z):
        return self.decoder(z).view(-1, 3, 64, 64)

"""
4. Training (modification allowed)
"""
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def loss_function(recon_x, x, mu, logvar):
        recon_loss = F.l1_loss(recon_x, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_div

    vae = VAE(latent_dim=32).to(device)
    optimizer = optim.AdamW(vae.parameters(), lr=1e-3, weight_decay=1e-4)

    num_epochs = 500
    warmup_epochs = 10

    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs)  # 490 epochs

    for epoch in range(num_epochs):
        vae.train()
        total_loss = 0
        for x, _ in dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = vae(x)
            loss = loss_function(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader.dataset):.4f}, LR: {current_lr:.6f}")

    total_params = sum(p.numel() for p in vae.parameters()) / (1024 * 1024)
    print(f"Model size: {total_params:.2f} M")

    vae.eval()
    with torch.no_grad():
        z = torch.randn(5000, 32).to(device)
        fake_images = vae.decode(z)
    fake_images = (fake_images * 255).clamp(0, 255).to(torch.uint8).float() / 255.0
    save_dir = "C:\\Users\\dan_t\\OneDrive\\Desktop\\CS424\\models\\v5.4\\generated_images\\"
    os.makedirs(save_dir, exist_ok=True)

    for i, img in enumerate(fake_images):
        vutils.save_image(img, f"{save_dir}image_{i}.jpg")

    # metrics = torch_fidelity.calculate_metrics(
    #     input1=save_dir,
    #     input2="C:\\Users\\dan_t\\OneDrive\\Desktop\\CS424\\VAE_generation\\VAE_generation\\train",
    #     cuda=True, fid=True, isc=True)

    # fid_score = metrics['frechet_inception_distance']
    # is_score = metrics['inception_score_mean']
    # s_value = (fid_score / is_score) ** 0.5
    # print(f"FID: {fid_score}")
    # print(f"IS: {is_score}")
    # print(f"S value (GMS): {s_value}")

if __name__ == '__main__':
    main()
