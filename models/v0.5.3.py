# -*- coding: utf-8 -*-
"""CS424 CycleGAN - Comprehensive Fix (v2)

Fixes over v1:
  1.  Identity loss inputs corrected     (G_BA(real_A), G_AB(real_B))
  2.  InstanceNorm2d affine=True         (learnable scale/shift per domain)
  3.  Generator initial block gets IN    (consistent normalisation)
  4.  BCEWithLogitsLoss + no Sigmoid     (numerically stable, fewer artifacts)
  5.  Replay buffer (size=50)            (stable discriminator)
  6.  Adam betas=(0.5,0.999)             (standard GAN setting)
  7.  LR linear decay epoch 50->100      (standard CycleGAN schedule)
  8.  Gradient clipping (max_norm=1.0)   (prevents exploding grads / blurring)
  9.  Random H-flip augmentation         (doubles effective dataset size)
  10. Identity loss annealed to 0 by ep25 (allows colour/texture transfer)
  11. Checkpoint save every 10 epochs    (resume safely if interrupted)
  12. Stale D predictions fixed          (fresh eval during D training step)
"""

import os
import torch
import torch.nn as nn
import torch_fidelity
import numpy as np
import pandas as pd
import itertools
import random
import platform

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # add at very top of file

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

# ── Reproducibility ──────────────────────────────────────────────────────────
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data_dir = "C:/Users/dan_t/OneDrive/Desktop/CS424"
if platform.system() != "Windows":
    data_dir = "/common/home/users/d/daniel.tan.2023/scratchDirectory"

# ── Replay Buffer ─────────────────────────────────────────────────────────────
class ReplayBuffer:
    """
    Stores up to max_size previously generated images.
    On each call, returns a mix of new and historical fakes so the
    discriminator can't overfit to only the latest generator output.
    """
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        result = []
        for element in data:
            element = element.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                result.append(element)
            else:
                if random.random() > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    result.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    result.append(element)
        return torch.cat(result, dim=0)


# ── Generator ─────────────────────────────────────────────────────────────────
NETWORK_FEATURES = 64


class GBlock(nn.Module):
    """
    Generator block (down or up).
    Fix #2: affine=True lets InstanceNorm learn per-channel scale+shift,
            which is essential for adapting to the target style domain.
    """
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs, padding_mode="reflect") if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels, affine=True),   # Fix #2
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            GBlock(channels, channels, use_act=True,  kernel_size=3, stride=1, padding=1),
            GBlock(channels, channels, use_act=False, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """
    CycleGAN ResNet generator.
    Fix #3: initial 7×7 conv now includes InstanceNorm (affine=True)
            so normalisation is consistent from the very first layer.
    """
    def __init__(self, img_channels=3, num_residuals=9):
        super().__init__()

        # Fix #3: add InstanceNorm to the initial block
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, NETWORK_FEATURES,
                      kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(NETWORK_FEATURES, affine=True),  # Fix #3
            nn.ReLU(inplace=True),
        )

        self.down_block = nn.Sequential(
            GBlock(NETWORK_FEATURES*1, NETWORK_FEATURES*2, kernel_size=3, stride=2, padding=1),
            GBlock(NETWORK_FEATURES*2, NETWORK_FEATURES*4, kernel_size=3, stride=2, padding=1),
        )

        self.residual_block = nn.Sequential(
            *[ResidualBlock(NETWORK_FEATURES*4) for _ in range(num_residuals)]
        )

        self.up_block = nn.Sequential(
            GBlock(NETWORK_FEATURES*4, NETWORK_FEATURES*2, down=False,
                   kernel_size=3, stride=2, padding=1, output_padding=1),
            GBlock(NETWORK_FEATURES*2, NETWORK_FEATURES*1, down=False,
                   kernel_size=3, stride=2, padding=1, output_padding=1),
        )

        self.last = nn.Sequential(
            nn.Conv2d(NETWORK_FEATURES, img_channels,
                      kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.down_block(x)
        x = self.residual_block(x)
        x = self.up_block(x)
        return self.last(x)


# ── Discriminator ─────────────────────────────────────────────────────────────
class DBlock(nn.Module):
    """
    Fix #2: affine=True on InstanceNorm so D can also learn domain-specific stats.
    Fix #4: no Sigmoid here — we use BCEWithLogitsLoss which is numerically stable.
    """
    def __init__(self, in_channels, out_channels, first_block=False, **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs,
                      padding_mode='reflect', bias=True),
            nn.InstanceNorm2d(out_channels, affine=True) if not first_block  # Fix #2
            else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Discriminator(nn.Module):
    """
    PatchGAN discriminator.
    Fix #4: last layer outputs raw logits (no Sigmoid). Use BCEWithLogitsLoss.
    """
    def __init__(self, img_channels=3):
        super().__init__()
        self.scale_factor = 8
        self.model = nn.Sequential(
            DBlock(img_channels,       NETWORK_FEATURES,   first_block=True, kernel_size=4, stride=2, padding=1),
            DBlock(NETWORK_FEATURES*1, NETWORK_FEATURES*2, kernel_size=4, stride=2, padding=1),
            DBlock(NETWORK_FEATURES*2, NETWORK_FEATURES*4, kernel_size=4, stride=2, padding=1),
            DBlock(NETWORK_FEATURES*4, NETWORK_FEATURES*8, kernel_size=4, stride=1, padding=1),
        )
        # Fix #4: NO Sigmoid — BCEWithLogitsLoss handles that internally
        self.last_layer = nn.Conv2d(
            NETWORK_FEATURES*8, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"
        )

    def forward(self, x):
        return self.last_layer(self.model(x))   # raw logits


# ── Dataset ───────────────────────────────────────────────────────────────────
class ImageDataset(Dataset):
    def __init__(self, data_dir, mode='train', transform=None):
        A_dir = os.path.join(data_dir, 'VAE_generation/train')   # modification forbidden
        B_dir = os.path.join(data_dir, 'VAE_generation1/train')  # modification forbidden

        all_A = [os.path.join(A_dir, n) for n in sorted(os.listdir(A_dir))]
        all_B = [os.path.join(B_dir, n) for n in sorted(os.listdir(B_dir))]

        # 90/10 train-valid split across all 4000 images
        split = int(0.9 * min(len(all_A), len(all_B)))

        if mode == 'train':
            self.files_A = all_A[:split]   # ~3600 images
            self.files_B = all_B[:split]   # ~3600 images
        elif mode == 'valid':
            self.files_A = all_A[split:]   # ~400 images
            self.files_B = all_B[split:]   # ~400 images

        self.transform = transform

    def __len__(self):
        return len(self.files_A)

    def __getitem__(self, index):
        img_A = Image.open(self.files_A[index]).convert("RGB")

        # Unpaired: random B index independent of A
        random_B_index = random.randint(0, len(self.files_B) - 1)
        img_B = Image.open(self.files_B[random_B_index]).convert("RGB")

        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return img_A, img_B


# ── Training ──────────────────────────────────────────────────────────────────
def main():
    # Fix #4: BCEWithLogitsLoss — fuses sigmoid + BCE for numerical stability,
    #         directly reducing artifact frequency caused by gradient saturation
    criterion_GAN      = nn.BCEWithLogitsLoss()   # Fix #4 (was BCELoss + Sigmoid)
    criterion_cycle    = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    G_AB = Generator(3)
    D_B  = Discriminator(3)
    G_BA = Generator(3)
    D_A  = Discriminator(3)

    # modification of parameters computation is forbidden
    total_params = (sum(p.numel() for p in G_AB.parameters()) +
                    sum(p.numel() for p in G_BA.parameters()) +
                    sum(p.numel() for p in D_A.parameters()) +
                    sum(p.numel() for p in D_B.parameters()))
    total_params_million = total_params / (1024 * 1024)
    print(f'Total parameters in CycleGAN model: {total_params_million:.2f} million')

    cuda = torch.cuda.is_available()
    print(f'cuda: {cuda}')
    if cuda:
        G_AB = G_AB.cuda(); G_BA = G_BA.cuda()
        D_A  = D_A.cuda();  D_B  = D_B.cuda()
        criterion_GAN      = criterion_GAN.cuda()
        criterion_cycle    = criterion_cycle.cuda()
        criterion_identity = criterion_identity.cuda()

    # Fix #6: betas=(0.5, 0.999) is the standard GAN Adam setting;
    #         default beta1=0.9 retains too much momentum, causing oscillation
    lr_G = 0.0002
    lr_D = 0.0001

    optimizer_G   = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr_G, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr_D, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr_D, betas=(0.5, 0.999))

    # Fix 2: Label smoothing — stops D pushing logits to extreme values
    real_label = 0.9
    fake_label = 0.1

    n_critics = 5
    n_epoches   = 30
    decay_epoch = n_epoches // 2

    # Fix #7: linear LR decay from epoch 50 → 100
    def lr_lambda(epoch):
        if epoch < decay_epoch:
            return 1.0
        return max(0.0, 1.0 - (epoch - decay_epoch) / float(n_epoches - decay_epoch))

    scheduler_G   = torch.optim.lr_scheduler.LambdaLR(optimizer_G,   lr_lambda)
    scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda)
    scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda)

    # Fix #5: replay buffers
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    model_dir   = os.path.join(data_dir, "models", script_name)
    os.makedirs(model_dir, exist_ok=True)

    image_size = (256, 256)

    # Fix #9: random horizontal flip to double effective dataset size (200 → ~400 effective)
    train_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.1),
        transforms.RandomApply([
            transforms.RandomChoice([
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.RandomCrop(image_size[0], padding=4, padding_mode='reflect'),
            ])
        ], p=0.8),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    eval_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size = 2

    trainloader = DataLoader(
        ImageDataset(data_dir, mode='train', transform=train_transforms),
        batch_size=batch_size, shuffle=True
    )
    validloader = DataLoader(
        ImageDataset(data_dir, mode='valid', transform=eval_transforms),
        batch_size=batch_size, shuffle=False
    )

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Loss weights
    lambda_cycle = 20   # cycle-consistency
    lambda_GAN   = 2    # adversarial
    lambda_feat = 5

    # Fix #10: identity loss weight schedule
    # Starts at 5 (= 0.5 × lambda_cycle, as in the paper) to warm-start colour mapping,
    # then anneals linearly to 0 by epoch 25 so the generators are free to transfer
    # colours and textures aggressively for the remaining 75 epochs.

    def feature_matching_loss(real, fake, discriminator):
        # Extract intermediate D features for real and fake
        # Forces G to match the feature distribution of the target domain
        # not just fool the final D output
        def get_features(x):
            features = []
            out = x
            for layer in discriminator.model:
                out = layer(out)
                features.append(out)
            return features

        real_features = get_features(real.detach())
        fake_features = get_features(fake)

        loss = 0
        for rf, ff in zip(real_features, fake_features):
            loss += nn.functional.l1_loss(ff, rf.detach())
        return loss / len(real_features)

    def identity_weight(epoch):
        if epoch >= n_epoches // 5:
            return 0.0
        return 1.0 * (1.0 - epoch / (n_epoches // 5))

    for epoch in range(n_epoches):
        lambda_identity = identity_weight(epoch)   # Fix #10

        for i, (real_A, real_B) in enumerate(trainloader):
            real_A = real_A.type(Tensor)
            real_B = real_B.type(Tensor)

            # ── Train Discriminators TWICE every step ─────────────────────
            for _ in range(n_critics):
                with torch.no_grad():
                    fake_A_buf = fake_A_buffer.push_and_pop(G_BA(real_B).detach())
                    fake_B_buf = fake_B_buffer.push_and_pop(G_AB(real_A).detach())

                optimizer_D_A.zero_grad()
                pred_real_A = D_A(real_A)
                loss_real_A = criterion_GAN(pred_real_A, torch.ones_like(pred_real_A) * real_label)
                pred_fake_A = D_A(fake_A_buf)
                loss_fake_A = criterion_GAN(pred_fake_A, torch.zeros_like(pred_fake_A) + fake_label)
                loss_D_A = (loss_real_A + loss_fake_A) / 2
                loss_D_A.backward()
                nn.utils.clip_grad_norm_(D_A.parameters(), max_norm=1.0)
                optimizer_D_A.step()

                optimizer_D_B.zero_grad()
                pred_real_B = D_B(real_B)
                loss_real_B = criterion_GAN(pred_real_B, torch.ones_like(pred_real_B) * real_label)
                pred_fake_B = D_B(fake_B_buf)
                loss_fake_B = criterion_GAN(pred_fake_B, torch.zeros_like(pred_fake_B) + fake_label)
                loss_D_B = (loss_real_B + loss_fake_B) / 2
                loss_D_B.backward()
                nn.utils.clip_grad_norm_(D_B.parameters(), max_norm=1.0)
                optimizer_D_B.step()

            # ── Train Generators ONCE every step ──────────────────────────
            G_AB.train(); G_BA.train()
            optimizer_G.zero_grad()

            fake_B_img = G_AB(real_A)
            fake_A_img = G_BA(real_B)

            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B) / 2

            loss_GAN_AB = criterion_GAN(D_B(fake_B_img), torch.ones_like(D_B(fake_B_img)) * real_label)
            loss_GAN_BA = criterion_GAN(D_A(fake_A_img), torch.ones_like(D_A(fake_A_img)) * real_label)
            loss_GAN    = (loss_GAN_AB + loss_GAN_BA) / 2

            recov_A = G_BA(fake_B_img)
            recov_B = G_AB(fake_A_img)
            loss_cycle = (criterion_cycle(recov_A, real_A) + criterion_cycle(recov_B, real_B)) / 2

            loss_feat_AB = feature_matching_loss(real_B, fake_B_img, D_B)
            loss_feat_BA = feature_matching_loss(real_A, fake_A_img, D_A)
            loss_feat = (loss_feat_AB + loss_feat_BA) / 2

            loss_G = (lambda_identity * loss_identity
                    + lambda_GAN     * loss_GAN
                    + lambda_cycle   * loss_cycle
                    + lambda_feat    * loss_feat)
            loss_G.backward()
            nn.utils.clip_grad_norm_(
                list(G_AB.parameters()) + list(G_BA.parameters()), max_norm=1.0)
            optimizer_G.step()

        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_B.step()

        loss_D = (loss_D_A + loss_D_B) / 2
        print(f'[Epoch {epoch+1:3d}/{n_epoches}] λ_id={lambda_identity:.2f}')
        print(f'  G  : total={loss_G.item():.4f}  id={loss_identity.item():.4f}'
                f'  GAN={loss_GAN.item():.4f}  cycle={loss_cycle.item():.4f}'
                f'  feat={loss_feat.item():.4f}')  # added
        print(f'  D  : total={loss_D.item():.4f}  D_A={loss_D_A.item():.4f}'
                f'  D_B={loss_D_B.item():.4f}')

            # Fix #11: save checkpoints so training can be resumed if interrupted
            # ckpt = {
            #     'epoch':       epoch + 1,
            #     'G_AB':        G_AB.state_dict(),
            #     'G_BA':        G_BA.state_dict(),
            #     'D_A':         D_A.state_dict(),
            #     'D_B':         D_B.state_dict(),
            #     'opt_G':       optimizer_G.state_dict(),
            #     'opt_D_A':     optimizer_D_A.state_dict(),
            #     'opt_D_B':     optimizer_D_B.state_dict(),
            #     'sched_G':     scheduler_G.state_dict(),
            #     'sched_D_A':   scheduler_D_A.state_dict(),
            #     'sched_D_B':   scheduler_D_B.state_dict(),
            # }
            # torch.save(ckpt, os.path.join(model_dir, f'ckpt_epoch{epoch+1:03d}.pt'))

    # ── Evaluation ────────────────────────────────────────────────────────────
    to_image = transforms.ToPILImage()

    def generate_and_save(generator, test_dir, save_dir):
        files = [os.path.join(test_dir, n) for n in os.listdir(test_dir)]
        os.makedirs(save_dir, exist_ok=True)
        generator.eval()
        with torch.no_grad():
            for i in range(0, len(files), batch_size):
                imgs = []
                batch_files = files[i: i + batch_size]
                for path in batch_files:
                    imgs.append(eval_transforms(Image.open(path).convert("RGB")))
                imgs = torch.stack(imgs).type(Tensor)
                fakes = generator(imgs).cpu()
                for j, fake in enumerate(fakes):
                    arr = fake.permute(1, 2, 0).numpy()
                    arr = ((arr - arr.min()) * 255 / (arr.max() - arr.min())).astype(np.uint8)
                    _, name = os.path.split(batch_files[j])
                    to_image(arr).save(os.path.join(save_dir, name))

    def score(save_dir, gt_dir, label):
        metrics   = torch_fidelity.calculate_metrics(
            input1=save_dir, input2=gt_dir, cuda=cuda, fid=True, isc=True)
        fid, isc  = metrics["frechet_inception_distance"], metrics["inception_score_mean"]
        if isc > 0:
            s = np.sqrt(fid / isc)
            print(f"Geometric Mean Score ({label}): {s:.5f}")
            return s
        print(f"IS is 0 for {label}, GMS cannot be computed!")
        return None

    # Step 8: A → B
    generate_and_save(
        G_AB,
        os.path.join(data_dir, 'VAE_generation/test'),   # modification forbidden
        os.path.join(model_dir, 'generated_B_images')
    )
    s1 = score(
        os.path.join(model_dir, 'generated_B_images'),
        os.path.join(data_dir,  'VAE_generation1/test'),
        'A→B'
    )

    # Step 8: B → A
    generate_and_save(
        G_BA,
        os.path.join(data_dir, 'VAE_generation1/test'),  # modification forbidden
        os.path.join(model_dir, 'generated_A_images')
    )
    s2 = score(
        os.path.join(model_dir, 'generated_A_images'),
        os.path.join(data_dir,  'VAE_generation/test'),
        'B→A'
    )

    if s1 is not None and s2 is not None:
        s_value = np.round((s1 + s2) / 2, 5)
        df      = pd.DataFrame({'id': [1], 'label': [s_value]})
        csv_path = os.path.join(model_dir, "Userid.csv")
        df.to_csv(csv_path, index=False)
        print(f"Final score: {s_value}  —  saved to {csv_path}")


if __name__ == "__main__":
    main()