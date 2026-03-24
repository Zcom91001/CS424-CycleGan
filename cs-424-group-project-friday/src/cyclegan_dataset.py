from pathlib import Path
import random

import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _list_images(root):
    root = Path(root)
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
    return sorted(files)


def _pil_to_tensor_rgb(img, image_size):
    if image_size is not None:
        img = img.resize((image_size, image_size), Image.BICUBIC)
    img = img.convert("RGB")
    h = img.size[1]
    w = img.size[0]
    raw = bytearray(img.tobytes())
    data = torch.tensor(raw, dtype=torch.uint8)
    data = data.view(h, w, 3).permute(2, 0, 1).float() / 255.0
    # Normalize to [-1, 1] to match tanh generators.
    return data * 2.0 - 1.0


class UnpairedImageDataset(Dataset):
    def __init__(self, domain_a_dir, domain_b_dir, image_size=128):
        self.domain_a_dir = Path(domain_a_dir).resolve()
        self.domain_b_dir = Path(domain_b_dir).resolve()
        self.a_files = _list_images(self.domain_a_dir)
        self.b_files = _list_images(self.domain_b_dir)
        if len(self.a_files) == 0 or len(self.b_files) == 0:
            raise ValueError(
                "Both domain directories must contain image files.\n"
                f"Resolved domain_a_dir: {self.domain_a_dir} (found {len(self.a_files)} files)\n"
                f"Resolved domain_b_dir: {self.domain_b_dir} (found {len(self.b_files)} files)\n"
                f"Supported extensions: {sorted(IMAGE_EXTS)}"
            )
        self.image_size = int(image_size) if image_size else None
        self.length = max(len(self.a_files), len(self.b_files))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        a_path = self.a_files[idx % len(self.a_files)]
        b_path = random.choice(self.b_files)

        with Image.open(a_path) as a_img:
            a_tensor = _pil_to_tensor_rgb(a_img, self.image_size)
        with Image.open(b_path) as b_img:
            b_tensor = _pil_to_tensor_rgb(b_img, self.image_size)

        return {
            "A": a_tensor,
            "B": b_tensor,
            "A_path": str(a_path),
            "B_path": str(b_path),
        }


def load_fixed_samples(domain_dir, n_images=16, image_size=128):
    files = _list_images(domain_dir)[: int(n_images)]
    out = []
    for p in files:
        with Image.open(p) as img:
            out.append(_pil_to_tensor_rgb(img, image_size))
    if len(out) == 0:
        raise ValueError(f"No images found in {domain_dir}")
    return torch.stack(out, dim=0)
