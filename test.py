# Run this and share the output
from PIL import Image
import numpy as np
import os

data_dir = "./"

def domain_stats(file_list, name, n=100):
    means, stds = [], []
    for f in file_list[:n]:
        img = np.array(Image.open(f).convert("RGB")) / 255.0
        means.append(img.mean(axis=(0,1)))
        stds.append(img.std(axis=(0,1)))
    means = np.array(means).mean(axis=0)
    stds  = np.array(stds).mean(axis=0)
    print(f"{name} — mean RGB: {means.round(3)}  std RGB: {stds.round(3)}")

all_A = sorted([os.path.join(data_dir, 'VAE_generation/train', n) 
                for n in os.listdir(os.path.join(data_dir, 'VAE_generation/train'))])
all_B = sorted([os.path.join(data_dir, 'VAE_generation1/train', n) 
                for n in os.listdir(os.path.join(data_dir, 'VAE_generation1/train'))])

domain_stats(all_A, "Cats (A)")
domain_stats(all_B, "Dogs (B)")