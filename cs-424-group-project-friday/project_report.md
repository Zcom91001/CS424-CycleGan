# Group Project Report
## CS424: Generative AI for Vision

This report summarizes the CycleGAN image-to-image translation pipeline in this repository. The work targets unpaired domain transfer between:

- Domain A: `VAE_generation`
- Domain B: `VAE_generation1`

Quality is evaluated with the Geometric Mean Score:

- `GMS = sqrt(FID / IS)` (lower is better)
- `GMS_avg = mean(GMS_b2a, GMS_a2b)` (the headline metric)

The archived runs in [outputs/runs/](outputs/runs/) cover five comparable CycleGAN configurations:

| Run | Discriminator | Replay buffer | λ_cycle | Epochs trained |
|---|---|---|---:|---:|
| `my_cyclegan_run` | PatchGAN | no | 10 | up to 100 |
| `cyclegan_baseline_20` | PatchGAN | no | 10 | 20 |
| `cyclegan_baseline_50` | PatchGAN | no | 10 | 50 |
| `cyclegan_relay` | PatchGAN | yes (`pool=50`) | 10 | up to 150 |
| `cyclegan_multiscale_disc_150` | Multi-scale (3 scales, SN) | no | 15 | 150 |

---

## 1. Model Architecture

### 1.1 Generators

Both `g_ab` and `g_ba` are ResNet-style image-translation networks defined in [src/cyclegan_models.py:22-61](src/cyclegan_models.py#L22-L61):

- `ReflectionPad2d(3)` → `7x7 Conv` stem (stride 1)
- two `3x3` strided downsampling convolutions (channel doubling)
- `n_blocks = 6` residual blocks (`InstanceNorm`, `ReLU`)
- two `ConvTranspose2d` upsampling layers (channel halving)
- `ReflectionPad2d(3)` → `7x7 Conv` → `Tanh`

Generator capacity is identical across all archived runs: `ngf = 64`, `n_blocks = 6`. The smaller `cyclegan_quick` sanity preset uses `ngf = 32`, `n_blocks = 3` for fast iteration only.

### 1.2 Discriminators

Two discriminator variants are implemented in [src/cyclegan_models.py:68-192](src/cyclegan_models.py#L68-L192).

**PatchGAN baseline** (`NLayerPatchDiscriminator`)
- 4×4 strided convolutions, three layers, classifies real/fake at patch level
- LeakyReLU(0.2), InstanceNorm
- Used by `my_cyclegan_run`, `cyclegan_baseline_*`, `cyclegan_relay`

**Multi-scale discriminator** (`MultiScaleDiscriminator`)
- `num_scales = 3` PatchGANs running on progressively `AvgPool`-downsampled inputs
- `n_layers = 3`
- `use_spectral_norm = True`
- `use_instance_norm = True`
- Used by `cyclegan_multiscale_disc_150`

### 1.3 Parameter Counts

| Variant | Trainable parameters (2G + 2D) |
|---|---:|
| Baseline PatchGAN CycleGAN | ≈ 21.2 M |
| Multi-scale CycleGAN | ≈ 32.3 M |

The multi-scale variant adds roughly 50 % more parameters, all on the discriminator side.

---

## 2. Training Loss

The training loop is in [src/train_cyclegan.py:222-300](src/train_cyclegan.py#L222-L300).

### 2.1 Generator Objective

```
g_total = g_adv_ab + g_adv_ba
        + λ_cycle_a * cycle_a + λ_cycle_b * cycle_b
        + λ_identity_a * id_a + λ_identity_b * id_b
```

- Adversarial: `MSELoss` (LSGAN-style)
- Cycle consistency: `L1Loss` on `g_ba(g_ab(a))` and `g_ab(g_ba(b))`
- Identity: `L1Loss` on `g_ba(a)` and `g_ab(b)`

For the multi-scale discriminator, `_mse_gan_loss` averages MSE over all scales ([train_cyclegan.py:50-55](src/train_cyclegan.py#L50-L55)).

### 2.2 Discriminator Objective

```
d_loss = 0.5 * (MSE(D(real), 1) + MSE(D(fake.detach()), 0))
```

Both `d_a` and `d_b` are trained independently each step.

### 2.3 Loss Weights by Run

| Run | λ_cycle_a | λ_cycle_b | λ_id_a | λ_id_b |
|---|---:|---:|---:|---:|
| `my_cyclegan_run` | 10 | 10 | 5 | 5 |
| `cyclegan_baseline_20` | 10 | 10 | 5 | 5 |
| `cyclegan_baseline_50` | 10 | 10 | 5 | 5 |
| `cyclegan_relay` | 10 | 10 | 5 | 5 |
| `cyclegan_multiscale_disc_150` | **15** | **15** | 5 | 5 |

The multi-scale run is the only configuration that strengthens cycle consistency, biasing the generators toward stronger structural preservation to balance the more aggressive discriminator.

---

## 3. Data Augmentation

The dataset pipeline in [src/cyclegan_dataset.py](src/cyclegan_dataset.py) is intentionally minimal.

For each `(a, b)` sample (`UnpairedImageDataset.__getitem__`):

1. Load PIL image, `convert("RGB")`
2. `BICUBIC` resize to `image_size × image_size` (128 in all archived runs)
3. Cast to `float32`, scale to `[0, 1]`
4. Normalize to `[-1, 1]` to match the `Tanh` generator output

What is **not** used:
- random crop / random resized crop
- horizontal or vertical flips
- color jitter, grayscale, blur
- rotation or affine transforms

The only stochasticity is that domain B is sampled randomly per index (`b_path = random.choice(self.b_files)`), giving unpaired pairings each epoch. All performance differences between runs therefore come from architecture, loss weights, training duration, and discriminator design — not from augmentation engineering.

---

## 4. Optimisation

### 4.1 Optimizer

All runs use Adam ([train_cyclegan.py:173-179](src/train_cyclegan.py#L173-L179)):

- generators: single Adam over `g_ab.parameters() ∪ g_ba.parameters()`
- discriminators: separate Adam optimizers for `d_a` and `d_b`
- `lr_g = lr_d = 2e-4`
- `beta1 = 0.5`, `beta2 = 0.999`
- no learning-rate schedule — fixed LR throughout training

### 4.2 Training Setup

| Setting | Value |
|---|---|
| `image_size` | 128 |
| `batch_size` | 1 |
| `num_workers` | 4 |
| Train images per domain | 4000 |
| Test images per domain | 1000 |
| Steps per epoch | 4000 |
| Seed | 42 |

### 4.3 Checkpoint Selection

The script periodically (`best_eval_every_epochs = 5`) runs `evaluate.py` on the candidate checkpoint, parses `gms_avg`, and saves `best.pth` whenever the score drops ([train_cyclegan.py:339-360](src/train_cyclegan.py#L339-L360)). If a per-epoch eval fails it falls back to minimizing `g_total`. `final.pth` is saved unconditionally at the end of training. This is important: the best epoch is rarely the last epoch.

---

## 5. Key Improvement Summary

The repository shows four meaningful changes layered on top of a textbook CycleGAN baseline.

### 5.1 Stable training infrastructure

`cyclegan_baseline_20` is the first clean reference run. Compared to the earlier `my_cyclegan_run` it uses:
- the unified `train_cyclegan.py` script
- `best_eval_every_epochs = 5` per-epoch evaluation gating `best.pth`
- consistent dataset paths from the repo

### 5.2 Replay buffer (`cyclegan_relay`)

Adds a `fake_pool_size = 50` image pool so that discriminators see a mix of historical and current fakes. Same generator/discriminator/loss weights as the baseline.

### 5.3 Longer training (`cyclegan_baseline_50`, `cyclegan_relay`, `cyclegan_multiscale_disc_150`)

Training budget increases from 20 → 50 → 150 epochs. With per-epoch eval, this also expands the search space for the best checkpoint.

### 5.4 Multi-scale discriminator + stronger cycle weight (`cyclegan_multiscale_disc_150`)

The most architecturally significant change:
- 3-scale PatchGAN with spectral normalization
- λ_cycle raised from 10 → 15 to balance the stronger discriminator

This is the run that achieves the best archived score in the workspace (`GMS_avg = 3.95350` at epoch 85, recorded in [outputs/runs/cyclegan_multiscale_disc_150/evaluate/eval_20260402_005656.json](outputs/runs/cyclegan_multiscale_disc_150/evaluate/eval_20260402_005656.json)).

---

## 6. Findings and Insights — Comparison at the 20-epoch mark

To make the runs comparable, every metric below is taken from the **epoch-20 evaluation** of each run, read from `outputs/runs/<run>/evaluate/by_epoch/epoch_020/eval_*.json` (or the equivalent `20epochs/` folder for the older runs). This isolates the effect of each configuration before training-budget differences kick in.

### 6.1 Headline table at epoch 20

| Run | Disc. | Replay | λ_cycle | B→A FID | B→A IS | B→A GMS | A→B FID | A→B IS | A→B GMS | **GMS_avg** |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `my_cyclegan_run` | PatchGAN | no | 10 | 55.094 | 2.123 | 5.094 | 127.517 | 5.143 | 4.980 | **5.0368** |
| `cyclegan_baseline_20` | PatchGAN | no | 10 | 40.942 | 1.713 | 4.888 | 111.116 | 4.814 | 4.804 | **4.8462** |
| `cyclegan_multiscale_disc_150` | Multi-scale (SN) | no | 15 | 51.867 | 2.167 | 4.892 | 109.173 | 5.123 | 4.616 | **4.7543** |
| `cyclegan_baseline_50` | PatchGAN | no | 10 | 39.965 | 1.770 | 4.752 | 111.347 | 5.249 | 4.606 | **4.6791** |
| `cyclegan_relay` | PatchGAN | yes (50) | 10 | 44.773 | 1.940 | 4.804 | 107.050 | 5.178 | 4.547 | **4.6753** |

Sorted by `GMS_avg` (best at the bottom). Differences across runs at 20 epochs are small — about 0.36 of GMS between worst and best — which is the appropriate scale for what 20 epochs of training can establish.

### 6.2 Finding 1 — Infrastructure alone is the biggest jump

`my_cyclegan_run` → `cyclegan_baseline_20` keeps the same model, loss weights and training duration, yet `GMS_avg` improves from `5.0368` to `4.8462` (Δ ≈ −0.19). The B→A FID alone falls from `55.09` to `40.94`. The only thing that changed is the unified training script and per-epoch best-checkpoint selection — the early ad-hoc run was simply not picking the best epoch.

**Insight:** before any architectural ablation, the largest single gain came from disciplined checkpointing.

### 6.3 Finding 2 — Replay buffer is the best 20-epoch configuration

At exactly 20 epochs, `cyclegan_relay` has the lowest `GMS_avg` (`4.6753`), narrowly ahead of the longer-trained `cyclegan_baseline_50` evaluated at epoch 20 (`4.6791`). It also achieves the best A→B FID (`107.05`) of all five runs at this checkpoint.

**Insight:** the historical-fake pool stabilizes early-epoch discriminator learning, which is exactly when CycleGAN benefits most from the regularization. Note: the baseline-50 epoch-20 row is statistically very close — replay's edge here is marginal, not decisive.

### 6.4 Finding 3 — The multi-scale discriminator is *underperforming* at 20 epochs

The multi-scale run lands at `GMS_avg = 4.7543` after 20 epochs — better than the early baselines but worse than both the replay buffer and the 50-epoch baseline at the same epoch. Looking at the per-epoch training metrics in [outputs/runs/cyclegan_multiscale_disc_150/metrics.csv](outputs/runs/cyclegan_multiscale_disc_150/metrics.csv):

- `g_total` at epoch 20 = `5.83` (vs `4.27` for baseline_20)
- `d_a_loss` = `0.044`, `d_b_loss` = `0.064` (vs `0.133`/`0.160` for baseline_20)

The multi-scale discriminator with spectral norm is roughly 3× harder for the generators to fool at this point in training — D losses are very low while G losses stay high. The model has not yet found its operating point.

**Insight:** the multi-scale variant is intentionally a long-horizon configuration. Its raised cycle weight (15 vs 10) and stronger discriminator only pay off given a much longer training budget. The archived `best.pth` at epoch 85 reaches `GMS_avg = 3.95350`, well below anything any run achieves at epoch 20.

### 6.5 Finding 4 — Direction asymmetry is consistent across runs

Across **every** configuration at 20 epochs, A→B FID is roughly 2–3× larger than B→A FID:

| Run | B→A FID | A→B FID | Ratio |
|---|---:|---:|---:|
| `my_cyclegan_run` | 55.09 | 127.52 | 2.31 |
| `cyclegan_baseline_20` | 40.94 | 111.12 | 2.71 |
| `cyclegan_baseline_50` | 39.96 | 111.35 | 2.79 |
| `cyclegan_multiscale_disc_150` | 51.87 | 109.17 | 2.10 |
| `cyclegan_relay` | 44.77 | 107.05 | 2.39 |

**Insight:** A→B is the harder direction for this dataset — domain B has higher visual complexity (its IS is consistently `~5.1`, vs `~1.7–2.2` for domain A), so generators struggle more to match its distribution. Importantly, GMS partially hides this because A→B's higher IS pulls its `sqrt(FID/IS)` down. The multi-scale run is the only one that begins to flatten this ratio (2.10), suggesting its discriminator is more useful in the harder direction even at 20 epochs where its absolute scores still lag.

### 6.6 Finding 5 — At 20 epochs the runs have not yet differentiated meaningfully

The five runs span only `4.6753 → 5.0368` in `GMS_avg`, a range of `0.36`. By contrast, the gap between the best 20-epoch run and the best overall checkpoint (multi-scale at epoch 85, `GMS_avg = 3.95350`) is `0.72` — twice as large as the spread between 20-epoch configurations.

**Insight:** at 20 epochs, the choice of run matters less than the choice of training duration. Architectural changes only translate to large metric gains once the model has had enough training steps to actually exploit them. For this dataset and this codebase, 20 epochs is roughly an early-progress checkpoint, not a finished model.

### 6.7 Final summary at the 20-epoch mark

1. **Best 20-epoch run:** `cyclegan_relay` (`GMS_avg = 4.6753`), narrowly ahead of `cyclegan_baseline_50` evaluated at the same epoch.
2. **Worst 20-epoch run:** the original `my_cyclegan_run` (`GMS_avg = 5.0368`), held back by ad-hoc checkpoint selection rather than by model design.
3. **Most under-rated at 20 epochs:** `cyclegan_multiscale_disc_150` — its 20-epoch `GMS_avg` is only mid-pack (`4.7543`), but it is the only run that eventually wins overall, reaching `GMS_avg = 3.95350` at epoch 85.
4. **Easy-direction vs hard-direction asymmetry** is consistent across every run, and persists into longer training; B→A is consistently the easier direction.
5. **The largest 20-epoch improvements** come from training infrastructure (best-epoch selection) and replay-buffer stabilization, not from architecture alone — at 20 epochs the multi-scale architecture has not yet had enough training time to differentiate from the baseline.

---

## 7. Findings and Insights — Comparison at 100 and 150 epochs

Only three of the five archived runs trained beyond 50 epochs, so the longer-horizon comparison is necessarily a subset:

| Run | Available checkpoints with eval JSON |
|---|---|
| `my_cyclegan_run` | up to 100 (`evaluate/100epochs/`) |
| `cyclegan_relay` | 110 and 150 (`evaluate/110epochs/`, `evaluate/150epochs/`) |
| `cyclegan_multiscale_disc_150` | every 5 epochs from 5 to 150 |

`cyclegan_baseline_20` and `cyclegan_baseline_50` cannot participate — despite the `_50` name, the second baseline only has eval JSONs for epochs 5–20 in [outputs/runs/cyclegan_baseline_50/evaluate/by_epoch/](outputs/runs/cyclegan_baseline_50/evaluate/by_epoch/). **No run in the workspace reached 250 epochs**, so a 250-epoch comparison is not possible.

There is also one provenance caveat for the relay numbers: the 150-epoch eval in [outputs/runs/cyclegan_relay/evaluate/150epochs/eval_20260327_091903.json](outputs/runs/cyclegan_relay/evaluate/150epochs/eval_20260327_091903.json) was actually run from a different sub-config — `lambda_cycle = 20` (not 10) and from `final.pth` rather than the per-epoch candidate. The 110-epoch eval is the cleaner comparison point for the original `lambda_cycle = 10` replay setup.

### 7.1 Headline table at ~100 epochs

| Run | Epoch | Disc. | λ_cycle | B→A FID | B→A IS | B→A GMS | A→B FID | A→B IS | A→B GMS | **GMS_avg** |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `my_cyclegan_run` | 100 | PatchGAN | 10 | 34.755 | 1.923 | 4.251 | 93.765 | 4.948 | 4.353 | **4.3021** |
| `cyclegan_relay` | 110 | PatchGAN | 10 | 37.169 | 1.878 | 4.449 | 86.256 | 5.314 | 4.029 | **4.2389** |
| `cyclegan_multiscale_disc_150` | 100 | Multi-scale (SN) | 15 | **90.868** | **6.125** | 3.852 | 98.168 | 5.346 | 4.285 | **4.0686** |

### 7.2 Headline table at ~150 epochs

| Run | Epoch | Disc. | λ_cycle | B→A FID | B→A IS | B→A GMS | A→B FID | A→B IS | A→B GMS | **GMS_avg** |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cyclegan_relay` (final, λ=20) | 150 | PatchGAN | 20 | 39.132 | 1.946 | 4.484 | 94.813 | 5.323 | 4.221 | **4.3523** |
| `cyclegan_multiscale_disc_150` | 150 | Multi-scale (SN) | 15 | 86.386 | 5.779 | 3.866 | 99.580 | 5.333 | 4.321 | **4.0937** |
| `cyclegan_multiscale_disc_150` (best) | **85** | Multi-scale (SN) | 15 | 107.744 | 7.930 | 3.686 | 97.008 | 5.445 | 4.221 | **3.9535** |

### 7.3 Finding 6 — The multi-scale run wins by GMS but loses by FID

This is the most important observation in the long-horizon comparison, and it inverts the intuitive reading of the table.

At epoch 100, `multi-scale` has the lowest `GMS_avg` (`4.0686`), but its B→A FID is **`90.87`** — over 2.4× worse than `my_cyclegan_run`'s `34.76` at the same epoch. The reason `GMS_avg` still favours multi-scale is that its B→A `IS` is **`6.13`** versus `my_cyclegan_run`'s `1.92`, and `GMS = sqrt(FID / IS)` divides one by the other. A 3× higher IS is enough to mask a 2.6× higher FID.

The same pattern holds at the multi-scale "best" checkpoint (epoch 85): `B→A FID = 107.74` with `IS = 7.93`. This is the headline result in the report (`GMS_avg = 3.9535`), but its FID alone is the *worst* B→A FID in the entire workspace.

**Insight:** the multi-scale + spectral-norm + λ_cycle=15 configuration has converged onto a different solution from the PatchGAN runs. It produces images that the Inception classifier finds **more diverse / more confident** (much higher IS) but **distributionally further from the test set** (much higher FID). Whether this is "better" depends entirely on which metric the project commits to. If FID is the gold standard, the PatchGAN replay run at epoch 110 (`B→A FID = 37.17`) is the best model in the workspace. If `GMS_avg` is the gold standard, the multi-scale run at epoch 85 wins.

This is also a warning about `GMS = sqrt(FID/IS)` as an evaluation metric: it can be moved by either factor independently, and a model can climb the leaderboard purely by inflating IS.

### 7.4 Finding 7 — `my_cyclegan_run` and `cyclegan_relay` track each other closely on FID

Stripping out the multi-scale outlier, the two PatchGAN long-horizon runs show that the replay buffer's marginal-at-20-epochs advantage is also marginal at 100+ epochs:

| Direction | `my_cyclegan_run` @ 100 | `cyclegan_relay` @ 110 | Δ (relay − no-replay) |
|---|---:|---:|---:|
| B→A FID | 34.755 | 37.169 | +2.41 |
| A→B FID | 93.765 | 86.256 | −7.51 |
| GMS_avg | 4.3021 | 4.2389 | −0.063 |

Replay loses slightly on B→A FID but wins by ~7.5 FID points on the harder A→B direction, and the net `GMS_avg` is essentially a tie. This is consistent with the canonical CycleGAN paper finding that the image pool helps mostly with discriminator stability rather than with absolute quality on simple translations.

**Insight:** for this dataset, the replay buffer is a wash on the easier direction and a small but real win on the harder one. It is not the dominant factor in long-horizon performance.

### 7.5 Finding 8 — Multi-scale peaks early and then drifts

Tracking the multi-scale run across its own checkpoints:

| Epoch | B→A FID | B→A IS | B→A GMS | A→B FID | A→B IS | A→B GMS | **GMS_avg** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 51.87 | 2.17 | 4.892 | 109.17 | 5.12 | 4.616 | 4.7543 |
| 50 | 67.60 | 3.45 | 4.424 | 109.00 | 5.49 | 4.457 | 4.4408 |
| **85** | **107.74** | **7.93** | **3.686** | 97.01 | 5.44 | 4.221 | **3.9535** |
| 100 | 90.87 | 6.12 | 3.852 | 98.17 | 5.35 | 4.285 | 4.0686 |
| 150 | 86.39 | 5.78 | 3.866 | 99.58 | 5.33 | 4.321 | 4.0937 |

The "best" checkpoint at epoch 85 is exactly the point where B→A IS spikes to its global maximum (`7.93`). After that, IS falls back toward `~5.8` and `GMS_avg` rises again. This is not a quality improvement followed by overfitting — the FID never gets better. It is `IS` rising and falling.

**Insight:** what `best_eval_every_epochs = 5` is selecting in the multi-scale run is not "the most realistic checkpoint", it is "the checkpoint at which the Inception classifier is most confident on B→A samples". Anyone using this best.pth in production should sanity-check the actual generated images — they may look more saturated or more "category-like" than at epoch 50, but not necessarily more realistic.

### 7.6 Final summary at 100 / 150 epochs

1. **Best `GMS_avg` overall:** `cyclegan_multiscale_disc_150` at epoch 85 (`3.9535`), and it stays the leader through epochs 100 and 150.
2. **Best `FID` overall (B→A):** `my_cyclegan_run` at epoch 100 (`34.755`) — the smallest, simplest archived setup. The multi-scale run never gets within 50 FID points of this on the B→A direction.
3. **Replay buffer is essentially tied with no-replay** in long-horizon training (`4.2389` vs `4.3021` `GMS_avg`); it helps slightly on the harder A→B direction.
4. **The multi-scale run does not improve monotonically.** Its score moves `4.7543 → 4.4408 → 3.9535 → 4.0686 → 4.0937` from epoch 20 to 150; the apparent peak at epoch 85 is driven by an IS spike, not by an FID improvement.
5. **No 250-epoch run exists in the workspace.** The longest available training horizon is 150 epochs.
6. **The metric matters more than the model at this scale.** Switching from `GMS_avg` to plain B→A FID would completely reorder the leaderboard and crown `my_cyclegan_run` as the best model. Pinning down which metric the project commits to is more impactful than any of the architectural changes tested here.
