# CycleGAN Replay Buffer Report

## 1) Architecture of `cyclegan_replay_buffer`

The replay-buffer training pipeline is implemented in `src/train_cyclegan_replay_buffer.py`, with model blocks in `src/cyclegan_models.py`.

### 1.1 Model Components

- **Generators**: `g_ab` and `g_ba` are both `ResnetGenerator`.
- **Discriminators**: `d_a` and `d_b` are both `PatchDiscriminator` (PatchGAN style).
- **Generator structure** (`ResnetGenerator`):
1. 7x7 reflection-padded conv stem (`in_channels -> ngf`, default `ngf=64`)
2. Two downsampling conv blocks (stride 2), channel progression `64 -> 128 -> 256`
3. `n_blocks` residual blocks at bottleneck (default `n_blocks=6`)
4. Two transposed-conv upsampling blocks back to `64` channels
5. 7x7 reflection-padded output conv + `tanh`
- **Discriminator structure** (`PatchDiscriminator`):
1. 4x4 conv stack with strides `[2,2,2,1,1]`
2. Channels progress `64 -> 128 -> 256 -> 512 -> 1`
3. LeakyReLU + InstanceNorm on intermediate layers

### 1.2 Replay Buffer (Image Pool)

The key replay-buffer mechanism is `ImagePool`:

- Configurable by `train.fake_pool_size` (default `50` in `configs/cyclegan_baseline.yaml`).
- For each fake image sent to discriminator training:
1. If pool not full: store and return current fake image.
2. If pool full: with 50% probability, return a randomly sampled historical fake and replace that slot with current fake; otherwise return current fake.

This is applied independently for both domains:

- `fake_a_pool` for training `d_a`
- `fake_b_pool` for training `d_b`

This design reduces discriminator oscillation by mixing recent and historical generated samples.

### 1.3 Training Objective and Update Order

Per batch, updates follow the standard CycleGAN order:

1. **Generator update** (`optim_g`):
- Adversarial losses: `MSE(d_b(g_ab(A)), 1)` and `MSE(d_a(g_ba(B)), 1)`
- Cycle-consistency losses: `L1(g_ba(g_ab(A)), A)` and `L1(g_ab(g_ba(B)), B)`
- Identity losses: `L1(g_ba(A), A)` and `L1(g_ab(B), B)`
- Weighted sum:
  - `lambda_cycle_a = 10`
  - `lambda_cycle_b = 10`
  - `lambda_identity_a = 5`
  - `lambda_identity_b = 5`

2. **Discriminator A update** (`optim_d_a`):
- Real loss on `A`
- Fake loss on replay-buffered `fake_a`
- Final discriminator loss: `0.5 * (real + fake)`

3. **Discriminator B update** (`optim_d_b`):
- Real loss on `B`
- Fake loss on replay-buffered `fake_b`
- Final discriminator loss: `0.5 * (real + fake)`

### 1.4 Best Checkpoint Selection

- Training writes `metrics.csv` each epoch.
- Candidate checkpoint is evaluated periodically (default every 5 epochs).
- Best checkpoint is chosen by minimizing **`gms_avg`** (from `src/evaluate.py`), with fallback to minimizing `g_total` if eval is unavailable.

---

## 2) Results in the `evaluate` Folder

This section summarizes:

- `outputs/runs/cyclegan_relay/evaluate/20epochs/eval_20260325_020249.json`
- `outputs/runs/cyclegan_relay/evaluate/50epochs/eval_20260324_185202.json`

### 2.1 Quantitative Metrics (lower FID and lower GMS are better; higher IS is better)

| Setting | B->A FID | B->A IS | B->A GMS | A->B FID | A->B IS | A->B GMS | `gms_avg` |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20 epochs (`eval_20260325_020249`) | 44.7734 | 1.9402 | 4.8038 | 107.0504 | 5.1782 | 4.5468 | **4.67528** |
| 50 epochs (`eval_20260324_185202`) | 37.8388 | 1.8055 | 4.5779 | 106.7513 | 4.4342 | 4.9066 | **4.74226** |

Both evaluations were run on 1000 generated vs 1000 real images per direction.

### 2.2 Interpretation

- **Overall metric selection**: by `gms_avg`, the **20-epoch checkpoint is slightly better** (`4.67528` vs `4.74226`).
- **Direction-level tradeoff**:
  - **B->A improved** from 20 to 50 epochs (FID and GMS both lower).
  - **A->B degraded** from 20 to 50 epochs (IS dropped and GMS increased), despite a tiny FID decrease.
- This indicates asymmetric training behavior: longer training helped one mapping but hurt the other in terms of the combined quality score used for model selection.

---

## 3) Conclusion

`cyclegan_replay_buffer` follows standard CycleGAN (dual generators/discriminators + cycle/identity losses) and adds a 50-image fake replay pool per domain to stabilize discriminator learning. In the available `cyclegan_relay/evaluate` results, the 20-epoch checkpoint gives the better overall bidirectional score (`gms_avg`), even though 50 epochs gives stronger B->A metrics.
