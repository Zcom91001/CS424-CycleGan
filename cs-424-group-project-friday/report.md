# CycleGAN (No Replay Buffer) Report

## Overview

This report summarizes the **no-replay CycleGAN** training pipeline (based on the script you provided) and the evaluation results from:

- `outputs/runs/my_cyclegan_run/evaluate/20epochs/eval_20260324_215631.json`
- `outputs/runs/my_cyclegan_run/evaluate/50epochs/eval_20260324_065229.json`
- `outputs/runs/my_cyclegan_run/evaluate/100epochs/eval_20260324_111455.json`
- `outputs/runs/my_cyclegan_run/evaluate/100epochs/eval_20260324_111456.json`
- `outputs/runs/my_cyclegan_run/evaluate/20epochs/joseph_cycle_gan_no_replay.out`

## Training Architecture (No Replay)

### Model setup

- Two generators: `g_ab` (A->B) and `g_ba` (B->A)
- Two discriminators: `d_a` (domain A) and `d_b` (domain B)
- Built via `build_cyclegan_models(ngf=64, ndf=64, n_blocks=6)`

### Data and loader

From the provided YAML:

- Train A: `image_image_translation/image_image_translation/VAE_generation/train`
- Train B: `image_image_translation/image_image_translation/VAE_generation1/train`
- Test A: `image_image_translation/image_image_translation/VAE_generation/test`
- Test B: `image_image_translation/image_image_translation/VAE_generation1/test`
- `image_size=128`, `batch_size=1`, `num_workers=4`, seed `42`

### Losses and optimization

- Adversarial loss: `MSELoss`
- Cycle-consistency loss: `L1Loss`
- Identity loss: `L1Loss`
- Generator objective:
  - `g_adv_ab + g_adv_ba + 10*cycle_a + 10*cycle_b + 5*id_a + 5*id_b`
- Discriminator objective (for each domain):
  - `0.5 * (real_mse + fake_mse)`
- Optimizers: Adam
  - `lr_g=0.0002`, `lr_d=0.0002`, `beta1=0.5`, `beta2=0.999`

### Key property: no replay buffer

Unlike replay-buffer training, this script feeds discriminators with **current-batch fake samples only**:

- `pred_fake_a_detached = d_a(fake_a.detach())`
- `pred_fake_b_detached = d_b(fake_b.detach())`

No image pool / historical fake sampling is used.

### Checkpointing and model selection

- Per-epoch metrics appended to `metrics.csv`
- `best.pth` selected by **minimum training `g_total`**
- `final.pth` saved at training end
- Auto evaluation is run via `run_auto_evaluation(...)`

## Evaluation Results

### Quantitative summary

Lower FID and lower GMS are better; higher IS is better.

| Eval file | Epochs | B->A FID | B->A IS | B->A GMS | A->B FID | A->B IS | A->B GMS | GMS_avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `eval_20260324_215631.json` | 20 | 55.0940 | 2.1231 | 5.0941 | 127.5168 | 5.1427 | 4.9795 | 5.03682 |
| `eval_20260324_065229.json` | 50 | 45.9058 | 2.0699 | 4.7093 | 95.0013 | 4.7398 | 4.4770 | 4.59315 |
| `eval_20260324_111455.json` | 100 | 34.7551 | 1.9233 | 4.2510 | 93.7649 | 4.9481 | 4.3531 | 4.30205 |
| `eval_20260324_111456.json` | 100 | 34.7551 | 1.9233 | 4.2510 | 93.7649 | 4.9481 | 4.3531 | 4.30205 |

Notes:

- Both 100-epoch JSON files contain identical metrics.
- Each direction uses `generated_images=1000` and `real_images=1000`.

## Trend Analysis

From 20 -> 50 -> 100 epochs:

- **Overall quality improves steadily** by `GMS_avg`:
  - `5.03682 -> 4.59315 -> 4.30205`
- **B->A translation improves strongly**:
  - FID drops from `55.09` to `34.76`
- **A->B translation also improves**:
  - FID drops from `127.52` to `93.76`
- IS fluctuates but combined metric (`GMS`) improves in both directions by 100 epochs.

## Run-log Highlights (`joseph_cycle_gan_no_replay.out`)

From the provided log file:

- Training ran on CUDA with `4000 steps/epoch`.
- Job completed successfully (`State: COMPLETED`, exit code `0`).
- Wall-clock time reported: `01:12:10`.
- Auto-evaluation outputs match the 20-epoch JSON metrics listed above.

## Conclusion

The no-replay CycleGAN setup is functioning correctly and shows consistent improvement as training epochs increase. Among the provided evaluation artifacts, the best result is at **100 epochs** with **`GMS_avg = 4.30205`**.


### Next Step

Good starting grid:

cycle=15, identity=5 (more structure preservation) (worse then original)
cycle=20, identity=5 (strong structure constraint)
cycle=10, identity=7.5 (less color/style drift)
cycle=10, identity=10 (strong color preservation)