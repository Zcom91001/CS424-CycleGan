# Minimal CycleGAN Experiment Repo

This repo is a lightweight, reproducible scaffold for CS424 Task 1 report logging and ablation tracking.

## What it logs per run
Each run writes to `outputs/runs/<run_name>/`:
- `config_snapshot.yaml`
- `metrics.csv` (epoch losses + official score)
- `summary.json` (includes parameter count)
- `notes.txt`
- `checkpoints/last.pt` and `checkpoints/best.pt`
- `plots/losses.png`

It also updates root-level `experiment_tracker.csv`.

## Folder structure
- `configs/`: baseline + ablation + final configs
- `src/`: training, evaluation, plotting, utilities
- `outputs/runs/`: experiment artifacts
- `experiment_tracker.csv`: ablation comparison table

## Quick start
Run from repo root (`cs-424-group-project-friday`):

```bash
python src/train.py --config configs/baseline.yaml --run-name baseline_run --notes "Baseline reference run"
python src/evaluate.py --run-name baseline_run --visual-quality-notes "Stable color transfer" --keep-change yes
```

Run ablations:

```bash
python src/train.py --config configs/ablation_lsgan.yaml --run-name abl_lsgan
python src/train.py --config configs/ablation_reflection_pad.yaml --run-name abl_reflect
python src/train.py --config configs/ablation_lr_decay.yaml --run-name abl_lr_decay
python src/train.py --config configs/ablation_model_unet.yaml --run-name abl_model_unet
```

Run final:

```bash
python src/train.py --config configs/final.yaml --run-name final_run --notes "Combined best settings"
```

## Directional tuning (A->B vs B->A)
If one direction improves while the other regresses, you can weight each direction separately in config:

```yaml
loss:
  lambda_adv_ab: 1.0
  lambda_adv_ba: 1.3
  lambda_cycle_a: 10.0
  lambda_cycle_b: 12.0
  lambda_identity_a: 5.0
  lambda_identity_b: 6.0

optimizer:
  lr: 0.0002
  lr_g_ab_mult: 1.0
  lr_g_ba_mult: 1.2
  lr_d_a_mult: 1.1
  lr_d_b_mult: 1.0
```

This keeps one shared CycleGAN run but gives stronger updates to the weaker direction.

## Notes
- This is a minimal PyTorch CycleGAN/CUT/GAN scaffold focused on reproducibility.
- The unified evaluator is now folder-based and B->A metric-oriented:
  - FID and IS are computed with `torch_fidelity`
  - `GMS` is derived as `sqrt(FID / IS)` (same convention as your VAE notebook)

## Real CycleGAN Scripts
New script-based CycleGAN workflow (PyTorch):

```bash
# Install deps (in your venv)
pip install -r requirements-cyclegan.txt

# 1) Train (quick sanity run)
python src/train_cyclegan.py --config configs/cyclegan_quick.yaml --run-name cyclegan_quick

# 2) Metric evaluation from folders (B->A outputs vs real A validation images)
python src/evaluate.py --generated-dir outputs/runs/cyclegan_quick/samples --real-dir image_image_translation/image_image_translation/VAE_generation/test --out-json outputs/runs/cyclegan_quick/eval/metrics.json --out-csv outputs/runs/cyclegan_quick/eval/metrics.csv

# 3) Plot loss graph from CSV
python src/plot_cyclegan_losses.py --run-name cyclegan_quick
```

Baseline full run config:
- `configs/cyclegan_baseline.yaml`

Outputs are written to:
- `outputs/runs/<run_name>/checkpoints/*.pth`
- `outputs/runs/<run_name>/metrics.csv`
- `outputs/runs/<run_name>/plots/losses.png`

## One-Way GAN (B->A Only)
If you only want input from `VAE_generation1` (Domain B) and outputs in `VAE_generation` style (Domain A):

```bash
# Train one-way GAN (B->A)
python -u src/train_b2a_gan.py --config configs/gan_b2a_quick.yaml --run-name gan_b2a_quick

# Metric evaluation from folders
python src/evaluate.py --generated-dir outputs/runs/gan_b2a_quick/samples --real-dir image_image_translation/image_image_translation/VAE_generation/test --out-json outputs/runs/gan_b2a_quick/eval/metrics.json --out-csv outputs/runs/gan_b2a_quick/eval/metrics.csv

# Baseline (longer run + metric evaluation)
python -u src/train_b2a_gan.py --config configs/gan_b2a_baseline.yaml --run-name gan_b2a_baseline
python src/evaluate.py --generated-dir outputs/runs/gan_b2a_baseline/samples --real-dir image_image_translation/image_image_translation/VAE_generation/test --out-json outputs/runs/gan_b2a_baseline/eval/metrics.json --out-csv outputs/runs/gan_b2a_baseline/eval/metrics.csv
```

Generated images for evaluation are produced by `src/evaluate.py` (checkpoint mode),
not during training.

Shortcut mode:
```bash
run.cmd gan_b2a
```

## One-Way CUT (B->A Only)
CUT-style one-way translation (GAN + PatchNCE), recommended for unpaired single-direction transfer:

```bash
# Quick sanity run + metric evaluation
python -u src/train_b2a_cut.py --config configs/gan_b2a_cut_quick.yaml --run-name gan_b2a_cut_quick
python src/evaluate.py --generated-dir outputs/runs/gan_b2a_cut_quick/samples --real-dir image_image_translation/image_image_translation/VAE_generation/test --out-json outputs/runs/gan_b2a_cut_quick/eval/metrics.json --out-csv outputs/runs/gan_b2a_cut_quick/eval/metrics.csv

# Baseline full run + metric evaluation
python -u src/train_b2a_cut.py --config configs/gan_b2a_cut_baseline.yaml --run-name gan_b2a_cut_baseline
python src/evaluate.py --generated-dir outputs/runs/gan_b2a_cut_baseline/samples --real-dir image_image_translation/image_image_translation/VAE_generation/test --out-json outputs/runs/gan_b2a_cut_baseline/eval/metrics.json --out-csv outputs/runs/gan_b2a_cut_baseline/eval/metrics.csv
python src/evaluate.py --checkpoint <path_to_best.pth> --input-b-dir <B_test_folder> --generated-dir <output_generated_folder> --real-dir <A_test_folder> --model-type gan

```

Shortcut mode:
```bash
run.cmd gan_b2a_cut
```

Generated images for evaluation are produced by `src/evaluate.py` (checkpoint mode),
not during training.
