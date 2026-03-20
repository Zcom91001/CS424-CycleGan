# Minimal CycleGAN Experiment Repo

This repo is a lightweight, reproducible scaffold for CS424 Task 1 report logging and ablation tracking.

## What it logs per run
Each run writes to `outputs/runs/<run_name>/`:
- `config_snapshot.yaml`
- `metrics.csv` (epoch losses + official score)
- `summary.json` (includes parameter count)
- `notes.txt`
- `checkpoints/last.pt` and `checkpoints/best.pt`
- `samples/epoch_XXX.png`
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

## Notes
- This is a minimal, dependency-free (Python stdlib only) CycleGAN-style scaffold for reporting workflow.
- It is designed for experiment tracking and reproducibility, not for SOTA image translation quality.
- Model selection is config-driven via `model.name` in YAML.
- Supported models: `toy_resnet`, `toy_unet`, `toy_mobilenet`, `toy_dense`, `toy_transformer`, `toy_shallow`.
- `src/evaluate.py` follows CS424 evaluation structure (A->B, B->A, GMS average, `Userid.csv`) using a dependency-free proxy metric.
