# How To Run This Repo

This guide shows exactly how to run the training/evaluation pipeline and where to find report artifacts.

## 1. Open the project folder
From PowerShell:

```powershell
cd "c:\Users\josep\Y3S2\CS424\Group Project\cs-424-group-project-friday"
```

## 2. What "run ablation configs" means
An ablation run changes one thing at a time and keeps the rest fixed.

In this repo, the main ablations are:
- `configs/ablation_lsgan.yaml` (changes GAN loss)
- `configs/ablation_reflection_pad.yaml` (changes generator padding)
- `configs/ablation_lr_decay.yaml` (changes LR schedule)
- `configs/ablation_model_unet.yaml` (changes model architecture)

You run each config as a separate experiment with a unique `--run-name`.

## 3. Run baseline

```powershell
python src/train.py --config configs/baseline.yaml --run-name baseline_run --notes "Baseline reference"
python src/evaluate.py --run-name baseline_run --visual-quality-notes "Add your visual notes" --keep-change yes
```

## 4. Run all ablations

```powershell
python src/train.py --config configs/ablation_lsgan.yaml --run-name abl_lsgan --notes "Ablation: LSGAN"
python src/evaluate.py --run-name abl_lsgan --visual-quality-notes "Add notes" --keep-change tbd --change "GAN objective: BCE -> LSGAN"

python src/train.py --config configs/ablation_reflection_pad.yaml --run-name abl_reflection --notes "Ablation: reflection padding"
python src/evaluate.py --run-name abl_reflection --visual-quality-notes "Add notes" --keep-change tbd --change "Padding: zero -> reflection"

python src/train.py --config configs/ablation_lr_decay.yaml --run-name abl_lr_decay --notes "Ablation: LR decay"
python src/evaluate.py --run-name abl_lr_decay --visual-quality-notes "Add notes" --keep-change tbd --change "Schedule: fixed LR -> decay"

python src/train.py --config configs/ablation_model_unet.yaml --run-name abl_model_unet --notes "Ablation: model toy_unet"
python src/evaluate.py --run-name abl_model_unet --visual-quality-notes "Add notes" --keep-change tbd --change "Model: toy_resnet -> toy_unet"
```

## 5.1 Switch model using config
Model choice is controlled by `model.name`:

```yaml
model:
  name: toy_resnet
  hidden_size: 64
  use_reflection_pad: false
  reflection_bonus: 0.02
```

Supported values now:
- `toy_resnet`
- `toy_unet`
- `toy_mobilenet`
- `toy_dense`
- `toy_transformer`
- `toy_shallow`

## 6. Run final config

```powershell
python src/train.py --config configs/final.yaml --run-name final_run --notes "Final selected settings"
python src/evaluate.py --run-name final_run --visual-quality-notes "Add notes" --keep-change yes --change "Final combined settings"
```

## 7. Where outputs go
Each run writes to:

```text
outputs/runs/<run_name>/
```

Expected files per run:
- `config_snapshot.yaml`
- `metrics.csv`
- `summary.json`
- `notes.txt`
- `checkpoints/last.pt`
- `checkpoints/best.pt`
- `samples/epoch_XXX.png`
- `plots/losses.png`

## 8. Compare experiments
Open:

```text
experiment_tracker.csv
```

This is the main table for ablation comparison in your report.

## 9. Optional quick smoke test
If you want a fast check before full runs:

```powershell
python src/train.py --config configs/baseline.yaml --run-name quick_test --epochs 3 --notes "Smoke test"
python src/evaluate.py --run-name quick_test --visual-quality-notes "Smoke test output" --keep-change no
```
