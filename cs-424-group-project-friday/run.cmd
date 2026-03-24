@echo off
setlocal

REM Usage:
REM   run.cmd ablations   -> run core ablations
REM   run.cmd screen      -> run baseline screening for all models
REM   run.cmd directional_ab -> run A->B focused directional tuning
REM   run.cmd directional -> run B->A focused directional tuning
REM   run.cmd gan_b2a     -> run one-way GAN baseline (B->A only)
REM   run.cmd gan_b2a_cut -> run one-way CUT baseline (B->A only)
REM   run.cmd all         -> run both
REM   run.cmd             -> default: ablations

set MODE=%~1
if "%MODE%"=="" set MODE=ablations

if /I "%MODE%"=="ablations" goto :RUN_ABLATIONS
if /I "%MODE%"=="screen" goto :RUN_SCREEN
if /I "%MODE%"=="directional_ab" goto :RUN_DIRECTIONAL_AB
if /I "%MODE%"=="directional" goto :RUN_DIRECTIONAL
if /I "%MODE%"=="gan_b2a" goto :RUN_GAN_B2A
if /I "%MODE%"=="gan_b2a_cut" goto :RUN_GAN_B2A_CUT
if /I "%MODE%"=="all" goto :RUN_ALL

echo Invalid mode: %MODE%
echo Valid modes: ablations ^| screen ^| directional_ab ^| directional ^| gan_b2a ^| gan_b2a_cut ^| all
exit /b 1

:RUN_ABLATIONS
echo [1/4] Running ablation_lsgan
python src/train.py --config configs/ablation_lsgan.yaml --run-name abl_lsgan --notes "Ablation: LSGAN"
if errorlevel 1 exit /b 1
python src/evaluate.py --run-name abl_lsgan --visual-quality-notes "Add notes" --keep-change tbd --change "GAN objective: BCE -> LSGAN"
if errorlevel 1 exit /b 1

echo [2/4] Running ablation_reflection_pad
python src/train.py --config configs/ablation_reflection_pad.yaml --run-name abl_reflection --notes "Ablation: reflection padding"
if errorlevel 1 exit /b 1
python src/evaluate.py --run-name abl_reflection --visual-quality-notes "Add notes" --keep-change tbd --change "Padding: zero -> reflection"
if errorlevel 1 exit /b 1

echo [3/4] Running ablation_lr_decay
python src/train.py --config configs/ablation_lr_decay.yaml --run-name abl_lr_decay --notes "Ablation: LR decay"
if errorlevel 1 exit /b 1
python src/evaluate.py --run-name abl_lr_decay --visual-quality-notes "Add notes" --keep-change tbd --change "Schedule: fixed LR -> decay"
if errorlevel 1 exit /b 1

echo [4/4] Running ablation_model_unet
python src/train.py --config configs/ablation_model_unet.yaml --run-name abl_model_unet --notes "Ablation: model toy_unet"
if errorlevel 1 exit /b 1
python src/evaluate.py --run-name abl_model_unet --visual-quality-notes "Add notes" --keep-change tbd --change "Model: toy_resnet -> toy_unet"
if errorlevel 1 exit /b 1

echo Ablations completed.
if /I "%MODE%"=="all" goto :RUN_SCREEN
exit /b 0

:RUN_SCREEN
echo [1/6] Screening toy_resnet
python src/train.py --config configs/baseline_toy_resnet.yaml --run-name screen_toy_resnet --notes "Model screening"
if errorlevel 1 exit /b 1
python src/evaluate.py --run-name screen_toy_resnet --visual-quality-notes "Model screening" --keep-change tbd --change "Model screening"
if errorlevel 1 exit /b 1

echo [2/6] Screening toy_unet
python src/train.py --config configs/baseline_toy_unet.yaml --run-name screen_toy_unet --notes "Model screening"
if errorlevel 1 exit /b 1
python src/evaluate.py --run-name screen_toy_unet --visual-quality-notes "Model screening" --keep-change tbd --change "Model screening"
if errorlevel 1 exit /b 1

echo [3/6] Screening toy_mobilenet
python src/train.py --config configs/baseline_toy_mobilenet.yaml --run-name screen_toy_mobilenet --notes "Model screening"
if errorlevel 1 exit /b 1
python src/evaluate.py --run-name screen_toy_mobilenet --visual-quality-notes "Model screening" --keep-change tbd --change "Model screening"
if errorlevel 1 exit /b 1

echo [4/6] Screening toy_dense
python src/train.py --config configs/baseline_toy_dense.yaml --run-name screen_toy_dense --notes "Model screening"
if errorlevel 1 exit /b 1
python src/evaluate.py --run-name screen_toy_dense --visual-quality-notes "Model screening" --keep-change tbd --change "Model screening"
if errorlevel 1 exit /b 1

echo [5/6] Screening toy_transformer
python src/train.py --config configs/baseline_toy_transformer.yaml --run-name screen_toy_transformer --notes "Model screening"
if errorlevel 1 exit /b 1
python src/evaluate.py --run-name screen_toy_transformer --visual-quality-notes "Model screening" --keep-change tbd --change "Model screening"
if errorlevel 1 exit /b 1

echo [6/6] Screening toy_shallow
python src/train.py --config configs/baseline_toy_shallow.yaml --run-name screen_toy_shallow --notes "Model screening"
if errorlevel 1 exit /b 1
python src/evaluate.py --run-name screen_toy_shallow --visual-quality-notes "Model screening" --keep-change tbd --change "Model screening"
if errorlevel 1 exit /b 1

echo Model screening completed.
exit /b 0

:RUN_DIRECTIONAL
echo Running directional B->A focused tuning
python src/train.py --config configs/directional_ba_focus.yaml --run-name directional_ba_focus --notes "Directional tuning: prioritize B->A"
if errorlevel 1 exit /b 1
python src/evaluate.py --run-name directional_ba_focus --visual-quality-notes "Directional tuning run" --keep-change tbd --change "Directional loss/LR weighting for B->A"
if errorlevel 1 exit /b 1

echo Directional tuning run completed.
exit /b 0

:RUN_DIRECTIONAL_AB
echo Running directional A->B focused tuning
python src/train.py --config configs/directional_ab_focus.yaml --run-name directional_ab_focus --notes "Directional tuning: prioritize A->B"
if errorlevel 1 exit /b 1
python src/evaluate.py --run-name directional_ab_focus --visual-quality-notes "Directional tuning run" --keep-change tbd --change "Directional loss/LR weighting for A->B"
if errorlevel 1 exit /b 1

echo Directional A->B tuning run completed.
exit /b 0

:RUN_GAN_B2A
echo Running one-way GAN baseline (B->A only)
python src/train_b2a_gan.py --config configs/gan_b2a_baseline.yaml --run-name gan_b2a_baseline
if errorlevel 1 exit /b 1
python src/evaluate_b2a_gan.py --config configs/gan_b2a_baseline.yaml --run-name gan_b2a_baseline --epochs 0,25,50,75,100
if errorlevel 1 exit /b 1

echo One-way GAN baseline completed.
exit /b 0

:RUN_GAN_B2A_CUT
echo Running one-way CUT baseline (B->A only)
python src/train_b2a_cut.py --config configs/gan_b2a_cut_baseline.yaml --run-name gan_b2a_cut_baseline
if errorlevel 1 exit /b 1
python src/evaluate_b2a_cut.py --config configs/gan_b2a_cut_baseline.yaml --run-name gan_b2a_cut_baseline --epochs 0,25,50,75,100
if errorlevel 1 exit /b 1

echo One-way CUT baseline completed.
exit /b 0

:RUN_ALL
goto :RUN_ABLATIONS


python src/evaluate.py --checkpoint outputs/runs/gan_b2a_baseline/checkpoints/best.pth --input-b-dir image_image_translation/image_image_translation/VAE_generation1/test --generated-dir outputs\runs\gan_b2a_baseline\samples --real-dir image_image_translation\image_image_translation\VAE_generation\test --model-type gan --out-json outputs\runs\gan_b2a_baseline\eval