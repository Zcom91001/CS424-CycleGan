@echo off
setlocal

:BASE
python src/train_b2a_gan.py --config configs/gan_b2a_baseline.yaml --run-name gan_b2a_baseline

:BASE_CUT
python src/train_b2a_gan.py --config configs/gan_b2a_cut_baseline.yaml --run-name gan_b2a_cut_baseline

python src/evaluate.py --checkpoint outputs\runs\gan_b2a_cut_baseline\checkpoints\best.pth --input-b-dir image_image_translation/image_image_translation/VAE_generation1/test --generated-dir outputs\runs\gan_b2a_cut_baseline\samples --real-dir image_image_translation\image_image_translation\VAE_generation\test --model-type gan --out-json outputs\runs\gan_b2a_cut_baseline\evaluate\eval.json