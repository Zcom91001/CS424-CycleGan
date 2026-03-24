Add a fake-image replay buffer for discriminator training.
This is a classic CycleGAN stabilizer and usually helps a lot with oscillation/mode collapse.
Edit: [train_cyclegan.py](c:\Users\josep\Y3S2\CS424\Group Project\CS424-CycleGan\cs-424-group-project-friday\src\train_cyclegan.py)

Use a learning-rate decay schedule (keep LR constant for first half, then linear decay to 0).
Your run uses fixed LR right now; decay usually improves final image quality.
Edit: [train_cyclegan.py](c:\Users\josep\Y3S2\CS424\Group Project\CS424-CycleGan\cs-424-group-project-friday\src\train_cyclegan.py)

Tune loss weights (lambda_cycle, lambda_identity) per your domains.
If outputs change too much structurally, raise cycle loss. If colors drift, raise identity loss.
Edit: [cyclegan_baseline.yaml](c:\Users\josep\Y3S2\CS424\Group Project\CS424-CycleGan\cs-424-group-project-friday\configs\cyclegan_baseline.yaml)

Increase model capacity/resolution carefully.
Try n_blocks=9 and larger ngf/ndf if VRAM allows; quality often improves.
Edit: [cyclegan_baseline.yaml](c:\Users\josep\Y3S2\CS424\Group Project\CS424-CycleGan\cs-424-group-project-friday\configs\cyclegan_baseline.yaml)

Improve data preprocessing/augmentation.
Use random resize-crop, horizontal flips, and consistent normalization; CycleGAN is very data-sensitive.
Edit: [cyclegan_dataset.py](c:\Users\josep\Y3S2\CS424\Group Project\CS424-CycleGan\cs-424-group-project-friday\src\cyclegan_dataset.py)

Train longer with checkpoint selection based on validation metrics, not only training g_total.
Adversarial losses can look good while visual quality worsens.

Add better evaluation signals (FID/KID/LPIPS + qualitative grid per epoch).
This makes hyperparameter tuning much faster and less guessy.

If still unstable, add discriminator regularization (spectral norm or light R1 penalty).
This is usually a second-stage improvement after 1-3 above.