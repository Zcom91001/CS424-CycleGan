import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from cyclegan_dataset import UnpairedImageDataset
from cyclegan_io import (
    append_metrics_row,
    ensure_run_dirs,
    run_auto_evaluation,
    save_checkpoint,
    save_loss_plot,
)
from cyclegan_models import build_cyclegan_models
from evaluate import evaluate_bidirectional_generators
from yaml_utils import load_yaml


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_device(batch, device):
    return batch.to(device, non_blocking=True)


def _resolve_device(requested):
    req = str(requested).strip().lower()
    if req.startswith("cuda") and not torch.cuda.is_available():
        print("Requested CUDA but CUDA is unavailable in this PyTorch build. Falling back to CPU.")
        return torch.device("cpu")
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(req)


def _as_discriminator_outputs(prediction):
    if isinstance(prediction, (list, tuple)):
        return list(prediction)
    return [prediction]


def _mse_gan_loss(prediction, target_is_real, mse):
    losses = []
    for pred in _as_discriminator_outputs(prediction):
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        losses.append(mse(pred, target))
    return sum(losses) / len(losses)


class ReplayBuffer:
    """Mix recent and historical fakes to reduce discriminator oscillation."""

    def __init__(self, max_size=50):
        self.max_size = max(0, int(max_size))
        self.data = []

    def push_and_pop(self, batch):
        if self.max_size <= 0:
            return batch

        samples = []
        for element in batch.detach():
            element = element.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(element.clone())
                samples.append(element)
                continue

            if random.random() > 0.5:
                idx = random.randint(0, self.max_size - 1)
                samples.append(self.data[idx].clone())
                self.data[idx] = element.clone()
            else:
                samples.append(element)
        return torch.cat(samples, dim=0)


def _build_linear_decay_scheduler(optimizer, total_epochs, start_epoch):
    total_epochs = max(1, int(total_epochs))
    start_epoch = min(max(0, int(start_epoch)), total_epochs)

    def lr_lambda(epoch_index):
        epoch_num = epoch_index + 1
        if epoch_num <= start_epoch:
            return 1.0
        decay_span = max(1, total_epochs - start_epoch)
        progress = min(epoch_num - start_epoch, decay_span)
        return max(0.0, 1.0 - (progress / decay_span))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def _run_eval_for_checkpoint(repo_root, run_dir, cfg, checkpoint_path, epoch, g_ab, g_ba, max_images=None):
    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    domain_a_test = data_cfg.get("domain_a_test")
    domain_b_test = data_cfg.get("domain_b_test")
    if not domain_a_test or not domain_b_test:
        print(
            "Per-epoch eval skipped: data.domain_a_test/domain_b_test not set in config.",
            flush=True,
        )
        return None

    repo_root = Path(repo_root).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    eval_dir = Path(run_dir).resolve() / "evaluate" / "by_epoch" / f"epoch_{int(epoch):03d}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    try:
        payload = evaluate_bidirectional_generators(
            g_ab=g_ab,
            g_ba=g_ba,
            input_a_dir=repo_root / str(domain_a_test),
            input_b_dir=repo_root / str(domain_b_test),
            generated_a_dir=eval_dir / "generated_A",
            generated_b_dir=eval_dir / "generated_B",
            real_a_dir=repo_root / str(domain_a_test),
            real_b_dir=repo_root / str(domain_b_test),
            image_size=int(data_cfg.get("image_size", 128)),
            batch_size=16,  # Match evaluate.py CLI default to keep behavior unchanged.
            max_images=max_images,
            use_cuda=torch.cuda.is_available(),
            out_json=eval_dir,
            checkpoint_path=checkpoint_path,
            config_used=cfg,
        )
    except Exception as exc:
        print(f"Per-epoch eval failed for epoch {epoch}: {exc}", flush=True)
        return None

    try:
        score = payload.get("gms_avg")
        if score is None:
            return None
        return float(score)
    except Exception as exc:
        print(f"Failed to parse eval output for epoch {epoch}: {exc}", flush=True)
        return None


def main():
    parser = argparse.ArgumentParser(description="Train CycleGAN on unpaired A/B image domains.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    repo_root = Path(__file__).resolve().parents[1]

    run_name = args.run_name or str(cfg.get("experiment_name", "cyclegan_run"))
    dirs = ensure_run_dirs(repo_root, run_name)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    loss_cfg = cfg.get("loss", {})
    data_cfg = cfg.get("data", {})
    device = _resolve_device(train_cfg.get("device", "auto"))
    discriminator_cfg = model_cfg.get("discriminator", {}) if isinstance(model_cfg, dict) else {}

    dataset = UnpairedImageDataset(
        domain_a_dir=repo_root / str(data_cfg.get("domain_a_train")),
        domain_b_dir=repo_root / str(data_cfg.get("domain_b_train")),
        image_size=int(data_cfg.get("image_size", 128)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        drop_last=True,
    )

    g_ab, g_ba, d_a, d_b = build_cyclegan_models(
        ngf=int(model_cfg.get("ngf", 64)),
        ndf=int(model_cfg.get("ndf", 64)),
        n_blocks=int(model_cfg.get("n_blocks", 6)),
        device=device,
        discriminator_cfg=discriminator_cfg,
    )

    lr_g = float(train_cfg.get("lr_g", 0.0002))
    lr_d = float(train_cfg.get("lr_d", 0.0002))
    betas = (float(train_cfg.get("beta1", 0.5)), float(train_cfg.get("beta2", 0.999)))
    n_epochs = int(args.epochs or train_cfg.get("epochs", 100))

    optim_g = torch.optim.Adam(
        list(g_ab.parameters()) + list(g_ba.parameters()),
        lr=lr_g,
        betas=betas,
    )
    optim_d_a = torch.optim.Adam(d_a.parameters(), lr=lr_d, betas=betas)
    optim_d_b = torch.optim.Adam(d_b.parameters(), lr=lr_d, betas=betas)

    lr_schedule = str(train_cfg.get("lr_schedule", "linear_decay")).strip().lower()
    lr_decay_start_epoch = int(train_cfg.get("lr_decay_start_epoch", max(1, n_epochs // 2)))
    sched_g = None
    sched_d_a = None
    sched_d_b = None
    if lr_schedule == "linear_decay":
        sched_g = _build_linear_decay_scheduler(optim_g, n_epochs, lr_decay_start_epoch)
        sched_d_a = _build_linear_decay_scheduler(optim_d_a, n_epochs, lr_decay_start_epoch)
        sched_d_b = _build_linear_decay_scheduler(optim_d_b, n_epochs, lr_decay_start_epoch)
    elif lr_schedule not in {"", "none"}:
        raise ValueError(f"Unsupported lr_schedule: {lr_schedule}")

    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    lambda_cycle_a = float(loss_cfg.get("lambda_cycle_a", 10.0))
    lambda_cycle_b = float(loss_cfg.get("lambda_cycle_b", 10.0))
    lambda_identity_a = float(loss_cfg.get("lambda_identity_a", 5.0))
    lambda_identity_b = float(loss_cfg.get("lambda_identity_b", 5.0))

    replay_buffer_size = int(train_cfg.get("replay_buffer_size", 0))
    fake_a_buffer = ReplayBuffer(replay_buffer_size)
    fake_b_buffer = ReplayBuffer(replay_buffer_size)

    log_every_steps = int(train_cfg.get("log_every_steps", 50))
    select_best_by = str(train_cfg.get("select_best_by", "eval_gms")).strip().lower()
    best_eval_every_epochs = max(1, int(train_cfg.get("best_eval_every_epochs", 1)))
    best_eval_max_images = train_cfg.get("best_eval_max_images", None)
    best_score_name = "gms_avg"

    metrics_csv = dirs["run_dir"] / "metrics.csv"
    best_score = float("inf")
    best_epoch = None
    has_best_checkpoint = False
    final_payload = None
    print(f"Starting training on device={device} with {len(loader)} steps/epoch for {n_epochs} epochs", flush=True)

    for epoch in range(1, n_epochs + 1):
        g_ab.train()
        g_ba.train()
        d_a.train()
        d_b.train()

        running = {
            "g_total": 0.0,
            "g_adv_ab": 0.0,
            "g_adv_ba": 0.0,
            "cycle_a": 0.0,
            "cycle_b": 0.0,
            "id_a": 0.0,
            "id_b": 0.0,
            "d_a_loss": 0.0,
            "d_b_loss": 0.0,
        }
        num_steps = 0

        for batch in loader:
            real_a = _to_device(batch["A"], device)
            real_b = _to_device(batch["B"], device)
            num_steps += 1

            # -------------------------
            # Train Generators
            # -------------------------
            optim_g.zero_grad(set_to_none=True)

            fake_b = g_ab(real_a)
            fake_a = g_ba(real_b)

            rec_a = g_ba(fake_b)
            rec_b = g_ab(fake_a)

            id_a = g_ba(real_a)
            id_b = g_ab(real_b)

            pred_fake_b = d_b(fake_b)
            pred_fake_a = d_a(fake_a)
            g_adv_ab = _mse_gan_loss(pred_fake_b, target_is_real=True, mse=mse)
            g_adv_ba = _mse_gan_loss(pred_fake_a, target_is_real=True, mse=mse)
            cycle_a = l1(rec_a, real_a)
            cycle_b = l1(rec_b, real_b)
            id_loss_a = l1(id_a, real_a)
            id_loss_b = l1(id_b, real_b)

            g_total = (
                g_adv_ab
                + g_adv_ba
                + lambda_cycle_a * cycle_a
                + lambda_cycle_b * cycle_b
                + lambda_identity_a * id_loss_a
                + lambda_identity_b * id_loss_b
            )
            g_total.backward()
            optim_g.step()

            # -------------------------
            # Train Discriminator A
            # -------------------------
            optim_d_a.zero_grad(set_to_none=True)
            pred_real_a = d_a(real_a)
            fake_a_for_d = fake_a_buffer.push_and_pop(fake_a)
            pred_fake_a_detached = d_a(fake_a_for_d.detach())
            d_a_real_loss = _mse_gan_loss(pred_real_a, target_is_real=True, mse=mse)
            d_a_fake_loss = _mse_gan_loss(pred_fake_a_detached, target_is_real=False, mse=mse)
            d_a_loss = 0.5 * (d_a_real_loss + d_a_fake_loss)
            d_a_loss.backward()
            optim_d_a.step()

            # -------------------------
            # Train Discriminator B
            # -------------------------
            optim_d_b.zero_grad(set_to_none=True)
            pred_real_b = d_b(real_b)
            fake_b_for_d = fake_b_buffer.push_and_pop(fake_b)
            pred_fake_b_detached = d_b(fake_b_for_d.detach())
            d_b_real_loss = _mse_gan_loss(pred_real_b, target_is_real=True, mse=mse)
            d_b_fake_loss = _mse_gan_loss(pred_fake_b_detached, target_is_real=False, mse=mse)
            d_b_loss = 0.5 * (d_b_real_loss + d_b_fake_loss)
            d_b_loss.backward()
            optim_d_b.step()

            running["g_total"] += float(g_total.item())
            running["g_adv_ab"] += float(g_adv_ab.item())
            running["g_adv_ba"] += float(g_adv_ba.item())
            running["cycle_a"] += float(cycle_a.item())
            running["cycle_b"] += float(cycle_b.item())
            running["id_a"] += float(id_loss_a.item())
            running["id_b"] += float(id_loss_b.item())
            running["d_a_loss"] += float(d_a_loss.item())
            running["d_b_loss"] += float(d_b_loss.item())

            if log_every_steps > 0 and (num_steps % log_every_steps == 0):
                print(
                    f"[Epoch {epoch:03d}/{n_epochs}] step {num_steps}/{len(loader)} "
                    f"G={g_total.item():.4f} D_A={d_a_loss.item():.4f} D_B={d_b_loss.item():.4f}",
                    flush=True,
                )

        if num_steps == 0:
            raise RuntimeError("No training batches produced. Check dataset paths and batch size.")

        epoch_row = {
            "epoch": epoch,
            "g_total": f"{running['g_total'] / num_steps:.6f}",
            "g_adv_ab": f"{running['g_adv_ab'] / num_steps:.6f}",
            "g_adv_ba": f"{running['g_adv_ba'] / num_steps:.6f}",
            "cycle_a": f"{running['cycle_a'] / num_steps:.6f}",
            "cycle_b": f"{running['cycle_b'] / num_steps:.6f}",
            "cycle_total": f"{(running['cycle_a'] + running['cycle_b']) / num_steps:.6f}",
            "id_a": f"{running['id_a'] / num_steps:.6f}",
            "id_b": f"{running['id_b'] / num_steps:.6f}",
            "d_a_loss": f"{running['d_a_loss'] / num_steps:.6f}",
            "d_b_loss": f"{running['d_b_loss'] / num_steps:.6f}",
            "lr_g": f"{optim_g.param_groups[0]['lr']:.8f}",
            "lr_d": f"{optim_d_a.param_groups[0]['lr']:.8f}",
        }
        append_metrics_row(metrics_csv, epoch_row)

        ckpt_payload = {
            "epoch": epoch,
            "config": cfg,
            "g_ab": g_ab.state_dict(),
            "g_ba": g_ba.state_dict(),
            "d_a": d_a.state_dict(),
            "d_b": d_b.state_dict(),
            "optim_g": optim_g.state_dict(),
            "optim_d_a": optim_d_a.state_dict(),
            "optim_d_b": optim_d_b.state_dict(),
            "sched_g": sched_g.state_dict() if sched_g is not None else None,
            "sched_d_a": sched_d_a.state_dict() if sched_d_a is not None else None,
            "sched_d_b": sched_d_b.state_dict() if sched_d_b is not None else None,
            "metrics": epoch_row,
        }
        final_payload = ckpt_payload

        candidate_ckpt = dirs["checkpoints"] / "_candidate_for_eval.pth"
        save_checkpoint(candidate_ckpt, ckpt_payload)

        score = None
        score_name = None
        if select_best_by == "eval_gms" and (epoch % best_eval_every_epochs == 0):
            score = _run_eval_for_checkpoint(
                repo_root=repo_root,
                run_dir=dirs["run_dir"],
                cfg=cfg,
                checkpoint_path=candidate_ckpt,
                epoch=epoch,
                g_ab=g_ab,
                g_ba=g_ba,
                max_images=best_eval_max_images,
            )
            score_name = best_score_name if score is not None else None

        if score is None:
            score = float(epoch_row["g_total"])
            score_name = "g_total_fallback"

        if score < best_score:
            best_score = score
            best_epoch = epoch
            has_best_checkpoint = True
            save_checkpoint(dirs["checkpoints"] / "best.pth", ckpt_payload)

        print(
            f"[Epoch {epoch:03d}/{n_epochs}] "
            f"G={float(epoch_row['g_total']):.4f} "
            f"D_A={float(epoch_row['d_a_loss']):.4f} "
            f"D_B={float(epoch_row['d_b_loss']):.4f} "
            f"best_metric={score_name}:{score:.5f}",
            flush=True,
        )

        if sched_g is not None:
            sched_g.step()
            sched_d_a.step()
            sched_d_b.step()

    if final_payload is None:
        final_payload = {
            "epoch": 0,
            "config": cfg,
            "g_ab": g_ab.state_dict(),
            "g_ba": g_ba.state_dict(),
            "d_a": d_a.state_dict(),
            "d_b": d_b.state_dict(),
            "optim_g": optim_g.state_dict(),
            "optim_d_a": optim_d_a.state_dict(),
            "optim_d_b": optim_d_b.state_dict(),
            "sched_g": sched_g.state_dict() if sched_g is not None else None,
            "sched_d_a": sched_d_a.state_dict() if sched_d_a is not None else None,
            "sched_d_b": sched_d_b.state_dict() if sched_d_b is not None else None,
            "metrics": {},
        }
    if not has_best_checkpoint:
        save_checkpoint(dirs["checkpoints"] / "best.pth", final_payload)
        print("No best checkpoint was selected during training; saved final checkpoint as best.pth.", flush=True)

    save_checkpoint(dirs["checkpoints"] / "final.pth", final_payload)

    save_loss_plot(metrics_csv, dirs["plots"] / "losses.png")
    run_auto_evaluation(repo_root=repo_root, dirs=dirs, cfg=cfg, model_type="gan")
    if best_epoch is not None:
        print(f"Best checkpoint selected at epoch {best_epoch} using minimized score={best_score:.5f}.", flush=True)
    print(f"Run complete: {run_name}", flush=True)
    print(f"Artifacts: {dirs['run_dir']}", flush=True)


if __name__ == "__main__":
    main()
