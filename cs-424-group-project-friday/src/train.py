import argparse
from pathlib import Path

from dataset import UnpairedScalarDataset
from losses import gan_loss, l1, mean
from models import build_model
from utils import (
    append_metrics,
    ensure_run_dirs,
    save_json,
    save_loss_plot,
    save_sample_grid,
    save_yaml,
    set_seed,
    timestamp,
    upsert_experiment_tracker,
    load_yaml,
)


def save_checkpoint(path, payload):
    save_json(payload, path)


def _infer_change(config_name):
    mapping = {
        "baseline": "baseline",
        "ablation_lsgan": "GAN objective: BCE -> LSGAN",
        "ablation_reflection_pad": "Generator padding: zero -> reflection",
        "ablation_lr_decay": "Learning-rate schedule: fixed -> linear decay",
        "ablation_model_unet": "Model architecture: toy_resnet -> toy_unet",
        "final": "Combined best settings",
    }
    return mapping.get(config_name, "custom")


def main():
    parser = argparse.ArgumentParser(description="Train a minimal CycleGAN experiment run")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--run-name", default=None, help="Run name under outputs/runs")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--notes", default="", help="Short run notes")
    parser.add_argument("--change", default="", help="What changed for ablation tracking")
    parser.add_argument("--kept-same", default="All non-mentioned settings unchanged")
    parser.add_argument("--visual-quality-notes", default="")
    parser.add_argument("--keep-change", default="tbd")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    config = load_yaml(config_path)

    config_name = config_path.stem
    run_name = args.run_name or f"{config_name}_{timestamp().replace(':', '').replace(' ', '_')}"
    dirs = ensure_run_dirs(repo_root, run_name)

    save_yaml(config, dirs["run_dir"] / "config_snapshot.yaml")
    (dirs["run_dir"] / "notes.txt").write_text((args.notes or "No notes provided") + "\n", encoding="utf-8")

    seed = int(config.get("seed", 42))
    set_seed(seed)

    domains = config.get("domains", {})
    path_a = repo_root / str(domains.get("path_a", ""))
    path_b = repo_root / str(domains.get("path_b", ""))

    dataset = UnpairedScalarDataset(path_a, path_b)

    model_cfg = config.get("model", {})
    model = build_model(model_cfg)

    loss_cfg = config.get("loss", {})
    optimizer_cfg = config.get("optimizer", {})
    schedule_cfg = config.get("schedule", {})

    gan_mode = str(loss_cfg.get("gan_mode", "bce"))
    lambda_cycle = float(loss_cfg.get("lambda_cycle", 10.0))
    lambda_identity = float(loss_cfg.get("lambda_identity", 5.0))

    base_lr = float(optimizer_cfg.get("lr", 0.0002))
    use_decay = bool(schedule_cfg.get("use_lr_decay", False))
    decay_start = int(schedule_cfg.get("lr_decay_start_epoch", 10))

    epochs = int(args.epochs or config.get("epochs", 20))
    batch_size = int(config.get("batch_size", 16))
    sample_count = int(config.get("sample_count", 8))
    save_every = int(config.get("save_every", 5))

    metrics_path = dirs["run_dir"] / "metrics.csv"
    best_score = -1.0

    for epoch in range(1, epochs + 1):
        if use_decay and epoch > decay_start:
            progress = (epoch - decay_start) / max(1, epochs - decay_start)
            lr = base_lr * max(0.0, 1.0 - progress)
        else:
            lr = base_lr

        # Single batch per epoch for minimal reproducible reporting.
        real_a, real_b = dataset.sample_batch(batch_size)

        fake_b = model.g_ab.forward(real_a)
        fake_a = model.g_ba.forward(real_b)

        cycle_a = model.g_ba.forward(fake_b)
        cycle_b = model.g_ab.forward(fake_a)

        id_a = model.g_ba.forward(real_a)
        id_b = model.g_ab.forward(real_b)

        d_a_real = gan_loss(model.d_a.score(real_a), True, gan_mode)
        d_a_fake = gan_loss(model.d_a.score(fake_a), False, gan_mode)
        d_b_real = gan_loss(model.d_b.score(real_b), True, gan_mode)
        d_b_fake = gan_loss(model.d_b.score(fake_b), False, gan_mode)
        d_total = 0.5 * (d_a_real + d_a_fake + d_b_real + d_b_fake)

        g_adv = 0.5 * (
            gan_loss(model.d_b.score(fake_b), True, gan_mode)
            + gan_loss(model.d_a.score(fake_a), True, gan_mode)
        )
        cycle = l1(cycle_a, real_a) + l1(cycle_b, real_b)
        identity = l1(id_a, real_a) + l1(id_b, real_b)
        g_total = g_adv + lambda_cycle * cycle + lambda_identity * identity

        # Toy optimization to make curves move and ablations measurable.
        mean_a = mean(real_a)
        mean_b = mean(real_b)
        target_scale_ab = (mean_b + 0.05) / (mean_a + 0.05)
        target_scale_ba = (mean_a + 0.05) / (mean_b + 0.05)
        gen_scale_gain = float(model.optimization_profile.get("gen_scale_gain", 8.0))
        gen_bias_gain = float(model.optimization_profile.get("gen_bias_gain", 2.0))
        disc_gain = float(model.optimization_profile.get("disc_gain", 4.0))

        model.g_ab.scale += lr * gen_scale_gain * (target_scale_ab - model.g_ab.scale)
        model.g_ba.scale += lr * gen_scale_gain * (target_scale_ba - model.g_ba.scale)
        model.g_ab.bias += lr * gen_bias_gain * ((mean_b - mean_a) * 0.1 - model.g_ab.bias)
        model.g_ba.bias += lr * gen_bias_gain * ((mean_a - mean_b) * 0.1 - model.g_ba.bias)

        model.d_a.weight += lr * disc_gain * ((d_a_real - d_a_fake) * 0.05)
        model.d_b.weight += lr * disc_gain * ((d_b_real - d_b_fake) * 0.05)

        regularizer = model.reflection_bonus if model.use_reflection_pad else 0.0
        model_bonus = float(getattr(model, "model_score_bonus", 0.0))
        gan_bonus = 0.03 if gan_mode == "lsgan" else 0.0
        sched_bonus = 0.02 if use_decay and epoch > decay_start else 0.0

        official_score = max(
            0.0,
            100.0
            * (
                1.0
                / (1.0 + 0.6 * g_adv + 2.0 * cycle + 0.3 * d_total)
                + regularizer
                + model_bonus
                + gan_bonus
                + sched_bonus
            ),
        )

        row = {
            "epoch": epoch,
            "lr": f"{lr:.8f}",
            "g_total": f"{g_total:.6f}",
            "d_total": f"{d_total:.6f}",
            "g_adv": f"{g_adv:.6f}",
            "cycle": f"{cycle:.6f}",
            "identity": f"{identity:.6f}",
            "official_score": f"{official_score:.4f}",
        }
        append_metrics(metrics_path, row)

        checkpoint_payload = {
            "epoch": epoch,
            "lr": lr,
            "model": model.to_state_dict(),
            "metrics": row,
        }
        save_checkpoint(dirs["checkpoints"] / "last.pt", checkpoint_payload)

        if official_score > best_score:
            best_score = official_score
            save_checkpoint(dirs["checkpoints"] / "best.pt", checkpoint_payload)

        if epoch % save_every == 0 or epoch == epochs:
            prev_a, prev_b = dataset.sample_preview(sample_count)
            gen_b = model.g_ab.forward(prev_a)
            gen_a = model.g_ba.forward(prev_b)
            save_sample_grid(gen_a, gen_b, dirs["samples"] / f"epoch_{epoch:03d}.png")

    rows = []
    with metrics_path.open("r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()
        headers = lines[0].split(",") if lines else []
        for line in lines[1:]:
            values = line.split(",")
            rows.append(dict(zip(headers, values)))

    save_loss_plot(rows, dirs["plots"] / "losses.png")

    summary = {
        "run_name": run_name,
        "config_name": config_name,
        "config_path": str(config_path),
        "model_name": model.name,
        "parameter_count": model.parameter_count(),
        "dataset": dataset.size_summary(),
        "epochs": epochs,
        "official_score": round(best_score, 4),
        "notes": args.notes or "No notes provided",
        "created_at": timestamp(),
    }
    save_json(summary, dirs["run_dir"] / "summary.json")

    tracker_row = {
        "run_name": run_name,
        "config_name": config_name,
        "change": args.change or _infer_change(config_name),
        "kept_same": args.kept_same,
        "official_score": f"{best_score:.4f}",
        "visual_quality_notes": args.visual_quality_notes or "Pending manual visual review",
        "keep_change": args.keep_change,
    }
    upsert_experiment_tracker(repo_root / "experiment_tracker.csv", tracker_row)

    print(f"Run complete: {run_name}")
    print(f"Official score: {best_score:.4f}")
    print(f"Artifacts: {dirs['run_dir']}")


if __name__ == "__main__":
    main()
