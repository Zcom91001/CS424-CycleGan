import csv
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image


def ensure_run_dirs(repo_root, run_name):
    run_dir = Path(repo_root) / "outputs" / "runs" / run_name
    checkpoints = run_dir / "checkpoints"
    samples = run_dir / "samples"
    plots = run_dir / "plots"
    for p in [run_dir, checkpoints, samples, plots]:
        p.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_dir,
        "checkpoints": checkpoints,
        "samples": samples,
        "plots": plots,
    }


def append_metrics_row(csv_path, row):
    csv_path = Path(csv_path)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_metrics(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            if not row:
                continue
            # Guard against duplicated header rows that can appear after resumed/appended runs.
            epoch_raw = str(row.get("epoch", "")).strip().lower()
            if epoch_raw in {"", "epoch"}:
                continue
            rows.append(row)
        return rows


def tensor_to_pil(img_tensor):
    # img_tensor expected in [-1, 1], shape (3, H, W)
    x = img_tensor.detach().cpu().clamp(-1.0, 1.0)
    x = (x + 1.0) * 0.5
    x = (x * 255.0).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(x)


def save_image_grid(batch_tensor, out_path, nrow=4):
    out_path = Path(out_path)
    imgs = [tensor_to_pil(t) for t in batch_tensor]
    w, h = imgs[0].size
    ncol = int(nrow)
    n = len(imgs)
    nrows = (n + ncol - 1) // ncol
    canvas = Image.new("RGB", (ncol * w, nrows * h), (255, 255, 255))
    for i, img in enumerate(imgs):
        r = i // ncol
        c = i % ncol
        canvas.paste(img, (c * w, r * h))
    canvas.save(out_path)


def save_loss_plot(metrics_csv_path, out_png_path):
    rows = read_metrics(metrics_csv_path)
    if not rows:
        return

    # Auto-detect training schema from CSV columns and plot available loss curves.
    if {"d_b_loss", "cycle_total"}.issubset(rows[0].keys()):
        parsed = []
        for r in rows:
            try:
                parsed.append(
                    (
                        int(str(r["epoch"]).strip()),
                        float(str(r["g_total"]).strip()),
                        float(str(r["d_a_loss"]).strip()),
                        float(str(r["d_b_loss"]).strip()),
                        float(str(r["cycle_total"]).strip()),
                    )
                )
            except (KeyError, ValueError, TypeError):
                continue
        if not parsed:
            return
        epochs = [x[0] for x in parsed]
        g_total = [x[1] for x in parsed]
        d_a = [x[2] for x in parsed]
        d_b = [x[3] for x in parsed]
        cycle = [x[4] for x in parsed]

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, g_total, label="g_total")
        plt.plot(epochs, d_a, label="d_a_loss")
        plt.plot(epochs, d_b, label="d_b_loss")
        plt.plot(epochs, cycle, label="cycle_total")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("CycleGAN Loss Curves")
        plt.xlim(left=1)
        plt.ylim(bottom=0)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_png_path, dpi=200)
        plt.close()
        return

    # One-way GAN/CUT schema.
    parsed = []
    for r in rows:
        try:
            parsed.append(
                (
                    int(str(r["epoch"]).strip()),
                    float(str(r["g_total"]).strip()),
                    float(str(r.get("g_adv_ba", "nan")).strip()),
                    float(str(r.get("d_a_loss", "nan")).strip()),
                    float(str(r.get("id_a", "nan")).strip()),
                    float(str(r.get("nce_total", "nan")).strip()),
                    float(str(r.get("perc_a", "nan")).strip()),
                )
            )
        except (KeyError, ValueError, TypeError):
            continue
    if not parsed:
        return

    epochs = [x[0] for x in parsed]
    g_total = [x[1] for x in parsed]
    g_adv = [x[2] for x in parsed]
    d_a = [x[3] for x in parsed]
    id_a = [x[4] for x in parsed]
    nce_total = [x[5] for x in parsed]
    perc_a = [x[6] for x in parsed]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, g_total, label="g_total")
    if any(v == v for v in g_adv):  # NaN-safe check
        plt.plot(epochs, g_adv, label="g_adv_ba")
    if any(v == v for v in d_a):
        plt.plot(epochs, d_a, label="d_a_loss")
    if any(v == v for v in id_a):
        plt.plot(epochs, id_a, label="id_a")
    if any(v == v for v in nce_total):
        plt.plot(epochs, nce_total, label="nce_total")
    if any(v == v for v in perc_a):
        plt.plot(epochs, perc_a, label="perc_a")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAN/CUT Loss Curves")
    plt.xlim(left=1)
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=200)
    plt.close()


def save_checkpoint(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def run_auto_evaluation(repo_root, dirs, cfg, model_type):
    eval_cfg = cfg.get("eval", {}) if isinstance(cfg, dict) else {}
    if bool(eval_cfg.get("auto_run", True)) is False:
        print("Auto-eval disabled via config: eval.auto_run=false", flush=True)
        return None

    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    domain_a_test = data_cfg.get("domain_a_test")
    domain_b_test = data_cfg.get("domain_b_test")
    if not domain_a_test or not domain_b_test:
        print("Auto-eval skipped: data.domain_a_test/domain_b_test not set in config.", flush=True)
        return None

    repo_root = Path(repo_root).resolve()
    checkpoints_dir = Path(dirs["checkpoints"]).resolve()
    best_ckpt = checkpoints_dir / "best.pth"
    final_ckpt = checkpoints_dir / "final.pth"
    if best_ckpt.exists():
        checkpoint_path = best_ckpt
    elif final_ckpt.exists():
        checkpoint_path = final_ckpt
    else:
        print("Auto-eval skipped: neither best.pth nor final.pth exists.", flush=True)
        return None

    eval_dir = Path(dirs["run_dir"]).resolve() / "evaluate"
    generated_dir = eval_dir / "generated"
    generated_a_dir = eval_dir / "generated_A"
    generated_b_dir = eval_dir / "generated_B"
    input_b_dir = repo_root / str(domain_b_test)
    input_a_dir = repo_root / str(domain_a_test)
    real_dir = repo_root / str(domain_a_test)
    real_a_dir = repo_root / str(domain_a_test)
    real_b_dir = repo_root / str(domain_b_test)

    if str(model_type).lower() == "gan":
        cmd = [
            sys.executable,
            str(repo_root / "src" / "evaluate.py"),
            "--bidirectional",
            "--checkpoint",
            str(checkpoint_path),
            "--input-a-dir",
            str(input_a_dir),
            "--input-b-dir",
            str(input_b_dir),
            "--generated-a-dir",
            str(generated_a_dir),
            "--generated-b-dir",
            str(generated_b_dir),
            "--real-a-dir",
            str(real_a_dir),
            "--real-b-dir",
            str(real_b_dir),
            "--model-type",
            str(model_type),
            "--image-size",
            str(int(data_cfg.get("image_size", 128))),
            "--allow-nonempty-generated-dir",
            "--out-json",
            str(eval_dir),
            "--submission-csv",
            str(eval_dir / "Userid.csv"),
        ]
    else:
        cmd = [
            sys.executable,
            str(repo_root / "src" / "evaluate.py"),
            "--checkpoint",
            str(checkpoint_path),
            "--input-b-dir",
            str(input_b_dir),
            "--generated-dir",
            str(generated_dir),
            "--real-dir",
            str(real_dir),
            "--model-type",
            str(model_type),
            "--image-size",
            str(int(data_cfg.get("image_size", 128))),
            "--allow-nonempty-generated-dir",
            "--out-json",
            str(eval_dir),
        ]

    max_images = eval_cfg.get("max_images")
    if max_images is not None:
        cmd.extend(["--max-images", str(int(max_images))])

    print("Starting auto-eval on best checkpoint...", flush=True)
    try:
        subprocess.run(cmd, check=True)
        print(f"Auto-eval complete: {eval_dir}", flush=True)
        return eval_dir
    except Exception as exc:
        print(f"Auto-eval failed: {exc}", flush=True)
        return None
