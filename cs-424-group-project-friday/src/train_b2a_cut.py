import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cut_models import CutGenerator, PatchDiscriminator, PatchSampleMLP, PatchNCELoss, init_weights
from cyclegan_dataset import UnpairedImageDataset
from cyclegan_io import ensure_run_dirs, run_auto_evaluation, save_checkpoint
from yaml_utils import load_yaml

METRICS_FIELDS = ["epoch", "g_total", "g_adv_ba", "nce_total", "nce_idt", "d_a_loss", "lr_g", "lr_d", "lr_f"]


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_device(requested):
    req = str(requested).strip().lower()
    if req.startswith("cuda") and not torch.cuda.is_available():
        print("Requested CUDA but CUDA is unavailable in this PyTorch build. Falling back to CPU.")
        return torch.device("cpu")
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(req)


def _append_metrics_row(csv_path, row):
    csv_path = Path(csv_path)
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_header = next(reader, [])
        if existing_header != METRICS_FIELDS:
            migrated_rows = []
            with csv_path.open("r", newline="", encoding="utf-8") as f:
                old_rows = list(csv.DictReader(f))
            for old in old_rows:
                migrated_rows.append(
                    {
                        "epoch": old.get("epoch", ""),
                        "g_total": old.get("g_total", ""),
                        "g_adv_ba": old.get("g_adv_ba", ""),
                        "nce_total": old.get("nce_total", ""),
                        "nce_idt": old.get("nce_idt", ""),
                        "d_a_loss": old.get("d_a_loss", ""),
                        "lr_g": old.get("lr_g", ""),
                        "lr_d": old.get("lr_d", ""),
                        "lr_f": old.get("lr_f", ""),
                    }
                )
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=METRICS_FIELDS)
                writer.writeheader()
                for migrated in migrated_rows:
                    writer.writerow(migrated)

    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _save_loss_plot(metrics_csv, out_png):
    metrics_csv = Path(metrics_csv)
    if not metrics_csv.exists():
        return
    with metrics_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    epochs = [int(r["epoch"]) for r in rows]
    g_total = [float(r["g_total"]) for r in rows]
    g_adv = [float(r["g_adv_ba"]) for r in rows]
    nce = [float(r["nce_total"]) for r in rows]
    d_a = [float(r["d_a_loss"]) for r in rows]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, g_total, label="g_total")
    plt.plot(epochs, g_adv, label="g_adv_ba")
    plt.plot(epochs, nce, label="nce_total")
    plt.plot(epochs, d_a, label="d_a_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("B->A CUT Loss Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train one-way CUT-style model for B->A translation.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    repo_root = Path(__file__).resolve().parents[1]
    run_name = args.run_name or str(cfg.get("experiment_name", "gan_b2a_cut_run"))
    dirs = ensure_run_dirs(repo_root, run_name)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    loss_cfg = cfg.get("loss", {})
    device = _resolve_device(train_cfg.get("device", "auto"))

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

    ngf = int(model_cfg.get("ngf", 64))
    ndf = int(model_cfg.get("ndf", 64))
    n_blocks = int(model_cfg.get("n_blocks", 6))
    proj_dim = int(model_cfg.get("proj_dim", 256))
    num_patches = int(model_cfg.get("num_patches", 256))

    g_ba = CutGenerator(3, 3, ngf=ngf, n_blocks=n_blocks).to(device)
    d_a = PatchDiscriminator(3, ndf=ndf).to(device)
    in_channels_list = [ngf, ngf * 2, ngf * 4, ngf * 4]
    f_mlp = PatchSampleMLP(in_channels_list=in_channels_list, proj_dim=proj_dim).to(device)

    g_ba.apply(init_weights)
    d_a.apply(init_weights)
    f_mlp.apply(init_weights)

    lr_g = float(train_cfg.get("lr_g", 0.0002))
    lr_d = float(train_cfg.get("lr_d", 0.0002))
    lr_f = float(train_cfg.get("lr_f", lr_g))
    betas = (float(train_cfg.get("beta1", 0.5)), float(train_cfg.get("beta2", 0.999)))

    optim_g = torch.optim.Adam(g_ba.parameters(), lr=lr_g, betas=betas)
    optim_d = torch.optim.Adam(d_a.parameters(), lr=lr_d, betas=betas)
    optim_f = torch.optim.Adam(f_mlp.parameters(), lr=lr_f, betas=betas)

    mse = nn.MSELoss()
    nce_criterion = PatchNCELoss(temperature=float(loss_cfg.get("nce_temperature", 0.07)))

    lambda_nce = float(loss_cfg.get("lambda_nce", 1.0))
    lambda_identity = float(loss_cfg.get("lambda_identity_a", 1.0))
    use_nce_idt = bool(loss_cfg.get("use_nce_identity", True))

    n_epochs = int(args.epochs or train_cfg.get("epochs", 100))
    log_every_steps = int(train_cfg.get("log_every_steps", 50))

    metrics_csv = dirs["run_dir"] / "metrics.csv"
    best_score = float("inf")
    final_payload = None

    print(
        f"Starting B->A CUT training on device={device} with {len(loader)} steps/epoch for {n_epochs} epochs",
        flush=True,
    )

    for epoch in range(1, n_epochs + 1):
        g_ba.train()
        d_a.train()
        f_mlp.train()

        running = {
            "g_total": 0.0,
            "g_adv_ba": 0.0,
            "nce_total": 0.0,
            "nce_idt": 0.0,
            "d_a_loss": 0.0,
        }
        num_steps = 0

        for batch in loader:
            real_a = batch["A"].to(device, non_blocking=True)
            real_b = batch["B"].to(device, non_blocking=True)
            num_steps += 1

            # -------------------------
            # Train G and F (CUT loss)
            # -------------------------
            optim_g.zero_grad(set_to_none=True)
            optim_f.zero_grad(set_to_none=True)

            fake_a, feats_q = g_ba(real_b, return_features=True)
            _, feats_k = g_ba(real_b, return_features=True)

            pred_fake = d_a(fake_a)
            g_adv_ba = mse(pred_fake, torch.ones_like(pred_fake, device=device))

            # Use early/mid features for NCE
            feats_q_sel = [feats_q[0], feats_q[1], feats_q[2], feats_q[3]]
            feats_k_sel = [feats_k[0], feats_k[1], feats_k[2], feats_k[3]]
            proj_q, sample_ids = f_mlp(feats_q_sel, num_patches=num_patches)
            proj_k, _ = f_mlp(feats_k_sel, num_patches=num_patches, patch_ids=sample_ids)

            nce_loss = 0.0
            for fq, fk in zip(proj_q, proj_k):
                nce_loss = nce_loss + nce_criterion(fq, fk)
            nce_loss = nce_loss / len(proj_q)

            nce_idt = torch.tensor(0.0, device=device)
            if use_nce_idt:
                _, feats_idt_q = g_ba(real_a, return_features=True)
                _, feats_idt_k = g_ba(real_a, return_features=True)
                feats_iq_sel = [feats_idt_q[0], feats_idt_q[1], feats_idt_q[2], feats_idt_q[3]]
                feats_ik_sel = [feats_idt_k[0], feats_idt_k[1], feats_idt_k[2], feats_idt_k[3]]
                proj_iq, sample_ids_i = f_mlp(feats_iq_sel, num_patches=num_patches)
                proj_ik, _ = f_mlp(feats_ik_sel, num_patches=num_patches, patch_ids=sample_ids_i)
                for fq, fk in zip(proj_iq, proj_ik):
                    nce_idt = nce_idt + nce_criterion(fq, fk)
                nce_idt = nce_idt / len(proj_iq)

            g_total = g_adv_ba + lambda_nce * nce_loss + lambda_identity * nce_idt
            g_total.backward()
            optim_g.step()
            optim_f.step()

            # -------------------------
            # Train D_A
            # -------------------------
            optim_d.zero_grad(set_to_none=True)
            pred_real = d_a(real_a)
            pred_fake_detached = d_a(fake_a.detach())
            d_real_loss = mse(pred_real, torch.ones_like(pred_real, device=device))
            d_fake_loss = mse(pred_fake_detached, torch.zeros_like(pred_fake_detached, device=device))
            d_a_loss = 0.5 * (d_real_loss + d_fake_loss)
            d_a_loss.backward()
            optim_d.step()

            running["g_total"] += float(g_total.item())
            running["g_adv_ba"] += float(g_adv_ba.item())
            running["nce_total"] += float(nce_loss.item())
            running["nce_idt"] += float(nce_idt.item())
            running["d_a_loss"] += float(d_a_loss.item())

            if log_every_steps > 0 and (num_steps % log_every_steps == 0):
                print(
                    f"[Epoch {epoch:03d}/{n_epochs}] step {num_steps}/{len(loader)} "
                    f"G={g_total.item():.4f} ADV={g_adv_ba.item():.4f} NCE={nce_loss.item():.4f} D_A={d_a_loss.item():.4f}",
                    flush=True,
                )

        if num_steps == 0:
            raise RuntimeError("No batches produced. Check dataset paths and batch size.")

        epoch_row = {
            "epoch": epoch,
            "g_total": f"{running['g_total'] / num_steps:.6f}",
            "g_adv_ba": f"{running['g_adv_ba'] / num_steps:.6f}",
            "nce_total": f"{running['nce_total'] / num_steps:.6f}",
            "nce_idt": f"{running['nce_idt'] / num_steps:.6f}",
            "d_a_loss": f"{running['d_a_loss'] / num_steps:.6f}",
            "lr_g": f"{lr_g:.8f}",
            "lr_d": f"{lr_d:.8f}",
            "lr_f": f"{lr_f:.8f}",
        }
        _append_metrics_row(metrics_csv, epoch_row)

        ckpt_payload = {
            "epoch": epoch,
            "config": cfg,
            "g_ba": g_ba.state_dict(),
            "d_a": d_a.state_dict(),
            "f_mlp": f_mlp.state_dict(),
            "optim_g": optim_g.state_dict(),
            "optim_d": optim_d.state_dict(),
            "optim_f": optim_f.state_dict(),
            "metrics": epoch_row,
        }
        final_payload = ckpt_payload

        score = float(epoch_row["g_total"])
        if score < best_score:
            best_score = score
            save_checkpoint(dirs["checkpoints"] / "best.pth", ckpt_payload)

        print(
            f"[Epoch {epoch:03d}/{n_epochs}] "
            f"G={float(epoch_row['g_total']):.4f} "
            f"ADV={float(epoch_row['g_adv_ba']):.4f} "
            f"NCE={float(epoch_row['nce_total']):.4f} "
            f"D_A={float(epoch_row['d_a_loss']):.4f}",
            flush=True,
        )

    if final_payload is None:
        final_payload = {
            "epoch": 0,
            "config": cfg,
            "g_ba": g_ba.state_dict(),
            "d_a": d_a.state_dict(),
            "f_mlp": f_mlp.state_dict(),
            "optim_g": optim_g.state_dict(),
            "optim_d": optim_d.state_dict(),
            "optim_f": optim_f.state_dict(),
            "metrics": {},
        }
    save_checkpoint(dirs["checkpoints"] / "final.pth", final_payload)

    _save_loss_plot(metrics_csv, dirs["plots"] / "losses.png")
    run_auto_evaluation(repo_root=repo_root, dirs=dirs, cfg=cfg, model_type="cut")
    print(f"Run complete: {run_name}", flush=True)
    print(f"Artifacts: {dirs['run_dir']}", flush=True)


if __name__ == "__main__":
    main()
