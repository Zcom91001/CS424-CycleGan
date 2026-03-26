import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import re

import torch
from PIL import Image


from cyclegan_models import ResnetGenerator
from cyclegan_io import tensor_to_pil


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _count_images(root):
    root = Path(root)
    if not root.exists():
        return 0
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _list_images(root):
    root = Path(root)
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def _pil_to_tensor_rgb(img, image_size):
    if image_size is not None:
        img = img.resize((image_size, image_size), Image.BICUBIC)
    img = img.convert("RGB")
    h = img.size[1]
    w = img.size[0]
    raw = bytearray(img.tobytes())
    data = torch.tensor(raw, dtype=torch.uint8)
    data = data.view(h, w, 3).permute(2, 0, 1).float() / 255.0
    # Keep the same [-1, 1] normalization used in training.
    return data * 2.0 - 1.0


def _build_generator(model_type, ngf, n_blocks, device):
    if model_type == "cut":
        model = CutGenerator(3, 3, ngf=ngf, n_blocks=n_blocks)
    elif model_type == "gan":
        model = ResnetGenerator(3, 3, ngf=ngf, n_blocks=n_blocks)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return model.to(device)


def _infer_model_type(payload, user_choice):
    if user_choice != "auto":
        return user_choice
    if isinstance(payload, dict) and "f_mlp" in payload:
        return "cut"
    # Both CycleGAN and one-way GAN use ResnetGenerator for g_ba.
    return "gan"


def _extract_generator_state(ckpt, key="g_ba"):
    # Training scripts save payloads with keys like "g_ab" and/or "g_ba".
    if isinstance(ckpt, dict) and key in ckpt:
        return ckpt[key]
    # Fallback: allow raw state_dict checkpoints.
    if isinstance(ckpt, dict):
        if key != "g_ba":
            raise ValueError(f"Checkpoint does not contain '{key}'.")
        return ckpt
    raise ValueError(f"Checkpoint format is not supported. Expected a dict with key '{key}'.")


def _infer_hparams_from_state_dict(state, model_type):
    if not isinstance(state, dict):
        return None, None

    if model_type == "gan":
        ngf = None
        if "net.1.weight" in state:
            ngf = int(state["net.1.weight"].shape[0])
        block_ids = set()
        for k in state.keys():
            m = re.match(r"net\.(\d+)\.block\.1\.weight$", str(k))
            if m:
                block_ids.add(int(m.group(1)))
        n_blocks = len(block_ids) if block_ids else None
        return ngf, n_blocks

    if model_type == "cut":
        ngf = None
        if "stem.1.weight" in state:
            ngf = int(state["stem.1.weight"].shape[0])
        block_ids = set()
        for k in state.keys():
            m = re.match(r"resblocks\.(\d+)\.block\.1\.weight$", str(k))
            if m:
                block_ids.add(int(m.group(1)))
        n_blocks = len(block_ids) if block_ids else None
        return ngf, n_blocks

    return None, None


def _generate_images_from_checkpoint(
    checkpoint_path,
    input_dir,
    generated_dir,
    model_type,
    ngf,
    n_blocks,
    image_size,
    batch_size,
    max_images,
    device,
    generator_key,
):
    checkpoint_path = Path(checkpoint_path).resolve()
    input_dir = Path(input_dir).resolve()
    generated_dir = Path(generated_dir).resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    input_images = _list_images(input_dir)
    if len(input_images) == 0:
        raise ValueError(f"No images found in input folder: {input_dir}")
    if max_images is not None:
        input_images = input_images[: int(max_images)]

    ckpt = torch.load(str(checkpoint_path), map_location=device)
    chosen_model = _infer_model_type(ckpt, model_type)
    state = _extract_generator_state(ckpt, key=generator_key)

    ckpt_ngf = None
    ckpt_n_blocks = None
    if isinstance(ckpt, dict):
        model_cfg = ckpt.get("config", {}).get("model", {})
        if isinstance(model_cfg, dict):
            if model_cfg.get("ngf") is not None:
                ckpt_ngf = int(model_cfg.get("ngf"))
            if model_cfg.get("n_blocks") is not None:
                ckpt_n_blocks = int(model_cfg.get("n_blocks"))

    state_ngf, state_n_blocks = _infer_hparams_from_state_dict(state, chosen_model)

    resolved_ngf = int(ngf) if ngf is not None else (ckpt_ngf or state_ngf or 64)
    resolved_n_blocks = int(n_blocks) if n_blocks is not None else (ckpt_n_blocks or state_n_blocks or 6)

    generator = _build_generator(
        chosen_model, ngf=resolved_ngf, n_blocks=resolved_n_blocks, device=device
    )
    generator.load_state_dict(state, strict=True)
    generator.eval()

    generated_dir.mkdir(parents=True, exist_ok=True)
    total = len(input_images)
    with torch.inference_mode():
        for start in range(0, total, int(batch_size)):
            batch_paths = input_images[start : start + int(batch_size)]
            batch_tensors = []
            for p in batch_paths:
                with Image.open(p) as img:
                    batch_tensors.append(_pil_to_tensor_rgb(img, image_size=image_size))
            x_b = torch.stack(batch_tensors, dim=0).to(device)
            fake_a = generator(x_b)

            for src_path, out_tensor in zip(batch_paths, fake_a):
                # Save as PNG for consistent metric input format.
                out_path = generated_dir / f"{src_path.stem}.png"
                tensor_to_pil(out_tensor).save(out_path)

    return {
        "generated_count": len(input_images),
        "model_type": chosen_model,
        "generator_key": generator_key,
        "config_used": ckpt.get("config") if isinstance(ckpt, dict) else None,
    }


def _compute_metrics_for_dirs(generated_dir, real_dir, use_cuda):
    generated_dir = Path(generated_dir).resolve()
    real_dir = Path(real_dir).resolve()

    if not generated_dir.exists():
        raise FileNotFoundError(f"Generated folder does not exist: {generated_dir}")
    if not real_dir.exists():
        raise FileNotFoundError(f"Real folder does not exist: {real_dir}")

    n_gen = _count_images(generated_dir)
    n_real = _count_images(real_dir)
    if n_gen == 0:
        raise ValueError(f"No images found in generated folder: {generated_dir}")
    if n_real == 0:
        raise ValueError(f"No images found in real folder: {real_dir}")

    import torch_fidelity

    metrics = torch_fidelity.calculate_metrics(
        input1=str(generated_dir),
        input2=str(real_dir),
        cuda=use_cuda,
        fid=True,
        isc=True,
    )

    fid_score = float(metrics["frechet_inception_distance"])
    is_mean = float(metrics["inception_score_mean"])
    is_std = float(metrics["inception_score_std"])
    gms_score = (fid_score / is_mean) ** 0.5 if is_mean > 0 else None

    return {
        "generated_dir": str(generated_dir),
        "real_dir": str(real_dir),
        "generated_images": n_gen,
        "real_images": n_real,
        "fid": fid_score,
        "is_mean": is_mean,
        "is_std": is_std,
        "gms": gms_score,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate generated images with FID/IS and derived GMS=sqrt(FID/IS).\n"
            "Single mode: evaluate one direction (default, backward compatible).\n"
            "Bidirectional mode: generate/evaluate A->B and B->A, then average both GMS values."
        )
    )
    parser.add_argument(
        "--generated-dir",
        default=None,
        help="Folder containing generated B->A images (or output folder for generation mode)",
    )
    parser.add_argument(
        "--real-dir",
        default=None,
        help="Folder containing real validation/target A images",
    )
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="Enable two-direction evaluation (A->B and B->A) and output their averaged GMS.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional path to model checkpoint. If set, script generates B->A images before computing metrics.",
    )
    parser.add_argument(
        "--input-b-dir",
        default=None,
        help="Folder of source B images to translate when --checkpoint is provided.",
    )
    parser.add_argument(
        "--input-a-dir",
        default=None,
        help="Folder of source A images to translate in --bidirectional mode.",
    )
    parser.add_argument(
        "--generated-a-dir",
        default=None,
        help="Output folder for generated B->A images in --bidirectional mode.",
    )
    parser.add_argument(
        "--generated-b-dir",
        default=None,
        help="Output folder for generated A->B images in --bidirectional mode.",
    )
    parser.add_argument(
        "--real-a-dir",
        default=None,
        help="Folder containing real A images (target for B->A) in --bidirectional mode.",
    )
    parser.add_argument(
        "--real-b-dir",
        default=None,
        help="Folder containing real B images (target for A->B) in --bidirectional mode.",
    )
    parser.add_argument(
        "--model-type",
        default="auto",
        choices=["auto", "gan", "cut"],
        help="Generator type for checkpoint loading. 'auto' detects CUT by checkpoint keys.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Resize B inputs before generation (must match training resolution).",
    )
    parser.add_argument(
        "--ngf",
        type=int,
        default=None,
        help="Generator ngf used during training (auto-inferred from checkpoint if omitted)",
    )
    parser.add_argument(
        "--n-blocks",
        type=int,
        default=None,
        help="Generator residual block count (auto-inferred from checkpoint if omitted)",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Generation batch size")
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap on number of B images to generate/evaluate.",
    )
    parser.add_argument(
        "--allow-nonempty-generated-dir",
        action="store_true",
        help=(
            "Allow generation into a folder that already contains images. "
            "Without this flag, generation mode requires an empty/non-existent generated folder "
            "to avoid mixing old and new outputs."
        ),
    )
    parser.add_argument(
        "--cuda",
        default="auto",
        choices=["auto", "true", "false"],
        help="Use CUDA for metric computation",
    )
    parser.add_argument("--out-json", default=None, help="Optional output JSON path")
    parser.add_argument("--out-csv", default=None, help="Optional output CSV path")
    parser.add_argument(
        "--submission-csv",
        default=None,
        help="Optional path to save submission-style CSV with columns: id,label where label is avg GMS.",
    )
    args = parser.parse_args()

    try:
        import torch_fidelity
    except ImportError as exc:
        raise ImportError(
            "torch_fidelity is required for evaluation. Install with: pip install torch-fidelity"
        ) from exc

    if args.cuda == "auto":
        use_cuda = torch.cuda.is_available()
    else:
        use_cuda = args.cuda == "true"
    device = "cuda" if use_cuda else "cpu"

    if args.bidirectional:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required in --bidirectional mode")
        if args.input_a_dir is None or args.input_b_dir is None:
            raise ValueError("--input-a-dir and --input-b-dir are required in --bidirectional mode")
        if args.generated_a_dir is None or args.generated_b_dir is None:
            raise ValueError("--generated-a-dir and --generated-b-dir are required in --bidirectional mode")
        if args.real_a_dir is None or args.real_b_dir is None:
            raise ValueError("--real-a-dir and --real-b-dir are required in --bidirectional mode")
        if args.model_type == "cut":
            raise ValueError("--bidirectional mode is not supported for CUT checkpoints (one-way generator).")

        generated_a_dir = Path(args.generated_a_dir).resolve()
        generated_b_dir = Path(args.generated_b_dir).resolve()
        existing_a = _count_images(generated_a_dir)
        existing_b = _count_images(generated_b_dir)
        if not args.allow_nonempty_generated_dir:
            if existing_a > 0:
                raise ValueError(
                    f"Generated A folder is not empty ({existing_a} images): {generated_a_dir}\n"
                    "Use an empty folder or pass --allow-nonempty-generated-dir to proceed."
                )
            if existing_b > 0:
                raise ValueError(
                    f"Generated B folder is not empty ({existing_b} images): {generated_b_dir}\n"
                    "Use an empty folder or pass --allow-nonempty-generated-dir to proceed."
                )

        gen_a_info = _generate_images_from_checkpoint(
            checkpoint_path=args.checkpoint,
            input_dir=args.input_b_dir,
            generated_dir=generated_a_dir,
            model_type=args.model_type,
            ngf=args.ngf,
            n_blocks=args.n_blocks,
            image_size=args.image_size,
            batch_size=args.batch_size,
            max_images=args.max_images,
            device=device,
            generator_key="g_ba",
        )
        print(
            "generated_from_checkpoint="
            f"{Path(args.checkpoint).resolve()} model={gen_a_info['model_type']} "
            f"generator=g_ba count={gen_a_info['generated_count']}"
        )

        gen_b_info = _generate_images_from_checkpoint(
            checkpoint_path=args.checkpoint,
            input_dir=args.input_a_dir,
            generated_dir=generated_b_dir,
            model_type=args.model_type,
            ngf=args.ngf,
            n_blocks=args.n_blocks,
            image_size=args.image_size,
            batch_size=args.batch_size,
            max_images=args.max_images,
            device=device,
            generator_key="g_ab",
        )
        print(
            "generated_from_checkpoint="
            f"{Path(args.checkpoint).resolve()} model={gen_b_info['model_type']} "
            f"generator=g_ab count={gen_b_info['generated_count']}"
        )

        b2a_metrics = _compute_metrics_for_dirs(generated_a_dir, args.real_a_dir, use_cuda=use_cuda)
        a2b_metrics = _compute_metrics_for_dirs(generated_b_dir, args.real_b_dir, use_cuda=use_cuda)

        print("B->A metrics")
        print(f"generated_dir={b2a_metrics['generated_dir']} images={b2a_metrics['generated_images']}")
        print(f"real_dir={b2a_metrics['real_dir']} images={b2a_metrics['real_images']}")
        print(f"FID={b2a_metrics['fid']:.6f}")
        print(f"IS_mean={b2a_metrics['is_mean']:.6f} IS_std={b2a_metrics['is_std']:.6f}")
        if b2a_metrics["gms"] is None:
            print("GMS=undefined (IS_mean<=0)")
        else:
            print(f"GMS={b2a_metrics['gms']:.6f}")

        print("A->B metrics")
        print(f"generated_dir={a2b_metrics['generated_dir']} images={a2b_metrics['generated_images']}")
        print(f"real_dir={a2b_metrics['real_dir']} images={a2b_metrics['real_images']}")
        print(f"FID={a2b_metrics['fid']:.6f}")
        print(f"IS_mean={a2b_metrics['is_mean']:.6f} IS_std={a2b_metrics['is_std']:.6f}")
        if a2b_metrics["gms"] is None:
            print("GMS=undefined (IS_mean<=0)")
        else:
            print(f"GMS={a2b_metrics['gms']:.6f}")

        gms_values = [v for v in [b2a_metrics["gms"], a2b_metrics["gms"]] if v is not None]
        gms_avg = round(sum(gms_values) / len(gms_values), 5) if gms_values else None
        if gms_avg is None:
            print("GMS_avg=undefined (both directions IS_mean<=0)")
        else:
            print(f"GMS_avg={gms_avg:.5f}")

        payload = {
            "config_used": gen_a_info.get("config_used"),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "b2a": b2a_metrics,
            "a2b": a2b_metrics,
            "gms_avg": gms_avg,
        }
    else:
        if args.generated_dir is None or args.real_dir is None:
            raise ValueError("--generated-dir and --real-dir are required in single-direction mode")

        gen_dir = Path(args.generated_dir).resolve()
        real_dir = Path(args.real_dir).resolve()

        generation_info = None
        if args.checkpoint is not None:
            if args.input_b_dir is None:
                raise ValueError("--input-b-dir is required when --checkpoint is provided")
            existing = _count_images(gen_dir)
            if existing > 0 and not args.allow_nonempty_generated_dir:
                raise ValueError(
                    f"Generated folder is not empty ({existing} images): {gen_dir}\n"
                    "Use an empty folder or pass --allow-nonempty-generated-dir to proceed."
                )
            generation_info = _generate_images_from_checkpoint(
                checkpoint_path=args.checkpoint,
                input_dir=args.input_b_dir,
                generated_dir=gen_dir,
                model_type=args.model_type,
                ngf=args.ngf,
                n_blocks=args.n_blocks,
                image_size=args.image_size,
                batch_size=args.batch_size,
                max_images=args.max_images,
                device=device,
                generator_key="g_ba",
            )
            print(
                "generated_from_checkpoint="
                f"{Path(args.checkpoint).resolve()} model={generation_info['model_type']} "
                f"count={generation_info['generated_count']}"
            )

        metrics_single = _compute_metrics_for_dirs(gen_dir, real_dir, use_cuda=use_cuda)

        print(f"generated_dir={metrics_single['generated_dir']} images={metrics_single['generated_images']}")
        print(f"real_dir={metrics_single['real_dir']} images={metrics_single['real_images']}")
        print(f"FID={metrics_single['fid']:.6f}")
        print(f"IS_mean={metrics_single['is_mean']:.6f} IS_std={metrics_single['is_std']:.6f}")
        if metrics_single["gms"] is None:
            print("GMS=undefined (IS_mean<=0)")
        else:
            print(f"GMS={metrics_single['gms']:.6f}")

        payload = {
            "config_used": generation_info.get("config_used") if generation_info is not None else None,
            "generated_dir": metrics_single["generated_dir"],
            "real_dir": metrics_single["real_dir"],
            "generated_images": metrics_single["generated_images"],
            "real_images": metrics_single["real_images"],
            "fid": metrics_single["fid"],
            "is": metrics_single["is_mean"],
            "is_std": metrics_single["is_std"],
            "gms": metrics_single["gms"],
        }

    if args.out_json:
        out_json = Path(args.out_json)
        # Allow passing either a file path or a directory path.
        if out_json.exists() and out_json.is_dir():
            out_json = out_json / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        elif out_json.suffix.lower() != ".json":
            out_json = out_json / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            # Enforce eval_<datetime>.json naming even when a filename is supplied.
            out_json = out_json.parent / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"saved_json={out_json}")

    if args.out_csv:
        out_csv = Path(args.out_csv)
        # Allow passing either a file path or a directory path.
        if out_csv.exists() and out_csv.is_dir():
            out_csv = out_csv / "metrics.csv"
        elif out_csv.suffix.lower() != ".csv":
            out_csv = out_csv / "metrics.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        if args.bidirectional:
            fieldnames = [
                "checkpoint",
                "gms_avg",
                "b2a_generated_dir",
                "b2a_real_dir",
                "b2a_generated_images",
                "b2a_real_images",
                "b2a_fid",
                "b2a_is_mean",
                "b2a_is_std",
                "b2a_gms",
                "a2b_generated_dir",
                "a2b_real_dir",
                "a2b_generated_images",
                "a2b_real_images",
                "a2b_fid",
                "a2b_is_mean",
                "a2b_is_std",
                "a2b_gms",
            ]
            row = {
                "checkpoint": payload.get("checkpoint"),
                "gms_avg": payload.get("gms_avg"),
                "b2a_generated_dir": payload["b2a"]["generated_dir"],
                "b2a_real_dir": payload["b2a"]["real_dir"],
                "b2a_generated_images": payload["b2a"]["generated_images"],
                "b2a_real_images": payload["b2a"]["real_images"],
                "b2a_fid": payload["b2a"]["fid"],
                "b2a_is_mean": payload["b2a"]["is_mean"],
                "b2a_is_std": payload["b2a"]["is_std"],
                "b2a_gms": payload["b2a"]["gms"],
                "a2b_generated_dir": payload["a2b"]["generated_dir"],
                "a2b_real_dir": payload["a2b"]["real_dir"],
                "a2b_generated_images": payload["a2b"]["generated_images"],
                "a2b_real_images": payload["a2b"]["real_images"],
                "a2b_fid": payload["a2b"]["fid"],
                "a2b_is_mean": payload["a2b"]["is_mean"],
                "a2b_is_std": payload["a2b"]["is_std"],
                "a2b_gms": payload["a2b"]["gms"],
            }
        else:
            fieldnames = [
                "generated_dir",
                "real_dir",
                "generated_images",
                "real_images",
                "fid",
                "is",
                "is_std",
                "gms",
            ]
            row = payload
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerow(row)
        print(f"saved_csv={out_csv}")

    if args.submission_csv:
        out_submission = Path(args.submission_csv)
        if out_submission.suffix.lower() != ".csv":
            out_submission = out_submission / "Userid.csv"
        out_submission.parent.mkdir(parents=True, exist_ok=True)
        label = payload.get("gms_avg") if args.bidirectional else payload.get("gms")
        with out_submission.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "label"])
            writer.writeheader()
            writer.writerow({"id": 1, "label": label})
        print(f"saved_submission_csv={out_submission}")


if __name__ == "__main__":
    main()
