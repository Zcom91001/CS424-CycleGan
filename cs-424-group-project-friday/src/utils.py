import csv
import json
import math
import os
import random
import struct
import zlib
from datetime import datetime
from pathlib import Path


def _parse_scalar(value: str):
    v = value.strip()
    if not v:
        return ""
    lower = v.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"null", "none"}:
        return None
    try:
        if "." in v:
            return float(v)
        return int(v)
    except ValueError:
        pass
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    return v


def load_yaml(path):
    """Minimal YAML reader for simple key/value and nested dictionaries."""
    text = Path(path).read_text(encoding="utf-8")
    root = {}
    stack = [(-1, root)]

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        if ":" not in content:
            continue
        key, value = content.split(":", 1)
        key = key.strip()
        value = value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]

        if value == "":
            node = {}
            current[key] = node
            stack.append((indent, node))
        else:
            current[key] = _parse_scalar(value)

    return root


def _dump_yaml_lines(data, indent=0):
    lines = []
    pad = " " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{pad}{key}:")
            lines.extend(_dump_yaml_lines(value, indent + 2))
        else:
            if isinstance(value, bool):
                rendered = "true" if value else "false"
            elif value is None:
                rendered = "null"
            else:
                rendered = str(value)
            lines.append(f"{pad}{key}: {rendered}")
    return lines


def save_yaml(data, path):
    lines = _dump_yaml_lines(data)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_run_dirs(root, run_name):
    run_dir = Path(root) / "outputs" / "runs" / run_name
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


def append_metrics(metrics_path, row):
    metrics_path = Path(metrics_path)
    exists = metrics_path.exists()
    with metrics_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_metrics(metrics_path):
    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        return []
    with metrics_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_json(data, path):
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def set_seed(seed):
    random.seed(int(seed))


def _png_chunk(chunk_type, data):
    chunk = chunk_type + data
    crc = zlib.crc32(chunk) & 0xFFFFFFFF
    return struct.pack("!I", len(data)) + chunk + struct.pack("!I", crc)


def write_png(path, width, height, pixels):
    """Write an RGB PNG from a 2D list of (r, g, b) tuples."""
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)

    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter type None
        row = pixels[y]
        for x in range(width):
            r, g, b = row[x]
            raw.extend((r & 255, g & 255, b & 255))

    idat = zlib.compress(bytes(raw), level=9)
    png_bytes = signature + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")
    Path(path).write_bytes(png_bytes)


def _draw_rect(canvas, x0, y0, x1, y1, color):
    h = len(canvas)
    w = len(canvas[0]) if h else 0
    for y in range(max(0, y0), min(h, y1 + 1)):
        for x in range(max(0, x0), min(w, x1 + 1)):
            canvas[y][x] = color


def _draw_line(canvas, x0, y0, x1, y1, color):
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    while True:
        if 0 <= y0 < len(canvas) and 0 <= x0 < len(canvas[0]):
            canvas[y0][x0] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def save_loss_plot(metrics_rows, out_path):
    width, height = 900, 500
    bg = (245, 246, 250)
    canvas = [[bg for _ in range(width)] for _ in range(height)]

    margin_left = 60
    margin_right = 20
    margin_top = 20
    margin_bottom = 50
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    _draw_rect(canvas, margin_left, margin_top, width - margin_right, height - margin_bottom, (255, 255, 255))

    if not metrics_rows:
        write_png(out_path, width, height, canvas)
        return

    fields = [
        ("g_total", (231, 76, 60)),
        ("d_total", (52, 152, 219)),
        ("cycle", (46, 204, 113)),
    ]

    values = []
    for row in metrics_rows:
        for name, _ in fields:
            values.append(float(row.get(name, 0.0)))

    v_min = min(values)
    v_max = max(values)
    if math.isclose(v_min, v_max):
        v_max = v_min + 1.0

    n = len(metrics_rows)

    def xy(epoch_idx, value):
        if n == 1:
            x = margin_left + plot_w // 2
        else:
            x = margin_left + int(epoch_idx * plot_w / (n - 1))
        norm = (value - v_min) / (v_max - v_min)
        y = margin_top + int((1.0 - norm) * plot_h)
        return x, y

    _draw_line(canvas, margin_left, margin_top, margin_left, height - margin_bottom, (180, 180, 180))
    _draw_line(canvas, margin_left, height - margin_bottom, width - margin_right, height - margin_bottom, (180, 180, 180))

    for name, color in fields:
        prev = None
        for i, row in enumerate(metrics_rows):
            point = xy(i, float(row.get(name, 0.0)))
            if prev is not None:
                _draw_line(canvas, prev[0], prev[1], point[0], point[1], color)
            prev = point

    write_png(out_path, width, height, canvas)


def save_sample_grid(values_a, values_b, out_path, cols=4, cell=48):
    values = list(values_a) + list(values_b)
    rows = (len(values) + cols - 1) // cols
    width = cols * cell
    height = rows * cell
    canvas = [[(250, 250, 250) for _ in range(width)] for _ in range(height)]

    for idx, value in enumerate(values):
        row = idx // cols
        col = idx % cols
        x0 = col * cell
        y0 = row * cell
        tone = max(0, min(255, int(value * 255)))
        color = (tone, 120, 255 - tone)
        for y in range(y0 + 2, y0 + cell - 2):
            for x in range(x0 + 2, x0 + cell - 2):
                canvas[y][x] = color

    write_png(out_path, width, height, canvas)


def upsert_experiment_tracker(tracker_path, row):
    tracker_path = Path(tracker_path)
    headers = [
        "run_name",
        "config_name",
        "change",
        "kept_same",
        "official_score",
        "visual_quality_notes",
        "keep_change",
    ]

    data = []
    if tracker_path.exists():
        with tracker_path.open("r", newline="", encoding="utf-8") as f:
            data = list(csv.DictReader(f))

    replaced = False
    for i, old in enumerate(data):
        if old.get("run_name") == row.get("run_name"):
            data[i] = row
            replaced = True
            break
    if not replaced:
        data.append(row)

    with tracker_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for item in data:
            writer.writerow({h: item.get(h, "") for h in headers})
