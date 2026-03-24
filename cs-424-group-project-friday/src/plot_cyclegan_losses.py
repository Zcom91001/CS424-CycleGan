import argparse
from pathlib import Path

from cyclegan_io import read_metrics, save_loss_plot

# How to run (from repo root):
#   python src/plot_cyclegan_losses.py --run-name cyclegan_quick
# This reads:
#   outputs/runs/<run-name>/metrics.csv
# and writes:
#   outputs/runs/<run-name>/plots/losses.png


def main():
    parser = argparse.ArgumentParser(description="Plot CycleGAN losses from metrics CSV.")
    parser.add_argument("--run-name", required=True, help="Run folder under outputs/runs")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    run_dir = repo_root / "outputs" / "runs" / args.run_name
    metrics_csv = run_dir / "metrics.csv"
    out_png = run_dir / "plots" / "losses.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)

    if not metrics_csv.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {metrics_csv}")
    rows = read_metrics(metrics_csv)
    if not rows:
        raise RuntimeError(f"No valid metric rows found in: {metrics_csv}")

    save_loss_plot(metrics_csv, out_png)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
