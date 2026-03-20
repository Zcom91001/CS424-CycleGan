import argparse
from pathlib import Path

from utils import read_metrics, save_loss_plot


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from metrics.csv")
    parser.add_argument("--metrics", required=True, help="Path to metrics.csv")
    parser.add_argument("--out", required=True, help="Output PNG path")
    args = parser.parse_args()

    rows = read_metrics(args.metrics)
    save_loss_plot(rows, args.out)
    print(f"Saved plot to {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
