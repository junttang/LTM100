"""Plot per-bin request counts after the fixed-delay example finishes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def plot_counts(paths: list[Path], output: Path) -> None:
    # Plotting is an example-only dependency and never runs in the load loop.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not paths:
        raise ValueError("no request-counts.csv files found")
    figure, axes = plt.subplots(
        len(paths), 1, figsize=(12, 3 * len(paths)), squeeze=False, layout="constrained"
    )
    for axis, path in zip(axes[:, 0], paths):
        with path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        if not rows:
            raise ValueError(f"empty request counts: {path}")
        starts = [float(row["elapsed_start_s"]) / 60 for row in rows]
        widths = [float(row["interval_seconds"]) / 60 for row in rows]
        adds = [int(row["add_started"]) for row in rows]
        searches = [int(row["search_started"]) for row in rows]
        axis.bar(starts, adds, width=widths, align="edge", color="#2563eb", label="Add")
        axis.bar(
            starts,
            searches,
            width=widths,
            bottom=adds,
            align="edge",
            color="#ea580c",
            label="Search",
        )
        axis.set_title(path.parent.name.replace("users-", "Users: "), loc="left")
        bin_seconds = float(rows[0]["interval_seconds"])
        axis.set_ylabel(f"Requests / {bin_seconds:g} s bin")
        axis.set_xlim(0, starts[-1] + widths[-1])
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.legend(loc="upper left", ncols=2)
        axis.set_xlabel("Elapsed time (minutes)")
    subtitle = "Request counts at fixed-delay backend entry"
    summary_path = paths[0].with_name("summary.json")
    if summary_path.exists():
        config = json.loads(summary_path.read_text(encoding="utf-8"))["meta"]["config"]
        subtitle = (
            f"Search {config['search_latency'] * 1000:g} ms · "
            f"add {config['add_latency'] * 1000:g} ms · "
            f"one user starts every {config['user_start_interval']:g} s\n"
            f"Answer {config['answer_time']:g} ± "
            f"{config['answer_time_variation_seconds']:g} s · "
            f"user gap {config['user_gap']:g} ± "
            f"{config['user_gap_variation_seconds']:g} s"
        )
    figure.suptitle(
        f"Chat-replay dispatch pattern · fixed-delay backend\n{subtitle}", fontsize=14
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, metadata={"Date": None} if output.suffix == ".svg" else None)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(
        args.input.glob("users-*/request-counts.csv"),
        key=lambda path: int(path.parent.name.removeprefix("users-")),
    )
    plot_counts(paths, args.output)


if __name__ == "__main__":
    main()
