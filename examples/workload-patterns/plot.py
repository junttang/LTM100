"""Plot dispatch counts, concurrency, and bursts after the example finishes."""

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
    from matplotlib.ticker import MaxNLocator

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
        peak = max(add + search for add, search in zip(adds, searches))
        axis.set_ylim(0, peak * 1.3 if peak else 1)
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
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
    parser.add_argument(
        "--kind", choices=("requests", "inflight", "bursts"), default="requests"
    )
    args = parser.parse_args()
    filename = {
        "requests": "request-counts.csv",
        "inflight": "inflight.csv",
        "bursts": "burst-histogram.csv",
    }[args.kind]
    paths = sorted(
        args.input.glob(f"users-*/{filename}"),
        key=lambda path: int(path.parent.name.removeprefix("users-")),
    )
    if args.kind == "requests":
        plot_counts(paths, args.output)
    elif args.kind == "inflight":
        plot_inflight(paths, args.output)
    else:
        plot_bursts(paths, args.output)


def plot_inflight(paths: list[Path], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    if not paths:
        raise ValueError("no inflight.csv files found")
    figure, axes = plt.subplots(
        len(paths), 1, figsize=(12, 3 * len(paths)), squeeze=False, layout="constrained"
    )
    for axis, path in zip(axes[:, 0], paths):
        with path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        times = [float(row["elapsed_start_s"]) / 60 for row in rows]
        times.append(times[-1] + float(rows[-1]["interval_seconds"]) / 60)
        adds = [float(row["add_mean"]) for row in rows]
        searches = [float(row["search_mean"]) for row in rows]
        peaks = [int(row["total_peak"]) for row in rows]
        axis.stackplot(
            times,
            adds + [adds[-1]],
            searches + [searches[-1]],
            step="post",
            colors=["#2563eb", "#ea580c"],
            labels=["Add (mean)", "Search (mean)"],
        )
        axis.step(
            times,
            peaks + [peaks[-1]],
            where="post",
            color="#334155",
            linewidth=0.9,
            label="Total (peak)",
        )
        axis.set_title(path.parent.name.replace("users-", "Users: "), loc="left")
        axis.set_ylabel("Requests in flight")
        axis.set_xlabel("Elapsed time (minutes)")
        axis.set_xlim(0, times[-1])
        axis.set_ylim(0, max(peaks) * 1.3 if max(peaks) else 1)
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.legend(loc="upper left", ncols=3)
    interval = float(rows[0]["interval_seconds"])
    figure.suptitle(
        "In-flight requests · fixed-delay backend\n"
        f"{interval:g} s bins: time-weighted means and exact simultaneous peaks",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, metadata={"Date": None} if output.suffix == ".svg" else None)
    plt.close(figure)


def plot_bursts(paths: list[Path], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    if not paths:
        raise ValueError("no burst-histogram.csv files found")
    figure, axes = plt.subplots(
        len(paths),
        1,
        figsize=(10, 2.7 * len(paths)),
        squeeze=False,
        layout="constrained",
    )
    for axis, path in zip(axes[:, 0], paths):
        payload = json.loads(
            path.with_name("pattern-statistics.json").read_text(encoding="utf-8")
        )
        with path.open(newline="", encoding="utf-8") as file:
            rows = [row for row in csv.DictReader(file) if row["op_type"] == "all"]
        counts = [int(row["requests_per_bin"]) for row in rows]
        probabilities = [float(row["fraction"]) for row in rows]
        bars = axis.bar(counts, probabilities, width=0.65, color="#334155")
        axis.bar_label(
            bars,
            labels=[f"{value:.2%}" for value in probabilities],
            padding=3,
            fontsize=9,
        )
        stats = payload["stats"]["all"]
        axis.set_title(
            f"{path.parent.name.replace('users-', 'Users: ')} · "
            f"empty {stats['empty_bin_fraction']:.2%} · "
            f"p95 {stats['p95_requests_per_bin']:g} · "
            f"p99 {stats['p99_requests_per_bin']:g} · "
            f"max {stats['max_requests_per_bin']:g}",
            loc="left",
        )
        axis.set_ylabel("Fraction of bins")
        axis.set_xlabel("Total requests starting in one bin (add + search)")
        axis.set_ylim(0, 1.1)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set_xlim(-0.6, max(counts) + 0.6)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    figure.suptitle(
        "Short-bin request distribution · fixed-delay backend\n"
        f"{payload['window_start_s'] / 60:g}–{payload['window_end_s'] / 60:g} min · "
        f"{payload['burst_interval_seconds'] * 1000:g} ms bins · empty bins included",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, metadata={"Date": None} if output.suffix == ".svg" else None)
    plt.close(figure)


if __name__ == "__main__":
    main()
