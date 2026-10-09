"""Reconstruct concurrency and short-bin request distributions from raw records."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from ltm100.metrics.aggregate import _percentiles


def _time(value: float) -> Fraction:
    """Use the recorded decimal value without introducing binary rounding.

    Convert absolute timestamps before subtracting the window origin. Exact
    arithmetic keeps decimal boundaries and adjacent representable floats
    distinct, including large epoch timestamps. Output values remain floats.
    """
    return Fraction(str(value))


def _bucket_index(offset: Fraction, interval: Fraction) -> int:
    """Assign an offset to its half-open bin without rounding or tolerance."""
    return offset // interval


def inflight_rows(
    records: list[dict], *, start: float, end: float, interval: float
) -> list[dict]:
    """Integrate exact half-open request lifetimes into mean/peak bin counts.

    Events at the same timestamp are applied together. A completed request
    and a new request at that timestamp do not overlap, and zero-duration
    requests contribute no residence time or concurrency peak.
    """
    if not all(math.isfinite(value) for value in (start, end, interval)):
        raise ValueError("boundaries and interval must be finite")
    if end <= start or interval <= 0:
        raise ValueError("end must exceed start and interval must be > 0")
    origin, limit, width = _time(start), _time(end), _time(interval)
    duration = limit - origin
    rows = [
        {
            "elapsed_start_s": float(index * width),
            "interval_seconds": float(min(width, duration - index * width)),
            "add_mean": Fraction(0),
            "search_mean": Fraction(0),
            "total_mean": 0.0,
            "add_peak": 0,
            "search_peak": 0,
            "total_peak": 0,
        }
        for index in range(math.ceil(duration / width))
    ]
    events: dict[Fraction, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        if record["status"] == "rejected":
            continue
        left = max(_time(record["started_at"]), origin) - origin
        right = min(_time(record["ended_at"]), limit) - origin
        if right <= left:
            continue
        op = record["op_type"]
        if op not in ("add", "search"):
            raise ValueError(f"unsupported operation: {op}")
        events[left][op] += 1
        events[right][op] -= 1

    counts = {"add": 0, "search": 0}

    def integrate(left: Fraction, right: Fraction) -> None:
        while left < right:
            index = _bucket_index(left, width)
            boundary = min((index + 1) * width, duration, right)
            row = rows[index]
            for op, count in counts.items():
                row[f"{op}_mean"] += count * (boundary - left)
                row[f"{op}_peak"] = max(row[f"{op}_peak"], count)
            row["total_peak"] = max(row["total_peak"], sum(counts.values()))
            left = boundary

    previous = Fraction(0)
    for timestamp, changes in sorted(events.items()):
        integrate(previous, timestamp)
        for op, delta in changes.items():
            counts[op] += delta
        previous = timestamp
    integrate(previous, duration)
    for index, row in enumerate(rows):
        exposure = min(width, duration - index * width)
        for op in counts:
            row[f"{op}_mean"] = float(row[f"{op}_mean"] / exposure)
        row["total_mean"] = row["add_mean"] + row["search_mean"]
    return rows


def burst_distribution(
    records: list[dict], *, start: float, end: float, interval: float
) -> tuple[list[dict], dict]:
    """Count starts in [start, end), including every empty full-width bin."""
    if not all(math.isfinite(value) for value in (start, end, interval)):
        raise ValueError("boundaries and interval must be finite")
    if end <= start or interval <= 0:
        raise ValueError("end must exceed start and interval must be > 0")
    origin, limit, width = _time(start), _time(end), _time(interval)
    ratio = (limit - origin) / width
    if not math.isclose(ratio, round(ratio), rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError("window width must be a whole number of burst intervals")
    size = round(ratio)
    if size < 1:
        raise ValueError("window must contain at least one burst interval")
    bins = {op: [0] * size for op in ("add", "search", "all")}
    for record in records:
        if record["status"] == "rejected" or not start <= record["started_at"] < end:
            continue
        op = record["op_type"]
        if op not in ("add", "search"):
            raise ValueError(f"unsupported operation: {op}")
        index = min(
            _bucket_index(_time(record["started_at"]) - origin, width), size - 1
        )
        bins[op][index] += 1
        bins["all"][index] += 1
    rows = []
    stats = {}
    for op, values in bins.items():
        histogram = Counter(values)
        cumulative = 0
        for count in range(max(values) + 1):
            frequency = histogram[count]
            cumulative += frequency
            rows.append(
                {
                    "op_type": op,
                    "requests_per_bin": count,
                    "bins": frequency,
                    "fraction": frequency / size,
                    "cumulative_fraction": cumulative / size,
                }
            )
        percentiles = _percentiles(values)
        stats[op] = {
            "requests": sum(values),
            "requests_s": sum(values) / (end - start),
            "mean_requests_per_bin": sum(values) / size,
            "p95_requests_per_bin": percentiles["p95"],
            "p99_requests_per_bin": percentiles["p99"],
            "max_requests_per_bin": percentiles["max"],
            "empty_bin_fraction": histogram[0] / size,
        }
    return rows, stats


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def analyze_case(
    source: Path,
    output: Path,
    *,
    inflight_interval: float,
    burst_interval: float,
    window_start: float,
    window_end: float,
) -> None:
    meta = json.loads((source / "summary.json").read_text(encoding="utf-8"))["meta"]
    start = meta["measurement_started_at"]
    end = meta["measurement_ended_at"]
    if not 0 <= window_start < window_end <= end - start:
        raise ValueError("comparison window must lie inside the measured run")
    records = [
        json.loads(line)
        for line in (source / "raw.ndjson").read_text(encoding="utf-8").splitlines()
    ]
    inflight = inflight_rows(records, start=start, end=end, interval=inflight_interval)
    distribution, stats = burst_distribution(
        records,
        start=start + window_start,
        end=start + window_end,
        interval=burst_interval,
    )
    window_concurrency = inflight_rows(
        records,
        start=start + window_start,
        end=start + window_end,
        interval=window_end - window_start,
    )[0]
    for op in stats:
        prefix = "total" if op == "all" else op
        stats[op]["mean_in_flight"] = window_concurrency[f"{prefix}_mean"]
        stats[op]["peak_in_flight"] = window_concurrency[f"{prefix}_peak"]
    output.mkdir(parents=True, exist_ok=True)
    write_csv(inflight, output / "inflight.csv")
    write_csv(distribution, output / "burst-histogram.csv")
    payload = {
        "inflight_interval_seconds": inflight_interval,
        "burst_interval_seconds": burst_interval,
        "window_start_s": window_start,
        "window_end_s": window_end,
        "window_bins": round((window_end - window_start) / burst_interval),
        "stats": stats,
    }
    (output / "pattern-statistics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inflight-interval", type=float, default=5)
    parser.add_argument("--burst-interval", type=float, default=0.1)
    parser.add_argument("--window-start", type=float, default=1200)
    parser.add_argument("--window-end", type=float, default=1800)
    args = parser.parse_args()
    sources = sorted(
        args.input.glob("users-*/raw.ndjson"),
        key=lambda path: int(path.parent.name.removeprefix("users-")),
    )
    if not sources:
        parser.error("no users-*/raw.ndjson files found")
    for raw_path in sources:
        analyze_case(
            raw_path.parent,
            args.output / raw_path.parent.name,
            inflight_interval=args.inflight_interval,
            burst_interval=args.burst_interval,
            window_start=args.window_start,
            window_end=args.window_end,
        )


if __name__ == "__main__":
    main()
