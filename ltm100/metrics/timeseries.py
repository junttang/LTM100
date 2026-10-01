"""Fixed-interval client-observed E2E performance aggregation."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from ltm100.core.op import OpResult
from ltm100.metrics.aggregate import _percentiles


def aggregate_timeseries(
    results: Iterable[OpResult],
    *,
    interval_seconds: float,
    measurement_started_at: float,
    measurement_ended_at: float,
) -> list[dict[str, Any]]:
    """Aggregate request starts and completions into fixed wall-clock buckets.

    Successful request latency is attributed to the bucket in which the
    response completed. Requests rejected by the client-side bounded queue
    were never dispatched to the backend, so they contribute only to the
    rejected count. Empty buckets are retained to make stalls visible.
    """
    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("interval_seconds must be a finite number > 0")
    if not math.isfinite(measurement_started_at) or not math.isfinite(
        measurement_ended_at
    ):
        raise ValueError("measurement boundaries must be finite")
    if measurement_ended_at < measurement_started_at:
        raise ValueError("measurement_ended_at must be >= measurement_started_at")

    results = list(results)
    duration = measurement_ended_at - measurement_started_at
    bucket_count = max(1, math.ceil(duration / interval_seconds))
    op_types = sorted({result.type.value for result in results})

    def new_bucket() -> dict[str, Any]:
        return {
            "started": 0,
            "completed": 0,
            "successful": 0,
            "errors": 0,
            "rejected": 0,
            "latencies_ms": [],
            "n_items": [],
            "empty": 0,
        }

    buckets = {
        op_type: [new_bucket() for _ in range(bucket_count)] for op_type in op_types
    }

    def bucket_index(timestamp: float) -> int:
        offset = max(timestamp - measurement_started_at, 0.0)
        return min(int(offset / interval_seconds), bucket_count - 1)

    for result in results:
        state = buckets[result.type.value]
        if result.status == "rejected":
            state[bucket_index(result.started_at)]["rejected"] += 1
            continue

        state[bucket_index(result.started_at)]["started"] += 1
        completed = state[bucket_index(result.ended_at)]
        completed["completed"] += 1
        if result.status == "ok":
            completed["successful"] += 1
            completed["latencies_ms"].append(
                (result.ended_at - result.started_at) * 1000.0
            )
            completed["n_items"].append(result.n_items)
            if result.n_items == 0:
                completed["empty"] += 1
        else:
            completed["errors"] += 1

    rows: list[dict[str, Any]] = []
    in_flight = dict.fromkeys(op_types, 0)
    for index in range(bucket_count):
        elapsed_start = index * interval_seconds
        elapsed_end = min((index + 1) * interval_seconds, duration)
        width = elapsed_end - elapsed_start
        bucket_started_at = measurement_started_at + elapsed_start
        bucket_ended_at = measurement_started_at + elapsed_end

        per_op_rows: list[dict[str, Any]] = []
        for op_type in op_types:
            bucket = buckets[op_type][index]
            in_flight[op_type] += bucket["started"] - bucket["completed"]
            latencies = bucket["latencies_ms"]
            items = bucket["n_items"]
            latency = _percentiles(latencies)
            row = {
                "bucket_index": index,
                "elapsed_start_s": elapsed_start,
                "elapsed_end_s": elapsed_end,
                "interval_seconds": width,
                "bucket_started_at": bucket_started_at,
                "bucket_ended_at": bucket_ended_at,
                "op_type": op_type,
                "started": bucket["started"],
                "completed": bucket["completed"],
                "successful": bucket["successful"],
                "errors": bucket["errors"],
                "rejected": bucket["rejected"],
                "successful_ops_s": bucket["successful"] / width if width else 0.0,
                "in_flight_end": in_flight[op_type],
                "latency_samples": len(latencies),
                "latency_mean_ms": (
                    sum(latencies) / len(latencies) if latencies else None
                ),
                "latency_p50_ms": latency["p50"] if latencies else None,
                "latency_p95_ms": latency["p95"] if latencies else None,
                "latency_p99_ms": latency["p99"] if latencies else None,
                "latency_max_ms": latency["max"] if latencies else None,
                "n_items_mean": sum(items) / len(items) if items else None,
                "empty_rate": (
                    bucket["empty"] / len(items)
                    if op_type == "search" and items
                    else None
                ),
            }
            per_op_rows.append(row)
            rows.append(row)

        rows.append(
            {
                "bucket_index": index,
                "elapsed_start_s": elapsed_start,
                "elapsed_end_s": elapsed_end,
                "interval_seconds": width,
                "bucket_started_at": bucket_started_at,
                "bucket_ended_at": bucket_ended_at,
                "op_type": "all",
                "started": sum(row["started"] for row in per_op_rows),
                "completed": sum(row["completed"] for row in per_op_rows),
                "successful": sum(row["successful"] for row in per_op_rows),
                "errors": sum(row["errors"] for row in per_op_rows),
                "rejected": sum(row["rejected"] for row in per_op_rows),
                "successful_ops_s": (
                    sum(row["successful"] for row in per_op_rows) / width
                    if width
                    else 0.0
                ),
                "in_flight_end": sum(in_flight.values()),
                "latency_samples": "",
                "latency_mean_ms": None,
                "latency_p50_ms": None,
                "latency_p95_ms": None,
                "latency_p99_ms": None,
                "latency_max_ms": None,
                "n_items_mean": None,
                "empty_rate": None,
            }
        )
    return rows


__all__ = ["aggregate_timeseries"]
