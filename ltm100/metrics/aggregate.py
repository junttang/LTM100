"""Post-run aggregation of per-request results into a summary.

Computes count, throughput (ops/s), QPS, and latency percentiles (p50/p90/
p95/p99/max), broken down by op type, plus an overall total throughput/QPS and
error rate. Per-op latency percentiles are kept separate from the top-level
summary because mixing add/search latencies into one distribution is ambiguous;
the overall view reports throughput only. Pure function over a list of OpResult.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

from ltm100.core.op import OpResult


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    ordered = sorted(values)
    n = len(ordered)

    def pct(p: float) -> float:
        if n == 1:
            return ordered[0]
        # Nearest-rank method.
        rank = math.ceil(p / 100 * n)
        rank = max(1, min(rank, n))
        return ordered[rank - 1]

    return {
        "p50": pct(50),
        "p90": pct(90),
        "p95": pct(95),
        "p99": pct(99),
        "max": ordered[-1],
    }


def aggregate(results: Iterable[OpResult]) -> dict:
    results = list(results)
    if not results:
        return {
            "total": 0,
            "throughput_ops_s": 0.0,
            "qps": 0.0,
            "by_op": {},
            "error_rate": 0.0,
            "wall_seconds": 0.0,
        }

    by_op: dict[str, list[OpResult]] = defaultdict(list)
    for r in results:
        by_op[r.type.value].append(r)

    start = min(r.started_at for r in results)
    end = max(r.ended_at for r in results)
    wall = max(end - start, 0.0)

    summary_by_op: dict[str, dict] = {}
    total = 0
    total_errors = 0
    for op_type, items in by_op.items():
        latencies_ms = [(r.ended_at - r.started_at) * 1000.0 for r in items]
        errors = sum(1 for r in items if r.status != "ok")
        total += len(items)
        total_errors += errors
        summary_by_op[op_type] = {
            "count": len(items),
            "throughput_ops_s": len(items) / wall if wall > 0 else 0.0,
            "qps": len(items) / wall if wall > 0 else 0.0,
            "latency_ms": {
                "mean": sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0,
                **_percentiles(latencies_ms),
            },
            "errors": errors,
            "error_rate": errors / len(items) if items else 0.0,
        }

    overall_throughput = total / wall if wall > 0 else 0.0

    return {
        "total": total,
        "throughput_ops_s": overall_throughput,
        "qps": overall_throughput,
        "by_op": summary_by_op,
        "error_rate": total_errors / total if total else 0.0,
        "wall_seconds": round(wall, 6),
    }


__all__ = ["aggregate"]
