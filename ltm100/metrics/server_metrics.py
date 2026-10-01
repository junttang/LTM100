"""Server-side latency metrics: Prometheus scrape parsing, deltas, quantiles.

A Prometheus server exposes cumulative histograms at a text endpoint. Every
latency histogram `X` appears as three series families:

    X_count{labels}          total runs since the process started
    X_sum{labels}            total seconds since the process started
    X_bucket{le=...,labels}  runs whose duration was <= le (cumulative)

Because the values are process-lifetime cumulative, a workload's numbers are
the difference of two snapshots (before/after). A negative difference means the
server restarted mid-window; that series is reported as reset and excluded
rather than averaged into nonsense.

This module is pure stdlib and free of I/O: the CLI scrapes through the
backend adapter, everything here works on the returned text.

Quantiles from bucketed counts use the standard linear interpolation inside
the bracketing bucket:

    quantile ~= le[k-1] + (le[k]-le[k-1]) * (r - C[k-1]) / (C[k] - C[k-1])

where r = q * delta_count and C is the cumulative delta count. A target that
lands in the +Inf bucket is reported as None with a "beyond buckets" note: the
true value exists but exceeds the finest bucket the server publishes.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from itertools import pairwise
from typing import Any

# A parsed sample is keyed by (metric name, sorted label pairs). Labels stay
# strings; the bucket parser is the one that turns `le` into a float.
SampleKey = tuple[str, tuple[tuple[str, str], ...]]

# The rows the summary report shows: MemMachine's add-pipeline phases (5),
# search-pipeline phases (4), and the two request-level http paths. Names are
# what lands in the CSV `series` column; the selector matches the histogram's
# labels (le aside). http_request_duration_seconds also carries method/status,
# which we sum over: the path is the request-level story, the status split
# stays available in the raw output.
#
# This table is MemMachine's view of its own latencies, not a generic schema:
# everything else in this module (parser, delta, quantiles, reporting) is
# backend-neutral, and a second backend implementing the metrics query
# contributes its own row table in place of this one (see DESIGN.md §7.3).
REPORT_SERIES: list[tuple[str, str, dict[str, str]]] = [
    *[
        (
            f'encode_phase{{phase="{phase}"}}',
            "event_memory_encode_events_phase_seconds",
            {"phase": phase},
        )
        for phase in (
            "segmentation",
            "derivation",
            "embedding",
            "segment_store",
            "vector_store",
        )
    ],
    *[
        (
            f'query_phase{{phase="{phase}"}}',
            "event_memory_query_phase_seconds",
            {"phase": phase},
        )
        for phase in ("embedding", "vector_query", "segment_query", "scoring")
    ],
    (
        'http_request{{path suffix "/memories"}}',
        "http_request_duration_seconds",
        {"path": "*/memories"},
    ),
    (
        'http_request{{path suffix "/memories/search"}}',
        "http_request_duration_seconds",
        {"path": "*/memories/search"},
    ),
]

_QUANTILES = (0.5, 0.9, 0.99)


# -- exposition parsing ------------------------------------------------------


def _split_labels(blob: str) -> dict[str, str]:
    """Split a label list like `a="1",b="x, y"` on commas outside quotes."""
    out: dict[str, str] = {}
    parts: list[str] = []
    current: list[str] = []
    in_quote = False
    escaped = False
    for ch in blob:
        if escaped:
            current.append(ch)
            escaped = False
        elif in_quote and ch == "\\":
            current.append(ch)
            escaped = True
        elif ch == '"':
            in_quote = not in_quote
            current.append(ch)
        elif ch == "," and not in_quote:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    for part in parts:
        if not part.strip():
            continue
        key, _, value = part.partition("=")
        value = value.strip()
        if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        # Undo the three escapes the exposition format defines.
        value = value.replace('\\"', '"').replace("\\\\", "\\").replace("\\n", "\n")
        out[key.strip()] = value
    return out


def _find_labels_end(line: str) -> int:
    """Index of the `}` closing the label set, or -1 when the line has none."""
    if "{" not in line:
        return -1
    in_quote = False
    escaped = False
    for i, ch in enumerate(line):
        if escaped:
            escaped = False
        elif in_quote and ch == "\\":
            escaped = True
        elif ch == '"':
            in_quote = not in_quote
        elif ch == "}" and not in_quote:
            return i
    return -1


def parse_prometheus(text: str) -> dict[SampleKey, float]:
    """Parse exposition text into {(name, labels): value}.

    TYPE/HELP comments and sample timestamps are skipped; unparseable lines
    are ignored rather than raising -- a scrape that gained an exotic series
    should still yield the histograms we report on.
    """
    samples: dict[SampleKey, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        end = _find_labels_end(line)
        if end >= 0:
            name_blob, rest = line[: end + 1], line[end + 1 :]
            name, _, label_blob = name_blob.partition("{")
            labels = _split_labels(label_blob[:-1])
        else:
            name, _, rest = line.partition(" ")
            labels = {}
        fields = rest.split()
        if not fields:
            continue
        try:
            value = float(fields[0])
        except ValueError:
            continue
        if math.isnan(value):
            continue
        if "le" in labels:
            # Normalize +Inf so the two snapshots key identically; ordering
            # happens at query time by parsing back to float, not here.
            labels["le"] = "inf" if labels["le"] in ("+Inf", "Inf") else labels["le"]
        key: SampleKey = (name, tuple(sorted(labels.items())))
        samples[key] = value
    return samples


def diff(
    before: dict[SampleKey, float], after: dict[SampleKey, float]
) -> tuple[dict[SampleKey, float], set[SampleKey]]:
    """Per-series deltas plus the set of series involved in a counter reset.

    A negative delta is a restart: the cumulative counter went back to zero
    mid-window, so neither the magnitude nor the sign of the delta means
    anything. Series present on one side only (new label values mid-run, or
    vanished label values) are reported through the reset set for the same
    reason: the before/after pair cannot describe what happened in between.
    """
    deltas: dict[SampleKey, float] = {}
    resets: set[SampleKey] = set()
    for key, value in after.items():
        if key not in before:
            resets.add(key)
            continue
        delta = value - before[key]
        if delta < 0:
            resets.add(key)
            continue
        deltas[key] = delta
    for key in before:
        if key not in after:
            resets.add(key)
    return deltas, resets


# -- quantiles ----------------------------------------------------------------


def quantile(
    buckets: dict[float, float], count: float, q: float
) -> tuple[float | None, bool]:
    """Estimate quantile `q` from cumulative delta buckets.

    `buckets` maps le -> cumulative delta count. Returns (value or None,
    beyond_buckets). None+beyond_buckets means the target mass fell in the
    +Inf bucket: at least `count - C[last finite le]` samples were slower
    than the largest finite boundary, whose exact value the server does not
    publish.
    """
    if count <= 0:
        return None, False
    finite = sorted(le for le in buckets if le != math.inf)
    target = q * count
    prev_c = 0.0
    prev_le = 0.0
    for le in finite:
        c = buckets[le]
        if c >= target:
            if c == prev_c:  # no mass in this bucket's step; report its edge
                return le, False
            return prev_le + (le - prev_le) * (target - prev_c) / (c - prev_c), False
        prev_c, prev_le = c, le
    # Only reached if no finite bucket covers the target mass: the remainder
    # lives in +Inf (or the bucket set is truncated), whose value is unknown.
    return None, True


# -- report building -----------------------------------------------------------


def _labels_dict(labels: tuple[tuple[str, str], ...]) -> dict[str, str]:
    return dict(labels)


def _matches(labels: dict[str, str], selector: dict[str, str]) -> bool:
    for key, want in selector.items():
        got = labels.get(key)
        if want.startswith("*"):
            # Suffix match. What the HTTP middleware puts in `path` is a
            # deployment detail -- some builds record the router-relative
            # `/memories`, others the full `/api/v2/memories`. The suffix is
            # the part both agree on, and it is anchored at a `/` so
            # `/memories` does not also match `/memories/search`.
            if got is None or not got.endswith(want[1:]):
                return False
        elif got != want:
            return False
    return True


def histogram_delta(
    name: str,
    selector: dict[str, str],
    deltas: dict[SampleKey, float],
    resets: set[SampleKey],
) -> dict[str, Any] | None:
    """Aggregate the _count/_sum/_bucket deltas of one histogram selector.

    Extra labels beyond the selector (e.g. http status codes) are summed over:
    summing cumulative bucket counts yields the cumulative distribution of the
    combined population, which is exactly what the quantiles need.
    """

    def matches(key: SampleKey) -> str | None:
        metric, label_pairs = key
        labels = _labels_dict(label_pairs)
        if metric == f"{name}_count" and _matches(labels, selector):
            return "count"
        if metric == f"{name}_sum" and _matches(labels, selector):
            return "sum"
        if metric == f"{name}_bucket" and _matches(
            {k: v for k, v in labels.items() if k != "le"}, selector
        ):
            return "bucket"
        return None

    count = 0.0
    total = 0.0
    buckets: dict[float, float] = {}
    seen = False
    reset = False
    for key, value in deltas.items():
        kind = matches(key)
        if kind is None:
            continue
        seen = True
        if kind == "count":
            count += value
        elif kind == "sum":
            total += value
        else:
            le_raw = _labels_dict(key[1])["le"]
            try:
                le = math.inf if le_raw == "inf" else float(le_raw)
            except ValueError:
                continue
            buckets[le] = buckets.get(le, 0.0) + value
    # A reset series lives in `resets`, not `deltas` (its delta was discarded
    # as meaningless), so the reset flag needs its own scan over the same
    # selector.
    for key in resets:
        if matches(key) is not None:
            reset = True
            seen = True
    if not seen:
        return None
    return {"count": count, "sum": total, "buckets": buckets, "reset": reset}


def build_report(before_text: str, after_text: str, *, window: str) -> dict[str, Any]:
    """Turn two exposition snapshots into the summary section.

    The returned dict carries: window, status, warnings, rows (the phase/http
    table, one dict per series), and raw (every parsed series of both
    snapshots, including the secondary component metrics -- the CLI writes
    that to server_metrics_raw.json and strips it from summary.json).
    """
    before = parse_prometheus(before_text)
    after = parse_prometheus(after_text)
    deltas, resets = diff(before, after)

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for series, name, selector in REPORT_SERIES:
        row: dict[str, Any] = {
            "series": series,
            "delta_count": None,
            "delta_sum_s": None,
            "mean_s": None,
            "p50": None,
            "p90": None,
            "p99": None,
            "note": "",
        }
        hist = histogram_delta(name, selector, deltas, resets)
        if hist is None:
            row["note"] = "no data"
        elif hist["reset"]:
            row["note"] = "counter reset during run (excluded)"
            warnings.append(f"{series}: counter reset during the window")
        elif hist["count"] == 0:
            row["note"] = "not executed"
        else:
            row["delta_count"] = int(hist["count"])
            row["delta_sum_s"] = hist["sum"]
            row["mean_s"] = hist["sum"] / hist["count"]
            notes = []
            for q, col in zip(_QUANTILES, ("p50", "p90", "p99")):
                value, beyond = quantile(hist["buckets"], hist["count"], q)
                row[col] = value
                if beyond:
                    notes.append(f"{col} beyond buckets")
            if not hist["buckets"]:
                notes.append("no buckets exposed")
            row["note"] = "; ".join(notes)
        rows.append(row)

    if any(
        not row["note"].startswith("counter reset") and row["delta_count"]
        for row in rows
    ):
        warnings.append(
            "bucket quantiles are interpolated inside the server's bucket "
            "edges; phases faster than the smallest bucket have no distribution"
        )

    raw = {
        "before": _series_list(before),
        "after": _series_list(after),
    }
    return {
        "window": window,
        "status": "ok",
        "warnings": warnings,
        "rows": rows,
        "raw": raw,
    }


def _series_list(parsed: dict[SampleKey, float]) -> list[dict[str, Any]]:
    """Flatten parsed samples into a JSON-friendly list of series."""
    return [
        {"name": name, "labels": dict(labels), "value": value}
        for (name, labels), value in sorted(parsed.items())
    ]


# -- snapshot capture around the measured window -------------------------------


class SnapshotCollector:
    """Scrapes the server's metrics endpoint at the start and end of the
    measured window. Never raises: a metrics failure degrades the report, it
    does not cancel the benchmark.

    The CLI wires `start`/`end` into the runner's optional measure hooks; the
    captured text is read back with `result()` after the run.

    Each snapshot opens and closes its own connection rather than holding a
    session open across the run: with --procs 1 the hooks execute inside the
    shard's own event loop, which a session opened elsewhere could not serve,
    and holding the connection across the measured window would let the
    observer's own keep-alive sit on the server for minutes.
    """

    def __init__(
        self,
        client: Any,
        *,
        interval_seconds: float | None = None,
    ) -> None:
        if interval_seconds is not None and (
            not math.isfinite(interval_seconds) or interval_seconds <= 0
        ):
            raise ValueError("interval_seconds must be a finite number > 0")
        self._client = client
        self.interval_seconds = interval_seconds
        self.before: str | None = None
        self.after: str | None = None
        self.errors: list[str] = []
        self._snapshots: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_monotonic: float | None = None

    async def start(self) -> None:
        text, error = await self._scrape("before")
        self.before = text
        self._started_monotonic = time.monotonic()
        self._append_snapshot(text=text, error=error, elapsed=0.0)
        if self.interval_seconds is not None:
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._sample_loop,
                name="ltm100-server-metrics",
                daemon=True,
            )
            self._thread.start()

    async def end(self) -> None:
        self._stop.set()
        if self._thread is not None:
            await asyncio.to_thread(self._thread.join)
            self._thread = None
        text, error = await self._scrape("after")
        self.after = text
        started = self._started_monotonic
        elapsed = time.monotonic() - started if started is not None else 0.0
        self._append_snapshot(text=text, error=error, elapsed=elapsed)

    async def _scrape(self, slot: str) -> tuple[str | None, str | None]:
        try:
            async with self._client:
                text = await self._client.server_metrics_snapshot()
        except Exception as e:  # noqa: BLE001 - metrics never fail the run
            error = f"{slot} snapshot failed: {type(e).__name__}: {e}"
            with self._lock:
                self.errors.append(error)
            return None, error
        if not isinstance(text, str):
            error = f"{slot} snapshot returned {type(text).__name__}, expected str"
            with self._lock:
                self.errors.append(error)
            return None, error
        return text, None

    def _sample_loop(self) -> None:
        assert self.interval_seconds is not None
        assert self._started_monotonic is not None
        sequence = 1
        while True:
            deadline = self._started_monotonic + sequence * self.interval_seconds
            if self._stop.wait(max(0.0, deadline - time.monotonic())):
                return
            text, error = asyncio.run(self._scrape(f"interval {sequence}"))
            if self._stop.is_set():
                return
            elapsed = time.monotonic() - self._started_monotonic
            self._append_snapshot(text=text, error=error, elapsed=elapsed)
            # A slow scrape may overrun one or more scheduled boundaries. Skip
            # those boundaries rather than firing catch-up requests back to
            # back and adding observer load exactly when the server is slow.
            next_boundary = math.floor(elapsed / self.interval_seconds) + 1
            sequence = max(sequence + 1, next_boundary)

    def _append_snapshot(
        self,
        *,
        text: str | None,
        error: str | None,
        elapsed: float,
    ) -> None:
        snapshot = {
            "captured_at": time.time(),
            "elapsed_s": max(0.0, elapsed),
            "text": text,
            "error": error,
        }
        with self._lock:
            self._snapshots.append(snapshot)

    def result(self) -> dict[str, Any]:
        with self._lock:
            errors = list(self.errors)
            snapshots = [dict(snapshot) for snapshot in self._snapshots]
        return {
            "before": self.before,
            "after": self.after,
            "errors": errors,
            "interval_seconds": self.interval_seconds,
            "snapshots": snapshots,
        }


def finish(capture: dict[str, Any], *, window: str) -> dict[str, Any]:
    """Assemble the summary section from a collector result (or the parent's
    equivalent before/after pair)."""
    warnings: list[str] = list(capture.get("errors") or [])
    before, after = capture.get("before"), capture.get("after")
    if before is None or after is None:
        warnings.append("server metrics disabled: a snapshot could not be captured")
        return {
            "enabled": False,
            "window": window,
            "status": "failed",
            "warnings": warnings,
            "rows": [],
        }
    report = build_report(before, after, window=window)
    report["warnings"] = warnings + report["warnings"]
    report["enabled"] = True
    return report


def finish_timeseries(capture: dict[str, Any]) -> dict[str, Any]:
    """Build adjacent-snapshot latency deltas for periodic server metrics.

    A failed scrape invalidates only intervals touching that boundary. It is
    never bridged with the next successful scrape because doing so would label
    a multi-interval delta as one interval and distort the trace.
    """
    interval = capture.get("interval_seconds")
    snapshots = capture.get("snapshots") or []
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    failed_scrapes = sum(snapshot.get("text") is None for snapshot in snapshots)
    valid_intervals = 0

    for index, (before, after) in enumerate(pairwise(snapshots)):
        elapsed_start = float(before.get("elapsed_s", 0.0))
        elapsed_end = float(after.get("elapsed_s", elapsed_start))
        common = {
            "bucket_index": index,
            "elapsed_start_s": elapsed_start,
            "elapsed_end_s": elapsed_end,
            "interval_seconds": max(0.0, elapsed_end - elapsed_start),
            "bucket_started_at": before.get("captured_at"),
            "bucket_ended_at": after.get("captured_at"),
        }
        before_text, after_text = before.get("text"), after.get("text")
        if before_text is None or after_text is None:
            note = (
                "; ".join(
                    value
                    for value in (before.get("error"), after.get("error"))
                    if value
                )
                or "snapshot unavailable"
            )
            warnings.append(f"interval {index}: {note}")
            rows.append(
                {
                    **common,
                    "series": "",
                    "status": "failed",
                    "delta_count": None,
                    "delta_sum_s": None,
                    "mean_s": None,
                    "p50": None,
                    "p90": None,
                    "p99": None,
                    "note": note,
                }
            )
            continue

        report = build_report(before_text, after_text, window=f"interval_{index}")
        valid_intervals += 1
        for report_row in report["rows"]:
            note = report_row.get("note", "")
            status = "reset" if note.startswith("counter reset") else "ok"
            rows.append({**common, **report_row, "status": status})

    if len(snapshots) < 2 or valid_intervals == 0:
        status = "failed"
        warnings.append("server metrics time series has no valid snapshot interval")
    elif failed_scrapes:
        status = "partial"
    else:
        status = "ok"
    return {
        "enabled": bool(interval is not None and valid_intervals > 0),
        "status": status,
        "interval_seconds": interval,
        "intervals": max(0, len(snapshots) - 1),
        "valid_intervals": valid_intervals,
        "failed_scrapes": failed_scrapes,
        "warnings": warnings,
        "rows": rows,
    }


__all__ = [
    "REPORT_SERIES",
    "SampleKey",
    "SnapshotCollector",
    "build_report",
    "diff",
    "finish",
    "finish_timeseries",
    "histogram_delta",
    "parse_prometheus",
    "quantile",
]
