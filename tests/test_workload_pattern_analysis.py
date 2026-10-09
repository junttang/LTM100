"""Post-run concurrency uses exact lifetimes; burst counts include idle bins."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from decimal import Decimal
from itertools import pairwise
from pathlib import Path
from xml.etree import ElementTree

import pytest

EXAMPLES = Path(__file__).parents[1] / "examples" / "workload-patterns"


def load_example(name):
    spec = importlib.util.spec_from_file_location(
        f"pattern_analysis_{name}", EXAMPLES / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


analysis = load_example("analyze")
plot = load_example("plot")


def record(op, start, end, status="ok"):
    return {"op_type": op, "started_at": start, "ended_at": end, "status": status}


@pytest.mark.parametrize("origin", [0, 17.25, 1_700_000_000.25])
def test_decimal_boundary_requests_occupy_distinct_burst_bins(origin):
    records = [
        record("search", origin + 1.1, origin + 1.2),
        record("search", origin + 1.2, origin + 1.3),
    ]
    rows, stats = analysis.burst_distribution(
        records, start=origin, end=origin + 2, interval=0.1
    )
    assert {
        row["requests_per_bin"]: row["bins"] for row in rows if row["op_type"] == "all"
    } == {0: 18, 1: 2}
    assert stats["all"]["p99_requests_per_bin"] == 1
    assert stats["all"]["max_requests_per_bin"] == 1
    assert stats["all"]["empty_bin_fraction"] == 0.9


@pytest.mark.parametrize("origin", [0, 17.25, 1_700_000_000.25])
def test_decimal_lifetime_boundary_has_no_phantom_peak(origin):
    rows = analysis.inflight_rows(
        [record("search", origin + 1.2, origin + 1.3)],
        start=origin,
        end=origin + 2,
        interval=0.1,
    )
    assert len(rows) == 20
    assert rows[12]["search_mean"] == rows[12]["search_peak"] == 1
    assert all(
        row["total_mean"] == row["total_peak"] == 0
        for index, row in enumerate(rows)
        if index != 12
    )


@pytest.mark.parametrize("origin", [0, 17.25, 1_700_000_000.25])
def test_boundary_adjacent_floats_remain_distinct_without_epsilon(origin):
    boundary = origin + 1.2
    before = math.nextafter(boundary, -math.inf)
    after = math.nextafter(boundary, math.inf)
    following = math.nextafter(after, math.inf)
    # A real, one-representable-step request before a boundary must survive.
    rows = analysis.inflight_rows(
        [record("add", before, boundary)],
        start=origin,
        end=origin + 2,
        interval=0.1,
    )
    assert rows[11]["add_mean"] > 0
    assert rows[11]["total_peak"] == 1
    assert rows[12]["total_mean"] == rows[12]["total_peak"] == 0
    records = [
        record("add", before, boundary),
        record("add", boundary, after),
        record("add", after, following),
    ]
    rows = analysis.inflight_rows(records, start=origin, end=origin + 2, interval=0.1)
    assert rows[11]["total_peak"] == rows[12]["total_peak"] == 1
    assert all(row["total_peak"] <= 1 for row in rows)
    _, stats = analysis.burst_distribution(
        records, start=origin, end=origin + 2, interval=0.1
    )
    assert stats["all"]["requests"] == 3
    assert stats["all"]["max_requests_per_bin"] == 2
    assert stats["all"]["empty_bin_fraction"] == 0.9


def test_decimal_bin_count_and_end_exclusion():
    records = [record("add", 0.3, 0.4)]
    rows = analysis.inflight_rows(records, start=0, end=0.3, interval=0.1)
    assert len(rows) == 3
    assert all(row["interval_seconds"] == 0.1 for row in rows)
    assert all(row["total_peak"] == row["total_mean"] == 0 for row in rows)
    _, stats = analysis.burst_distribution(records, start=0, end=0.3, interval=0.1)
    assert stats["all"]["requests"] == 0
    assert stats["all"]["empty_bin_fraction"] == 1
    rows = analysis.inflight_rows([], start=0, end=0.07, interval=0.01)
    assert len(rows) == 7  # Binary division previously produced an eighth bin.
    assert all(row["interval_seconds"] == 0.01 for row in rows)


@pytest.mark.parametrize("origin", [0, 17.25, 1_700_000_000.25])
@pytest.mark.parametrize("interval", [0.01, 0.1, 0.2])
def test_decimal_bins_match_independent_overlap_and_histogram_reference(
    origin, interval
):
    # Decimal arithmetic is independent of the implementation's Fraction sweep.
    def decimal(value):
        return Decimal(str(value))

    starts = [
        float(decimal(origin) + decimal(index) * decimal(interval))
        for index in range(15)
    ]
    records = [
        record(
            "add" if index % 2 else "search",
            start,
            float(decimal(start) + decimal(interval) * Decimal("1.5")),
        )
        for index, start in enumerate(starts)
    ]
    records += [
        record("add", math.nextafter(start, -math.inf), start) for start in starts[1:]
    ]
    end = float(decimal(origin) + decimal(interval) * 20)
    rows = analysis.inflight_rows(records, start=origin, end=end, interval=interval)
    expected_starts = []
    for index, row in enumerate(rows):
        left = decimal(origin) + index * decimal(interval)
        right = left + decimal(interval)
        expected_starts.append(
            sum(left <= decimal(item["started_at"]) < right for item in records)
        )
        for op, prefix in (("add", "add"), ("search", "search"), (None, "total")):
            selected = [item for item in records if op is None or item["op_type"] == op]
            area = sum(
                max(
                    Decimal(0),
                    min(right, decimal(item["ended_at"]))
                    - max(left, decimal(item["started_at"])),
                )
                for item in selected
            )
            times = sorted(
                {left, right}
                | {
                    decimal(item[field])
                    for item in selected
                    for field in ("started_at", "ended_at")
                    if left < decimal(item[field]) < right
                }
            )
            peak = max(
                sum(
                    decimal(item["started_at"])
                    <= (a + b) / 2
                    < decimal(item["ended_at"])
                    for item in selected
                )
                for a, b in pairwise(times)
            )
            assert row[f"{prefix}_mean"] == pytest.approx(
                float(area / (right - left)), rel=1e-15, abs=1e-30
            )
            assert row[f"{prefix}_peak"] == peak
    histogram, _ = analysis.burst_distribution(
        records, start=origin, end=end, interval=interval
    )
    assert {
        row["requests_per_bin"]: row["bins"]
        for row in histogram
        if row["op_type"] == "all"
    } == {
        count: expected_starts.count(count) for count in range(max(expected_starts) + 1)
    }


def test_exact_overlap_mean_and_peak_with_tied_end_and_start():
    records = [
        record("search", 0, 0.6),
        record("add", 0.6, 0.7),
        record("add", 0.2, 0.4),
        record("add", 0.7, 0.7),
        record("search", 0.3, 0.3, "rejected"),
    ]
    rows = analysis.inflight_rows(records, start=0, end=1, interval=0.5)
    assert rows[0]["search_mean"] == pytest.approx(1)
    assert rows[0]["add_mean"] == pytest.approx(0.4)
    assert rows[0]["total_mean"] == pytest.approx(1.4)
    assert rows[0]["total_peak"] == 2
    assert rows[1]["add_mean"] == pytest.approx(0.2)
    assert rows[1]["search_mean"] == pytest.approx(0.2)
    assert rows[1]["add_peak"] == rows[1]["search_peak"] == 1
    assert rows[1]["total_peak"] == 1  # Individual peaks never occur together.


def test_clips_lifetimes_at_both_window_boundaries_and_partial_bin():
    rows = analysis.inflight_rows(
        [record("search", -1, 0.25), record("add", 0.75, 2)],
        start=0,
        end=1.25,
        interval=0.5,
    )
    assert [row["interval_seconds"] for row in rows] == [0.5, 0.5, 0.25]
    assert [row["total_mean"] for row in rows] == [0.5, 0.5, 1]
    assert sum(row["total_mean"] * row["interval_seconds"] for row in rows) == 0.75


def test_peak_preserves_short_request_that_endpoint_sampling_would_miss():
    row = analysis.inflight_rows(
        [record("add", 1, 1.001)],
        start=0,
        end=5,
        interval=5,
    )[0]
    assert row["total_peak"] == 1
    assert row["total_mean"] == pytest.approx(0.001 / 5)


def test_concurrency_matches_independent_overlap_area_and_midpoint_reference():
    records = [
        record("add" if index % 2 else "search", index * 0.125, index * 0.125 + 0.75)
        for index in range(20)
    ]
    rows = analysis.inflight_rows(records, start=0, end=4, interval=0.5)
    critical_times = sorted(
        {
            0,
            4,
            *(row["started_at"] for row in records),
            *(row["ended_at"] for row in records),
        }
    )
    for row in rows:
        left = row["elapsed_start_s"]
        right = left + row["interval_seconds"]
        area = sum(
            max(0, min(right, item["ended_at"]) - max(left, item["started_at"]))
            for item in records
        )
        assert row["total_mean"] * row["interval_seconds"] == pytest.approx(area)
        times = sorted(
            {left, right, *(time for time in critical_times if left < time < right)}
        )
        peak = max(
            sum(
                item["started_at"] <= (a + b) / 2 < item["ended_at"] for item in records
            )
            for a, b in pairwise(times)
        )
        assert row["total_peak"] == peak


def test_burst_distribution_uses_starts_half_open_window_and_empty_bins():
    records = [
        record("add", 0.5, 1.5),  # Active in the window, but started before it.
        record("search", 1, 2),
        record("add", 1.2, 1.3, "error"),
        record("search", 1.5, 2),
        record("search", 1.5, 1.5, "rejected"),
        record("add", 2.8, 2.9),
        record("add", 3, 3.1),  # End boundary belongs to the next window.
    ]
    rows, stats = analysis.burst_distribution(records, start=1, end=3, interval=0.5)
    histogram = {
        row["requests_per_bin"]: row for row in rows if row["op_type"] == "all"
    }
    assert {count: row["bins"] for count, row in histogram.items()} == {
        0: 1,
        1: 2,
        2: 1,
    }
    assert histogram[0]["fraction"] == 0.25
    assert histogram[2]["cumulative_fraction"] == 1
    assert stats["all"]["requests"] == 4
    assert stats["all"]["requests_s"] == 2
    assert stats["all"]["p99_requests_per_bin"] == 2
    assert stats["all"]["empty_bin_fraction"] == 0.25
    assert stats["add"]["requests"] == stats["search"]["requests"] == 2


def test_empty_records_preserve_idle_time_and_zero_percentiles():
    rows = analysis.inflight_rows([], start=0, end=10, interval=5)
    assert len(rows) == 2
    assert all(row["total_mean"] == row["total_peak"] == 0 for row in rows)
    histogram, stats = analysis.burst_distribution([], start=0, end=10, interval=0.1)
    assert all(row["bins"] == 100 and row["fraction"] == 1 for row in histogram)
    assert (
        stats["all"]["p95_requests_per_bin"]
        == stats["all"]["p99_requests_per_bin"]
        == 0
    )


@pytest.mark.parametrize(
    "function", [analysis.inflight_rows, analysis.burst_distribution]
)
@pytest.mark.parametrize(
    "start,end,interval", [(0, 1, 0), (0, 0, 1), (0, 1, float("nan"))]
)
def test_invalid_analysis_intervals_fail(function, start, end, interval):
    with pytest.raises(ValueError):
        function([], start=start, end=end, interval=interval)


def test_partial_burst_bin_is_rejected_instead_of_biasing_distribution():
    with pytest.raises(ValueError, match="whole number"):
        analysis.burst_distribution([], start=0, end=1.05, interval=0.1)


def test_analysis_outputs_and_both_graphs_reproduce_without_raw(tmp_path):
    source = tmp_path / "source" / "users-2"
    source.mkdir(parents=True)
    (source / "summary.json").write_text(
        json.dumps({"meta": {"measurement_started_at": 0, "measurement_ended_at": 10}})
    )
    records = [record("search", 0, 0.6), record("add", 0.6, 0.7), record("add", 7, 7.1)]
    (source / "raw.ndjson").write_text("\n".join(json.dumps(row) for row in records))
    output = tmp_path / "derived" / "users-2"
    analysis.analyze_case(
        source,
        output,
        inflight_interval=5,
        burst_interval=0.1,
        window_start=5,
        window_end=10,
    )
    payload = json.loads((output / "pattern-statistics.json").read_text())
    assert payload["window_bins"] == 50
    assert payload["stats"]["all"]["requests"] == 1
    assert payload["stats"]["all"]["mean_in_flight"] == pytest.approx(0.1 / 5)
    with (output / "inflight.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert not (output / "raw.ndjson").exists()
    pytest.importorskip("matplotlib")
    for kind, function, filename in (
        ("inflight", plot.plot_inflight, "inflight.csv"),
        ("bursts", plot.plot_bursts, "burst-histogram.csv"),
    ):
        image = tmp_path / f"{kind}.svg"
        function([output / filename], image)
        assert ElementTree.parse(image).getroot().tag.endswith("svg")
