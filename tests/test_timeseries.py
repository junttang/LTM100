"""Tests for fixed-interval client-observed E2E reporting."""

from __future__ import annotations

import csv
import json

import pytest
import yaml

from ltm100.core.multiproc import ShardResult
from ltm100.core.op import OpResult, OpType
from ltm100.core.runner import SessionAdmissionStats
from ltm100.metrics.report import write_timeseries_csv
from ltm100.metrics.timeseries import aggregate_timeseries


def _row(rows, bucket: int, op_type: str):
    return next(
        row
        for row in rows
        if row["bucket_index"] == bucket and row["op_type"] == op_type
    )


def test_timeseries_tracks_starts_completions_inflight_and_empty_buckets():
    results = [
        OpResult(OpType.ADD, "u0", 100.2, 101.2, "ok", n_items=5),
        OpResult(OpType.SEARCH, "u1", 100.3, 100.4, "ok", n_items=0),
        OpResult(
            OpType.SEARCH,
            "u2",
            101.1,
            102.4,
            "error",
            error_kind="timeout",
        ),
        OpResult(
            OpType.SEARCH,
            "u3",
            101.5,
            101.5,
            "rejected",
            error_kind="queue_full",
        ),
    ]

    rows = aggregate_timeseries(
        results,
        interval_seconds=1.0,
        measurement_started_at=100.0,
        measurement_ended_at=103.0,
    )

    assert len(rows) == 9  # add, search, all for every one-second bucket
    add_0 = _row(rows, 0, "add")
    assert add_0["started"] == 1
    assert add_0["completed"] == 0
    assert add_0["in_flight_end"] == 1
    assert add_0["latency_samples"] == 0

    add_1 = _row(rows, 1, "add")
    assert add_1["started"] == 0
    assert add_1["completed"] == 1
    assert add_1["successful_ops_s"] == 1.0
    assert add_1["in_flight_end"] == 0
    assert add_1["latency_mean_ms"] == pytest.approx(1000.0)
    assert add_1["n_items_mean"] == 5.0

    search_0 = _row(rows, 0, "search")
    assert search_0["successful"] == 1
    assert search_0["latency_p95_ms"] == pytest.approx(100.0)
    assert search_0["empty_rate"] == 1.0

    search_1 = _row(rows, 1, "search")
    assert search_1["started"] == 1
    assert search_1["rejected"] == 1
    assert search_1["completed"] == 0
    assert search_1["in_flight_end"] == 1

    search_2 = _row(rows, 2, "search")
    assert search_2["completed"] == 1
    assert search_2["errors"] == 1
    assert search_2["in_flight_end"] == 0

    # The empty add bucket remains present, so a completion stall is visible.
    add_2 = _row(rows, 2, "add")
    assert add_2["started"] == add_2["completed"] == 0
    assert add_2["latency_mean_ms"] is None

    all_0 = _row(rows, 0, "all")
    assert all_0["started"] == 2
    assert all_0["completed"] == 1
    assert all_0["in_flight_end"] == 1
    assert all_0["latency_mean_ms"] is None


def test_timeseries_uses_actual_width_for_final_partial_bucket():
    rows = aggregate_timeseries(
        [OpResult(OpType.ADD, "u0", 14.1, 14.5, "ok")],
        interval_seconds=2.0,
        measurement_started_at=10.0,
        measurement_ended_at=15.0,
    )

    final = _row(rows, 2, "add")
    assert final["elapsed_start_s"] == 4.0
    assert final["elapsed_end_s"] == 5.0
    assert final["interval_seconds"] == 1.0
    assert final["successful_ops_s"] == 1.0


@pytest.mark.parametrize("interval", [0.0, -1.0, float("inf"), float("nan")])
def test_timeseries_rejects_invalid_interval(interval):
    with pytest.raises(ValueError, match="interval_seconds"):
        aggregate_timeseries(
            [],
            interval_seconds=interval,
            measurement_started_at=0.0,
            measurement_ended_at=1.0,
        )


def test_write_timeseries_csv_formats_timestamps_and_empty_metrics(tmp_path):
    rows = aggregate_timeseries(
        [OpResult(OpType.ADD, "u0", 100.2, 101.2, "ok")],
        interval_seconds=1.0,
        measurement_started_at=100.0,
        measurement_ended_at=102.0,
    )
    path = tmp_path / "timeseries.csv"

    write_timeseries_csv(rows, path)

    with open(path, newline="", encoding="utf-8") as file:
        written = list(csv.DictReader(file))
    assert written[0]["bucket_started_at"] == "1970-01-01T00:01:40+00:00"
    assert written[0]["op_type"] == "add"
    assert written[0]["latency_mean_ms"] == ""
    assert written[1]["op_type"] == "all"
    assert written[1]["latency_mean_ms"] == ""


def test_cli_writes_timeseries_without_raw_output(tmp_path, monkeypatch, capsys):
    from ltm100 import cli

    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "dataset": {"name": "synthetic"},
                "backend": {
                    "name": "memmachine",
                    "base_url": "http://localhost:8080",
                },
            }
        )
    )
    results = [
        OpResult(OpType.ADD, "u0", 100.2, 100.4, "ok", n_items=1),
        OpResult(OpType.ADD, "u0", 101.2, 101.5, "ok", n_items=1),
    ]
    monkeypatch.setattr(cli, "_backend_build", lambda cfg: {})

    def fake_run_shards(entry, args, procs, **kwargs):
        assert procs == 2
        assert callable(kwargs["on_measure_start"])
        assert callable(kwargs["on_measure_end"])
        kwargs["on_measure_start"]()
        kwargs["on_measure_end"]()
        return ShardResult(
            results,
            SessionAdmissionStats(),
            measurement_started_at=100.0,
            measurement_ended_at=102.0,
        )

    monkeypatch.setattr(cli, "run_shards", fake_run_shards)
    output = tmp_path / "out"
    args = cli.build_parser().parse_args(
        [
            "run",
            "--config",
            str(config),
            "--scenario",
            "add-load",
            "--ops",
            "2",
            "--users",
            "2",
            "--procs",
            "2",
            "--output",
            str(output),
            "--time-series-interval",
            "1",
        ]
    )

    assert cli._run(args) == 0
    payload = json.loads(capsys.readouterr().out.split("\nreports written")[0])
    assert payload["meta"]["measurement_started_at"] == "1970-01-01T00:01:40+00:00"
    assert payload["meta"]["measurement_ended_at"] == "1970-01-01T00:01:42+00:00"
    assert payload["meta"]["time_series_interval"] == 1.0
    assert (output / "timeseries.csv").exists()
    assert not (output / "raw.ndjson").exists()


def test_cli_timeseries_requires_positive_interval_and_output(tmp_path, monkeypatch):
    from ltm100 import cli

    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "dataset": {"name": "synthetic"},
                "backend": {
                    "name": "memmachine",
                    "base_url": "http://localhost:8080",
                },
            }
        )
    )
    monkeypatch.setattr(cli, "_backend_build", lambda cfg: pytest.fail("too late"))
    common = [
        "run",
        "--config",
        str(config),
        "--scenario",
        "add-load",
        "--ops",
        "1",
    ]

    args = cli.build_parser().parse_args(
        [*common, "--output", str(tmp_path / "out"), "--time-series-interval", "0"]
    )
    with pytest.raises(ValueError, match="finite number > 0"):
        cli._run(args)

    args = cli.build_parser().parse_args([*common, "--time-series-interval", "1"])
    with pytest.raises(ValueError, match="requires --output"):
        cli._run(args)
