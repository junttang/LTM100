"""Tests for summary/raw report writers."""

from __future__ import annotations

import csv
import json

from ltm100.core.op import OpResult, OpType
from ltm100.metrics.aggregate import aggregate
from ltm100.metrics.report import (
    write_raw_ndjson,
    write_summary_csv,
    write_summary_json,
)


def _results():
    return [
        OpResult(OpType.ADD, "u0", 1.0, 1.3, "ok", n_items=5),
        OpResult(OpType.SEARCH, "u1", 2.0, 2.4, "error", error_kind="timeout"),
        OpResult(OpType.SEARCH, "u0", 3.0, 3.2, "ok"),
    ]


def test_write_summary_json_roundtrips(tmp_path):
    summary = aggregate(_results())
    p = tmp_path / "summary.json"
    write_summary_json(summary, p, meta={"users": 2, "seed": 0})
    data = json.loads(p.read_text())
    assert data["meta"]["users"] == 2
    assert data["summary"]["total"] == 3
    assert set(data["summary"]["by_op"]) == {"add", "search"}


def test_write_summary_csv_has_rows_per_op(tmp_path):
    summary = aggregate(_results())
    p = tmp_path / "summary.csv"
    write_summary_csv(summary, p)
    with open(p) as f:
        rows = list(csv.reader(f))
    assert rows[0][0] == "op_type"
    op_types = {r[0] for r in rows[1:]}
    assert op_types == {"add", "search"}


def test_write_raw_ndjson_one_line_per_result(tmp_path):
    p = tmp_path / "raw.ndjson"
    write_raw_ndjson(_results(), p)
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 3
    first = json.loads(lines[0])
    assert first["op_type"] == "add"
    assert first["status"] == "ok"
    assert abs(first["latency_ms"] - 300.0) < 1e-6
    err = json.loads(lines[1])
    assert err["status"] == "error"
    assert err["error_kind"] == "timeout"
