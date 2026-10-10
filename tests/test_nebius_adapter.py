"""Nebius loading and source normalization without network access."""

from __future__ import annotations

import builtins
import copy
import json
import sys
from types import SimpleNamespace
from typing import ClassVar

import pytest

from ltm100.adapters.datasets.nebius import DATASET_ID, DEFAULT_REVISION, NebiusAdapter


def _call(call_id="view-0", name="str_replace_editor", arguments=None):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(
                arguments or {"command": "view", "path": "/repo/a.py"}
            ),
        },
    }


def _record(index=0):
    """Small authored fixture with the public source schema, not source data."""
    return {
        "trajectory_id": f"trajectory-{index}",
        "instance_id": f"example/repo-{index}",
        "repo": "example/repo",
        "exit_status": "submit",
        "resolved": index % 2,
        "trajectory": [
            {"role": "system", "content": "Agent instructions"},
            {"role": "user", "content": f"Fix bug {index}"},
            {"role": "assistant", "content": "", "tool_calls": [_call()]},
            {
                "role": "tool",
                "content": f"File contents {index}",
                "name": "str_replace_editor",
                "tool_call_id": "view-0",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _call("finish-0", "finish", {"message": f"Fixed bug {index}"})
                ],
            },
        ],
        "tools": [{"type": "function", "function": {"name": "str_replace_editor"}}],
        "model_patch": "unused patch",
    }


def _adapter(records):
    adapter = NebiusAdapter()
    adapter._records = records
    return adapter


@pytest.mark.parametrize("suffix", [".json", ".jsonl"])
@pytest.mark.parametrize("length,expected", [(None, 3), (0, 0), (1, 1), (20, 3)])
def test_local_loading_and_length(tmp_path, suffix, length, expected):
    path = tmp_path / f"source{suffix}"
    records = [_record(i) for i in range(3)]
    content = (
        json.dumps(records)
        if suffix == ".json"
        else "\n".join(map(json.dumps, records))
    )
    path.write_text(content)
    adapter = NebiusAdapter(path=str(path), length=length)

    assert len(adapter._load()) == expected
    assert adapter._load() is adapter._load()
    if expected:
        assert adapter._task(0).trajectory_id == "trajectory-0"


def test_jsonl_skips_blank_lines_and_stops_at_limit(tmp_path):
    path = tmp_path / "source.jsonl"
    path.write_text("\n" + json.dumps(_record()) + "\n\ninvalid tail")
    assert len(NebiusAdapter(path=str(path), length=1)._load()) == 1
    with pytest.raises(ValueError, match="source.jsonl:4"):
        NebiusAdapter(path=str(path))._load()


@pytest.mark.parametrize("payload", ["", "{}", '"text"'])
def test_json_requires_array(tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_text(payload)
    with pytest.raises(ValueError, match="must contain an array"):
        NebiusAdapter(path=str(path))._load()


def test_local_source_ignores_hugging_face(monkeypatch, tmp_path):
    path = tmp_path / "source.json"
    path.write_text(json.dumps([_record()]))
    monkeypatch.setitem(sys.modules, "datasets", None)
    assert (
        NebiusAdapter(path=str(path), revision="unused")._task(0).repo == "example/repo"
    )


def test_local_missing_file_and_unknown_format(tmp_path):
    with pytest.raises(FileNotFoundError):
        NebiusAdapter(path=str(tmp_path / "missing.json"))._load()
    with pytest.raises(ValueError, match=".json, .jsonl, or .parquet"):
        NebiusAdapter(path=str(tmp_path / "source.txt"))._load()


@pytest.mark.parametrize("length", [-1, True, 1.5, "2"])
def test_invalid_length(length):
    with pytest.raises(ValueError, match="length"):
        NebiusAdapter(length=length)


@pytest.mark.parametrize("chunk_chars", [0, -1, True, 1.5, "2"])
def test_invalid_chunk_size(chunk_chars):
    with pytest.raises(ValueError, match="chunk_chars"):
        NebiusAdapter(chunk_chars=chunk_chars)


@pytest.mark.parametrize("revision", [None, "", " "])
def test_invalid_revision(revision):
    with pytest.raises(ValueError, match="revision"):
        NebiusAdapter(revision=revision)


class _ArrowRecords(list):
    column_names: ClassVar[list[str]] = ["trajectory"]

    def select(self, indices):
        return _ArrowRecords(self[i] for i in indices)

    def to_list(self):
        raise AssertionError("the whole Arrow dataset must not be materialized")


@pytest.mark.parametrize("local", [False, True])
def test_arrow_loader_contract(monkeypatch, tmp_path, local):
    calls = []
    records = _ArrowRecords([_record(i) for i in range(3)])

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return records

    monkeypatch.setitem(
        sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset)
    )
    path = str(tmp_path / "source.parquet") if local else None
    adapter = NebiusAdapter(path=path, cache_dir="cache", revision="pinned", length=2)
    assert len(adapter._load()) == 2
    assert adapter._task(1).trajectory_id == "trajectory-1"
    assert calls == [
        (("parquet",), {"data_files": path, "split": "train", "cache_dir": "cache"})
        if local
        else (
            (DATASET_ID,),
            {"split": "train", "revision": "pinned", "cache_dir": "cache"},
        )
    ]


def test_default_hugging_face_revision():
    assert NebiusAdapter().revision == DEFAULT_REVISION


def test_arrow_dependency_error(monkeypatch):
    original = builtins.__import__

    def without_datasets(name, *args, **kwargs):
        if name == "datasets":
            raise ImportError("not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_datasets)
    with pytest.raises(ImportError, match=r"\[datasets\]"):
        NebiusAdapter()._load()


def test_loader_failure_propagates_without_fallback(monkeypatch):
    failure = OSError("cache is inaccessible")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fail))
    with pytest.raises(OSError) as error:
        NebiusAdapter()._load()
    assert error.value is failure


def test_arrow_missing_schema(monkeypatch):
    records = _ArrowRecords([_record()])
    records.column_names = ["wrong"]
    monkeypatch.setitem(
        sys.modules, "datasets", SimpleNamespace(load_dataset=lambda *a, **k: records)
    )
    with pytest.raises(ValueError, match="trajectory column"):
        NebiusAdapter()._load()


def test_structured_task_retains_empty_assistant_and_finish():
    task = _adapter([_record()])._task(0)
    assert [m.role for m in task.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    call = task.messages[2].tool_calls[0]
    assert task.messages[2].content == ""
    assert call.id == task.messages[3].tool_call_id == "view-0"
    assert call.name == task.messages[3].name == "str_replace_editor"
    assert call.arguments == {"command": "view", "path": "/repo/a.py"}
    assert task.final_response == "Fixed bug 0"
    assert task.exit_status == "submit"
    assert task.resolved is False  # submit is not equivalent to solving the issue


@pytest.mark.parametrize("decoded", [False, True])
def test_normalization_does_not_mutate_source_or_share_arguments(decoded):
    record = _record()
    if decoded:
        record["trajectory"][2]["tool_calls"][0]["function"]["arguments"] = {
            "nested": ["original"]
        }
    original = copy.deepcopy(record)
    adapter = _adapter([record])
    task = adapter._task(0)
    task.messages[2].tool_calls[0].arguments["new"] = "changed"
    if decoded:
        task.messages[2].tool_calls[0].arguments["nested"].append("changed")
    assert record == original
    assert "new" not in adapter._task(0).messages[2].tool_calls[0].arguments


def test_multiple_tool_calls_and_unknown_tool_preserve_order():
    record = _record()
    record["trajectory"][2]["tool_calls"].append(_call("other-0", "future_tool"))
    record["trajectory"].insert(
        4,
        {
            "role": "tool",
            "content": "other result",
            "name": "future_tool",
            "tool_call_id": "other-0",
        },
    )
    task = _adapter([record])._task(0)
    assert [c.name for c in task.messages[2].tool_calls] == [
        "str_replace_editor",
        "future_tool",
    ]
    assert task.messages[4].content == "other result"


def test_interrupted_attempt_keeps_unobserved_terminal_action():
    record = _record()
    record["exit_status"] = "RuntimeError: maximum iteration"
    record["trajectory"] = record["trajectory"][:3]
    task = _adapter([record])._task(0)
    assert task.final_response is None
    assert task.messages[-1].tool_calls[0].name == "str_replace_editor"
    assert len(task.messages) == 3


@pytest.mark.parametrize(
    "mutation,pattern",
    [
        (lambda r: r.pop("trajectory_id"), "trajectory_id"),
        (lambda r: r.update(repo=""), "repo"),
        (lambda r: r.update(resolved=2), "resolved"),
        (lambda r: r.update(trajectory=[]), "non-empty list"),
        (lambda r: r["trajectory"][1].update(role="unknown"), "unsupported role"),
        (lambda r: r["trajectory"][1].update(content=None), "content"),
        (lambda r: r["trajectory"][2].update(tool_calls={}), "tool_calls"),
        (
            lambda r: r["trajectory"][2]["tool_calls"][0]["function"].update(
                arguments="bad"
            ),
            "Expecting value",
        ),
        (
            lambda r: r["trajectory"][2]["tool_calls"][0]["function"].update(
                arguments="[]"
            ),
            "arguments",
        ),
        (lambda r: r["trajectory"][3].update(tool_call_id="unknown"), "unmatched"),
        (lambda r: r["trajectory"][3].update(name="wrong"), "unmatched"),
        (
            lambda r: r["trajectory"].insert(4, copy.deepcopy(r["trajectory"][3])),
            "duplicate tool observation",
        ),
        (
            lambda r: r["trajectory"][4]["tool_calls"][0].update(id="view-0"),
            "duplicate tool call",
        ),
        (lambda r: r["trajectory"].pop(3), "missing tool observation"),
        (
            lambda r: r["trajectory"][4]["tool_calls"][0]["function"].update(
                arguments="{}"
            ),
            "finish message",
        ),
        (lambda r: r["trajectory"].pop(1), "no user request"),
    ],
)
def test_malformed_records_fail_with_source_row_context(mutation, pattern):
    record = _record()
    mutation(record)
    with pytest.raises(ValueError, match=f"row 0:.*{pattern}"):
        _adapter([record])._task(0)


def test_source_commands_are_never_executed(tmp_path):
    target = tmp_path / "must-not-exist"
    record = _record()
    record["trajectory"][2]["tool_calls"][0]["function"]["arguments"] = json.dumps(
        {"command": f"touch {target}"}
    )
    _adapter([record])._task(0)
    assert not target.exists()
