"""Nebius loading and source normalization without network access."""

from __future__ import annotations

import builtins
import copy
import json
import random
import sys
from types import SimpleNamespace

import pytest

from ltm100.adapters.datasets import nebius as nebius_module
from ltm100.adapters.datasets.nebius import DATASET_ID, DEFAULT_REVISION, NebiusAdapter
from ltm100.config import AdapterConfig, build_dataset
from ltm100.core.scenarios import ChatReplay


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


@pytest.mark.parametrize("local", [False, True])
def test_parquet_loader_contract(monkeypatch, tmp_path, local):
    downloads = []
    reads = []
    records = [_record(i) for i in range(3)]

    def download(**kwargs):
        downloads.append(kwargs)
        return str(tmp_path / "downloaded.parquet")

    def read(source, length):
        reads.append((source, length))
        return records[:length]

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download)
    )
    monkeypatch.setattr(nebius_module, "_ParquetRecords", read)
    path = str(tmp_path / "source.parquet") if local else None
    adapter = NebiusAdapter(path=path, cache_dir="cache", revision="pinned", length=2)
    assert len(adapter._load()) == 2
    assert adapter._task(1).trajectory_id == "trajectory-1"
    assert downloads == (
        []
        if local
        else [
            {
                "repo_id": DATASET_ID,
                "filename": "trajectories.parquet",
                "repo_type": "dataset",
                "revision": "pinned",
                "cache_dir": "cache",
            }
        ]
    )
    assert reads == [
        (tmp_path / ("source.parquet" if local else "downloaded.parquet"), 2)
    ]


def test_default_hugging_face_revision():
    assert NebiusAdapter().revision == DEFAULT_REVISION


def test_arrow_dependency_error(monkeypatch):
    original = builtins.__import__

    def without_datasets(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ImportError("not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_datasets)
    with pytest.raises(ImportError, match=r"\[datasets\]"):
        NebiusAdapter()._load()


def test_loader_failure_propagates_without_fallback(monkeypatch):
    failure = OSError("cache is inaccessible")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=fail)
    )
    with pytest.raises(OSError) as error:
        NebiusAdapter()._load()
    assert error.value is failure


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


@pytest.mark.parametrize("tasks,users", [(8, 3), (3, 3), (3, 8), (1, 4)])
@pytest.mark.parametrize("seed", [0, 42, -7])
def test_user_assignment_and_finite_task_stream(tasks, users, seed):
    adapter = _adapter([_record(i) for i in range(tasks)])
    user_ids = adapter.users(users, seed=seed)
    shuffled = list(range(tasks))
    random.Random(seed).shuffle(shuffled)
    assert len(set(user_ids)) == users
    sizes = []
    all_indices = []
    for i, user in enumerate(user_ids):
        expected = shuffled[i::users] if tasks >= users else [shuffled[i % tasks]]
        assert adapter.task_indices(user) == tuple(expected)
        assigned = list(adapter.task_stream(user))
        assert [task.trajectory_id for task in assigned] == [
            f"trajectory-{j}" for j in expected
        ]
        assert assigned == list(adapter.task_stream(user))
        sizes.append(len(assigned))
        all_indices.extend(expected)
    assert max(sizes) - min(sizes) <= 1
    if tasks >= users:
        assert sorted(all_indices) == list(range(tasks))


def test_seed_changes_assignment_without_changing_source():
    adapter = _adapter([_record(i) for i in range(10)])
    users = adapter.users(3, seed=0)
    original = [adapter.task_indices(user) for user in users]
    assert adapter.users(3, seed=42) == users
    changed = [adapter.task_indices(user) for user in users]
    assert changed != original
    adapter.users(3, seed=0)
    assert [adapter.task_indices(user) for user in users] == original


def test_empty_source_and_zero_users():
    assert _adapter([]).users(3, seed=0) == []
    assert _adapter([_record()]).users(0, seed=0) == []


@pytest.mark.parametrize("users", [-1, True, 1.5])
def test_invalid_user_count(users):
    with pytest.raises(ValueError, match="n_users"):
        _adapter([_record()]).users(users, seed=0)


def test_unknown_user_does_not_fall_back_to_first_record():
    adapter = _adapter([_record()])
    with pytest.raises(ValueError, match="call users"):
        list(adapter.task_stream("unknown"))
    adapter.users(1, seed=0)
    with pytest.raises(ValueError, match="unknown nebius user"):
        list(adapter.memory_stream("neb_user_00001"))


def test_invalid_source_fails_at_user_initialization():
    record = _record()
    record["trajectory"][2]["tool_calls"][0]["function"]["arguments"] = "bad"
    with pytest.raises(ValueError, match="row 0"):
        _adapter([record]).users(1, seed=0)


def test_duplicate_trajectory_ids_fail_before_run():
    with pytest.raises(ValueError, match="duplicate nebius trajectory_id"):
        _adapter([_record(), _record()]).users(2, seed=0)


def test_generic_memory_projection_retains_tools_roles_and_final_response():
    adapter = _adapter([_record()])
    user = adapter.users(1, seed=0)[0]
    items = list(adapter.memory_stream(user))
    assert [item.role for item in items] == ["user", "assistant", "tool", "assistant"]
    assert items[0].content == "Fix bug 0"
    assert (
        items[1].content
        == 'Tool call: str_replace_editor\nArguments: {"command": "view", "path": "/repo/a.py"}'
    )
    assert items[2].content == "Tool result: str_replace_editor\n\nFile contents 0"
    assert items[3].content == "Fixed bug 0"
    assert all(item.producer == user and item.metadata == {} for item in items)
    assert all("Agent instructions" not in item.content for item in items)
    assert items == list(adapter.memory_stream(user))


def test_final_answer_keeps_assistant_content_without_duplicate_finish():
    record = _record()
    record["trajectory"][-1]["content"] = "Explanation before final answer"
    adapter = _adapter([record])
    user = adapter.users(1, seed=0)[0]
    final = list(adapter.memory_stream(user))[-1].content
    assert final == "Explanation before final answer\n\nFixed bug 0"
    record["trajectory"][-1]["content"] = "Fixed bug 0"
    assert list(adapter.memory_stream(user))[-1].content == "Fixed bug 0"


def test_interrupted_projection_does_not_add_a_final_response():
    record = _record()
    record["trajectory"] = record["trajectory"][:3]
    record["exit_status"] = "RuntimeError: maximum iteration"
    adapter = _adapter([record])
    user = adapter.users(1, seed=0)[0]
    assert len(list(adapter.memory_stream(user))) == 2
    assert next(adapter.task_stream(user)).final_response is None


@pytest.mark.parametrize("content", ["가" * 101, "longword" * 40, "a b c " * 100])
def test_projection_chunking_does_not_split_structured_source(content):
    record = _record()
    record["trajectory"][1]["content"] = content
    adapter = _adapter([record])
    adapter.chunk_chars = 50
    user = adapter.users(1, seed=0)[0]
    items = list(adapter.memory_stream(user))
    assert all(0 < len(item.content) <= 50 for item in items)
    assert next(adapter.task_stream(user)).messages[1].content == content
    assert all(item.role and item.producer == user for item in items)


def test_nebius_registry_and_dialogue_capability_rejection():
    adapter = build_dataset(AdapterConfig("nebius", {"length": 2}))
    assert isinstance(adapter, NebiusAdapter)
    assert not hasattr(adapter, "turn_stream")
    assert not hasattr(adapter, "session_stream")
    with pytest.raises(ValueError, match="turn_stream"):
        ChatReplay().validate(adapter)


@pytest.mark.parametrize(
    "arguments", ['{"value": NaN}', '{"value": 1e400}', {"value": float("inf")}]
)
def test_nonfinite_arguments_fail_before_projection(arguments):
    record = _record()
    record["trajectory"][2]["tool_calls"][0]["function"]["arguments"] = arguments
    with pytest.raises(ValueError, match="JSON compliant"):
        _adapter([record]).users(1, seed=0)


def test_additional_framework_user_messages_are_preserved():
    record = _record()
    record["trajectory"].insert(
        4, {"role": "user", "content": "Please continue working on the task."}
    )
    adapter = _adapter([record])
    user = adapter.users(1, seed=0)[0]
    task = next(adapter.task_stream(user))
    assert task.messages[4].role == "user"
    assert task.messages[4].content == "Please continue working on the task."
    assert task.final_response == "Fixed bug 0"
    assert list(adapter.memory_stream(user))[3].content == task.messages[4].content
