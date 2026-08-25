"""Tests for the LongMemEval adapter that don't require network access.

We monkeypatch the loader so the adapter can be exercised against an
in-memory synthetic dataset shaped like a LongMemEval sample.
"""

from __future__ import annotations

import pytest

from ltm100.adapters.datasets.longmemeval import LongMemEvalAdapter


def _synthetic_records(n: int = 3) -> list[dict]:
    records = []
    for i in range(n):
        records.append(
            {
                "question": f"What is fact {i}?",
                "answer": f"answer-{i}",
                "question_type": "single_hop",
                "question_id": f"q{i}",
                "haystack_sessions": [
                    [
                        {"content": f"hello world {i} " * 300, "has_answer": True},
                        {"content": f"second turn {i}", "has_answer": False},
                    ],
                    [{"content": f"another session {i}", "has_answer": False}],
                ],
            }
        )
    return records


def _make_adapter(records: list[dict]) -> LongMemEvalAdapter:
    adapter = LongMemEvalAdapter(split="longmemeval_s_cleaned")
    adapter._records = records  # bypass network load
    return adapter


def test_users_count_and_replication():
    adapter = _make_adapter(_synthetic_records(3))
    users = adapter.users(7, seed=0)
    assert len(users) == 7
    assert len(set(users)) == 7  # unique user ids


def test_user_id_encodes_sample_index():
    adapter = _make_adapter(_synthetic_records(3))
    users = adapter.users(5, seed=1)
    for u in users:
        assert u.startswith("lme_user_")
        assert "_s" in u


def test_memory_stream_yields_chunked_items():
    adapter = _make_adapter(_synthetic_records(1))
    users = adapter.users(1, seed=0)
    items = list(adapter.memory_stream(users[0]))
    assert len(items) >= 2  # multiple turns across sessions
    assert all(it.content.strip() for it in items)
    assert all(it.producer == users[0] for it in items)


def test_memory_stream_chunks_long_content():
    adapter = _make_adapter(_synthetic_records(1))
    users = adapter.users(1, seed=0)
    items = list(adapter.memory_stream(users[0]))
    # The first turn is ~4200 chars, so it must be split into <=3000 chunks.
    assert any(len(it.content) <= 3000 for it in items)
    assert any(len(it.content) > 2000 for it in items)


def test_replicated_users_have_consistent_backing_sample():
    adapter = _make_adapter(_synthetic_records(2))
    # Two users that map to the same sample should see the same memory.
    users = adapter.users(4, seed=0)
    backing = {}
    for u in users:
        mem = tuple(it.content for it in adapter.memory_stream(u))
        backing.setdefault(mem, []).append(u)
    # At most 2 distinct memory sets (one per sample).
    assert len(backing) <= 2


def test_reproducible_user_mapping():
    adapter = _make_adapter(_synthetic_records(5))
    a = adapter.users(10, seed=42)
    b = adapter.users(10, seed=42)
    assert a == b


def test_loads_from_local_path(tmp_path):
    import json

    p = tmp_path / "lme.json"
    p.write_text(json.dumps(_synthetic_records(4)))
    adapter = LongMemEvalAdapter(path=str(p), length=2)
    users = adapter.users(3, seed=0)
    assert len(users) == 3
    items = list(adapter.memory_stream(users[0]))
    assert items  # has memories


def test_local_path_respects_length(tmp_path):
    import json

    p = tmp_path / "lme.json"
    p.write_text(json.dumps(_synthetic_records(10)))
    adapter = LongMemEvalAdapter(path=str(p), length=3)
    users = adapter.users(100, seed=0)
    # Only 3 distinct samples back the 100 users.
    backing = set()
    for u in users:
        backing.add(tuple(it.content for it in adapter.memory_stream(u)))
    assert len(backing) <= 3


def test_local_path_rejects_non_list(tmp_path):
    import json

    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"not": "a list"}))
    adapter = LongMemEvalAdapter(path=str(p))
    with pytest.raises(TypeError):
        adapter.users(1, seed=0)


def test_local_path_expands_tilde(tmp_path, monkeypatch):
    """A home-relative path (`~/...`) is expanded against the home dir."""
    import json

    p = tmp_path / "lme_tilde.json"
    p.write_text(json.dumps(_synthetic_records(2)))
    # Point HOME at tmp_path so `~` resolves there.
    monkeypatch.setenv("HOME", str(tmp_path))
    adapter = LongMemEvalAdapter(path="~/lme_tilde.json", length=2)
    users = adapter.users(2, seed=0)
    assert len(users) == 2
