"""Tests for the synthetic dataset adapter."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict

import pytest

from ltm100.adapters.datasets.synthetic import SyntheticAdapter
from ltm100.config import build_dataset, load_config


def _corpus(seed, *, memories=4):
    ds = SyntheticAdapter(memories_per_user=memories, content_chars=80, categories=3)
    return {user: list(ds.memory_stream(user)) for user in ds.users(3, seed=seed)}


def test_seed_zero_preserves_existing_corpus_bytes():
    serialized = json.dumps(
        {user: [asdict(item) for item in items] for user, items in _corpus(0).items()},
        sort_keys=True,
    )
    assert hashlib.sha256(serialized.encode()).hexdigest() == (
        "bb66b5863c3100ef554d611d98a78df8ae6faba2d81c7a1466cbad619c7a7003"
    )


@pytest.mark.parametrize("seed", [1, 42, -7, 2**40 + 3])
def test_run_seed_changes_content_without_changing_identities(seed):
    zero = _corpus(0)
    actual = _corpus(seed)
    assert actual.keys() == zero.keys()
    for user in zero:
        assert [item.content for item in actual[user]] != [
            item.content for item in zero[user]
        ]
        assert [item.producer for item in actual[user]] == [
            item.producer for item in zero[user]
        ]
        assert [item.metadata for item in actual[user]] == [
            item.metadata for item in zero[user]
        ]


def test_seed_uses_full_integer_value():
    assert _corpus(0) != _corpus(2**32)
    assert _corpus(7) != _corpus(7 + 2**32)
    assert _corpus(7) != _corpus(-7)


@pytest.mark.parametrize("seed", [0, 42, -7])
def test_repeated_and_independent_adapters_are_reproducible(seed):
    expected = _corpus(seed)
    assert _corpus(seed) == expected
    ds = SyntheticAdapter(memories_per_user=4, content_chars=80, categories=3)
    users = ds.users(3, seed=seed)
    assert {user: list(ds.memory_stream(user)) for user in users} == expected
    assert {user: list(ds.memory_stream(user)) for user in reversed(users)} == expected


@pytest.mark.parametrize("seed", [0, 42, -7])
def test_corpus_growth_preserves_prefix(seed):
    small = _corpus(seed, memories=4)
    large = _corpus(seed, memories=9)
    for user in small:
        assert large[user][:4] == small[user]


def test_seed_zero_is_default_before_user_initialization():
    ds = SyntheticAdapter(memories_per_user=4, content_chars=80, categories=3)
    assert list(ds.memory_stream("syn_user_00000")) == _corpus(0)["syn_user_00000"]


def test_reseeding_does_not_mutate_an_already_started_stream():
    ds = SyntheticAdapter(memories_per_user=4, content_chars=80, categories=3)
    user = ds.users(1, seed=42)[0]
    stream = ds.memory_stream(user)
    first = next(stream)
    ds.users(1, seed=7)
    assert [first, *stream] == _corpus(42)[user]
    assert list(ds.memory_stream(user)) == _corpus(7)[user]
    ds.users(1)
    assert list(ds.memory_stream(user)) == _corpus(0)[user]


@pytest.mark.parametrize("seed", [0, 42, -7])
def test_separate_interpreters_ignore_python_hash_randomization(seed):
    script = """
import json
from dataclasses import asdict
from ltm100.adapters.datasets.synthetic import SyntheticAdapter
ds = SyntheticAdapter(memories_per_user=4, content_chars=80, categories=3)
users = ds.users(3, seed=SEED)
print(json.dumps({u: [asdict(i) for i in ds.memory_stream(u)] for u in users}, sort_keys=True))
""".replace("SEED", str(seed))
    expected = json.dumps(
        {
            user: [asdict(item) for item in items]
            for user, items in _corpus(seed).items()
        },
        sort_keys=True,
    )
    for hash_seed in ("1", "123"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        actual = subprocess.check_output(
            [sys.executable, "-c", script], env=env, text=True
        )
        assert actual.strip() == expected


def test_users_unique_and_counted():
    ds = SyntheticAdapter()
    users = ds.users(7, seed=0)
    assert len(users) == 7
    assert len(set(users)) == 7


def test_memory_stream_count():
    ds = SyntheticAdapter(memories_per_user=25)
    users = ds.users(1, seed=0)
    items = list(ds.memory_stream(users[0]))
    assert len(items) == 25
    assert all(it.producer == users[0] for it in items)


def test_reproducible_content_across_calls():
    ds = SyntheticAdapter(memories_per_user=10)
    users = ds.users(1, seed=0)
    a = [it.content for it in ds.memory_stream(users[0])]
    b = [it.content for it in ds.memory_stream(users[0])]
    assert a == b


def test_different_users_different_content():
    ds = SyntheticAdapter(memories_per_user=5)
    users = ds.users(2, seed=0)
    a = [it.content for it in ds.memory_stream(users[0])]
    b = [it.content for it in ds.memory_stream(users[1])]
    assert a != b


def test_registry_resolves_synthetic(tmp_path):
    import yaml

    cfg = {
        "dataset": {"name": "synthetic", "memories_per_user": 12},
        "backend": {"name": "memmachine", "base_url": "http://localhost:8080"},
    }
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    bc = load_config(str(p))
    ds = build_dataset(bc.dataset)
    assert isinstance(ds, SyntheticAdapter)
    assert ds.memories_per_user == 12
