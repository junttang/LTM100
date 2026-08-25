"""Tests for the synthetic dataset adapter."""

from __future__ import annotations

from ltm100.adapters.datasets.synthetic import SyntheticAdapter
from ltm100.config import build_dataset, load_config


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
