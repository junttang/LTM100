"""Tests for the MemVerge extensions: sharding, rungs, env expansion, items."""

from __future__ import annotations

import itertools
import os

import pytest

from ltm100.adapters.backends.memmachine import MemMachineClient
from ltm100.adapters.datasets.synthetic import SyntheticAdapter
from ltm100.core import scenarios
from ltm100.core.config import RunConfig
from ltm100.core.scenarios import AddLoad, Mixed, SearchLoad
from ltm100.core.op import OpResult, OpType
from ltm100.metrics.aggregate import aggregate


class _Shardable:
    """Minimal stand-in exposing shard_users against a given config."""

    def __init__(self, cfg):
        self.config = cfg

    from ltm100.core.runner import LoadRunner as _LR

    shard_users = _LR.shard_users


def test_single_proc_is_identity():
    users = [f"u{i}" for i in range(10)]
    cfg = RunConfig(users=10, duration=1.0, procs=1)
    assert _Shardable(cfg).shard_users(users) == users


def test_shards_partition_users_exactly_once():
    users = [f"u{i}" for i in range(12)]
    procs = 4
    seen: list[str] = []
    for i in range(procs):
        cfg = RunConfig(users=12, duration=1.0, procs=procs, proc_index=i)
        shard = _Shardable(cfg).shard_users(users)
        assert shard, "every shard must get work"
        seen.extend(shard)
    assert sorted(seen) == sorted(users)
    assert len(seen) == len(set(seen))


def test_procs_may_not_exceed_users():
    with pytest.raises(ValueError, match="exceeds users"):
        RunConfig(users=2, duration=1.0, procs=4)


def test_proc_index_must_be_in_range():
    with pytest.raises(ValueError, match="proc_index"):
        RunConfig(users=8, duration=1.0, procs=2, proc_index=2)







def _result(op: OpType, n_items: int, status: str = "ok") -> OpResult:
    return OpResult(
        type=op, user_id="u", started_at=0.0, ended_at=0.1,
        status=status, n_items=n_items,
    )


def test_empty_rate_separates_working_search_from_silent_search():
    all_empty = aggregate([_result(OpType.SEARCH, 0) for _ in range(4)])
    assert all_empty["error_rate"] == 0.0
    assert all_empty["by_op"]["search"]["items"]["empty_rate"] == 1.0

    productive = aggregate([_result(OpType.SEARCH, 5) for _ in range(4)])
    assert productive["by_op"]["search"]["items"]["empty_rate"] == 0.0
    assert productive["by_op"]["search"]["items"]["mean"] == 5.0


def test_errored_ops_are_excluded_from_empty_rate():
    mixed = aggregate([
        _result(OpType.SEARCH, 5),
        _result(OpType.SEARCH, 0, status="error"),
    ])
    # the errored op has no result count to speak of; it must not read as empty
    assert mixed["by_op"]["search"]["items"]["empty_rate"] == 0.0
    assert mixed["by_op"]["search"]["errors"] == 1


# -- the build recorded in the report ---------------------------------------

class _Cfg:
    def __init__(self, name="memmachine", options=None):
        self.name = name
        self.options = options or {}


class _Bundle:
    def __init__(self, backend):
        self.backend = backend


def _patch_backend(monkeypatch, factory):
    import ltm100.cli as cli
    monkeypatch.setattr(cli, "build_backend", lambda cfg: factory())


def test_build_version_is_recorded(monkeypatch):
    from ltm100.cli import _backend_build

    class Healthy:
        async def __aenter__(self): return self
        async def __aexit__(self, *e): return None
        async def health(self):
            return {"status": "healthy", "service": "memmachine", "version": "0.3.9.post1"}

    _patch_backend(monkeypatch, Healthy)
    assert _backend_build(_Bundle(_Cfg())) == {
        "build": "0.3.9.post1", "service": "memmachine",
    }


def test_an_unstamped_build_is_reported_as_such(monkeypatch):
    from ltm100.cli import _backend_build

    class Unstamped:
        async def __aenter__(self): return self
        async def __aexit__(self, *e): return None
        async def health(self):
            return {"service": "memmachine", "version": "0.0.0"}

    _patch_backend(monkeypatch, Unstamped)
    # 0.0.0 is a real answer and must survive verbatim: it is the signal that an
    # image was built without SCM_VERSION, not a missing value to paper over.
    assert _backend_build(_Bundle(_Cfg()))["build"] == "0.0.0"


def test_a_failed_probe_does_not_break_the_report(monkeypatch):
    from ltm100.cli import _backend_build

    class Unreachable:
        async def __aenter__(self): return self
        async def __aexit__(self, *e): return None
        async def health(self):
            raise ConnectionError("refused")

    _patch_backend(monkeypatch, Unreachable)
    assert "unavailable" in _backend_build(_Bundle(_Cfg()))["build"]


def test_a_backend_without_health_is_simply_omitted(monkeypatch):
    from ltm100.cli import _backend_build

    class NoHealth:
        async def __aenter__(self): return self
        async def __aexit__(self, *e): return None

    _patch_backend(monkeypatch, NoHealth)
    assert _backend_build(_Bundle(_Cfg())) == {}


def test_the_report_meta_carries_the_build(tmp_path, monkeypatch, capsys):
    """The probe is only useful if _run actually puts it in the report."""
    import json as _json
    import yaml as _yaml

    import ltm100.cli as cli

    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml.safe_dump({
        "dataset": {"name": "synthetic", "length": 2},
        "backend": {"name": "memmachine", "base_url": "http://localhost:8080"},
    }))

    monkeypatch.setattr(cli, "_backend_build",
                        lambda cfg: {"build": "9.9.9+test", "service": "memmachine"})
    monkeypatch.setattr(cli, "run_shards", lambda entry, args, procs: [])

    args = cli.build_parser().parse_args([
        "run", "--config", str(cfg_path), "--scenario", "chat-replay",
        "--users", "2", "--duration", "1",
    ])
    assert cli._run(args) == 0

    meta = _json.loads(capsys.readouterr().out)["meta"]
    assert meta["build"] == "9.9.9+test"
    assert meta["service"] == "memmachine"


# -- whole-run load is divided across shards --------------------------------

def test_shares_sum_to_the_total():
    from ltm100.cli import _split
    for total, procs in ((100, 4), (10, 4), (7, 3), (0, 4), (1, 4)):
        parts = [_split(total, procs, i) for i in range(procs)]
        assert sum(parts) == total, (total, procs, parts)


def test_single_process_is_unchanged():
    from ltm100.cli import _split
    assert _split(100, 1, 0) == 100


def test_sharding_divides_concurrency_rate_ops_and_queue():
    """Each shard runs its own runner, so an undivided budget applies N times."""
    import ltm100.cli as cli

    args = cli.build_parser().parse_args([
        "run", "--config", "x", "--scenario", "chat-replay",
        "--users", "24", "--duration", "10", "--procs", "4",
        "--global-concurrency", "100",
        "--model", "open", "--arrival-rate", "40", "--session-ops", "12",
        "--queue-bound", "20",
    ])
    shares = []
    for i in range(4):
        args.proc_index = i
        cfg = cli._build_run_config(args)
        shares.append(cfg)
    assert sum(c.global_concurrency for c in shares) == 100
    assert sum(c.queue_bound for c in shares) == 20
    assert abs(sum(c.arrival_rate for c in shares) - 40) < 1e-9
    # per-session, not per-run: must NOT be divided
    assert all(c.session_ops == 12 for c in shares)
    # users are partitioned by shard_users, so the config keeps the full count
    assert all(c.users == 24 for c in shares)


def test_length_zero_yields_no_samples(tmp_path):
    """The streaming path and the json.load fallback must agree."""
    import json
    from ltm100.adapters.datasets.longmemeval import LongMemEvalAdapter

    p = tmp_path / "d.json"
    p.write_text(json.dumps([{"haystack_sessions": [[{"role": "user", "content": "a"}]],
                              "haystack_session_ids": ["s1"]}] * 3))
    assert LongMemEvalAdapter(path=str(p), length=0)._load() == []


def test_sharding_divides_a_count_based_budget():
    """--ops caps the whole run, so each shard gets a share of it."""
    import ltm100.cli as cli

    args = cli.build_parser().parse_args([
        "run", "--config", "x", "--scenario", "chat-replay",
        "--users", "8", "--ops", "1000", "--procs", "3",
    ])
    total = 0
    for i in range(3):
        args.proc_index = i
        total += cli._build_run_config(args).ops
    assert total == 1000


# -- server-side search knobs: expand_context and filter ---------------------

def _captured_search_payload(**query_kwargs):
    """Run MemMachineClient.search against a stub transport, return the payload."""
    import asyncio
    from ltm100.adapters.backends.memmachine import MemMachineClient
    from ltm100.common import QueryItem

    seen = {}

    class _Stub:
        headers: dict = {}
        async def request(self, method, path, json=None, params=None):
            seen["path"] = path
            seen["payload"] = json
            return {"content": {"episodic_memory": {"long_term_memory": {"episodes": []}}}}

    c = MemMachineClient()
    c._transport = _Stub()
    asyncio.run(c.search("u0", QueryItem(query="q", **query_kwargs)))
    return seen["payload"]


def test_defaults_omit_both_knobs():
    """A default run's payload must be unchanged by this feature."""
    p = _captured_search_payload()
    assert "expand_context" not in p
    assert "filter" not in p
    assert set(p) == {"org_id", "project_id", "query", "top_k", "types"}


def test_expand_context_is_sent_when_set():
    p = _captured_search_payload(expand_context=3)
    assert p["expand_context"] == 3


def test_filter_is_sent_verbatim_when_set():
    # the platform's filter language is metadata.<field>=<value>, exact match
    p = _captured_search_payload(filter="metadata.category=cat_3")
    assert p["filter"] == "metadata.category=cat_3"
    assert "expand_context" not in p


def test_zero_expand_is_treated_as_off():
    assert "expand_context" not in _captured_search_payload(expand_context=0)


def test_scenarios_propagate_the_knobs_to_every_query():
    """A knob set on the CLI is useless if the scenario drops it."""
    from ltm100.core.scenarios import get_scenario
    from ltm100.adapters.datasets.synthetic import SyntheticAdapter

    ds = SyntheticAdapter(memories_per_user=6)
    for name in ("search-load", "mixed"):
        s = get_scenario(name, expand_context=2, filter="metadata.category=cat_1")
        # plan() loops forever to sustain load; islice or it eats all memory
        ops = list(itertools.islice(s.plan("u0", ds, {"seed": 0}), 12))
        queries = [o.query for o in ops if getattr(o, "query", None) is not None]
        assert queries, f"{name} produced no queries"
        assert all(q.expand_context == 2 for q in queries), name
        assert all(q.filter == "metadata.category=cat_1" for q in queries), name


def test_negative_expand_is_rejected():
    from ltm100.core.scenarios import get_scenario
    for name in ("search-load", "mixed", "chat-replay"):
        with pytest.raises(ValueError, match="expand_context"):
            get_scenario(name, expand_context=-1)


def test_synthetic_categories_give_a_field_to_filter_on():
    from ltm100.adapters.datasets.synthetic import SyntheticAdapter

    off = list(SyntheticAdapter(memories_per_user=10).memory_stream("u0"))
    assert all(m.metadata == {} for m in off), "must be inert when unset"

    on = list(SyntheticAdapter(memories_per_user=100, categories=10).memory_stream("u0"))
    cats = [m.metadata["category"] for m in on]
    assert len(set(cats)) == 10
    assert cats.count("cat_3") == 10          # one value selects ~1/N


def test_mcp_refuses_the_knobs_it_cannot_honour():
    """Silently ignoring them would label a baseline search as filtered."""
    import asyncio
    from ltm100.adapters.backends.memmachine_mcp import MemMachineMcpClient
    from ltm100.common import QueryItem

    c = MemMachineMcpClient.__new__(MemMachineMcpClient)
    c.org_prefix = "t"
    for q in (QueryItem(query="x", expand_context=2),
              QueryItem(query="x", filter="metadata.category=cat_3")):
        with pytest.raises(ValueError, match="MCP backend supports neither"):
            asyncio.run(c.search("u0", q))


def test_mcp_refuses_metadata_it_would_drop():
    import asyncio
    from ltm100.adapters.backends.memmachine_mcp import MemMachineMcpClient
    from ltm100.common import MemoryItem

    c = MemMachineMcpClient.__new__(MemMachineMcpClient)
    c.org_prefix = "t"
    with pytest.raises(ValueError, match="cannot store item metadata"):
        asyncio.run(c.add("u0", [MemoryItem(content="a", metadata={"category": "cat_1"})]))


class _CountingSynthetic(SyntheticAdapter):
    """Counts how many times a plan materializes the user's stream."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.builds = 0

    def memory_stream(self, user):
        self.builds += 1
        return super().memory_stream(user)


def _plan_ops(scenario, dataset, n):
    return [
        (op.type, tuple(getattr(i, "query", getattr(i, "content", None)) for i in op.items))
        for op in itertools.islice(scenario.plan("syn_user_00000", dataset, {"seed": 0}), n)
    ]


@pytest.mark.parametrize(
    "make_scenario",
    [lambda: SearchLoad(top_k=20), lambda: AddLoad(), lambda: Mixed()],
    ids=["search-load", "add-load", "mixed"],
)
def test_plans_build_the_corpus_once_per_user_not_once_per_session(make_scenario):
    """Under --model open a session opens per arrival, and planning runs on the
    event loop; rebuilding the corpus each time starves the loop and wedges the
    run. Guards the cache in _memories/_query_pool."""
    scenarios._MEMO.clear()
    dataset = _CountingSynthetic(memories_per_user=200)
    scenario = make_scenario()
    for _ in range(5):
        _plan_ops(scenario, dataset, 3)
    assert dataset.builds == 1, f"corpus rebuilt {dataset.builds} times across 5 sessions"


@pytest.mark.parametrize(
    "make_scenario",
    [lambda: SearchLoad(top_k=20), lambda: AddLoad(), lambda: Mixed()],
    ids=["search-load", "add-load", "mixed"],
)
def test_caching_does_not_change_the_plan(make_scenario):
    """The cache must be invisible: a cached run and an uncached one have to
    yield the same ops, or it would silently alter the workload."""
    scenarios._MEMO.clear()
    cached = _plan_ops(make_scenario(), SyntheticAdapter(memories_per_user=200), 50)

    real_memories, real_pool = scenarios._memories, scenarios._query_pool
    try:
        scenarios._memories = lambda d, u: list(d.memory_stream(u))
        scenarios._query_pool = lambda sc, d, u: scenarios._content_queries(
            list(d.memory_stream(u)), sc.top_k, sc.expand_context, sc.filter
        )
        uncached = _plan_ops(make_scenario(), SyntheticAdapter(memories_per_user=200), 50)
    finally:
        scenarios._memories, scenarios._query_pool = real_memories, real_pool

    assert cached == uncached
