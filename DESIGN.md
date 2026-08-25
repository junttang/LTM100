# LTM100 — Multi-User Load Benchmark for Long-Term Memory Systems

> Status: **Implemented (baseline verified against a live MemMachine
> server).** This document is the design reference; the README is the fast
> entry point and `docs/` holds the per-topic detail.
> Last updated: 2026-08-25

## 1. Purpose

LTM100 is a benchmark for evaluating Long-Term Memory (LTM) software solutions
(e.g. MemMachine, Mem0) in a **server-client, multi-user** setting. Its core is
not retrieval/answer quality — it is **how an LTM server behaves when many users
repeatedly perform `add`, `search`, and `add&search` operations against a single
endpoint.**

The benchmark generates realistic and synthetic load, drives a running LTM
server through its client API, and extracts performance/load metrics
(latency, throughput, concurrency behavior). It is built to be **reusable and
publicly releasable**: datasets and LTM backends are pluggable, the run
configuration is simple, and defaults are sensible.

### What this benchmark measures

- Storage performance / throughput (`add`).
- Multi-user behavior under concurrency (`add`, `search`, mixed).
- Scalability / concurrency: QPS, latency percentiles (p50/p99), error rate.

### What this benchmark does NOT measure

- Retrieval / recall quality (no precision/recall/MRR/answer correctness).
- Memory isolation correctness as a dedicated test. Per-user scoping
  (`user_id` → backend tenant key) is assumed to be implemented correctly by
  each backend; isolation is an implicit property of correct per-user behavior,
  not a separate assertion.
- Server-side resource usage (CPU/mem/IO). Those are collected **by the
  server side separately**; LTM100 only collects client-observable metrics.

## 2. Goals & Non-Goals

**Goals**
- Pluggable datasets (LongMemEval first; BEAM, LoCoMo, others later).
- Pluggable LTM backends (MemMachine first; Mem0, others later).
- Pluggable transport (REST first; MCP later) under a unified backend adapter.
- Multiple load scenarios: pure load tests and realistic per-user patterns.
- Reproducible (seeded), with variance runs available.
- Easy to adopt: minimal config, clear plugin contracts, good docs.

**Non-Goals**
- Answer-quality evaluation.
- Client-side process/container isolation per user (server already isolates).
- Built-in server resource monitoring.

## 3. Architecture Overview

Three pluggable axes, plus a load core and a metrics recorder:

```
                 +------------------------+
                 |   Scenario (op mix +   |
                 |   data; closed/open    |
                 |   consume schedule)    |
                 +-----------+------------+
                             |
                             v
   +-------------------+   +------------------+   +-------------------+
   |  DatasetAdapter   |-->|   Load Core      |-->|  MetricsRecorder  |
   |  (per-user add    |   | (asyncio users)  |   | (per-request,      |
   |   + turn streams) |   |                  |   |  by op type)       |
   +-------------------+   +--------+---------+   +-------------------+
                                    |
                                    v
                          +-------------------+
                          |  LTMClient (async)|  <-- backend adapter
                          +---------+---------+
                                    |
                          +---------+---------+
                          |  Transport (REST /|
                          |  MCP / ...)       |
                          +-------------------+
                                    |
                                    v
                          [ Running LTM Server ]
```

- **DatasetAdapter**: turns a raw dataset into per-user streams of `add`
  payloads (and optional dialogue `turn_stream` for chat-replay). Knows
  nothing about backends or search queries — search queries are
  content-derived by the scenarios.
- **LTMClient** (backend adapter): async `add` / `search` against a specific
  backend, with per-user tenant scoping. Knows nothing about datasets or
  scenarios.
- **Transport**: the wire protocol under a backend adapter (REST first, MCP
  later). Backend adapters delegate to a transport.
- **Load Core**: orchestrates N virtual users (asyncio tasks), drives them
  through a Scenario, and records metrics.
- **Scenario**: defines *how* a user emits requests (inter-arrival, op mix,
  termination). Closed and open models share one runner.
- **MetricsRecorder**: collects per-request timing by op type; emits a
  summary + optional raw stream.

## 4. Pluggable Interfaces

All interface contracts below are the design intent; exact signatures are
finalized at implementation time.

### 4.1 DatasetAdapter

A dataset adapter exposes per-user workloads without leaking the raw dataset
shape into the core.

```python
class DatasetAdapter(Protocol):
    name: str

    def users(self, n_users: int, *, seed: int) -> list[UserId]:
        """Return n_users virtual-user identifiers (may replicate samples)."""

    def memory_stream(self, user: UserId) -> Iterator[MemoryItem]:
        """Yield add payloads for this user (in ingestion order)."""

    def turn_stream(self, user: UserId) -> Iterator[Turn]:
        """Optional: structured conversation turns, for chat-replay."""
```

- `MemoryItem`: `content` + optional `timestamp`, `producer`, `role`,
  `metadata`.
- `Turn`: `(role, items)` for dialogue datasets (optional; used by
  `chat-replay`).
- **No `query_stream`.** Search queries are **content-derived** by the
  scenarios — built from the user's own `memory_stream` items (for
  `search-load` / `mixed`) or from `turn_stream` user-turn content (for
  `chat-replay`). A dataset evaluation question is therefore not exposed as
  a search stream; this keeps the query pool large (one query per stored
  unit) so cycling does not naively repeat a single query and warm a server
  result cache.
- `QueryItem`: query string + optional expected fields (gold answer etc. are
  **not** scored — kept only for optional traceability/debugging).
- **Replication by default**: `n_users` is independent of dataset size; an
  adapter maps virtual users onto dataset samples (1:1 when enough samples,
  replication otherwise). The virtual-user count we drive is what matters,
  not the dataset's own user count.
- LongMemEval mapping: a sample's `haystack_sessions` → one user's
  `memory_stream` (flattened, chunked <=3000 chars) **and** `turn_stream`
  (per-turn role + chunked items, for chat-replay).

### 4.2 LTMClient (backend adapter)

```python
class LTMClient(Protocol):
    name: str

    async def setup(self, users: list[UserId]) -> None:
        """Per-run provisioning (e.g. create a project/tenant per user)."""

    async def add(self, user: UserId, items: list[MemoryItem]) -> list[str]:
        """Store memories for user; return backend ids."""

    async def search(self, user: UserId, query: QueryItem) -> list[ResultItem]:
        """Retrieve memories for user (scoped to this user only).

        The search depth (`top_k`) is carried on `query.top_k` (default 20,
        set via the scenario `top_k` param / `--top-k`), not as a separate
        argument."""

    async def teardown(self, users: list[UserId], *, delete: bool) -> None:
        """Optional cleanup (delete per-user state) per run."""
```

- Per-user scoping is the adapter's responsibility: it maps `UserId` to the
  backend's tenant key (MemMachine: `org_id`/`project_id` → `session_key`).
- The adapter is async; sync SDKs (e.g. Mem0) are wrapped via an executor.
- `setup`/`teardown` are out-of-measurement phases.

### 4.3 Transport (under a backend adapter)

```python
class Transport(Protocol):
    async def request(self, op: str, payload: dict) -> dict: ...
```

- REST is the first transport. MCP is a second transport with the **same**
  `LTMClient` contract; choosing it is a config switch, not a code fork.
- Backends that have only an SDK (no server) implement `LTMClient` directly.

## 5. Load Model

Hybrid: both **closed** and **open** models are supported, sharing one runner.

### 5.1 Closed model (load test) — implemented first

- Fixed `N` concurrent virtual users (asyncio tasks), each looping:
  emit request → await response → (optional think time) → next request.
- Per-user **in-flight = 1** by default (sequential within a user).
  Expandable to per-user in-flight > 1 later; behind a parameter.
- Optional **global concurrency cap** (`asyncio.Semaphore(K)`) to test "max
  concurrency K" load independent of N.

### 5.2 Open model (mixed) — implemented second

- Users arrive over time per an arrival process (Poisson by default, custom
  inter-arrival distributions pluggable).
- Each user performs a bounded number of ops then leaves.
- Concurrency is a *result* of arrival rate vs. service rate, not fixed.
- **Congestion policy** (decided at implementation): when arrival rate
  exceeds server capacity, define a bounded queue / rejection, and record
  rejections + queue depth as metrics. No silent dropping.

### 5.3 Shared runner

Both models reduce to "a virtual user coroutine emits requests over time";
the difference is the *emit schedule* (closed: think-time loop; open:
inter-arrival). The runner is the same; the Scenario provides the schedule.

See [`docs/load-models.md`](./docs/load-models.md) for the implemented
mechanics (closed loop, Poisson arrivals, the bounded-queue rejection
policy) and when to use each model.

## 6. Scenarios

A Scenario decides, per virtual user, the **op mix** and **emit schedule**.

```python
class Scenario(Protocol):
    name: str
    def plan(self, user: UserId, dataset: DatasetAdapter, rng_state: dict) -> Iterator[Op]:
        """Yield the sequence of ops for this user, in order. `rng_state`
        carries the seeded RNG state so schedules are reproducible."""
    def validate(self, dataset: DatasetAdapter) -> None: ...
```

Initial scenarios (`chat-replay` is the primary workload; the rest are
auxiliary load probes):
1. **`chat-replay`**: replay a chatbot-with-LTM workload over the dataset's
   `turn_stream` (recall before a user turn, then ingest the turn), with a
   configurable recall cadence (`search_every`) and LLM answer / user think
   time (`answer_time`, `user_gap`). The primary workload.
2. **`add-load`**: each user streams `memory_stream` back-to-back (wrapping),
   max concurrency. Pure storage throughput.
3. **`search-load`**: users run pre-ingested, content-derived searches
   forever. Pure search throughput/latency.
4. **`mixed`**: a controllable add/search mixture (op mix via
   `search_weight`), needs no `turn_stream` so it works with the synthetic
   dataset. Under the open model its per-session slice gives a quick
   congestion probe; under closed it loops like any other scenario.

Every scenario runs under **both** closed and open load models. The op mix
(add vs search) is owned by the Scenario plan for **both** models — the open
model's arriving sessions consume a bounded number of ops from the same
`plan()` interface the closed model loops over. There is no runner-level
op-mix weight; `mixed` takes a `search_weight` constructor param instead.
All scenario plans are **infinite** (they wrap their stream), so a duration
run sustains load instead of going idle when a finite stream is exhausted.

## 7. Metrics

All metrics are **client-observable** and **separated by op type**
(`add` vs `search`).

Per-request recorded fields: `op_type`, `user_id`, `started_at`, `ended_at`,
`latency_ms`, `status` (ok / error / rejected), `error_kind`, `bytes_in/out`
(optional).

Aggregated summary:
- Count per op type.
- Throughput (ops/s) per op type.
- QPS (total and per op type).
- Latency: mean, p50, p90, p95, p99, max per op type.
- Error rate (% and by kind).
- Concurrency (observed concurrent in-flight over time, for open model).

Recording:
- **Default**: in-memory per-request list, aggregated post-run. Good for
  small/medium scale and full reproducibility/debugging.
- **Optional**: NDJSON streaming (large scale), enabled via flag.
- p99 computed post-run (sort / numpy).

Server-side resource metrics are **not** collected here; the server exports
its own (e.g. Prometheus) and is scraped separately.

## 8. Run Lifecycle

1. **Load config** (YAML) — backend endpoint/auth, transport, adapter choices.
2. **Resolve adapters** — dataset + LTM client (+ transport).
3. **Provision** (`LTMClient.setup`) — per-user tenants created. (out of measure)
4. **Optional warm-up / pre-ingest** — fill each user's memories (a fraction
   of `memory_stream`) before the measured run; excluded from metrics. Enabled
   with `--preingest` and `--preingest-fraction`; applied under the global
   concurrency cap.
5. **Measured run** — scenario drives users; MetricsRecorder collects.
6. **Drain** — in-flight requests complete (or timeout).
7. **Aggregate & report** — summary JSON/CSV + optional raw NDJSON.
8. **Teardown** (`LTMClient.teardown`, `delete=True`) — optional per run;
   also exposed as a standalone cleanup command.

Termination: count-based (total K ops) **or** time-based (T seconds). Ramp-up
is optional; warm-up time is excluded from steady-state metrics.

## 9. Configuration

**YAML** (stable, per-environment): backend endpoint/auth, transport choice,
dataset adapter, LTM client adapter, defaults.

**CLI** (per-run, changed often): `--users N`, `--scenario`, `--duration` /
`--ops`, `--seed`, `--model`, `--global-concurrency`, `--warmup`, `--rampup`,
`--preingest`, `--preingest-fraction`, `--arrival-rate`, `--session-ops`,
`--queue-bound`, `--search-weight`, `--top-k`, `--think`, `--search-every`,
`--answer-time`, `--user-gap`, `--raw`, `--no-delete-on-exit`, `--output`.
See `ltm100 run --help` for the authoritative list.

## 10. Reproducibility

- Seeded RNG by default (`--seed`). Used for: virtual-user→sample mapping,
  op ordering, think time, inter-arrival, op mix.
- Different seeds produce variance runs; same seed + same config reproduces.
- Wall-clock timing is inherently non-deterministic (network/server); the
  *load shape* (order, mix, arrival) is deterministic under a seed.

## 11. Project Layout (proposed)

```
ltm100/
  ltm100/
    core/            # load core, runner, scenarios
    adapters/
      datasets/      # longmemeval.py, ...
      backends/       # memmachine.py, mem0.py, ...
      transports/     # rest.py, mcp.py
    metrics/         # recorder, aggregation, report
    config.py
    cli.py
  datasets/          # adapter-specific data access (not raw data)
  examples/          # sample configs + run commands
  docs/
  tests/
  README.md
  DESIGN.md          # this file
```

## 12. Initial Baseline (implemented)

First concrete adapters, end-to-end, **implemented and verified against a live
MemMachine server (v0.3.10)** via a smoke run:
- Dataset: **LongMemEval** (`xiaowu0162/longmemeval-cleaned`, `longmemeval_s_cleaned`),
  loadable from HuggingFace **or** from a pre-downloaded local JSON file
  (`path` option). A **Synthetic** adapter (`synthetic`) is also provided for
  fast, dependency-free load testing.
- Backend: **MemMachine** over **REST** (`/api/v2`), with
  `UserId → {org_id, project_id}` → `session_key = f"{org_id}/{project_id}"`,
  and a second transport over **MCP** (same `LTMClient` contract).
- Scenarios: `chat-replay` (primary), `add-load`, `search-load`, `mixed` —
  all run under both closed and open models; all plans wrap their streams.
  (`add-search-mixed` was retired early on — its `search_every` cadence
  moved to `chat-replay`.) Scenario params: `--think`, `--search-every`,
  `--search-weight`, `--top-k`, `--answer-time`, `--user-gap`.
- CLI: `ltm100 run`, `ltm100 cleanup`; reports: `summary.json`, `summary.csv`,
  optional `raw.ndjson`.

Verified: health, add/search, per-user isolation, count- and time-based
termination, global concurrency cap, reproducibility (same seed), report
generation, and cleanup (teardown delete). No code bugs found.

### Known limitation: backend `types` is hardcoded

The MemMachine adapter currently sends `types: ["episodic"]` for both add and
search (semantic memory excluded to avoid LLM-based background-processing
noise in load measurement). Making `types` configurable is a deferred TODO.

### Known caveat: MemMachine `projects/list` is eventually consistent

After `teardown(delete=True)` succeeds (server returns 204 and logs
`Deleted session`), an immediate `projects/list` call may still show the
project. It disappears shortly after. Cleanup verification must wait a moment
or check delete response status, not trust an immediate list.

## 13. Status of Open Questions

Resolved during implementation:
- Async signatures: `Protocol` for `LTMClient` / `DatasetAdapter` / `Transport`;
  concrete classes (`MemMachineClient`, `LongMemEvalAdapter`, `RestTransport`).
- Per-user in-flight = 1 is enforced in the runner; >1 is a future runner
  parameter (not Scenario-level).
- NDJSON raw format: per-request `{op_type, user_id, started_at, ended_at,
  latency_ms, status, error_kind, n_items}`.
- **Warm-up pre-ingest** (§8 step 4): the runner pre-ingests each user's
  memories (fraction configurable) before the measured run, under the global
  concurrency cap. Excluded from metrics.
- **Open model + congestion policy**: a Poisson arrival process spawns
  arriving sessions, each consuming a bounded number of ops from the Scenario
  plan (`mixed`). A bounded queue on the global concurrency cap rejects
  overload as `status="rejected"` (zero latency, `error_kind="queue_full"`).
  Op mix is owned by the Scenario plan (not a runner-level weight), so the
  open and closed models share one Scenario interface.
- **MCP transport**: a second transport under the same `LTMClient` contract
  (`MemMachineMcpClient` + `McpTransport` over `fastmcp`). The measured
  add/search ops call MemMachine's `add_memory`/`search_memory` MCP tools;
  lifecycle (project create/delete) stays on REST, since MCP has no
  project-management tools (hybrid lifecycle). Tenancy is passed as MCP tool
  arguments. The MCP `add_memory` writes all memory types (episodic + semantic),
  unlike the episodic-only REST add, so MCP add latency is not directly
  comparable to REST add latency; documented rather than worked around.
- **chat-replay LLM timing**: `chat-replay` models the LLM answer time
  (`answer_time`) and the user's think/typing time (`user_gap`) as
  Exponential-mean delays attached to specific ops, defaulting to 0
  (back-to-back). Search depth is configurable via `top_k` (default 20).
  All applied uniformly to every user for now.

Still open / next work (priority order):
1. **Mem0 backend adapter** — a second LTM solution under the `LTMClient`
   contract, to compare two LTM solutions on the same workload. Likely
   SDK-based (serverless), so a sync SDK wrapped via an executor is a design
   point to confirm.
2. **Per-user in-flight > 1** — currently fixed at 1 in the runner; make it a
   runner parameter so peak-concurrency measurement is not capped at N.
3. **Per-user-group finer control** — define user groups with their own
   `answer_time`/`user_gap`/`top_k` and a per-group user ratio, plus a
   per-user (or per-group) duration / "aggressiveness" knob. (Implement
   after the new timing/top_k params are validated to move load on a live
   server.)
4. **Configurable memory types** — replace the REST adapter's hardcoded
   episodic-only `types` with a config option (semantic adds LLM background
   processing load). The MCP transport is already all-types by the tool's
   design. Synergy with Mem0.
5. **Additional datasets** (BEAM, LoCoMo) via the `DatasetAdapter` extension
   (must implement `memory_stream`, and `turn_stream` if dialogue).
6. **Ramp-up / warm-up steady-state filtering** — the `warmup` field exists;
   verify steady-state metric exclusion at scale.

## 14. Glossary

- **Virtual user**: an asyncio task simulating one user, identified by a
  `UserId`; maps to a backend tenant.
- **Tenant key**: backend-specific per-user isolation key (MemMachine:
  `session_key`).
- **In-flight**: a request sent but not yet responded to.
- **Closed model**: fixed number of concurrent users, each looping.
- **Open model**: users arrive over time; concurrency is emergent.
- **Op**: a single `add` or `search` request.
