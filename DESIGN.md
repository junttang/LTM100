# LTM100 — Multi-User Load Benchmark for Long-Term Memory Systems

> Status: **Design draft (no implementation yet).**
> Last updated: 2026-08-24

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
                 |   Scenario (closed /   |
                 |   open / mixed)        |
                 +-----------+------------+
                             |
                             v
   +-------------------+   +------------------+   +-------------------+
   |  DatasetAdapter   |-->|   Load Core      |-->|  MetricsRecorder  |
   |  (per-user add +  |   | (asyncio users)  |   | (per-request,      |
   |   query streams)  |   |                  |   |  by op type)       |
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
  payloads and `search` queries. Knows nothing about backends.
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

    def query_stream(self, user: UserId) -> Iterator[QueryItem]:
        """Yield search queries for this user (may repeat / interleave)."""
```

- `MemoryItem`: content + optional metadata (timestamp, producer, role).
- `QueryItem`: query string + optional expected fields (gold answer etc. are
  **not** scored — kept only for optional traceability/debugging).
- **Replication by default**: `n_users` is independent of dataset size; an
  adapter maps virtual users onto dataset samples (1:1 when enough samples,
  replication otherwise). The virtual-user count we drive is what matters,
  not the dataset's own user count.
- LongMemEval mapping: a sample's `haystack_sessions` → one user's
  `memory_stream`; the sample's `question/answer` → that user's
  `query_stream`.

### 4.2 LTMClient (backend adapter)

```python
class LTMClient(Protocol):
    name: str

    async def setup(self, users: list[UserId]) -> None:
        """Per-run provisioning (e.g. create a project/tenant per user)."""

    async def add(self, user: UserId, items: list[MemoryItem]) -> list[str]:
        """Store memories for user; return backend ids."""

    async def search(self, user: UserId, query: QueryItem, top_k: int) -> list[ResultItem]:
        """Retrieve memories for user (scoped to this user only)."""

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

### 5.2 Open model (realistic) — implemented second

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

## 6. Scenarios

A Scenario decides, per virtual user, the **op mix** and **emit schedule**.

```python
class Scenario(Protocol):
    name: str
    def plan(self, user: UserId, dataset: DatasetAdapter, rng: Random) -> Iterator[Op]:
        """Yield (op_type, payload, think_or_interarrival) for this user."""
```

Initial scenarios (easiest first):
1. **`add-load`**: each user streams `memory_stream` back-to-back, max
   concurrency. Pure storage throughput.
2. **`search-load`**: users run pre-ingested `query_stream` repeatedly. Pure
   search throughput/latency.
3. **`add-search-mixed`**: interleaved add & search per user.
4. **`realistic`** (open model): per-user inter-arrival, bounded session
   length, op mix weighted toward search with occasional adds.

Scenarios 1–3 use the closed model; scenario 4 uses the open model.

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
4. **Optional warm-up / pre-ingest** — fill memories; excluded from metrics.
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
`--ops`, `--seed`, `--global-concurrency`, `--warmup`, `--ramp-up`,
`--stream-metrics`, `--output`, `--delete-on-exit`.

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

## 12. Initial Baseline

First concrete adapters, end-to-end:
- Dataset: **LongMemEval** (`xiaowu0162/longmemeval-cleaned`, `longmemeval_s_cleaned`).
- Backend: **MemMachine** over **REST** (`/api/v2`), with
  `UserId → {org_id, project_id}` → `session_key = f"{org_id}/{project_id}"`.
- Scenario: `add-load` → `search-load` → `add-search-mixed`.

## 13. Open Questions (deferred to implementation)

- Exact async signatures of `LTMClient` / `DatasetAdapter` (Protocol vs ABC).
- Open-model congestion policy details (queue bound, backpressure, rejection
  semantics) and how rejections surface in metrics.
- Whether per-user in-flight > 1 is parameterized on the Scenario or the
  runner.
- NDJSON streaming format and live-progress reporting shape.
- Whether MCP transport needs a separate `Transport` impl or a thin client
  wrapper (depends on the MCP SDK's request model at implementation time).

## 14. Glossary

- **Virtual user**: an asyncio task simulating one user, identified by a
  `UserId`; maps to a backend tenant.
- **Tenant key**: backend-specific per-user isolation key (MemMachine:
  `session_key`).
- **In-flight**: a request sent but not yet responded to.
- **Closed model**: fixed number of concurrent users, each looping.
- **Open model**: users arrive over time; concurrency is emergent.
- **Op**: a single `add` or `search` request.
