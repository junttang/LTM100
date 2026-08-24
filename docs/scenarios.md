# Scenarios

This document describes what each LTM100 scenario tests and exactly how a
virtual user behaves under it: how users are driven, how many run at once,
the concurrency model, and the `add`/`search` operations issued. Read it
alongside [`DESIGN.md`](../DESIGN.md) for the contracts and the
[README](../README.md) for copy-pasteable run commands.

LTM100 is backend-pluggable: scenarios talk to an `LTMClient` adapter, never
to a specific server. This document is therefore written at the adapter
contract level. Backend-specific details (MemMachine's REST endpoints,
tenancy key, request bodies) are isolated to the last section,
[MemMachine backend specifics](#memmachine-backend-specifics).

## Shared behavior

All scenarios share the same primitives. Understanding these first makes the
per-scenario differences small.

### Users and tenancy

A virtual user is **one asyncio coroutine** in a single process (not a
container or process). `--users N` spawns N coroutines, each identified by a
`UserId` string.

Per-user isolation is the **adapter's** responsibility: it maps a `UserId`
to whatever tenant key the backend uses, so that a user only ever searches
its own memories. The adapter's `setup` provisions per-user tenants and
`teardown` removes them.

When `--users N` exceeds the dataset's unique samples, samples are
replicated, so N is the driven virtual-user count, independent of dataset
size.

### Concurrency

Concurrency is controlled on two layers:

1. **Per-user in-flight = 1 (fixed).** Each user emits one request at a time
   and waits for the response before sending the next. There is never more
   than one in-flight request per user.
2. **`--global-concurrency C` (optional, 0 = no cap).** A global bound on the
   total number of in-flight requests across all users, enforced with an
   `asyncio.Semaphore`. Even with N users firing simultaneously, at most C
   requests execute at once.

### The add and search operations

A scenario emits `Op`s of two kinds, carried over the `LTMClient` contract:

- **ADD** — `client.add(user, items: list[MemoryItem]) -> list[str]`. Stores
  one or more memory units for the user and returns backend ids. Each
  `MemoryItem` carries a `content` string plus optional `producer`, `role`,
  `timestamp`, and `metadata`. The adapter decides how a batch of items maps
  to backend requests (it may batch them). One `add` op records the number of
  ids returned as `n_items`.
- **SEARCH** — `client.search(user, query: QueryItem) -> list[ResultItem]`.
  Retrieves memories scoped to this user only. The `QueryItem` carries a
  `query` string and a `top_k` (default 20). One `search` op records the
  number of results returned as `n_items`.

Per-request recording: `op_type`, `user_id`, `started_at`, `ended_at`,
`status` (ok / error / rejected), `error_kind`, `n_items`.

### Termination

Every run terminates by **either** `--duration SECONDS` or `--ops N`
(exactly one is required). The open model additionally requires `--duration`.

### Op mix ownership

The op mix (add vs search) is owned by the scenario for **both** load models.
There is no runner-level mix weight. In the open model, arriving sessions
consume ops from the same `Scenario.plan()` interface the closed model loops
over; the runner only decides *how many* ops each session takes.

---

## `add-load` (closed)

**Tests:** pure storage (ingest) throughput — how fast the backend stores
memories.

**User behavior:** each user iterates its `memory_stream` from start to end,
emitting one `Op(ADD, items=[item])` per item, back to back. When the stream
is exhausted the user stops (the plan is finite). No search is ever issued.

**Users:** `--users N`, N coroutines launched together.

**Concurrency:** per-user in-flight 1; N users add in parallel, so
simultaneous `add` calls = `min(N, C)`. With no global cap, N concurrent adds.

**add:** back-to-back, `delay=0`. One op per memory item. Synthetic yields
`memories_per_user` (default 100) items; LongMemEval yields one add per
haystack session.

**search:** none.

**Termination:** `--duration` (time bound when memories are large) or
`--ops`. Users that finish early simply exit.

---

## `search-load` (closed)

**Tests:** pure search throughput and latency — read-path load against
pre-populated memory.

**User behavior:** each user loops its `query_stream` repeatedly (capped at
10,000 iterations), emitting SEARCH ops. No `add` during the measured run,
so **memory must already be present** (use `--preingest`).

**Users:** `--users N`.

**Concurrency:** per-user in-flight 1; users loop forever, so only
`--duration`/`--ops` terminates the run.

**search:** each query carries a small think time
(`delay = uniform(0, 0.02)`) so users drift out of lockstep and do not all
fire the same query simultaneously. Queries cycle through the user's
`query_stream`. `top_k` from the `QueryItem` (default 20).

**add:** none during measurement.

**Pre-ingest:** essential here. `--preingest` fills each user's memories
before the measured run, under the global concurrency cap, ingesting a
`--preingest-fraction` (default 1.0 = all) of each user's `memory_stream`.
Pre-ingest is excluded from metrics.

**Termination:** `--duration` or `--ops` (users do not self-terminate).

---

## `add-search-mixed` (closed)

**Tests:** mixed workload — a single user's lifetime interleaving ingestion
with retrieval, mirroring a user who stores memories and occasionally
recalls them.

**User behavior:** each user walks its `memory_stream`, adding items in
batches of `add_batch` (default 1). After every `search_every` (default 20)
adds, it issues one search from its `query_stream` (cycled). If the query
stream is empty, this degenerates to `add-load`.

**Users:** `--users N`.

**Concurrency:** per-user in-flight 1; the add-20-then-search-1 pattern
progresses per user under a fixed seed.

**add:** items batched `add_batch` per op (default 1 → one item per op),
`delay=0`.
**search:** one every `search_every` adds, query cycled from
`query_stream`, `delay=0`, `top_k` default 20.

**Parameters:** constructor `search_every=20`, `add_batch=1`. These are not
currently exposed on the CLI (defaults are used); they can be surfaced if
needed.

**Pre-ingest:** not needed — the user adds its own memories and searches
against them as it goes.

**Termination:** the user stops when its `memory_stream` is exhausted
(finite plan), or on `--duration`/`--ops`.

---

## `realistic` (open)

**Tests:** arrival-driven load — real traffic is not N fixed looping users
but users arriving and leaving over time. This scenario exercises emergent
concurrency and the congestion/rejection policy under overload.

This is the only scenario intended for the **open** model. The runner, with
`--model open`, consumes this scenario's plan.

**User behavior — two-stage:**

1. **Arrival process (runner-owned, not scenario).** A Poisson process spawns
   sessions at `--arrival-rate λ`. Inter-arrival is `expovariate(λ)`. Each
   arrival draws the next user round-robin from the `--users N` pool (so the
   tenant already exists) and starts one **session**. Sessions keep arriving
   until the deadline.

2. **Session (consumes the scenario plan).** An arriving user consumes up to
   `--session-ops` ops from `self.scenario.plan(...)`, then leaves. The op
   mix is decided by the scenario:
   - `rng.random() < search_weight` (default 0.8) → SEARCH (query cycled)
   - else → ADD (one item from `memory_stream`, cycled when exhausted)
   - each op carries think jitter (`delay = uniform(0, think)`, `think`
     default 0.05)

**Users:** `--users N` is the **tenant identity pool, not the concurrent
count**. Concurrency is emergent — a function of arrival rate vs service
rate. The same user may appear in multiple concurrent sessions (same tenant,
isolation preserved).

**Concurrency (congestion policy — the core output):**
- `--global-concurrency C`: max in-flight.
- `--queue-bound Q`: how many requests beyond C may wait in queue.
- When in-flight reaches `C + Q`, the next request is **rejected**
  (`status=rejected`, `error_kind=queue_full`, zero latency) — it is recorded
  but not executed.
- `Q=0` rejects immediately on C saturation. `Q>0` lets up to Q requests
  queue (busy-wait in 0.005s steps) before acquiring a slot.

**add:** when not a search, one item per op, `delay = uniform(0, 0.05)`.
**search:** query cycled, `top_k` default 20, `delay = uniform(0, 0.05)`.

**Parameters:** `--search-weight` (default 0.8, forwarded to the scenario
constructor); `think` default 0.05 (not currently CLI-exposed). The open-model
knobs `--arrival-rate`, `--session-ops`, `--queue-bound` live on `RunConfig`.

**Pre-ingest:** recommended — arriving users search against memory that
should already exist.

**Termination:** `--duration` is required (open model enforces `duration > 0`).
Sessions arrive until the deadline, then in-flight sessions drain.

---

## Comparison

| | add-load | search-load | add-search-mixed | realistic |
|---|---|---|---|---|
| load model | closed | closed | closed | open |
| ops | add only | search only | add + search interleaved | search-weighted add + search |
| user lifetime | finite (stream exhausted) | infinite (loop) | finite (stream exhausted) | per-session (arrival → `session_ops`) |
| concurrency | N fixed, parallel add | N fixed, parallel search | N fixed, mixed | emergent (Poisson arrivals) |
| precondition | none | preingest required | none | preingest recommended |
| termination | duration/ops | duration/ops | duration/ops | duration required |
| op scheduling | back-to-back | think 0–0.02s | back-to-back | think 0–0.05s |
| key output | write throughput | read latency | read/write mix | rejection/congestion metrics |

---

## MemMachine backend specifics

The above describes behavior at the `LTMClient` contract level. When the
backend is **MemMachine over REST** (the current baseline adapter,
`ltm100/adapters/backends/memmachine.py`), the contract maps to concrete
requests as follows. This is the only backend-specific section; a future
adapter (e.g. Mem0) would map the same contract differently.

**Tenancy.** MemMachine's multi-tenancy is
`session_key = f"{org_id}/{project_id}"`. The adapter maps a user to a single
org and one project per user:

```
session_key = f"{org_prefix}/user_{UserId}"
```

So **one user = one MemMachine project** under a shared org. `setup` creates
one project per user (409 "already exists" is tolerated for reruns);
`teardown` deletes them.

**add** maps to `POST /api/v2/memories`:

```
org_id, project_id              # from the user's session_key
types: ["episodic"]             # episodic only (hardcoded; semantic is a future option)
messages: [{ content, producer, role?, timestamp?, metadata? }]
```

`MemoryItem.content` is passed through; `producer` is the `UserId`; `role`,
`timestamp`, `metadata` are forwarded if present (metadata values are
stringified). The adapter chunks `items` into batches of `add_batch_size`
(YAML, default 50) per request. Most scenarios pass one item per op, so one
op typically becomes one request.

**search** maps to `POST /api/v2/memories/search`:

```
org_id, project_id
query: <string>
top_k: 20                      # from the QueryItem
types: ["episodic"]
```

The response's `content.episodic_memory.long_term_memory.episodes` is parsed
into `ResultItem`s (`content`, `score`, `uid`, `metadata`).

**Endpoints used:**

```
POST /api/v2/projects          create a per-user project (setup)
POST /api/v2/projects/delete   delete a project (teardown)
POST /api/v2/memories          add memories
POST /api/v2/memories/search   search memories
GET  /api/v2/health            readiness check
```

> Note: MemMachine's `projects/list` is eventually consistent — an immediate
> list after delete may still show a project before it disappears. The
> delete itself is confirmed by the server's response, not by listing.
