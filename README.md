# LTM100

LTM100 is a multi-user **load benchmark** for Long-Term Memory (LTM) systems
(e.g. MemMachine, Mem0). It drives many virtual users performing `add`,
`search`, and `add&search` operations against a single LTM endpoint and
reports client-observable performance metrics: throughput, QPS, latency
percentiles, and error rate, all split by operation type.

This benchmark measures **load, concurrency, and scalability behavior** —
not retrieval or answer quality. There are no precision/recall/MRR metrics.
Server-side resource metrics (CPU, memory, etc.) are collected separately by
the server itself.

For the full design, see [`DESIGN.md`](./DESIGN.md). Topic docs live under
[`docs/`](./docs/): [`docs/scenarios.md`](./docs/scenarios.md) for the
per-scenario data flow, [`docs/load-models.md`](./docs/load-models.md) for
the closed/open load models and the congestion policy.

## Status

Version: **v0.3.1** (see [Versioning](#versioning)).

Early development. Datasets and LTM backends are pluggable; the initial
baseline is the LongMemEval dataset + the MemMachine backend over REST.

Implemented:
- Closed and open load models (every scenario runs under both; see below).
- Scenarios: `chat-replay` (the primary workload), `add-load`, `search-load`,
  `mixed`. All wrap their data streams to sustain load for a duration run,
  and all search on content-derived queries (from the user's own memories /
  conversation turns).
- Congestion policy (bounded queue with rejection) for the open model.
- Warm-up / pre-ingest before the measured run.
- Datasets: LongMemEval (local file or HuggingFace), synthetic.
- Backend: MemMachine (REST and MCP transports).
- Configurable scenario parameters on the CLI (`--think`, `--search-every`,
  `--search-weight`).
- Reports: summary JSON/CSV + optional raw NDJSON.

Planned:
- Mem0 backend adapter.
- Additional datasets (BEAM, LoCoMo).

## Install

```sh
pip install -e ".[dev]"
# To use the LongMemEval adapter via HuggingFace also:
pip install -e ".[datasets]"
# To use the MemMachine MCP transport also:
pip install -e ".[mcp]"
```

The `ltm100` CLI is the entry point.

## Configuration

A run takes two inputs:

- A **YAML config file** (stable, per environment): backend endpoint/auth,
  transport, and the dataset and backend adapter choices. See
  `examples/memmachine.yaml` and `examples/synthetic.yaml`.
- **CLI flags** (per run): number of users, scenario, duration/ops, seed,
  load model, concurrency, warm-up, and output. See `ltm100 run --help`.

Edit `examples/*.yaml` to point at your LTM server (`backend.base_url`) and
pick a dataset.

## Datasets

- **`longmemeval`** — the LongMemEval-cleaned dataset. One sample's
  `haystack_sessions` becomes a user's add stream; the same session structure
  also drives `chat-replay`'s recall (user/assistant turns). Loads from a
  local JSON file (`path:`) or, if `path` is omitted, downloads the split
  from HuggingFace. `length:` caps the number of samples.
- **`synthetic`** — deterministically generated per-user content from the
  seed. No download; ideal for fast, reproducible load tests. Tunable via
  `memories_per_user`, `content_chars`. (No conversation turns, so
  `chat-replay` is not usable with it.)

When the virtual-user count exceeds the dataset's unique samples, samples
are replicated so N is the driven user count, independent of dataset size.

### What gets added and searched

A scenario decides *when* and *how often* to issue add/search, but **not what
content** — that comes from the dataset adapter:

- **add** operates on `dataset.memory_stream(user)`: LongMemEval turns the
  sample's haystack into <=3000-char chunks (one turn may yield several
  items); synthetic yields a fixed number of deterministic items. Items are
  stored as **episodic** memory (`producer` = the user id).
- **search** queries are **content-derived**, not taken from a dataset
  evaluation question. `search-load` and `mixed` build the query pool
  from the user's own `memory_stream` items (one query per stored unit), so
  the pool is large and cycling it does not naively repeat a single query
  (which would warm a server result cache and understate latency). They
  cycle the pool with a rotating per-pass offset and a small think jitter so
  users drift out of lockstep. `chat-replay` instead derives each recall
  query from the upcoming user turn's content via the dataset's
  `turn_stream(user)` (LongMemEval only), so recall is driven by the
  conversation itself. LTM100 measures load, not recall — there are no
  precision/recall metrics and gold `expected` fields, if any, are carried
  for tracing only and never scored.

See [`docs/scenarios.md`](./docs/scenarios.md) for the full per-scenario
data-flow detail.

## Scenarios

A scenario turns each user's dataset streams into a sequence of operations.
The op mix (add vs search) is owned by the scenario for both load models.
**Every scenario runs under both load models** — closed (a fixed pool of
looping users) and open (Poisson arrivals with a congestion/rejection
policy). The scenario owns the op mix and data; the runner owns the consume
schedule (think-time loop vs arrival-driven sessions). See [Load
models](#load-models).

| | chat-replay | add-load | search-load | mixed |
| --- | --- | --- | --- | --- |
| load model | closed + open | closed + open | closed + open | closed + open |
| ops | recall + add per turn | add only | search only | search-weighted add + search |
| user lifetime | wraps the dialogue | wraps to sustain | wraps to sustain | per-session (arrival → `session_ops`) |
| concurrency | N fixed, in-order replay | N fixed, parallel add | N fixed, parallel search | emergent under open; N fixed under closed |
| precondition | dataset with `turn_stream` | none | preingest required | preingest recommended |
| termination | duration/ops | duration/ops | duration/ops | duration required (open) |
| op scheduling | think 0–0.05s | back-to-back | think 0–0.02s | think 0–0.05s |
| key output | chatbot-LTM integration load | write throughput | read latency | rejection/congestion metrics (open) |

For a detailed, per-scenario walkthrough — exactly how a virtual user
behaves, how many run, the concurrency model, and the `add`/`search`
operations at the backend-contract level (with MemMachine specifics
isolated to one section) — see [`docs/scenarios.md`](./docs/scenarios.md).

## Load models

A **load model** is the consume schedule — when and how many users emit the
scenario's ops. It is an axis *independent* of the scenario: the scenario
owns the op mix and data, the runner owns the schedule. Every scenario runs
under both models.

- **Closed** (`--model closed`, default): a fixed pool of `--users N`
  virtual users, each looping its plan with in-flight = 1 per user (send,
  await response, think, next). An optional `--global-concurrency` cap
  bounds total in-flight ops. Concurrency is fixed at N — use it to measure
  peak throughput/latency at a fixed concurrency.
- **Open** (`--model open`): users arrive per a Poisson process
  (`--arrival-rate`), each running `--session-ops` ops then leaving.
  Concurrency is emergent (a function of arrival rate vs service rate). A
  `--queue-bound` beyond the global concurrency cap controls how many
  requests queue before being **rejected** (`status=rejected`,
  `error_kind=queue_full`, zero latency) — the rejection rate and the
  arrival rate where it starts are the open model's key output. Use it to
  find the overload threshold.

Because every scenario's plan wraps its data stream, a `--duration` run
sustains load instead of going idle once a finite stream is exhausted.

For the full mechanics (closed loop vs arrival process, the congestion
policy, and when to use which) see [`docs/load-models.md`](./docs/load-models.md).

## Quick start

First, make sure your LTM server is up (e.g. MemMachine at
`http://localhost:8080`), then run one of the scenarios below.

### Chatbot-LTM integration (`chat-replay`, the primary workload)

Replay a multi-turn dialogue as a chatbot-with-LTM would: before each user
turn, recall (search) against the user's utterance, then ingest both the
user and assistant turns. `--search-every N` throttles the recall cadence
(default 1 = recall before every user turn; N>1 recalls only every Nth
user turn). Requires a dataset with a `turn_stream` (LongMemEval, not
synthetic); the run fails loudly otherwise. The dialogue wraps, so a
`--duration` run replays it as many times as needed.

```sh
ltm100 run --config examples/memmachine.yaml \
    --scenario chat-replay --users 10 --duration 60 --seed 0 \
    --output out/chat-replay
```

Run the same chatbot workload under realistic arrival timing: users arrive
per a Poisson process and the open model's congestion policy applies (the
recall cadence and turn content are unchanged) — see [Load
models](#load-models).

```sh
ltm100 run --config examples/memmachine.yaml \
    --scenario chat-replay --users 10 --duration 30 --seed 0 \
    --model open --arrival-rate 2.0 --session-ops 12 \
    --global-concurrency 8 --queue-bound 4 \
    --output out/chat-replay-open
```

### Storage throughput (`add-load`, closed)

Pure add pressure — measure how fast the server ingests memories.

```sh
ltm100 run --config examples/synthetic.yaml \
    --scenario add-load --users 20 --duration 30 --seed 0 \
    --output out/add-load
```

### Search throughput & latency (`search-load`, closed)

Pre-ingest memories, then loop searches. `--preingest` fills each user's
memories before the measured run; `--preingest-fraction` controls how much.

```sh
ltm100 run --config examples/synthetic.yaml \
    --scenario search-load --users 20 --duration 30 --seed 0 \
    --preingest --preingest-fraction 1.0 \
    --output out/search-load
```

### Arrival-driven congestion probe (`mixed`, open)

A flat add/search mixture with a tunable search ratio — needs no dialogue,
so it runs against the synthetic dataset. Users arrive at 5/s, each doing 6
ops (80% search). A global cap of 4 in-flight with a queue of 4 buffers
bursts; overload beyond that is rejected.

```sh
ltm100 run --config examples/synthetic.yaml \
    --scenario mixed --users 8 --duration 20 --seed 0 \
    --model open --arrival-rate 5.0 --session-ops 6 \
    --queue-bound 4 --global-concurrency 4 --search-weight 0.8 \
    --preingest --preingest-fraction 0.5 \
    --output out/mixed
```

To explicitly exercise rejection, raise the arrival rate far above the
service rate and set `--queue-bound 0` (reject immediately when the cap is
saturated):

```sh
ltm100 run --config examples/synthetic.yaml \
    --scenario mixed --users 4 --duration 12 --seed 1 \
    --model open --arrival-rate 80.0 --session-ops 8 \
    --queue-bound 0 --global-concurrency 2 --search-weight 1.0 \
    --raw --output out/congestion
```

### MCP transport (same workload, MCP tools)

Drive add/search through MemMachine's `add_memory` / `search_memory` MCP tools
instead of REST — same `LTMClient` contract, so the workload and flags are
identical; only the config changes. Useful to compare REST vs MCP overhead on
the same load. (Requires `pip install -e ".[mcp]"`.)

```sh
ltm100 run --config examples/memmachine-mcp.yaml \
    --scenario chat-replay --users 20 --duration 30 --seed 0 \
    --output out/mcp-chat
```

### Common flags

- `--duration SECONDS` or `--ops N`: how a run terminates (one is required).
- `--users N`: number of virtual users.
- `--seed N`: reproducible load shape (varies with N for variance runs).
- `--global-concurrency N`: cap total in-flight ops (0 = no cap).
- `--rampup SECONDS`: stagger user start to avoid a thundering herd.
- `--search-weight F`: (mixed) fraction of ops that are search (0..1).
- `--think SECONDS`: (mixed, chat-replay) max think-time jitter per op.
- `--search-every N`: (chat-replay) issue a recall search every N user turns
  (default 1 = every user turn).
- `--answer-time SECONDS`: (chat-replay) mean LLM answer-generation time after
  a user turn (Exponential; 0 = back-to-back, default). Applied to all users.
- `--user-gap SECONDS`: (chat-replay) mean user think/typing time before the
  next turn (Exponential; 0 = back-to-back, default). Applied to all users.
- `--raw`: also write per-request `raw.ndjson`.
- `--no-delete-on-exit`: keep per-user state after the run.

## Reports

With `--output DIR`, LTM100 writes:

- `summary.json` — aggregated metrics (count, throughput, QPS, latency
  percentiles p50/p90/p95/p99/max, error rate) per op type, plus run meta.
- `summary.csv` — the same summary as a flat table.
- `raw.ndjson` (with `--raw`) — one line per request.

## Cleanup per-user state

Per-run state (e.g. MemMachine projects) is deleted on exit by default. To
delete it without running a benchmark:

```sh
ltm100 cleanup --config examples/memmachine.yaml --users 50
```

> Note: MemMachine's `projects/list` is eventually consistent — an immediate
> list after delete may still show the project before it disappears. The
> delete itself is confirmed by the server's response.

## Pluggable axes

- **Dataset adapters** (`ltm100/adapters/datasets/`): LongMemEval, synthetic.
- **Backend adapters** (`ltm100/adapters/backends/`): MemMachine (REST),
  MemMachine-MCP.
- **Transports** (`ltm100/adapters/transports/`): REST, MCP (`fastmcp`).

The MCP transport drives `add`/`search` through MemMachine's `add_memory` /
`search_memory` MCP tools (same `LTMClient` contract) so a workload can be
compared REST-vs-MCP. Lifecycle (project create/delete) stays on REST, since
MCP has no project-management tools. Note the MCP `add_memory` writes all
memory types (episodic + semantic), unlike the episodic-only REST add, so MCP
add latency is not directly comparable to REST add latency.

See [`DESIGN.md`](./DESIGN.md) for the adapter contracts and how to add a new
dataset or backend.

## Tests

```sh
pytest -q
```

## Versioning

Releases are marked with git tags (`vMAJOR.MINOR.PATCH`). The current release
is **v0.3.1**. Tag a release at a stable, documented milestone:

```sh
git tag v0.3.1
git push origin v0.3.1
```

During 0.x, each minor bump marks a meaningful, tested milestone (a coherent
set of features verified against a live server). Breaking changes bump the
minor version while still in 0.x.
