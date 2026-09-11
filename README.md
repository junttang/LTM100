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

## Status

Version: **v0.4.0** (see [Versioning](#versioning)).

Early development. Datasets and LTM backends are pluggable; the initial
baseline is the LongMemEval dataset + the MemMachine backend over REST.

Implemented:
- Closed and open load models (every scenario runs under both).
- Scenarios: `chat-replay` (the primary workload), `add-load`, `search-load`,
  `mixed`. All wrap their data streams to sustain load and search on
  content-derived queries.
- Congestion policy (bounded queue with rejection) for the open model.
- Warm-up / pre-ingest before the measured run.
- Datasets: LongMemEval (local file or HuggingFace), synthetic.
- Backend: MemMachine (REST and MCP transports).
- Configurable scenario parameters on the CLI (`--think`, `--search-every`,
  `--search-weight`, `--top-k`, `--answer-time`, `--user-gap`).
- Reports: summary JSON/CSV + optional raw NDJSON.

Planned (see [DESIGN.md](./DESIGN.md) for the full roadmap):
- Mem0 backend adapter; per-user in-flight > 1; per-user-group finer control;
  configurable memory types; additional datasets (BEAM, LoCoMo).

## Install

```sh
pip install -e ".[dev]"
# To use the LongMemEval adapter via HuggingFace also:
pip install -e ".[datasets]"
# To use the MemMachine MCP transport also:
pip install -e ".[mcp]"
```

The `ltm100` CLI is the entry point. In this environment it is invoked as
`python -m ltm100.cli` if the console script is not on PATH.

## Configuration

A run takes two inputs:

- A **YAML config file** (stable, per environment): backend endpoint/auth,
  transport, and the dataset and backend adapter choices. See
  `examples/memmachine.yaml` and `examples/synthetic.yaml`.
- **CLI flags** (per run): number of users, scenario, duration/ops, seed,
  load model, concurrency, warm-up, and output. See `ltm100 run --help`.

Edit `examples/*.yaml` to point at your LTM server (`backend.base_url`) and
pick a dataset.

## Quick start

First, make sure your LTM server is up (e.g. MemMachine at
`http://localhost:8080`), then run one of the scenarios below. `chat-replay`
is the primary workload; the others are auxiliary load probes.

### Chatbot-LTM integration (`chat-replay`, the primary workload)

Replay a multi-turn dialogue as a chatbot-with-LTM would: before each user
turn, recall (search) against the user's utterance, then ingest both the
user and assistant turns. Requires a dataset with a `turn_stream`
(LongMemEval, not synthetic); the run fails loudly otherwise. The dialogue
wraps, so a `--duration` run replays it as many times as needed.

```sh
ltm100 run --config examples/memmachine.yaml \
    --scenario chat-replay --users 10 --duration 60 --seed 0 \
    --output out/chat-replay
```

Model the LLM answer time and the user's typing time so the load shape
resembles a real chatbot session (both default to 0 = back-to-back):

```sh
ltm100 run --config examples/memmachine.yaml \
    --scenario chat-replay --users 10 --duration 60 --seed 0 \
    --answer-time 2.0 --user-gap 3.0 \
    --output out/chat-replay
```

Run the same chatbot workload under realistic arrival timing (open model):
users arrive per a Poisson process and the congestion policy applies — the
recall cadence and turn content are unchanged.

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
so it runs against the synthetic dataset. Overload beyond the queue bound
is rejected; sweep `--arrival-rate` to find the server's capacity.

```sh
ltm100 run --config examples/synthetic.yaml \
    --scenario mixed --users 8 --duration 20 --seed 0 \
    --model open --arrival-rate 5.0 --session-ops 6 \
    --queue-bound 4 --global-concurrency 4 --search-weight 0.8 \
    --preingest --preingest-fraction 0.5 \
    --output out/mixed
```

### MCP transport (same workload, MCP tools)

Drive add/search through MemMachine's `add_memory` / `search_memory` MCP
tools instead of REST — same `LTMClient` contract, so the workload and flags
are identical; only the config changes. Useful to compare REST vs MCP
overhead on the same load. (Requires `pip install -e ".[mcp]"`.)

```sh
ltm100 run --config examples/memmachine-mcp.yaml \
    --scenario chat-replay --users 20 --duration 30 --seed 0 \
    --output out/mcp-chat
```

### Common flags

- `--duration SECONDS` or `--ops N`: how a run terminates (one is required).
- `--users N`: number of virtual users.
- `--seed N`: reproducible load shape (varies with N for variance runs).
- `--model closed|open`: load model (see [Load models](docs/load-models.md)).
- `--global-concurrency N`: cap total in-flight ops (0 = no cap).
- `--arrival-rate F` / `--session-ops N` / `--queue-bound N`: open-model knobs.
- `--rampup SECONDS`: stagger user start to avoid a thundering herd.
- `--search-weight F`: (mixed) fraction of ops that are search (0..1).
- `--top-k N`: (search-load, mixed, chat-replay) memories returned per search
  (default 20). Applied to all users.
- `--think SECONDS`: (mixed, chat-replay) max think-time jitter per op.
- `--search-every N`: (chat-replay) recall every N user turns (default 1).
- `--answer-time SECONDS`: (chat-replay) mean LLM answer time after a user
  turn (Exponential; 0 = back-to-back, default). All users.
- `--user-gap SECONDS`: (chat-replay) mean user think/typing time before the
  next turn (Exponential; 0 = back-to-back, default). All users.
- `--preingest` / `--preingest-fraction F`: pre-fill memories before the run.
- `--raw`: also write per-request `raw.ndjson`.
- `--no-delete-on-exit`: keep per-user state after the run.

See `ltm100 run --help` for the complete list.

## Backend setup

LTM100 talks to an LTM server through a pluggable **backend adapter**
(`LTMClient` contract). The initial baseline backend is **MemMachine**,
usable over **REST** or **MCP**:

- **MemMachine (REST)** — `examples/memmachine.yaml`. Points `backend.base_url`
  at the server (e.g. `http://localhost:8080`); `org_prefix` namespaces
  per-user projects (`session_key = f"{org_prefix}/user_{UserId}"`); one
  user = one MemMachine project, created on setup and deleted on teardown.
- **MemMachine (MCP)** — `examples/memmachine-mcp.yaml`. Same workload via
  `add_memory` / `search_memory` MCP tools. Lifecycle stays on REST (MCP has
  no project-management tools). Note `add_memory` writes all memory types
  (episodic + semantic), unlike the episodic-only REST add, so MCP add
  latency is not directly comparable to REST add latency.

Verify the server is up before a run (MemMachine: `GET /api/v2/health`).
Additional backends (e.g. Mem0) and how to add a new one are described in
[`DESIGN.md`](./DESIGN.md).

## Reports

With `--output DIR`, LTM100 writes:

- `summary.json` — aggregated metrics: an overall total throughput/QPS at the
  top level, plus count, throughput, QPS, latency percentiles p50/p90/p95/p99/max,
  and error rate per op type, plus run meta.
- `summary.csv` — the same summary as a flat table, with an overall `all` row
  (throughput/qps only; latency cells blank since mixing add/search latencies is
  ambiguous).
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

## Documentation

This README is a fast entry point. Detail lives under [`docs/`](./docs/) and
[`DESIGN.md`](./DESIGN.md):

- [`docs/scenarios.md`](./docs/scenarios.md) — per-scenario data flow: how a
  virtual user behaves, concurrency, the `add`/`search` ops at the
  backend-contract level, plus the scenario comparison table.
- [`docs/load-models.md`](./docs/load-models.md) — the closed/open load
  models, the congestion/rejection policy, and when to use which.
- [`DESIGN.md`](./DESIGN.md) — full design: pluggable adapter contracts,
  metrics, run lifecycle, reproducibility, project layout, roadmap.

## Tests

```sh
pytest -q
```

## Versioning

Releases are marked with git tags (`vMAJOR.MINOR.PATCH`). The current release
is **v0.4.0**. Tag a release at a stable, documented milestone:

```sh
git tag v0.4.0
git push origin v0.4.0
```

During 0.x, each minor bump marks a meaningful, tested milestone (a coherent
set of features verified against a live server). Breaking changes bump the
minor version while still in 0.x.
