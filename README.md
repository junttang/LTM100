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

For the full design, see [`DESIGN.md`](./DESIGN.md).

## Status

Version: **v0.1.0** (see [Versioning](#versioning)).

Early development. Datasets and LTM backends are pluggable; the initial
baseline is the LongMemEval dataset + the MemMachine backend over REST.

Implemented:
- Closed and open load models.
- Scenarios: `add-load`, `search-load`, `add-search-mixed`, `realistic`.
- Congestion policy (bounded queue with rejection) for the open model.
- Warm-up / pre-ingest before the measured run.
- Datasets: LongMemEval (local file or HuggingFace), synthetic.
- Backend: MemMachine (REST).
- Reports: summary JSON/CSV + optional raw NDJSON.

Planned:
- Mem0 backend adapter.
- MCP transport under the same `LTMClient` contract.
- Additional datasets (BEAM, LoCoMo).

## Install

```sh
pip install -e ".[dev]"
# To use the LongMemEval adapter via HuggingFace also:
pip install -e ".[datasets]"
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
  `haystack_sessions` becomes a user's add stream; its `question/answer`
  becomes the search stream. Loads from a local JSON file (`path:`) or, if
  `path` is omitted, downloads the split from HuggingFace. `length:` caps the
  number of samples.
- **`synthetic`** — deterministically generated per-user content from the
  seed. No download; ideal for fast, reproducible load tests. Tunable via
  `memories_per_user`, `queries_per_user`, `content_chars`.

When the virtual-user count exceeds the dataset's unique samples, samples
are replicated so N is the driven user count, independent of dataset size.

### What gets added and searched

A scenario decides *when* and *how often* to issue add/search, but **not what
content** — that comes from the dataset adapter:

- **add** operates on `dataset.memory_stream(user)`: LongMemEval turns the
  sample's haystack into <=3000-char chunks (one turn may yield several
  items); synthetic yields a fixed number of deterministic items. Items are
  stored as **episodic** memory (`producer` = the user id).
- **search** operates on `dataset.query_stream(user)`: LongMemEval yields a
  single query per sample (the `question`); synthetic yields a few. Search
  scenarios cycle this finite query pool round-robin, with a small think
  jitter so users drift out of lockstep. Gold answer fields (`expected`) are
  carried for tracing only and are never scored — LTM100 measures load, not
  recall. `chat-replay` is the exception: it derives each recall query from
  the upcoming user turn's content via the dataset's optional
  `turn_stream(user)` (LongMemEval only), so recall is driven by the
  conversation itself rather than `query_stream`.

See [`docs/scenarios.md`](./docs/scenarios.md) for the full per-scenario
data-flow detail.

## Scenarios

A scenario turns each user's dataset streams into a sequence of operations.
The op mix (add vs search) is owned by the scenario for both load models.

| | add-load | search-load | add-search-mixed | chat-replay | realistic |
| --- | --- | --- | --- | --- | --- |
| load model | closed | closed | closed | closed | open |
| ops | add only | search only | add + search interleaved | recall + add per turn | search-weighted add + search |
| user lifetime | finite (stream exhausted) | infinite (loop) | finite (stream exhausted) | finite (dialogue replayed) | per-session (arrival → `session_ops`) |
| concurrency | N fixed, parallel add | N fixed, parallel search | N fixed, mixed | N fixed, in-order replay | emergent (Poisson arrivals) |
| precondition | none | preingest required | none | dataset with `turn_stream` | preingest recommended |
| termination | duration/ops | duration/ops | duration/ops | duration/ops | duration required |
| op scheduling | back-to-back | think 0–0.02s | back-to-back | think 0–0.05s | think 0–0.05s |
| key output | write throughput | read latency | read/write mix | chatbot-LTM integration load | rejection/congestion metrics |

For a detailed, per-scenario walkthrough — exactly how a virtual user
behaves, how many run, the concurrency model, and the `add`/`search`
operations at the backend-contract level (with MemMachine specifics
isolated to one section) — see [`docs/scenarios.md`](./docs/scenarios.md).

## Load models

- **Closed** (`--model closed`, default): a fixed number of virtual users,
  each looping its scenario plan with in-flight = 1 per user. An optional
  `--global-concurrency` cap bounds total in-flight ops.
- **Open** (`--model open`): users arrive per a Poisson process
  (`--arrival-rate`), each running `--session-ops` ops then leaving.
  Concurrency is emergent (a function of arrival rate vs service rate). A
  `--queue-bound` beyond the global concurrency cap controls how many
  requests queue before being **rejected** (`status=rejected`,
  `error_kind=queue_full`, zero latency).

## Quick start

First, make sure your LTM server is up (e.g. MemMachine at
`http://localhost:8080`), then run one of the scenarios below.

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

### Mixed workload (`add-search-mixed`, closed)

Interleaved add and search per user, mirroring a single user's lifetime.

```sh
ltm100 run --config examples/synthetic.yaml \
    --scenario add-search-mixed --users 50 --duration 60 --seed 0 \
    --output out/mixed
```

### Chatbot-LTM integration (`chat-replay`, closed)

Replay a multi-turn dialogue as a chatbot-with-LTM would: before each user
turn, recall (search) against the user's utterance, then ingest both the
user and assistant turns. Requires a dataset with a `turn_stream`
(LongMemEval, not synthetic); the run fails loudly otherwise.

```sh
ltm100 run --config examples/memmachine.yaml \
    --scenario chat-replay --users 10 --duration 60 --seed 0 \
    --output out/chat-replay
```

### Arrival-driven load with congestion (`realistic`, open)

Users arrive at 5/s, each doing 6 ops (80% search). A global cap of 4
in-flight with a queue of 4 buffers bursts; overload beyond that is rejected.

```sh
ltm100 run --config examples/synthetic.yaml \
    --scenario realistic --users 8 --duration 20 --seed 0 \
    --model open --arrival-rate 5.0 --session-ops 6 \
    --queue-bound 4 --global-concurrency 4 --search-weight 0.8 \
    --preingest --preingest-fraction 0.5 \
    --output out/realistic
```

To explicitly exercise rejection, raise the arrival rate far above the
service rate and set `--queue-bound 0` (reject immediately when the cap is
saturated):

```sh
ltm100 run --config examples/synthetic.yaml \
    --scenario realistic --users 4 --duration 12 --seed 1 \
    --model open --arrival-rate 80.0 --session-ops 8 \
    --queue-bound 0 --global-concurrency 2 --search-weight 1.0 \
    --raw --output out/congestion
```

### Common flags

- `--duration SECONDS` or `--ops N`: how a run terminates (one is required).
- `--users N`: number of virtual users.
- `--seed N`: reproducible load shape (varies with N for variance runs).
- `--global-concurrency N`: cap total in-flight ops (0 = no cap).
- `--rampup SECONDS`: stagger user start to avoid a thundering herd.
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
- **Backend adapters** (`ltm100/adapters/backends/`): MemMachine (REST).
- **Transports** (`ltm100/adapters/transports/`): REST; MCP planned under the
  same `LTMClient` contract.

See [`DESIGN.md`](./DESIGN.md) for the adapter contracts and how to add a new
dataset or backend.

## Tests

```sh
pytest -q
```

## Versioning

Releases are marked with git tags (`vMAJOR.MINOR.PATCH`). The current release
is **v0.1.0**. Tag a release at a stable, documented milestone:

```sh
git tag v0.1.0
git push origin v0.1.0
```

During 0.x, each minor bump marks a meaningful, tested milestone (a coherent
set of features verified against a live server). Breaking changes bump the
minor version while still in 0.x.
