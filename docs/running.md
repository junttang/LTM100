# Running LTM100

This guide covers configuration, common workload commands, load controls,
backend-specific behavior, reports, and cleanup. For scenario internals, see
[`scenarios.md`](./scenarios.md); for closed and open scheduling, see
[`load-models.md`](./load-models.md).

## Configuration

A run combines two inputs:

- A **YAML config file** for the dataset, backend adapter, endpoint, and stable
  environment-specific options.
- **CLI flags** for the scenario, users, termination, load model, concurrency,
  pre-ingest, ramp-up, and output.

Start from [`examples/`](../examples/README.md), update `backend.base_url`, and
verify that the target LTM service is running. Scenario selection stays on the
CLI so the same backend/dataset config can drive multiple workloads.

LongMemEval supports all four scenarios. Synthetic data supports `add-load`,
`search-load`, and `mixed`, but not `chat-replay`, because it has no structured
dialogue `turn_stream`.

## Workload examples

### Chatbot-LTM integration

`chat-replay` recalls against each user utterance and then stores the user and
assistant turns. It requires a dialogue dataset such as LongMemEval.

```sh
ltm100 run --config examples/memmachine.yaml \
    --scenario chat-replay --users 10 --duration 60 --seed 0 \
    --output out/chat-replay
```

Model the LLM answer gap and the user's reading/typing gap:

```sh
ltm100 run --config examples/memmachine.yaml \
    --scenario chat-replay --users 10 --duration 60 --seed 0 \
    --answer-time 2.0 --user-gap 3.0 \
    --output out/chat-replay
```

Run the same workload with arrival-driven sessions:

```sh
ltm100 run --config examples/memmachine.yaml \
    --scenario chat-replay --users 10 --duration 30 --seed 0 \
    --model open --arrival-rate 2.0 --session-ops 12 \
    --global-concurrency 8 --queue-bound 4 \
    --output out/chat-replay-open
```

### Add throughput

`add-load` continuously ingests memory items. It works with both LongMemEval
and synthetic data.

```sh
ltm100 run --config examples/synthetic.yaml \
    --scenario add-load --users 20 --duration 30 --seed 0 \
    --output out/add-load
```

### Search throughput and latency

`search-load` performs search only during measurement. Pre-ingest the corpus so
the measured run searches populated user memory.

```sh
ltm100 run --config examples/synthetic.yaml \
    --scenario search-load --users 20 --duration 30 --seed 0 \
    --preingest --preingest-fraction 1.0 \
    --output out/search-load
```

### Mixed open-model load

`mixed` emits a configurable add/search ratio. Under the open model it is a
lightweight congestion probe for finding the arrival rate at which rejections
begin.

```sh
ltm100 run --config examples/synthetic.yaml \
    --scenario mixed --users 8 --duration 20 --seed 0 \
    --model open --arrival-rate 5.0 --session-ops 6 \
    --global-concurrency 4 --queue-bound 4 --search-weight 0.8 \
    --preingest --preingest-fraction 0.5 \
    --output out/mixed
```

### MCP transport

The MCP adapter sends measured add/search operations through `add_memory` and
`search_memory`; project setup and teardown remain on REST. The example uses
LongMemEval and requires the `datasets` and `mcp` extras.

```sh
pip install -e ".[dev,datasets,mcp]"

ltm100 run --config examples/memmachine-mcp.yaml \
    --scenario chat-replay --users 20 --duration 30 --seed 0 \
    --output out/mcp-chat
```

### Search expansion and filtering

Supporting REST adapters can forward `expand_context` and metadata filters.
The synthetic dataset writes `metadata.category` when its `categories` option
is enabled.

```sh
ltm100 run --config examples/synthetic.yaml \
    --scenario search-load --users 20 --duration 30 --seed 0 \
    --preingest --expand 2 --filter metadata.category=cat_3 \
    --output out/filtered-search
```

### Multi-process client scaling

One asyncio process can saturate a CPU core before a fast server reaches
capacity. `--procs` shards users and whole-run budgets across OS processes.

```sh
ltm100 run --config examples/memmachine.yaml \
    --scenario chat-replay --users 100 --duration 60 --seed 0 \
    --procs 4 --output out/scaled
```

## Common flags

- `--duration SECONDS` or `--ops N`: termination condition; one is required.
- `--users N`: virtual-user or tenant pool size.
- `--seed N`: reproducible load-shape seed.
- `--model closed|open`: scheduling model.
- `--global-concurrency N`: maximum total in-flight operations (`0` = no cap).
- `--arrival-rate F`, `--session-ops N`, `--queue-bound N`: open-model controls.
- `--procs N`: number of load-generator processes.
- `--rampup SECONDS`: stagger closed-model user startup.
- `--preingest`, `--preingest-fraction F`: populate memory before measurement.
- `--top-k N`: search result count for search-capable scenarios.
- `--expand N`: server-side context expansion, when supported.
- `--filter EXPR`: server-side metadata filter, when supported.
- `--search-weight F`: search fraction for `mixed`.
- `--think SECONDS`: maximum uniform think jitter for `mixed` and
  `chat-replay` operations.
- `--search-every N`: recall cadence in `chat-replay`.
- `--answer-time SECONDS`: mean LLM answer delay in `chat-replay`.
- `--user-gap SECONDS`: mean user reading/typing delay in `chat-replay`.
- `--server-metrics`: collect backend-provided server latency metrics.
- `--raw`: write per-request `raw.ndjson`.
- `--no-delete-on-exit`: preserve per-run backend state.

Run `ltm100 run --help` for the complete and authoritative option list.

## Backend notes

### MemMachine REST

[`examples/memmachine.yaml`](../examples/memmachine.yaml) maps each virtual
user to a project under `org_prefix`. It uses episodic add/search operations
and supports `--expand`, `--filter`, and `--server-metrics`.

The optional `project_id` and `filter_by_producer` settings support shared-
project isolation experiments. See the isolation-scope section in
[`scenarios.md`](./scenarios.md) before interpreting those results.

### MemMachine MCP

[`examples/memmachine-mcp.yaml`](../examples/memmachine-mcp.yaml) uses MCP for
measured add/search operations and REST for lifecycle management. MCP add
writes all memory types rather than the REST adapter's episodic-only payload,
so REST and MCP add latency are not a transport-only comparison.

The MCP tools expose neither context expansion nor metadata filtering and do
not accept item metadata. The adapter rejects unsupported combinations rather
than silently dropping them.

### Mem0 OSS REST

[`examples/mem0.yaml`](../examples/mem0.yaml) maps each virtual user to a
namespaced Mem0 `user_id`. Its default `infer: false` stores each input as one
memory without LLM fact extraction, preserving LTM100's item accounting. Set
`infer: true` to include Mem0's extraction pipeline. Mem0 supports `--filter`
but not `--expand`.

## Reports

With `--output DIR`, LTM100 writes:

- `summary.json`: metadata and per-operation aggregates. Offered, accepted,
  successful, backend-error, and rejected populations are separate.
  `throughput_ops_s`, `qps`, and latency percentiles contain successful
  requests only. Search `items.empty_rate` is the fraction of successful
  searches returning no items.
- `summary.csv`: flattened summary rows. The overall row omits mixed-operation
  latency because combining add and search latency is ambiguous.
- `raw.ndjson`: per-request records when `--raw` is enabled.
- `server_metrics.csv` and `server_metrics_raw.json`: server-side latency
  deltas when `--server-metrics` is enabled and supported by the adapter.

`meta` records the whole-run configuration, server build, and timestamps that
bracket the complete run lifecycle. With one process, server metrics bracket
the measured window. With multiple processes, the parent brackets the whole
run and reports `window: "whole_run"`, which includes setup and pre-ingest.

Server-side resource utilization such as CPU, memory, storage, and network is
outside LTM100's client report and should be collected from the system under
test.

## Cleanup

Per-run backend state is deleted on exit unless `--no-delete-on-exit` is set.
To delete user state without running a benchmark:

```sh
ltm100 cleanup --config examples/memmachine.yaml --users 50
```

Lifecycle behavior is backend-specific. For example, a MemMachine project list
may remain eventually consistent briefly after a confirmed delete response.

## Reproducible runs

- Keep the YAML, complete CLI command, LTM server build, and hardware topology
  with each result.
- Use the same dataset split, seed, user count, and pre-ingest fraction when
  comparing systems.
- Confirm that search runs return non-empty results; a low error rate alone
  does not prove that the corpus was populated correctly.
- Increase `--procs` only after checking whether the load-generator process is
  the bottleneck.
- Treat REST, MCP, extraction-enabled, and extraction-disabled paths as
  different workloads unless their server-side behavior is equivalent.
