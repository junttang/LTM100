# LTM100

Multi-user load benchmark for Long-Term Memory (LTM) systems (e.g. MemMachine,
Mem0). It drives many virtual users performing `add`, `search`, and
`add&search` operations against a single LTM endpoint and reports
client-observable performance metrics (throughput, QPS, latency percentiles,
error rate).

This benchmark measures **load and concurrency behavior**, not retrieval or
answer quality. Server-side resource metrics are collected separately by the
server.

See [`DESIGN.md`](./DESIGN.md) for the full design.

## Status

Early development. Datasets and LTM backends are pluggable; the initial
baseline is LongMemEval + MemMachine (REST).

## Install

```sh
pip install -e ".[dev]"
# To use the LongMemEval adapter (HF datasets) also:
pip install -e ".[datasets]"
```

## Run

```sh
# Edit examples/memmachine.yaml to point at your MemMachine server, then:
ltm100 run --config examples/memmachine.yaml \
    --scenario add-search-mixed --users 50 --duration 60 --seed 0
```

Reports (summary JSON/CSV, optional raw NDJSON) are written to `--output`.

## Cleanup per-user state

```sh
ltm100 cleanup --config examples/memmachine.yaml --users 50
```

## Pluggable axes

- **Dataset adapters** (`ltm100/adapters/datasets/`): LongMemEval today.
- **Backend adapters** (`ltm100/adapters/backends/`): MemMachine (REST) today.
- **Transports** (`ltm100/adapters/transports/`): REST today; MCP planned under
  the same `LTMClient` contract.

See [`DESIGN.md`](./DESIGN.md) for the adapter contracts.

## Tests

```sh
pytest -q
```
