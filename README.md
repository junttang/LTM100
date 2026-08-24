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

## Quick start (planned)

```sh
pip install -e .
ltm100 run --config examples/memmachine.yaml \
    --users 50 --scenario add-search-mixed --duration 60 --seed 0
```
