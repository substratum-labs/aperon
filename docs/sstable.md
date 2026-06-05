# Memory SSTable Baseline Comparison & Benchmarks

Aperon's Memory SSTable MVP has a reproducible local comparison harness covering the SSTable flat generator, SSTable array-like generator, SSTable pivot-prefix generator, optional SSTable HTLA/tangent generator, naive JSONL scan, in-memory flat scan, and vector-only flat scan.

---

## Reproducible Benchmarking

The benchmark runs the tiny `examples/aperon_memory.jsonl` / `examples/query_prefix8.json` case and deterministic synthetic scenarios.

### Full Scale Benchmark

```bash
cargo run -p aperon-core --bin memory_sstable_bench -- \
  --records 100000 \
  --segments 100 \
  --queries 100
```

### Fast Smoke Run

```bash
cargo run -p aperon-core --bin memory_sstable_bench -- \
  --records 1000 \
  --segments 10 \
  --queries 10
```

---

## Metrics Captured

The benchmark reports the following metrics for comparison across query paths:
- **Build Time**: Time taken to compile and package the segments.
- **Storage Footprint**: Disk bytes occupied by manifests (`.apmf`), record segments (`.apms`), and optional index sidecars (`.apmv`).
- **Query Latency**: Microseconds per search query.
- **Semantic Evals**: The number of inner-product vector calculations performed (representing index pruning efficiency).
- **Candidate Recall**: Precision at the candidate generation step before rerank.
- **Top-K Correctness**: Final precision/召回 overlap compared to brute-force exact scan.
- **Fork Overhead**: Branching speed and target manifest size when checking out speculative memory paths.

---

## Scenarios Log

Each scenario run writes structured machine-readable outputs under its scenario directory:
- `summary.json`: Schema version, scenario metadata, artifact paths, and path performance summaries.
- `metrics.jsonl`: Stable database metrics per path for append/merge benchmark tracking.

Current deterministic validation scenarios include:
- `tiny-prefix8`: Basic check matching the toy memory files.
- `metadata-selective`: Performance under tight metadata filters.
- `symbol-selective`: Performance under tight symbol filters.
- `broad-semantic`: General semantic recall performance.
- `branch-fork`: Verifies fork operation overhead.
- `adversarial`: Drives corner-case inputs to test reliability.
- `fallback`: Drives low-budget path misses to record index fallback rates.
