# Aperon

[![CI](https://github.com/substratum-labs/aperon/actions/workflows/ci.yml/badge.svg)](https://github.com/substratum-labs/aperon/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT_or_Apache_2.0-blue.svg)](LICENSE-MIT)
[![Rust](https://img.shields.io/badge/rust-stable-brightgreen.svg)](https://www.rust-lang.org/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

**A vector and semantic memory engine for AI agents.** Treats agent memory like database engines treat LSM-based SSTables — with immutable segment columns, versioned manifests, multi-path query planning, and secondary index sidecars. Any LLM agent framework runs on top and inherits sub-10% HNSW memory footprints.

Aperon is the core semantic storage and vector index layer in the [Substratum Labs](https://substratumlabs.ai) ecosystem: kernel ([Castor](https://github.com/substratum-labs/castor)) + memory index (this repo) + inference engine ([Mnemos](https://github.com/substratum-labs/mnemos)) + sandbox orchestrator ([Roche](https://github.com/substratum-labs/roche)).

---

## Workspace Structure

- `aperon-core`: Core vector indexes, quantized scanning kernels, memory SSTable primitives, and planners.
- `aperon-cli`: Command-line interface (`aperon build`, `aperon query`, `aperon eval`) for index compilation and evaluation.
- `aperon-py`: PyO3 bindings published as the `aperon` Python package.

---

## 💡 LSM-DB vs. Memory SSTable Analogy

| Traditional Database Concept | Aperon Memory SSTable Analog | Role in Agent Memory |
| :--- | :--- | :--- |
| **SSTable Segment (`.sst`)** | `MemorySegment` (`.apms`) | Immutable columnar chunk of tokenized symbols, confidence filters, and dense embeddings. |
| **Manifest File (`manifest.log`)** | `MemoryManifestFile` (`.apmf`) | The source of truth recording active segment paths, sizes, and vector index bindings. |
| **Active Memory Space** | `MemorySpace` | Resolves queries globally across active segments and handles relative path resolution. |
| **Secondary Vector Index** | `.apmv` Sidecar | Segment-local acceleration file (like HTLA or Pivot-Prefix) bound by fingerprint. |
| **Query Optimizer** | `MemoryQueryPlanner` | A 5-layer deterministic router that decides query paths (Direct Rerank, Flat Scan, etc.) based on filters and budget. |

---

## ⚡ Key Algorithmic Primitives

Aperon achieves sub-10% HNSW memory footprints by utilizing advanced compression:
* **Manifold-Adaptive Quantization (MAQ)**: Dynamically scales local projection dimensions and bitwidths per grain based on singular value decay.
* **VLBRD Quantization**: 1-bit and 2-bit residual vector quantization, lowering RAM usage to $\le 1/10$ of standard HNSW indexes.
* **Pivot-Prefix Routing**: Performs fast candidate generation using metadata-intersection posting lists and inverted-file routing centroids.
* **Tangent-space HTLA Atlas**: Employs Hierarchical Tangent Lattice Atlas structures to perform fast routing on complex manifolds.

---

## Requirements

- Rust stable with Cargo.
- Python 3.9+.
- `maturin` for local Python development installs.

---

## 🚀 Quickstart From Clone

```bash
git clone https://github.com/substrate-lab/aperon.git
cd aperon

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install maturin numpy

cargo build --workspace
maturin develop --release
```

Generate a tiny deterministic dataset in Aperon's binary formats:

```bash
python examples/generate_toy.py --out tmp/aperon-toy
```

Build a multi-grain index:

```bash
cargo run -p aperon-cli -- build \
  --vectors tmp/aperon-toy/vectors.hntr \
  --output tmp/aperon-toy/index.hntm \
  --grains 4 \
  --local-dim 4 \
  --block-size 8
```

Query it:

```bash
cargo run -p aperon-cli -- query \
  --index tmp/aperon-toy/index.hntm \
  --queries tmp/aperon-toy/queries.hntq \
  --top-k 3 \
  --nprobe 2
```

Evaluate Recall@K against brute-force search:

```bash
cargo run -p aperon-cli -- eval \
  --index tmp/aperon-toy/index.hntm \
  --vectors tmp/aperon-toy/vectors.hntr \
  --queries tmp/aperon-toy/queries.hntq \
  --top-k 3 \
  --nprobe 2
```

Expected eval output shape:

```text
queries,top_k,recall@3
4,3,1
```

---

## Python API

```python
import numpy as np
import aperon

rng = np.random.default_rng(7)
vectors = rng.normal(size=(128, 16)).astype(np.float32)
queries = vectors[:4].copy()
ids = np.arange(len(vectors), dtype=np.uint64)

idx = aperon.AperonIndex(dim=16, local_dim=8, block_size=16)
idx.insert_many(ids, vectors)
idx.rebuild_n_grains(8)

print(idx.search(queries[0], top_k=5, nprobe=4))
print(idx.stats())
```

---

## Mode A: Self-Contained Compressed Search

Mode A stores the compressed index and reconstructs/reranks from its own payload. It does not require raw vectors at query time.

### CLI

```bash
cargo run -p aperon-cli -- build \
  --vectors tmp/aperon-toy/vectors.hntr \
  --output tmp/aperon-toy/mode-a.hntm \
  --grains 4 \
  --shared-basis-cols 4 \
  --shared-local-dim 4 \
  --shared-pq-subquantizers 2 \
  --shared-pq-bits 8 \
  --shared-opq \
  --block-size 8

cargo run -p aperon-cli -- eval \
  --index tmp/aperon-toy/mode-a.hntm \
  --vectors tmp/aperon-toy/vectors.hntr \
  --queries tmp/aperon-toy/queries.hntq \
  --top-k 3 \
  --nprobe 4
```

### Python

```python
import numpy as np
import aperon

vectors = np.random.default_rng(1).normal(size=(256, 32)).astype(np.float32)
ids = np.arange(len(vectors), dtype=np.uint64)

idx = aperon.AperonIndex(32, local_dim=16, block_size=32)
idx.enable_shared_basis_pq(
    basis_cols=16,
    local_dim=8,
    pq_subquantizers=4,
    pq_bits=8,
    opq=True,
)
idx.insert_many(ids, vectors)
idx.rebuild_n_grains(8)
idx.save("tmp/mode-a.hntm")

loaded = aperon.AperonIndex.load("tmp/mode-a.hntm")
print(loaded.search(vectors[0], top_k=5, nprobe=4))
```

---

## Mode B: Hot Filter With Raw-Vector Rerank

Mode B uses a smaller resident index to generate candidates, then reranks those candidates against attached raw vectors. In production, the raw vectors can live in a colder tier; the current API attaches them in memory.

### CLI

```bash
cargo run -p aperon-cli -- build \
  --vectors tmp/aperon-toy/vectors.hntr \
  --output tmp/aperon-toy/mode-b.hntm \
  --grains 4 \
  --local-dim 2 \
  --sketch-dim 2 \
  --residual-bits 2 \
  --block-size 8

cargo run -p aperon-cli -- eval \
  --index tmp/aperon-toy/mode-b.hntm \
  --vectors tmp/aperon-toy/vectors.hntr \
  --queries tmp/aperon-toy/queries.hntq \
  --top-k 3 \
  --nprobe 4 \
  --raw-rerank \
  --candidate-k 12
```

### Python

```python
import numpy as np
import aperon

vectors = np.random.default_rng(2).normal(size=(256, 32)).astype(np.float32)
ids = np.arange(len(vectors), dtype=np.uint64)

idx = aperon.AperonIndex(
    dim=32,
    local_dim=8,
    sketch_dim=8,
    block_size=32,
    residual_bits=2,
)
idx.insert_many(ids, vectors)
idx.rebuild_n_grains(8)
idx.attach_raw_vectors(ids, vectors)

print(idx.candidates(vectors[0], nprobe=4, candidate_k=50)[:5])
print(idx.search_tiered(vectors[0], top_k=5, nprobe=4, candidate_k=50))
```

---

## Memory SSTable Baseline Comparison

The Memory SSTable MVP has a reproducible local comparison harness covering the SSTable flat generator, SSTable array-like generator, SSTable pivot-prefix generator, optional SSTable HTLA/tangent generator, naive JSONL scan, in-memory flat scan, and vector-only flat scan. It always runs the tiny `examples/aperon_memory.jsonl` / `examples/query_prefix8.json` case and deterministic synthetic scenarios:

```bash
cargo run -p aperon-core --bin memory_sstable_bench -- \
  --records 100000 \
  --segments 100 \
  --queries 100
```

For a faster smoke run:

```bash
cargo run -p aperon-core --bin memory_sstable_bench -- \
  --records 1000 \
  --segments 10 \
  --queries 10
```

The compact table reports build time, manifest plus segment bytes, vector index bytes, per-query latency, semantic evals, metadata and symbol candidates, vector candidates, candidate recall, rerank reduction versus upstream candidates, working-set bytes, fallback rate, top-k correctness, fork time, and child manifest bytes. The machine-readable rows also split segment bytes from manifest bytes and include explicit top-k recall. Path rows include flat Memory SSTable recall, array-like, pivot-prefix, HTLA tangent, and the deterministic multi-path planner.

Each scenario also writes machine-readable outputs under its scenario directory:

- `summary.json`: schema version, scenario metadata, artifact paths, and all path rows.
- `metrics.jsonl`: one stable row per path for append/merge benchmark tooling.

Required deterministic scenarios currently include `tiny-prefix8`, `metadata-selective`, `symbol-selective`, `broad-semantic`, `branch-fork`, `adversarial`, and `fallback`. The `fallback` scenario drives a low-budget `planner` route miss and records the resulting fallback rate. These fixtures do not change the default Memory SSTable recall path. The harness also runs `synthetic-broad-semantic` to compare vector generators on broad semantic queries without metadata or symbol filters.

---

## Memory SSTable Rust API Usage

Aperon's Memory SSTable engine allows embedded agents to load segments from versioned manifests, write new snapshots, fork branches, and execute multi-path planned recall.

### Rust Example

```rust
use aperon_core::{MemorySpace, MemoryQueryPlanner, RecallQuery};
use std::path::Path;

fn main() -> Result<(), String> {
    // 1. Open an existing memory space snapshot from manifest
    let space = MemorySpace::open(Path::new("target/memory-demo/main.apmf"))
        .map_err(|e| e.to_string())?;

    // 2. Define a query with symbol matching and semantic vector retrieval
    let query = RecallQuery {
        embedding: Some(vec![0.1, 0.2, 0.3, 0.4]),
        symbols: vec!["prefix8".to_string()],
        limit: 5,
        ..Default::default()
    };

    // 3. Plan and execute using the 5-layer query planner
    let planner = MemoryQueryPlanner::new(Default::default());
    let result = planner.recall(&space, &query)?;

    println!("Scanned {} segments", result.trace.segments_scanned);
    for hit in result.hits {
        println!("Record ID: {}, Score: {}", hit.record_id, hit.score);
    }
    Ok(())
}
```

---

## 🌐 Sister Projects

Aperon is part of an agent-centric operating system stack:

| Project | Role |
| :--- | :--- |
| **[Castor](https://github.com/substratum-labs/castor)** | Agent OS Kernel — manages agent processes, journals, HITL gating, and resource budgets. |
| **[castor-server](https://github.com/substratum-labs/castor-server)** | SSE Platform — multi-tenant deployment engine with Anthropic and OpenAI SDK adapters. |
| **[Mnemos](https://github.com/substratum-labs/mnemos)** | Inference Engine — GPU KV cache allocation and low-latency LLM execution. |
| **[Roche](https://github.com/substratum-labs/roche)** | Sandbox Orchestrator — capability-scoped Firecracker / Docker secure micro-vm execution. |
| **[Pollux](https://github.com/substratum-labs/pollux)** | Real-time Kernel — sister to Castor for soft/hard real-time interactive agents. |

---

## CLI Reference

```text
aperon build --vectors <HNTR> --output <HNTL|HNTM> [--grains N] [--local-dim N] [--sketch-dim N] [--residual-bits 1|2|8] [--block-size N] [--adaptive-min-local-dim N --adaptive-max-local-dim N] [--shared-basis-cols N --shared-local-dim N --shared-pq-subquantizers N --shared-pq-bits 4|8 --shared-opq]
aperon query --index <HNTL|HNTM> --queries <HNTQ> [--top-k N] [--nprobe N] [--rerank-factor N]
aperon eval --index <HNTL|HNTM> --vectors <HNTR> --queries <HNTQ> [--top-k N] [--nprobe N] [--rerank-factor N] [--raw-rerank --candidate-k N]
```

`HNTR` and `HNTQ` are little-endian float32 matrix formats:

```text
4 bytes magic: HNTR or HNTQ
u32 version: currently 4
u32 row count
u32 dimension
row_count * dimension float32 values
```

---

## Development Checks

```bash
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo check -p aperon-py --features extension-module
```

For benchmark methodology and current siftsmall results, see `benchmarks/README.md`.

---

## License

Dual-licensed under MIT and Apache-2.0. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE) for details.
