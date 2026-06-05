# Aperon

[![CI](https://github.com/substratum-labs/aperon/actions/workflows/ci.yml/badge.svg)](https://github.com/substratum-labs/aperon/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT_or_Apache_2.0-blue.svg)](LICENSE-MIT)
[![Rust](https://img.shields.io/badge/rust-stable-brightgreen.svg)](https://www.rust-lang.org/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

**A vector and semantic memory engine for AI agents.** Treats agent memory like database engines treat LSM-based SSTables — with immutable segment columns, versioned manifests, multi-path query planning, and secondary index sidecars. Any LLM agent framework runs on top and inherits sub-10% HNSW memory footprints.

---

## Workspace Structure

- `aperon-core`: Core vector indexes, quantized scanning kernels, memory SSTable primitives, and planners.
- `aperon-cli`: Command-line interface (`aperon build`, `aperon query`, `aperon eval`) for index compilation and evaluation.
- `aperon-py`: PyO3 bindings published as the `aperon` Python package.

For detailed indexing modes and benchmark summaries, see:
- [Indexing Modes (Mode A & Mode B)](docs/modes.md) — Self-contained compressed search vs. raw-vector hot filtering.
- [Memory SSTable Benchmarks](docs/sstable.md) — Comparison parameters, metrics, and validation scenarios.

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

Generate a tiny dataset in Aperon's binary formats:

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

## CLI Reference

```text
aperon build --vectors <HNTR> --output <HNTL|HNTM> [--grains N] [--local-dim N] [--sketch-dim N] [--residual-bits 1|2|8] [--block-size N] [--adaptive-min-local-dim N --adaptive-max-local-dim N] [--shared-basis-cols N --shared-local-dim N --shared-pq-subquantizers N --shared-pq-bits 4|8 --shared-opq]
aperon query --index <HNTL|HNTM> --queries <HNTQ> [--top-k N] [--nprobe N] [--rerank-factor N]
aperon eval --index <HNTL|HNTM> --vectors <HNTR> --queries <HNTQ> [--top-k N] [--nprobe N] [--rerank-factor N] [--raw-rerank --candidate-k N]
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
