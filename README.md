# Aperon

[![CI](https://github.com/substratum-labs/aperon/actions/workflows/ci.yml/badge.svg)](https://github.com/substratum-labs/aperon/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT_or_Apache_2.0-blue.svg)](LICENSE-MIT)
[![Rust](https://img.shields.io/badge/rust-stable-brightgreen.svg)](https://www.rust-lang.org/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

**Aperon is a transformative, high-risk, leading-edge vector and semantic memory engine that compresses agent-native memory spaces to sub-10% of standard HNSW footprints with zero pointer-chasing CPU cache misses.** Rather than incrementally optimizing established vector databases, it introduces an entirely unproven, non-evolutionary indexing paradigm: an embedded Memory SSTable engine powered by topological manifold-adaptive quantization (MAQ) and pointerless tangent-lattice atlas (HTLA) routing to achieve sub-5ns distance evaluation latency.

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

## 🚀 End-to-End Memory SSTable Walkthrough

Aperon is designed as an agent-native **Memory SSTable engine**. Instead of simple in-memory vector indexing, it models memory as log-structured segments, versioned manifests, and multi-path recall queries combining symbol filters with quantized vector routing.

Here is the complete walkthrough of compiling memory records, executing recall, and branching/forking the memory space.

### 1. Setup & Demo CLI Walkthrough

Clone the repository and build the binary:

```bash
git clone https://github.com/substrate-lab/aperon.git
cd aperon

# Build Rust binaries and compile the Python bindings
cargo build --workspace
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip maturin numpy
maturin develop --release
```

Run the end-to-end companion demo binary (`memory_sstable_demo`) to write segments, recall records, and fork branches:

```bash
# A. Build segment files (.apms) and manifest (.apmf) from a JSONL log of memory inputs
cargo run -p aperon-core --bin memory_sstable_demo -- build \
  --input examples/aperon_memory.jsonl \
  --out target/memory-demo

# B. Recall memory records combining timestamp, scope, symbols, and semantic embeddings
cargo run -p aperon-core --bin memory_sstable_demo -- recall \
  --manifest target/memory-demo/main.apmf \
  --query examples/query_prefix8.json

# C. Fork a zero-copy branch manifest inheriting all segment files
cargo run -p aperon-core --bin memory_sstable_demo -- fork \
  --manifest target/memory-demo/main.apmf \
  --branch experimental-branch \
  --out target/memory-demo/fork.apmf
```

### 2. Rust Programmatic API

Here is the exact same build, recall, and fork workflow implemented programmatically in Rust:

```rust
use std::path::{Path, PathBuf};
use aperon_core::{
    MemoryRecordInput, MemorySegment, MemoryManifestFile, MemoryManifestSegment,
    MemorySpace, RecallQuery, stable_memory_branch_id
};

fn main() -> Result<(), String> {
    let out_dir = Path::new("target/readme-demo");
    std::fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;

    // 1. Build and write immutable columnar segment files (.apms) from memory inputs
    let records = vec![MemoryRecordInput {
        record_id: 171001,
        scope_id: 7,
        timestamp: 171,
        source_id: 1,
        confidence: 0.96,
        text: "T-171 proved the route kernel is fast.".to_string(),
        embedding: vec![1.0, 0.05, 0.0, 0.0],
        symbols: vec!["T-171".to_string(), "route-kernel".to_string()],
    }];
    let segment = MemorySegment::build(171, 4, records)?;
    let segment_path = out_dir.join("segment-171.apms");
    segment.write(&segment_path).map_err(|e| e.to_string())?;

    // 2. Write a versioned manifest file (.apmf) tracking the active segments
    let manifest = MemoryManifestFile::new(
        None, // parent_manifest_id
        stable_memory_branch_id("main"),
        vec![MemoryManifestSegment {
            segment_id: 171,
            path: PathBuf::from("segment-171.apms"),
            vector_sidecar: None,
        }],
    );
    let manifest_path = out_dir.join("main.apmf");
    manifest.write(&manifest_path).map_err(|e| e.to_string())?;

    // 3. Open the MemorySpace snapshot and query it with combined symbol + semantic filters
    let space = MemorySpace::open(&manifest_path).map_err(|e| e.to_string())?;
    let query = RecallQuery {
        embedding: Some(vec![1.0, 0.0, 0.28, 0.0]),
        symbols: vec!["route-kernel".to_string()],
        scope_id: Some(7),
        limit: 5,
        ..Default::default()
    };
    let result = space.recall(&query)?;

    println!("Scanned {} segments", result.trace.segments_scanned);
    for hit in result.hits {
        println!("Hit record ID: {}, Score: {}", hit.record_id, hit.score);
    }

    // 4. Fork the memory space to create a zero-copy child manifest
    let fork_path = out_dir.join("fork.apmf");
    space.fork("experimental-branch", &fork_path).map_err(|e| e.to_string())?;
    Ok(())
}
```


---

## 🐍 Python API

Aperon exposes both low-level quantized vector index structures and high-level Memory SSTable primitives to Python. For more comprehensive details, see [docs/python_api.md](docs/python_api.md).

### High-Level Memory SSTable Engine

Exposes immutable segment building, manifest versioning, planned semantic recall, and zero-copy branching checkouts:

```python
import aperon

# 1. Compile immutable memory segments from record dicts
records = [{
    "record_id": 191001,
    "scope_id": 7,
    "timestamp": 191,
    "source_id": 1,
    "confidence": 0.97,
    "text": "T-191 exposes Memory SSTable bindings.",
    "embedding": [1.0, 0.0, 0.0, 0.0],
    "symbols": ["T-191", "python"],
}]

segment = aperon.MemorySegment.build(segment_id=191, dim=4, records=records)
segment.write("target/python-demo/segment-191.apms")

# 2. Write a versioned manifest file (.apmf) tracking the active segments
manifest = aperon.MemoryManifestFile(
    branch="main",
    segments=[{"segment_id": 191, "path": "segment-191.apms"}],
)
manifest.write("target/python-demo/main.apmf")

# 3. Open the MemorySpace snapshot and query it with combined symbol + semantic filters
space = aperon.MemorySpace.open("target/python-demo/main.apmf")
query = aperon.RecallQuery(
    embedding=[1.0, 0.0, 0.0, 0.0],
    symbols=["python"],
    scope_id=7,
    limit=5,
)
result = space.recall(query)

for hit in result["hits"]:
    print(f"Hit record ID: {hit['record_id']}, Score: {hit['score']}")

# 4. Fork the memory space to create a zero-copy child manifest
space.fork("python-experimental-branch", "target/python-demo/fork.apmf")
```

### Low-Level Quantized Vector Indexing

If you only need standalone, high-performance quantized vector indexing (without LSM segments, manifests, and symbolic filtering metadata), you can use the low-level `AperonIndex` class:

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
