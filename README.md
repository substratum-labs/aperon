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

Aperon provides a high-level, thread-safe, and crash-safe `Collection` API. This API automatically manages:
* **Write-Ahead Logging (WAL)**: All inserts/deletes are appended to WAL before memtable updates for crash recovery.
* **Segment Compaction**: Automated or manual memory flushes serialize memtable entries into immutable columnar `.apms` segment files.
* **Manifest Synchronization**: Transparently writes and updates versioned manifest logs (`main.apmf`) recording active segments.
* **5-Layer Query Planning**: Recall queries automatically choose the most efficient execution path based on filters, symbols, and computational budget.

Here is the mutable `Collection` workflow implemented programmatically in Rust:

```rust
use std::path::Path;
use std::collections::BTreeMap;
use aperon_core::{Collection, MemoryRecordInput, RecallQuery};

fn main() -> Result<(), String> {
    let out_dir = Path::new("target/readme-demo");

    // 1. Open or create a Collection (automatically replays WAL & opens the MemorySpace)
    let mut collection = Collection::open(out_dir, "agent-memory".to_string())
        .map_err(|e| e.to_string())?;

    // 2. Insert records with vector embeddings, symbols, optional vector ID, and metadata map
    let mut metadata = BTreeMap::new();
    metadata.insert("project".to_string(), "castor".to_string());
    metadata.insert("importance".to_string(), "high".to_string());

    let record = MemoryRecordInput {
        record_id: 20260607,
        scope_id: 42,
        timestamp: 1717750800,
        source_id: 1,
        confidence: 0.98,
        text: "MVP Collection API is fully integrated and thread-safe.".to_string(),
        embedding: vec![0.15, -0.42, 0.88, 0.02],
        symbols: vec!["mvp".to_string(), "collection-api".to_string()],
        vector_id: Some("vec-record-001".to_string()),
        metadata,
    };

    collection.insert(record).map_err(|e| e.to_string())?;

    // 3. Recall memory records combining timestamp, scope, symbols, and semantic embeddings
    let mut query_metadata_filter = BTreeMap::new();
    query_metadata_filter.insert("project".to_string(), "castor".to_string());

    let query = RecallQuery {
        embedding: Some(vec![0.12, -0.40, 0.85, 0.0]),
        symbols: vec!["mvp".to_string()],
        scope_id: Some(42),
        metadata_filter: query_metadata_filter,
        limit: 5,
        ..Default::default()
    };

    let result = collection.recall(&query)?;

    println!("Recall paths traversed: {:?}", result.trace.access_paths);
    for hit in result.hits {
        println!(
            "Hit ID: {}, Vector ID: {:?}, Score: {}, Text: '{}'",
            hit.record_id, hit.vector_id, hit.score, hit.text
        );
    }

    // 4. Compact the memtable and sync all changes to disk
    collection.flush().map_err(|e| e.to_string())?;

    Ok(())
}
```

---

## 🐍 Python API

Aperon exposes the high-level `Collection` engine and low-level quantized vector index structures to Python. For more comprehensive details, see [docs/python_api.md](docs/python_api.md).

### High-Level Memory Collection Engine

Exposes the mutable thread-safe Collection API supporting inserts, batch inserts, symbolic filters, semantic search, and automatic file management:

```python
from pathlib import Path
import aperon

# 1. Open or create the collection (replays WAL log automatically)
collection_dir = Path("target/python-demo")
collection = aperon.Collection.open(collection_dir, "agent-memory")

# 2. Insert records using standard Python dictionaries
record = {
    "record_id": 20260607,
    "scope_id": 42,
    "timestamp": 1717750800,
    "source_id": 1,
    "confidence": 0.98,
    "text": "MVP Collection API is fully integrated and thread-safe.",
    "embedding": [0.15, -0.42, 0.88, 0.02],
    "symbols": ["mvp", "collection-api"],
    "vector_id": "vec-record-001",
    "metadata": {"project": "castor", "importance": "high"},
}

collection.insert(record)

# 3. Query the collection using RecallQuery and PyO3 bindings
query = aperon.RecallQuery(
    embedding=[0.12, -0.40, 0.85, 0.0],
    symbols=["mvp"],
    scope_id=42,
    metadata_filter={"project": "castor"},
    limit=5,
)
result = collection.recall(query)

# 4. Iterate over the results
for hit in result["hits"]:
    print(f"Hit ID: {hit['record_id']}, Score: {hit['score']}, Text: {hit['text']}")

# 5. Flush and compact the active memtable to disk
collection.flush()
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
