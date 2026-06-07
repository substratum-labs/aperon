# Compatibility and Deprecation Policy

This document details the binary format compatibility policy and the public API deprecation and stability policy for Aperon.

---

## 1. Binary Compatibility Policy

Aperon is designed to preserve access to on-disk vector indices and segment files. We classify compatibility constraints across two main categories of file formats: the **Active Memory SSTable Formats** and the **Legacy Index Formats**.

### 1.1 Active Memory SSTable Formats

The active storage engine uses three types of files to persist and recall memory records:

| Extension | Format Name | Role | Magic Header | Version | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `.apms` | Memory Segment | Immutable record segment | `APMS` | `0` | Stores record headers, text bytes, embedding rows, metadata columns, symbol postings, tombstone state, stats, and checksum. |
| `.apmf` | Manifest File | Space snapshot / fork boundary | N/A (JSON) | `0` | Stores branch IDs, parent manifest references, and list of ordered segment/sidecar references. |
| `.apmv` | Vector Sidecar | Optional routing artifact | `APMV` | `0` | Contains candidate generator artifacts (e.g. array-like or pivot-prefix tables) for segment acceleration. |

#### Compatibility Rules:
* **Forward Compatibility**: Older manifest and segment readers (e.g., v0) must ignore unknown fields when opening newer manifest formats, unless those fields are explicitly declared as required.
* **Sidecar Binding**: A `.apmv` sidecar file binds to exactly one `.apms` segment file by segment ID, segment version, record count, embedding dimension, and checksum fingerprint. A mismatch causes validation to fail.
* **Missing Optional Sidecars**: If an optional sidecar (`required == false`) is missing, mismatched, or corrupted, the system falls back to flat/direct scan recall paths and records `fallback_used` in the trace.
* **Missing Required Sidecars**: If a required sidecar is missing, mismatched, or corrupted, the system fails fast before executing queries or returning partial results.

---

### 1.2 Legacy Index Formats (`.hntl` / `.hntm`)

Aperon maintains backward compatibility with legacy formats used in previous MVP releases. These files are parsed by `load_legacy_index` into single or multi-grain legacy representations.

#### Single Grain Format (`.hntl` / `HNTL`)
* **Version 1**: Basic HNTL single-grain index. Residual bits are not serialized and default to 8.
* **Version 2, 3, 4**: Serializes the `residual_bits` parameter explicitly.

#### Multi-Grain Format (`.hntm` / `HNTM`)
* **Version 1**: Basic HNTM index. Inherits parameters dynamically.
* **Version 2**: Adds explicit `residual_bits` serialization.
* **Version 3**: Serializes explicit parameters (`local_dim`, `block_size`, `sketch_dim`, `residual_bits`) inside each individual grain.
* **Version 4**: Introduces layout format byte checks (`V4_FORMAT_LEGACY`, `V4_FORMAT_SHARED_PQ`, `V4_FORMAT_LATTICE_LEGACY`, `V4_FORMAT_LATTICE_SHARED_PQ`) and optional HLR / Shared PQ tables.

#### Breaking Change Rejection:
* Files missing proper magic headers (`HNTL` or `HNTM`) or declaring unsupported versions (e.g., version `>= 5`) correctly fail validation with `std::io::ErrorKind::InvalidData`.

---

## 2. API Deprecation and Stability Policy

Aperon stabilizes the public search entry points to ensure API predictability for both Rust and Python clients.

### 2.1 Stable Search APIs

Starting with version `0.1.0`, clients should select either Mode A or Mode B search pathways based on whether raw vectors are resident for query-time reranking.

#### Mode A: Self-Contained Search (Reranking via compressed reconstructions)
* **Rust**: `AperonIndex::search_mode_a(&self, query: &[f32], top_k: usize, nprobe: usize, rerank_factor: usize) -> Result<Vec<ScoredVector>, String>`
* **Python**: `AperonIndex.search_mode_a(query: List[float], top_k: int, nprobe: int, rerank_factor: int) -> List[Tuple[int, float]]`
* **Python Batch**: `AperonIndex.search_many_mode_a(queries: np.ndarray, top_k: int, nprobe: int, rerank_factor: int) -> List[List[Tuple[int, float]]]`

#### Mode B: Tiered Search (Reranking via attached raw vectors)
* **Rust**: `AperonIndex::search_mode_b(&self, query: &[f32], top_k: usize, nprobe: usize, candidate_k: usize) -> Result<Vec<ScoredVector>, String>`
* **Python**: `AperonIndex.search_mode_b(query: List[float], top_k: int, nprobe: int, candidate_k: int) -> List[Tuple[int, float]]`
* **Python Batch**: `AperonIndex.search_many_mode_b(queries: np.ndarray, top_k: int, nprobe: int, candidate_k: int) -> List[List[Tuple[int, float]]]`

---

### 2.2 Deprecated Legacy Search APIs

To prevent breaking existing integrations, legacy search methods are preserved as deprecated aliases. 

| Deprecated Method | Replacement | Deprecation Warning Level |
| :--- | :--- | :--- |
| `search(...)` | `search_mode_a(...)` | Rust compiler warning / Python `DeprecationWarning` |
| `search_with_nprobe(...)` | `search_mode_a(...)` | Rust compiler warning / Python `DeprecationWarning` |
| `search_tiered(...)` | `search_mode_b(...)` | Rust compiler warning / Python `DeprecationWarning` |
| `search_tiered_with_nprobe(...)` | `search_mode_b(...)` | Rust compiler warning / Python `DeprecationWarning` |
| `search_many(...)` | `search_many_mode_a(...)` | Rust compiler warning / Python `DeprecationWarning` |
| `search_many_tiered(...)` | `search_many_mode_b(...)` | Rust compiler warning / Python `DeprecationWarning` |

When calling these deprecated endpoints in Python, a `DeprecationWarning` is emitted using the standard `warnings` module. Legacy methods will be removed in a future major version release.
