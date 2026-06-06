# Python API Reference

Aperon exposes high-performance index and routing primitives to Python through PyO3 bindings.

---

## 1. `AperonIndex`

The primary class for building and querying quantized vector search indexes.

### Constructor
```python
aperon.AperonIndex(
    dim: int, 
    local_dim: int = None, 
    sketch_dim: int = 0, 
    block_size: int = 64, 
    rerank_factor: int = 4, 
    residual_bits: int = 8
)
```
- `dim`: Input embedding dimension.
- `local_dim`: Projection dimension for manifold-local grains.
- `sketch_dim`: Bitwidth allocation dimension for quantized sketches.
- `block_size`: Number of vectors per compressed SoA block.
- `rerank_factor`: Over-sampling ratio for final candidate reranking.
- `residual_bits`: Number of bits used for VLBRD direction encoding.

### Core Methods

* **`insert(id_or_vector, vector=None) -> int`**  
  Inserts a single vector. If `vector` is omitted, the ID defaults to the current vector count.
* **`insert_many(ids: np.ndarray, matrix: np.ndarray) -> int`**  
  Batch inserts a matrix of float32 embeddings matching the corresponding list of IDs.
* **`rebuild_n_grains(grains: int)`**  
  Reclusters the inserted dataset into `N` manifold-local grains.
* **`save(path: str)`**  
  Serializes the index to disk.
* **`load(path: str) -> AperonIndex`**  
  Loads a serialized index from disk.
* **`search(query: np.ndarray, top_k: int, nprobe: int) -> list`**  
  Executes search queries under Mode A (Self-Contained).
* **`attach_raw_vectors(ids: np.ndarray, matrix: np.ndarray)`**  
  Attaches raw vectors to enable query-time exact reranking under Mode B.
* **`candidates(query: np.ndarray, nprobe: int, candidate_k: int) -> list`**  
  Retrieves candidate vector IDs before exact reranking.
* **`search_tiered(query: np.ndarray, top_k: int, nprobe: int, candidate_k: int) -> list`**  
  Executes tiered search queries under Mode B (Hot Filter + Rerank).
* **`stats() -> dict`**  
  Returns diagnostic statistics (e.g., number of grains, sizes, resident memory).

---

## 2. Memory SSTable Bindings

The high-level Memory SSTable API exposes immutable segment files, versioned manifests, symbolic filters, semantic recall, query tracing, and zero-copy manifest forks.

### `RecallQuery`
```python
aperon.RecallQuery(
    embedding: list[float] | None = None,
    symbols: list[str] = [],
    scope_id: int | None = None,
    time_start: int | None = None,
    time_end: int | None = None,
    min_confidence: float | None = None,
    limit: int = 10,
    candidate_budget: int | None = None,
)
```

All constructor fields are exposed as mutable Python properties.

### `MemorySegment`
* **`MemorySegment.build(segment_id: int, dim: int, records: list[dict]) -> MemorySegment`**  
  Builds an immutable segment from record dictionaries containing `record_id`, `scope_id`, `timestamp`, `source_id`, `confidence`, `text`, `embedding`, and `symbols`.
* **`write(path: str)`**  
  Writes an `.apms` segment file.
* **`read(path: str) -> MemorySegment`**  
  Loads an `.apms` segment file.
* **`len() -> int`**  
  Returns the number of records in the segment.

### `MemoryManifestFile`
```python
aperon.MemoryManifestFile(
    branch: str,
    segments: list[dict],
    parent_manifest_id: int | None = None,
)
```

Each segment dictionary must contain `segment_id` and `path`. Segment paths may be relative to the manifest file directory.

* **`write(path: str)`**  
  Writes a versioned `.apmf` manifest.
* **`read(path: str) -> MemoryManifestFile`**  
  Loads a manifest file.
* **`manifest_id`, `parent_manifest_id`, `branch_id`**  
  Read-only manifest metadata properties.

### `MemorySpace`
* **`MemorySpace.open(manifest_path: str) -> MemorySpace`**  
  Opens a manifest-backed memory space and loads active segment files.
* **`recall(query: RecallQuery) -> dict`**  
  Returns `{"hits": [...], "trace": {...}}`, where hit dictionaries contain `record_id`, `score`, `semantic_distance`, `symbol_matches`, `confidence`, `timestamp`, and `text`.
* **`fork(branch: str, out_path: str)`**  
  Writes a zero-copy child manifest pointing at the same segment files.

```python
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
segment.write("segment-191.apms")

manifest = aperon.MemoryManifestFile(
    branch="main",
    segments=[{"segment_id": 191, "path": "segment-191.apms"}],
)
manifest.write("main.apmf")

space = aperon.MemorySpace.open("main.apmf")
query = aperon.RecallQuery(
    embedding=[1.0, 0.0, 0.0, 0.0],
    symbols=["python"],
    scope_id=7,
    limit=5,
)
result = space.recall(query)
space.fork("experiment", "experiment.apmf")
```

---

## 3. `HlrRouter`

A hierarchical routing coordinator for mapping vectors to clustered grid keys.

### Constructor
```python
aperon.HlrRouter(dim: int, vectors: np.ndarray, layer_configs: list[tuple[int, float]])
```

### Core Methods
* **`route(query: list[float], nprobe: int) -> list[int]`**  
  Routes a single query vector to its nearest centroid keys.
* **`route_many(queries: np.ndarray, nprobe: int) -> list[list[int]]`**  
  Batch routes multiple queries.

---

## 4. `HtlaRouter`

A tangent-space Hierarchical Tangent Lattice Atlas router.

### Constructor
```python
aperon.HtlaRouter(dim: int, vectors: np.ndarray, levels: int, chart_dim: int)
```

### Core Methods
* **`route(query: list[float], beam: int, pool: int, final_nprobe: int = 16) -> dict`**  
  Performs lattice routing and returns a dictionary containing:
  - `candidates`: List of nearby vector IDs.
  - `final_nprobe`: Final centroid count evaluated.
  - `fallback`: Boolean indicating if route-risk fallback was triggered.
  - `working_set_bytes`: Working memory footprint of the routing path.
