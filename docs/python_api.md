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

## 2. `HlrRouter`

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

## 3. `HtlaRouter`

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
