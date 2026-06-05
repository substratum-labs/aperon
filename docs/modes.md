# Indexing Modes: Mode A & Mode B

Aperon supports two distinct execution modes depending on whether raw vectors are kept resident for query-time reranking.

---

## Mode A: Self-Contained Compressed Search

Mode A stores the compressed index and reconstructs/reranks from its own payload. It does not require raw vectors at query time.

### Command Line Interface (CLI)

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

### Python API

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

### Python API

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
