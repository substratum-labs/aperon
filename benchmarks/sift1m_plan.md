# T-151 Larger Dataset Benchmark Plan (SIFT1M)

This plan defines the evaluation criteria, data preparation, baseline comparisons, and execution targets for scaling the Aperon Vector SSTable substrate to million-scale vector datasets.

---

## 1. Objectives

- **Scale Validation**: Confirm search latency and recall scaling from $10^4$ (siftsmall) to $10^6$ (SIFT1M) vectors.
- **Resource Profiling**: Measure the memory footprint of HNSW index alternatives vs Aperon's compact vector index layout.
- **Planner Sanity**: Verify that the multi-path `MemoryQueryPlanner` adapts to large-scale index sizes correctly.

---

## 2. Dataset & Setup

- **Dataset**: SIFT1M (1,000,000 base vectors, 10,000 query vectors, 100 dimensions, L2 distance).
- **Storage Tiering Profiles**:
  - *Full DRAM*: Keep all embeddings resident in memory.
  - *Tiered SQ8*: Offload vectors to disk in 8-bit scalar quantization format.
  - *Tiered F16*: Offload vectors to disk in 16-bit half-precision floating-point format.

---

## 3. Baselines

We will compare Aperon against:
1. **Faiss Flat IVF**: Standard inverted file index with exact L2 scanning of candidates.
2. **HNSW**: High-accuracy graph-based ANN search (measuring memory vs recall).
3. **Aperon SSTable Index (Mode A)**: Bounded candidate generation via pivot routing + exact Rerank.

---

## 4. Key Metrics to Collect

Report the following metrics for each configuration:
- **Recall@10**: Proportion of true nearest neighbors retrieved in top 10 results.
- **Query Latency (p50 / p95 / p99)**: Search latency in milliseconds.
- **Query QPS**: Queries processed per second.
- **Index Build Time**: Time to build the index structure in seconds.
- **DRAM Footprint**: Memory occupied by index structures.
- **Cold Bytes Read**: Disk reads triggered per query in tiered mode.

---

## 5. Target Acceptance Thresholds

| Index Type | Recall@10 Target | QPS Target | DRAM Footprint Target |
| :--- | :--- | :--- | :--- |
| **HNSW (Baseline)** | >= 0.98 | >= 1,000 | ~450 MB |
| **Aperon (Mode A - DRAM)** | >= 0.92 | >= 300 | <= 100 MB (< 0.25x HNSW) |
| **Aperon (Mode A - SQ8)** | >= 0.88 | >= 150 | <= 20 MB (< 0.05x HNSW) |
