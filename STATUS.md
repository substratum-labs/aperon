# STATUS.md - Aperon

## Current Status: ACTIVE

## Key Metadata
- **Ecosystem Context:** [castor](file:///Users/yong/projects/castor), [castor-internal](file:///Users/yong/projects/castor-internal), [substratum-papers](file:///Users/yong/projects/substratum-papers)
- **Active Branch:** main

---

## Detailed Task Checklist
- [x] Phase 1: Prototype HNTL vector indexing model in C++/Python/Rust
- [x] Run benchmarks on synthetic anisotropic manifold datasets ($d=768$, $N=10{,}000$)
- [x] Capture hardware performance counters via Apple `kperf` PMU API
- [x] Draft Aperon Theory Paper (`papers/castor/aperon/theory-paper/`)
- [ ] Scale evaluation of HNTL to standard SIFT1M and GIST1M datasets
- [ ] Implement dynamic index update protocols (preservation of flat layouts without global rewrites)
- [ ] Port Block-SoA SIMD scan kernels to GPU (CUDA/Metal)

---

## Progress Logs
### 2026-06-05
* Completed M1 Milestone Technical Report on HNTL.
* Benchmarked Apple M2 Max: Block-SoA NEON SIMD achieves **4.137 ns/vector** scan throughput, representing a **3.61x speedup** over pointer-chasing graph traversal.
* Verified that local PCA captures $96.3\%$ of manifold variance, matching HNSW accuracy (Rerank Recall@10 = 1.0000) with $4.7\times$ index memory reduction.
