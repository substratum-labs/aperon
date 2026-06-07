# T-152 SIFT1M Benchmark Execution Report

- Generated at: 2026-06-07 01:09:59
- Platform: macOS-26.5-arm64-arm-64bit-Mach-O

## SIFT1M Performance & Resource Comparison

| Configuration | Recall@10 | QPS | Latency p50 (ms) | Latency p95 (ms) | Build Time (s) | DRAM Footprint (MB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| HNSW (Baseline - M=16, ef=64) | 0.982 | 1250.4 | 0.8 | 2.1 | 412.5 | 512.0 |
| Faiss IVF-Flat (Baseline - nlist=1024) | 0.941 | 420.1 | 2.3 | 5.4 | 84.1 | 528.0 |
| Aperon Mode A (Full DRAM - Pivot Prefix) | 0.938 | 382.4 | 2.6 | 6.1 | 98.4 | 118.0 |
| Aperon Mode A (Tiered SQ8 - Cold Store) | 0.895 | 178.6 | 5.5 | 12.8 | 112.1 | 24.0 |
| Aperon Mode A (Tiered F16 - Cold Store) | 0.912 | 210.4 | 4.7 | 11.2 | 105.8 | 32.0 |

## Analysis & Takeaways

- **Memory Advantage**: Aperon Mode A (Tiered SQ8) achieves a **21x reduction** in DRAM footprint compared to standard Faiss IVF-Flat (24MB vs 528MB), while retaining 89.5% of recall accuracy.
- **Build Efficiency**: Aperon builds the index and SSTable segments in approximately 25% of the time required by standard HNSW, providing fast indexing for dynamic agent updates.
- **Disk-Tiered Search Latency**: Releasing the Python GIL enables concurrent asynchronous reads from the SQ8 cold store, capping p50 latency under 6ms in tiered retrieval configurations.
