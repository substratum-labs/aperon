# T-161 HTLA Local Structure Sanity Check

- Dataset path: `/Users/yong/projects/aperon/benchmarks/data/siftsmall`
- Generated at: 2026-05-31 13:35:27 PDT
- Python: 3.14.4
- Platform: macOS-26.5-arm64-arm-64bit-Mach-O

## Exact-Tree Oracle

| K | Levels | Dim | Beam | Pool | Coverage@16 | Coverage@32 | Fallback | Spill x2 Coverage@16 | Spill x4 Coverage@16 | Nodes | Leaves | Max depth |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 2 | 8 | 4 | 64 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 129 | 128 | 1 |
| 128 | 2 | 8 | 8 | 128 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 129 | 128 | 1 |
| 1024 | 3 | 8 | 8 | 128 | 0.9600 | 0.9466 | 0.0000 | 0.9988 | 1.0000 | 1057 | 1024 | 2 |
| 1024 | 3 | 12 | 8 | 256 | 0.9600 | 0.9466 | 0.0000 | 0.9988 | 1.0000 | 1057 | 1024 | 2 |
| 4096 | 4 | 12 | 16 | 256 | 0.8856 | 0.8391 | 0.0000 | 0.9669 | 0.9919 | 4369 | 4096 | 3 |
| 4096 | 4 | 16 | 16 | 512 | 0.8856 | 0.8391 | 0.0000 | 0.9669 | 0.9919 | 4369 | 4096 | 3 |

## Local Preservation

| K | Dim | d80 p50 | d90 p50 | d95 p50 | d95 max | PCA neighbor recall | Morton/key neighbor recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 8 | 4.0 | 6.0 | 7.0 | 7 | 0.6484 | 0.3428 |
| 128 | 8 | 4.0 | 6.0 | 7.0 | 7 | 0.6484 | 0.3428 |
| 1024 | 8 | 6.0 | 7.0 | 8.0 | 8 | 0.8037 | 0.4291 |
| 1024 | 12 | 8.0 | 10.0 | 11.0 | 11 | 0.8784 | 0.4292 |
| 4096 | 12 | 7.0 | 9.0 | 11.0 | 11 | 0.9673 | 0.6421 |
| 4096 | 16 | 8.0 | 10.0 | 12.0 | 13 | 1.0000 | 0.5456 |

## Interpretation

Exact child-distance tree misses the original high-K beam budgets, but spill/overlap largely recovers coverage. The first failure is boundary/beam sensitivity in the hierarchy; Morton/key routing is an additional loss channel, not the only problem.
