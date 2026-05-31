# T-162 HTLA Overlap / Spill Router Prototype

- Dataset path: `/Users/yong/projects/aperon/benchmarks/data/siftsmall`
- Generated at: 2026-05-31 14:11:53 PDT
- Python: 3.14.4
- Platform: macOS-26.5-arm64-arm-64bit-Mach-O
- Final exact rerank: `final_nprobe=16` over the bounded candidate pool
- PCA routing uses parent-local chart distance plus the query residual constant for cross-parent score comparability.
- Morton/key ordering is reported only as a local-preservation diagnostic.

## Route Results

| K | Levels | Chart dim | Router | Beam | Pool | Spill | Coverage@16 | Pool coverage@32 | QPS | Route us/q | Child evals/q | Candidates/q | Route bytes | Workset bytes/q | Build s | Fallback |
| ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 3 | 12 | exact | 8 | 256 | x1 | 0.9600 | 0.9466 | 5500.2 | 181.8 | 288.0 | 256.0 | 1,166,128 | 147,456 | 0.0819 | 0.0000 |
| 1024 | 3 | 12 | pca | 8 | 256 | x1 | 0.9563 | 0.9416 | 7943.0 | 125.9 | 288.0 | 256.0 | 1,436,464 | 161,712 | 0.0819 | 0.0000 |
| 1024 | 3 | 12 | exact | 8 | 256 | x2 | 0.9988 | 0.9991 | 2930.2 | 341.3 | 544.0 | 256.0 | 1,166,128 | 278,528 | 0.0824 | 0.0000 |
| 1024 | 3 | 12 | pca | 8 | 256 | x2 | 0.9988 | 0.9988 | 4303.9 | 232.3 | 544.0 | 256.0 | 1,436,464 | 305,456 | 0.0824 | 0.0000 |
| 1024 | 3 | 12 | exact | 8 | 256 | x4 | 1.0000 | 1.0000 | 1513.4 | 660.8 | 1056.0 | 256.0 | 1,166,128 | 540,672 | 0.0689 | 0.0000 |
| 1024 | 3 | 12 | pca | 8 | 256 | x4 | 1.0000 | 1.0000 | 2246.5 | 445.1 | 1056.0 | 256.0 | 1,436,464 | 592,944 | 0.0689 | 0.0000 |
| 4096 | 4 | 16 | exact | 16 | 512 | x1 | 0.8856 | 0.8391 | 2835.2 | 352.7 | 528.0 | 256.0 | 4,849,584 | 270,336 | 0.2711 | 0.0000 |
| 4096 | 4 | 16 | pca | 16 | 512 | x1 | 0.8856 | 0.8391 | 3802.7 | 263.0 | 528.0 | 256.0 | 7,505,328 | 306,240 | 0.2711 | 0.0000 |
| 4096 | 4 | 16 | exact | 16 | 512 | x2 | 0.9669 | 0.9522 | 1827.7 | 547.1 | 784.0 | 512.0 | 4,849,584 | 401,408 | 0.2509 | 0.0000 |
| 4096 | 4 | 16 | pca | 16 | 512 | x2 | 0.9669 | 0.9522 | 2351.9 | 425.2 | 784.0 | 512.0 | 7,505,328 | 454,720 | 0.2509 | 0.0000 |
| 4096 | 4 | 16 | exact | 16 | 512 | x4 | 0.9919 | 0.9900 | 1090.6 | 916.9 | 1296.0 | 512.0 | 4,849,584 | 663,552 | 0.2483 | 0.0000 |
| 4096 | 4 | 16 | pca | 16 | 512 | x4 | 0.9919 | 0.9900 | 1430.6 | 699.0 | 1296.0 | 512.0 | 7,505,328 | 751,680 | 0.2483 | 0.0000 |

## Spill Sensitivity

| K | Router | Spill x1 | Spill x2 | Spill x4 |
| ---: | :--- | ---: | ---: | ---: |
| 1024 | exact | 0.9600 | 0.9988 | 1.0000 |
| 1024 | pca | 0.9563 | 0.9988 | 1.0000 |
| 4096 | exact | 0.8856 | 0.9669 | 0.9919 |
| 4096 | pca | 0.8856 | 0.9669 | 0.9919 |

## Local PCA Diagnostics

| K | Chart dim | Nodes | Leaves | Max depth | d80 p50 | d90 p50 | d95 p50 | d95 max | PCA neighbor recall | Morton/key neighbor recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 12 | 1057 | 1024 | 2 | 8.0 | 10.0 | 11.0 | 11 | 0.8784 | 0.4292 |
| 4096 | 16 | 4369 | 4096 | 3 | 8.0 | 10.0 | 12.0 | 13 | 1.0000 | 0.5456 |

## Interpretation

Positive signal: PCA child-distance overlap routing reaches coverage@16 >= 0.99 at K=1024 and K=4096 with spill <= x4 under the requested pool caps. This remains worth a Rust implementation pass.
