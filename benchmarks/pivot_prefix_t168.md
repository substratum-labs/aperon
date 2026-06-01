# T-168 Permutation-Prefix Pivot Posting Router Prototype

- Dataset path: `benchmarks/data/siftsmall`
- Generated at: 2026-05-31 17:05:24 PDT
- Python: 3.14.4
- Platform: macOS-26.5-arm64-arm-64bit-Mach-O
- Final exact rerank: `final_nprobe=16` over the bounded centroid candidate pool.
- Blocks reuse the T-165 deterministic balanced block layout.
- Routing computes query pivot prefix, unions pivot postings, scores candidate blocks by overlap or weighted overlap, scans selected blocks contiguously, then exact-reranks centroids.

## Minimum Matrix

| K | Block size | Blocks | Pivots | Prefix | Top blocks | Pool | Mode | Coverage@16 | Pool coverage@32 | QPS | Route us/q | Pivot evals/q | Posting entries/q | Duplicate block rate | Selected blocks/q | Centroid evals/q | Candidates/q | Resident bytes | Workset bytes/q | Build s | Fallback |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 32 | 32 | 32 | 4 | 16 | 256 | union | 0.9644 | 0.9528 | 12696.6 | 78.8 | 32.0 | 17.1 | 0.4537 | 9.3 | 298.9 | 236.5 | 562,824 | 169,624 | 0.0329 | 0.0000 |
| 1024 | 32 | 32 | 32 | 8 | 16 | 256 | union | 0.9925 | 0.9956 | 8956.0 | 111.7 | 32.0 | 72.2 | 0.7759 | 15.9 | 510.4 | 256.0 | 564,104 | 278,334 | 0.0316 | 0.0000 |
| 1024 | 64 | 16 | 32 | 8 | 16 | 256 | union | 0.9994 | 0.9997 | 8819.9 | 113.4 | 16.0 | 64.4 | 0.8526 | 9.5 | 607.4 | 256.0 | 546,248 | 319,660 | 0.0186 | 0.0000 |
| 4096 | 32 | 128 | 64 | 4 | 32 | 512 | union | 0.9519 | 0.9397 | 6381.3 | 156.7 | 64.0 | 34.9 | 0.3290 | 23.4 | 749.8 | 498.2 | 2,217,992 | 417,135 | 0.4538 | 0.0000 |
| 4096 | 32 | 128 | 64 | 8 | 32 | 512 | union | 0.9856 | 0.9778 | 4833.6 | 206.9 | 64.0 | 142.2 | 0.6548 | 32.0 | 1023.4 | 512.0 | 2,223,112 | 558,170 | 0.4311 | 0.0000 |
| 4096 | 64 | 64 | 64 | 8 | 32 | 512 | union | 0.9981 | 0.9981 | 3836.0 | 260.7 | 64.0 | 68.0 | 0.6538 | 23.6 | 1509.1 | 512.0 | 2,184,968 | 806,128 | 0.2222 | 0.0000 |

## Sensitivity Rows

| K | Block size | Blocks | Pivots | Prefix | Top blocks | Pool | Mode | Coverage@16 | Pool coverage@32 | QPS | Route us/q | Pivot evals/q | Posting entries/q | Duplicate block rate | Selected blocks/q | Centroid evals/q | Candidates/q | Resident bytes | Workset bytes/q | Build s | Fallback |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 32 | 128 | 64 | 8 | 32 | 512 | weighted | 0.9856 | 0.9772 | 4420.1 | 226.2 | 64.0 | 142.2 | 0.6548 | 32.0 | 1023.4 | 512.0 | 2,223,112 | 558,170 | 0.4419 | 0.0000 |
| 4096 | 32 | 128 | 64 | 8 | 16 | 512 | weighted | 0.9106 | 0.8953 | 6175.4 | 161.9 | 64.0 | 142.2 | 0.6548 | 16.0 | 512.0 | 512.0 | 2,223,112 | 296,353 | 0.4419 | 0.0000 |
| 4096 | 32 | 128 | 64 | 8 | 64 | 512 | weighted | 0.9950 | 0.9944 | 3156.8 | 316.8 | 64.0 | 142.2 | 0.6548 | 49.0 | 1568.6 | 512.0 | 2,223,112 | 837,353 | 0.4419 | 0.0000 |
| 4096 | 32 | 128 | 64 | 12 | 32 | 512 | union | 0.9850 | 0.9797 | 3673.2 | 272.2 | 64.0 | 321.8 | 0.8106 | 32.0 | 1024.0 | 512.0 | 2,228,232 | 559,715 | 0.4884 | 0.0000 |
| 4096 | 32 | 128 | 64 | 12 | 32 | 512 | weighted | 0.9881 | 0.9831 | 3636.1 | 275.0 | 64.0 | 321.8 | 0.8106 | 32.0 | 1024.0 | 512.0 | 2,228,232 | 559,715 | 0.4579 | 0.0000 |

## Baseline Comparison

| Router | K | Coverage@16 | Pool coverage@32 | Route us/q | Route evals/q | Candidates/q | Resident bytes | Workset bytes/q | Build s |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T-165 block graph best practical | 1024 | 0.9994 | 0.9984 | 127.1 | 32.0 | 256.0 | 545,956 | 279,936 | 0.0285 |
| T-165 block graph best practical | 4096 | 0.9944 | 0.9947 | 345.3 | 128.0 | 512.0 | 2,183,748 | 595,456 | 0.3424 |
| T-167 pivot sketch l_inf | 1024 | 0.9988 | 0.9991 | 107.7 | 32.0 | 256.0 | 565,380 | 282,880 | 0.0335 |
| T-167 pivot sketch l_inf | 4096 | 0.9981 | 0.9984 | 325.4 | 64.0 | 512.0 | 2,171,140 | 1,073,664 | 0.2234 |

## Verdict

Positive signal: permutation-prefix postings reach coverage@16 >= 0.99 and reduce route working-set bytes versus T-167 dense pivot scans.
