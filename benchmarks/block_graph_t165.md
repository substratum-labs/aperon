# T-165 Pointerless Block Graph Router Prototype

- Dataset path: `benchmarks/data/siftsmall`
- Generated at: 2026-05-31 15:20:34 PDT
- Python: 3.14.4
- Platform: macOS-26.5-arm64-arm-64bit-Mach-O
- Final exact rerank: `final_nprobe=16` over the bounded centroid candidate pool.
- Centroids are deterministically sampled, clustered into near-fixed-size blocks, and stored as a contiguous payload array with int32 offsets.
- Block routing uses a fixed-width `block_neighbors[B, M]` int32 matrix over block representatives; selected blocks are scanned contiguously before bounded exact centroid rerank.

## Minimum Matrix

| K | Block size | Blocks | Block M | Entries | Rounds | Beam blocks | Pool | Coverage@16 | Pool coverage@32 | QPS | Route us/q | Block evals/q | Centroid evals/q | Selected blocks/q | Candidates/q | Graph bytes | Workset bytes/q | Build s | Fallback | Neighbor recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 16 | 64 | 8 | 8 | 4 | 16 | 256 | 0.9812 | 0.9666 | 7494.1 | 133.4 | 64.0 | 256.0 | 16.0 | 256.0 | 563,492 | 166,400 | 0.0505 | 0.0000 | 0.9414 |
| 1024 | 32 | 32 | 8 | 8 | 4 | 16 | 256 | 0.9994 | 0.9984 | 7866.9 | 127.1 | 32.0 | 512.0 | 16.0 | 256.0 | 545,956 | 279,936 | 0.0285 | 0.0000 | 0.9727 |
| 1024 | 64 | 16 | 8 | 8 | 4 | 16 | 256 | 1.0000 | 1.0000 | 5702.3 | 175.4 | 16.0 | 1024.0 | 16.0 | 256.0 | 537,188 | 533,184 | 0.0150 | 0.0000 | 0.9922 |
| 4096 | 16 | 256 | 8 | 16 | 5 | 32 | 512 | 0.9762 | 0.9666 | 2807.4 | 356.2 | 214.4 | 512.0 | 32.0 | 512.0 | 2,253,892 | 379,097 | 0.7165 | 0.0000 | 0.8306 |
| 4096 | 32 | 128 | 8 | 16 | 5 | 32 | 512 | 0.9944 | 0.9947 | 2895.8 | 345.3 | 128.0 | 1024.0 | 32.0 | 512.0 | 2,183,748 | 595,456 | 0.3424 | 0.0000 | 0.9004 |
| 4096 | 64 | 64 | 8 | 16 | 5 | 32 | 512 | 1.0000 | 1.0000 | 2588.8 | 386.3 | 64.0 | 2048.0 | 32.0 | 512.0 | 2,148,676 | 1,084,160 | 0.1784 | 0.0000 | 0.9434 |

## Sensitivity

| K | Block size | Blocks | Block M | Entries | Rounds | Beam blocks | Pool | Coverage@16 | Pool coverage@32 | QPS | Route us/q | Block evals/q | Centroid evals/q | Selected blocks/q | Candidates/q | Graph bytes | Workset bytes/q | Build s | Fallback | Neighbor recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 16 | 256 | 16 | 16 | 5 | 32 | 512 | 0.9762 | 0.9666 | 2409.3 | 415.1 | 241.3 | 512.0 | 32.0 | 512.0 | 2,262,084 | 397,785 | 0.7092 | 0.0000 | 0.9517 |
| 4096 | 32 | 128 | 16 | 16 | 5 | 32 | 512 | 0.9944 | 0.9947 | 2770.7 | 360.9 | 128.0 | 1024.0 | 32.0 | 512.0 | 2,187,844 | 599,552 | 0.3451 | 0.0000 | 0.9678 |
| 1024 | 32 | 32 | 8 | 8 | 3 | 16 | 256 | 0.9994 | 0.9984 | 7802.2 | 128.2 | 32.0 | 512.0 | 16.0 | 256.0 | 545,956 | 279,936 | 0.0285 | 0.0000 | 0.9727 |
| 1024 | 32 | 32 | 8 | 8 | 5 | 16 | 256 | 0.9994 | 0.9984 | 7722.8 | 129.5 | 32.0 | 512.0 | 16.0 | 256.0 | 545,956 | 279,936 | 0.0285 | 0.0000 | 0.9727 |
| 4096 | 32 | 128 | 8 | 16 | 5 | 16 | 512 | 0.9563 | 0.9487 | 3862.1 | 258.9 | 124.5 | 512.0 | 16.0 | 512.0 | 2,183,748 | 329,915 | 0.3424 | 0.0000 | 0.9004 |
| 4096 | 32 | 128 | 8 | 16 | 5 | 64 | 512 | 0.9988 | 0.9994 | 2097.2 | 476.8 | 128.0 | 2048.0 | 64.0 | 512.0 | 2,183,748 | 1,119,744 | 0.3424 | 0.0000 | 0.9004 |

## Baselines

| K | Router | Coverage@16 | Pool coverage@32 | Route us/q | Evals/q | Candidates/q | Resident bytes | Workset bytes/q | Build s |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | T-162 overlap PCA spill x4 | 1.0000 | 1.0000 | 445.1 | 1056.0 | 256.0 | 1,436,464 | 592,944 | 0.0689 |
| 4096 | T-162 overlap PCA spill x4 | 0.9919 | 0.9900 | 699.0 | 1296.0 | 512.0 | 7,505,328 | 751,680 | 0.2483 |
| 1024 | T-164 diverse point graph M16 | 0.9981 | 0.9966 | 332.8 | 896.0 | 255.1 | 589,856 | 182,330 | 0.2585 |
| 4096 | T-164 diverse point graph M16 | 0.9988 | 0.9966 | 972.6 | 2304.0 | 512.0 | 2,359,360 | 474,973 | 1.1511 |

## Interpretation

Positive signal: block graph routing reaches coverage@16 >= 0.99 at both K values and is route-time competitive with the T-162 overlap PCA baseline while using regular block scans.
