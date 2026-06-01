# T-169 Block-Aware Monotonic / RNG Graph Router Prototype

- Dataset path: `benchmarks/data/siftsmall`
- Generated at: 2026-05-31 17:16:20 PDT
- Python: 3.14.4
- Platform: macOS-26.5-arm64-arm-64bit-Mach-O
- Final exact rerank: `final_nprobe=16` over the bounded centroid candidate pool.
- Blocks reuse the T-165 deterministic balanced block layout.
- `rng` greedily keeps candidates whose distance to already selected neighbors is at least the candidate distance from the source block; deterministic kNN fill preserves fixed width.
- `alpha` is a more permissive RNG variant using `candidate_distance / 1.25` as the separation threshold.

## Minimum Matrix

| K | Block size | Blocks | Candidate M | Final M | Entries | Rounds | Beam blocks | Pool | Prune | Coverage@16 | Pool coverage@32 | QPS | Route us/q | Block evals/q | Centroid evals/q | Selected blocks/q | Candidates/q | Graph bytes | Workset bytes/q | Build s | Fallback | Neighbor recall | Edge diversity |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 32 | 32 | 32 | 8 | 8 | 4 | 16 | 256 | rng | 0.9988 | 0.9991 | 8098.2 | 123.5 | 32.0 | 512.0 | 16.0 | 256.0 | 545,956 | 279,936 | 0.0382 | 0.0000 | 0.9570 | 1.2693 |
| 1024 | 64 | 16 | 32 | 8 | 8 | 4 | 16 | 256 | rng | 1.0000 | 1.0000 | 6005.3 | 166.5 | 16.0 | 1024.0 | 16.0 | 256.0 | 537,188 | 533,184 | 0.0197 | 0.0000 | 0.9766 | 1.3090 |
| 4096 | 32 | 128 | 32 | 8 | 16 | 5 | 32 | 512 | rng | 0.9944 | 0.9931 | 2968.7 | 336.8 | 128.0 | 1024.0 | 32.0 | 512.0 | 2,183,748 | 595,456 | 0.4739 | 0.0000 | 0.8838 | 1.3846 |
| 4096 | 64 | 64 | 32 | 8 | 16 | 5 | 32 | 512 | rng | 1.0000 | 1.0000 | 2648.2 | 377.6 | 64.0 | 2048.0 | 32.0 | 512.0 | 2,148,676 | 1,084,160 | 0.2384 | 0.0000 | 0.9395 | 1.3559 |
| 4096 | 32 | 128 | 64 | 8 | 16 | 5 | 32 | 512 | rng | 0.9944 | 0.9931 | 3007.6 | 332.5 | 128.0 | 1024.0 | 32.0 | 512.0 | 2,183,748 | 595,455 | 0.4746 | 0.0000 | 0.8604 | 1.4180 |
| 4096 | 32 | 128 | 64 | 16 | 16 | 5 | 32 | 512 | rng | 0.9944 | 0.9931 | 2866.6 | 348.8 | 128.0 | 1024.0 | 32.0 | 512.0 | 2,187,844 | 599,552 | 0.4853 | 0.0000 | 0.9697 | 1.2601 |

## Sensitivity Rows

| K | Block size | Blocks | Candidate M | Final M | Entries | Rounds | Beam blocks | Pool | Prune | Coverage@16 | Pool coverage@32 | QPS | Route us/q | Block evals/q | Centroid evals/q | Selected blocks/q | Candidates/q | Graph bytes | Workset bytes/q | Build s | Fallback | Neighbor recall | Edge diversity |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 32 | 128 | 64 | 8 | 16 | 5 | 16 | 512 | rng | 0.9644 | 0.9509 | 4066.3 | 245.9 | 121.9 | 512.0 | 16.0 | 512.0 | 2,183,748 | 328,569 | 0.4746 | 0.0000 | 0.8604 | 1.4180 |
| 4096 | 32 | 128 | 64 | 8 | 16 | 5 | 64 | 512 | rng | 0.9969 | 0.9972 | 2146.1 | 466.0 | 128.0 | 2048.0 | 64.0 | 512.0 | 2,183,748 | 1,119,744 | 0.4746 | 0.0000 | 0.8604 | 1.4180 |
| 4096 | 32 | 128 | 64 | 8 | 16 | 5 | 32 | 512 | alpha | 0.9944 | 0.9931 | 2904.2 | 344.3 | 128.0 | 1024.0 | 32.0 | 512.0 | 2,183,748 | 595,456 | 0.4675 | 0.0000 | 0.7354 | 1.4817 |
| 1024 | 32 | 32 | 32 | 8 | 8 | 4 | 16 | 256 | knn | 0.9988 | 0.9991 | 7617.3 | 131.3 | 32.0 | 512.0 | 16.0 | 256.0 | 545,956 | 279,936 | 0.0333 | 0.0000 | 1.0000 | 1.1655 |
| 4096 | 32 | 128 | 32 | 8 | 16 | 5 | 32 | 512 | knn | 0.9944 | 0.9931 | 3011.6 | 332.1 | 128.0 | 1024.0 | 32.0 | 512.0 | 2,183,748 | 595,454 | 0.4440 | 0.0000 | 1.0000 | 1.3089 |
| 4096 | 32 | 128 | 64 | 8 | 16 | 5 | 32 | 512 | knn | 0.9944 | 0.9931 | 3007.3 | 332.5 | 128.0 | 1024.0 | 32.0 | 512.0 | 2,183,748 | 595,454 | 0.4467 | 0.0000 | 1.0000 | 1.3089 |

## T-165 Baselines

| Router | K | Coverage@16 | Pool coverage@32 | Route us/q | Block evals/q | Candidates/q | Resident bytes | Workset bytes/q | Build s |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T-165 block graph bs32 beam16 | 1024 | 0.9994 | 0.9984 | 127.1 | 32.0 | 256.0 | 545,956 | 279,936 | 0.0285 |
| T-165 block graph bs32 beam16 | 4096 | 0.9563 | 0.9487 | 258.9 | 124.5 | 512.0 | 2,183,748 | 329,915 | 0.3424 |
| T-165 block graph bs32 beam32 | 4096 | 0.9944 | 0.9947 | 345.3 | 128.0 | 512.0 | 2,183,748 | 595,456 | 0.3424 |
| T-165 block graph bs32 beam32 | 4096 | 0.9944 | 0.9947 | 360.9 | 128.0 | 512.0 | 2,187,844 | 599,552 | 0.3451 |
| T-165 block graph bs32 beam64 | 4096 | 0.9988 | 0.9994 | 476.8 | 128.0 | 512.0 | 2,183,748 | 1,119,744 | 0.3424 |
| T-165 block graph bs64 beam32 | 4096 | 1.0000 | 1.0000 | 386.3 | 64.0 | 512.0 | 2,148,676 | 1,084,160 | 0.1784 |

## Verdict

Mixed signal: RNG/monotonic edges improve K=4096 beam-16 sensitivity, but same-beam coverage does not beat T-165 enough to clearly replace it.
