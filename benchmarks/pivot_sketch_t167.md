# T-167 Pivot-Sketch / LAESA-Style Block Router Prototype

- Dataset path: `benchmarks/data/siftsmall`
- Generated at: 2026-05-31 16:52:10 PDT
- Python: 3.14.4
- Platform: macOS-26.5-arm64-arm-64bit-Mach-O
- Final exact rerank: `final_nprobe=16` over the bounded centroid candidate pool.
- Centroids use the T-162/T-165 deterministic `linspace` sample policy.
- Blocks reuse the T-165 deterministic balanced block layout shape: contiguous int32 payload plus offsets.
- Routing computes query-to-pivot distances, dense-scores every block signature row with the configured signature metric (`l1`, `l_inf`, or `l2`), scans selected blocks contiguously, then exact-reranks centroids.

## Minimum Matrix

| K | Block size | Blocks | Pivots | Top blocks | Pool | Signature | Coverage@16 | Pool coverage@32 | QPS | Route us/q | Pivot evals/q | Block rows/q | Centroid evals/q | Selected blocks/q | Candidates/q | Resident bytes | Workset bytes/q | Build s | Fallback |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 16 | 64 | 16 | 16 | 256 | l1 | 0.7650 | 0.7612 | 14243.3 | 70.2 | 16.0 | 64.0 | 256.0 | 16.0 | 256.0 | 573,700 | 143,872 | 0.0607 | 0.0000 |
| 1024 | 32 | 32 | 16 | 16 | 256 | l1 | 0.9762 | 0.9812 | 9841.5 | 101.6 | 16.0 | 32.0 | 512.0 | 16.0 | 256.0 | 555,140 | 272,640 | 0.0346 | 0.0000 |
| 1024 | 32 | 32 | 32 | 16 | 256 | l1 | 0.9825 | 0.9866 | 9419.1 | 106.2 | 32.0 | 32.0 | 512.0 | 16.0 | 256.0 | 565,380 | 282,880 | 0.0329 | 0.0000 |
| 1024 | 32 | 32 | 32 | 16 | 256 | l_inf | 0.9988 | 0.9991 | 9286.6 | 107.7 | 32.0 | 32.0 | 512.0 | 16.0 | 256.0 | 565,380 | 282,880 | 0.0335 | 0.0000 |
| 4096 | 16 | 256 | 32 | 32 | 512 | l1 | 0.6306 | 0.6300 | 7170.6 | 139.5 | 32.0 | 256.0 | 512.0 | 32.0 | 512.0 | 2,294,788 | 313,344 | 1.0166 | 0.0000 |
| 4096 | 32 | 128 | 32 | 32 | 512 | l1 | 0.7281 | 0.7344 | 5081.9 | 196.8 | 32.0 | 128.0 | 1024.0 | 32.0 | 512.0 | 2,212,356 | 558,080 | 0.4835 | 0.0000 |
| 4096 | 64 | 64 | 32 | 32 | 512 | l1 | 0.9869 | 0.9872 | 2983.1 | 335.2 | 32.0 | 64.0 | 2048.0 | 32.0 | 512.0 | 2,171,140 | 1,073,664 | 0.2463 | 0.0000 |
| 4096 | 64 | 64 | 32 | 32 | 512 | l_inf | 0.9981 | 0.9984 | 3073.3 | 325.4 | 32.0 | 64.0 | 2048.0 | 32.0 | 512.0 | 2,171,140 | 1,073,664 | 0.2234 | 0.0000 |

## Sensitivity Rows

| K | Block size | Blocks | Pivots | Top blocks | Pool | Signature | Coverage@16 | Pool coverage@32 | QPS | Route us/q | Pivot evals/q | Block rows/q | Centroid evals/q | Selected blocks/q | Candidates/q | Resident bytes | Workset bytes/q | Build s | Fallback |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 32 | 128 | 64 | 32 | 512 | l1 | 0.7063 | 0.7041 | 4895.0 | 204.3 | 64.0 | 128.0 | 1024.0 | 32.0 | 512.0 | 2,245,124 | 590,848 | 0.4579 | 0.0000 |
| 4096 | 32 | 128 | 32 | 16 | 512 | l1 | 0.5012 | 0.5106 | 7727.2 | 129.4 | 32.0 | 128.0 | 512.0 | 16.0 | 512.0 | 2,212,356 | 295,936 | 0.4835 | 0.0000 |
| 4096 | 32 | 128 | 32 | 64 | 512 | l1 | 0.9731 | 0.9784 | 2900.5 | 344.8 | 32.0 | 128.0 | 2048.0 | 64.0 | 512.0 | 2,212,356 | 1,082,368 | 0.4835 | 0.0000 |
| 4096 | 32 | 128 | 32 | 64 | 512 | l_inf | 0.9944 | 0.9931 | 2935.6 | 340.6 | 32.0 | 128.0 | 2048.0 | 64.0 | 512.0 | 2,212,356 | 1,082,368 | 0.4504 | 0.0000 |
| 4096 | 32 | 128 | 32 | 64 | 512 | l2 | 0.9925 | 0.9931 | 2915.0 | 343.0 | 32.0 | 128.0 | 2048.0 | 64.0 | 512.0 | 2,212,356 | 1,082,368 | 0.4503 | 0.0000 |
| 4096 | 64 | 64 | 32 | 32 | 512 | l2 | 0.9950 | 0.9944 | 3055.7 | 327.3 | 32.0 | 64.0 | 2048.0 | 32.0 | 512.0 | 2,171,140 | 1,073,664 | 0.2248 | 0.0000 |
| 4096 | 32 | 128 | 32 | 32 | 512 | fp16 | 0.7281 | 0.7344 | 5024.6 | 199.0 | 32.0 | 128.0 | 1024.0 | 32.0 | 512.0 | 2,204,164 | 549,888 | 0.4623 | 0.0000 |
| 4096 | 32 | 128 | 32 | 32 | 512 | uint16 | 0.7281 | 0.7344 | 4976.7 | 200.9 | 32.0 | 128.0 | 1024.0 | 32.0 | 512.0 | 2,204,420 | 549,888 | 0.4515 | 0.0000 |

## Baseline Comparison

| Router | K | Coverage@16 | Pool coverage@32 | Route us/q | Route evals/q | Candidates/q | Resident bytes | Workset bytes/q | Build s |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T-162 overlap PCA spill x4 | 1024 | 1.0000 | 1.0000 | 445.1 | 1056.0 | 256.0 | 1,436,464 | 592,944 | 0.0689 |
| T-165 block graph best practical | 1024 | 0.9994 | 0.9984 | 127.1 | 32.0 | 256.0 | 545,956 | 279,936 | 0.0285 |
| T-166 landmark multi-probe | 1024 | 0.8650 | 0.8006 | 37.8 | 64.0 | 105.0 | 41,220 | 87,975 | 0.0567 |
| T-162 overlap PCA spill x4 | 4096 | 0.9919 | 0.9900 | 699.0 | 1296.0 | 512.0 | 7,505,328 | 751,680 | 0.2483 |
| T-165 block graph best practical | 4096 | 0.9944 | 0.9947 | 345.3 | 128.0 | 512.0 | 2,183,748 | 595,456 | 0.3424 |
| T-166 landmark multi-probe | 4096 | 0.8919 | 0.8528 | 78.6 | 256.0 | 215.9 | 164,868 | 244,608 | 0.9448 |

## Verdict

Positive signal: pivot-sketch routing reaches coverage@16 >= 0.99 at both K values and is route-time competitive with T-165 while using dense sequential signature scans.

## Notes

- `l1` stores fp32 pivot-distance signatures and scores `sum(abs(block_sig - query_sig))`.
- `l_inf` stores fp32 signatures and scores `max(abs(block_sig - query_sig))`, the LAESA/AESA-style pivot lower-bound score.
- `l2` stores fp32 signatures and scores Euclidean distance between pivot-distance signatures.
- `fp16` and `uint16` sensitivity rows change only the resident signature format; scoring dequantizes/casts in this Python prototype and currently uses the L1 score path.
- Workset bytes estimate the dense pivot reads, dense block-signature scan, selected contiguous centroid scan, and score scratch. It is a prototype accounting model, not a hardware counter.
