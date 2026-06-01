# T-166 Landmark Multi-Probe Router Prototype

- Dataset path: `/Users/yong/projects/aperon/benchmarks/data/siftsmall`
- Generated at: 2026-05-31 15:57:52 PDT
- Python: 3.14.4
- Platform: macOS-26.5-arm64-arm-64bit-Mach-O
- Final exact rerank: `final_nprobe=16` over the bounded candidate pool.
- Centroids use the same deterministic linspace sampling policy as T-162.
- Landmark sets use deterministic balanced k-means representatives over sampled centroids.
- Postings are fixed-cap int32 lists per landmark; `top-2` assignment stores overlapping centroid IDs and duplicate rate is measured after top-R probe union.

## Minimum Matrix

| K | Landmarks | Assignment | Probes | Posting cap | Pool | Coverage@16 | Pool coverage@32 | QPS | Route us/q | Landmark evals/q | Posting entries/q | Candidates/q | Duplicate rate | Route bytes | Workset bytes/q | Build s | Fallback | Retained entries | Retained rate |
| ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 64 | top-1 | 4 | 64 | 256 | 0.7450 | 0.6406 | 41463.7 | 24.1 | 64.0 | 63.9 | 63.9 | 0.0000 | 37,124 | 66,251 | 0.0581 | 0.0000 | 1,024 | 1.0000 |
| 1024 | 64 | top-2 | 4 | 64 | 256 | 0.8650 | 0.8006 | 26438.7 | 37.8 | 64.0 | 130.4 | 105.0 | 0.1951 | 41,220 | 87,975 | 0.0567 | 0.0000 | 2,048 | 1.0000 |
| 1024 | 128 | top-1 | 4 | 64 | 256 | 0.6012 | 0.4863 | 22616.1 | 44.2 | 128.0 | 32.5 | 32.5 | 0.0000 | 70,148 | 82,555 | 0.1204 | 0.0000 | 1,024 | 1.0000 |
| 1024 | 128 | top-2 | 4 | 64 | 256 | 0.7444 | 0.6469 | 36016.6 | 27.8 | 128.0 | 66.0 | 55.5 | 0.1578 | 74,244 | 94,696 | 0.1164 | 0.0000 | 2,048 | 1.0000 |
| 4096 | 128 | top-1 | 8 | 128 | 512 | 0.8775 | 0.8466 | 13609.8 | 73.5 | 128.0 | 256.9 | 256.9 | 0.0000 | 82,436 | 200,141 | 0.4792 | 0.0000 | 4,096 | 1.0000 |
| 4096 | 128 | top-2 | 8 | 128 | 512 | 0.9500 | 0.9344 | 8382.6 | 119.3 | 128.0 | 519.2 | 402.6 | 0.2241 | 98,820 | 277,441 | 0.4439 | 0.0000 | 8,192 | 1.0000 |
| 4096 | 256 | top-1 | 8 | 128 | 512 | 0.7931 | 0.7347 | 19784.0 | 50.5 | 256.0 | 128.7 | 128.7 | 0.0000 | 148,484 | 198,495 | 1.0596 | 0.0000 | 4,096 | 1.0000 |
| 4096 | 256 | top-2 | 8 | 128 | 512 | 0.8919 | 0.8528 | 12719.7 | 78.6 | 256.0 | 263.9 | 215.9 | 0.1815 | 164,868 | 244,608 | 0.9448 | 0.0000 | 8,192 | 1.0000 |

## Sensitivity

| K | Landmarks | Assignment | Probes | Posting cap | Pool | Coverage@16 | Pool coverage@32 | QPS | Route us/q | Landmark evals/q | Posting entries/q | Candidates/q | Duplicate rate | Route bytes | Workset bytes/q | Build s | Fallback | Retained entries | Retained rate |
| ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

## Baselines

| K | Router | Coverage@16 | Pool coverage@32 | Route us/q | Evals/q | Candidates/q | Resident bytes | Workset bytes/q | Build s |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | T-162 overlap PCA spill x4 | 1.0000 | 1.0000 | 445.1 | 1056.0 | 256.0 | 1,436,464 | 592,944 | 0.0689 |
| 4096 | T-162 overlap PCA spill x4 | 0.9919 | 0.9900 | 699.0 | 1296.0 | 512.0 | 7,505,328 | 751,680 | 0.2483 |
| 1024 | T-165 block graph size32 M8 | 0.9994 | 0.9984 | 127.1 | 32.0 | 256.0 | 545,956 | 279,936 | 0.0285 |
| 4096 | T-165 block graph size32 M8 | 0.9944 | 0.9947 | 345.3 | 128.0 | 512.0 | 2,183,748 | 595,456 | 0.3424 |

## Interpretation

Negative signal: landmark postings stay simple and regular, but they miss high-K boundary cases under the requested pool caps while T-162/T-165 baselines clear coverage@16 >= 0.99.
