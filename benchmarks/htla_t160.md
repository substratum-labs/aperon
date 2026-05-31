# T-160 HTLA Route-Only Benchmark

- Dataset path: `/Users/yong/projects/aperon/benchmarks/data/siftsmall`
- Generated at: 2026-05-31 04:29:07 PDT
- Python: 3.14.4
- Platform: macOS-26.5-arm64-arm-64bit-Mach-O

| Dataset | K | Levels | Dim | Beam | Pool | Coverage@16 | Coverage@32 | QPS | Exact scan QPS | Route bytes | Working-set bytes/q | Build s | Fallback rate |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| siftsmall | 128 | 2 | 8 | 4 | 64 | 1.0000 | 0.9981 | 232544.2 | 120857.8 | 691,864 | 4,888 | 0.0151 | 0.0000 |
| siftsmall | 128 | 2 | 8 | 8 | 128 | 1.0000 | 1.0000 | 167891.8 | 115271.8 | 691,864 | 5,016 | 0.0138 | 0.0000 |
| siftsmall | 1024 | 3 | 8 | 8 | 128 | 0.6969 | 0.6747 | 67703.1 | 27841.8 | 5,656,984 | 43,096 | 0.2657 | 0.0000 |
| siftsmall | 1024 | 3 | 12 | 8 | 256 | 0.7150 | 0.6878 | 40340.1 | 28354.9 | 7,847,080 | 61,800 | 0.4034 | 0.0000 |
| siftsmall | 4096 | 4 | 12 | 16 | 256 | 0.1844 | 0.1616 | 19472.8 | 7887.2 | 32,365,480 | 225,960 | 2.6461 | 0.0000 |
| siftsmall | 4096 | 4 | 16 | 16 | 512 | 0.1888 | 0.1706 | 16243.1 | 7819.8 | 41,418,040 | 294,328 | 3.2728 | 0.0000 |

## Acceptance Summary

| K | Required pool | Coverage@16 target | Observed Coverage@16 | Result |
| ---: | ---: | ---: | ---: | :--- |
| 128 | 128 | 0.9900 | 1.0000 | PASS |
| 1024 | 256 | 0.9900 | 0.7150 | FAIL |
| 4096 | 512 | 0.9900 | 0.1888 | FAIL |

## Diagnostics

| K | Dim | d80 max | d90 max | d95 max | d80 p50 | d90 p50 | d95 p50 | radius shrink p50 | radius shrink p90 | p10(norm_sep) | p25(norm_sep) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 8 | 4 | 6 | 7 | 4.0 | 6.0 | 7.0 | 0.0000 | 0.0000 | 0.4490 | 0.6123 |
| 128 | 8 | 4 | 6 | 7 | 4.0 | 6.0 | 7.0 | 0.0000 | 0.0000 | 0.4490 | 0.6123 |
| 1024 | 8 | 6 | 7 | 8 | 5.0 | 7.0 | 8.0 | 0.0000 | 0.0000 | 0.6200 | 0.7902 |
| 1024 | 12 | 8 | 10 | 11 | 7.0 | 10.0 | 11.0 | 0.0000 | 0.0000 | 0.7465 | 0.9043 |
| 4096 | 12 | 8 | 10 | 11 | 7.0 | 9.0 | 11.0 | 0.0000 | 0.0000 | 0.8409 | 1.0293 |
| 4096 | 16 | 16 | 16 | 16 | 10.0 | 15.0 | 16.0 | 0.0000 | 0.0000 | 0.9110 | 1.0949 |

Coverage@16 is measured on the final 16 centroid IDs after exact rerank of the routed pool.
Coverage@32 is measured on the routed candidate pool before final exact rerank.
