# T-171 Rust Pivot-Prefix Route Kernel Prototype

- Dataset path: `benchmarks/data/siftsmall`
- Final exact rerank: `final_nprobe=16`
- Hot query allocation counter starts after router and scratch construction.

| Router | K | Block size | Blocks | Pivots | Prefix | Top blocks | Pool | Mode | Coverage@16 | Pool coverage@32 | QPS | Route us/q | Posting entries/q | Duplicate rate | Selected blocks/q | Centroid evals/q | Resident bytes | Workset bytes/q | Hot allocs |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T-171 pivot prefix | 1024 | 32 | 32 | 32 | 4 | 16 | 256 | union | 0.9575 | 0.9534 | 120236.0 | 8.3 | 17.1 | 0.4535 | 9.5 | 302.7 | 562440 | 171575 | 0 |
| T-171 pivot prefix | 1024 | 32 | 32 | 32 | 8 | 16 | 256 | union | 0.9962 | 0.9969 | 73239.1 | 13.7 | 71.5 | 0.7752 | 15.8 | 507.2 | 563336 | 276618 | 0 |
| T-171 pivot prefix | 1024 | 64 | 16 | 16 | 8 | 16 | 256 | union | 0.9988 | 0.9994 | 70712.4 | 14.1 | 64.0 | 0.8625 | 8.8 | 563.2 | 545864 | 296976 | 0 |
| T-171 pivot prefix | 4096 | 32 | 128 | 64 | 4 | 32 | 512 | union | 0.9519 | 0.9397 | 48915.6 | 20.4 | 34.9 | 0.3290 | 23.4 | 749.8 | 2216456 | 417100 | 0 |
| T-171 pivot prefix | 4096 | 32 | 128 | 64 | 8 | 32 | 512 | union | 0.9856 | 0.9778 | 35109.9 | 28.5 | 142.2 | 0.6548 | 32.0 | 1023.4 | 2220040 | 558027 | 0 |
| T-171 pivot prefix | 4096 | 64 | 64 | 64 | 8 | 32 | 512 | union | 0.9981 | 0.9981 | 24410.6 | 41.0 | 68.0 | 0.6538 | 23.6 | 1509.1 | 2183432 | 806060 | 0 |
| T-171 pivot prefix | 4096 | 32 | 128 | 64 | 8 | 32 | 512 | weighted | 0.9856 | 0.9772 | 35187.0 | 28.4 | 142.2 | 0.6548 | 32.0 | 1023.4 | 2220040 | 558027 | 0 |
| T-171 pivot prefix | 4096 | 32 | 128 | 64 | 8 | 16 | 512 | weighted | 0.9106 | 0.8953 | 67595.2 | 14.8 | 142.2 | 0.6548 | 16.0 | 512.0 | 2220040 | 296211 | 0 |
| T-171 pivot prefix | 4096 | 32 | 128 | 64 | 8 | 64 | 512 | weighted | 0.9950 | 0.9944 | 23025.9 | 43.4 | 142.2 | 0.6548 | 49.0 | 1568.6 | 2220040 | 837211 | 0 |
| T-171 pivot prefix | 4096 | 32 | 128 | 64 | 12 | 32 | 512 | union | 0.9850 | 0.9797 | 34615.2 | 28.9 | 321.8 | 0.8106 | 32.0 | 1024.0 | 2223624 | 559393 | 0 |
| T-171 pivot prefix | 4096 | 32 | 128 | 64 | 12 | 32 | 512 | weighted | 0.9881 | 0.9831 | 34949.9 | 28.6 | 321.8 | 0.8106 | 32.0 | 1024.0 | 2223624 | 559393 | 0 |
| T-171 dense l_inf fallback | 4096 | 64 | 64 | 64 | 0 | 32 | 512 | l_inf | 1.0000 | 1.0000 | 17854.5 | 56.0 | 0.0 | 0.0000 | 32.0 | 2048.0 | 2195716 | 1098240 | 0 |

## Verdict

PASS: Rust route kernel meets the T-171 coverage and hot-query allocation gates.
