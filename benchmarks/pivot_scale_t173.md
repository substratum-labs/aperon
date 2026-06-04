# T-173 Pivot Routing Scale Validation

- Dataset path: `benchmarks/data/siftsmall`
- Available dataset: `siftsmall_base` with `10000` vectors; SIFT1M/T-151/T-152 data was not present locally.
- Final exact rerank: `final_nprobe=16`.
- HNSW/Faiss baseline: unavailable because T-152 is still `[PROPOSED]` and no local baseline artifact exists.
- Block graph baseline: compare against pinned T-165 artifact for K=4096 (`coverage@16=0.9944`, route `345.3 us/q`, workset `595,456 bytes/q`).

| Router | K | Block | Blocks | Pivots | Prefix | Top blocks | Pool | Mode | Coverage@16 | Pool@32 | Route us/q | Build s | Resident bytes | Workset bytes/q | Posting entries/q | Duplicate rate | Centroid evals/q |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pivot_prefix | 4096 | 64 | 64 | 64 | 8 | 32 | 512 | union | 0.9956 | 0.9950 | 57.5 | 0.102 | 2183432 | 804092 | 67.8 | 0.6528 | 1505.3 |
| pivot_prefix | 4096 | 32 | 128 | 64 | 8 | 64 | 512 | weighted | 0.9931 | 0.9941 | 45.4 | 0.170 | 2220040 | 842482 | 147.2 | 0.6643 | 1578.9 |
| dense_l_inf | 4096 | 64 | 64 | 64 | 0 | 32 | 512 | l_inf | 1.0000 | 1.0000 | 57.7 | 0.093 | 2195716 | 1098240 | 0.0 | 0.0000 | 2048.0 |
| pivot_prefix | 8192 | 64 | 128 | 96 | 8 | 48 | 768 | union | 0.9931 | 0.9906 | 70.9 | 0.339 | 4350216 | 1286085 | 93.7 | 0.5940 | 2414.1 |
| pivot_prefix | 8192 | 64 | 128 | 96 | 12 | 48 | 768 | union | 0.9969 | 0.9969 | 86.2 | 0.339 | 4353800 | 1560795 | 213.1 | 0.7538 | 2949.1 |
| pivot_prefix | 8192 | 32 | 256 | 96 | 8 | 96 | 768 | weighted | 0.9975 | 0.9962 | 81.9 | 0.714 | 4423432 | 1334496 | 203.0 | 0.6003 | 2506.6 |
| dense_l_inf | 8192 | 64 | 128 | 96 | 0 | 48 | 768 | l_inf | 0.9981 | 0.9975 | 92.2 | 0.335 | 4391428 | 1672192 | 0.0 | 0.0000 | 3072.0 |
| pivot_prefix | 10000 | 64 | 157 | 128 | 8 | 64 | 1024 | union | 0.9862 | 0.9819 | 70.3 | 0.513 | 5316372 | 1247976 | 85.6 | 0.5779 | 2307.8 |
| pivot_prefix | 10000 | 64 | 157 | 128 | 8 | 96 | 1024 | union | 0.9862 | 0.9819 | 69.7 | 0.517 | 5316372 | 1247976 | 85.6 | 0.5779 | 2307.8 |
| pivot_prefix | 10000 | 64 | 157 | 128 | 12 | 64 | 1024 | union | 0.9994 | 0.9988 | 102.6 | 0.525 | 5320768 | 1810207 | 197.1 | 0.7251 | 3404.4 |
| pivot_prefix | 10000 | 64 | 157 | 128 | 12 | 96 | 1024 | union | 0.9994 | 0.9991 | 103.5 | 0.521 | 5320768 | 1843118 | 197.1 | 0.7251 | 3468.6 |
| pivot_prefix | 10000 | 32 | 313 | 128 | 8 | 128 | 1024 | weighted | 0.9938 | 0.9928 | 89.8 | 1.077 | 5405604 | 1369656 | 184.8 | 0.5703 | 2543.4 |
| dense_l_inf | 10000 | 64 | 157 | 128 | 0 | 64 | 1024 | l_inf | 1.0000 | 1.0000 | 124.8 | 0.512 | 5386936 | 2237308 | 0.0 | 0.0000 | 4082.3 |

## Scale Interpretation

Within the available `siftsmall_base` scale sweep, pivot-prefix coverage bottoms out at `0.9862`. The largest observed posting fanout is `213.1` entries/query and the largest duplicate rate is `0.7538`. Prefix 8 is fast but becomes profile-sensitive at K=10000/block64; raising top blocks from 64 to 96 does not help because the touched block set is already smaller than the budget. Prefix 12 or a block32/top128 profile recovers coverage. Prefix 12 increases fanout and duplicate pressure, so it should be treated as a fallback profile rather than the default.

## Recommendation

NO-GO for direct AperonIndex integration: the default prefix8/block64 profile misses `coverage@16 >= 0.99` at K=10000 on the available local dataset, and increasing top-block budget alone does not fix it. Keep pivot-prefix as the candidate, but require planner fallback to prefix12, block32 profiles, or dense `l_inf`, and rerun on SIFT1M/T-152 before integration.

## Verdict

NO-GO: local scale sweep found coverage or duplicate-rate failure.
