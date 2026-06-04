# T-172 Quantized Pivot Routing Prototype

- Dataset path: `benchmarks/data/siftsmall`
- Dense rows use T-167 `l_inf` scoring with final exact rerank.
- Prefix rows use T-168 pivot-prefix routing; packed row changes ID/token width only, so routing semantics match baseline.

| Layout | Coverage@16 | Pool coverage@32 | Route us/q | Resident bytes | Workset bytes/q | Posting bytes/block | Quant max abs err | Quant mean abs err | Hot allocs |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense_l_inf_f32 | 1.0000 | 1.0000 | 61.4 | 2195716 | 1098240 | 0.00 | 0.000000 | 0.000000 | 0 |
| dense_l_inf_fp16 | 1.0000 | 1.0000 | 106.4 | 2187524 | 1090048 | 0.00 | 0.250000 | 0.066267 | 0 |
| dense_l_inf_uint16 | 1.0000 | 1.0000 | 102.7 | 2187532 | 1090048 | 0.00 | 0.004761 | 0.002367 | 0 |
| prefix_baseline_u32 | 0.9981 | 0.9981 | 41.8 | 2183432 | 806060 | 44.06 | 0.000000 | 0.000000 | 0 |
| prefix_packed_u16 | 0.9981 | 0.9981 | 41.5 | 2174216 | 800940 | 28.06 | 0.000000 | 0.000000 | 0 |

## Recommendation

Use u16 block/posting IDs for pivot-prefix layouts when `blocks <= 65535`, keep f32 pivots and centroid vectors for now, and use uint16 dense `l_inf` signatures as the fallback layout. fp16 is viable but uint16 has lower signature error at the same 2-byte footprint.

## Verdict

PASS: compact pivot layouts preserve the T-167/T-168 positive signal on siftsmall.
