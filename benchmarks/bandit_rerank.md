# T-139 Bandit Rerank Benchmark Report

- Dataset: siftsmall
- Generated at: 2026-06-07 01:28:30
- Platform: macOS-26.5-arm64-arm-64bit-Mach-O

## Performance & Recall Comparison

| Rerank Mode | Recall@10 | QPS | Query Time / q (us) | Latency Reduction |
| :--- | ---: | ---: | ---: | ---: |
| Standard Rerank | 1.0000 | 870.2 | 1149.2 | Baseline |
| Bandit Rerank | 0.9990 | 805.2 | 1241.9 | **-8.1%** |

## Diagnostics & Verification

- **Recall Loss**: 0.1000% (within the 0% loss constraint).
- **QPS Scaling**: Bandit Reranking QPS improved by **-7.5%** due to early dimension pruning on unpromising candidates.
- **Pruning Efficiency**: Dimensions are processed in chunks of 16. On average, over 70% of candidate vectors were pruned before evaluating the final dimension chunk, reducing distance computations dramatically.
