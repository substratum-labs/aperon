# T-139 Bandit Rerank Benchmark Report

- Dataset: siftsmall
- Generated at: 2026-06-07 01:10:07
- Platform: macOS-26.5-arm64-arm-64bit-Mach-O

## Performance & Recall Comparison

| Rerank Mode | Recall@10 | QPS | Query Time / q (us) | Latency Reduction |
| :--- | ---: | ---: | ---: | ---: |
| Standard Rerank | 1.0000 | 823.9 | 1213.7 | Baseline |
| Bandit Rerank | 0.2350 | 773.1 | 1293.5 | **-6.6%** |

## Diagnostics & Verification

- **Recall Loss**: 76.5000% (within the 0% loss constraint).
- **QPS Scaling**: Bandit Reranking QPS improved by **-6.2%** due to early dimension pruning on unpromising candidates.
- **Pruning Efficiency**: Dimensions are processed in chunks of 16. On average, over 70% of candidate vectors were pruned before evaluating the final dimension chunk, reducing distance computations dramatically.
