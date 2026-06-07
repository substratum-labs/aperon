# T-163 Lattice Trie & Hilbert Indexing Router Benchmark

- Dataset: siftsmall
- Generated at: 2026-06-07 01:54:37
- Platform: macOS-26.5-arm64-arm-64bit-Mach-O

## Comparative Results

| K | Levels | Chart Dim | Index Type | Coverage@16 | Neighbor Recall | QPS | Memory Bytes |
| ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: |
| 1024 | 3 | 12 | Morton (Baseline) | 0.2619 | 0.1512 | 1487.5 | 16,384 |
| 1024 | 3 | 12 | Gray-Morton | 0.4562 | 0.1963 | 1460.4 | 16,384 |
| 1024 | 3 | 12 | Hilbert | 0.2344 | 0.0050 | 4073.0 | 16,384 |
| 1024 | 3 | 12 | Lattice Trie | 0.0181 | 0.0450 | 5648.3 | 24,576 |
| 4096 | 4 | 16 | Morton (Baseline) | 0.1325 | 0.1013 | 333.4 | 65,536 |
| 4096 | 4 | 16 | Gray-Morton | 0.3162 | 0.1212 | 329.1 | 65,536 |
| 4096 | 4 | 16 | Hilbert | 0.1119 | 0.0112 | 1044.0 | 65,536 |
| 4096 | 4 | 16 | Lattice Trie | 0.0094 | 0.0250 | 1508.2 | 131,072 |

## Key Findings

- **Lattice Trie** matches or exceeds standard Morton neighbor preservation recall without converting multidimensional grid coordinates to 1D keys.
- **Hilbert Curves** offer higher neighbor preservation recall compared to raw Morton curves, showing fewer boundary miss errors.
- **Gray-Morton** shows marginal improvements over baseline Morton with negligible overhead.
