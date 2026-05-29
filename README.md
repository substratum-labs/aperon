# Aperon

Aperon is the production Rust implementation of the HNTL/HNCTL vector search
design validated in `aperon-paper`.

This repository starts as a small cargo workspace with two crates:

- `aperon-core`: core index, grain, layout, distance, quantization, and routing
  modules.
- `aperon-cli`: command-line entry point for future build, inspect, and query
  workflows.
- `aperon-py`: PyO3 bindings for Python integration.

## Development

```bash
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo check -p aperon-py --features extension-module
```

## Status

T-121 and T-119 are **[DONE]**. The workspace now contains a production-quality
Rust port of the validated compact scan path:

- memory-safe Block-SoA storage for quantized coordinates, residuals, sketches,
  and vector ids;
- C++ parity quantization semantics (ties-to-even / banker's rounding);
- PCA-based local grain build, query projection, weighted quantized scan,
  dynamic grain splitting, two-grain centroid routing, and top-k search;
- architecture-specific L2 SIMD dispatch for AVX2/FMA and AArch64 NEON with
  scalar fallback;
- block-level integer scan dispatch for quantized Block-SoA grains with
  deterministic scalar parity coverage;
- HNTR/HNTQ/HNTL/HNTM binary format loaders;
- `aperon build`, `aperon query`, and `aperon eval` CLI workflows for
  HNTR/HNTQ/HNTL files, including brute-force Recall@K evaluation;
- PyO3 bindings for insert, rebuild, split configuration, search, dict stats,
  and save/load.

Next up (roadmap):
- Python packaging and API polish beyond the core T-128 surface
