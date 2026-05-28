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
- HNTR/HNTQ/HNTL/HNTM binary format loaders;
- PyO3 bindings for insert, rebuild, split configuration, search, and stats.

Next up (roadmap):
- **T-124** — Index serialization writer (`write_legacy_index`)
- **T-122** — CLI `build` / `query` / `eval` subcommands
- **T-125** — GitHub Actions Python wheel CI (maturin)
