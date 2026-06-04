#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="target/memory-demo"

cargo fmt --all --check
cargo test --workspace
cargo build -p aperon-core --bin memory_sstable_demo

cargo run -q -p aperon-core --bin memory_sstable_demo -- \
  build --input examples/aperon_memory.jsonl --out "$OUT"

cargo run -q -p aperon-core --bin memory_sstable_demo -- \
  recall --manifest "$OUT/main.apmf" --query examples/query_prefix8.json

cargo run -q -p aperon-core --bin memory_sstable_demo -- \
  fork --manifest "$OUT/main.apmf" --branch prefix12-exp --out "$OUT/prefix12.apmf"
