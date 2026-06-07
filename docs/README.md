# Aperon Documentation System

Welcome to the **Aperon** documentation system. Aperon is a compact, agent-native vector and semantic memory engine designed to run with minimal memory overhead.

---

## Table of Contents

### 1. [Getting Started & Installation](../README.md#🚀-quickstart-from-clone)
How to clone, build the workspace, set up Python bindings, and build your first grain-local index.

### 2. [Indexing Modes (Mode A & Mode B)](modes.md)
Detailed walkthroughs and usage examples of Aperon's two operational modes:
* **Mode A**: Self-contained compressed search (no raw vectors needed at query time).
* **Mode B**: Hot semantic filter with raw-vector reranking from colder storage tiers.

### 3. [Memory SSTable & Benchmarks](sstable.md)
LSM-style memory space specifications, multi-path query planning, metrics definitions, and deterministic testing scenarios.

### 4. [Core Architecture & Quantization Primitives](architecture.md)
Deep dive into Aperon's mathematical compression algorithms, including Manifold-Adaptive Quantization (MAQ), VLBRD low-bit quantization, Pivot-Prefix routing, and Hierarchical Tangent Lattice Atlas (HTLA) routing.

### 5. [Python API Reference](python_api.md)
Reference guide for Python developers using the `AperonIndex`, `HlrRouter`, and `HtlaRouter` PyO3 bindings.

### 6. [Rust API Reference](rust_api.md)
Reference guide for Rust developers embedding `MemorySpace`, `MemoryQueryPlanner`, and `MemorySegment` into agent systems.

### 7. [Compatibility & Deprecation Policy](compatibility.md)
Guidelines for binary serialization compatibility and search FFI deprecation paths.

### 8. [Release Checklist](release.md)
Step-by-step verification, testing, and deployment procedure for publishing new crates and wheels.

