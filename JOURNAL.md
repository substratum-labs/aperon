# JOURNAL.md - Aperon

## Developer Deep Thoughts & Design Notes

### 2026-06-26: Rethinking Vector Indexing for AI Agents
*   **The Problem with Graphs (HNSW):** Proximity graphs dominate high-dimensional search but they have a heavy "pointer tax". Chasing memory pointers during graph traversal causes severe L1/L2 cache misses (up to $8.72\%$) and limits throughput. Additionally, graphs make it difficult to perform transactional database operations like instant zero-copy branching (which agents need for parallel counterfactual reasoning) and snapshotting, because updates trigger global re-wiring.
*   **The HNTL Philosophy:** 
    1.  *Manifold Geometries:* High-dimensional embeddings often lie on low-dimensional local manifolds. HNTL exploits this by splitting the space into local "grains" and projecting vectors onto local tangent spaces (tangent PCA), reducing a 768-dimension problem to 16-32 dimensions.
    2.  *Hardware Alignment:* Instead of graph-chasing, we do a sequential scan over a contiguous, pointerless **Block-SoA** memory layout. Since the layout is cache-aligned, the CPU can read the coordinates linearly, auto-vectorizing via SIMD registers (NEON, AVX2, AVX-512) with near-zero cache misses ($<0.01\%$).
    3.  *Immutable SSTables:* Since grains are geographically isolated and self-contained, updates do not trigger global graph updates. They map directly to Log-Structured Memory Segments (SSTables). This allows instant copy-on-write segment forks (branching) for parallel agent simulations.

### Next Directions
*   **GPU Block-SoA Scan:** Porting the Block-SoA sequential scan to warp-level parallel primitives. Sequential linear blocks are extremely friendly to GPU thread blocks since threads can scan lanes in parallel without divergence.
*   **Dynamic Partition Splitting:** Refining the lazy 2-means splitting protocol when grains grow beyond $N_{\max}$.
