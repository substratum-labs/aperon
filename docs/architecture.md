# Core Architecture & Compression Primitives

Aperon is designed to treat agent-native vector memory like traditional databases treat LSM-based SSTables. To run within memory-constrained embedded environments, it employs advanced manifold compression and routing techniques.

---

## 1. Manifold-Adaptive Quantization (MAQ)

Standard vector databases apply uniform quantization (e.g., product quantization) across the entire dataset. In contrast, MAQ adapts to local manifold structures:
- **Local Grains**: Vectors are grouped into local topological grains.
- **Singular Value Decay**: For each grain, Aperon performs local Principal Component Analysis (PCA) to measure topological dimensionality.
- **Dynamic Bit Allocation**: Dimensions experiencing rapid singular value decay receive fewer quantization bits, whereas principal dimensions receive higher bitwidths.
- **Impact**: Reduces total HNSW index size by up to 90% without sacrificing recall on local manifolds.

---

## 2. Very Low Bitrate Residual Quantization (VLBRD)

VLBRD extends standard quantization by tracking the residual error of projected coordinates:
- **Residual Direction**: Encodes residual error vectors using 1-bit or 2-bit signatures.
- **Contiguous Block-SoA**: Compressed structures are packed in contiguous block Structure-of-Arrays (SoA) layout on disk.
- **SIMD Scanning**: Custom NEON/AVX2 kernels scan these low-bit SoA blocks directly in register registers, keeping distance calculations under 5 ns per vector.

---

## 3. Pivot-Prefix Posting Router

To scale metadata filtering combined with vector searches, Aperon integrates a Pivot-Prefix Posting Router:
- **Metadata Intersections**: Pre-filters candidates by intersecting symbolic/metadata postings before routing.
- **Centroid Landmarks**: Maps remaining candidates to inverted-file centroid landmarks.
- **Prefix Matching**: Employs weighted overlap prefix scoring to rapidly rank centroids, bypassing expensive distance calls for pruned partitions.

---

## 4. Tangent-Space HTLA Atlas

For highly non-linear vector manifolds, Aperon utilizes a Hierarchical Tangent Lattice Atlas:
- **Local Charts**: Builds a hierarchy of parent-local tangent coordinate charts.
- **Lattice Keys**: Encodes tangent spaces into compact `u128` lattice coordinate keys.
- **Beam Routing**: Routes search beams along lattice transitions, achieving fast routing coverage on curved topological spaces without pointer overhead.
