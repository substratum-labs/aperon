# RFC: GPU-Native Vector Scan Engine & Compute Contracts

## Status
- **ID**: `RFC-234-A`
- **Status**: `PROPOSED`
- **Owner**: Gemini
- **Reviewer**: Yong
- **Target Project**: `aperon-core` / `aperon`

---

## 1. Context & Motivation

Aperon's vector search engine targets extremely high QPS (Queries Per Second) and low latency by eliminating the pointer tax (e.g., using contiguous Block-SoA memory layout instead of pointer-heavy graphs like HNSW). While our CPU SIMD kernels (AVX-512 and ARM NEON) provide state-of-the-art single-thread performance, they face two key bottlenecks during high-throughput workloads:

1.  **Memory Bandwidth Saturation**: CPU cores sharing a memory controller quickly saturate memory bandwidth when scanning millions of high-dimensional vectors in parallel.
2.  **Batch Multi-Query Latency**: In agentic multi-agent workloads, queries are often submitted in batches (e.g., concurrent memory retrievals or massive codebook training cycles). CPU SIMD cores struggle to maintain low sub-millisecond latencies under high batch sizes (e.g., batch size > 1000).

To resolve these constraints, this RFC proposes a **GPU-Native Vector Scan Engine** built on `wgpu` (Rust's portable, zero-copy WebGPU implementation). By executing massive parallel scans directly on the GPU compute pipeline, we tap into high-bandwidth memory (HBM/GDDR) and unified memory architectures (like Apple Silicon) while bypassing CPU instruction and bus transfer overhead.

---

## 2. Core Architecture & Memory Layout

### 2.1 Pointerless Zero-Copy Buffer Mapping
Aperon segments are laid out on disk as contiguous raw float slices in a custom `.apms` format. 
To achieve zero-copy execution:
- We mapping the `.apms` file directly to host memory using `mmap`.
- The GPU buffer is configured as a `MAP_READ | COPY_DST` resource (or host-shared on unified memory platforms).
- Shaders read directly from a 1D storage array mapping to the vector database, avoiding intermediate deserialization or pointer reconstruction.

### 2.2 Storage Layout (Block-SoA Struct)
The GPU buffer stores vectors in blocks of size $B = 64$ to align with GPU warp/subgroup sizes (typically 32 or 64 threads). 
Within each block, vectors are stored in Structure of Arrays (SoA) layout to allow coalesced memory reads across adjacent GPU threads:

```wgsl
struct VectorDbBlock {
    // Dimension-major layout within a block
    // block_data[d * B + i] represents dimension d, vector index i inside the block
    block_data: array<f32, D_DIM * 64>,
}

@group(0) @binding(0) var<storage, read> db: array<VectorDbBlock>;
@group(0) @binding(1) var<storage, read> queries: array<f32>; // packed queries [Q, D_DIM]
@group(0) @binding(2) var<storage, read_write> results: array<f32>; // output scores [Q, N_VECTORS]
```

---

## 3. GPU Compute Shader & Workgroup Dispatch Contracts

We utilize WGSL (WebGPU Shading Language) for our compute shaders, ensuring portability across macOS (Metal), Windows (Vulkan/DX12), and Linux.

### 3.1 Workgroup Dispatch Strategy
- **Workgroup Size**: $64 \times 1 \times 1$. Each workgroup is mapped to a single block database containing 64 vectors.
- **Thread Mapping**: Thread `local_invocation_id.x` (from 0 to 63) is assigned to scan a single vector index $i$ within that block.
- **Memory Coalescing**: When loading dimension $d$, all 64 threads load adjacent elements `block_data[d * 64 + i]` concurrently, matching the hardware's memory bus width and achieving maximum memory throughput.

### 3.2 Compute Shader Contract (WGSL draft)
```wgsl
// WGSL Compute Shader for L2 / Cosine Vector Scan
override D_DIM: u32; // Specialized constant for dimension
override B_SIZE: u32 = 64u;

struct VectorDbBlock {
    data: array<f32>,
}

@group(0) @binding(0) var<storage, read> db_buf: array<f32>; // Flat layout: [NumBlocks * D_DIM * 64]
@group(0) @binding(1) var<storage, read> query_buf: array<f32>; // Flat layout: [NumQueries * D_DIM]
@group(0) @binding(2) var<storage, read_write> score_buf: array<f32>; // Flat layout: [NumQueries * NumVectors]

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let vector_idx = global_id.x; // Global vector index to scan
    let local_idx = local_id.x;   // Index within the block
    let block_idx = workgroup_id.x;
    
    // Outer loop over queries in the batch
    let num_queries = arrayLength(&query_buf) / D_DIM;
    
    for (var q: u32 = 0u; q < num_queries; q = q + 1u) {
        let q_offset = q * D_DIM;
        var dist: f32 = 0.0;
        
        // Loop over dimensions, exploiting coalesced SoA memory loads
        for (var d: u32 = 0u; d < D_DIM; d = d + 1u) {
            let q_val = query_buf[q_offset + d];
            
            // SoA block index: block_idx * D_DIM * 64 + d * 64 + local_idx
            let db_val = db_buf[block_idx * D_DIM * 64u + d * 64u + local_idx];
            
            let diff = q_val - db_val;
            dist = dist + (diff * diff);
        }
        
        // Write the calculated score
        score_buf[q * arrayLength(&db_buf) / D_DIM + vector_idx] = dist;
    }
}
```

---

## 4. Rust API Contracts (`GpuScanEngine` Trait)

We define the primary trait `GpuScanEngine` to govern all GPU vector scanning activities in `aperon-core`:

```rust
use std::future::Future;
use std::pin::Pin;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuDistanceMetric {
    L2,
    Cosine,
    DotProduct,
}

#[derive(Debug, Clone)]
pub struct GpuScanResult {
    pub vector_id: u64,
    pub score: f32,
}

pub trait GpuScanEngine: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    /// Initializes the GPU adapter, device, command queue, and compiles WGSL shaders.
    fn init() -> impl Future<Output = Result<Self, Self::Error>> + Send
    where
        Self: Sized;

    /// Binds the memory-mapped vector database slice to a GPU storage buffer.
    /// In unified memory systems, this is a zero-copy pointer pass.
    fn bind_vector_database(
        &mut self,
        database_bytes: &[u8],
        dimension: usize,
        num_vectors: usize,
    ) -> Result<(), Self::Error>;

    /// Submits a batch of queries to the GPU for parallel distance computation.
    /// Returns a Future that yields a vector of top-K results for each query in the batch.
    fn batch_scan(
        &self,
        queries: &[f32],
        num_queries: usize,
        top_k: usize,
        metric: GpuDistanceMetric,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<Vec<GpuScanResult>>, Self::Error>> + Send>>;
}
```

---

## 5. Performance and Verification Targets

For the Phase 2 implementation, we define the following validation goals:

1.  **Bit-Accuracy**: GPU calculated scores must match the AVX-512/NEON CPU float results down to $10^{-5}$ relative tolerance.
2.  **Batch Scale-up**: At batch size $Q = 1024$, the GPU-native scan must achieve at least **3x throughput speedup** over CPU multi-thread execution.
3.  **Low Resource Latency**: Query queue submission overhead (host-to-device buffer mapping latency) must remain below **100 microseconds** on local hardware.

---

## 6. Detailed Implementation Strategy & Timeline

- **Phase 1 (T-234-A - This RFC)**: Approve compute contracts and trait interface.
- **Phase 2 (T-234-B)**: Build mock wgpu adapters and unit tests verifying the mathematical correctness of WGSL distance formulas on synthetic blocks.
- **Phase 3 (T-234-C)**: Implement wgpu host setup, pipeline bindings, command encoders, and integration tests.
- **Phase 4 (T-234-D)**: Benchmark latency vs CPU SIMD, write the code tour, and merge.
