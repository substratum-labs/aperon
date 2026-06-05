//! Core primitives for the Aperon vector search engine.

pub mod binary;
pub mod distance;
pub mod grain;
pub mod index;
pub mod layout;
pub mod memory_sstable;
pub mod pivot_prefix;
pub mod quantization;
pub mod routing;
mod scan;
pub mod shared;

pub use distance::{l2_squared, Distance};
pub use grain::{Grain, GrainId, ScoredVector};
pub use index::{AperonIndex, IndexStats};
pub use layout::{BlockSoaLayout, VectorId, DEFAULT_BLOCK_SIZE};
pub use memory_sstable::{
    stable_memory_branch_id, ArrayLikeMemoryVectorCandidateGenerator, ArrayLikeMemoryVectorIndex,
    FlatMemoryVectorCandidateGenerator, HtlaMemoryVectorCandidateGenerator, HtlaMemoryVectorConfig,
    LoadedMemorySegment, MemoryHit, MemoryManifest, MemoryManifestFile, MemoryManifestSegment,
    MemoryRecordInput, MemorySegment, MemorySpace, MemorySpaceRecallResult, MemorySpaceRecallTrace,
    MemorySpaceSegmentTrace, MemoryVectorCandidateGenerator, MemoryVectorRouteTrace,
    PivotPrefixMemoryVectorCandidateGenerator, RecallQuery, RecallResult, RecallTrace,
};
pub use pivot_prefix::{
    coverage, exact_topk, sample_centroids, DensePivotSketch, PivotPrefixConfig, PivotPrefixRouter,
    PivotRouteScratch, PrefixScoreMode, RouteMetrics, DEFAULT_FINAL_NPROBE,
};
pub use quantization::{QuantizationError, Quantizer};
pub use routing::{
    CentroidRouter, HierarchicalLatticeLayer, HierarchicalLatticeLayerConfig,
    HierarchicalLatticeRouter, HtlaDiagnostics, HtlaRoute, HtlaRouter, LatticeRouter, Route,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_tracks_inserted_vectors() {
        let mut index = AperonIndex::new(4);

        index.insert(7, [1.0, 2.0, 3.0, 4.0]).unwrap();
        index.insert(9, [2.0, 2.0, 3.0, 8.0]).unwrap();

        let stats = index.stats();
        assert_eq!(stats.dim, 4);
        assert_eq!(stats.vectors, 2);
        assert_eq!(stats.grains, 1);
    }

    #[test]
    fn router_prefers_nearest_centroid() {
        let mut router = CentroidRouter::new(2);
        router.add_centroid(GrainId::new(10), [0.0, 0.0]).unwrap();
        router.add_centroid(GrainId::new(20), [10.0, 10.0]).unwrap();

        let route = router.route(&[1.0, 1.0]).unwrap();
        assert_eq!(route.grain_id, GrainId::new(10));
    }
}
