use crate::distance::l2_squared_unchecked;
use crate::pivot_prefix::{PivotPrefixConfig, PivotPrefixRouter, PrefixScoreMode, RouteMetrics};
use crate::routing::HtlaRouter;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

const SEGMENT_MAGIC: &[u8; 4] = b"APMS";
const SEGMENT_VERSION: u32 = 1;
const VECTOR_SIDECAR_MAGIC: &[u8; 4] = b"APMV";
const VECTOR_SIDECAR_VERSION: u32 = 0;
const MANIFEST_SCHEMA_VERSION: u32 = 0;
const CHECKSUM_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const CHECKSUM_PRIME: u64 = 0x100000001b3;

#[derive(Clone, Debug, PartialEq)]
pub struct MemoryRecordInput {
    pub record_id: u64,
    pub scope_id: u32,
    pub timestamp: i64,
    pub source_id: u16,
    pub confidence: f32,
    pub text: String,
    pub embedding: Vec<f32>,
    pub symbols: Vec<String>,
    pub vector_id: Option<String>,
    pub metadata: std::collections::BTreeMap<String, String>,
}

impl MemoryRecordInput {
    pub fn passes_filters(&self, query: &RecallQuery) -> bool {
        if query
            .scope_id
            .is_some_and(|scope_id| self.scope_id != scope_id)
        {
            return false;
        }
        if query.time_start.is_some_and(|start| self.timestamp < start) {
            return false;
        }
        if query.time_end.is_some_and(|end| self.timestamp > end) {
            return false;
        }
        if query
            .min_confidence
            .is_some_and(|min| self.confidence < min)
        {
            return false;
        }
        if !query.symbols.is_empty() {
            let record_syms_normalized: HashSet<String> =
                self.symbols.iter().map(|s| normalize_symbol(s)).collect();
            for sym in &query.symbols {
                if !record_syms_normalized.contains(&normalize_symbol(sym)) {
                    return false;
                }
            }
        }
        if let Some(query_vid) = &query.vector_id {
            if self.vector_id.as_ref() != Some(query_vid) {
                return false;
            }
        }
        for (k, v) in &query.metadata_filter {
            if self.metadata.get(k) != Some(v) {
                return false;
            }
        }
        true
    }

    pub fn score(&self, query: &RecallQuery) -> f32 {
        let semantic_distance = query
            .embedding
            .as_ref()
            .map(|query_emb| l2_squared_unchecked(query_emb, &self.embedding));
        let semantic = semantic_distance.map_or(0.0, |dist| -dist);

        let symbol_matches = if query.symbols.is_empty() {
            0
        } else {
            let record_syms_normalized: HashSet<String> =
                self.symbols.iter().map(|s| normalize_symbol(s)).collect();
            query
                .symbols
                .iter()
                .filter(|sym| record_syms_normalized.contains(&normalize_symbol(sym)))
                .count()
        };

        let record_symbol_count = self
            .symbols
            .iter()
            .map(|s| normalize_symbol(s))
            .collect::<HashSet<_>>()
            .len();
        let symbol = symbol_matches as f32 * 2.0 + record_symbol_count as f32 * 0.01;
        let confidence = self.confidence;

        semantic + symbol + confidence
    }
}

/// An immutable, columnar SSTable segment representing a collection of memory records.
/// Contains pre-tokenized symbol indices, metadata filters, and dense embeddings for vector queries.
#[derive(Clone, Debug, PartialEq)]
pub struct MemorySegment {
    pub dim: usize,
    pub segment_id: u64,
    pub record_ids: Vec<u64>,
    pub scope_ids: Vec<u32>,
    pub timestamps: Vec<i64>,
    pub source_ids: Vec<u16>,
    pub confidences: Vec<f32>,
    pub text_offsets: Vec<u32>,
    pub text_bytes: Vec<u8>,
    pub embeddings: Vec<f32>,
    pub symbol_terms: Vec<String>,
    pub symbol_offsets: Vec<u32>,
    pub symbol_record_ids: Vec<u32>,
    pub vector_id_presence: Vec<u8>,
    pub vector_id_offsets: Vec<u32>,
    pub vector_id_bytes: Vec<u8>,
    pub metadata_offsets: Vec<u32>,
    pub metadata_bytes: Vec<u8>,
    pub cold_store: Option<std::sync::Arc<ColdVectorStore>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MemoryManifest {
    pub manifest_id: u64,
    pub parent_manifest_id: Option<u64>,
    pub branch_id: u64,
    pub segment_ids: Vec<u64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MemoryManifestSegment {
    pub segment_id: u64,
    pub path: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector_sidecar: Option<MemoryVectorSidecarRef>,
}

/// Reference to an external `.apmv` vector sidecar file associated with a segment.
/// Stores expected indexing parameters, checksums, and validation fingerprints to ensure version consistency.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MemoryVectorSidecarRef {
    pub path: PathBuf,
    #[serde(default)]
    pub required: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_sidecar_version: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_generator_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_generator_version: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sidecar_checksum: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub segment_fingerprint: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryVectorSidecarFile {
    pub version: u32,
    pub segment_id: u64,
    pub segment_version: u32,
    pub record_count: u32,
    pub dim: u32,
    pub segment_fingerprint: u64,
    pub generator_name: String,
    pub generator_version: u32,
    pub checksum: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MemoryManifestFile {
    pub version: u32,
    pub manifest_id: u64,
    pub parent_manifest_id: Option<u64>,
    pub branch_id: u64,
    pub segments: Vec<MemoryManifestSegment>,
}

/// A query structure specifying conditions for filtering and searching memory records.
/// Supports metadata-matching filters, confidence thresholds, and semantic routing parameters.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RecallQuery {
    pub embedding: Option<Vec<f32>>,
    pub symbols: Vec<String>,
    pub scope_id: Option<u32>,
    pub time_start: Option<i64>,
    pub time_end: Option<i64>,
    pub min_confidence: Option<f32>,
    pub limit: usize,
    pub candidate_budget: Option<usize>,
    pub vector_id: Option<String>,
    pub metadata_filter: std::collections::BTreeMap<String, String>,
    pub fallback_to_recon_on_cold: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MemoryHit {
    pub record_id: u64,
    pub score: f32,
    pub semantic_distance: Option<f32>,
    pub symbol_matches: usize,
    pub confidence: f32,
    pub timestamp: i64,
    pub text: String,
    pub vector_id: Option<String>,
    pub metadata: std::collections::BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RecallTrace {
    pub segment_id: u64,
    pub access_paths: Vec<&'static str>,
    pub records_total: usize,
    pub candidates_after_filters: usize,
    pub candidates_after_symbols: usize,
    pub vector_generator: &'static str,
    pub vector_candidates: usize,
    pub vector_route: Option<MemoryVectorRouteTrace>,
    pub planner: Option<MemoryQueryPlannerTrace>,
    pub semantic_evals: usize,
    pub returned: usize,
    pub cold_bytes_read: usize,
    pub read_amplification: f32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryQueryPlannerTrace {
    pub selected_path: &'static str,
    pub candidate_budget: usize,
    pub expanded_candidate_budget: Option<usize>,
    pub fallback_reason: Option<&'static str>,
    pub candidates_after_symbols: usize,
    pub final_candidates: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MemoryVectorRouteTrace {
    pub vector_index_bytes: usize,
    pub route_candidates: usize,
    pub posting_entries_touched: usize,
    pub duplicate_block_rate: f32,
    pub selected_blocks: usize,
    pub centroid_evals: usize,
    pub working_set_bytes: usize,
    pub fallback_used: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RecallResult {
    pub hits: Vec<MemoryHit>,
    pub trace: RecallTrace,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MemorySpaceSegmentTrace {
    pub segment_id: u64,
    pub pruned: bool,
    pub prune_reason: Option<&'static str>,
    pub trace: Option<RecallTrace>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MemorySpaceRecallTrace {
    pub manifest_id: u64,
    pub branch_id: u64,
    pub segments_considered: usize,
    pub segments_scanned: usize,
    pub segments_pruned: usize,
    pub semantic_evals: usize,
    pub returned: usize,
    pub segment_traces: Vec<MemorySpaceSegmentTrace>,
    pub cold_bytes_read: usize,
    pub read_amplification: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MemorySpaceRecallResult {
    pub hits: Vec<MemoryHit>,
    pub trace: MemorySpaceRecallTrace,
}

#[derive(Debug)]
pub struct MemorySpace {
    pub manifest: MemoryManifestFile,
    pub segments: Vec<LoadedMemorySegment>,
    pub memtable: std::sync::RwLock<MemTable>,
    pub wal_writer: std::sync::Mutex<WALWriter>,
    pub wal_path: PathBuf,
    pub manifest_path: PathBuf,
    pub max_memtable_records: usize,
    pub vector_encoding: Option<VectorEncoding>,
    pub raw_dram_budget: Option<usize>,
}

impl Clone for MemorySpace {
    fn clone(&self) -> Self {
        let memtable_guard = self.memtable.read().unwrap();
        let cloned_memtable = memtable_guard.clone();
        drop(memtable_guard);

        let wal_writer =
            WALWriter::open(&self.wal_path).expect("failed to open WAL for cloned MemorySpace");

        Self {
            manifest: self.manifest.clone(),
            segments: self.segments.clone(),
            memtable: std::sync::RwLock::new(cloned_memtable),
            wal_writer: std::sync::Mutex::new(wal_writer),
            wal_path: self.wal_path.clone(),
            manifest_path: self.manifest_path.clone(),
            max_memtable_records: self.max_memtable_records,
            vector_encoding: self.vector_encoding,
            raw_dram_budget: self.raw_dram_budget,
        }
    }
}

impl PartialEq for MemorySpace {
    fn eq(&self, other: &Self) -> bool {
        let self_memtable = self.memtable.read().unwrap();
        let other_memtable = other.memtable.read().unwrap();
        self.manifest == other.manifest
            && self.segments == other.segments
            && *self_memtable == *other_memtable
            && self.wal_path == other.wal_path
            && self.manifest_path == other.manifest_path
            && self.max_memtable_records == other.max_memtable_records
            && self.vector_encoding == other.vector_encoding
            && self.raw_dram_budget == other.raw_dram_budget
    }
}

pub trait MemoryVectorCandidateGenerator {
    fn name(&self) -> &'static str;

    fn candidates(
        &self,
        segment: &MemorySegment,
        query: &RecallQuery,
        candidates_after_symbols: &[u32],
    ) -> Result<Vec<u32>, String>;

    fn route_trace(&self) -> Option<MemoryVectorRouteTrace> {
        None
    }

    fn planner_trace(&self) -> Option<MemoryQueryPlannerTrace> {
        None
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FlatMemoryVectorCandidateGenerator;

impl MemoryVectorCandidateGenerator for FlatMemoryVectorCandidateGenerator {
    fn name(&self) -> &'static str {
        "flat"
    }

    fn candidates(
        &self,
        _segment: &MemorySegment,
        query: &RecallQuery,
        candidates_after_symbols: &[u32],
    ) -> Result<Vec<u32>, String> {
        let mut candidates = candidates_after_symbols.to_vec();
        if let Some(budget) = query.candidate_budget {
            candidates.truncate(budget);
        }
        Ok(candidates)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ArrayLikeMemoryVectorIndex {
    dim: usize,
    segment_id: u64,
    local_ids: Vec<u32>,
    embeddings: Vec<f32>,
}

impl ArrayLikeMemoryVectorIndex {
    pub fn build(segment: &MemorySegment) -> Self {
        Self {
            dim: segment.dim,
            segment_id: segment.segment_id,
            local_ids: (0..segment.len()).map(|local_id| local_id as u32).collect(),
            embeddings: segment.embeddings.clone(),
        }
    }

    pub fn len(&self) -> usize {
        self.local_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.local_ids.is_empty()
    }

    pub fn vector_index_bytes(&self) -> usize {
        self.local_ids.len() * std::mem::size_of::<u32>()
            + self.embeddings.len() * std::mem::size_of::<f32>()
    }

    pub fn segment_id(&self) -> u64 {
        self.segment_id
    }

    fn embedding_row(&self, local_id: u32) -> &[f32] {
        let local_id = local_id as usize;
        &self.embeddings[local_id * self.dim..(local_id + 1) * self.dim]
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ArrayLikeMemoryVectorCandidateGenerator {
    index: ArrayLikeMemoryVectorIndex,
}

impl ArrayLikeMemoryVectorCandidateGenerator {
    pub fn build(segment: &MemorySegment) -> Self {
        Self {
            index: ArrayLikeMemoryVectorIndex::build(segment),
        }
    }

    pub fn index(&self) -> &ArrayLikeMemoryVectorIndex {
        &self.index
    }
}

impl MemoryVectorCandidateGenerator for ArrayLikeMemoryVectorCandidateGenerator {
    fn name(&self) -> &'static str {
        "array_like"
    }

    fn candidates(
        &self,
        segment: &MemorySegment,
        query: &RecallQuery,
        candidates_after_symbols: &[u32],
    ) -> Result<Vec<u32>, String> {
        if self.index.segment_id != segment.segment_id {
            return Err(format!(
                "array-like vector index segment id mismatch: expected {}, got {}",
                segment.segment_id, self.index.segment_id
            ));
        }
        if self.index.dim != segment.dim || self.index.len() != segment.len() {
            return Err("array-like vector index layout mismatch".to_string());
        }

        let Some(query_embedding) = query.embedding.as_deref() else {
            return FlatMemoryVectorCandidateGenerator.candidates(
                segment,
                query,
                candidates_after_symbols,
            );
        };
        let Some(budget) = query.candidate_budget else {
            return Ok(candidates_after_symbols.to_vec());
        };
        if budget >= candidates_after_symbols.len() {
            return Ok(candidates_after_symbols.to_vec());
        }

        let mut scored = candidates_after_symbols
            .iter()
            .map(|&local_id| {
                let distance =
                    l2_squared_unchecked(query_embedding, self.index.embedding_row(local_id));
                (local_id, distance)
            })
            .collect::<Vec<_>>();
        scored.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        scored.truncate(budget);
        scored.sort_unstable_by_key(|&(local_id, _)| local_id);
        Ok(scored
            .into_iter()
            .map(|(local_id, _)| local_id)
            .collect::<Vec<_>>())
    }

    fn route_trace(&self) -> Option<MemoryVectorRouteTrace> {
        Some(MemoryVectorRouteTrace {
            vector_index_bytes: self.index.vector_index_bytes(),
            route_candidates: self.index.len(),
            posting_entries_touched: 0,
            duplicate_block_rate: 0.0,
            selected_blocks: 0,
            centroid_evals: self.index.len(),
            working_set_bytes: self.index.vector_index_bytes(),
            fallback_used: false,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PivotPrefixMemoryVectorCandidateGenerator {
    segment_id: u64,
    router: PivotPrefixRouter,
    last_trace: RefCell<Option<MemoryVectorRouteTrace>>,
}

impl PivotPrefixMemoryVectorCandidateGenerator {
    pub fn build(segment: &MemorySegment, config: PivotPrefixConfig) -> Result<Self, String> {
        let router = PivotPrefixRouter::build(&segment.embeddings, segment.dim, config)?;
        Ok(Self {
            segment_id: segment.segment_id,
            router,
            last_trace: RefCell::new(None),
        })
    }

    pub fn build_default(segment: &MemorySegment) -> Result<Self, String> {
        Self::build(segment, default_memory_pivot_prefix_config(segment))
    }

    pub fn resident_bytes(&self) -> usize {
        self.router.resident_bytes()
    }
}

impl MemoryVectorCandidateGenerator for PivotPrefixMemoryVectorCandidateGenerator {
    fn name(&self) -> &'static str {
        "pivot_prefix"
    }

    fn candidates(
        &self,
        segment: &MemorySegment,
        query: &RecallQuery,
        candidates_after_symbols: &[u32],
    ) -> Result<Vec<u32>, String> {
        self.last_trace.replace(None);
        if self.segment_id != segment.segment_id {
            return Err(format!(
                "pivot-prefix vector index segment id mismatch: expected {}, got {}",
                segment.segment_id, self.segment_id
            ));
        }

        let Some(query_embedding) = query.embedding.as_deref() else {
            return FlatMemoryVectorCandidateGenerator.candidates(
                segment,
                query,
                candidates_after_symbols,
            );
        };

        let mut scratch = self.router.scratch();
        let metrics = self.router.route(query_embedding, &mut scratch);
        self.last_trace.replace(Some(memory_vector_route_trace(
            self.resident_bytes(),
            metrics,
        )));

        let mut seen = BTreeSet::new();
        let mut routed = scratch
            .pool_candidates
            .iter()
            .copied()
            .filter(|local_id| {
                candidates_after_symbols.binary_search(local_id).is_ok() && seen.insert(*local_id)
            })
            .collect::<Vec<_>>();
        if let Some(budget) = query.candidate_budget {
            routed.truncate(budget);
        }
        routed.sort_unstable();

        if routed.is_empty() && !candidates_after_symbols.is_empty() {
            if let Some(trace) = self.last_trace.borrow_mut().as_mut() {
                trace.fallback_used = true;
            }
            let mut fallback = candidates_after_symbols.to_vec();
            if let Some(budget) = query.candidate_budget {
                fallback.truncate(budget);
            }
            return Ok(fallback);
        }
        Ok(routed)
    }

    fn route_trace(&self) -> Option<MemoryVectorRouteTrace> {
        self.last_trace.borrow().clone()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HtlaMemoryVectorConfig {
    pub levels: usize,
    pub chart_dim: usize,
    pub beam: usize,
    pub candidate_pool: usize,
    pub final_nprobe: usize,
    pub fallback_on_route_risk: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HtlaMemoryVectorCandidateGenerator {
    segment_id: u64,
    router: HtlaRouter,
    config: HtlaMemoryVectorConfig,
    fallback: ArrayLikeMemoryVectorCandidateGenerator,
    last_trace: RefCell<Option<MemoryVectorRouteTrace>>,
}

impl HtlaMemoryVectorCandidateGenerator {
    pub fn build(segment: &MemorySegment, config: HtlaMemoryVectorConfig) -> Result<Self, String> {
        let centroids = segment
            .embeddings
            .chunks_exact(segment.dim)
            .map(|row| row.to_vec())
            .collect::<Vec<_>>();
        let router = HtlaRouter::new(segment.dim, &centroids, config.levels, config.chart_dim)?;
        Ok(Self {
            segment_id: segment.segment_id,
            router,
            config,
            fallback: ArrayLikeMemoryVectorCandidateGenerator::build(segment),
            last_trace: RefCell::new(None),
        })
    }

    pub fn build_default(segment: &MemorySegment) -> Result<Self, String> {
        Self::build(segment, default_memory_htla_config(segment))
    }

    pub fn resident_bytes(&self) -> usize {
        self.router.resident_bytes()
    }
}

impl MemoryVectorCandidateGenerator for HtlaMemoryVectorCandidateGenerator {
    fn name(&self) -> &'static str {
        "htla_tangent"
    }

    fn candidates(
        &self,
        segment: &MemorySegment,
        query: &RecallQuery,
        candidates_after_symbols: &[u32],
    ) -> Result<Vec<u32>, String> {
        self.last_trace.replace(None);
        if self.segment_id != segment.segment_id {
            return Err(format!(
                "htla vector index segment id mismatch: expected {}, got {}",
                segment.segment_id, self.segment_id
            ));
        }

        let Some(query_embedding) = query.embedding.as_deref() else {
            return FlatMemoryVectorCandidateGenerator.candidates(
                segment,
                query,
                candidates_after_symbols,
            );
        };

        let budget = query
            .candidate_budget
            .unwrap_or(self.config.candidate_pool)
            .min(self.config.candidate_pool)
            .max(1);
        let route = self.router.route(
            query_embedding,
            self.config.beam,
            budget,
            self.config.final_nprobe.min(segment.len()).max(1),
        );
        self.last_trace.replace(Some(MemoryVectorRouteTrace {
            vector_index_bytes: self.resident_bytes(),
            route_candidates: route.candidates.len(),
            posting_entries_touched: 0,
            duplicate_block_rate: 0.0,
            selected_blocks: route.final_nprobe.len(),
            centroid_evals: route.candidates.len(),
            working_set_bytes: route.working_set_bytes,
            fallback_used: route.fallback,
        }));

        let mut seen = BTreeSet::new();
        let mut routed = route
            .candidates
            .iter()
            .map(|&local_id| local_id as u32)
            .filter(|local_id| {
                candidates_after_symbols.binary_search(local_id).is_ok() && seen.insert(*local_id)
            })
            .collect::<Vec<_>>();
        routed.truncate(budget);
        routed.sort_unstable();

        if (self.config.fallback_on_route_risk && route.fallback)
            || (routed.is_empty() && !candidates_after_symbols.is_empty())
        {
            if let Some(trace) = self.last_trace.borrow_mut().as_mut() {
                trace.fallback_used = true;
            }
            return self
                .fallback
                .candidates(segment, query, candidates_after_symbols);
        }
        Ok(routed)
    }

    fn route_trace(&self) -> Option<MemoryVectorRouteTrace> {
        self.last_trace.borrow().clone()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryQueryPlannerConfig {
    pub direct_candidate_threshold: usize,
    pub vector_candidate_budget: usize,
    pub fallback_budget_multiplier: usize,
    pub pivot_min_candidates: usize,
    pub htla_enabled: bool,
    pub htla_min_candidates: usize,
}

impl Default for MemoryQueryPlannerConfig {
    fn default() -> Self {
        Self {
            direct_candidate_threshold: 16,
            vector_candidate_budget: 32,
            fallback_budget_multiplier: 2,
            pivot_min_candidates: 32,
            htla_enabled: false,
            htla_min_candidates: 128,
        }
    }
}

/// A 5-layer query planner that routes queries through direct scans, flat scans,
/// array-like indexes, pivot-prefix indexes, or HTLA lanes based on available metadata and budget.
#[derive(Clone, Debug, PartialEq)]
pub struct MemoryQueryPlanner {
    config: MemoryQueryPlannerConfig,
    array_like: ArrayLikeMemoryVectorCandidateGenerator,
    pivot_prefix: Option<PivotPrefixMemoryVectorCandidateGenerator>,
    htla: Option<HtlaMemoryVectorCandidateGenerator>,
    last_route: RefCell<Option<MemoryVectorRouteTrace>>,
    last_trace: RefCell<Option<MemoryQueryPlannerTrace>>,
}

impl MemoryQueryPlanner {
    pub fn build(
        segment: &MemorySegment,
        config: MemoryQueryPlannerConfig,
    ) -> Result<Self, String> {
        let pivot_prefix = if segment.is_empty() {
            None
        } else {
            Some(PivotPrefixMemoryVectorCandidateGenerator::build_default(
                segment,
            )?)
        };
        let htla = if config.htla_enabled && !segment.is_empty() {
            Some(HtlaMemoryVectorCandidateGenerator::build_default(segment)?)
        } else {
            None
        };
        Ok(Self {
            config,
            array_like: ArrayLikeMemoryVectorCandidateGenerator::build(segment),
            pivot_prefix,
            htla,
            last_route: RefCell::new(None),
            last_trace: RefCell::new(None),
        })
    }

    pub fn build_default(segment: &MemorySegment) -> Result<Self, String> {
        Self::build(segment, MemoryQueryPlannerConfig::default())
    }

    pub fn resident_bytes(&self) -> usize {
        self.array_like.index().vector_index_bytes()
            + self
                .pivot_prefix
                .as_ref()
                .map(PivotPrefixMemoryVectorCandidateGenerator::resident_bytes)
                .unwrap_or(0)
            + self
                .htla
                .as_ref()
                .map(HtlaMemoryVectorCandidateGenerator::resident_bytes)
                .unwrap_or(0)
    }

    fn planned_budget(&self, query: &RecallQuery, candidates_after_symbols: usize) -> usize {
        let budget = match query.candidate_budget {
            Some(budget) => budget.max(1),
            None => self.config.vector_candidate_budget.max(query.limit.max(1)),
        };
        budget.min(candidates_after_symbols.max(1))
    }

    fn expanded_budget(&self, budget: usize, candidates_after_symbols: usize) -> usize {
        budget
            .saturating_mul(self.config.fallback_budget_multiplier.max(1))
            .max(budget + usize::from(budget < candidates_after_symbols))
            .min(candidates_after_symbols)
    }

    fn query_with_budget(query: &RecallQuery, budget: usize) -> RecallQuery {
        let mut planned_query = query.clone();
        planned_query.candidate_budget = Some(budget);
        planned_query
    }

    fn record_trace(
        &self,
        selected_path: &'static str,
        candidate_budget: usize,
        expanded_candidate_budget: Option<usize>,
        fallback_reason: Option<&'static str>,
        candidates_after_symbols: usize,
        final_candidates: usize,
    ) {
        self.last_trace.replace(Some(MemoryQueryPlannerTrace {
            selected_path,
            candidate_budget,
            expanded_candidate_budget,
            fallback_reason,
            candidates_after_symbols,
            final_candidates,
        }));
    }
}

impl MemoryVectorCandidateGenerator for MemoryQueryPlanner {
    fn name(&self) -> &'static str {
        "query_planner"
    }

    fn candidates(
        &self,
        segment: &MemorySegment,
        query: &RecallQuery,
        candidates_after_symbols: &[u32],
    ) -> Result<Vec<u32>, String> {
        self.last_route.replace(None);
        self.last_trace.replace(None);

        let candidates_len = candidates_after_symbols.len();
        let budget = self.planned_budget(query, candidates_len);
        if segment.embeddings.is_empty() {
            self.record_trace(
                "direct_rerank",
                budget,
                None,
                Some("cold_vector_store"),
                candidates_len,
                candidates_len,
            );
            return Ok(candidates_after_symbols.to_vec());
        }
        if candidates_len == 0 {
            self.record_trace("direct_rerank", budget, None, None, 0, 0);
            return Ok(Vec::new());
        }

        if query.embedding.is_none() {
            let mut candidates = candidates_after_symbols.to_vec();
            candidates.truncate(budget);
            self.record_trace(
                "direct_rerank",
                budget,
                None,
                Some("missing_embedding"),
                candidates_len,
                candidates.len(),
            );
            return Ok(candidates);
        }

        if candidates_len <= self.config.direct_candidate_threshold {
            self.record_trace(
                "direct_rerank",
                budget,
                None,
                None,
                candidates_len,
                candidates_len,
            );
            return Ok(candidates_after_symbols.to_vec());
        }

        let planned_query = Self::query_with_budget(query, budget);
        let (selected_path, generator): (&'static str, &dyn MemoryVectorCandidateGenerator) =
            match (&self.htla, &self.pivot_prefix) {
                (Some(ref htla), _)
                    if self.config.htla_enabled
                        && candidates_len >= self.config.htla_min_candidates =>
                {
                    ("htla_tangent", htla)
                }
                (_, Some(ref pivot)) if candidates_len >= self.config.pivot_min_candidates => {
                    ("pivot_prefix", pivot)
                }
                _ => ("array_like", &self.array_like),
            };

        let mut candidates =
            generator.candidates(segment, &planned_query, candidates_after_symbols)?;
        let route = generator.route_trace();
        let fallback_reason = if route.as_ref().is_some_and(|trace| trace.fallback_used) {
            Some("route_fallback")
        } else if candidates.is_empty() {
            Some("empty_vector_candidates")
        } else {
            None
        };

        if let Some(reason) = fallback_reason {
            let expanded_budget = self.expanded_budget(budget, candidates_len);
            let expanded_query = Self::query_with_budget(query, expanded_budget);
            candidates =
                self.array_like
                    .candidates(segment, &expanded_query, candidates_after_symbols)?;
            self.last_route.replace(route);
            self.record_trace(
                selected_path,
                budget,
                Some(expanded_budget),
                Some(reason),
                candidates_len,
                candidates.len(),
            );
            return Ok(candidates);
        }

        self.last_route.replace(route);
        self.record_trace(
            selected_path,
            budget,
            None,
            None,
            candidates_len,
            candidates.len(),
        );
        Ok(candidates)
    }

    fn route_trace(&self) -> Option<MemoryVectorRouteTrace> {
        self.last_route.borrow().clone()
    }

    fn planner_trace(&self) -> Option<MemoryQueryPlannerTrace> {
        self.last_trace.borrow().clone()
    }
}

impl MemorySegment {
    pub fn build(
        segment_id: u64,
        dim: usize,
        records: Vec<MemoryRecordInput>,
    ) -> Result<Self, String> {
        if dim == 0 {
            return Err("dim must be greater than zero".to_string());
        }
        let mut record_ids = Vec::with_capacity(records.len());
        let mut scope_ids = Vec::with_capacity(records.len());
        let mut timestamps = Vec::with_capacity(records.len());
        let mut source_ids = Vec::with_capacity(records.len());
        let mut confidences = Vec::with_capacity(records.len());
        let mut text_offsets = Vec::with_capacity(records.len() + 1);
        let mut text_bytes = Vec::new();
        let mut embeddings = Vec::with_capacity(records.len() * dim);
        let mut postings = BTreeMap::<String, BTreeSet<u32>>::new();

        let mut vector_id_presence = Vec::with_capacity(records.len());
        let mut vector_id_offsets = Vec::with_capacity(records.len() + 1);
        let mut vector_id_bytes = Vec::new();
        let mut metadata_offsets = Vec::with_capacity(records.len() + 1);
        let mut metadata_bytes = Vec::new();

        text_offsets.push(0);
        vector_id_offsets.push(0);
        metadata_offsets.push(0);

        for (local_id, record) in records.into_iter().enumerate() {
            if record.embedding.len() != dim {
                return Err(format!(
                    "record {} embedding dimension mismatch: expected {}, got {}",
                    record.record_id,
                    dim,
                    record.embedding.len()
                ));
            }
            record_ids.push(record.record_id);
            scope_ids.push(record.scope_id);
            timestamps.push(record.timestamp);
            source_ids.push(record.source_id);
            confidences.push(record.confidence);
            text_bytes.extend_from_slice(record.text.as_bytes());
            text_offsets.push(text_bytes.len() as u32);
            embeddings.extend_from_slice(&record.embedding);
            for symbol in record.symbols {
                postings
                    .entry(normalize_symbol(&symbol))
                    .or_default()
                    .insert(local_id as u32);
            }

            if let Some(ref vid) = record.vector_id {
                vector_id_presence.push(1);
                vector_id_bytes.extend_from_slice(vid.as_bytes());
            } else {
                vector_id_presence.push(0);
            }
            vector_id_offsets.push(vector_id_bytes.len() as u32);

            let meta_str = serde_json::to_string(&record.metadata).unwrap();
            metadata_bytes.extend_from_slice(meta_str.as_bytes());
            metadata_offsets.push(metadata_bytes.len() as u32);
        }

        let mut symbol_terms = Vec::with_capacity(postings.len());
        let mut symbol_offsets = Vec::with_capacity(postings.len() + 1);
        let mut symbol_record_ids = Vec::new();
        symbol_offsets.push(0);
        for (term, ids) in postings {
            symbol_terms.push(term);
            symbol_record_ids.extend(ids);
            symbol_offsets.push(symbol_record_ids.len() as u32);
        }

        Ok(Self {
            dim,
            segment_id,
            record_ids,
            scope_ids,
            timestamps,
            source_ids,
            confidences,
            text_offsets,
            text_bytes,
            embeddings,
            symbol_terms,
            symbol_offsets,
            symbol_record_ids,
            vector_id_presence,
            vector_id_offsets,
            vector_id_bytes,
            metadata_offsets,
            metadata_bytes,
            cold_store: None,
        })
    }

    pub fn len(&self) -> usize {
        self.record_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.record_ids.is_empty()
    }

    pub fn write(&self, path: impl AsRef<Path>) -> io::Result<()> {
        self.validate_layout().map_err(invalid_data)?;

        let record_count = checked_u32(self.len(), "record count")?;
        let dim = checked_u32(self.dim, "dimension")?;
        let text_bytes_len = checked_u64(self.text_bytes.len(), "text bytes length")?;
        let embedding_count = checked_u64(self.embeddings.len(), "embedding count")?;
        let symbol_count = checked_u32(self.symbol_terms.len(), "symbol count")?;
        let symbol_record_count =
            checked_u32(self.symbol_record_ids.len(), "symbol record id count")?;

        let mut bytes = Vec::new();
        bytes.extend_from_slice(SEGMENT_MAGIC);
        write_u32(&mut bytes, SEGMENT_VERSION);
        write_u64(&mut bytes, self.segment_id);
        write_u32(&mut bytes, dim);
        write_u32(&mut bytes, record_count);
        write_u64(&mut bytes, text_bytes_len);
        write_u64(&mut bytes, embedding_count);
        write_u32(&mut bytes, symbol_count);
        write_u32(&mut bytes, symbol_record_count);

        // Version 1 extra header fields
        write_u64(&mut bytes, self.vector_id_bytes.len() as u64);
        write_u64(&mut bytes, self.metadata_bytes.len() as u64);

        for value in &self.record_ids {
            write_u64(&mut bytes, *value);
        }
        for value in &self.scope_ids {
            write_u32(&mut bytes, *value);
        }
        for value in &self.timestamps {
            write_i64(&mut bytes, *value);
        }
        for value in &self.source_ids {
            write_u16(&mut bytes, *value);
        }
        for value in &self.confidences {
            write_f32(&mut bytes, *value);
        }
        for value in &self.text_offsets {
            write_u32(&mut bytes, *value);
        }
        bytes.extend_from_slice(&self.text_bytes);
        for value in &self.embeddings {
            write_f32(&mut bytes, *value);
        }
        for term in &self.symbol_terms {
            let term_bytes = term.as_bytes();
            write_u32(
                &mut bytes,
                checked_u32(term_bytes.len(), "symbol term length")?,
            );
            bytes.extend_from_slice(term_bytes);
        }
        for value in &self.symbol_offsets {
            write_u32(&mut bytes, *value);
        }
        for value in &self.symbol_record_ids {
            write_u32(&mut bytes, *value);
        }

        // Version 1 columns
        bytes.extend_from_slice(&self.vector_id_presence);
        for &offset in &self.vector_id_offsets {
            write_u32(&mut bytes, offset);
        }
        bytes.extend_from_slice(&self.vector_id_bytes);
        for &offset in &self.metadata_offsets {
            write_u32(&mut bytes, offset);
        }
        bytes.extend_from_slice(&self.metadata_bytes);

        let checksum = checksum64(&bytes);
        write_u64(&mut bytes, checksum);
        fs::write(path, bytes)
    }

    pub fn read(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path)?;
        if bytes.len() < 56 {
            return Err(invalid_data("segment file is too short"));
        }
        let (payload, footer) = bytes.split_at(bytes.len() - 8);
        let expected_checksum = read_footer_checksum(footer)?;
        let actual_checksum = checksum64(payload);
        if expected_checksum != actual_checksum {
            return Err(invalid_data("segment checksum mismatch"));
        }

        let mut reader = SegmentReader::new(payload);
        reader.expect_magic()?;
        let version = reader.read_u32()?;
        if version != 0 && version != 1 {
            return Err(invalid_data(format!(
                "unsupported memory segment version: {}",
                version
            )));
        }
        let segment_id = reader.read_u64()?;
        let dim = reader.read_u32()? as usize;
        let record_count = reader.read_u32()? as usize;
        let text_bytes_len = usize::try_from(reader.read_u64()?)
            .map_err(|_| invalid_data("text_bytes_len does not fit in usize"))?;
        let embedding_count = usize::try_from(reader.read_u64()?)
            .map_err(|_| invalid_data("embedding_count does not fit in usize"))?;
        let symbol_count = reader.read_u32()? as usize;
        let symbol_record_count = reader.read_u32()? as usize;

        let (vector_id_bytes_len, metadata_bytes_len) = if version == 1 {
            let v_len = usize::try_from(reader.read_u64()?)
                .map_err(|_| invalid_data("vector_id_bytes_len does not fit in usize"))?;
            let m_len = usize::try_from(reader.read_u64()?)
                .map_err(|_| invalid_data("metadata_bytes_len does not fit in usize"))?;
            (v_len, m_len)
        } else {
            (0, 0)
        };

        let record_ids = reader.read_u64_vec(record_count)?;
        let scope_ids = reader.read_u32_vec(record_count)?;
        let timestamps = reader.read_i64_vec(record_count)?;
        let source_ids = reader.read_u16_vec(record_count)?;
        let confidences = reader.read_f32_vec(record_count)?;
        let text_offsets = reader.read_u32_vec(record_count + 1)?;
        let text_bytes = reader.read_bytes(text_bytes_len)?.to_vec();
        let embeddings = reader.read_f32_vec(embedding_count)?;

        let mut symbol_terms = Vec::with_capacity(symbol_count);
        for _ in 0..symbol_count {
            let len = reader.read_u32()? as usize;
            let term = std::str::from_utf8(reader.read_bytes(len)?)
                .map_err(|_| invalid_data("symbol term is not valid utf-8"))?
                .to_string();
            symbol_terms.push(term);
        }
        let symbol_offsets = reader.read_u32_vec(symbol_count + 1)?;
        let symbol_record_ids = reader.read_u32_vec(symbol_record_count)?;

        let (
            vector_id_presence,
            vector_id_offsets,
            vector_id_bytes,
            metadata_offsets,
            metadata_bytes,
        ) = if version == 1 {
            let presence = reader.read_bytes(record_count)?.to_vec();
            let offsets = reader.read_u32_vec(record_count + 1)?;
            let v_bytes = reader.read_bytes(vector_id_bytes_len)?.to_vec();
            let m_offsets = reader.read_u32_vec(record_count + 1)?;
            let m_bytes = reader.read_bytes(metadata_bytes_len)?.to_vec();
            (presence, offsets, v_bytes, m_offsets, m_bytes)
        } else {
            let presence = vec![0; record_count];
            let offsets = vec![0; record_count + 1];
            let v_bytes = Vec::new();

            let empty_meta_bytes = b"{}";
            let mut m_bytes = Vec::new();
            let mut m_offsets = Vec::with_capacity(record_count + 1);
            m_offsets.push(0);
            for _ in 0..record_count {
                m_bytes.extend_from_slice(empty_meta_bytes);
                m_offsets.push(m_bytes.len() as u32);
            }
            (presence, offsets, v_bytes, m_offsets, m_bytes)
        };

        reader.expect_end()?;

        let apmc_path = path.with_extension("apmc");
        let cold_store = if apmc_path.exists() {
            Some(std::sync::Arc::new(ColdVectorStore::open(&apmc_path)?))
        } else {
            None
        };

        let segment = Self {
            dim,
            segment_id,
            record_ids,
            scope_ids,
            timestamps,
            source_ids,
            confidences,
            text_offsets,
            text_bytes,
            embeddings,
            symbol_terms,
            symbol_offsets,
            symbol_record_ids,
            vector_id_presence,
            vector_id_offsets,
            vector_id_bytes,
            metadata_offsets,
            metadata_bytes,
            cold_store,
        };
        segment.validate_layout().map_err(invalid_data)?;
        Ok(segment)
    }

    pub fn recall(&self, query: &RecallQuery) -> Result<RecallResult, String> {
        self.recall_with_vector_candidate_generator(query, &FlatMemoryVectorCandidateGenerator)
    }

    pub fn recall_with_vector_candidate_generator(
        &self,
        query: &RecallQuery,
        vector_candidate_generator: &(impl MemoryVectorCandidateGenerator + ?Sized),
    ) -> Result<RecallResult, String> {
        if let Some(embedding) = &query.embedding {
            if embedding.len() != self.dim {
                return Err(format!(
                    "query embedding dimension mismatch: expected {}, got {}",
                    self.dim,
                    embedding.len()
                ));
            }
        }

        let limit = query.limit.max(1);
        let mut access_paths = Vec::new();
        let mut candidates = Vec::with_capacity(self.len());
        for local_id in 0..self.len() {
            if self.passes_filters(local_id, query) {
                candidates.push(local_id as u32);
            }
        }
        if query.scope_id.is_some()
            || query.time_start.is_some()
            || query.time_end.is_some()
            || query.min_confidence.is_some()
            || query.vector_id.is_some()
            || !query.metadata_filter.is_empty()
        {
            access_paths.push("column_filters");
        }
        let candidates_after_filters = candidates.len();

        if !query.symbols.is_empty() {
            access_paths.push("symbol_postings");
            let postings = query
                .symbols
                .iter()
                .map(|symbol| self.symbol_postings(symbol))
                .collect::<Option<Vec<_>>>();
            if let Some(postings) = postings {
                candidates.retain(|id| postings.iter().all(|ids| ids.binary_search(id).is_ok()));
            } else {
                candidates.clear();
            }
        }
        let candidates_after_symbols = candidates.len();

        let candidates_after_symbols_slice = candidates.as_slice();
        let candidates =
            vector_candidate_generator.candidates(self, query, candidates_after_symbols_slice)?;
        self.validate_vector_candidates(&candidates, candidates_after_symbols_slice)?;
        let vector_candidates = candidates.len();
        let vector_route = vector_candidate_generator.route_trace();
        let planner = vector_candidate_generator.planner_trace();

        let mut cold_bytes_read = 0;
        let mut loaded_vectors = Vec::new();
        let fallback_to_recon = query.embedding.is_some()
            && self.embeddings.is_empty()
            && query.fallback_to_recon_on_cold;

        if query.embedding.is_some() && self.embeddings.is_empty() {
            if fallback_to_recon {
                // Bypasses reading from cold_store.
            } else {
                if let Some(cold_store) = &self.cold_store {
                    let (vecs, bytes_read) = cold_store
                        .read_vectors(&candidates)
                        .map_err(|e| e.to_string())?;
                    loaded_vectors = vecs;
                    cold_bytes_read = bytes_read;
                } else {
                    return Err(
                        "embeddings are empty but no cold vector store is available".to_string()
                    );
                }
            }
        }

        let mut scored = Vec::with_capacity(candidates.len());
        if query.embedding.is_some() {
            access_paths.push("semantic_rerank");
        }
        if fallback_to_recon {
            access_paths.push("fallback_to_recon");
        }

        let query_embedding = query.embedding.as_deref();
        for (i, &local_id) in candidates.iter().enumerate() {
            let local_id = local_id as usize;
            let semantic_distance = query_embedding.map(|embedding| {
                if fallback_to_recon {
                    0.0f32
                } else {
                    let emb_ref = if self.embeddings.is_empty() {
                        &loaded_vectors[i]
                    } else {
                        self.embedding_row(local_id)
                    };
                    l2_squared_unchecked(embedding, emb_ref).sqrt()
                }
            });
            let symbol_matches = self.symbol_match_count(local_id, &query.symbols);
            let score = self.score(local_id, semantic_distance, symbol_matches);
            scored.push((local_id, score, semantic_distance, symbol_matches));
        }
        scored.sort_unstable_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| self.record_ids[a.0].cmp(&self.record_ids[b.0]))
        });

        let hits = scored
            .iter()
            .take(limit)
            .map(
                |&(local_id, score, semantic_distance, symbol_matches)| MemoryHit {
                    record_id: self.record_ids[local_id],
                    score,
                    semantic_distance,
                    symbol_matches,
                    confidence: self.confidences[local_id],
                    timestamp: self.timestamps[local_id],
                    text: self.text(local_id).to_string(),
                    vector_id: if self.vector_id_presence[local_id] == 1 {
                        let start = self.vector_id_offsets[local_id] as usize;
                        let end = self.vector_id_offsets[local_id + 1] as usize;
                        std::str::from_utf8(&self.vector_id_bytes[start..end])
                            .map(String::from)
                            .ok()
                    } else {
                        None
                    },
                    metadata: {
                        let start = self.metadata_offsets[local_id] as usize;
                        let end = self.metadata_offsets[local_id + 1] as usize;
                        let meta_str =
                            std::str::from_utf8(&self.metadata_bytes[start..end]).unwrap_or("{}");
                        serde_json::from_str(meta_str).unwrap_or_default()
                    },
                },
            )
            .collect::<Vec<_>>();

        let bytes_per_vector = self.dim * 4;
        let returned_count = hits.len();
        let bytes_needed = returned_count * bytes_per_vector;
        let read_amplification = if fallback_to_recon {
            0.0
        } else if bytes_needed > 0 {
            cold_bytes_read as f32 / bytes_needed as f32
        } else {
            1.0
        };

        Ok(RecallResult {
            trace: RecallTrace {
                segment_id: self.segment_id,
                access_paths,
                records_total: self.len(),
                candidates_after_filters,
                candidates_after_symbols,
                vector_generator: vector_candidate_generator.name(),
                vector_candidates,
                vector_route,
                planner,
                semantic_evals: scored.len(),
                returned: hits.len(),
                cold_bytes_read,
                read_amplification,
            },
            hits,
        })
    }

    pub fn text(&self, local_id: usize) -> &str {
        let start = self.text_offsets[local_id] as usize;
        let end = self.text_offsets[local_id + 1] as usize;
        std::str::from_utf8(&self.text_bytes[start..end]).unwrap_or("")
    }

    pub fn get_record_input(&self, local_id: usize) -> MemoryRecordInput {
        let mut symbols = Vec::new();
        let target_local_id = local_id as u32;
        for (pos, term) in self.symbol_terms.iter().enumerate() {
            let start = self.symbol_offsets[pos] as usize;
            let end = self.symbol_offsets[pos + 1] as usize;
            let ids = &self.symbol_record_ids[start..end];
            if ids.binary_search(&target_local_id).is_ok() {
                symbols.push(term.clone());
            }
        }

        let vector_id = if self.vector_id_presence[local_id] == 1 {
            let start = self.vector_id_offsets[local_id] as usize;
            let end = self.vector_id_offsets[local_id + 1] as usize;
            std::str::from_utf8(&self.vector_id_bytes[start..end])
                .map(String::from)
                .ok()
        } else {
            None
        };
        let metadata = {
            let start = self.metadata_offsets[local_id] as usize;
            let end = self.metadata_offsets[local_id + 1] as usize;
            let meta_str = std::str::from_utf8(&self.metadata_bytes[start..end]).unwrap_or("{}");
            serde_json::from_str(meta_str).unwrap_or_default()
        };

        MemoryRecordInput {
            record_id: self.record_ids[local_id],
            scope_id: self.scope_ids[local_id],
            timestamp: self.timestamps[local_id],
            source_id: self.source_ids[local_id],
            confidence: self.confidences[local_id],
            text: self.text(local_id).to_string(),
            embedding: if self.embeddings.is_empty() {
                self.cold_store
                    .as_ref()
                    .and_then(|cs| cs.read_vector(local_id).ok())
                    .unwrap_or_default()
            } else {
                self.embedding_row(local_id).to_vec()
            },
            symbols,
            vector_id,
            metadata,
        }
    }

    fn passes_filters(&self, local_id: usize, query: &RecallQuery) -> bool {
        if query
            .scope_id
            .is_some_and(|scope_id| self.scope_ids[local_id] != scope_id)
        {
            return false;
        }
        if query
            .time_start
            .is_some_and(|start| self.timestamps[local_id] < start)
        {
            return false;
        }
        if query
            .time_end
            .is_some_and(|end| self.timestamps[local_id] > end)
        {
            return false;
        }
        if query
            .min_confidence
            .is_some_and(|min| self.confidences[local_id] < min)
        {
            return false;
        }
        if let Some(query_vid) = &query.vector_id {
            let record_vid = if self.vector_id_presence[local_id] == 1 {
                let start = self.vector_id_offsets[local_id] as usize;
                let end = self.vector_id_offsets[local_id + 1] as usize;
                std::str::from_utf8(&self.vector_id_bytes[start..end]).ok()
            } else {
                None
            };
            if record_vid != Some(query_vid.as_str()) {
                return false;
            }
        }
        if !query.metadata_filter.is_empty() {
            let start = self.metadata_offsets[local_id] as usize;
            let end = self.metadata_offsets[local_id + 1] as usize;
            let meta_str = std::str::from_utf8(&self.metadata_bytes[start..end]).unwrap_or("{}");
            if meta_str == "{}" || meta_str.is_empty() {
                return false;
            }
            let record_meta: std::collections::BTreeMap<String, String> =
                serde_json::from_str(meta_str).unwrap_or_default();
            for (k, v) in &query.metadata_filter {
                if record_meta.get(k) != Some(v) {
                    return false;
                }
            }
        }
        true
    }

    fn symbol_postings(&self, symbol: &str) -> Option<&[u32]> {
        let symbol = normalize_symbol(symbol);
        let pos = self.symbol_terms.binary_search(&symbol).ok()?;
        let start = self.symbol_offsets[pos] as usize;
        let end = self.symbol_offsets[pos + 1] as usize;
        Some(&self.symbol_record_ids[start..end])
    }

    fn symbol_match_count(&self, local_id: usize, symbols: &[String]) -> usize {
        symbols
            .iter()
            .filter(|symbol| {
                self.symbol_postings(symbol)
                    .is_some_and(|ids| ids.binary_search(&(local_id as u32)).is_ok())
            })
            .count()
    }

    fn embedding_row(&self, local_id: usize) -> &[f32] {
        &self.embeddings[local_id * self.dim..(local_id + 1) * self.dim]
    }

    fn validate_vector_candidates(
        &self,
        candidates: &[u32],
        candidates_after_symbols: &[u32],
    ) -> Result<(), String> {
        for &local_id in candidates {
            if local_id as usize >= self.len() {
                return Err(format!(
                    "vector candidate local id out of range: {} >= {}",
                    local_id,
                    self.len()
                ));
            }
            if candidates_after_symbols.binary_search(&local_id).is_err() {
                return Err(format!(
                    "vector candidate local id was not produced by upstream filters: {}",
                    local_id
                ));
            }
        }
        Ok(())
    }

    fn score(&self, local_id: usize, semantic_distance: Option<f32>, symbol_matches: usize) -> f32 {
        let semantic = semantic_distance.map_or(0.0, |dist| -dist);
        let symbol = symbol_matches as f32 * 2.0 + self.record_symbol_count(local_id) as f32 * 0.01;
        let confidence = self.confidences[local_id];
        semantic + symbol + confidence
    }

    fn record_symbol_count(&self, local_id: usize) -> usize {
        let local_id = local_id as u32;
        self.symbol_record_ids
            .iter()
            .filter(|&&record_id| record_id == local_id)
            .count()
    }

    fn validate_layout(&self) -> Result<(), String> {
        if self.dim == 0 {
            return Err("dim must be greater than zero".to_string());
        }
        let record_count = self.record_ids.len();
        if self.scope_ids.len() != record_count
            || self.timestamps.len() != record_count
            || self.source_ids.len() != record_count
            || self.confidences.len() != record_count
        {
            return Err("record column length mismatch".to_string());
        }
        if self.text_offsets.len() != record_count + 1 {
            return Err("text_offsets must have record_count + 1 entries".to_string());
        }
        if self.text_offsets.first().copied() != Some(0) {
            return Err("text_offsets must start at zero".to_string());
        }
        if self.text_offsets.last().copied().map(usize::try_from) != Some(Ok(self.text_bytes.len()))
        {
            return Err("text_offsets must end at text_bytes length".to_string());
        }
        for window in self.text_offsets.windows(2) {
            if window[0] > window[1] {
                return Err("text_offsets must be monotonic".to_string());
            }
        }
        if std::str::from_utf8(&self.text_bytes).is_err() {
            return Err("text bytes must be valid utf-8".to_string());
        }
        if !self.embeddings.is_empty() && self.embeddings.len() != record_count * self.dim {
            return Err("embedding column length mismatch".to_string());
        }
        if self.symbol_offsets.len() != self.symbol_terms.len() + 1 {
            return Err("symbol_offsets must have symbol_count + 1 entries".to_string());
        }
        if self.symbol_offsets.first().copied() != Some(0) {
            return Err("symbol_offsets must start at zero".to_string());
        }
        if self.symbol_offsets.last().copied().map(usize::try_from)
            != Some(Ok(self.symbol_record_ids.len()))
        {
            return Err("symbol_offsets must end at symbol_record_ids length".to_string());
        }
        for window in self.symbol_offsets.windows(2) {
            if window[0] > window[1] {
                return Err("symbol_offsets must be monotonic".to_string());
            }
        }
        for term in &self.symbol_terms {
            if term != &normalize_symbol(term) {
                return Err("symbol terms must be normalized".to_string());
            }
        }
        for window in self.symbol_terms.windows(2) {
            if window[0] >= window[1] {
                return Err("symbol terms must be sorted and unique".to_string());
            }
        }
        for &local_id in &self.symbol_record_ids {
            if local_id as usize >= record_count {
                return Err("symbol posting local id out of range".to_string());
            }
        }

        if self.vector_id_presence.len() != record_count {
            return Err("vector_id_presence length mismatch".to_string());
        }
        if self.vector_id_offsets.len() != record_count + 1 {
            return Err("vector_id_offsets length mismatch".to_string());
        }
        if self.vector_id_offsets.first().copied() != Some(0) {
            return Err("vector_id_offsets must start at zero".to_string());
        }
        if self.vector_id_offsets.last().copied().map(usize::try_from)
            != Some(Ok(self.vector_id_bytes.len()))
        {
            return Err("vector_id_offsets must end at vector_id_bytes length".to_string());
        }
        for window in self.vector_id_offsets.windows(2) {
            if window[0] > window[1] {
                return Err("vector_id_offsets must be monotonic".to_string());
            }
        }
        if self.metadata_offsets.len() != record_count + 1 {
            return Err("metadata_offsets length mismatch".to_string());
        }
        if self.metadata_offsets.first().copied() != Some(0) {
            return Err("metadata_offsets must start at zero".to_string());
        }
        if self.metadata_offsets.last().copied().map(usize::try_from)
            != Some(Ok(self.metadata_bytes.len()))
        {
            return Err("metadata_offsets must end at metadata_bytes length".to_string());
        }
        for window in self.metadata_offsets.windows(2) {
            if window[0] > window[1] {
                return Err("metadata_offsets must be monotonic".to_string());
            }
        }
        Ok(())
    }
}

impl MemoryManifestFile {
    pub fn new(
        parent_manifest_id: Option<u64>,
        branch_id: u64,
        segments: Vec<MemoryManifestSegment>,
    ) -> Self {
        let manifest_id = stable_manifest_id(parent_manifest_id, branch_id, &segments);
        Self {
            version: MANIFEST_SCHEMA_VERSION,
            manifest_id,
            parent_manifest_id,
            branch_id,
            segments,
        }
    }

    pub fn write(&self, path: impl AsRef<Path>) -> io::Result<()> {
        self.validate().map_err(invalid_data)?;
        let bytes = serde_json::to_vec_pretty(self)
            .map_err(|err| invalid_data(format!("serialize manifest json: {err}")))?;
        fs::write(path, bytes)
    }

    pub fn read(path: impl AsRef<Path>) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        let manifest = serde_json::from_slice::<Self>(&bytes)
            .map_err(|err| invalid_data(format!("parse manifest json: {err}")))?;
        manifest.validate().map_err(invalid_data)?;
        Ok(manifest)
    }

    fn validate(&self) -> Result<(), String> {
        if self.version != MANIFEST_SCHEMA_VERSION {
            return Err(format!(
                "unsupported memory manifest version: {}",
                self.version
            ));
        }
        let expected_manifest_id =
            stable_manifest_id(self.parent_manifest_id, self.branch_id, &self.segments);
        if self.manifest_id != expected_manifest_id {
            return Err(format!(
                "manifest id mismatch: expected {}, got {}",
                expected_manifest_id, self.manifest_id
            ));
        }
        let mut segment_ids = BTreeSet::new();
        for segment in &self.segments {
            if !segment_ids.insert(segment.segment_id) {
                return Err("manifest segment ids must be unique".to_string());
            }
            if segment.path.as_os_str().is_empty() {
                return Err("manifest segment path must not be empty".to_string());
            }
            if segment.path.to_str().is_none() {
                return Err("manifest segment path must be valid utf-8".to_string());
            }
            if let Some(sidecar) = &segment.vector_sidecar {
                if sidecar.path.as_os_str().is_empty() {
                    return Err("manifest vector sidecar path must not be empty".to_string());
                }
                if sidecar.path.to_str().is_none() {
                    return Err("manifest vector sidecar path must be valid utf-8".to_string());
                }
                if sidecar
                    .expected_generator_name
                    .as_deref()
                    .is_some_and(str::is_empty)
                {
                    return Err(
                        "manifest vector sidecar expected generator name must not be empty"
                            .to_string(),
                    );
                }
            }
        }
        Ok(())
    }
}

impl MemoryVectorSidecarFile {
    pub fn for_segment_file(
        segment_path: impl AsRef<Path>,
        segment: &MemorySegment,
        generator_name: impl Into<String>,
        generator_version: u32,
    ) -> io::Result<Self> {
        Ok(Self {
            version: VECTOR_SIDECAR_VERSION,
            segment_id: segment.segment_id,
            segment_version: SEGMENT_VERSION,
            record_count: checked_u32(segment.len(), "record count")?,
            dim: checked_u32(segment.dim, "dimension")?,
            segment_fingerprint: segment_file_fingerprint(segment_path)?,
            generator_name: generator_name.into(),
            generator_version,
            checksum: 0,
        })
    }

    pub fn write(&self, path: impl AsRef<Path>) -> io::Result<()> {
        self.validate_layout().map_err(invalid_data)?;
        let generator_name = self.generator_name.as_bytes();
        let generator_name_len = checked_u32(generator_name.len(), "generator name length")?;

        let mut bytes = Vec::new();
        bytes.extend_from_slice(VECTOR_SIDECAR_MAGIC);
        write_u32(&mut bytes, self.version);
        write_u64(&mut bytes, self.segment_id);
        write_u32(&mut bytes, self.segment_version);
        write_u32(&mut bytes, self.record_count);
        write_u32(&mut bytes, self.dim);
        write_u64(&mut bytes, self.segment_fingerprint);
        write_u32(&mut bytes, self.generator_version);
        write_u32(&mut bytes, generator_name_len);
        bytes.extend_from_slice(generator_name);

        let checksum = checksum64(&bytes);
        write_u64(&mut bytes, checksum);
        fs::write(path, bytes)
    }

    pub fn read(path: impl AsRef<Path>) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        if bytes.len() < 52 {
            return Err(invalid_data("vector sidecar file is too short"));
        }
        let (payload, footer) = bytes.split_at(bytes.len() - 8);
        let expected_checksum = read_footer_checksum(footer)?;
        let actual_checksum = checksum64(payload);
        if expected_checksum != actual_checksum {
            return Err(invalid_data("vector sidecar checksum mismatch"));
        }

        let mut reader = SegmentReader::new(payload);
        let magic = reader.read_bytes(VECTOR_SIDECAR_MAGIC.len())?;
        if magic != VECTOR_SIDECAR_MAGIC {
            return Err(invalid_data("unsupported memory vector sidecar magic"));
        }
        let version = reader.read_u32()?;
        if version != VECTOR_SIDECAR_VERSION {
            return Err(invalid_data(format!(
                "unsupported memory vector sidecar version: {}",
                version
            )));
        }
        let segment_id = reader.read_u64()?;
        let segment_version = reader.read_u32()?;
        let record_count = reader.read_u32()?;
        let dim = reader.read_u32()?;
        let segment_fingerprint = reader.read_u64()?;
        let generator_version = reader.read_u32()?;
        let generator_name_len = reader.read_u32()? as usize;
        let generator_name = std::str::from_utf8(reader.read_bytes(generator_name_len)?)
            .map_err(|_| invalid_data("vector sidecar generator name is not valid utf-8"))?
            .to_string();
        reader.expect_end()?;

        let sidecar = Self {
            version,
            segment_id,
            segment_version,
            record_count,
            dim,
            segment_fingerprint,
            generator_name,
            generator_version,
            checksum: actual_checksum,
        };
        sidecar.validate_layout().map_err(invalid_data)?;
        Ok(sidecar)
    }

    fn validate_layout(&self) -> Result<(), String> {
        if self.version != VECTOR_SIDECAR_VERSION {
            return Err(format!(
                "unsupported memory vector sidecar version: {}",
                self.version
            ));
        }
        if self.segment_version != SEGMENT_VERSION {
            return Err(format!(
                "memory vector sidecar segment version mismatch: expected {}, got {}",
                SEGMENT_VERSION, self.segment_version
            ));
        }
        if self.dim == 0 {
            return Err("memory vector sidecar dimension must be greater than zero".to_string());
        }
        if self.generator_name.is_empty() {
            return Err("memory vector sidecar generator name must not be empty".to_string());
        }
        if self.generator_name.as_bytes().contains(&0) {
            return Err("memory vector sidecar generator name must not contain NUL".to_string());
        }
        Ok(())
    }

    fn validate_binding(
        &self,
        segment: &MemorySegment,
        segment_fingerprint: u64,
        reference: &MemoryVectorSidecarRef,
    ) -> Result<(), String> {
        if self.segment_id != segment.segment_id {
            return Err(format!(
                "memory vector sidecar segment id mismatch: expected {}, got {}",
                segment.segment_id, self.segment_id
            ));
        }
        if self.segment_version != SEGMENT_VERSION {
            return Err(format!(
                "memory vector sidecar segment version mismatch: expected {}, got {}",
                SEGMENT_VERSION, self.segment_version
            ));
        }
        if self.record_count as usize != segment.len() {
            return Err(format!(
                "memory vector sidecar record count mismatch: expected {}, got {}",
                segment.len(),
                self.record_count
            ));
        }
        if self.dim as usize != segment.dim {
            return Err(format!(
                "memory vector sidecar dimension mismatch: expected {}, got {}",
                segment.dim, self.dim
            ));
        }
        if self.segment_fingerprint != segment_fingerprint {
            return Err("memory vector sidecar segment fingerprint mismatch".to_string());
        }
        if let Some(expected) = reference.expected_sidecar_version {
            if self.version != expected {
                return Err(format!(
                    "memory vector sidecar version mismatch: expected {}, got {}",
                    expected, self.version
                ));
            }
        }
        if let Some(expected) = reference.expected_generator_name.as_deref() {
            if self.generator_name != expected {
                return Err(format!(
                    "memory vector sidecar generator mismatch: expected {}, got {}",
                    expected, self.generator_name
                ));
            }
        }
        if let Some(expected) = reference.expected_generator_version {
            if self.generator_version != expected {
                return Err(format!(
                    "memory vector sidecar generator version mismatch: expected {}, got {}",
                    expected, self.generator_version
                ));
            }
        }
        if let Some(expected) = reference.sidecar_checksum {
            if self.checksum != expected {
                return Err("memory vector sidecar checksum reference mismatch".to_string());
            }
        }
        if let Some(expected) = reference.segment_fingerprint {
            if self.segment_fingerprint != expected {
                return Err(
                    "memory vector sidecar segment fingerprint reference mismatch".to_string(),
                );
            }
        }
        Ok(())
    }
}

impl MemorySpace {
    pub fn open(manifest_path: impl AsRef<Path>) -> io::Result<Self> {
        let manifest_path = manifest_path.as_ref();
        let manifest = MemoryManifestFile::read(manifest_path)?;
        let base_dir = manifest_path.parent().unwrap_or_else(|| Path::new("."));
        let mut segments = Vec::with_capacity(manifest.segments.len());
        for entry in &manifest.segments {
            let segment_path = if entry.path.is_absolute() {
                entry.path.clone()
            } else {
                base_dir.join(&entry.path)
            };
            let segment = MemorySegment::read(&segment_path)?;
            if segment.segment_id != entry.segment_id {
                return Err(invalid_data(format!(
                    "manifest segment id {} does not match loaded segment id {}",
                    entry.segment_id, segment.segment_id
                )));
            }
            if let Some(sidecar_ref) = &entry.vector_sidecar {
                let sidecar_path = if sidecar_ref.path.is_absolute() {
                    sidecar_ref.path.clone()
                } else {
                    base_dir.join(&sidecar_ref.path)
                };
                let validate_result = Self::validate_vector_sidecar_ref(
                    &segment_path,
                    &segment,
                    &sidecar_path,
                    sidecar_ref,
                );
                if sidecar_ref.required {
                    validate_result?;
                }
            }
            segments.push(LoadedMemorySegment {
                stats: SegmentStats::from_segment(&segment),
                segment,
            });
        }

        let wal_path = base_dir.join("wal_active.apmw");
        let next_segment_id = manifest
            .segments
            .iter()
            .map(|s| s.segment_id)
            .max()
            .unwrap_or(0)
            + 1;
        let mut memtable = MemTable::new(next_segment_id);
        if wal_path.exists() {
            let mut reader = WALReader::open(&wal_path)?;
            while let Some(entry) = reader.next_entry()? {
                match entry {
                    WALEntry::Insert(record) => memtable.insert(record),
                    WALEntry::Delete(record_id) => memtable.delete(record_id),
                }
            }
        }
        let wal_writer = WALWriter::open(&wal_path)?;

        let mut space = Self {
            manifest,
            segments,
            memtable: std::sync::RwLock::new(memtable),
            wal_writer: std::sync::Mutex::new(wal_writer),
            wal_path,
            manifest_path: manifest_path.to_path_buf(),
            max_memtable_records: 5000,
            vector_encoding: None,
            raw_dram_budget: None,
        };
        space.enforce_dram_budget()?;
        Ok(space)
    }

    pub fn insert(&mut self, record: MemoryRecordInput) -> io::Result<()> {
        {
            let mut writer = self.wal_writer.lock().unwrap();
            writer.append_insert(&record)?;
            writer.sync()?;
        }
        let should_flush = {
            let mut mem = self.memtable.write().unwrap();
            mem.insert(record);
            mem.len() >= self.max_memtable_records
        };
        if should_flush {
            self.flush()?;
        }
        self.enforce_dram_budget()?;
        Ok(())
    }

    pub fn delete(&mut self, record_id: u64) -> io::Result<()> {
        {
            let mut writer = self.wal_writer.lock().unwrap();
            writer.append_delete(record_id)?;
            writer.sync()?;
        }
        {
            let mut mem = self.memtable.write().unwrap();
            mem.delete(record_id);
        }
        Ok(())
    }

    pub fn resident_raw_dram_bytes(&self) -> usize {
        let dim = self
            .segments
            .first()
            .map(|s| s.segment.dim)
            .or_else(|| {
                self.memtable
                    .read()
                    .unwrap()
                    .records
                    .first()
                    .map(|r| r.embedding.len())
            })
            .unwrap_or(0);
        let memtable_bytes = self.memtable.read().unwrap().records.len() * dim * 4;
        let segments_bytes: usize = self
            .segments
            .iter()
            .map(|s| s.segment.embeddings.len() * 4)
            .sum();
        memtable_bytes + segments_bytes
    }

    pub fn enforce_dram_budget(&mut self) -> std::io::Result<usize> {
        let mut total_offloaded = 0;
        if let Some(limit) = self.raw_dram_budget {
            while self.resident_raw_dram_bytes() > limit {
                let mut target_idx = None;
                for (idx, loaded_seg) in self.segments.iter().enumerate() {
                    if !loaded_seg.segment.embeddings.is_empty() {
                        target_idx = Some(idx);
                        break;
                    }
                }

                if let Some(idx) = target_idx {
                    let base_dir = self
                        .manifest_path
                        .parent()
                        .unwrap_or_else(|| Path::new("."));
                    let segment_id = self.segments[idx].segment.segment_id;
                    let segment_entry = self
                        .manifest
                        .segments
                        .iter()
                        .find(|s| s.segment_id == segment_id)
                        .cloned()
                        .ok_or_else(|| {
                            std::io::Error::new(
                                std::io::ErrorKind::NotFound,
                                format!("Segment ID {} not found in manifest", segment_id),
                            )
                        })?;
                    let segment_path = if segment_entry.path.is_absolute() {
                        segment_entry.path.clone()
                    } else {
                        base_dir.join(&segment_entry.path)
                    };
                    let apmc_path = segment_path.with_extension("apmc");

                    let segment = &mut self.segments[idx].segment;
                    if !apmc_path.exists() {
                        let encoding = self.vector_encoding.unwrap_or(VectorEncoding::Raw);
                        ColdVectorStore::write(
                            &apmc_path,
                            encoding,
                            segment.dim,
                            &segment.embeddings,
                        )?;
                    }

                    let cold = ColdVectorStore::open(&apmc_path)?;
                    segment.cold_store = Some(std::sync::Arc::new(cold));

                    let bytes_offloaded = segment.embeddings.len() * 4;
                    segment.embeddings = Vec::new();
                    total_offloaded += bytes_offloaded;
                } else {
                    break;
                }
            }
        }
        Ok(total_offloaded)
    }

    pub fn flush(&mut self) -> io::Result<()> {
        let mut mem = self.memtable.write().unwrap();
        if mem.records.is_empty() && mem.tombstones.is_empty() {
            return Ok(());
        }

        let space_dir = self.wal_path.parent().unwrap_or_else(|| Path::new("."));
        let mut next_segment_id = self
            .manifest
            .segments
            .iter()
            .map(|s| s.segment_id)
            .max()
            .unwrap_or(0)
            + 1;
        let mut new_manifest_segments = Vec::new();
        let mut files_to_delete = Vec::new();

        for loaded in &self.segments {
            let mut needs_compaction = false;
            for &rec_id in &loaded.segment.record_ids {
                if mem.is_deleted(rec_id) || mem.records.iter().any(|r| r.record_id == rec_id) {
                    needs_compaction = true;
                    break;
                }
            }

            if !needs_compaction {
                let entry = self
                    .manifest
                    .segments
                    .iter()
                    .find(|s| s.segment_id == loaded.segment.segment_id)
                    .unwrap();
                new_manifest_segments.push(entry.clone());
            } else {
                let mut compacted_records = Vec::new();
                for local_id in 0..loaded.segment.len() {
                    let rec_id = loaded.segment.record_ids[local_id];
                    if !mem.is_deleted(rec_id) && !mem.records.iter().any(|r| r.record_id == rec_id)
                    {
                        compacted_records.push(loaded.segment.get_record_input(local_id));
                    }
                }

                if !compacted_records.is_empty() {
                    let mut new_seg = MemorySegment::build(
                        next_segment_id,
                        loaded.segment.dim,
                        compacted_records,
                    )
                    .map_err(io::Error::other)?;
                    let new_seg_filename = format!("segment_{}.apms", next_segment_id);
                    let new_seg_path = space_dir.join(&new_seg_filename);
                    if let Some(encoding) = self.vector_encoding {
                        let apmc_filename = format!("segment_{}.apmc", next_segment_id);
                        let apmc_path = space_dir.join(&apmc_filename);
                        ColdVectorStore::write(
                            &apmc_path,
                            encoding,
                            loaded.segment.dim,
                            &new_seg.embeddings,
                        )?;
                        new_seg.embeddings = Vec::new();
                    }
                    new_seg.write(&new_seg_path)?;

                    new_manifest_segments.push(MemoryManifestSegment {
                        segment_id: next_segment_id,
                        path: PathBuf::from(new_seg_filename),
                        vector_sidecar: None,
                    });
                    next_segment_id += 1;
                }

                // Delete old segment
                let entry = self
                    .manifest
                    .segments
                    .iter()
                    .find(|s| s.segment_id == loaded.segment.segment_id)
                    .unwrap();
                let segment_path = if entry.path.is_absolute() {
                    entry.path.clone()
                } else {
                    space_dir.join(&entry.path)
                };
                files_to_delete.push(segment_path.clone());
                let apmc_path = segment_path.with_extension("apmc");
                if apmc_path.exists() {
                    files_to_delete.push(apmc_path);
                }

                if let Some(sidecar_ref) = &entry.vector_sidecar {
                    let sidecar_path = if sidecar_ref.path.is_absolute() {
                        sidecar_ref.path.clone()
                    } else {
                        space_dir.join(&sidecar_ref.path)
                    };
                    files_to_delete.push(sidecar_path);
                }
            }
        }

        if !mem.records.is_empty() {
            let dim = mem.records[0].embedding.len();
            let mut new_seg = MemorySegment::build(next_segment_id, dim, mem.records.clone())
                .map_err(io::Error::other)?;
            let new_seg_filename = format!("segment_{}.apms", next_segment_id);
            let new_seg_path = space_dir.join(&new_seg_filename);
            if let Some(encoding) = self.vector_encoding {
                let apmc_filename = format!("segment_{}.apmc", next_segment_id);
                let apmc_path = space_dir.join(&apmc_filename);
                ColdVectorStore::write(&apmc_path, encoding, dim, &new_seg.embeddings)?;
                new_seg.embeddings = Vec::new();
            }
            new_seg.write(&new_seg_path)?;

            new_manifest_segments.push(MemoryManifestSegment {
                segment_id: next_segment_id,
                path: PathBuf::from(new_seg_filename),
                vector_sidecar: None,
            });
            next_segment_id += 1;
        }

        let new_manifest = MemoryManifestFile::new(
            Some(self.manifest.manifest_id),
            self.manifest.branch_id,
            new_manifest_segments,
        );
        new_manifest.write(&self.manifest_path)?;

        let mut new_loaded_segments = Vec::with_capacity(new_manifest.segments.len());
        for entry in &new_manifest.segments {
            if let Some(existing) = self
                .segments
                .iter()
                .find(|s| s.segment.segment_id == entry.segment_id)
            {
                new_loaded_segments.push(existing.clone());
            } else {
                let segment_path = if entry.path.is_absolute() {
                    entry.path.clone()
                } else {
                    space_dir.join(&entry.path)
                };
                let segment = MemorySegment::read(&segment_path)?;
                new_loaded_segments.push(LoadedMemorySegment {
                    stats: SegmentStats::from_segment(&segment),
                    segment,
                });
            }
        }

        self.manifest = new_manifest;
        self.segments = new_loaded_segments;

        let _ = fs::remove_file(&self.wal_path);
        *mem = MemTable::new(next_segment_id);

        {
            let mut wal = self.wal_writer.lock().unwrap();
            *wal = WALWriter::open(&self.wal_path)?;
        }

        for path in files_to_delete {
            let _ = fs::remove_file(path);
        }

        Ok(())
    }

    fn validate_vector_sidecar_ref(
        segment_path: &Path,
        segment: &MemorySegment,
        sidecar_path: &Path,
        sidecar_ref: &MemoryVectorSidecarRef,
    ) -> io::Result<()> {
        let segment_fingerprint = segment_file_fingerprint(segment_path)?;
        let sidecar = MemoryVectorSidecarFile::read(sidecar_path)?;
        sidecar
            .validate_binding(segment, segment_fingerprint, sidecar_ref)
            .map_err(invalid_data)
    }

    pub fn recall(&self, query: &RecallQuery) -> Result<MemorySpaceRecallResult, String> {
        let limit = query.limit.max(1);
        let mut merged = Vec::new();
        let mut segment_traces = Vec::with_capacity(self.segments.len());
        let mut segments_scanned = 0;
        let mut segments_pruned = 0;
        let mut semantic_evals = 0;

        let mem = self.memtable.read().unwrap();

        for loaded in &self.segments {
            if let Some(reason) = loaded.stats.prune_reason(query) {
                segments_pruned += 1;
                segment_traces.push(MemorySpaceSegmentTrace {
                    segment_id: loaded.segment.segment_id,
                    pruned: true,
                    prune_reason: Some(reason),
                    trace: None,
                });
                continue;
            }

            segments_scanned += 1;
            let result = loaded.segment.recall(query)?;
            semantic_evals += result.trace.semantic_evals;
            for hit in result.hits {
                if mem.is_deleted(hit.record_id)
                    || mem.records.iter().any(|r| r.record_id == hit.record_id)
                {
                    continue;
                }
                merged.push((loaded.segment.segment_id, hit));
            }
            segment_traces.push(MemorySpaceSegmentTrace {
                segment_id: loaded.segment.segment_id,
                pruned: false,
                prune_reason: None,
                trace: Some(result.trace),
            });
        }

        // Query memtable records
        let mut mem_evals = 0;
        for record in mem.records() {
            if record.passes_filters(query) {
                if query.embedding.is_some() {
                    mem_evals += 1;
                }

                let score = record.score(query);

                let semantic_distance = query
                    .embedding
                    .as_ref()
                    .map(|query_emb| l2_squared_unchecked(query_emb, &record.embedding));

                let symbol_matches = if query.symbols.is_empty() {
                    0
                } else {
                    let record_syms_normalized: HashSet<String> =
                        record.symbols.iter().map(|s| normalize_symbol(s)).collect();
                    query
                        .symbols
                        .iter()
                        .filter(|sym| record_syms_normalized.contains(&normalize_symbol(sym)))
                        .count()
                };

                let hit = MemoryHit {
                    record_id: record.record_id,
                    score,
                    semantic_distance,
                    symbol_matches,
                    confidence: record.confidence,
                    timestamp: record.timestamp,
                    text: record.text.clone(),
                    vector_id: record.vector_id.clone(),
                    metadata: record.metadata.clone(),
                };
                merged.push((mem.segment_id, hit));
            }
        }

        if query.embedding.is_some() {
            semantic_evals += mem_evals;
        }

        merged.sort_unstable_by(|a, b| {
            b.1.score
                .total_cmp(&a.1.score)
                .then_with(|| a.0.cmp(&b.0))
                .then_with(|| a.1.record_id.cmp(&b.1.record_id))
        });
        let hits = merged
            .into_iter()
            .take(limit)
            .map(|(_, hit)| hit)
            .collect::<Vec<_>>();

        let total_cold_bytes_read = segment_traces
            .iter()
            .filter_map(|st| st.trace.as_ref())
            .map(|t| t.cold_bytes_read)
            .sum::<usize>();

        let dim = self.segments.first().map(|s| s.segment.dim).unwrap_or(0);
        let bytes_needed = hits.len() * dim * 4;
        let read_amplification = if bytes_needed > 0 {
            total_cold_bytes_read as f32 / bytes_needed as f32
        } else {
            1.0
        };

        Ok(MemorySpaceRecallResult {
            trace: MemorySpaceRecallTrace {
                manifest_id: self.manifest.manifest_id,
                branch_id: self.manifest.branch_id,
                segments_considered: self.segments.len(),
                segments_scanned,
                segments_pruned,
                semantic_evals,
                returned: hits.len(),
                segment_traces,
                cold_bytes_read: total_cold_bytes_read,
                read_amplification,
            },
            hits,
        })
    }

    pub fn fork(&self, branch_id_str: &str, out_manifest_path: impl AsRef<Path>) -> io::Result<()> {
        let branch_id = stable_memory_branch_id(branch_id_str);
        let child = MemoryManifestFile::new(
            Some(self.manifest.manifest_id),
            branch_id,
            self.manifest.segments.clone(),
        );
        child.write(out_manifest_path)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LoadedMemorySegment {
    pub segment: MemorySegment,
    stats: SegmentStats,
}

#[derive(Clone, Debug, PartialEq)]
struct SegmentStats {
    scope_min: Option<u32>,
    scope_max: Option<u32>,
    time_min: Option<i64>,
    time_max: Option<i64>,
}

impl SegmentStats {
    fn from_segment(segment: &MemorySegment) -> Self {
        Self {
            scope_min: segment.scope_ids.iter().min().copied(),
            scope_max: segment.scope_ids.iter().max().copied(),
            time_min: segment.timestamps.iter().min().copied(),
            time_max: segment.timestamps.iter().max().copied(),
        }
    }

    fn prune_reason(&self, query: &RecallQuery) -> Option<&'static str> {
        if let Some(scope_id) = query.scope_id {
            if self
                .scope_min
                .zip(self.scope_max)
                .is_some_and(|(min, max)| scope_id < min || scope_id > max)
            {
                return Some("scope_range");
            }
        }
        if let Some(start) = query.time_start {
            if self.time_max.is_some_and(|max| max < start) {
                return Some("time_range");
            }
        }
        if let Some(end) = query.time_end {
            if self.time_min.is_some_and(|min| min > end) {
                return Some("time_range");
            }
        }
        None
    }
}

fn normalize_symbol(symbol: &str) -> String {
    symbol.trim().to_ascii_lowercase()
}

fn checked_u32(value: usize, name: &str) -> io::Result<u32> {
    u32::try_from(value).map_err(|_| invalid_data(format!("{} does not fit in u32", name)))
}

fn checked_u64(value: usize, name: &str) -> io::Result<u64> {
    u64::try_from(value).map_err(|_| invalid_data(format!("{} does not fit in u64", name)))
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn default_memory_pivot_prefix_config(segment: &MemorySegment) -> PivotPrefixConfig {
    let record_count = segment.len().max(1);
    PivotPrefixConfig {
        block_size: record_count.clamp(1, 64),
        pivot_count: record_count.clamp(1, 16),
        prefix_len: 1,
        top_blocks: 1,
        candidate_pool: record_count,
        mode: PrefixScoreMode::Weighted,
        cluster_iters: 2,
    }
}

fn default_memory_htla_config(segment: &MemorySegment) -> HtlaMemoryVectorConfig {
    let record_count = segment.len().max(1);
    HtlaMemoryVectorConfig {
        levels: if record_count >= 8 { 3 } else { 2 },
        chart_dim: segment.dim.clamp(1, 4),
        beam: 4,
        candidate_pool: record_count.clamp(1, 64),
        final_nprobe: record_count.clamp(1, 16),
        fallback_on_route_risk: true,
    }
}

fn memory_vector_route_trace(
    vector_index_bytes: usize,
    metrics: RouteMetrics,
) -> MemoryVectorRouteTrace {
    MemoryVectorRouteTrace {
        vector_index_bytes,
        route_candidates: metrics.candidate_count,
        posting_entries_touched: metrics.posting_entries_touched,
        duplicate_block_rate: metrics.duplicate_block_rate,
        selected_blocks: metrics.selected_blocks,
        centroid_evals: metrics.centroid_evals,
        working_set_bytes: metrics.working_set_bytes,
        fallback_used: metrics.fallback,
    }
}

fn checksum64(bytes: &[u8]) -> u64 {
    let mut checksum = CHECKSUM_OFFSET_BASIS;
    for byte in bytes {
        checksum ^= u64::from(*byte);
        checksum = checksum.wrapping_mul(CHECKSUM_PRIME);
    }
    checksum
}

fn stable_manifest_id(
    parent_manifest_id: Option<u64>,
    branch_id: u64,
    segments: &[MemoryManifestSegment],
) -> u64 {
    let mut bytes = Vec::new();
    write_u32(&mut bytes, MANIFEST_SCHEMA_VERSION);
    match parent_manifest_id {
        Some(parent_manifest_id) => {
            bytes.push(1);
            write_u64(&mut bytes, parent_manifest_id);
        }
        None => {
            bytes.push(0);
            write_u64(&mut bytes, 0);
        }
    }
    write_u64(&mut bytes, branch_id);
    write_u64(&mut bytes, segments.len() as u64);
    for segment in segments {
        write_u64(&mut bytes, segment.segment_id);
        bytes.extend_from_slice(&stable_path_bytes(&segment.path));
        bytes.push(0);
    }
    checksum64(&bytes)
}

pub fn stable_memory_branch_id(branch_id_str: &str) -> u64 {
    checksum64(branch_id_str.as_bytes())
}

fn stable_path_bytes(path: &Path) -> Vec<u8> {
    path.to_string_lossy().replace('\\', "/").into_bytes()
}

fn write_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn write_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn write_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn write_i64(bytes: &mut Vec<u8>, value: i64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn write_f32(bytes: &mut Vec<u8>, value: f32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn read_footer_checksum(bytes: &[u8]) -> io::Result<u64> {
    let array: [u8; 8] = bytes
        .try_into()
        .map_err(|_| invalid_data("segment footer is malformed"))?;
    Ok(u64::from_le_bytes(array))
}

fn segment_file_fingerprint(path: impl AsRef<Path>) -> io::Result<u64> {
    let bytes = fs::read(path)?;
    if bytes.len() < 8 {
        return Err(invalid_data("segment file is too short"));
    }
    read_footer_checksum(&bytes[bytes.len() - 8..])
}

struct SegmentReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> SegmentReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn expect_magic(&mut self) -> io::Result<()> {
        let magic = self.read_bytes(SEGMENT_MAGIC.len())?;
        if magic != SEGMENT_MAGIC {
            return Err(invalid_data("unsupported memory segment magic"));
        }
        Ok(())
    }

    fn expect_end(&self) -> io::Result<()> {
        if self.offset != self.bytes.len() {
            return Err(invalid_data("trailing bytes in memory segment"));
        }
        Ok(())
    }

    fn check_remaining(&self, count: usize, size_of_element: usize) -> io::Result<()> {
        let remaining = self.bytes.len().saturating_sub(self.offset);
        if count
            .checked_mul(size_of_element)
            .is_none_or(|needed| needed > remaining)
        {
            return Err(invalid_data(
                "unexpected end of memory segment or size mismatch",
            ));
        }
        Ok(())
    }

    fn read_bytes(&mut self, len: usize) -> io::Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| invalid_data("segment offset overflow"))?;
        if end > self.bytes.len() {
            return Err(invalid_data("unexpected end of memory segment"));
        }
        let slice = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(slice)
    }

    fn read_u16(&mut self) -> io::Result<u16> {
        let bytes: [u8; 2] = self.read_bytes(2)?.try_into().unwrap();
        Ok(u16::from_le_bytes(bytes))
    }

    fn read_u32(&mut self) -> io::Result<u32> {
        let bytes: [u8; 4] = self.read_bytes(4)?.try_into().unwrap();
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_u64(&mut self) -> io::Result<u64> {
        let bytes: [u8; 8] = self.read_bytes(8)?.try_into().unwrap();
        Ok(u64::from_le_bytes(bytes))
    }

    fn read_i64(&mut self) -> io::Result<i64> {
        let bytes: [u8; 8] = self.read_bytes(8)?.try_into().unwrap();
        Ok(i64::from_le_bytes(bytes))
    }

    fn read_f32(&mut self) -> io::Result<f32> {
        let bytes: [u8; 4] = self.read_bytes(4)?.try_into().unwrap();
        Ok(f32::from_le_bytes(bytes))
    }

    fn read_u16_vec(&mut self, len: usize) -> io::Result<Vec<u16>> {
        self.check_remaining(len, 2)?;
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            values.push(self.read_u16()?);
        }
        Ok(values)
    }

    fn read_u32_vec(&mut self, len: usize) -> io::Result<Vec<u32>> {
        self.check_remaining(len, 4)?;
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            values.push(self.read_u32()?);
        }
        Ok(values)
    }

    fn read_u64_vec(&mut self, len: usize) -> io::Result<Vec<u64>> {
        self.check_remaining(len, 8)?;
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            values.push(self.read_u64()?);
        }
        Ok(values)
    }

    fn read_i64_vec(&mut self, len: usize) -> io::Result<Vec<i64>> {
        self.check_remaining(len, 8)?;
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            values.push(self.read_i64()?);
        }
        Ok(values)
    }

    fn read_f32_vec(&mut self, len: usize) -> io::Result<Vec<f32>> {
        self.check_remaining(len, 4)?;
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            values.push(self.read_f32()?);
        }
        Ok(values)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MemTable {
    pub segment_id: u64,
    pub records: Vec<MemoryRecordInput>,
    pub tombstones: HashSet<u64>,
}

impl MemTable {
    pub fn new(segment_id: u64) -> Self {
        Self {
            segment_id,
            records: Vec::new(),
            tombstones: HashSet::new(),
        }
    }

    pub fn insert(&mut self, record: MemoryRecordInput) {
        self.tombstones.remove(&record.record_id);
        if let Some(pos) = self
            .records
            .iter()
            .position(|r| r.record_id == record.record_id)
        {
            self.records[pos] = record;
        } else {
            self.records.push(record);
        }
    }

    pub fn delete(&mut self, record_id: u64) {
        self.tombstones.insert(record_id);
        self.records.retain(|r| r.record_id != record_id);
    }

    pub fn is_deleted(&self, record_id: u64) -> bool {
        self.tombstones.contains(&record_id)
    }

    pub fn get_record(&self, record_id: u64) -> Option<&MemoryRecordInput> {
        if self.is_deleted(record_id) {
            None
        } else {
            self.records.iter().find(|r| r.record_id == record_id)
        }
    }

    pub fn records(&self) -> &[MemoryRecordInput] {
        &self.records
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

#[cfg(unix)]
fn lock_file(file: &std::fs::File) -> io::Result<()> {
    use std::os::unix::io::AsRawFd;
    let fd = file.as_raw_fd();
    let res = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
    if res != 0 {
        let err = io::Error::last_os_error();
        if err.kind() == io::ErrorKind::WouldBlock || err.raw_os_error() == Some(libc::EWOULDBLOCK)
        {
            return Err(io::Error::new(
                io::ErrorKind::AddrInUse,
                "WAL file is already locked by another process or cloned MemorySpace writer",
            ));
        }
        return Err(err);
    }
    Ok(())
}

#[cfg(unix)]
fn unlock_file(file: &std::fs::File) -> io::Result<()> {
    use std::os::unix::io::AsRawFd;
    let fd = file.as_raw_fd();
    let res = unsafe { libc::flock(fd, libc::LOCK_UN) };
    if res != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(not(unix))]
fn lock_file(_file: &std::fs::File) -> io::Result<()> {
    Ok(())
}

#[cfg(not(unix))]
fn unlock_file(_file: &std::fs::File) -> io::Result<()> {
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub enum WALEntry {
    Insert(MemoryRecordInput),
    Delete(u64),
}

#[derive(Debug)]
pub struct WALWriter {
    file: File,
}

impl Drop for WALWriter {
    fn drop(&mut self) {
        let _ = unlock_file(&self.file);
    }
}

impl WALWriter {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = OpenOptions::new().append(true).create(true).open(path)?;
        lock_file(&file)?;
        Ok(Self { file })
    }

    pub fn append_insert(&mut self, record: &MemoryRecordInput) -> io::Result<()> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&record.record_id.to_le_bytes());
        payload.extend_from_slice(&record.scope_id.to_le_bytes());
        payload.extend_from_slice(&record.timestamp.to_le_bytes());
        payload.extend_from_slice(&record.source_id.to_le_bytes());
        payload.extend_from_slice(&record.confidence.to_le_bytes());

        // text
        let text_bytes = record.text.as_bytes();
        payload.extend_from_slice(&(text_bytes.len() as u32).to_le_bytes());
        payload.extend_from_slice(text_bytes);

        // embedding
        payload.extend_from_slice(&(record.embedding.len() as u32).to_le_bytes());
        for &val in &record.embedding {
            payload.extend_from_slice(&val.to_le_bytes());
        }

        // symbols
        payload.extend_from_slice(&(record.symbols.len() as u32).to_le_bytes());
        for sym in &record.symbols {
            let sym_bytes = sym.as_bytes();
            payload.extend_from_slice(&(sym_bytes.len() as u32).to_le_bytes());
            payload.extend_from_slice(sym_bytes);
        }

        // vector_id
        if let Some(ref vid) = record.vector_id {
            payload.push(1);
            let vid_bytes = vid.as_bytes();
            payload.extend_from_slice(&(vid_bytes.len() as u32).to_le_bytes());
            payload.extend_from_slice(vid_bytes);
        } else {
            payload.push(0);
        }

        // metadata
        payload.extend_from_slice(&(record.metadata.len() as u32).to_le_bytes());
        for (key, val) in &record.metadata {
            let key_bytes = key.as_bytes();
            payload.extend_from_slice(&(key_bytes.len() as u32).to_le_bytes());
            payload.extend_from_slice(key_bytes);

            let val_bytes = val.as_bytes();
            payload.extend_from_slice(&(val_bytes.len() as u32).to_le_bytes());
            payload.extend_from_slice(val_bytes);
        }

        self.write_entry(1, &payload)
    }

    pub fn append_delete(&mut self, record_id: u64) -> io::Result<()> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&record_id.to_le_bytes());
        self.write_entry(2, &payload)
    }

    fn write_entry(&mut self, entry_type: u8, payload: &[u8]) -> io::Result<()> {
        let mut entry = Vec::with_capacity(4 + 1 + 1 + 4 + payload.len() + 8);
        entry.extend_from_slice(b"APMW"); // Magic
        entry.push(0); // Version
        entry.push(entry_type); // EntryType
        entry.extend_from_slice(&(payload.len() as u32).to_le_bytes()); // Payload Size
        entry.extend_from_slice(payload);

        // Compute checksum
        let checksum = checksum64(&entry);
        entry.extend_from_slice(&checksum.to_le_bytes());

        self.file.write_all(&entry)
    }

    pub fn sync(&mut self) -> io::Result<()> {
        self.file.sync_all()
    }
}

#[derive(Debug)]
pub struct WALReader {
    bytes: Vec<u8>,
    offset: usize,
}

impl WALReader {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        Ok(Self { bytes, offset: 0 })
    }

    pub fn next_entry(&mut self) -> io::Result<Option<WALEntry>> {
        if self.offset == self.bytes.len() {
            return Ok(None);
        }

        // Check if there is enough data for a header (10 bytes)
        let entry_start = self.offset;
        if entry_start + 10 > self.bytes.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "unexpected EOF in header",
            ));
        }

        // Read header
        let magic = &self.bytes[entry_start..entry_start + 4];
        if magic != b"APMW" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid WAL magic",
            ));
        }

        let version = self.bytes[entry_start + 4];
        if version != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid WAL version",
            ));
        }

        let entry_type = self.bytes[entry_start + 5];
        if entry_type != 1 && entry_type != 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid WAL entry type",
            ));
        }

        let payload_size = u32::from_le_bytes(
            self.bytes[entry_start + 6..entry_start + 10]
                .try_into()
                .unwrap(),
        ) as usize;

        // Check if there is enough data for payload and checksum (8 bytes)
        let payload_start = entry_start + 10;
        if payload_start + payload_size > self.bytes.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "unexpected EOF in payload",
            ));
        }

        let checksum_start = payload_start + payload_size;
        if checksum_start + 8 > self.bytes.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "unexpected EOF in checksum",
            ));
        }

        // Validate checksum
        let expected_checksum = checksum64(&self.bytes[entry_start..checksum_start]);
        let read_checksum = u64::from_le_bytes(
            self.bytes[checksum_start..checksum_start + 8]
                .try_into()
                .unwrap(),
        );
        if expected_checksum != read_checksum {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "WAL checksum mismatch",
            ));
        }

        // Deserialize payload
        let payload_bytes = &self.bytes[payload_start..checksum_start];
        let mut p_offset = 0;

        let entry = match entry_type {
            1 => {
                // Insert
                if p_offset + 8 > payload_bytes.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "malformed payload (record_id)",
                    ));
                }
                let record_id =
                    u64::from_le_bytes(payload_bytes[p_offset..p_offset + 8].try_into().unwrap());
                p_offset += 8;

                if p_offset + 4 > payload_bytes.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "malformed payload (scope_id)",
                    ));
                }
                let scope_id =
                    u32::from_le_bytes(payload_bytes[p_offset..p_offset + 4].try_into().unwrap());
                p_offset += 4;

                if p_offset + 8 > payload_bytes.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "malformed payload (timestamp)",
                    ));
                }
                let timestamp =
                    i64::from_le_bytes(payload_bytes[p_offset..p_offset + 8].try_into().unwrap());
                p_offset += 8;

                if p_offset + 2 > payload_bytes.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "malformed payload (source_id)",
                    ));
                }
                let source_id =
                    u16::from_le_bytes(payload_bytes[p_offset..p_offset + 2].try_into().unwrap());
                p_offset += 2;

                if p_offset + 4 > payload_bytes.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "malformed payload (confidence)",
                    ));
                }
                let confidence =
                    f32::from_le_bytes(payload_bytes[p_offset..p_offset + 4].try_into().unwrap());
                p_offset += 4;

                // text
                if p_offset + 4 > payload_bytes.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "malformed payload (text length)",
                    ));
                }
                let text_len =
                    u32::from_le_bytes(payload_bytes[p_offset..p_offset + 4].try_into().unwrap())
                        as usize;
                p_offset += 4;

                if p_offset + text_len > payload_bytes.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "malformed payload (text bytes)",
                    ));
                }
                let text = String::from_utf8(payload_bytes[p_offset..p_offset + text_len].to_vec())
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
                p_offset += text_len;

                // embedding
                if p_offset + 4 > payload_bytes.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "malformed payload (embedding length)",
                    ));
                }
                let emb_len =
                    u32::from_le_bytes(payload_bytes[p_offset..p_offset + 4].try_into().unwrap())
                        as usize;
                p_offset += 4;

                if p_offset + emb_len * 4 > payload_bytes.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "malformed payload (embedding elements)",
                    ));
                }
                let mut embedding = Vec::with_capacity(emb_len);
                for _ in 0..emb_len {
                    let val = f32::from_le_bytes(
                        payload_bytes[p_offset..p_offset + 4].try_into().unwrap(),
                    );
                    embedding.push(val);
                    p_offset += 4;
                }

                // symbols
                if p_offset + 4 > payload_bytes.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "malformed payload (symbols length)",
                    ));
                }
                let syms_len =
                    u32::from_le_bytes(payload_bytes[p_offset..p_offset + 4].try_into().unwrap())
                        as usize;
                p_offset += 4;

                let mut symbols = Vec::with_capacity(syms_len);
                for _ in 0..syms_len {
                    if p_offset + 4 > payload_bytes.len() {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "malformed payload (symbol length)",
                        ));
                    }
                    let sym_len = u32::from_le_bytes(
                        payload_bytes[p_offset..p_offset + 4].try_into().unwrap(),
                    ) as usize;
                    p_offset += 4;

                    if p_offset + sym_len > payload_bytes.len() {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "malformed payload (symbol bytes)",
                        ));
                    }
                    let sym =
                        String::from_utf8(payload_bytes[p_offset..p_offset + sym_len].to_vec())
                            .map_err(|e| {
                                io::Error::new(io::ErrorKind::InvalidData, e.to_string())
                            })?;
                    symbols.push(sym);
                    p_offset += sym_len;
                }

                // vector_id
                if p_offset + 1 > payload_bytes.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "malformed payload (vector_id presence)",
                    ));
                }
                let presence = payload_bytes[p_offset];
                p_offset += 1;
                let vector_id = if presence == 1 {
                    if p_offset + 4 > payload_bytes.len() {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "malformed payload (vector_id length)",
                        ));
                    }
                    let len = u32::from_le_bytes(
                        payload_bytes[p_offset..p_offset + 4].try_into().unwrap(),
                    ) as usize;
                    p_offset += 4;
                    if p_offset + len > payload_bytes.len() {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "malformed payload (vector_id bytes)",
                        ));
                    }
                    let vid = String::from_utf8(payload_bytes[p_offset..p_offset + len].to_vec())
                        .map_err(|e| {
                        io::Error::new(io::ErrorKind::InvalidData, e.to_string())
                    })?;
                    p_offset += len;
                    Some(vid)
                } else {
                    None
                };

                // metadata
                if p_offset + 4 > payload_bytes.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "malformed payload (metadata len)",
                    ));
                }
                let meta_len =
                    u32::from_le_bytes(payload_bytes[p_offset..p_offset + 4].try_into().unwrap())
                        as usize;
                p_offset += 4;

                let mut metadata = BTreeMap::new();
                for _ in 0..meta_len {
                    if p_offset + 4 > payload_bytes.len() {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "malformed payload (metadata key len)",
                        ));
                    }
                    let key_len = u32::from_le_bytes(
                        payload_bytes[p_offset..p_offset + 4].try_into().unwrap(),
                    ) as usize;
                    p_offset += 4;
                    if p_offset + key_len > payload_bytes.len() {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "malformed payload (metadata key bytes)",
                        ));
                    }
                    let key =
                        String::from_utf8(payload_bytes[p_offset..p_offset + key_len].to_vec())
                            .map_err(|e| {
                                io::Error::new(io::ErrorKind::InvalidData, e.to_string())
                            })?;
                    p_offset += key_len;

                    if p_offset + 4 > payload_bytes.len() {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "malformed payload (metadata val len)",
                        ));
                    }
                    let val_len = u32::from_le_bytes(
                        payload_bytes[p_offset..p_offset + 4].try_into().unwrap(),
                    ) as usize;
                    p_offset += 4;
                    if p_offset + val_len > payload_bytes.len() {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "malformed payload (metadata val bytes)",
                        ));
                    }
                    let val =
                        String::from_utf8(payload_bytes[p_offset..p_offset + val_len].to_vec())
                            .map_err(|e| {
                                io::Error::new(io::ErrorKind::InvalidData, e.to_string())
                            })?;
                    p_offset += val_len;

                    metadata.insert(key, val);
                }

                if p_offset != payload_bytes.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "unconsumed bytes in Insert payload",
                    ));
                }

                WALEntry::Insert(MemoryRecordInput {
                    record_id,
                    scope_id,
                    timestamp,
                    source_id,
                    confidence,
                    text,
                    embedding,
                    symbols,
                    vector_id,
                    metadata,
                })
            }
            2 => {
                // Delete
                if p_offset + 8 > payload_bytes.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "malformed payload (delete record_id)",
                    ));
                }
                let record_id =
                    u64::from_le_bytes(payload_bytes[p_offset..p_offset + 8].try_into().unwrap());
                p_offset += 8;

                if p_offset != payload_bytes.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "unconsumed bytes in Delete payload",
                    ));
                }

                WALEntry::Delete(record_id)
            }
            _ => unreachable!(),
        };

        self.offset = checksum_start + 8;
        Ok(Some(entry))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum VectorEncoding {
    Raw,
    F16,
    SQ8,
}

pub fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let fraction = bits & 0x007fffff;

    if exp == 0 {
        sign
    } else if exp == 0xff {
        if fraction == 0 {
            sign | 0x7c00
        } else {
            sign | 0x7c00 | ((fraction >> 13) & 0x03ff) as u16 | 1
        }
    } else {
        let new_exp = exp - 127 + 15;
        if new_exp >= 31 {
            sign | 0x7c00
        } else if new_exp <= 0 {
            if new_exp < -10 {
                sign
            } else {
                let shift = 11 - new_exp;
                let subnormal_fraction = (0x00800000 | fraction) >> shift;
                sign | (subnormal_fraction as u16)
            }
        } else {
            let new_fraction = ((fraction + 0x00001000) >> 13) as u16;
            if new_fraction >= 0x0400 {
                let carried_exp = new_exp + 1;
                if carried_exp >= 31 {
                    sign | 0x7c00
                } else {
                    sign | ((carried_exp as u16) << 10)
                }
            } else {
                sign | ((new_exp as u16) << 10) | new_fraction
            }
        }
    }
}

pub fn f16_to_f32(val: u16) -> f32 {
    let sign = (val & 0x8000) as u32;
    let exp = ((val & 0x7c00) >> 10) as u32;
    let fraction = (val & 0x03ff) as u32;

    let res_bits = if exp == 0 {
        if fraction == 0 {
            sign << 16
        } else {
            let mut f = fraction;
            let mut e = 0;
            while (f & 0x0400) == 0 {
                f <<= 1;
                e += 1;
            }
            let new_exp = 127 - 15 - e + 1;
            let new_fraction = (f & 0x03ff) << 13;
            (sign << 16) | (new_exp << 23) | new_fraction
        }
    } else if exp == 31 {
        if fraction == 0 {
            (sign << 16) | 0x7f800000
        } else {
            (sign << 16) | 0x7f800000 | (fraction << 13)
        }
    } else {
        let new_exp = exp - 15 + 127;
        let new_fraction = fraction << 13;
        (sign << 16) | (new_exp << 23) | new_fraction
    };

    f32::from_bits(res_bits)
}

#[derive(Debug)]
pub struct ColdVectorStore {
    pub file: std::fs::File,
    pub encoding: VectorEncoding,
    pub record_count: usize,
    pub dim: usize,
}

impl PartialEq for ColdVectorStore {
    fn eq(&self, other: &Self) -> bool {
        self.encoding == other.encoding
            && self.record_count == other.record_count
            && self.dim == other.dim
    }
}

impl ColdVectorStore {
    pub fn write(
        path: impl AsRef<Path>,
        encoding: VectorEncoding,
        dim: usize,
        embeddings: &[f32],
    ) -> io::Result<()> {
        if dim == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "dimension must be greater than zero",
            ));
        }
        let record_count = if embeddings.is_empty() {
            0
        } else {
            embeddings.len() / dim
        };

        let mut payload = Vec::new();
        match encoding {
            VectorEncoding::Raw => {
                for &val in embeddings {
                    payload.extend_from_slice(&val.to_le_bytes());
                }
            }
            VectorEncoding::F16 => {
                for &val in embeddings {
                    let half = f32_to_f16(val);
                    payload.extend_from_slice(&half.to_le_bytes());
                }
            }
            VectorEncoding::SQ8 => {
                for chunk in embeddings.chunks_exact(dim) {
                    let abs_max = chunk.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
                    let scale = if abs_max == 0.0 { 0.0 } else { abs_max / 127.0 };
                    payload.extend_from_slice(&scale.to_le_bytes());
                    for &x in chunk {
                        let q = if scale == 0.0 {
                            0
                        } else {
                            (x / scale).round().clamp(-128.0, 127.0) as i8
                        };
                        payload.push(q as u8);
                    }
                }
            }
        }

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"APMC"); // Magic
        bytes.push(1); // Version
        let enc_byte = match encoding {
            VectorEncoding::Raw => 0,
            VectorEncoding::F16 => 1,
            VectorEncoding::SQ8 => 2,
        };
        bytes.push(enc_byte);
        bytes.extend_from_slice(&(record_count as u32).to_le_bytes());
        bytes.extend_from_slice(&(dim as u32).to_le_bytes());
        bytes.extend_from_slice(&payload);

        let checksum = checksum64(&bytes);
        bytes.extend_from_slice(&checksum.to_le_bytes());

        fs::write(path, bytes)
    }

    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let bytes = fs::read(&path)?;
        if bytes.len() < 14 + 8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "cold vector store file is too short",
            ));
        }
        let (payload, footer) = bytes.split_at(bytes.len() - 8);
        let read_checksum = u64::from_le_bytes(footer.try_into().unwrap());
        let expected_checksum = checksum64(payload);
        if read_checksum != expected_checksum {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "cold vector store checksum mismatch",
            ));
        }

        let magic = &payload[0..4];
        if magic != b"APMC" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid cold vector store magic",
            ));
        }
        let version = payload[4];
        if version != 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported cold vector store version",
            ));
        }
        let enc_byte = payload[5];
        let encoding = match enc_byte {
            0 => VectorEncoding::Raw,
            1 => VectorEncoding::F16,
            2 => VectorEncoding::SQ8,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid cold vector store encoding",
                ))
            }
        };
        let record_count = u32::from_le_bytes(payload[6..10].try_into().unwrap()) as usize;
        let dim = u32::from_le_bytes(payload[10..14].try_into().unwrap()) as usize;

        let file = fs::File::open(path)?;
        Ok(Self {
            file,
            encoding,
            record_count,
            dim,
        })
    }

    pub fn read_vector(&self, local_id: usize) -> io::Result<Vec<f32>> {
        use std::os::unix::fs::FileExt;
        if local_id >= self.record_count {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "local_id out of range",
            ));
        }

        match self.encoding {
            VectorEncoding::Raw => {
                let size = self.dim * 4;
                let offset = 14 + local_id * size;
                let mut buf = vec![0u8; size];
                self.file.read_exact_at(&mut buf, offset as u64)?;
                let mut vec = Vec::with_capacity(self.dim);
                for i in 0..self.dim {
                    let val = f32::from_le_bytes(buf[i * 4..(i + 1) * 4].try_into().unwrap());
                    vec.push(val);
                }
                Ok(vec)
            }
            VectorEncoding::F16 => {
                let size = self.dim * 2;
                let offset = 14 + local_id * size;
                let mut buf = vec![0u8; size];
                self.file.read_exact_at(&mut buf, offset as u64)?;
                let mut vec = Vec::with_capacity(self.dim);
                for i in 0..self.dim {
                    let val = u16::from_le_bytes(buf[i * 2..(i + 1) * 2].try_into().unwrap());
                    vec.push(f16_to_f32(val));
                }
                Ok(vec)
            }
            VectorEncoding::SQ8 => {
                let size = 4 + self.dim;
                let offset = 14 + local_id * size;
                let mut buf = vec![0u8; size];
                self.file.read_exact_at(&mut buf, offset as u64)?;
                let scale = f32::from_le_bytes(buf[0..4].try_into().unwrap());
                let mut vec = Vec::with_capacity(self.dim);
                for i in 0..self.dim {
                    let q = buf[4 + i] as i8;
                    vec.push(q as f32 * scale);
                }
                Ok(vec)
            }
        }
    }

    pub fn read_vectors(&self, local_ids: &[u32]) -> io::Result<(Vec<Vec<f32>>, usize)> {
        let bytes_per_vector = match self.encoding {
            VectorEncoding::Raw => self.dim * 4,
            VectorEncoding::F16 => self.dim * 2,
            VectorEncoding::SQ8 => 4 + self.dim,
        };
        let mut results = Vec::with_capacity(local_ids.len());
        for &id in local_ids {
            results.push(self.read_vector(id as usize)?);
        }
        let total_bytes_read = local_ids.len() * bytes_per_vector;
        Ok((results, total_bytes_read))
    }
}

pub struct Collection {
    pub name: String,
    pub space: MemorySpace,
}

impl Collection {
    pub fn open(collection_dir: impl AsRef<Path>, name: String) -> io::Result<Self> {
        let dir = collection_dir.as_ref();
        fs::create_dir_all(dir)?;
        let manifest_path = dir.join("main.apmf");
        if !manifest_path.exists() {
            let manifest = MemoryManifestFile::new(None, 0, Vec::new());
            manifest.write(&manifest_path)?;
        }
        let space = MemorySpace::open(&manifest_path)?;
        Ok(Self { name, space })
    }

    pub fn insert(&mut self, record: MemoryRecordInput) -> io::Result<()> {
        self.space.insert(record)
    }

    pub fn delete(&mut self, record_id: u64) -> io::Result<()> {
        self.space.delete(record_id)
    }

    pub fn recall(&self, query: &RecallQuery) -> Result<MemorySpaceRecallResult, String> {
        self.space.recall(query)
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.space.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn recalls_with_symbol_scope_and_semantic_rerank() {
        let segment = MemorySegment::build(
            7,
            3,
            vec![
                record(
                    1,
                    10,
                    100,
                    "prefix8 failed at K10000",
                    [1.0, 0.0, 0.0],
                    &["T-173", "prefix8"],
                ),
                record(
                    2,
                    10,
                    110,
                    "uint16 dense fallback is stable",
                    [0.0, 1.0, 0.0],
                    &["T-172", "uint16"],
                ),
                record(
                    3,
                    11,
                    120,
                    "unrelated other project note",
                    [1.0, 0.0, 0.0],
                    &["T-173"],
                ),
            ],
        )
        .unwrap();

        let result = segment
            .recall(&RecallQuery {
                embedding: Some(vec![1.0, 0.1, 0.0]),
                symbols: vec!["prefix8".to_string()],
                scope_id: Some(10),
                limit: 3,
                ..RecallQuery::default()
            })
            .unwrap();

        assert_eq!(result.hits.len(), 1);
        assert_eq!(result.hits[0].record_id, 1);
        assert_eq!(
            result.trace.access_paths,
            vec!["column_filters", "symbol_postings", "semantic_rerank"]
        );
        assert_eq!(result.trace.candidates_after_filters, 2);
        assert_eq!(result.trace.candidates_after_symbols, 1);
    }

    #[test]
    fn manifest_models_branchable_memory_views() {
        let base = MemoryManifest {
            manifest_id: 1,
            parent_manifest_id: None,
            branch_id: 42,
            segment_ids: vec![10, 11],
        };
        let branch = MemoryManifest {
            manifest_id: 2,
            parent_manifest_id: Some(base.manifest_id),
            branch_id: 43,
            segment_ids: vec![10, 11, 12],
        };
        assert_eq!(branch.parent_manifest_id, Some(1));
        assert_eq!(branch.segment_ids, vec![10, 11, 12]);
    }

    #[test]
    fn symbol_filter_requires_all_query_symbols() {
        let segment = MemorySegment::build(
            7,
            3,
            vec![
                record(
                    1,
                    10,
                    100,
                    "prefix8 failed at K10000",
                    [1.0, 0.0, 0.0],
                    &["prefix8", "planner-fallback"],
                ),
                record(
                    2,
                    10,
                    110,
                    "prefix8 alone is insufficient",
                    [1.0, 0.0, 0.0],
                    &["prefix8"],
                ),
                record(
                    3,
                    10,
                    120,
                    "fallback alone is insufficient",
                    [1.0, 0.0, 0.0],
                    &["planner-fallback"],
                ),
            ],
        )
        .unwrap();

        let result = segment
            .recall(&RecallQuery {
                symbols: vec!["prefix8".to_string(), "planner-fallback".to_string()],
                scope_id: Some(10),
                limit: 10,
                ..RecallQuery::default()
            })
            .unwrap();

        assert_eq!(result.hits.len(), 1);
        assert_eq!(result.hits[0].record_id, 1);
        assert_eq!(result.trace.candidates_after_symbols, 1);
    }

    #[test]
    fn symbol_score_uses_record_symbol_richness_after_intersection() {
        let segment = MemorySegment::build(
            7,
            3,
            vec![
                record(
                    1,
                    10,
                    100,
                    "required symbols only",
                    [1.0, 0.0, 0.0],
                    &["prefix8", "planner-fallback"],
                ),
                record(
                    2,
                    10,
                    100,
                    "required symbols plus context",
                    [1.0, 0.0, 0.0],
                    &["prefix8", "planner-fallback", "k10000"],
                ),
            ],
        )
        .unwrap();

        let result = segment
            .recall(&RecallQuery {
                symbols: vec!["prefix8".to_string(), "planner-fallback".to_string()],
                scope_id: Some(10),
                limit: 10,
                ..RecallQuery::default()
            })
            .unwrap();

        assert_eq!(
            result
                .hits
                .iter()
                .map(|hit| hit.record_id)
                .collect::<Vec<_>>(),
            vec![2, 1]
        );
        assert!(result.hits[0].score > result.hits[1].score);
    }

    #[test]
    fn manifest_file_round_trip_preserves_ordered_segments() {
        let manifest = MemoryManifestFile::new(
            None,
            42,
            vec![
                MemoryManifestSegment {
                    segment_id: 10,
                    path: PathBuf::from("segment-10.apms"),
                    vector_sidecar: None,
                },
                MemoryManifestSegment {
                    segment_id: 11,
                    path: PathBuf::from("segment-11.apms"),
                    vector_sidecar: None,
                },
            ],
        );
        let path = temp_manifest_path("round-trip");
        manifest.write(&path).unwrap();
        let manifest_bytes = fs::read(&path).unwrap();

        let loaded = MemoryManifestFile::read(&path).unwrap();
        fs::remove_file(&path).unwrap();

        assert_eq!(loaded, manifest);
        assert!(manifest_bytes.starts_with(b"{\n"));
        assert!(String::from_utf8(manifest_bytes)
            .unwrap()
            .contains("\"version\": 0"));
        assert_eq!(loaded.segments[0].segment_id, 10);
        assert_eq!(loaded.segments[1].segment_id, 11);
    }

    #[test]
    fn manifest_file_rejects_unsupported_version() {
        let mut manifest = MemoryManifestFile::new(
            None,
            42,
            vec![MemoryManifestSegment {
                segment_id: 10,
                path: PathBuf::from("segment-10.apms"),
                vector_sidecar: None,
            }],
        );
        manifest.version = 999;
        let path = temp_manifest_path("bad-version");
        fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();

        let error = MemoryManifestFile::read(&path).unwrap_err();
        fs::remove_file(&path).unwrap();

        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("version"));
    }

    #[test]
    fn manifest_file_round_trip_preserves_mixed_vector_sidecar_refs() {
        let without_sidecars = MemoryManifestFile::new(
            None,
            42,
            vec![
                MemoryManifestSegment {
                    segment_id: 10,
                    path: PathBuf::from("segment-10.apms"),
                    vector_sidecar: None,
                },
                MemoryManifestSegment {
                    segment_id: 11,
                    path: PathBuf::from("segment-11.apms"),
                    vector_sidecar: None,
                },
            ],
        );
        let with_sidecar = MemoryManifestFile::new(
            None,
            42,
            vec![
                MemoryManifestSegment {
                    segment_id: 10,
                    path: PathBuf::from("segment-10.apms"),
                    vector_sidecar: Some(sample_sidecar_ref("segment-10.apmv", false)),
                },
                MemoryManifestSegment {
                    segment_id: 11,
                    path: PathBuf::from("segment-11.apms"),
                    vector_sidecar: None,
                },
            ],
        );
        let path = temp_manifest_path("sidecar-round-trip");
        with_sidecar.write(&path).unwrap();

        let loaded = MemoryManifestFile::read(&path).unwrap();
        let manifest_bytes = fs::read(&path).unwrap();
        fs::remove_file(&path).unwrap();

        assert_eq!(loaded, with_sidecar);
        assert_eq!(with_sidecar.manifest_id, without_sidecars.manifest_id);
        assert!(String::from_utf8(manifest_bytes)
            .unwrap()
            .contains("\"vector_sidecar\""));
        assert!(loaded.segments[0].vector_sidecar.is_some());
        assert!(loaded.segments[1].vector_sidecar.is_none());
    }

    #[test]
    fn memory_space_opens_with_optional_missing_vector_sidecar() {
        let segment = sample_segment();
        let segment_path = temp_segment_path("optional-sidecar-segment");
        let manifest_path = temp_manifest_path("optional-sidecar-manifest");
        segment.write(&segment_path).unwrap();
        MemoryManifestFile::new(
            None,
            42,
            vec![MemoryManifestSegment {
                segment_id: segment.segment_id,
                path: segment_path.clone(),
                vector_sidecar: Some(sample_sidecar_ref("missing.apmv", false)),
            }],
        )
        .write(&manifest_path)
        .unwrap();

        let space = MemorySpace::open(&manifest_path).unwrap();

        fs::remove_file(&segment_path).unwrap();
        fs::remove_file(&manifest_path).unwrap();

        assert_eq!(space.segments.len(), 1);
        assert_eq!(space.segments[0].segment.segment_id, segment.segment_id);
    }

    #[test]
    fn memory_space_rejects_required_missing_vector_sidecar() {
        let segment = sample_segment();
        let segment_path = temp_segment_path("required-sidecar-segment");
        let manifest_path = temp_manifest_path("required-sidecar-manifest");
        segment.write(&segment_path).unwrap();
        MemoryManifestFile::new(
            None,
            42,
            vec![MemoryManifestSegment {
                segment_id: segment.segment_id,
                path: segment_path.clone(),
                vector_sidecar: Some(sample_sidecar_ref("missing.apmv", true)),
            }],
        )
        .write(&manifest_path)
        .unwrap();

        let error = MemorySpace::open(&manifest_path).unwrap_err();

        fs::remove_file(&segment_path).unwrap();
        fs::remove_file(&manifest_path).unwrap();

        assert_eq!(error.kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn memory_space_rejects_required_unsupported_vector_sidecar_version() {
        let segment = sample_segment();
        let segment_path = temp_segment_path("bad-sidecar-version-segment");
        let sidecar_path = temp_sidecar_path("bad-version");
        let manifest_path = temp_manifest_path("bad-sidecar-version-manifest");
        segment.write(&segment_path).unwrap();
        MemoryVectorSidecarFile::for_segment_file(&segment_path, &segment, "array_like", 0)
            .unwrap()
            .write(&sidecar_path)
            .unwrap();
        let mut sidecar_bytes = fs::read(&sidecar_path).unwrap();
        sidecar_bytes[4..8].copy_from_slice(&999u32.to_le_bytes());
        rewrite_checksum(&mut sidecar_bytes);
        fs::write(&sidecar_path, sidecar_bytes).unwrap();
        MemoryManifestFile::new(
            None,
            42,
            vec![MemoryManifestSegment {
                segment_id: segment.segment_id,
                path: segment_path.clone(),
                vector_sidecar: Some(MemoryVectorSidecarRef {
                    path: sidecar_path.clone(),
                    required: true,
                    expected_sidecar_version: Some(VECTOR_SIDECAR_VERSION),
                    expected_generator_name: Some("array_like".to_string()),
                    expected_generator_version: Some(0),
                    sidecar_checksum: None,
                    segment_fingerprint: None,
                }),
            }],
        )
        .write(&manifest_path)
        .unwrap();

        let error = MemorySpace::open(&manifest_path).unwrap_err();

        fs::remove_file(&segment_path).unwrap();
        fs::remove_file(&sidecar_path).unwrap();
        fs::remove_file(&manifest_path).unwrap();

        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("version"));
    }

    #[test]
    fn memory_space_resolves_vector_sidecar_relative_to_manifest_dir() {
        let segment = sample_segment();
        let base_dir = temp_space_dir("relative-sidecar");
        fs::create_dir_all(&base_dir).unwrap();
        let segment_path = base_dir.join("segment.apms");
        let sidecar_path = base_dir.join("segment.apmv");
        let manifest_path = base_dir.join("main.apmf");
        segment.write(&segment_path).unwrap();
        let sidecar =
            MemoryVectorSidecarFile::for_segment_file(&segment_path, &segment, "pivot_prefix", 3)
                .unwrap();
        sidecar.write(&sidecar_path).unwrap();
        let sidecar = MemoryVectorSidecarFile::read(&sidecar_path).unwrap();
        MemoryManifestFile::new(
            None,
            42,
            vec![MemoryManifestSegment {
                segment_id: segment.segment_id,
                path: PathBuf::from("segment.apms"),
                vector_sidecar: Some(MemoryVectorSidecarRef {
                    path: PathBuf::from("segment.apmv"),
                    required: true,
                    expected_sidecar_version: Some(VECTOR_SIDECAR_VERSION),
                    expected_generator_name: Some("pivot_prefix".to_string()),
                    expected_generator_version: Some(3),
                    sidecar_checksum: Some(sidecar.checksum),
                    segment_fingerprint: Some(sidecar.segment_fingerprint),
                }),
            }],
        )
        .write(&manifest_path)
        .unwrap();

        let space = MemorySpace::open(&manifest_path).unwrap();

        fs::remove_dir_all(&base_dir).unwrap();

        assert_eq!(space.segments.len(), 1);
        assert_eq!(space.segments[0].segment.segment_id, segment.segment_id);
    }

    #[test]
    fn memory_space_recalls_across_segments_with_deterministic_trace() {
        let segment_a = MemorySegment::build(
            10,
            3,
            vec![record(
                1,
                10,
                100,
                "manifest recall found base note",
                [1.0, 0.0, 0.0],
                &["T-177", "manifest"],
            )],
        )
        .unwrap();
        let segment_b = MemorySegment::build(
            11,
            3,
            vec![record(
                4,
                10,
                130,
                "manifest recall found branch note",
                [0.9, 0.0, 0.0],
                &["T-177", "branch"],
            )],
        )
        .unwrap();
        let segment_a_path = temp_segment_path("space-a");
        let segment_b_path = temp_segment_path("space-b");
        let manifest_path = temp_manifest_path("space");
        segment_a.write(&segment_a_path).unwrap();
        segment_b.write(&segment_b_path).unwrap();
        MemoryManifestFile::new(
            None,
            42,
            vec![
                MemoryManifestSegment {
                    segment_id: 10,
                    path: segment_a_path.clone(),
                    vector_sidecar: None,
                },
                MemoryManifestSegment {
                    segment_id: 11,
                    path: segment_b_path.clone(),
                    vector_sidecar: None,
                },
            ],
        )
        .write(&manifest_path)
        .unwrap();

        let space = MemorySpace::open(&manifest_path).unwrap();
        let query = RecallQuery {
            embedding: Some(vec![1.0, 0.0, 0.0]),
            symbols: vec!["t-177".to_string()],
            scope_id: Some(10),
            limit: 10,
            ..RecallQuery::default()
        };
        let first = space.recall(&query).unwrap();
        let second = space.recall(&query).unwrap();

        fs::remove_file(&segment_a_path).unwrap();
        fs::remove_file(&segment_b_path).unwrap();
        fs::remove_file(&manifest_path).unwrap();

        assert_eq!(first, second);
        assert_eq!(
            first
                .hits
                .iter()
                .map(|hit| hit.record_id)
                .collect::<Vec<_>>(),
            vec![1, 4]
        );
        assert_eq!(first.trace.segments_considered, 2);
        assert_eq!(first.trace.segments_scanned, 2);
        assert_eq!(first.trace.segments_pruned, 0);
        assert_eq!(first.trace.semantic_evals, 2);
        assert_eq!(first.trace.returned, 2);
        assert_eq!(first.trace.segment_traces.len(), 2);
        assert_eq!(first.trace.segment_traces[0].segment_id, 10);
        assert_eq!(first.trace.segment_traces[1].segment_id, 11);
    }

    #[test]
    fn memory_space_prunes_segments_by_scope_and_time_stats() {
        let segment = sample_segment();
        let segment_path = temp_segment_path("prune");
        let manifest_path = temp_manifest_path("prune");
        segment.write(&segment_path).unwrap();
        MemoryManifestFile::new(
            None,
            42,
            vec![MemoryManifestSegment {
                segment_id: segment.segment_id,
                path: segment_path.clone(),
                vector_sidecar: None,
            }],
        )
        .write(&manifest_path)
        .unwrap();

        let space = MemorySpace::open(&manifest_path).unwrap();
        let result = space
            .recall(&RecallQuery {
                scope_id: Some(99),
                time_start: Some(1000),
                limit: 5,
                ..RecallQuery::default()
            })
            .unwrap();

        fs::remove_file(&segment_path).unwrap();
        fs::remove_file(&manifest_path).unwrap();

        assert!(result.hits.is_empty());
        assert_eq!(result.trace.segments_considered, 1);
        assert_eq!(result.trace.segments_scanned, 0);
        assert_eq!(result.trace.segments_pruned, 1);
        assert_eq!(
            result.trace.segment_traces[0].prune_reason,
            Some("scope_range")
        );
    }

    #[test]
    fn memory_space_fork_writes_child_manifest_without_mutating_parent() {
        let segment = sample_segment();
        let segment_path = temp_segment_path("fork-segment");
        let parent_path = temp_manifest_path("fork-parent");
        let child_path = temp_manifest_path("fork-child");
        segment.write(&segment_path).unwrap();
        let parent = MemoryManifestFile::new(
            None,
            42,
            vec![MemoryManifestSegment {
                segment_id: segment.segment_id,
                path: segment_path.clone(),
                vector_sidecar: None,
            }],
        );
        parent.write(&parent_path).unwrap();
        let parent_before = fs::read(&parent_path).unwrap();

        let space = MemorySpace::open(&parent_path).unwrap();
        space.fork("prefix12-exp", &child_path).unwrap();
        let parent_after = fs::read(&parent_path).unwrap();
        let child = MemoryManifestFile::read(&child_path).unwrap();

        fs::remove_file(&segment_path).unwrap();
        fs::remove_file(&parent_path).unwrap();
        fs::remove_file(&child_path).unwrap();

        assert_eq!(parent_before, parent_after);
        assert_eq!(child.parent_manifest_id, Some(parent.manifest_id));
        assert_eq!(child.branch_id, stable_memory_branch_id("prefix12-exp"));
        assert_ne!(child.manifest_id, parent.manifest_id);
        assert_eq!(child.segments, parent.segments);
    }

    #[test]
    fn segment_file_round_trip_preserves_recall_and_layout() {
        let segment = sample_segment();
        let path = temp_segment_path("round-trip");
        segment.write(&path).unwrap();

        let loaded = MemorySegment::read(&path).unwrap();
        fs::remove_file(&path).unwrap();

        assert_eq!(loaded, segment);
        assert_eq!(loaded.dim, 3);
        assert_eq!(loaded.text_offsets, segment.text_offsets);
        assert_eq!(loaded.embeddings.len(), loaded.len() * loaded.dim);
        assert_eq!(
            loaded.symbol_terms,
            vec!["prefix8", "t-172", "t-173", "uint16"]
        );
        assert_eq!(loaded.symbol_offsets, segment.symbol_offsets);
        assert_eq!(loaded.symbol_record_ids, segment.symbol_record_ids);

        let query = RecallQuery {
            embedding: Some(vec![1.0, 0.1, 0.0]),
            symbols: vec!["prefix8".to_string()],
            scope_id: Some(10),
            limit: 3,
            ..RecallQuery::default()
        };
        assert_eq!(
            loaded.recall(&query).unwrap(),
            segment.recall(&query).unwrap()
        );
    }

    #[test]
    fn flat_vector_candidate_generator_preserves_recall_behavior() {
        let segment = sample_segment();
        let query = RecallQuery {
            embedding: Some(vec![1.0, 0.1, 0.0]),
            scope_id: Some(10),
            limit: 3,
            candidate_budget: Some(1),
            ..RecallQuery::default()
        };

        let default_result = segment.recall(&query).unwrap();
        let explicit_flat_result = segment
            .recall_with_vector_candidate_generator(&query, &FlatMemoryVectorCandidateGenerator)
            .unwrap();

        assert_eq!(explicit_flat_result, default_result);
        assert_eq!(default_result.trace.vector_generator, "flat");
        assert_eq!(default_result.trace.candidates_after_symbols, 2);
        assert_eq!(default_result.trace.vector_candidates, 1);
        assert_eq!(default_result.trace.semantic_evals, 1);
    }

    #[test]
    fn recall_consumes_custom_bounded_vector_candidates() {
        let segment = sample_segment();
        let query = RecallQuery {
            embedding: Some(vec![1.0, 0.1, 0.0]),
            scope_id: Some(10),
            limit: 3,
            ..RecallQuery::default()
        };

        let result = segment
            .recall_with_vector_candidate_generator(&query, &FixedCandidates(vec![1]))
            .unwrap();

        assert_eq!(result.trace.vector_generator, "fixed");
        assert_eq!(result.trace.candidates_after_filters, 2);
        assert_eq!(result.trace.candidates_after_symbols, 2);
        assert_eq!(result.trace.vector_candidates, 1);
        assert_eq!(result.trace.semantic_evals, 1);
        assert_eq!(
            result
                .hits
                .iter()
                .map(|hit| hit.record_id)
                .collect::<Vec<_>>(),
            vec![2]
        );
    }

    #[test]
    fn array_like_vector_candidate_generator_bounds_semantic_rerank() {
        let segment = sample_segment();
        let generator = ArrayLikeMemoryVectorCandidateGenerator::build(&segment);
        let query = RecallQuery {
            embedding: Some(vec![1.0, 0.1, 0.0]),
            scope_id: Some(10),
            limit: 3,
            candidate_budget: Some(1),
            ..RecallQuery::default()
        };

        let result = segment
            .recall_with_vector_candidate_generator(&query, &generator)
            .unwrap();

        assert_eq!(generator.index().segment_id(), segment.segment_id);
        assert_eq!(generator.index().len(), segment.len());
        assert_eq!(
            generator.index().vector_index_bytes(),
            segment.len() * std::mem::size_of::<u32>()
                + segment.embeddings.len() * std::mem::size_of::<f32>()
        );
        assert_eq!(result.trace.vector_generator, "array_like");
        assert_eq!(result.trace.candidates_after_filters, 2);
        assert_eq!(result.trace.candidates_after_symbols, 2);
        assert_eq!(result.trace.vector_candidates, 1);
        assert_eq!(result.trace.semantic_evals, 1);
        assert_eq!(
            result
                .hits
                .iter()
                .map(|hit| hit.record_id)
                .collect::<Vec<_>>(),
            vec![1]
        );
    }

    #[test]
    fn array_like_vector_candidate_generator_is_deterministic() {
        let segment = sample_segment();
        let generator = ArrayLikeMemoryVectorCandidateGenerator::build(&segment);
        let query = RecallQuery {
            embedding: Some(vec![1.0, 0.1, 0.0]),
            limit: 3,
            candidate_budget: Some(2),
            ..RecallQuery::default()
        };

        let first = segment
            .recall_with_vector_candidate_generator(&query, &generator)
            .unwrap();
        let second = segment
            .recall_with_vector_candidate_generator(&query, &generator)
            .unwrap();

        assert_eq!(first, second);
        assert_eq!(first.trace.vector_candidates, 2);
        assert_eq!(first.trace.semantic_evals, 2);
    }

    #[test]
    fn array_like_vector_candidate_generator_falls_back_without_embedding() {
        let segment = sample_segment();
        let generator = ArrayLikeMemoryVectorCandidateGenerator::build(&segment);
        let query = RecallQuery {
            scope_id: Some(10),
            limit: 3,
            candidate_budget: Some(1),
            ..RecallQuery::default()
        };

        let array_like = segment
            .recall_with_vector_candidate_generator(&query, &generator)
            .unwrap();
        let flat = segment
            .recall_with_vector_candidate_generator(&query, &FlatMemoryVectorCandidateGenerator)
            .unwrap();

        assert_eq!(array_like.hits, flat.hits);
        assert_eq!(array_like.trace.vector_generator, "array_like");
        assert_eq!(array_like.trace.vector_candidates, 1);
        assert_eq!(array_like.trace.semantic_evals, 1);
    }

    #[test]
    fn array_like_vector_candidate_generator_rejects_wrong_segment() {
        let segment = sample_segment();
        let other = MemorySegment::build(
            8,
            3,
            vec![record(
                9,
                10,
                100,
                "different segment",
                [1.0, 0.0, 0.0],
                &["prefix8"],
            )],
        )
        .unwrap();
        let generator = ArrayLikeMemoryVectorCandidateGenerator::build(&other);

        let error = segment
            .recall_with_vector_candidate_generator(
                &RecallQuery {
                    embedding: Some(vec![1.0, 0.0, 0.0]),
                    limit: 3,
                    candidate_budget: Some(1),
                    ..RecallQuery::default()
                },
                &generator,
            )
            .unwrap_err();

        assert!(error.contains("segment id mismatch"));
    }

    #[test]
    fn pivot_prefix_vector_candidate_generator_routes_before_rerank() {
        let segment = sample_segment();
        let generator = PivotPrefixMemoryVectorCandidateGenerator::build(
            &segment,
            PivotPrefixConfig {
                block_size: 1,
                pivot_count: 2,
                prefix_len: 1,
                top_blocks: 1,
                candidate_pool: 2,
                mode: PrefixScoreMode::Weighted,
                cluster_iters: 2,
            },
        )
        .unwrap();
        let query = RecallQuery {
            embedding: Some(vec![1.0, 0.1, 0.0]),
            scope_id: Some(10),
            limit: 3,
            candidate_budget: Some(1),
            ..RecallQuery::default()
        };

        let result = segment
            .recall_with_vector_candidate_generator(&query, &generator)
            .unwrap();
        let route = result.trace.vector_route.as_ref().unwrap();

        assert_eq!(result.trace.vector_generator, "pivot_prefix");
        assert_eq!(result.trace.candidates_after_filters, 2);
        assert_eq!(result.trace.vector_candidates, 1);
        assert_eq!(result.trace.semantic_evals, 1);
        assert_eq!(result.hits[0].record_id, 1);
        assert!(route.vector_index_bytes > 0);
        assert!(route.posting_entries_touched > 0);
        assert!(route.route_candidates <= 2);
        assert!(route.working_set_bytes > 0);
    }

    #[test]
    fn pivot_prefix_vector_candidate_generator_falls_back_after_empty_intersection() {
        let segment = sample_segment();
        let generator = PivotPrefixMemoryVectorCandidateGenerator::build(
            &segment,
            PivotPrefixConfig {
                block_size: 1,
                pivot_count: 2,
                prefix_len: 1,
                top_blocks: 1,
                candidate_pool: 2,
                mode: PrefixScoreMode::Weighted,
                cluster_iters: 2,
            },
        )
        .unwrap();
        let query = RecallQuery {
            embedding: Some(vec![1.0, 0.0, 0.0]),
            symbols: vec!["uint16".to_string()],
            limit: 3,
            candidate_budget: Some(1),
            ..RecallQuery::default()
        };

        let result = segment
            .recall_with_vector_candidate_generator(&query, &generator)
            .unwrap();

        assert_eq!(result.trace.candidates_after_symbols, 1);
        assert_eq!(result.trace.vector_candidates, 1);
        assert_eq!(result.trace.semantic_evals, 1);
        assert_eq!(result.hits[0].record_id, 2);
        assert!(result.trace.vector_route.unwrap().fallback_used);
    }

    #[test]
    fn htla_vector_candidate_generator_routes_deterministically() {
        let segment = htla_sample_segment();
        let generator = HtlaMemoryVectorCandidateGenerator::build(
            &segment,
            HtlaMemoryVectorConfig {
                levels: 3,
                chart_dim: 2,
                beam: 4,
                candidate_pool: 8,
                final_nprobe: 4,
                fallback_on_route_risk: false,
            },
        )
        .unwrap();
        let query = RecallQuery {
            embedding: Some(vec![10.1, 2.0, 0.0, 1.0]),
            limit: 4,
            candidate_budget: Some(8),
            ..RecallQuery::default()
        };

        let first = segment
            .recall_with_vector_candidate_generator(&query, &generator)
            .unwrap();
        let second = segment
            .recall_with_vector_candidate_generator(&query, &generator)
            .unwrap();
        let route = first.trace.vector_route.as_ref().unwrap();

        assert_eq!(first, second);
        assert_eq!(first.trace.vector_generator, "htla_tangent");
        assert!(first.trace.vector_candidates <= 8);
        assert!(route.vector_index_bytes > 0);
        assert!(route.working_set_bytes > 0);
        assert!(route.route_candidates <= 8);
        assert!(first
            .hits
            .iter()
            .any(|hit| hit.record_id == 10 || hit.record_id == 11));
    }

    #[test]
    fn htla_vector_candidate_generator_falls_back_on_route_risk() {
        let segment = htla_sample_segment();
        let generator = HtlaMemoryVectorCandidateGenerator::build(
            &segment,
            HtlaMemoryVectorConfig {
                levels: 3,
                chart_dim: 2,
                beam: 1,
                candidate_pool: 1,
                final_nprobe: 4,
                fallback_on_route_risk: true,
            },
        )
        .unwrap();
        let query = RecallQuery {
            embedding: Some(vec![10.1, 2.0, 0.0, 1.0]),
            limit: 4,
            candidate_budget: Some(4),
            ..RecallQuery::default()
        };

        let result = segment
            .recall_with_vector_candidate_generator(&query, &generator)
            .unwrap();

        assert_eq!(result.trace.vector_generator, "htla_tangent");
        assert!(result.trace.vector_route.unwrap().fallback_used);
        assert_eq!(result.trace.vector_candidates, 4);
        assert!(result
            .hits
            .iter()
            .any(|hit| hit.record_id == 10 || hit.record_id == 11));
    }

    #[test]
    fn query_planner_uses_direct_rerank_for_metadata_selective_query() {
        let segment = sample_segment();
        let planner = MemoryQueryPlanner::build_default(&segment).unwrap();
        let query = RecallQuery {
            embedding: Some(vec![1.0, 0.1, 0.0]),
            scope_id: Some(10),
            limit: 3,
            candidate_budget: Some(1),
            ..RecallQuery::default()
        };

        let result = segment
            .recall_with_vector_candidate_generator(&query, &planner)
            .unwrap();
        let plan = result.trace.planner.as_ref().unwrap();

        assert_eq!(result.trace.vector_generator, "query_planner");
        assert_eq!(plan.selected_path, "direct_rerank");
        assert_eq!(plan.candidates_after_symbols, 2);
        assert_eq!(plan.final_candidates, 2);
        assert_eq!(plan.candidate_budget, 1);
        assert_eq!(plan.fallback_reason, None);
        assert_eq!(result.trace.semantic_evals, 2);
    }

    #[test]
    fn query_planner_uses_direct_rerank_for_symbol_selective_query() {
        let segment = sample_segment();
        let planner = MemoryQueryPlanner::build_default(&segment).unwrap();
        let query = RecallQuery {
            embedding: Some(vec![0.0, 1.0, 0.0]),
            symbols: vec!["uint16".to_string()],
            limit: 3,
            candidate_budget: Some(1),
            ..RecallQuery::default()
        };

        let result = segment
            .recall_with_vector_candidate_generator(&query, &planner)
            .unwrap();
        let plan = result.trace.planner.as_ref().unwrap();

        assert_eq!(plan.selected_path, "direct_rerank");
        assert_eq!(plan.candidates_after_symbols, 1);
        assert_eq!(plan.final_candidates, 1);
        assert_eq!(result.hits[0].record_id, 2);
    }

    #[test]
    fn query_planner_uses_pivot_prefix_for_broad_semantic_query() {
        let segment = broad_planner_segment();
        let planner = MemoryQueryPlanner::build(
            &segment,
            MemoryQueryPlannerConfig {
                direct_candidate_threshold: 4,
                vector_candidate_budget: 8,
                fallback_budget_multiplier: 2,
                pivot_min_candidates: 8,
                htla_enabled: false,
                htla_min_candidates: 128,
            },
        )
        .unwrap();
        let query = RecallQuery {
            embedding: Some(vec![30.1, 0.0, 0.0]),
            limit: 4,
            candidate_budget: Some(8),
            ..RecallQuery::default()
        };

        let first = segment
            .recall_with_vector_candidate_generator(&query, &planner)
            .unwrap();
        let second = segment
            .recall_with_vector_candidate_generator(&query, &planner)
            .unwrap();
        let plan = first.trace.planner.as_ref().unwrap();

        assert_eq!(first, second);
        assert_eq!(plan.selected_path, "pivot_prefix");
        assert_eq!(plan.candidate_budget, 8);
        assert_eq!(plan.final_candidates, first.trace.vector_candidates);
        assert!(first.trace.vector_route.is_some());
        assert!(first.trace.semantic_evals <= 8);
        assert!(first.hits.iter().any(|hit| hit.record_id == 30));
    }

    #[test]
    fn query_planner_expands_budget_after_route_fallback() {
        let segment = adversarial_planner_segment();
        let planner = MemoryQueryPlanner::build(
            &segment,
            MemoryQueryPlannerConfig {
                direct_candidate_threshold: 0,
                vector_candidate_budget: 1,
                fallback_budget_multiplier: 2,
                pivot_min_candidates: 1,
                htla_enabled: false,
                htla_min_candidates: 128,
            },
        )
        .unwrap();
        let query = RecallQuery {
            embedding: Some(vec![0.0, 0.0, 0.0]),
            symbols: vec!["needle".to_string()],
            limit: 3,
            candidate_budget: Some(1),
            ..RecallQuery::default()
        };

        let result = segment
            .recall_with_vector_candidate_generator(&query, &planner)
            .unwrap();
        let plan = result.trace.planner.as_ref().unwrap();

        assert_eq!(plan.selected_path, "pivot_prefix");
        assert_eq!(plan.fallback_reason, Some("route_fallback"));
        assert_eq!(plan.expanded_candidate_budget, Some(1));
        assert_eq!(plan.final_candidates, 1);
        assert!(result.trace.vector_route.as_ref().unwrap().fallback_used);
        assert_eq!(result.hits[0].record_id, 127);
    }

    #[test]
    fn query_planner_falls_back_deterministically_without_embedding() {
        let segment = sample_segment();
        let planner = MemoryQueryPlanner::build_default(&segment).unwrap();
        let query = RecallQuery {
            scope_id: Some(10),
            limit: 3,
            candidate_budget: Some(1),
            ..RecallQuery::default()
        };

        let result = segment
            .recall_with_vector_candidate_generator(&query, &planner)
            .unwrap();
        let plan = result.trace.planner.as_ref().unwrap();

        assert_eq!(plan.selected_path, "direct_rerank");
        assert_eq!(plan.fallback_reason, Some("missing_embedding"));
        assert_eq!(plan.final_candidates, 1);
        assert_eq!(result.trace.semantic_evals, 1);
    }

    #[test]
    fn recall_rejects_out_of_range_vector_candidates() {
        let segment = sample_segment();
        let error = segment
            .recall_with_vector_candidate_generator(
                &RecallQuery {
                    limit: 3,
                    ..RecallQuery::default()
                },
                &FixedCandidates(vec![segment.len() as u32]),
            )
            .unwrap_err();

        assert!(error.contains("out of range"));
    }

    #[test]
    fn recall_rejects_vector_candidates_outside_upstream_filters() {
        let segment = sample_segment();
        let error = segment
            .recall_with_vector_candidate_generator(
                &RecallQuery {
                    scope_id: Some(10),
                    limit: 3,
                    ..RecallQuery::default()
                },
                &FixedCandidates(vec![2]),
            )
            .unwrap_err();

        assert!(error.contains("upstream filters"));
    }

    #[test]
    fn segment_file_round_trip_accepts_short_symbol_terms() {
        let segment = MemorySegment::build(
            7,
            3,
            vec![record(
                1,
                10,
                100,
                "one byte symbol",
                [1.0, 0.0, 0.0],
                &["x"],
            )],
        )
        .unwrap();
        let path = temp_segment_path("short-symbol");
        segment.write(&path).unwrap();

        let loaded = MemorySegment::read(&path).unwrap();
        fs::remove_file(&path).unwrap();

        assert_eq!(loaded.symbol_terms, vec!["x"]);
        assert_eq!(loaded, segment);
    }

    #[test]
    fn segment_file_rejects_header_sized_file_without_footer_payload() {
        let path = temp_segment_path("too-short");
        fs::write(&path, vec![0; 55]).unwrap();

        let error = MemorySegment::read(&path).unwrap_err();
        fs::remove_file(&path).unwrap();

        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("too short"));
    }

    #[test]
    fn segment_file_rejects_bad_magic() {
        let segment = sample_segment();
        let path = temp_segment_path("bad-magic");
        segment.write(&path).unwrap();

        let mut bytes = fs::read(&path).unwrap();
        bytes[0] = b'X';
        rewrite_checksum(&mut bytes);
        fs::write(&path, bytes).unwrap();

        let error = MemorySegment::read(&path).unwrap_err();
        fs::remove_file(&path).unwrap();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("magic"));
    }

    #[test]
    fn segment_file_rejects_bad_version() {
        let segment = sample_segment();
        let path = temp_segment_path("bad-version");
        segment.write(&path).unwrap();

        let mut bytes = fs::read(&path).unwrap();
        bytes[4..8].copy_from_slice(&999u32.to_le_bytes());
        rewrite_checksum(&mut bytes);
        fs::write(&path, bytes).unwrap();

        let error = MemorySegment::read(&path).unwrap_err();
        fs::remove_file(&path).unwrap();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("version"));
    }

    #[test]
    fn segment_file_rejects_checksum_mismatch() {
        let segment = sample_segment();
        let path = temp_segment_path("bad-checksum");
        segment.write(&path).unwrap();

        let mut bytes = fs::read(&path).unwrap();
        bytes[16] ^= 0xff;
        fs::write(&path, bytes).unwrap();

        let error = MemorySegment::read(&path).unwrap_err();
        fs::remove_file(&path).unwrap();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("checksum"));
    }

    fn sample_segment() -> MemorySegment {
        MemorySegment::build(
            7,
            3,
            vec![
                record(
                    1,
                    10,
                    100,
                    "prefix8 failed at K10000",
                    [1.0, 0.0, 0.0],
                    &["T-173", "prefix8"],
                ),
                record(
                    2,
                    10,
                    110,
                    "uint16 dense fallback is stable",
                    [0.0, 1.0, 0.0],
                    &["T-172", "uint16"],
                ),
                record(
                    3,
                    11,
                    120,
                    "unrelated other project note",
                    [1.0, 0.0, 0.0],
                    &["T-173"],
                ),
            ],
        )
        .unwrap()
    }

    fn htla_sample_segment() -> MemorySegment {
        MemorySegment::build(
            70,
            4,
            (0..64)
                .map(|i| {
                    record_vec(
                        i as u64,
                        1,
                        1_700_000_000 + i as i64,
                        "htla synthetic chart record",
                        vec![i as f32, (i % 8) as f32, 0.0, 1.0],
                        &["htla"],
                    )
                })
                .collect(),
        )
        .unwrap()
    }

    fn broad_planner_segment() -> MemorySegment {
        MemorySegment::build(
            71,
            3,
            (0..48)
                .map(|i| {
                    record(
                        i as u64,
                        1,
                        1_700_010_000 + i as i64,
                        "broad planner synthetic record",
                        [i as f32, 0.0, 0.0],
                        &["broad"],
                    )
                })
                .collect(),
        )
        .unwrap()
    }

    fn adversarial_planner_segment() -> MemorySegment {
        MemorySegment::build(
            72,
            3,
            (0..128)
                .map(|i| {
                    let symbols = if i == 127 {
                        vec!["needle"]
                    } else {
                        vec!["haystack"]
                    };
                    record(
                        i as u64,
                        1,
                        1_700_020_000 + i as i64,
                        "adversarial planner synthetic record",
                        [i as f32, 0.0, 0.0],
                        &symbols,
                    )
                })
                .collect(),
        )
        .unwrap()
    }

    fn temp_segment_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "aperon-memory-segment-{}-{}-{}.apms",
            name,
            std::process::id(),
            nanos
        ))
    }

    fn temp_manifest_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let unique_dir = std::env::temp_dir().join(format!(
            "aperon-memory-test-{}-{}-{}",
            name,
            std::process::id(),
            nanos
        ));
        std::fs::create_dir_all(&unique_dir).unwrap();
        unique_dir.join("main.apmf")
    }

    fn temp_sidecar_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "aperon-memory-sidecar-{}-{}-{}.apmv",
            name,
            std::process::id(),
            nanos
        ))
    }

    fn temp_space_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "aperon-memory-space-{}-{}-{}",
            name,
            std::process::id(),
            nanos
        ))
    }

    fn sample_sidecar_ref(path: impl Into<PathBuf>, required: bool) -> MemoryVectorSidecarRef {
        MemoryVectorSidecarRef {
            path: path.into(),
            required,
            expected_sidecar_version: Some(VECTOR_SIDECAR_VERSION),
            expected_generator_name: Some("array_like".to_string()),
            expected_generator_version: Some(0),
            sidecar_checksum: None,
            segment_fingerprint: None,
        }
    }

    fn rewrite_checksum(bytes: &mut [u8]) {
        let checksum_start = bytes.len() - 8;
        let checksum = checksum64(&bytes[..checksum_start]);
        bytes[checksum_start..].copy_from_slice(&checksum.to_le_bytes());
    }

    struct FixedCandidates(Vec<u32>);

    impl MemoryVectorCandidateGenerator for FixedCandidates {
        fn name(&self) -> &'static str {
            "fixed"
        }

        fn candidates(
            &self,
            _segment: &MemorySegment,
            _query: &RecallQuery,
            _candidates_after_symbols: &[u32],
        ) -> Result<Vec<u32>, String> {
            Ok(self.0.clone())
        }
    }

    fn record(
        record_id: u64,
        scope_id: u32,
        timestamp: i64,
        text: &str,
        embedding: [f32; 3],
        symbols: &[&str],
    ) -> MemoryRecordInput {
        MemoryRecordInput {
            record_id,
            scope_id,
            timestamp,
            source_id: 1,
            confidence: 1.0,
            text: text.to_string(),
            embedding: embedding.to_vec(),
            symbols: symbols.iter().map(|symbol| symbol.to_string()).collect(),
            vector_id: None,
            metadata: std::collections::BTreeMap::new(),
        }
    }

    fn record_vec(
        record_id: u64,
        scope_id: u32,
        timestamp: i64,
        text: &str,
        embedding: Vec<f32>,
        symbols: &[&str],
    ) -> MemoryRecordInput {
        MemoryRecordInput {
            record_id,
            scope_id,
            timestamp,
            source_id: 1,
            confidence: 1.0,
            text: text.to_string(),
            embedding,
            symbols: symbols.iter().map(|symbol| symbol.to_string()).collect(),
            vector_id: None,
            metadata: std::collections::BTreeMap::new(),
        }
    }

    #[test]
    fn test_memtable_operations() {
        let mut memtable = MemTable::new(42);
        assert_eq!(memtable.segment_id, 42);
        assert_eq!(memtable.len(), 0);
        assert!(memtable.records().is_empty());

        let rec1 = record(1, 10, 100, "hello", [1.0, 0.0, 0.0], &["tag1"]);
        let rec2 = record(2, 10, 110, "world", [0.0, 1.0, 0.0], &["tag2"]);

        // Insert rec1
        memtable.insert(rec1.clone());
        assert_eq!(memtable.len(), 1);
        assert_eq!(memtable.get_record(1), Some(&rec1));
        assert_eq!(memtable.get_record(2), None);
        assert!(!memtable.is_deleted(1));

        // Insert rec2
        memtable.insert(rec2.clone());
        assert_eq!(memtable.len(), 2);
        assert_eq!(memtable.get_record(2), Some(&rec2));

        // Overwrite rec1
        let rec1_updated = record(
            1,
            10,
            120,
            "hello updated",
            [1.5, 0.0, 0.0],
            &["tag1", "updated"],
        );
        memtable.insert(rec1_updated.clone());
        assert_eq!(memtable.len(), 2);
        assert_eq!(memtable.get_record(1), Some(&rec1_updated));

        // Delete rec2
        memtable.delete(2);
        assert_eq!(memtable.len(), 1);
        assert!(memtable.is_deleted(2));
        assert_eq!(memtable.get_record(2), None);

        // Re-insert rec2
        let rec2_reinserted = record(2, 10, 130, "world returns", [0.0, 2.0, 0.0], &["tag2"]);
        memtable.insert(rec2_reinserted.clone());
        assert_eq!(memtable.len(), 2);
        assert!(!memtable.is_deleted(2));
        assert_eq!(memtable.get_record(2), Some(&rec2_reinserted));

        // Delete a non-existent record
        memtable.delete(999);
        assert_eq!(memtable.len(), 2); // len shouldn't change
        assert!(memtable.is_deleted(999));
        assert_eq!(memtable.get_record(999), None);

        // Inspect records slice
        let records = memtable.records();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].record_id, 1);
        assert_eq!(records[1].record_id, 2);
    }

    fn temp_wal_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "aperon-memory-wal-{}-{}-{}.apmw",
            name,
            std::process::id(),
            nanos
        ))
    }

    #[test]
    fn test_wal_exclusive_lock() {
        let path = temp_wal_path("exclusive_lock");

        // 1. Open first writer (holds lock)
        let _writer1 = WALWriter::open(&path).unwrap();

        // 2. Open second writer on the same path (should fail because of lock)
        let writer2_res = WALWriter::open(&path);
        assert!(writer2_res.is_err());
        let err = writer2_res.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::AddrInUse);
        assert!(err.to_string().contains("locked"));

        // 3. Drop first writer (releases lock)
        drop(_writer1);

        // 4. Open second writer again (should succeed now)
        let _writer2 = WALWriter::open(&path).unwrap();

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_wal_roundtrip() {
        let path = temp_wal_path("roundtrip");
        let rec1 = record(1, 10, 100, "hello", [1.0, 0.0, 0.0], &["tag1"]);
        let rec2 = record(2, 10, 110, "world", [0.0, 1.0, 0.0], &["tag2"]);

        // Write entries
        {
            let mut writer = WALWriter::open(&path).unwrap();
            writer.append_insert(&rec1).unwrap();
            writer.append_delete(3).unwrap();
            writer.append_insert(&rec2).unwrap();
            writer.sync().unwrap();
        }

        // Read entries
        {
            let mut reader = WALReader::open(&path).unwrap();

            let entry1 = reader.next_entry().unwrap().unwrap();
            assert_eq!(entry1, WALEntry::Insert(rec1));

            let entry2 = reader.next_entry().unwrap().unwrap();
            assert_eq!(entry2, WALEntry::Delete(3));

            let entry3 = reader.next_entry().unwrap().unwrap();
            assert_eq!(entry3, WALEntry::Insert(rec2));

            let entry4 = reader.next_entry().unwrap();
            assert!(entry4.is_none());
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_wal_corrupted_checksum() {
        let path = temp_wal_path("corrupted_checksum");
        let rec = record(1, 10, 100, "hello", [1.0, 0.0, 0.0], &["tag1"]);

        {
            let mut writer = WALWriter::open(&path).unwrap();
            writer.append_insert(&rec).unwrap();
            writer.sync().unwrap();
        }

        // Mutate a byte in the middle of the file
        let mut data = std::fs::read(&path).unwrap();
        if data.len() > 12 {
            data[12] ^= 0xFF; // Corrupt a byte in the payload
            std::fs::write(&path, data).unwrap();
        }

        // Try reading it back
        let mut reader = WALReader::open(&path).unwrap();
        let res = reader.next_entry();
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("checksum mismatch"));

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_wal_partial_write() {
        let path = temp_wal_path("partial_write");
        let rec = record(1, 10, 100, "hello", [1.0, 0.0, 0.0], &["tag1"]);

        {
            let mut writer = WALWriter::open(&path).unwrap();
            writer.append_insert(&rec).unwrap();
            writer.sync().unwrap();
        }

        // Truncate the file to various partial lengths
        let original_data = std::fs::read(&path).unwrap();
        let original_len = original_data.len();

        // Truncate during header (say 5 bytes)
        {
            let partial_data = &original_data[..5];
            std::fs::write(&path, partial_data).unwrap();

            let mut reader = WALReader::open(&path).unwrap();
            let res = reader.next_entry();
            assert!(res.is_err());
            assert_eq!(res.unwrap_err().kind(), io::ErrorKind::UnexpectedEof);
        }

        // Truncate during payload (say 15 bytes, if total size is larger)
        if original_len > 15 {
            let partial_data = &original_data[..15];
            std::fs::write(&path, partial_data).unwrap();

            let mut reader = WALReader::open(&path).unwrap();
            let res = reader.next_entry();
            assert!(res.is_err());
            assert_eq!(res.unwrap_err().kind(), io::ErrorKind::UnexpectedEof);
        }

        // Truncate before checksum is complete (say original_len - 4)
        if original_len > 8 {
            let partial_data = &original_data[..original_len - 4];
            std::fs::write(&path, partial_data).unwrap();

            let mut reader = WALReader::open(&path).unwrap();
            let res = reader.next_entry();
            assert!(res.is_err());
            assert_eq!(res.unwrap_err().kind(), io::ErrorKind::UnexpectedEof);
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_memory_space_crash_recovery() {
        let manifest_path = temp_manifest_path("crash_recovery");
        let dir = manifest_path.parent().unwrap();
        std::fs::create_dir_all(dir).unwrap();

        // 1. Create a minimal valid manifest file
        let manifest = MemoryManifestFile::new(None, 10, Vec::new());
        manifest.write(&manifest_path).unwrap();

        // 2. Open the space (which creates a fresh WAL)
        let mut space = MemorySpace::open(&manifest_path).unwrap();
        assert_eq!(space.memtable.read().unwrap().len(), 0);

        // 3. Insert record 1
        let rec1 = record(1, 10, 100, "hello space", [1.0, 0.0, 0.0], &["crash"]);
        space.insert(rec1.clone()).unwrap();

        // 4. Insert record 2
        let rec2 = record(2, 10, 110, "to delete", [0.0, 1.0, 0.0], &["transient"]);
        space.insert(rec2.clone()).unwrap();

        // 5. Delete record 2
        space.delete(2).unwrap();

        // Check current in-memory state
        {
            let query = RecallQuery {
                limit: 10,
                ..Default::default()
            };
            let result = space.recall(&query).unwrap();
            assert_eq!(result.hits.len(), 1);
            assert_eq!(result.hits[0].record_id, 1);
            assert_eq!(result.hits[0].text, "hello space");
        }

        // 6. Simulate crash by dropping the space
        drop(space);

        // 7. Re-open the space from the same manifest
        let space_recovered = MemorySpace::open(&manifest_path).unwrap();

        // 8. Verify the state was recovered from WAL
        {
            let query = RecallQuery {
                limit: 10,
                ..Default::default()
            };
            let result = space_recovered.recall(&query).unwrap();
            assert_eq!(result.hits.len(), 1);
            assert_eq!(result.hits[0].record_id, 1);
            assert_eq!(result.hits[0].text, "hello space");

            // Verify that record 2 is indeed deleted/not present
            let mem = space_recovered.memtable.read().unwrap();
            assert!(mem.is_deleted(2));
            assert_eq!(mem.get_record(2), None);
        }

        // Clean up
        let wal_path = dir.join("wal_active.apmw");
        std::fs::remove_file(wal_path).ok();
        std::fs::remove_file(manifest_path).ok();
    }

    #[test]
    fn test_auto_flush() {
        let manifest_path = temp_manifest_path("auto_flush");
        let dir = manifest_path.parent().unwrap();
        std::fs::create_dir_all(dir).unwrap();

        // Create initial manifest
        let manifest = MemoryManifestFile::new(None, 10, Vec::new());
        manifest.write(&manifest_path).unwrap();

        let mut space = MemorySpace::open(&manifest_path).unwrap();
        space.max_memtable_records = 2; // Auto-flush threshold

        let rec1 = record(1, 10, 100, "rec1", [1.0, 0.0, 0.0], &["tag"]);
        let rec2 = record(2, 10, 110, "rec2", [0.0, 1.0, 0.0], &["tag"]);
        let rec3 = record(3, 10, 120, "rec3", [0.0, 0.0, 1.0], &["tag"]);

        // Insert first record -> in memtable (len = 1)
        space.insert(rec1.clone()).unwrap();
        assert_eq!(space.memtable.read().unwrap().len(), 1);
        assert!(space.segments.is_empty());

        // Insert second record -> triggers auto-flush because len becomes 2 >= 2
        space.insert(rec2.clone()).unwrap();
        // After flush, memtable is reset to empty, and a new segment is loaded
        assert_eq!(space.memtable.read().unwrap().len(), 0);
        assert_eq!(space.segments.len(), 1);
        assert_eq!(space.segments[0].segment.len(), 2);

        // Insert third record -> in memtable (len = 1)
        space.insert(rec3.clone()).unwrap();
        assert_eq!(space.memtable.read().unwrap().len(), 1);
        assert_eq!(space.segments.len(), 1);

        // Query all records
        let query = RecallQuery {
            limit: 10,
            ..Default::default()
        };
        let result = space.recall(&query).unwrap();
        assert_eq!(result.hits.len(), 3);

        let ids: Vec<u64> = result.hits.iter().map(|h| h.record_id).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
        assert!(ids.contains(&3));

        // Clean up
        for segment_ref in &space.manifest.segments {
            let p = dir.join(&segment_ref.path);
            std::fs::remove_file(p).ok();
        }
        std::fs::remove_file(dir.join("segment_1.apms")).ok();
        std::fs::remove_file(dir.join("segment_2.apms")).ok();
        std::fs::remove_file(dir.join("wal_active.apmw")).ok();
        std::fs::remove_file(manifest_path).ok();
    }

    #[test]
    fn test_flush_compaction_tombstones() {
        let manifest_path = temp_manifest_path("flush_compaction");
        let dir = manifest_path.parent().unwrap();
        std::fs::create_dir_all(dir).unwrap();

        // Create initial manifest
        let manifest = MemoryManifestFile::new(None, 10, Vec::new());
        manifest.write(&manifest_path).unwrap();

        let mut space = MemorySpace::open(&manifest_path).unwrap();
        space.max_memtable_records = 10;

        let rec1 = record(1, 10, 100, "rec1", [1.0, 0.0, 0.0], &["tag"]);
        let rec2 = record(2, 10, 110, "rec2", [0.0, 1.0, 0.0], &["tag"]);
        let rec3 = record(3, 10, 120, "rec3", [0.0, 0.0, 1.0], &["tag"]);

        space.insert(rec1.clone()).unwrap();
        space.insert(rec2.clone()).unwrap();
        space.insert(rec3.clone()).unwrap();

        // Delete record 2 (tombstone added, rec2 removed from memtable records)
        space.delete(2).unwrap();

        // Force a flush
        space.flush().unwrap();

        // Manifest should point to the newly flushed segment (id 1)
        assert_eq!(space.manifest.segments.len(), 1);
        let segment_ref = &space.manifest.segments[0];
        let segment_path = dir.join(&segment_ref.path);

        // Read segment directly from disk
        let segment_on_disk = MemorySegment::read(&segment_path).unwrap();

        // Confirm that rec2 (id 2) is NOT in the record_ids of the segment on disk (physically purged)
        assert_eq!(segment_on_disk.len(), 2);
        assert!(!segment_on_disk.record_ids.contains(&2));
        assert!(segment_on_disk.record_ids.contains(&1));
        assert!(segment_on_disk.record_ids.contains(&3));

        // Clean up
        std::fs::remove_file(segment_path).ok();
        std::fs::remove_file(dir.join("wal_active.apmw")).ok();
        std::fs::remove_file(manifest_path).ok();
    }

    #[test]
    fn test_collection_ops() {
        let manifest_path = temp_manifest_path("collection_ops");
        let collection_dir = manifest_path.parent().unwrap().to_path_buf();
        let name = "test_col".to_string();

        // 1. Open collection
        let mut col = Collection::open(&collection_dir, name).unwrap();
        assert_eq!(col.name, "test_col");

        // 2. Insert records with vector_id and metadata
        let mut metadata = BTreeMap::new();
        metadata.insert("category".to_string(), "sports".to_string());
        metadata.insert("sub".to_string(), "soccer".to_string());

        let rec1 = MemoryRecordInput {
            record_id: 101,
            scope_id: 10,
            timestamp: 1000,
            source_id: 1,
            confidence: 0.9,
            text: "Soccer match today".to_string(),
            embedding: vec![1.0, 0.0, 0.0],
            symbols: vec!["sports".to_string()],
            vector_id: Some("vec_101".to_string()),
            metadata: metadata.clone(),
        };

        let mut metadata2 = BTreeMap::new();
        metadata2.insert("category".to_string(), "sports".to_string());
        metadata2.insert("sub".to_string(), "basketball".to_string());

        let rec2 = MemoryRecordInput {
            record_id: 102,
            scope_id: 10,
            timestamp: 1001,
            source_id: 1,
            confidence: 0.95,
            text: "Basketball game tonight".to_string(),
            embedding: vec![0.0, 1.0, 0.0],
            symbols: vec!["sports".to_string()],
            vector_id: Some("vec_102".to_string()),
            metadata: metadata2,
        };

        col.insert(rec1).unwrap();
        col.insert(rec2).unwrap();

        // 3. Query with vector_id
        let mut query = RecallQuery {
            vector_id: Some("vec_101".to_string()),
            limit: 10,
            ..Default::default()
        };
        let res = col.recall(&query).unwrap();
        assert_eq!(res.hits.len(), 1);
        assert_eq!(res.hits[0].record_id, 101);
        assert_eq!(res.hits[0].vector_id, Some("vec_101".to_string()));
        assert_eq!(res.hits[0].metadata.get("sub").unwrap(), "soccer");

        // 4. Query with metadata filter
        let mut filter = BTreeMap::new();
        filter.insert("category".to_string(), "sports".to_string());
        filter.insert("sub".to_string(), "basketball".to_string());
        query.vector_id = None;
        query.metadata_filter = filter;
        let res = col.recall(&query).unwrap();
        assert_eq!(res.hits.len(), 1);
        assert_eq!(res.hits[0].record_id, 102);

        // 5. Delete and verify
        col.delete(101).unwrap();
        let mut filter = BTreeMap::new();
        filter.insert("category".to_string(), "sports".to_string());
        query.metadata_filter = filter;
        let res = col.recall(&query).unwrap();
        assert_eq!(res.hits.len(), 1);
        assert_eq!(res.hits[0].record_id, 102);

        // 6. Flush collection
        col.flush().unwrap();
    }

    #[test]
    fn test_cold_vector_store_raw_f16_sq8() {
        for encoding in &[
            VectorEncoding::Raw,
            VectorEncoding::F16,
            VectorEncoding::SQ8,
        ] {
            let manifest_path = temp_manifest_path(&format!("cold_{:?}", encoding));
            let dir = manifest_path.parent().unwrap().to_path_buf();

            // Create initial manifest
            let manifest = MemoryManifestFile::new(None, 10, Vec::new());
            manifest.write(&manifest_path).unwrap();

            let mut space = MemorySpace::open(&manifest_path).unwrap();
            space.vector_encoding = Some(*encoding);

            // Insert 3 records with embeddings
            let rec1 = record(1, 10, 100, "rec1", [1.0, 0.0, 0.0], &["tag"]);
            let rec2 = record(2, 10, 110, "rec2", [0.0, 1.0, 0.0], &["tag"]);
            let rec3 = record(3, 10, 120, "rec3", [0.0, 0.0, 1.0], &["tag"]);

            space.insert(rec1).unwrap();
            space.insert(rec2).unwrap();
            space.insert(rec3).unwrap();

            // Flush space -> creates companion cold vector store
            space.flush().unwrap();
            drop(space);

            // Confirm that the .apmc file exists on disk
            let apmc_path = dir.join("segment_1.apmc");
            assert!(apmc_path.exists());

            // Re-open space
            let re_space = MemorySpace::open(&manifest_path).unwrap();
            assert_eq!(re_space.segments.len(), 1);

            let segment = &re_space.segments[0].segment;
            assert!(segment.embeddings.is_empty());
            assert!(segment.cold_store.is_some());

            // Perform recall queries
            let query = RecallQuery {
                embedding: Some(vec![1.0, 0.1, 0.0]),
                limit: 10,
                ..Default::default()
            };

            let res = re_space.recall(&query).unwrap();
            assert_eq!(res.hits.len(), 3);
            // First hit should be record 1
            assert_eq!(res.hits[0].record_id, 1);
            // Verify positive trace metrics
            assert!(res.trace.cold_bytes_read > 0);
            assert!(res.trace.read_amplification > 0.0);

            // Clean up
            for segment_ref in &re_space.manifest.segments {
                let p = dir.join(&segment_ref.path);
                std::fs::remove_file(p).ok();
            }
            std::fs::remove_file(apmc_path).ok();
            std::fs::remove_file(dir.join("wal_active.apmw")).ok();
            std::fs::remove_file(manifest_path).ok();
        }
    }

    #[test]
    fn test_elastic_fallback_cache() {
        let manifest_path = temp_manifest_path("elastic_fallback_cache");
        let dir = manifest_path.parent().unwrap().to_path_buf();

        // Create initial manifest
        let manifest = MemoryManifestFile::new(None, 10, Vec::new());
        manifest.write(&manifest_path).unwrap();

        let mut space = MemorySpace::open(&manifest_path).unwrap();
        // Insert 3 records with embeddings (each has 3 float32 elements -> 12 bytes raw embedding size)
        let rec1 = record(1, 10, 100, "rec1", [1.0, 0.0, 0.0], &["tag"]);
        let rec2 = record(2, 10, 110, "rec2", [0.0, 1.0, 0.0], &["tag"]);
        let rec3 = record(3, 10, 120, "rec3", [0.0, 0.0, 1.0], &["tag"]);

        space.insert(rec1).unwrap();
        space.insert(rec2).unwrap();
        space.insert(rec3).unwrap();

        // Flush space to disk to write a segment
        space.flush().unwrap();

        // Check DRAM resident raw bytes before setting budget
        // Segment has 3 records * 3 dim * 4 bytes = 36 bytes.
        // MemTable is empty, so 0 bytes.
        // Total = 36 bytes.
        assert_eq!(space.resident_raw_dram_bytes(), 36);
        assert!(!space.segments[0].segment.embeddings.is_empty());

        // 1. Setting raw_dram_budget triggers offload on insert when budget is breached
        space.raw_dram_budget = Some(20);
        let rec4 = record(4, 10, 130, "rec4", [1.0, 1.0, 1.0], &["tag"]);
        space.insert(rec4).unwrap();

        // After insert, raw dram bytes limit (20) was breached.
        // The segment should have been offloaded to companion .apmc file.
        // The resident bytes now should only be the new record in memtable: 1 record * 3 dim * 4 = 12 bytes.
        assert_eq!(space.resident_raw_dram_bytes(), 12);
        assert!(space.segments[0].segment.embeddings.is_empty());
        assert!(space.segments[0].segment.cold_store.is_some());
        let apmc_path = dir.join("segment_1.apmc");
        assert!(apmc_path.exists());

        // Drop space to release lock on WAL
        drop(space);

        // 2. Open enforces budget
        let mut space2 = MemorySpace::open(&manifest_path).unwrap();
        // At this point, the segment is loaded with embeddings because they exist in the segment file.
        // We set the budget and enforce it, simulating startup enforcement when a budget is configured.
        space2.raw_dram_budget = Some(20);
        space2.enforce_dram_budget().unwrap();
        assert!(space2.segments[0].segment.embeddings.is_empty());
        assert!(space2.segments[0].segment.cold_store.is_some());

        // 3. fallback_to_recon_on_cold query bypasses cold store reads and registers "fallback_to_recon"
        // and cold_bytes_read = 0 in the trace.
        let query_normal = RecallQuery {
            embedding: Some(vec![1.0, 0.0, 0.0]),
            fallback_to_recon_on_cold: false,
            limit: 10,
            ..Default::default()
        };
        let res_normal = space2.recall(&query_normal).unwrap();
        let segment_trace_normal = res_normal.trace.segment_traces[0].trace.as_ref().unwrap();
        assert!(!segment_trace_normal
            .access_paths
            .contains(&"fallback_to_recon"));
        assert!(segment_trace_normal.cold_bytes_read > 0);

        let query_fallback = RecallQuery {
            embedding: Some(vec![1.0, 0.0, 0.0]),
            fallback_to_recon_on_cold: true,
            limit: 10,
            ..Default::default()
        };
        let res_fallback = space2.recall(&query_fallback).unwrap();
        let segment_trace_fallback = res_fallback.trace.segment_traces[0].trace.as_ref().unwrap();
        assert!(segment_trace_fallback
            .access_paths
            .contains(&"fallback_to_recon"));
        assert_eq!(segment_trace_fallback.cold_bytes_read, 0);
        assert_eq!(segment_trace_fallback.read_amplification, 0.0);

        // Clean up
        let segments_to_cleanup = space2.manifest.segments.clone();
        drop(space2);

        for segment_ref in &segments_to_cleanup {
            let p = dir.join(&segment_ref.path);
            std::fs::remove_file(p).ok();
        }
        std::fs::remove_file(apmc_path).ok();
        std::fs::remove_file(dir.join("wal_active.apmw")).ok();
        std::fs::remove_file(manifest_path).ok();
    }
}
