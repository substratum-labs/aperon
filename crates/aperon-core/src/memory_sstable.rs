use crate::distance::l2_squared_unchecked;
use crate::pivot_prefix::{PivotPrefixConfig, PivotPrefixRouter, PrefixScoreMode, RouteMetrics};
use crate::routing::HtlaRouter;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

const SEGMENT_MAGIC: &[u8; 4] = b"APMS";
const SEGMENT_VERSION: u32 = 0;
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
}

#[derive(Clone, Debug, PartialEq)]
pub struct MemorySpaceRecallResult {
    pub hits: Vec<MemoryHit>,
    pub trace: MemorySpaceRecallTrace,
}

/// An open memory snapshot that manages a set of loaded `MemorySegment`s and resolves
/// global queries across them. Handles relative path resolution and sidecar validation.
#[derive(Clone, Debug, PartialEq)]
pub struct MemorySpace {
    pub manifest: MemoryManifestFile,
    pub segments: Vec<LoadedMemorySegment>,
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

        text_offsets.push(0);
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

        let checksum = checksum64(&bytes);
        write_u64(&mut bytes, checksum);
        fs::write(path, bytes)
    }

    pub fn read(path: impl AsRef<Path>) -> io::Result<Self> {
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
        if version != SEGMENT_VERSION {
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
        reader.expect_end()?;

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

        let mut scored = Vec::with_capacity(candidates.len());
        if query.embedding.is_some() {
            access_paths.push("semantic_rerank");
        }
        let query_embedding = query.embedding.as_deref();
        for local_id in candidates {
            let local_id = local_id as usize;
            let semantic_distance = query_embedding.map(|embedding| {
                l2_squared_unchecked(embedding, self.embedding_row(local_id)).sqrt()
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
                },
            )
            .collect::<Vec<_>>();

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
            },
            hits,
        })
    }

    pub fn text(&self, local_id: usize) -> &str {
        let start = self.text_offsets[local_id] as usize;
        let end = self.text_offsets[local_id + 1] as usize;
        std::str::from_utf8(&self.text_bytes[start..end]).unwrap_or("")
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
        if self.embeddings.len() != record_count * self.dim {
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
        Ok(Self { manifest, segments })
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
                merged.push((loaded.segment.segment_id, hit));
            }
            segment_traces.push(MemorySpaceSegmentTrace {
                segment_id: loaded.segment.segment_id,
                pruned: false,
                prune_reason: None,
                trace: Some(result.trace),
            });
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
        std::env::temp_dir().join(format!(
            "aperon-memory-manifest-{}-{}-{}.apmf",
            name,
            std::process::id(),
            nanos
        ))
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
        }
    }
}
