# Rust API Reference

Aperon's core Memory SSTable engine is exposed to Rust embedders through the `aperon-core` crate.

---

## 1. `MemorySpace`

Represents an active, queryable semantic space loaded from a manifest log.

### Associated Methods
* **`open(manifest_path: impl AsRef<Path>) -> io::Result<Self>`**  
  Loads a versioned `.apmf` manifest file, resolves relative paths, loads and validates segment IDs, and runs strict sidecar index validation checks.
* **`recall(&self, query: &RecallQuery) -> Result<MemorySpaceRecallResult, String>`**  
  Performs queries across all loaded segments and merges the resulting candidate lists deterministically.

---

## 2. `MemoryQueryPlanner`

A 5-layer query planner that routes queries through direct metadata scans, flat vector scans, array-like indexes, pivot-prefix indexes, or HTLA routing pipelines.

### Associated Methods
* **`build(segment: &MemorySegment, config: MemoryQueryPlannerConfig) -> Result<Self, String>`**  
  Builds a query planner instance for a specific segment.
* **`build_default(segment: &MemorySegment) -> Result<Self, String>`**  
  Builds a query planner instance for a specific segment with default configurations.

### Trait Implementation (`MemoryVectorCandidateGenerator` for `MemoryQueryPlanner`)
* **`candidates(&self, segment: &MemorySegment, query: &RecallQuery, candidates_after_symbols: &[u32]) -> Result<Vec<u32>, String>`**  
  Routes the query across active segment access paths and generates matching candidates.


### Config Options (`MemoryQueryPlannerConfig`)
- `direct_candidate_threshold`: Scans metadata directly if candidates are below this limit (bypassing vector indexes).
- `vector_candidate_budget`: Maximum vector candidates requested from indexing paths.
- `fallback_budget_multiplier`: Multiplies the candidate budget when index fallbacks are triggered.
- `pivot_min_candidates`: Minimum candidates needed to qualify for Pivot-Prefix routing.
- `htla_enabled`: Boolean flag enabling HTLA tangent routing.
- `htla_min_candidates`: Minimum candidates needed to qualify for HTLA routing.

---

## 3. `MemorySegment`

An immutable columnar chunk representing a compiled partition of records.

### Associated Methods
* **`build(dim: usize, segment_id: u64, records: &[MemoryRecordInput]) -> Result<Self, String>`**  
  Compiles a list of raw memory records into the columnar format.
* **`write(&self, path: impl AsRef<Path>) -> io::Result<()>`**  
  Serializes the columnar segment to disk.
* **`read(path: impl AsRef<Path>) -> io::Result<Self>`**  
  Loads a serialized segment from disk.

---

## 4. `RecallQuery`

The query payload structure used to filter and scan vector memory.

### Struct Fields
- `embedding: Option<Vec<f32>>`: Optional query vector embedding for semantic search.
- `symbols: Vec<String>`: Tokenized metadata keywords that must match.
- `scope_id: Option<u32>`: Optional scope identifier.
- `time_start: Option<i64>`: Optional timestamp range start.
- `time_end: Option<i64>`: Optional timestamp range end.
- `min_confidence: Option<f32>`: Minimum threshold filter.
- `limit: usize`: Top-K hits requested.
- `candidate_budget: Option<usize>`: Custom candidate generator limit override.
