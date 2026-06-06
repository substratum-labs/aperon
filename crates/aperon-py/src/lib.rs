use aperon_core::{
    binary::{load_legacy_index, write_legacy_index},
    stable_memory_branch_id, AperonIndex, HierarchicalLatticeLayerConfig,
    HierarchicalLatticeRouter, HtlaRouter, MemoryHit, MemoryManifestFile, MemoryManifestSegment,
    MemoryQueryPlannerTrace, MemoryRecordInput, MemorySegment, MemorySpace, MemorySpaceRecallTrace,
    MemorySpaceSegmentTrace, MemoryVectorRouteTrace, RecallQuery, RecallTrace, VectorId,
};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};
use std::{fs::File, io::BufWriter, path::PathBuf};

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[pyclass(name = "AperonIndex")]
struct PyAperonIndex {
    inner: AperonIndex,
}

#[pyclass(name = "HlrRouter")]
struct PyHlrRouter {
    inner: HierarchicalLatticeRouter,
}

#[pyclass(name = "HtlaRouter")]
struct PyHtlaRouter {
    inner: HtlaRouter,
}

#[pyclass(name = "RecallQuery")]
#[derive(Clone)]
struct PyRecallQuery {
    inner: RecallQuery,
}

#[pyclass(name = "MemorySegment")]
#[derive(Clone)]
struct PyMemorySegment {
    inner: MemorySegment,
}

#[pyclass(name = "MemoryManifestFile")]
#[derive(Clone)]
struct PyMemoryManifestFile {
    inner: MemoryManifestFile,
}

#[pyclass(name = "MemorySpace")]
#[derive(Clone)]
struct PyMemorySpace {
    inner: MemorySpace,
}

#[pymethods]
impl PyRecallQuery {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        embedding=None,
        symbols=Vec::new(),
        scope_id=None,
        time_start=None,
        time_end=None,
        min_confidence=None,
        limit=10,
        candidate_budget=None
    ))]
    fn new(
        embedding: Option<Vec<f32>>,
        symbols: Vec<String>,
        scope_id: Option<u32>,
        time_start: Option<i64>,
        time_end: Option<i64>,
        min_confidence: Option<f32>,
        limit: usize,
        candidate_budget: Option<usize>,
    ) -> Self {
        Self {
            inner: RecallQuery {
                embedding,
                symbols,
                scope_id,
                time_start,
                time_end,
                min_confidence,
                limit,
                candidate_budget,
            },
        }
    }

    #[getter]
    fn embedding(&self) -> Option<Vec<f32>> {
        self.inner.embedding.clone()
    }

    #[setter]
    fn set_embedding(&mut self, embedding: Option<Vec<f32>>) {
        self.inner.embedding = embedding;
    }

    #[getter]
    fn symbols(&self) -> Vec<String> {
        self.inner.symbols.clone()
    }

    #[setter]
    fn set_symbols(&mut self, symbols: Vec<String>) {
        self.inner.symbols = symbols;
    }

    #[getter]
    fn scope_id(&self) -> Option<u32> {
        self.inner.scope_id
    }

    #[setter]
    fn set_scope_id(&mut self, scope_id: Option<u32>) {
        self.inner.scope_id = scope_id;
    }

    #[getter]
    fn time_start(&self) -> Option<i64> {
        self.inner.time_start
    }

    #[setter]
    fn set_time_start(&mut self, time_start: Option<i64>) {
        self.inner.time_start = time_start;
    }

    #[getter]
    fn time_end(&self) -> Option<i64> {
        self.inner.time_end
    }

    #[setter]
    fn set_time_end(&mut self, time_end: Option<i64>) {
        self.inner.time_end = time_end;
    }

    #[getter]
    fn min_confidence(&self) -> Option<f32> {
        self.inner.min_confidence
    }

    #[setter]
    fn set_min_confidence(&mut self, min_confidence: Option<f32>) {
        self.inner.min_confidence = min_confidence;
    }

    #[getter]
    fn limit(&self) -> usize {
        self.inner.limit
    }

    #[setter]
    fn set_limit(&mut self, limit: usize) {
        self.inner.limit = limit;
    }

    #[getter]
    fn candidate_budget(&self) -> Option<usize> {
        self.inner.candidate_budget
    }

    #[setter]
    fn set_candidate_budget(&mut self, candidate_budget: Option<usize>) {
        self.inner.candidate_budget = candidate_budget;
    }
}

#[pymethods]
impl PyMemorySegment {
    #[classmethod]
    fn build(
        _cls: &Bound<'_, PyType>,
        segment_id: u64,
        dim: usize,
        records: Vec<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let records = records
            .into_iter()
            .map(|record| memory_record_from_dict(&record))
            .collect::<PyResult<Vec<_>>>()?;
        let inner = MemorySegment::build(segment_id, dim, records).map_err(value_error)?;
        Ok(Self { inner })
    }

    fn write(&self, path: PathBuf) -> PyResult<()> {
        self.inner.write(path).map_err(io_error)
    }

    #[classmethod]
    fn read(_cls: &Bound<'_, PyType>, path: PathBuf) -> PyResult<Self> {
        let inner = MemorySegment::read(path).map_err(io_error)?;
        Ok(Self { inner })
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

#[pymethods]
impl PyMemoryManifestFile {
    #[new]
    #[pyo3(signature = (branch, segments, parent_manifest_id=None))]
    fn new(
        branch: &str,
        segments: Vec<Bound<'_, PyDict>>,
        parent_manifest_id: Option<u64>,
    ) -> PyResult<Self> {
        let branch_id = stable_memory_branch_id(branch);
        let segments = segments
            .into_iter()
            .map(|segment| memory_manifest_segment_from_dict(&segment))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: MemoryManifestFile::new(parent_manifest_id, branch_id, segments),
        })
    }

    #[getter]
    fn manifest_id(&self) -> u64 {
        self.inner.manifest_id
    }

    #[getter]
    fn parent_manifest_id(&self) -> Option<u64> {
        self.inner.parent_manifest_id
    }

    #[getter]
    fn branch_id(&self) -> u64 {
        self.inner.branch_id
    }

    fn write(&self, path: PathBuf) -> PyResult<()> {
        self.inner.write(path).map_err(io_error)
    }

    #[classmethod]
    fn read(_cls: &Bound<'_, PyType>, path: PathBuf) -> PyResult<Self> {
        let inner = MemoryManifestFile::read(path).map_err(io_error)?;
        Ok(Self { inner })
    }
}

#[pymethods]
impl PyMemorySpace {
    #[classmethod]
    fn open(_cls: &Bound<'_, PyType>, manifest_path: PathBuf) -> PyResult<Self> {
        let inner = MemorySpace::open(manifest_path).map_err(io_error)?;
        Ok(Self { inner })
    }

    fn recall(&self, py: Python<'_>, query: &PyRecallQuery) -> PyResult<PyObject> {
        let result = self.inner.recall(&query.inner).map_err(value_error)?;
        let out = PyDict::new(py);
        out.set_item("hits", memory_hits_to_py(py, &result.hits)?)?;
        out.set_item("trace", memory_space_trace_to_py(py, &result.trace)?)?;
        Ok(out.into())
    }

    fn fork(&self, branch: &str, out_path: PathBuf) -> PyResult<()> {
        self.inner.fork(branch, out_path).map_err(io_error)
    }
}

#[pymethods]
impl PyHlrRouter {
    #[new]
    #[pyo3(signature = (dim, vectors, layer_configs))]
    fn new(
        dim: usize,
        vectors: &Bound<'_, PyAny>,
        layer_configs: Vec<(usize, f32)>,
    ) -> PyResult<Self> {
        let vectors = extract_matrix(vectors)?;
        let configs = hlr_configs(layer_configs);
        let inner = HierarchicalLatticeRouter::new(dim, &configs, &vectors).map_err(value_error)?;
        Ok(Self { inner })
    }

    fn route(&self, query: Vec<f32>, nprobe: usize) -> Vec<usize> {
        self.inner.route_nprobe(&query, nprobe)
    }

    fn route_many(&self, queries: PyReadonlyArray2<'_, f32>, nprobe: usize) -> Vec<Vec<usize>> {
        queries
            .as_array()
            .outer_iter()
            .map(|row| {
                let query = row.iter().copied().collect::<Vec<_>>();
                self.inner.route_nprobe(&query, nprobe)
            })
            .collect()
    }

    fn residual_energy(&self) -> Vec<f32> {
        self.inner.residual_energy.clone()
    }
}

#[pymethods]
impl PyHtlaRouter {
    #[new]
    #[pyo3(signature = (dim, vectors, levels, chart_dim))]
    fn new(
        dim: usize,
        vectors: &Bound<'_, PyAny>,
        levels: usize,
        chart_dim: usize,
    ) -> PyResult<Self> {
        let vectors = extract_matrix(vectors)?;
        let inner = HtlaRouter::new(dim, &vectors, levels, chart_dim).map_err(value_error)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (query, beam, pool, final_nprobe=16))]
    fn route(
        &self,
        query: Vec<f32>,
        beam: usize,
        pool: usize,
        final_nprobe: usize,
    ) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let route = self.inner.route(&query, beam, pool, final_nprobe);
            let dict = PyDict::new(py);
            dict.set_item("candidates", route.candidates)?;
            dict.set_item("final_nprobe", route.final_nprobe)?;
            dict.set_item("fallback", route.fallback)?;
            dict.set_item("working_set_bytes", route.working_set_bytes)?;
            Ok(dict.into())
        })
    }

    #[pyo3(signature = (queries, beam, pool, final_nprobe=16))]
    fn route_many(
        &self,
        queries: PyReadonlyArray2<'_, f32>,
        beam: usize,
        pool: usize,
        final_nprobe: usize,
    ) -> PyResult<Vec<PyObject>> {
        Python::with_gil(|py| {
            queries
                .as_array()
                .outer_iter()
                .map(|row| {
                    let query = row.iter().copied().collect::<Vec<_>>();
                    let route = self.inner.route(&query, beam, pool, final_nprobe);
                    let dict = PyDict::new(py);
                    dict.set_item("candidates", route.candidates)?;
                    dict.set_item("final_nprobe", route.final_nprobe)?;
                    dict.set_item("fallback", route.fallback)?;
                    dict.set_item("working_set_bytes", route.working_set_bytes)?;
                    Ok(dict.into())
                })
                .collect()
        })
    }

    fn resident_bytes(&self) -> usize {
        self.inner.resident_bytes()
    }

    fn diagnostics(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let d = &self.inner.diagnostics;
            let dict = PyDict::new(py);
            dict.set_item("pca_energy", d.pca_energy.clone())?;
            dict.set_item("d80", d.d80.clone())?;
            dict.set_item("d90", d.d90.clone())?;
            dict.set_item("d95", d.d95.clone())?;
            dict.set_item("residual_energy", d.residual_energy.clone())?;
            dict.set_item("radius_shrink", d.radius_shrink.clone())?;
            dict.set_item("norm_sep_p10", d.norm_sep_p10)?;
            dict.set_item("norm_sep_p25", d.norm_sep_p25)?;
            Ok(dict.into())
        })
    }
}

#[pymethods]
impl PyAperonIndex {
    #[new]
    #[pyo3(signature = (dim, local_dim=None, sketch_dim=0, block_size=64, rerank_factor=4, residual_bits=8))]
    fn new(
        dim: usize,
        local_dim: Option<usize>,
        sketch_dim: usize,
        block_size: usize,
        rerank_factor: usize,
        residual_bits: u8,
    ) -> PyResult<Self> {
        let mut inner =
            AperonIndex::with_options(dim, local_dim.unwrap_or(dim), sketch_dim, block_size);
        inner
            .set_residual_bits(residual_bits)
            .map_err(value_error)?;
        inner.set_rerank_factor(rerank_factor);
        Ok(Self { inner })
    }

    fn set_rerank_factor(&mut self, factor: usize) {
        self.inner.set_rerank_factor(factor);
    }

    fn get_rerank_factor(&self) -> usize {
        self.inner.rerank_factor()
    }

    #[pyo3(signature = (id_or_vector, vector=None))]
    fn insert(
        &mut self,
        id_or_vector: &Bound<'_, PyAny>,
        vector: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<u64> {
        let (id, vector) = match vector {
            Some(vector) => (id_or_vector.extract::<u64>()?, extract_vector(vector)?),
            None => (
                self.inner.stats().vectors as u64,
                extract_vector(id_or_vector)?,
            ),
        };

        self.inner
            .insert(VectorId::new(id), vector)
            .map_err(value_error)?;
        Ok(id)
    }

    fn insert_many(
        &mut self,
        ids: &Bound<'_, PyAny>,
        matrix: &Bound<'_, PyAny>,
    ) -> PyResult<usize> {
        let ids = extract_ids(ids)?;
        let vectors = extract_matrix(matrix)?;
        if ids.len() != vectors.len() {
            return Err(value_error(format!(
                "ids length mismatch: expected {}, got {}",
                vectors.len(),
                ids.len()
            )));
        }
        for vector in &vectors {
            if vector.len() != self.inner.dim() {
                return Err(value_error(format!(
                    "dimension mismatch: expected {}, got {}",
                    self.inner.dim(),
                    vector.len()
                )));
            }
        }

        let count = vectors.len();
        for (id, vector) in ids.into_iter().zip(vectors) {
            self.inner
                .insert(VectorId::new(id), vector)
                .map_err(value_error)?;
        }
        Ok(count)
    }

    fn attach_raw_vectors(
        &mut self,
        ids: &Bound<'_, PyAny>,
        matrix: &Bound<'_, PyAny>,
    ) -> PyResult<usize> {
        let ids = extract_ids(ids)?;
        let vectors = extract_matrix(matrix)?;
        let count = vectors.len();
        let ids = ids.into_iter().map(VectorId::new).collect::<Vec<_>>();
        self.inner
            .attach_raw_vectors(&ids, &vectors)
            .map_err(value_error)?;
        Ok(count)
    }

    fn rebuild_single_grain(&mut self) -> PyResult<()> {
        self.inner.rebuild_single_grain().map_err(value_error)
    }

    fn rebuild_two_grains(&mut self) -> PyResult<()> {
        self.inner.rebuild_two_grains().map_err(value_error)
    }

    fn rebuild_n_grains(&mut self, grains: usize) -> PyResult<()> {
        self.inner.rebuild_n_grains(grains).map_err(value_error)
    }

    fn enable_dynamic_splitting(&mut self, split_threshold: usize) -> PyResult<()> {
        self.inner
            .enable_dynamic_splitting(split_threshold)
            .map_err(value_error)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        min_local_dim,
        max_local_dim,
        min_sketch_dim=0,
        max_sketch_dim=0,
        min_residual_bits=1,
        max_residual_bits=2,
        variance_target=0.9
    ))]
    fn enable_adaptive_quantization(
        &mut self,
        min_local_dim: usize,
        max_local_dim: usize,
        min_sketch_dim: usize,
        max_sketch_dim: usize,
        min_residual_bits: u8,
        max_residual_bits: u8,
        variance_target: f32,
    ) -> PyResult<()> {
        self.inner
            .enable_adaptive_quantization(
                min_local_dim,
                max_local_dim,
                min_sketch_dim,
                max_sketch_dim,
                min_residual_bits,
                max_residual_bits,
                variance_target,
            )
            .map_err(value_error)
    }

    #[pyo3(signature = (basis_cols=64, local_dim=16, pq_subquantizers=8, pq_bits=4, opq=false))]
    fn enable_shared_basis_pq(
        &mut self,
        basis_cols: usize,
        local_dim: usize,
        pq_subquantizers: usize,
        pq_bits: u8,
        opq: bool,
    ) -> PyResult<()> {
        self.inner
            .enable_shared_basis_pq(basis_cols, local_dim, pq_subquantizers, pq_bits, opq)
            .map_err(value_error)
    }

    #[pyo3(signature = (routing_dim=4, spacing=0.5))]
    fn enable_lattice_routing(&mut self, routing_dim: usize, spacing: f32) -> PyResult<()> {
        self.inner
            .enable_lattice_routing(routing_dim, spacing)
            .map_err(value_error)
    }

    /// Enable Hierarchical Lattice Routing (HLR) with multi-layer residual PCA.
    ///
    /// Args:
    ///     layer_configs: List of (routing_dim, spacing) tuples, one per HLR layer.
    #[pyo3(signature = (layer_configs))]
    fn enable_hlr_routing(&mut self, layer_configs: Vec<(usize, f32)>) -> PyResult<()> {
        let configs = hlr_configs(layer_configs);
        self.inner.enable_hlr_routing(&configs).map_err(value_error)
    }

    #[pyo3(signature = (query, top_k, nprobe=None, rerank_factor=None))]
    fn search(
        &self,
        query: Vec<f32>,
        top_k: usize,
        nprobe: Option<usize>,
        rerank_factor: Option<usize>,
    ) -> PyResult<Vec<(u64, f64)>> {
        let results = match (nprobe, rerank_factor) {
            (Some(np), Some(rf)) => self
                .inner
                .search_with_nprobe_internal(&query, top_k, np, rf),
            (Some(np), None) => self.inner.search_with_nprobe(&query, top_k, np),
            (None, Some(rf)) => self.inner.search_internal(&query, top_k, rf),
            (None, None) => self.inner.search(&query, top_k),
        }
        .map_err(value_error)?;

        Ok(results
            .into_iter()
            .map(|scored| (scored.id.get(), scored.distance))
            .collect())
    }

    #[pyo3(signature = (queries, top_k, nprobe=None, rerank_factor=None))]
    fn search_many(
        &self,
        queries: PyReadonlyArray2<'_, f32>,
        top_k: usize,
        nprobe: Option<usize>,
        rerank_factor: Option<usize>,
    ) -> PyResult<Vec<Vec<(u64, f64)>>> {
        let array = queries.as_array();
        let mut all_results = Vec::with_capacity(array.shape()[0]);
        for row in array.outer_iter() {
            let query: Vec<f32> = row.iter().copied().collect();
            let results = match (nprobe, rerank_factor) {
                (Some(np), Some(rf)) => self
                    .inner
                    .search_with_nprobe_internal(&query, top_k, np, rf),
                (Some(np), None) => self.inner.search_with_nprobe(&query, top_k, np),
                (None, Some(rf)) => self.inner.search_internal(&query, top_k, rf),
                (None, None) => self.inner.search(&query, top_k),
            }
            .map_err(value_error)?;

            let mapped = results
                .into_iter()
                .map(|scored| (scored.id.get(), scored.distance))
                .collect();
            all_results.push(mapped);
        }
        Ok(all_results)
    }

    #[pyo3(signature = (query, top_k, nprobe, candidate_k=100))]
    fn search_tiered(
        &self,
        query: Vec<f32>,
        top_k: usize,
        nprobe: usize,
        candidate_k: usize,
    ) -> PyResult<Vec<(u64, f64)>> {
        let results = self
            .inner
            .search_tiered_with_nprobe(&query, top_k, nprobe, candidate_k)
            .map_err(value_error)?;
        Ok(results
            .into_iter()
            .map(|scored| (scored.id.get(), scored.distance))
            .collect())
    }

    #[pyo3(signature = (queries, top_k, nprobe, candidate_k=100))]
    fn search_many_tiered(
        &self,
        queries: PyReadonlyArray2<'_, f32>,
        top_k: usize,
        nprobe: usize,
        candidate_k: usize,
    ) -> PyResult<Vec<Vec<(u64, f64)>>> {
        let array = queries.as_array();
        let mut all_results = Vec::with_capacity(array.shape()[0]);
        for row in array.outer_iter() {
            let query = row.iter().copied().collect::<Vec<_>>();
            let results = self
                .inner
                .search_tiered_with_nprobe(&query, top_k, nprobe, candidate_k)
                .map_err(value_error)?;
            all_results.push(
                results
                    .into_iter()
                    .map(|scored| (scored.id.get(), scored.distance))
                    .collect(),
            );
        }
        Ok(all_results)
    }

    #[pyo3(signature = (query, nprobe, candidate_k=100))]
    fn candidates(
        &self,
        query: Vec<f32>,
        nprobe: usize,
        candidate_k: usize,
    ) -> PyResult<Vec<(u64, f64)>> {
        let results = self
            .inner
            .candidate_pool_with_nprobe(&query, nprobe, candidate_k)
            .map_err(value_error)?;
        Ok(results
            .into_iter()
            .map(|scored| (scored.id.get(), scored.distance))
            .collect())
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stats = self.inner.stats();
        let out = PyDict::new(py);
        out.set_item("dim", stats.dim)?;
        out.set_item("grains", stats.grains)?;
        out.set_item("vectors", stats.vectors)?;
        out.set_item("grain_sizes", stats.grain_sizes)?;
        out.set_item("residual_bits", stats.residual_bits)?;
        out.set_item("encoded_bytes", stats.encoded_bytes)?;
        out.set_item("grain_local_dims", stats.grain_local_dims)?;
        out.set_item("grain_sketch_dims", stats.grain_sketch_dims)?;
        out.set_item(
            "grain_residual_bits",
            stats
                .grain_residual_bits
                .into_iter()
                .map(usize::from)
                .collect::<Vec<_>>(),
        )?;
        Ok(out)
    }

    fn save(&self, path: PathBuf) -> PyResult<()> {
        let legacy = self
            .inner
            .to_legacy_index()
            .ok_or_else(|| value_error("index has no serializable grains".to_string()))?;
        let file = File::create(path).map_err(io_error)?;
        write_legacy_index(BufWriter::new(file), &legacy).map_err(io_error)
    }

    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, path: PathBuf) -> PyResult<Self> {
        Self::load_path(path)
    }
}

impl PyAperonIndex {
    fn load_path(path: PathBuf) -> PyResult<Self> {
        let file = File::open(path).map_err(io_error)?;
        let legacy = load_legacy_index(file).map_err(io_error)?;
        let inner = AperonIndex::from_legacy_index(legacy).map_err(value_error)?;
        Ok(Self { inner })
    }
}

#[pymodule]
fn aperon(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", VERSION)?;
    module.add_class::<PyAperonIndex>()?;
    module.add_class::<PyHlrRouter>()?;
    module.add_class::<PyHtlaRouter>()?;
    module.add_class::<PyRecallQuery>()?;
    module.add_class::<PyMemorySegment>()?;
    module.add_class::<PyMemoryManifestFile>()?;
    module.add_class::<PyMemorySpace>()?;
    Ok(())
}

fn memory_record_from_dict(dict: &Bound<'_, PyDict>) -> PyResult<MemoryRecordInput> {
    Ok(MemoryRecordInput {
        record_id: required_item(dict, "record_id")?.extract()?,
        scope_id: required_item(dict, "scope_id")?.extract()?,
        timestamp: required_item(dict, "timestamp")?.extract()?,
        source_id: required_item(dict, "source_id")?.extract()?,
        confidence: required_item(dict, "confidence")?.extract()?,
        text: required_item(dict, "text")?.extract()?,
        embedding: required_item(dict, "embedding")?.extract()?,
        symbols: required_item(dict, "symbols")?.extract()?,
    })
}

fn memory_manifest_segment_from_dict(dict: &Bound<'_, PyDict>) -> PyResult<MemoryManifestSegment> {
    Ok(MemoryManifestSegment {
        segment_id: required_item(dict, "segment_id")?.extract()?,
        path: PathBuf::from(required_item(dict, "path")?.extract::<String>()?),
        vector_sidecar: None,
    })
}

fn required_item<'py>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    dict.get_item(key)?
        .ok_or_else(|| value_error(format!("missing required key: {key}")))
}

fn memory_hits_to_py(py: Python<'_>, hits: &[MemoryHit]) -> PyResult<Vec<PyObject>> {
    hits.iter()
        .map(|hit| {
            let dict = PyDict::new(py);
            dict.set_item("record_id", hit.record_id)?;
            dict.set_item("score", hit.score)?;
            dict.set_item("semantic_distance", hit.semantic_distance)?;
            dict.set_item("symbol_matches", hit.symbol_matches)?;
            dict.set_item("confidence", hit.confidence)?;
            dict.set_item("timestamp", hit.timestamp)?;
            dict.set_item("text", &hit.text)?;
            Ok(dict.into())
        })
        .collect()
}

fn memory_space_trace_to_py(py: Python<'_>, trace: &MemorySpaceRecallTrace) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("manifest_id", trace.manifest_id)?;
    dict.set_item("branch_id", trace.branch_id)?;
    dict.set_item("segments_considered", trace.segments_considered)?;
    dict.set_item("segments_scanned", trace.segments_scanned)?;
    dict.set_item("segments_pruned", trace.segments_pruned)?;
    dict.set_item("semantic_evals", trace.semantic_evals)?;
    dict.set_item("returned", trace.returned)?;
    let segment_traces = trace
        .segment_traces
        .iter()
        .map(|segment_trace| memory_space_segment_trace_to_py(py, segment_trace))
        .collect::<PyResult<Vec<_>>>()?;
    dict.set_item("segment_traces", segment_traces)?;
    Ok(dict.into())
}

fn memory_space_segment_trace_to_py(
    py: Python<'_>,
    trace: &MemorySpaceSegmentTrace,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("segment_id", trace.segment_id)?;
    dict.set_item("pruned", trace.pruned)?;
    dict.set_item("prune_reason", trace.prune_reason)?;
    dict.set_item(
        "trace",
        match &trace.trace {
            Some(trace) => Some(recall_trace_to_py(py, trace)?),
            None => None,
        },
    )?;
    Ok(dict.into())
}

fn recall_trace_to_py(py: Python<'_>, trace: &RecallTrace) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("segment_id", trace.segment_id)?;
    dict.set_item("access_paths", trace.access_paths.clone())?;
    dict.set_item("records_total", trace.records_total)?;
    dict.set_item("candidates_after_filters", trace.candidates_after_filters)?;
    dict.set_item("candidates_after_symbols", trace.candidates_after_symbols)?;
    dict.set_item("vector_generator", trace.vector_generator)?;
    dict.set_item("vector_candidates", trace.vector_candidates)?;
    dict.set_item(
        "vector_route",
        match &trace.vector_route {
            Some(route) => Some(vector_route_trace_to_py(py, route)?),
            None => None,
        },
    )?;
    dict.set_item(
        "planner",
        match &trace.planner {
            Some(planner) => Some(planner_trace_to_py(py, planner)?),
            None => None,
        },
    )?;
    dict.set_item("semantic_evals", trace.semantic_evals)?;
    dict.set_item("returned", trace.returned)?;
    Ok(dict.into())
}

fn vector_route_trace_to_py(py: Python<'_>, trace: &MemoryVectorRouteTrace) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("vector_index_bytes", trace.vector_index_bytes)?;
    dict.set_item("route_candidates", trace.route_candidates)?;
    dict.set_item("posting_entries_touched", trace.posting_entries_touched)?;
    dict.set_item("duplicate_block_rate", trace.duplicate_block_rate)?;
    dict.set_item("selected_blocks", trace.selected_blocks)?;
    dict.set_item("centroid_evals", trace.centroid_evals)?;
    dict.set_item("working_set_bytes", trace.working_set_bytes)?;
    dict.set_item("fallback_used", trace.fallback_used)?;
    Ok(dict.into())
}

fn planner_trace_to_py(py: Python<'_>, trace: &MemoryQueryPlannerTrace) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("selected_path", trace.selected_path)?;
    dict.set_item("candidate_budget", trace.candidate_budget)?;
    dict.set_item("expanded_candidate_budget", trace.expanded_candidate_budget)?;
    dict.set_item("fallback_reason", trace.fallback_reason)?;
    dict.set_item("candidates_after_symbols", trace.candidates_after_symbols)?;
    dict.set_item("final_candidates", trace.final_candidates)?;
    Ok(dict.into())
}

fn hlr_configs(layer_configs: Vec<(usize, f32)>) -> Vec<HierarchicalLatticeLayerConfig> {
    layer_configs
        .into_iter()
        .map(|(routing_dim, spacing)| HierarchicalLatticeLayerConfig {
            routing_dim,
            spacing,
        })
        .collect()
}

fn extract_vector(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    if let Ok(vector) = obj.extract::<Vec<f32>>() {
        return Ok(vector);
    }
    if let Ok(array) = obj.extract::<PyReadonlyArray1<'_, f32>>() {
        return Ok(array.as_array().iter().copied().collect());
    }
    obj.extract::<Vec<f32>>()
}

fn extract_matrix(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f32>>> {
    if let Ok(matrix) = obj.extract::<Vec<Vec<f32>>>() {
        return Ok(matrix);
    }
    if let Ok(array) = obj.extract::<PyReadonlyArray2<'_, f32>>() {
        return Ok(array
            .as_array()
            .outer_iter()
            .map(|row| row.iter().copied().collect())
            .collect());
    }
    obj.extract::<Vec<Vec<f32>>>()
}

fn extract_ids(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u64>> {
    if let Ok(ids) = obj.extract::<Vec<u64>>() {
        return Ok(ids);
    }
    if let Ok(array) = obj.extract::<PyReadonlyArray1<'_, u64>>() {
        return Ok(array.as_array().iter().copied().collect());
    }
    obj.extract::<Vec<u64>>()
}

fn value_error(message: String) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(message)
}

fn io_error(error: std::io::Error) -> PyErr {
    pyo3::exceptions::PyOSError::new_err(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, process};

    #[test]
    #[allow(deprecated)]
    fn stats_returns_python_dict() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let mut index = PyAperonIndex::new(2, None, 0, 4, 4, 8).unwrap();
            let id = 7_u64.into_pyobject(py).unwrap();
            let vector = vec![1.0, 2.0].into_pyobject(py).unwrap();
            index.insert(id.as_any(), Some(vector.as_any())).unwrap();

            let stats = index.stats(py).unwrap();

            assert_eq!(
                stats
                    .get_item("dim")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                2
            );
            assert_eq!(
                stats
                    .get_item("grains")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                1
            );
            assert_eq!(
                stats
                    .get_item("vectors")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                1
            );
            assert_eq!(
                stats
                    .get_item("grain_sizes")
                    .unwrap()
                    .unwrap()
                    .extract::<Vec<usize>>()
                    .unwrap(),
                vec![1]
            );
        });
    }

    #[test]
    #[allow(deprecated)]
    fn stats_returns_python_grain_sizes_for_multi_grain_index() {
        pyo3::prepare_freethreaded_python();
        let mut index = PyAperonIndex::new(2, Some(2), 0, 2, 4, 8).unwrap();
        Python::with_gil(|py| {
            for cluster in 0..4 {
                let base = cluster as f32 * 100.0;
                for offset in 0..2 {
                    let id = ((cluster * 10 + offset) as u64).into_pyobject(py).unwrap();
                    let vector = vec![base + offset as f32, base].into_pyobject(py).unwrap();
                    index.insert(id.as_any(), Some(vector.as_any())).unwrap();
                }
            }
            index.rebuild_n_grains(4).unwrap();

            let stats = index.stats(py).unwrap();

            assert_eq!(
                stats
                    .get_item("grain_sizes")
                    .unwrap()
                    .unwrap()
                    .extract::<Vec<usize>>()
                    .unwrap(),
                vec![2, 2, 2, 2]
            );
        });
    }

    #[test]
    #[allow(deprecated)]
    fn save_and_load_round_trip_searchable_index() {
        pyo3::prepare_freethreaded_python();
        let path = std::env::temp_dir().join(format!(
            "aperon-py-save-load-{}-{}.hntl",
            process::id(),
            unique_suffix()
        ));
        let mut index = PyAperonIndex::new(2, Some(2), 0, 4, 4, 8).unwrap();
        Python::with_gil(|py| {
            for (id, vector) in [
                (0_u64, vec![0.0, 0.0]),
                (1_u64, vec![10.0, 0.0]),
                (2_u64, vec![0.0, 10.0]),
            ] {
                let id = id.into_pyobject(py).unwrap();
                let vector = vector.into_pyobject(py).unwrap();
                index.insert(id.as_any(), Some(vector.as_any())).unwrap();
            }
        });
        index.rebuild_single_grain().unwrap();

        index.save(path.clone()).unwrap();
        let loaded = PyAperonIndex::load_path(path.clone()).unwrap();
        fs::remove_file(path).unwrap();

        let results = loaded.search(vec![9.0, 0.0], 1, None, None).unwrap();
        assert_eq!(results[0].0, 1);
        Python::with_gil(|py| {
            let stats = loaded.stats(py).unwrap();
            assert_eq!(
                stats
                    .get_item("vectors")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                3
            );
        });
    }

    fn unique_suffix() -> u128 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    }

    #[test]
    #[allow(deprecated)]
    fn insert_many_accepts_python_sequences() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let mut index = PyAperonIndex::new(2, None, 0, 4, 4, 8).unwrap();
            let ids = vec![10_u64, 11_u64].into_pyobject(py).unwrap();
            let matrix = vec![vec![1.0, 0.0], vec![2.0, 0.0]]
                .into_pyobject(py)
                .unwrap();

            let inserted = index.insert_many(ids.as_any(), matrix.as_any()).unwrap();

            assert_eq!(inserted, 2);
            assert_eq!(
                index.search(vec![1.8, 0.0], 1, None, None).unwrap()[0].0,
                11
            );
        });
    }

    #[test]
    #[allow(deprecated)]
    fn search_reranks_and_improves_accuracy_py() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let mut index = PyAperonIndex::new(3, Some(2), 1, 4, 4, 8).unwrap();
            let ids = vec![1_u64, 2_u64, 3_u64].into_pyobject(py).unwrap();
            let matrix = vec![
                vec![0.0, 0.0, 0.0],
                vec![10.0, 0.0, 0.0],
                vec![0.0, 10.0, 0.0],
            ]
            .into_pyobject(py)
            .unwrap();
            index.insert_many(ids.as_any(), matrix.as_any()).unwrap();
            index.rebuild_single_grain().unwrap();

            // Reranking should find the correct nearest neighbor
            let results = index.search(vec![9.0, 0.0, 0.0], 1, None, Some(4)).unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].0, 2);
            assert!((results[0].1 - 1.0).abs() < 1.0);
        });
    }

    #[test]
    #[allow(deprecated)]
    fn memory_sstable_python_workflow_round_trip() {
        pyo3::prepare_freethreaded_python();
        let dir = std::env::temp_dir().join(format!(
            "aperon-py-memory-sstable-{}-{}",
            process::id(),
            unique_suffix()
        ));
        fs::create_dir_all(&dir).unwrap();
        let segment_path = dir.join("segment-191.apms");
        let manifest_path = dir.join("main.apmf");
        let fork_path = dir.join("fork.apmf");

        Python::with_gil(|py| {
            let record_a = PyDict::new(py);
            record_a.set_item("record_id", 191001_u64).unwrap();
            record_a.set_item("scope_id", 7_u32).unwrap();
            record_a.set_item("timestamp", 191_i64).unwrap();
            record_a.set_item("source_id", 1_u16).unwrap();
            record_a.set_item("confidence", 0.97_f32).unwrap();
            record_a
                .set_item("text", "T-191 exposes Memory SSTable bindings.")
                .unwrap();
            record_a
                .set_item("embedding", vec![1.0_f32, 0.0, 0.0, 0.0])
                .unwrap();
            record_a
                .set_item("symbols", vec!["T-191", "python"])
                .unwrap();

            let record_b = PyDict::new(py);
            record_b.set_item("record_id", 191002_u64).unwrap();
            record_b.set_item("scope_id", 7_u32).unwrap();
            record_b.set_item("timestamp", 192_i64).unwrap();
            record_b.set_item("source_id", 1_u16).unwrap();
            record_b.set_item("confidence", 0.90_f32).unwrap();
            record_b
                .set_item("text", "Unrelated memory record.")
                .unwrap();
            record_b
                .set_item("embedding", vec![0.0_f32, 1.0, 0.0, 0.0])
                .unwrap();
            record_b.set_item("symbols", vec!["other"]).unwrap();

            let segment = PyMemorySegment::build(
                &py.get_type::<PyMemorySegment>(),
                191,
                4,
                vec![record_a, record_b],
            )
            .unwrap();
            assert_eq!(segment.len(), 2);
            segment.write(segment_path.clone()).unwrap();

            let manifest_segment = PyDict::new(py);
            manifest_segment.set_item("segment_id", 191_u64).unwrap();
            manifest_segment
                .set_item("path", "segment-191.apms")
                .unwrap();
            let manifest = PyMemoryManifestFile::new("main", vec![manifest_segment], None).unwrap();
            manifest.write(manifest_path.clone()).unwrap();

            let space = PyMemorySpace::open(&py.get_type::<PyMemorySpace>(), manifest_path.clone())
                .unwrap();
            let query = PyRecallQuery::new(
                Some(vec![1.0, 0.0, 0.0, 0.0]),
                vec!["python".to_string()],
                Some(7),
                None,
                None,
                Some(0.95),
                5,
                Some(10),
            );

            let result = space.recall(py, &query).unwrap();
            let result = result.bind(py).downcast::<PyDict>().unwrap();
            let hits = result
                .get_item("hits")
                .unwrap()
                .unwrap()
                .extract::<Vec<Bound<'_, PyDict>>>()
                .unwrap();
            assert_eq!(hits.len(), 1);
            assert_eq!(
                hits[0]
                    .get_item("record_id")
                    .unwrap()
                    .unwrap()
                    .extract::<u64>()
                    .unwrap(),
                191001
            );

            let trace = result.get_item("trace").unwrap().unwrap();
            let trace = trace.downcast::<PyDict>().unwrap();
            assert_eq!(
                trace
                    .get_item("segments_scanned")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                1
            );
            assert_eq!(
                trace
                    .get_item("returned")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                1
            );

            space
                .fork("python-experimental-branch", fork_path.clone())
                .unwrap();
            let forked = PyMemoryManifestFile::read(
                &py.get_type::<PyMemoryManifestFile>(),
                fork_path.clone(),
            )
            .unwrap();
            assert_eq!(forked.parent_manifest_id(), Some(manifest.manifest_id()));
            assert_ne!(forked.manifest_id(), manifest.manifest_id());
        });

        fs::remove_file(segment_path).unwrap();
        fs::remove_file(manifest_path).unwrap();
        fs::remove_file(fork_path).unwrap();
        fs::remove_dir(dir).unwrap();
    }
}
