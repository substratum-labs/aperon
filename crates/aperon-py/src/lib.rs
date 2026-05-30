use aperon_core::{
    binary::{load_legacy_index, write_legacy_index},
    AperonIndex, VectorId,
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
    Ok(())
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
}
