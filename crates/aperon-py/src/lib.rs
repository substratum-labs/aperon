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
    #[pyo3(signature = (dim, local_dim=None, sketch_dim=0, block_size=64))]
    fn new(dim: usize, local_dim: Option<usize>, sketch_dim: usize, block_size: usize) -> Self {
        let inner =
            AperonIndex::with_options(dim, local_dim.unwrap_or(dim), sketch_dim, block_size);
        Self { inner }
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

    #[pyo3(signature = (query, top_k, nprobe=None))]
    fn search(
        &self,
        query: Vec<f32>,
        top_k: usize,
        nprobe: Option<usize>,
    ) -> PyResult<Vec<(u64, f64)>> {
        let results = match nprobe {
            Some(nprobe) => self.inner.search_with_nprobe(&query, top_k, nprobe),
            None => self.inner.search(&query, top_k),
        }
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
            let mut index = PyAperonIndex::new(2, None, 0, 4);
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
        let mut index = PyAperonIndex::new(2, Some(2), 0, 4);
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

        let results = loaded.search(vec![9.0, 0.0], 1, None).unwrap();
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
            let mut index = PyAperonIndex::new(2, None, 0, 4);
            let ids = vec![10_u64, 11_u64].into_pyobject(py).unwrap();
            let matrix = vec![vec![1.0, 0.0], vec![2.0, 0.0]]
                .into_pyobject(py)
                .unwrap();

            let inserted = index.insert_many(ids.as_any(), matrix.as_any()).unwrap();

            assert_eq!(inserted, 2);
            assert_eq!(index.search(vec![1.8, 0.0], 1, None).unwrap()[0].0, 11);
        });
    }
}
