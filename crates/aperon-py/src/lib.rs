use aperon_core::{
    binary::{load_legacy_index, write_legacy_index},
    AperonIndex, VectorId,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};
use std::{fs::File, io::BufWriter, path::PathBuf};

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

    fn insert(&mut self, id: u64, vector: Vec<f32>) -> PyResult<()> {
        self.inner
            .insert(VectorId::new(id), vector)
            .map_err(value_error)
    }

    fn rebuild_single_grain(&mut self) -> PyResult<()> {
        self.inner.rebuild_single_grain().map_err(value_error)
    }

    fn rebuild_two_grains(&mut self) -> PyResult<()> {
        self.inner.rebuild_two_grains().map_err(value_error)
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
fn aperon_py(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyAperonIndex>()?;
    Ok(())
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
            index.insert(7, vec![1.0, 2.0]).unwrap();

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
        index.insert(0, vec![0.0, 0.0]).unwrap();
        index.insert(1, vec![10.0, 0.0]).unwrap();
        index.insert(2, vec![0.0, 10.0]).unwrap();
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
}
