use aperon_core::{AperonIndex, VectorId};
use pyo3::prelude::*;

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

    fn stats(&self) -> (usize, usize, usize) {
        let stats = self.inner.stats();
        (stats.dim, stats.grains, stats.vectors)
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
