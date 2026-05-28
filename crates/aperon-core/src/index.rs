use crate::{
    distance::l2_squared, grain::ScoredVector, Grain, GrainId, VectorId, DEFAULT_BLOCK_SIZE,
};

/// Minimal in-memory index skeleton.
#[derive(Clone, Debug)]
pub struct AperonIndex {
    dim: usize,
    local_dim: usize,
    sketch_dim: usize,
    block_size: usize,
    grains: Vec<Grain>,
    centroids: Vec<Vec<f32>>,
    grain_ids: Vec<Vec<VectorId>>,
    grain_vectors: Vec<Vec<Vec<f32>>>,
    ids: Vec<VectorId>,
    raw_vectors: Vec<Vec<f32>>,
    split_threshold: Option<usize>,
}

impl AperonIndex {
    pub fn new(dim: usize) -> Self {
        let local_dim = dim;
        Self {
            dim,
            local_dim,
            sketch_dim: 0,
            block_size: DEFAULT_BLOCK_SIZE,
            grains: vec![Grain::new(GrainId::new(0), dim)],
            centroids: vec![vec![0.0; dim]],
            grain_ids: vec![Vec::new()],
            grain_vectors: vec![Vec::new()],
            ids: Vec::new(),
            raw_vectors: Vec::new(),
            split_threshold: None,
        }
    }

    pub fn with_options(
        dim: usize,
        local_dim: usize,
        sketch_dim: usize,
        block_size: usize,
    ) -> Self {
        Self {
            dim,
            local_dim: local_dim.min(dim),
            sketch_dim,
            block_size,
            grains: vec![Grain::new(GrainId::new(0), dim)],
            centroids: vec![vec![0.0; dim]],
            grain_ids: vec![Vec::new()],
            grain_vectors: vec![Vec::new()],
            ids: Vec::new(),
            raw_vectors: Vec::new(),
            split_threshold: None,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn enable_dynamic_splitting(&mut self, split_threshold: usize) -> Result<(), String> {
        if split_threshold < self.block_size {
            return Err(format!(
                "split threshold must be at least block_size {}",
                self.block_size
            ));
        }
        self.split_threshold = Some(split_threshold);
        Ok(())
    }

    pub fn insert(
        &mut self,
        id: impl Into<VectorId>,
        vector: impl Into<Vec<f32>>,
    ) -> Result<(), String> {
        let id = id.into();
        let vector = vector.into();
        if vector.len() != self.dim {
            return Err(format!(
                "dimension mismatch: expected {}, got {}",
                self.dim,
                vector.len()
            ));
        }

        let route = if self.grains.len() == 1 {
            0
        } else {
            self.route(&vector)?[0]
        };

        self.grains[route].insert(id, vector.clone())?;
        self.grain_ids[route].push(id);
        self.grain_vectors[route].push(vector.clone());
        self.ids.push(id);
        self.raw_vectors.push(vector);

        if self
            .split_threshold
            .is_some_and(|threshold| self.grain_ids[route].len() >= threshold)
        {
            self.split_grain(route)?;
        }
        Ok(())
    }

    pub fn rebuild_single_grain(&mut self) -> Result<(), String> {
        self.grains = vec![Grain::build(
            GrainId::new(0),
            &self.raw_vectors,
            &self.ids,
            self.dim,
            self.local_dim,
            self.sketch_dim,
            self.block_size,
        )?];
        self.centroids = vec![mean_vector(&self.raw_vectors, self.dim)];
        self.grain_ids = vec![self.ids.clone()];
        self.grain_vectors = vec![self.raw_vectors.clone()];
        Ok(())
    }

    pub fn rebuild_two_grains(&mut self) -> Result<(), String> {
        if self.raw_vectors.len() < 2 {
            return self.rebuild_single_grain();
        }

        let split = two_means(&self.raw_vectors, self.dim);
        let mut ids0 = Vec::new();
        let mut ids1 = Vec::new();
        let mut vectors0 = Vec::new();
        let mut vectors1 = Vec::new();
        for (idx, assignment) in split.assignments.iter().enumerate() {
            if *assignment == 0 {
                ids0.push(self.ids[idx]);
                vectors0.push(self.raw_vectors[idx].clone());
            } else {
                ids1.push(self.ids[idx]);
                vectors1.push(self.raw_vectors[idx].clone());
            }
        }

        if ids0.is_empty() || ids1.is_empty() {
            return self.rebuild_single_grain();
        }

        self.grains = vec![
            Grain::build(
                GrainId::new(0),
                &vectors0,
                &ids0,
                self.dim,
                self.local_dim,
                self.sketch_dim,
                self.block_size,
            )?,
            Grain::build(
                GrainId::new(1),
                &vectors1,
                &ids1,
                self.dim,
                self.local_dim,
                self.sketch_dim,
                self.block_size,
            )?,
        ];
        self.centroids = vec![split.centroid0, split.centroid1];
        self.grain_ids = vec![ids0, ids1];
        self.grain_vectors = vec![vectors0, vectors1];
        Ok(())
    }

    pub fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<ScoredVector>, String> {
        let mut results = Vec::new();
        for grain in self.routed_grains(query)? {
            results.extend(grain.scan(query, top_k)?);
        }
        results.sort_by(|a, b| {
            a.distance
                .total_cmp(&b.distance)
                .then_with(|| a.id.cmp(&b.id))
        });
        results.truncate(top_k);
        Ok(results)
    }

    pub fn search_with_nprobe(
        &self,
        query: &[f32],
        top_k: usize,
        nprobe: usize,
    ) -> Result<Vec<ScoredVector>, String> {
        let mut routes = self.route(query)?;
        routes.truncate(nprobe.max(1));

        let mut results = Vec::new();
        for route in routes {
            results.extend(self.grains[route].scan(query, top_k)?);
        }
        results.sort_by(|a, b| {
            a.distance
                .total_cmp(&b.distance)
                .then_with(|| a.id.cmp(&b.id))
        });
        results.truncate(top_k);
        Ok(results)
    }

    fn routed_grains(&self, query: &[f32]) -> Result<Vec<&Grain>, String> {
        Ok(self
            .route(query)?
            .into_iter()
            .map(|idx| &self.grains[idx])
            .collect())
    }

    fn route(&self, query: &[f32]) -> Result<Vec<usize>, String> {
        if query.len() != self.dim {
            return Err(format!(
                "dimension mismatch: expected {}, got {}",
                self.dim,
                query.len()
            ));
        }

        let mut routes = self
            .centroids
            .iter()
            .enumerate()
            .map(|(idx, centroid)| {
                let distance = l2_squared(query, centroid).unwrap_or(f32::INFINITY);
                (idx, distance)
            })
            .collect::<Vec<_>>();
        routes.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        Ok(routes.into_iter().map(|(idx, _)| idx).collect())
    }

    fn split_grain(&mut self, grain_idx: usize) -> Result<(), String> {
        let ids = self.grain_ids[grain_idx].clone();
        let vectors = self.grain_vectors[grain_idx].clone();
        if ids.len() < self.block_size * 2 {
            return Ok(());
        }

        let split = two_means(&vectors, self.dim);
        let mut ids0 = Vec::new();
        let mut ids1 = Vec::new();
        let mut vectors0 = Vec::new();
        let mut vectors1 = Vec::new();
        for (idx, assignment) in split.assignments.iter().enumerate() {
            if *assignment == 0 {
                ids0.push(ids[idx]);
                vectors0.push(vectors[idx].clone());
            } else {
                ids1.push(ids[idx]);
                vectors1.push(vectors[idx].clone());
            }
        }

        if ids0.len() < self.block_size || ids1.len() < self.block_size {
            return Ok(());
        }

        self.grains[grain_idx] = Grain::build(
            GrainId::new(grain_idx as u64),
            &vectors0,
            &ids0,
            self.dim,
            self.local_dim,
            self.sketch_dim,
            self.block_size,
        )?;
        self.centroids[grain_idx] = split.centroid0;
        self.grain_ids[grain_idx] = ids0;
        self.grain_vectors[grain_idx] = vectors0;

        let new_idx = self.grains.len();
        self.grains.push(Grain::build(
            GrainId::new(new_idx as u64),
            &vectors1,
            &ids1,
            self.dim,
            self.local_dim,
            self.sketch_dim,
            self.block_size,
        )?);
        self.centroids.push(split.centroid1);
        self.grain_ids.push(ids1);
        self.grain_vectors.push(vectors1);
        Ok(())
    }

    pub fn stats(&self) -> IndexStats {
        IndexStats {
            dim: self.dim,
            grains: self.grains.len(),
            vectors: self.grains.iter().map(Grain::len).sum(),
        }
    }
}

/// Small status snapshot exposed by the core crate and CLI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndexStats {
    pub dim: usize,
    pub grains: usize,
    pub vectors: usize,
}

#[derive(Clone, Debug)]
struct TwoMeansSplit {
    assignments: Vec<usize>,
    centroid0: Vec<f32>,
    centroid1: Vec<f32>,
}

fn two_means(vectors: &[Vec<f32>], dim: usize) -> TwoMeansSplit {
    let mut centroid0 = vectors[0].clone();
    let mut centroid1 = vectors
        .iter()
        .max_by(|a, b| {
            l2_squared(a, &centroid0)
                .unwrap_or(0.0)
                .total_cmp(&l2_squared(b, &centroid0).unwrap_or(0.0))
        })
        .cloned()
        .unwrap_or_else(|| vec![0.0; dim]);
    let mut assignments = vec![0; vectors.len()];

    for _ in 0..20 {
        let mut next = vec![0; vectors.len()];
        let mut sums = [vec![0.0; dim], vec![0.0; dim]];
        let mut counts = [0_usize, 0_usize];

        for (idx, vector) in vectors.iter().enumerate() {
            let d0 = l2_squared(vector, &centroid0).unwrap_or(f32::INFINITY);
            let d1 = l2_squared(vector, &centroid1).unwrap_or(f32::INFINITY);
            let assignment = usize::from(d1 < d0);
            next[idx] = assignment;
            counts[assignment] += 1;
            for (sum, value) in sums[assignment].iter_mut().zip(vector) {
                *sum += value;
            }
        }

        if counts[0] == 0 || counts[1] == 0 {
            break;
        }

        for (centroid, (sum, count)) in [&mut centroid0, &mut centroid1]
            .into_iter()
            .zip(sums.into_iter().zip(counts))
        {
            for (slot, value) in centroid.iter_mut().zip(sum) {
                *slot = value / count as f32;
            }
        }

        if next == assignments {
            break;
        }
        assignments = next;
    }

    TwoMeansSplit {
        assignments,
        centroid0,
        centroid1,
    }
}

fn mean_vector(vectors: &[Vec<f32>], dim: usize) -> Vec<f32> {
    if vectors.is_empty() {
        return vec![0.0; dim];
    }
    let mut mean = vec![0.0; dim];
    for vector in vectors {
        for (slot, value) in mean.iter_mut().zip(vector) {
            *slot += value;
        }
    }
    for value in &mut mean {
        *value /= vectors.len() as f32;
    }
    mean
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebuilds_and_searches_quantized_single_grain() {
        let mut index = AperonIndex::with_options(3, 2, 1, 4);
        index.insert(1, [0.0, 0.0, 0.0]).unwrap();
        index.insert(2, [10.0, 0.0, 0.0]).unwrap();
        index.insert(3, [0.0, 10.0, 0.0]).unwrap();
        index.rebuild_single_grain().unwrap();

        let results = index.search(&[9.0, 0.0, 0.0], 2).unwrap();

        assert_eq!(results[0].id, VectorId::new(2));
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn routes_search_across_two_grains() {
        let mut index = AperonIndex::with_options(2, 2, 0, 4);
        index.insert(1, [0.0, 0.0]).unwrap();
        index.insert(2, [1.0, 0.0]).unwrap();
        index.insert(10, [100.0, 100.0]).unwrap();
        index.insert(11, [101.0, 100.0]).unwrap();
        index.rebuild_two_grains().unwrap();

        let stats = index.stats();
        assert_eq!(stats.grains, 2);

        let results = index.search_with_nprobe(&[100.2, 100.0], 1, 1).unwrap();
        assert_eq!(results[0].id, VectorId::new(10));
    }

    #[test]
    fn dynamic_insert_splits_large_grains() {
        let mut index = AperonIndex::with_options(2, 2, 0, 2);
        index.enable_dynamic_splitting(4).unwrap();
        for i in 0..4 {
            index.insert(i as u64, [i as f32, 0.0]).unwrap();
        }

        let stats = index.stats();
        assert_eq!(stats.vectors, 4);
        assert_eq!(stats.grains, 2);
    }
}
