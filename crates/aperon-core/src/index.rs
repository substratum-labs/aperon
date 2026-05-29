use crate::{
    binary::{LegacyIndex, LegacyMultiGrain, LegacySingleGrain},
    distance::l2_squared,
    grain::ScoredVector,
    Grain, GrainId, VectorId, DEFAULT_BLOCK_SIZE,
};
use std::collections::HashMap;

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
    id_to_index: HashMap<VectorId, usize>,
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
            id_to_index: HashMap::new(),
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
            id_to_index: HashMap::new(),
            ids: Vec::new(),
            raw_vectors: Vec::new(),
            split_threshold: None,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn from_legacy_index(index: LegacyIndex) -> Result<Self, String> {
        match index {
            LegacyIndex::Single(single) => {
                let dim = single.dimension as usize;
                let local_dim = single.local_dim as usize;
                let sketch_dim = single.sketch_dim as usize;
                let block_size = single.block_size as usize;
                let grain = Grain::from_legacy_single(GrainId::new(0), single)?;
                let ids = grain.vector_ids().to_vec();
                Ok(Self {
                    dim,
                    local_dim,
                    sketch_dim,
                    block_size,
                    centroids: vec![grain.centroid().to_vec()],
                    grain_ids: vec![ids],
                    id_to_index: HashMap::new(),
                    ids: Vec::new(),
                    raw_vectors: Vec::new(),
                    grains: vec![grain],
                    split_threshold: None,
                })
            }
            LegacyIndex::Multi(multi) => {
                let dim = multi.dimension as usize;
                let local_dim = multi.local_dim as usize;
                let sketch_dim = multi.sketch_dim as usize;
                let block_size = multi.block_size as usize;
                let grains = multi
                    .grains
                    .into_iter()
                    .enumerate()
                    .map(|(idx, grain)| Grain::from_legacy_single(GrainId::new(idx as u64), grain))
                    .collect::<Result<Vec<_>, _>>()?;
                let grain_ids = grains
                    .iter()
                    .map(|grain| grain.vector_ids().to_vec())
                    .collect::<Vec<_>>();
                let centroids = multi
                    .centroids
                    .chunks_exact(dim)
                    .map(|chunk| chunk.to_vec())
                    .collect::<Vec<_>>();
                if centroids.len() != grains.len() {
                    return Err(format!(
                        "centroid count mismatch: expected {}, got {}",
                        grains.len(),
                        centroids.len()
                    ));
                }
                Ok(Self {
                    dim,
                    local_dim,
                    sketch_dim,
                    block_size,
                    grains,
                    centroids,
                    grain_ids,
                    id_to_index: HashMap::new(),
                    ids: Vec::new(),
                    raw_vectors: Vec::new(),
                    split_threshold: None,
                })
            }
        }
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
        self.id_to_index.insert(id, self.ids.len());
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
        if ids.len() < self.block_size * 2 {
            return Ok(());
        }

        let mut vectors = Vec::with_capacity(ids.len());
        for id in &ids {
            if let Some(&idx) = self.id_to_index.get(id) {
                vectors.push(self.raw_vectors[idx].clone());
            } else {
                return Err(format!("VectorId {:?} not found in raw_vectors", id));
            }
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

        Ok(())
    }

    pub fn stats(&self) -> IndexStats {
        IndexStats {
            dim: self.dim,
            grains: self.grains.len(),
            vectors: self.grains.iter().map(Grain::len).sum(),
        }
    }

    /// Extract the first (and typically only) grain as a `LegacySingleGrain`
    /// for serialisation. Returns `None` if no grain has been built yet.
    pub fn to_legacy_single(&self) -> Option<LegacySingleGrain> {
        let grain = self.grains.first()?;
        Some(self.legacy_single_for_grain(grain))
    }

    pub fn to_legacy_index(&self) -> Option<LegacyIndex> {
        if self.grains.len() <= 1 {
            return self.to_legacy_single().map(LegacyIndex::Single);
        }

        Some(LegacyIndex::Multi(LegacyMultiGrain {
            num_centroids: self.centroids.len() as u32,
            num_vectors: self.grains.iter().map(Grain::len).sum::<usize>() as u32,
            dimension: self.dim as u32,
            local_dim: self.local_dim as u32,
            block_size: self.block_size as u32,
            sketch_dim: self.sketch_dim as u32,
            centroids: self
                .centroids
                .iter()
                .flat_map(|centroid| centroid.iter().copied())
                .collect(),
            grains: self
                .grains
                .iter()
                .map(|grain| self.legacy_single_for_grain(grain))
                .collect(),
        }))
    }

    fn legacy_single_for_grain(&self, grain: &Grain) -> LegacySingleGrain {
        LegacySingleGrain {
            num_vectors: grain.len() as u32,
            dimension: self.dim as u32,
            local_dim: grain.local_dim() as u32,
            block_size: grain.block_size() as u32,
            sketch_dim: grain.sketch_dim() as u32,
            mean: grain.mean().to_vec(),
            projection: grain.projection().to_vec(),
            proj_scales: grain.proj_scales().to_vec(),
            residual_scale: grain.residual_scale(),
            sketch_projection: grain.sketch_projection().to_vec(),
            sketch_scales: grain.sketch_scales().to_vec(),
            block_data: grain.raw_block_bytes(),
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
    fn exports_multi_grain_legacy_index() {
        let mut index = AperonIndex::with_options(2, 2, 0, 4);
        index.insert(1, [0.0, 0.0]).unwrap();
        index.insert(2, [1.0, 0.0]).unwrap();
        index.insert(10, [100.0, 100.0]).unwrap();
        index.insert(11, [101.0, 100.0]).unwrap();
        index.rebuild_two_grains().unwrap();

        let legacy = index.to_legacy_index().unwrap();
        let loaded = AperonIndex::from_legacy_index(legacy).unwrap();

        let stats = loaded.stats();
        assert_eq!(stats.grains, 2);
        assert_eq!(stats.vectors, 4);
        let results = loaded.search_with_nprobe(&[100.2, 100.0], 1, 1).unwrap();
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
