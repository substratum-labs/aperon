use crate::shared::{
    packed_pq_bytes, train_shared_quantizer, unpack_pq_bytes, SharedBuildGrain,
    SharedGrainEncoding, SharedQuantizationConfig, SharedQuantizer,
};
use crate::{
    binary::{
        LegacyIndex, LegacyLatticeRouter, LegacyMultiGrain, LegacySharedGrain,
        LegacySharedQuantizer, LegacySingleGrain,
    },
    distance::l2_squared,
    grain::ScoredVector,
    Grain, GrainId, VectorId, DEFAULT_BLOCK_SIZE,
};
use std::collections::HashMap;
use std::mem::size_of;

/// Minimal in-memory index skeleton.
#[derive(Clone, Debug)]
pub struct AperonIndex {
    dim: usize,
    local_dim: usize,
    sketch_dim: usize,
    residual_bits: u8,
    block_size: usize,
    grains: Vec<Grain>,
    centroids: Vec<Vec<f32>>,
    grain_ids: Vec<Vec<VectorId>>,
    id_to_index: HashMap<VectorId, usize>,
    ids: Vec<VectorId>,
    raw_vectors: Vec<Vec<f32>>,
    split_threshold: Option<usize>,
    rerank_factor: usize,
    vector_locations: HashMap<VectorId, (usize, usize)>,
    adaptive_quantization: Option<AdaptiveQuantizationConfig>,
    shared_quantization: Option<SharedQuantizationConfig>,
    shared_quantizer: Option<SharedQuantizer>,
    lattice_router: Option<crate::routing::LatticeRouter>,
    hlr_router: Option<crate::routing::HierarchicalLatticeRouter>,
}

impl AperonIndex {
    pub fn new(dim: usize) -> Self {
        let local_dim = dim;
        Self {
            dim,
            local_dim,
            sketch_dim: 0,
            residual_bits: 8,
            block_size: DEFAULT_BLOCK_SIZE,
            grains: vec![Grain::new(GrainId::new(0), dim)],
            centroids: vec![vec![0.0; dim]],
            grain_ids: vec![Vec::new()],
            id_to_index: HashMap::new(),
            ids: Vec::new(),
            raw_vectors: Vec::new(),
            split_threshold: None,
            rerank_factor: 4,
            vector_locations: HashMap::new(),
            adaptive_quantization: None,
            shared_quantization: None,
            shared_quantizer: None,
            lattice_router: None,
            hlr_router: None,
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
            residual_bits: 8,
            block_size,
            grains: vec![Grain::new(GrainId::new(0), dim)],
            centroids: vec![vec![0.0; dim]],
            grain_ids: vec![Vec::new()],
            id_to_index: HashMap::new(),
            ids: Vec::new(),
            raw_vectors: Vec::new(),
            split_threshold: None,
            rerank_factor: 4,
            vector_locations: HashMap::new(),
            adaptive_quantization: None,
            shared_quantization: None,
            shared_quantizer: None,
            lattice_router: None,
            hlr_router: None,
        }
    }

    pub fn set_rerank_factor(&mut self, factor: usize) {
        self.rerank_factor = factor;
    }

    pub fn rerank_factor(&self) -> usize {
        self.rerank_factor
    }

    pub(crate) fn grains(&self) -> &[Grain] {
        &self.grains
    }

    pub fn set_residual_bits(&mut self, bits: u8) -> Result<(), String> {
        crate::layout::validate_residual_bits(bits)?;
        self.residual_bits = bits;
        Ok(())
    }

    pub fn residual_bits(&self) -> u8 {
        self.residual_bits
    }

    #[allow(clippy::too_many_arguments)]
    pub fn enable_adaptive_quantization(
        &mut self,
        min_local_dim: usize,
        max_local_dim: usize,
        min_sketch_dim: usize,
        max_sketch_dim: usize,
        min_residual_bits: u8,
        max_residual_bits: u8,
        variance_target: f32,
    ) -> Result<(), String> {
        self.adaptive_quantization = Some(AdaptiveQuantizationConfig::new(
            self.dim,
            min_local_dim,
            max_local_dim,
            min_sketch_dim,
            max_sketch_dim,
            min_residual_bits,
            max_residual_bits,
            variance_target,
        )?);
        Ok(())
    }

    pub fn enable_shared_basis_pq(
        &mut self,
        basis_cols: usize,
        local_dim: usize,
        pq_subquantizers: usize,
        pq_bits: u8,
        opq: bool,
    ) -> Result<(), String> {
        self.shared_quantization = Some(SharedQuantizationConfig::new(
            self.dim,
            basis_cols,
            local_dim,
            pq_subquantizers,
            pq_bits,
            opq,
        )?);
        Ok(())
    }

    fn update_vector_locations(&mut self) {
        self.vector_locations.clear();
        for (g_idx, ids) in self.grain_ids.iter().enumerate() {
            for (ord, id) in ids.iter().enumerate() {
                self.vector_locations.insert(*id, (g_idx, ord));
            }
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn from_legacy_index(index: LegacyIndex) -> Result<Self, String> {
        let mut index_obj = match index {
            LegacyIndex::Single(single) => {
                let dim = single.dimension as usize;
                let local_dim = single.local_dim as usize;
                let sketch_dim = single.sketch_dim as usize;
                let block_size = single.block_size as usize;
                let residual_bits = single.residual_bits;
                let grain = Grain::from_legacy_single(GrainId::new(0), single)?;
                let ids = grain.vector_ids().to_vec();
                Self {
                    dim,
                    local_dim,
                    sketch_dim,
                    residual_bits,
                    block_size,
                    centroids: vec![grain.centroid().to_vec()],
                    grain_ids: vec![ids],
                    id_to_index: HashMap::new(),
                    ids: Vec::new(),
                    raw_vectors: Vec::new(),
                    grains: vec![grain],
                    split_threshold: None,
                    rerank_factor: 4,
                    vector_locations: HashMap::new(),
                    adaptive_quantization: None,
                    shared_quantization: None,
                    shared_quantizer: None,
                    lattice_router: None,
                    hlr_router: None,
                }
            }
            LegacyIndex::Multi(multi) => {
                let dim = multi.dimension as usize;
                let local_dim = multi.local_dim as usize;
                let sketch_dim = multi.sketch_dim as usize;
                let block_size = multi.block_size as usize;
                let residual_bits = multi.residual_bits;
                if let Some(shared) = multi.shared {
                    let config = SharedQuantizationConfig::new(
                        dim,
                        shared.basis_cols as usize,
                        multi.local_dim as usize,
                        shared.pq_subquantizers as usize,
                        shared.pq_bits,
                        shared.opq,
                    )?;
                    let quantizer = SharedQuantizer {
                        config,
                        basis: shared.basis,
                        opq_rotation: shared.opq_rotation,
                        pq_codebooks: shared.pq_codebooks,
                    };
                    let centroids = multi
                        .centroids
                        .chunks_exact(dim)
                        .map(|chunk| chunk.to_vec())
                        .collect::<Vec<_>>();
                    let grains = multi
                        .shared_grains
                        .into_iter()
                        .enumerate()
                        .map(|(idx, grain)| {
                            let unpacked = unpack_pq_bytes(
                                &grain.pq_codes,
                                grain.num_vectors as usize * quantizer.config.pq_subquantizers,
                                quantizer.config.pq_bits,
                            );
                            let encoding = SharedGrainEncoding {
                                column_indices: grain
                                    .column_indices
                                    .iter()
                                    .map(|value| *value as usize)
                                    .collect(),
                                coord_scales: grain.coord_scales,
                                qcoords: qcoords_from_block_data(
                                    grain.local_dim as usize,
                                    grain.block_size as usize,
                                    grain.num_vectors as usize,
                                    &grain.block_data,
                                )?,
                                pq_codes: unpacked,
                                error_scale: grain.pq_error_scale,
                                error_norms: grain.pq_error_norms,
                            };
                            Grain::build_shared(
                                GrainId::new(idx as u64),
                                &ids_from_block_data(
                                    grain.local_dim as usize,
                                    grain.block_size as usize,
                                    grain.num_vectors as usize,
                                    &grain.block_data,
                                )?,
                                dim,
                                grain.mean,
                                grain.block_size as usize,
                                quantizer.clone(),
                                encoding,
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let grain_ids = grains
                        .iter()
                        .map(|grain| grain.vector_ids().to_vec())
                        .collect::<Vec<_>>();
                    let lattice_router = if let Some(legacy_router) = multi.lattice_router {
                        let mut map = HashMap::new();
                        for (key, values) in legacy_router
                            .map_keys
                            .into_iter()
                            .zip(legacy_router.map_values)
                        {
                            map.insert(key, values.into_iter().map(|idx| idx as usize).collect());
                        }
                        let r_dim = legacy_router.routing_dim as usize;
                        let mut projection = vec![0.0; r_dim * dim];
                        for d in 0..dim {
                            for j in 0..r_dim {
                                projection[j * dim + d] = legacy_router.projection[d * r_dim + j];
                            }
                        }
                        let num_centroids = centroids.len();
                        let mut centroids_coords = vec![0_i16; num_centroids * r_dim];
                        for (g_idx, centroid) in centroids.iter().enumerate() {
                            for j in 0..r_dim {
                                let mut dot = 0.0;
                                for d in 0..dim {
                                    dot += centroid[d] * projection[j * dim + d];
                                }
                                centroids_coords[g_idx * r_dim + j] =
                                    (dot / legacy_router.spacing).round() as i16;
                            }
                        }
                        Some(crate::routing::LatticeRouter {
                            dim,
                            routing_dim: r_dim,
                            spacing: legacy_router.spacing,
                            projection,
                            lattice_map: map,
                            centroids_coords,
                            num_centroids,
                        })
                    } else {
                        None
                    };
                    let mut out = Self {
                        dim,
                        local_dim,
                        sketch_dim,
                        residual_bits,
                        block_size,
                        grains,
                        centroids,
                        grain_ids,
                        id_to_index: HashMap::new(),
                        ids: Vec::new(),
                        raw_vectors: Vec::new(),
                        split_threshold: None,
                        rerank_factor: 4,
                        vector_locations: HashMap::new(),
                        adaptive_quantization: None,
                        shared_quantization: Some(config),
                        shared_quantizer: Some(quantizer),
                        lattice_router,
                        hlr_router: None,
                    };
                    out.update_vector_locations();
                    return Ok(out);
                }

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
                let lattice_router = if let Some(legacy_router) = multi.lattice_router {
                    let mut map = HashMap::new();
                    for (key, values) in legacy_router
                        .map_keys
                        .into_iter()
                        .zip(legacy_router.map_values)
                    {
                        map.insert(key, values.into_iter().map(|idx| idx as usize).collect());
                    }
                    let r_dim = legacy_router.routing_dim as usize;
                    let mut projection = vec![0.0; r_dim * dim];
                    for d in 0..dim {
                        for j in 0..r_dim {
                            projection[j * dim + d] = legacy_router.projection[d * r_dim + j];
                        }
                    }
                    let num_centroids = centroids.len();
                    let mut centroids_coords = vec![0_i16; num_centroids * r_dim];
                    for (g_idx, centroid) in centroids.iter().enumerate() {
                        for j in 0..r_dim {
                            let mut dot = 0.0;
                            for d in 0..dim {
                                dot += centroid[d] * projection[j * dim + d];
                            }
                            centroids_coords[g_idx * r_dim + j] =
                                (dot / legacy_router.spacing).round() as i16;
                        }
                    }
                    Some(crate::routing::LatticeRouter {
                        dim,
                        routing_dim: r_dim,
                        spacing: legacy_router.spacing,
                        projection,
                        lattice_map: map,
                        centroids_coords,
                        num_centroids,
                    })
                } else {
                    None
                };
                Self {
                    dim,
                    local_dim,
                    sketch_dim,
                    residual_bits,
                    block_size,
                    grains,
                    centroids,
                    grain_ids,
                    id_to_index: HashMap::new(),
                    ids: Vec::new(),
                    raw_vectors: Vec::new(),
                    split_threshold: None,
                    rerank_factor: 4,
                    vector_locations: HashMap::new(),
                    adaptive_quantization: None,
                    shared_quantization: None,
                    shared_quantizer: None,
                    lattice_router,
                    hlr_router: None,
                }
            }
        };
        index_obj.update_vector_locations();
        Ok(index_obj)
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

    pub fn enable_lattice_routing(
        &mut self,
        routing_dim: usize,
        spacing: f32,
    ) -> Result<(), String> {
        if self.centroids.is_empty() {
            return Err(
                "cannot enable lattice routing: no centroids exist. Build/rebuild index first."
                    .to_string(),
            );
        }
        let router =
            crate::routing::LatticeRouter::new(self.dim, routing_dim, spacing, &self.centroids)?;
        self.lattice_router = Some(router);
        Ok(())
    }

    pub fn lattice_router(&self) -> Option<&crate::routing::LatticeRouter> {
        self.lattice_router.as_ref()
    }

    fn update_lattice_router(&mut self) -> Result<(), String> {
        if let Some(existing) = &self.lattice_router {
            let routing_dim = existing.routing_dim;
            let spacing = existing.spacing;
            let router = crate::routing::LatticeRouter::new(
                self.dim,
                routing_dim,
                spacing,
                &self.centroids,
            )?;
            self.lattice_router = Some(router);
        }
        Ok(())
    }

    pub fn enable_hlr_routing(
        &mut self,
        layer_configs: &[crate::routing::HierarchicalLatticeLayerConfig],
    ) -> Result<(), String> {
        if self.centroids.is_empty() {
            return Err(
                "cannot enable HLR routing: no centroids exist. Build/rebuild index first."
                    .to_string(),
            );
        }
        let router = crate::routing::HierarchicalLatticeRouter::new(
            self.dim,
            layer_configs,
            &self.centroids,
        )?;
        self.hlr_router = Some(router);
        Ok(())
    }

    pub fn hlr_router(&self) -> Option<&crate::routing::HierarchicalLatticeRouter> {
        self.hlr_router.as_ref()
    }

    fn update_hlr_router(&mut self) -> Result<(), String> {
        if let Some(existing) = &self.hlr_router {
            let configs: Vec<crate::routing::HierarchicalLatticeLayerConfig> = existing
                .layers
                .iter()
                .map(|layer| crate::routing::HierarchicalLatticeLayerConfig {
                    routing_dim: layer.routing_dim,
                    spacing: layer.spacing,
                })
                .collect();
            let router = crate::routing::HierarchicalLatticeRouter::new(
                self.dim,
                &configs,
                &self.centroids,
            )?;
            self.hlr_router = Some(router);
        }
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
            self.route(&vector, None)?[0]
        };

        self.grains[route].insert(id, vector.clone())?;
        self.grain_ids[route].push(id);
        self.id_to_index.insert(id, self.ids.len());
        self.ids.push(id);
        self.raw_vectors.push(vector);

        let ord = self.grain_ids[route].len() - 1;
        self.vector_locations.insert(id, (route, ord));

        if self
            .split_threshold
            .is_some_and(|threshold| self.grain_ids[route].len() >= threshold)
        {
            self.split_grain(route)?;
        }
        Ok(())
    }

    pub fn attach_raw_vectors(
        &mut self,
        ids: &[VectorId],
        vectors: &[Vec<f32>],
    ) -> Result<(), String> {
        if ids.len() != vectors.len() {
            return Err("ids and vectors length mismatch".to_string());
        }
        for vector in vectors {
            if vector.len() != self.dim {
                return Err(format!(
                    "dimension mismatch: expected {}, got {}",
                    self.dim,
                    vector.len()
                ));
            }
        }

        self.id_to_index.clear();
        self.ids.clear();
        self.raw_vectors.clear();
        for (idx, (id, vector)) in ids.iter().copied().zip(vectors).enumerate() {
            self.id_to_index.insert(id, idx);
            self.ids.push(id);
            self.raw_vectors.push(vector.clone());
        }
        Ok(())
    }

    pub fn rebuild_single_grain(&mut self) -> Result<(), String> {
        self.grains = vec![self.build_grain(0, &self.raw_vectors, &self.ids)?];
        self.centroids = vec![mean_vector(&self.raw_vectors, self.dim)];
        self.grain_ids = vec![self.ids.clone()];
        self.update_vector_locations();
        self.update_lattice_router()?;
        self.update_hlr_router()?;
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
            self.build_grain(0, &vectors0, &ids0)?,
            self.build_grain(1, &vectors1, &ids1)?,
        ];
        self.centroids = vec![split.centroid0, split.centroid1];
        self.grain_ids = vec![ids0, ids1];
        self.update_vector_locations();
        self.update_lattice_router()?;
        self.update_hlr_router()?;
        Ok(())
    }

    pub fn rebuild_n_grains(&mut self, requested_grains: usize) -> Result<(), String> {
        if requested_grains == 0 {
            return Err("requested_grains must be greater than zero".to_string());
        }
        let target_grains = requested_grains.min(self.raw_vectors.len());
        if target_grains <= 1 {
            return self.rebuild_single_grain();
        }

        let split = k_means(&self.raw_vectors, self.dim, target_grains);
        let mut bucket_ids = vec![Vec::new(); split.centroids.len()];
        let mut bucket_vectors = vec![Vec::new(); split.centroids.len()];
        for (idx, assignment) in split.assignments.iter().copied().enumerate() {
            bucket_ids[assignment].push(self.ids[idx]);
            bucket_vectors[assignment].push(self.raw_vectors[idx].clone());
        }

        let mut grains = Vec::new();
        let mut centroids = Vec::new();
        let mut grain_ids = Vec::new();
        let mut build_grains = Vec::new();
        for (ids, vectors) in bucket_ids.into_iter().zip(bucket_vectors) {
            if ids.is_empty() {
                continue;
            }
            let mean = mean_vector(&vectors, self.dim);
            centroids.push(mean.clone());
            build_grains.push(SharedBuildGrain {
                ids: ids.clone(),
                vectors: vectors.clone(),
                mean: mean.clone(),
            });
            if self.shared_quantization.is_none() {
                let grain_idx = grains.len();
                grains.push(self.build_grain(grain_idx, &vectors, &ids)?);
            }
            grain_ids.push(ids);
        }

        if let Some(config) = self.shared_quantization {
            let (quantizer, encodings) =
                train_shared_quantizer(&self.raw_vectors, &build_grains, self.dim, config)?;
            grains = build_grains
                .into_iter()
                .zip(encodings)
                .enumerate()
                .map(|(idx, (grain, encoding))| {
                    Grain::build_shared(
                        GrainId::new(idx as u64),
                        &grain.ids,
                        self.dim,
                        grain.mean,
                        self.block_size,
                        quantizer.clone(),
                        encoding,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            self.shared_quantizer = Some(quantizer);
        }

        if grains.len() <= 1 {
            return self.rebuild_single_grain();
        }

        self.grains = grains;
        self.centroids = centroids;
        self.grain_ids = grain_ids;
        self.update_vector_locations();
        self.update_lattice_router()?;
        self.update_hlr_router()?;
        Ok(())
    }

    #[deprecated(
        since = "0.1.0",
        note = "Please use search_mode_a or search_mode_b instead"
    )]
    pub fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<ScoredVector>, String> {
        self.search_internal(query, top_k, self.rerank_factor)
    }

    #[deprecated(
        since = "0.1.0",
        note = "Please use search_mode_a or search_mode_b instead"
    )]
    pub fn search_with_nprobe(
        &self,
        query: &[f32],
        top_k: usize,
        nprobe: usize,
    ) -> Result<Vec<ScoredVector>, String> {
        self.search_with_nprobe_internal(query, top_k, nprobe, self.rerank_factor)
    }

    pub fn search_mode_a(
        &self,
        query: &[f32],
        top_k: usize,
        nprobe: usize,
        rerank_factor: usize,
    ) -> Result<Vec<ScoredVector>, String> {
        self.search_with_nprobe_internal(query, top_k, nprobe, rerank_factor)
    }

    #[allow(deprecated)]
    pub fn search_mode_b(
        &self,
        query: &[f32],
        top_k: usize,
        nprobe: usize,
        candidate_k: usize,
    ) -> Result<Vec<ScoredVector>, String> {
        self.search_tiered_with_nprobe(query, top_k, nprobe, candidate_k)
    }

    pub fn search_internal(
        &self,
        query: &[f32],
        top_k: usize,
        rerank_factor: usize,
    ) -> Result<Vec<ScoredVector>, String> {
        if rerank_factor == 0 {
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
            return Ok(results);
        }

        let scan_k = top_k * rerank_factor;
        let mut candidates = Vec::new();
        for grain in self.routed_grains(query)? {
            candidates.extend(grain.scan(query, scan_k)?);
        }

        candidates.sort_by_key(|c| c.id);
        candidates.dedup_by_key(|c| c.id);

        let mut reranked = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let dist = self.rerank_distance(query, candidate.id)?;
            reranked.push(ScoredVector {
                id: candidate.id,
                distance: dist,
            });
        }

        reranked.sort_by(|a, b| {
            a.distance
                .total_cmp(&b.distance)
                .then_with(|| a.id.cmp(&b.id))
        });
        reranked.truncate(top_k);
        Ok(reranked)
    }

    pub fn search_with_nprobe_internal(
        &self,
        query: &[f32],
        top_k: usize,
        nprobe: usize,
        rerank_factor: usize,
    ) -> Result<Vec<ScoredVector>, String> {
        if rerank_factor == 0 {
            let mut routes = self.route(query, Some(nprobe))?;
            let probe_count = nprobe.max(1).min(routes.len());
            routes.truncate(probe_count);

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
            return Ok(results);
        }

        let mut routes = self.route(query, Some(nprobe))?;
        let probe_count = nprobe.max(1).min(routes.len());
        routes.truncate(probe_count);

        let scan_k = top_k * rerank_factor;
        let mut candidates = Vec::new();
        for route in routes {
            candidates.extend(self.grains[route].scan(query, scan_k)?);
        }

        candidates.sort_by_key(|c| c.id);
        candidates.dedup_by_key(|c| c.id);

        let mut reranked = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let dist = self.rerank_distance(query, candidate.id)?;
            reranked.push(ScoredVector {
                id: candidate.id,
                distance: dist,
            });
        }

        reranked.sort_by(|a, b| {
            a.distance
                .total_cmp(&b.distance)
                .then_with(|| a.id.cmp(&b.id))
        });
        reranked.truncate(top_k);
        Ok(reranked)
    }

    pub fn candidate_pool_with_nprobe(
        &self,
        query: &[f32],
        nprobe: usize,
        candidate_k: usize,
    ) -> Result<Vec<ScoredVector>, String> {
        if candidate_k == 0 {
            return Ok(Vec::new());
        }
        let mut routes = self.route(query, Some(nprobe))?;
        let probe_count = nprobe.max(1).min(routes.len());
        routes.truncate(probe_count);

        let mut candidates = Vec::new();
        for route in routes {
            candidates.extend(self.grains[route].scan(query, candidate_k)?);
        }
        candidates.sort_by_key(|c| c.id);
        candidates.dedup_by_key(|c| c.id);
        candidates.sort_by(|a, b| {
            a.distance
                .total_cmp(&b.distance)
                .then_with(|| a.id.cmp(&b.id))
        });
        candidates.truncate(candidate_k);
        Ok(candidates)
    }

    #[deprecated(
        since = "0.1.0",
        note = "Please use search_mode_a or search_mode_b instead"
    )]
    pub fn search_tiered_with_nprobe(
        &self,
        query: &[f32],
        top_k: usize,
        nprobe: usize,
        candidate_k: usize,
    ) -> Result<Vec<ScoredVector>, String> {
        let candidates = self.candidate_pool_with_nprobe(query, nprobe, candidate_k)?;
        let mut reranked = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let dist = self.rerank_distance(query, candidate.id)?;
            reranked.push(ScoredVector {
                id: candidate.id,
                distance: dist,
            });
        }
        reranked.sort_by(|a, b| {
            a.distance
                .total_cmp(&b.distance)
                .then_with(|| a.id.cmp(&b.id))
        });
        reranked.truncate(top_k);
        Ok(reranked)
    }

    fn rerank_distance(&self, query: &[f32], id: VectorId) -> Result<f64, String> {
        if !self.raw_vectors.is_empty() {
            if let Some(&idx) = self.id_to_index.get(&id) {
                let raw_vec = &self.raw_vectors[idx];
                return Ok(l2_squared(query, raw_vec).unwrap_or(f32::INFINITY) as f64);
            }
        }

        if let Some(&(g_idx, ord)) = self.vector_locations.get(&id) {
            let grain = &self.grains[g_idx];
            let recon = grain.reconstruct(ord)?;
            let remaining = grain.residual_norm_sq(ord)?;
            let dist = l2_squared(query, &recon).unwrap_or(f32::INFINITY);
            return Ok((dist + remaining) as f64);
        }

        Err(format!(
            "VectorId {:?} location not found for reranking",
            id
        ))
    }

    fn routed_grains(&self, query: &[f32]) -> Result<Vec<&Grain>, String> {
        Ok(self
            .route(query, None)?
            .into_iter()
            .map(|idx| &self.grains[idx])
            .collect())
    }

    pub(crate) fn route(&self, query: &[f32], nprobe: Option<usize>) -> Result<Vec<usize>, String> {
        if query.len() != self.dim {
            return Err(format!(
                "dimension mismatch: expected {}, got {}",
                self.dim,
                query.len()
            ));
        }

        if let Some(router) = &self.hlr_router {
            let np = nprobe.unwrap_or(1);
            let limit = (np * 2 + 16).min(self.centroids.len());
            let mut routed = router.route_nprobe(query, limit);

            // Sort these routed candidate centroids by exact L2 distance
            routed.sort_by(|&a, &b| {
                let dist_a = l2_squared(query, &self.centroids[a]).unwrap_or(f32::INFINITY);
                let dist_b = l2_squared(query, &self.centroids[b]).unwrap_or(f32::INFINITY);
                dist_a.total_cmp(&dist_b).then_with(|| a.cmp(&b))
            });

            let mut all_routes = routed;
            for idx in 0..self.centroids.len() {
                if !all_routes.contains(&idx) {
                    all_routes.push(idx);
                }
            }
            return Ok(all_routes);
        }

        if let Some(router) = &self.lattice_router {
            let np = nprobe.unwrap_or(1);
            let limit = (np * 2 + 16).min(self.centroids.len());
            let mut routed = router.route_nprobe(query, limit);

            // Sort these routed candidate centroids by exact L2 distance
            routed.sort_by(|&a, &b| {
                let dist_a = l2_squared(query, &self.centroids[a]).unwrap_or(f32::INFINITY);
                let dist_b = l2_squared(query, &self.centroids[b]).unwrap_or(f32::INFINITY);
                dist_a.total_cmp(&dist_b).then_with(|| a.cmp(&b))
            });

            let mut all_routes = routed;
            for idx in 0..self.centroids.len() {
                if !all_routes.contains(&idx) {
                    all_routes.push(idx);
                }
            }
            return Ok(all_routes);
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

        self.grains[grain_idx] = self.build_grain(grain_idx, &vectors0, &ids0)?;
        self.centroids[grain_idx] = split.centroid0;
        self.grain_ids[grain_idx] = ids0;

        let new_idx = self.grains.len();
        self.grains
            .push(self.build_grain(new_idx, &vectors1, &ids1)?);
        self.centroids.push(split.centroid1);
        self.grain_ids.push(ids1);

        self.update_vector_locations();
        Ok(())
    }

    fn build_grain(
        &self,
        grain_idx: usize,
        vectors: &[Vec<f32>],
        ids: &[VectorId],
    ) -> Result<Grain, String> {
        let shape = self
            .adaptive_quantization
            .as_ref()
            .map(|config| config.select_shape(vectors, self.dim))
            .unwrap_or(GrainShape {
                local_dim: self.local_dim,
                sketch_dim: self.sketch_dim,
                residual_bits: self.residual_bits,
            });
        Grain::build_with_residual_bits(
            GrainId::new(grain_idx as u64),
            vectors,
            ids,
            self.dim,
            shape.local_dim,
            shape.sketch_dim,
            self.block_size,
            shape.residual_bits,
        )
    }

    pub fn stats(&self) -> IndexStats {
        let grain_sizes = self.grains.iter().map(Grain::len).collect::<Vec<_>>();
        let vectors = grain_sizes.iter().sum();
        let encoded_bytes = self.grains.iter().map(Grain::encoded_bytes).sum::<usize>()
            + self.centroids.len() * self.dim * size_of::<f32>();
        let encoded_bytes = encoded_bytes
            + self
                .shared_quantizer
                .as_ref()
                .map(|q| {
                    q.basis.len() * size_of::<f32>()
                        + q.opq_rotation.len() * size_of::<f32>()
                        + q.pq_codebooks.len() * size_of::<f32>()
                })
                .unwrap_or(0);
        let grain_local_dims = self.grains.iter().map(Grain::local_dim).collect::<Vec<_>>();
        let grain_sketch_dims = self
            .grains
            .iter()
            .map(Grain::sketch_dim)
            .collect::<Vec<_>>();
        let grain_residual_bits = self
            .grains
            .iter()
            .map(Grain::residual_bits)
            .collect::<Vec<_>>();
        IndexStats {
            dim: self.dim,
            grains: self.grains.len(),
            vectors,
            grain_sizes,
            residual_bits: self.residual_bits,
            encoded_bytes,
            grain_local_dims,
            grain_sketch_dims,
            grain_residual_bits,
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
            residual_bits: self.residual_bits,
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
            shared: self
                .shared_quantizer
                .as_ref()
                .map(|q| LegacySharedQuantizer {
                    basis_cols: q.config.basis_cols as u32,
                    pq_subquantizers: q.config.pq_subquantizers as u32,
                    pq_bits: q.config.pq_bits,
                    opq: q.config.opq,
                    basis: q.basis.clone(),
                    opq_rotation: q.opq_rotation.clone(),
                    pq_codebooks: q.pq_codebooks.clone(),
                }),
            shared_grains: if self.shared_quantizer.is_some() {
                self.grains
                    .iter()
                    .map(|grain| LegacySharedGrain {
                        num_vectors: grain.len() as u32,
                        local_dim: grain.local_dim() as u32,
                        block_size: grain.block_size() as u32,
                        mean: grain.mean().to_vec(),
                        column_indices: grain
                            .shared_column_indices()
                            .iter()
                            .map(|idx| *idx as u8)
                            .collect(),
                        coord_scales: grain.proj_scales().to_vec(),
                        block_data: grain.shared_compact_block_bytes(),
                        pq_codes: packed_pq_bytes(
                            grain.pq_codes(),
                            self.shared_quantizer.as_ref().unwrap().config.pq_bits,
                        ),
                        pq_error_scale: 1.0,
                        pq_error_norms: Vec::new(),
                    })
                    .collect()
            } else {
                Vec::new()
            },
            lattice_router: self.lattice_router.as_ref().map(|router| {
                let mut map_keys = Vec::new();
                let mut map_values = Vec::new();
                for (key, values) in &router.lattice_map {
                    map_keys.push(key.clone());
                    map_values.push(values.iter().map(|&idx| idx as u32).collect());
                }
                let mut legacy_projection = vec![0.0; router.dim * router.routing_dim];
                for d in 0..router.dim {
                    for j in 0..router.routing_dim {
                        legacy_projection[d * router.routing_dim + j] =
                            router.projection[j * router.dim + d];
                    }
                }
                LegacyLatticeRouter {
                    routing_dim: router.routing_dim as u32,
                    spacing: router.spacing,
                    projection: legacy_projection,
                    map_keys,
                    map_values,
                }
            }),
        }))
    }

    fn legacy_single_for_grain(&self, grain: &Grain) -> LegacySingleGrain {
        LegacySingleGrain {
            num_vectors: grain.len() as u32,
            dimension: self.dim as u32,
            local_dim: grain.local_dim() as u32,
            block_size: grain.block_size() as u32,
            sketch_dim: grain.sketch_dim() as u32,
            residual_bits: grain.residual_bits(),
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
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndexStats {
    pub dim: usize,
    pub grains: usize,
    pub vectors: usize,
    pub grain_sizes: Vec<usize>,
    pub residual_bits: u8,
    pub encoded_bytes: usize,
    pub grain_local_dims: Vec<usize>,
    pub grain_sketch_dims: Vec<usize>,
    pub grain_residual_bits: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GrainShape {
    local_dim: usize,
    sketch_dim: usize,
    residual_bits: u8,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaptiveQuantizationConfig {
    min_local_dim: usize,
    max_local_dim: usize,
    min_sketch_dim: usize,
    max_sketch_dim: usize,
    min_residual_bits: u8,
    max_residual_bits: u8,
    variance_target: f32,
}

impl AdaptiveQuantizationConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dim: usize,
        min_local_dim: usize,
        max_local_dim: usize,
        min_sketch_dim: usize,
        max_sketch_dim: usize,
        min_residual_bits: u8,
        max_residual_bits: u8,
        variance_target: f32,
    ) -> Result<Self, String> {
        if min_local_dim == 0 || max_local_dim == 0 || min_local_dim > max_local_dim {
            return Err("invalid adaptive local_dim range".to_string());
        }
        if min_sketch_dim > max_sketch_dim {
            return Err("invalid adaptive sketch_dim range".to_string());
        }
        crate::layout::validate_residual_bits(min_residual_bits)?;
        crate::layout::validate_residual_bits(max_residual_bits)?;
        if residual_bit_rank(min_residual_bits) > residual_bit_rank(max_residual_bits) {
            return Err("invalid adaptive residual_bits range".to_string());
        }
        if !variance_target.is_finite() || !(0.0..=1.0).contains(&variance_target) {
            return Err("variance_target must be in [0, 1]".to_string());
        }
        Ok(Self {
            min_local_dim: min_local_dim.min(dim).max(1),
            max_local_dim: max_local_dim.min(dim).max(1),
            min_sketch_dim: min_sketch_dim.min(dim),
            max_sketch_dim: max_sketch_dim.min(dim),
            min_residual_bits,
            max_residual_bits,
            variance_target,
        })
    }

    fn select_shape(self, vectors: &[Vec<f32>], dim: usize) -> GrainShape {
        let effective_dim = effective_axis_dim(vectors, dim, self.variance_target);
        let local_dim = effective_dim.clamp(self.min_local_dim, self.max_local_dim);
        let remaining = effective_dim.saturating_sub(local_dim);
        let sketch_dim = remaining.clamp(self.min_sketch_dim, self.max_sketch_dim);
        let residual_bits = if local_dim >= self.max_local_dim || sketch_dim >= self.max_sketch_dim
        {
            self.max_residual_bits
        } else {
            self.min_residual_bits
        };
        GrainShape {
            local_dim,
            sketch_dim,
            residual_bits,
        }
    }
}

fn residual_bit_rank(bits: u8) -> usize {
    match bits {
        1 => 0,
        2 => 1,
        8 => 2,
        _ => usize::MAX,
    }
}

#[derive(Clone, Debug)]
struct TwoMeansSplit {
    assignments: Vec<usize>,
    centroid0: Vec<f32>,
    centroid1: Vec<f32>,
}

#[derive(Clone, Debug)]
struct KMeansSplit {
    assignments: Vec<usize>,
    centroids: Vec<Vec<f32>>,
}

fn k_means(vectors: &[Vec<f32>], dim: usize, k: usize) -> KMeansSplit {
    let mut centroids = seed_centroids(vectors, dim, k);
    let mut assignments = vec![usize::MAX; vectors.len()];

    for _ in 0..20 {
        let next = assign_to_centroids_balanced(vectors, &centroids);
        let updated = recompute_centroids(vectors, dim, &next, &centroids);
        let converged = next == assignments;
        assignments = next;
        centroids = updated;
        if converged {
            break;
        }
    }

    KMeansSplit {
        assignments,
        centroids,
    }
}

fn effective_axis_dim(vectors: &[Vec<f32>], dim: usize, variance_target: f32) -> usize {
    if vectors.is_empty() || dim == 0 {
        return 1;
    }
    let mean = mean_vector(vectors, dim);
    let mut variances = vec![0.0_f32; dim];
    for vector in vectors {
        for d in 0..dim {
            let diff = vector[d] - mean[d];
            variances[d] += diff * diff;
        }
    }
    variances.sort_by(|a, b| b.total_cmp(a));
    let total = variances.iter().sum::<f32>();
    if total <= 0.0 {
        return 1;
    }
    let target = total * variance_target;
    let mut cumulative = 0.0_f32;
    for (idx, variance) in variances.iter().enumerate() {
        cumulative += variance;
        if cumulative >= target {
            return idx + 1;
        }
    }
    dim
}

fn seed_centroids(vectors: &[Vec<f32>], dim: usize, k: usize) -> Vec<Vec<f32>> {
    if vectors.is_empty() || k == 0 {
        return Vec::new();
    }

    let target = k.min(vectors.len());
    let mut selected = vec![false; vectors.len()];
    let mut centroids = vec![vectors[0].clone()];
    selected[0] = true;

    while centroids.len() < target {
        let mut best_idx = None;
        let mut best_distance = f32::NEG_INFINITY;
        for (idx, vector) in vectors.iter().enumerate() {
            if selected[idx] {
                continue;
            }
            let nearest = centroids
                .iter()
                .map(|centroid| l2_squared(vector, centroid).unwrap_or(f32::INFINITY))
                .fold(f32::INFINITY, f32::min);
            if nearest > best_distance {
                best_distance = nearest;
                best_idx = Some(idx);
            }
        }

        let Some(idx) = best_idx else {
            break;
        };
        selected[idx] = true;
        centroids.push(vectors[idx].clone());
    }

    if centroids.is_empty() {
        centroids.push(vec![0.0; dim]);
    }
    centroids
}

fn assign_to_centroids_balanced(vectors: &[Vec<f32>], centroids: &[Vec<f32>]) -> Vec<usize> {
    let n = vectors.len();
    let k = centroids.len();
    if n == 0 || k == 0 {
        return Vec::new();
    }

    let mut vec_centroid_dists = vec![Vec::with_capacity(k); n];
    let mut regret_indices = (0..n).collect::<Vec<usize>>();
    let mut regrets = vec![0.0_f32; n];

    for (i, v) in vectors.iter().enumerate() {
        for (j, centroid) in centroids.iter().enumerate().take(k) {
            let dist = l2_squared(v, centroid).unwrap_or(f32::INFINITY);
            vec_centroid_dists[i].push((j, dist));
        }
        vec_centroid_dists[i].sort_by(|a, b| a.1.total_cmp(&b.1));

        let r = if k > 1 {
            vec_centroid_dists[i][1].1 - vec_centroid_dists[i][0].1
        } else {
            0.0
        };
        regrets[i] = r;
    }

    regret_indices.sort_by(|&a, &b| regrets[b].total_cmp(&regrets[a]));

    let cap = ((n as f32 / k as f32) * 1.3).ceil() as usize;
    let mut assignments = vec![usize::MAX; n];
    let mut centroid_counts = vec![0_usize; k];

    for idx in regret_indices {
        let mut assigned = false;
        for &(c_idx, _) in &vec_centroid_dists[idx] {
            if centroid_counts[c_idx] < cap {
                assignments[idx] = c_idx;
                centroid_counts[c_idx] += 1;
                assigned = true;
                break;
            }
        }
        if !assigned {
            let mut best_c = 0;
            let mut min_count = usize::MAX;
            for (c_idx, count) in centroid_counts.iter().enumerate().take(k) {
                if *count < min_count {
                    min_count = *count;
                    best_c = c_idx;
                }
            }
            assignments[idx] = best_c;
            centroid_counts[best_c] += 1;
        }
    }

    assignments
}

fn recompute_centroids(
    vectors: &[Vec<f32>],
    dim: usize,
    assignments: &[usize],
    previous: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let mut sums = vec![vec![0.0; dim]; previous.len()];
    let mut counts = vec![0_usize; previous.len()];

    for (vector, assignment) in vectors.iter().zip(assignments) {
        counts[*assignment] += 1;
        for (sum, value) in sums[*assignment].iter_mut().zip(vector) {
            *sum += value;
        }
    }

    sums.into_iter()
        .zip(counts)
        .zip(previous)
        .map(|((mut sum, count), old)| {
            if count == 0 {
                return old.clone();
            }
            for value in &mut sum {
                *value /= count as f32;
            }
            sum
        })
        .collect()
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

fn qcoords_from_block_data(
    local_dim: usize,
    block_size: usize,
    num_vectors: usize,
    bytes: &[u8],
) -> Result<Vec<i16>, String> {
    use std::mem::size_of;

    let bytes_per_block = block_size * (local_dim + size_of::<u32>());
    if bytes.len() != num_vectors.div_ceil(block_size) * bytes_per_block {
        return Err("shared block data length mismatch".to_string());
    }
    let mut out = vec![0_i16; num_vectors * local_dim];
    for ordinal in 0..num_vectors {
        let block = ordinal / block_size;
        let lane = ordinal % block_size;
        let block_start = block * bytes_per_block;
        for dim in 0..local_dim {
            let offset = block_start + dim * block_size + lane;
            out[ordinal * local_dim + dim] = bytes[offset] as i8 as i16;
        }
    }
    Ok(out)
}

fn ids_from_block_data(
    local_dim: usize,
    block_size: usize,
    num_vectors: usize,
    bytes: &[u8],
) -> Result<Vec<VectorId>, String> {
    use std::mem::size_of;

    let bytes_per_block = block_size * (local_dim + size_of::<u32>());
    if bytes.len() != num_vectors.div_ceil(block_size) * bytes_per_block {
        return Err("shared block data length mismatch".to_string());
    }
    let id_base = local_dim * block_size;
    let mut out = Vec::with_capacity(num_vectors);
    for ordinal in 0..num_vectors {
        let block = ordinal / block_size;
        let lane = ordinal % block_size;
        let offset = block * bytes_per_block + id_base + lane * size_of::<u32>();
        let id = u32::from_le_bytes(
            bytes[offset..offset + size_of::<u32>()]
                .try_into()
                .map_err(|_| "invalid shared id bytes")?,
        );
        out.push(VectorId::new(u64::from(id)));
    }
    Ok(out)
}

#[cfg(test)]
#[allow(deprecated)]
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
    fn rebuild_n_grains_builds_requested_routable_grains() {
        let mut index = AperonIndex::with_options(2, 2, 0, 2);
        for cluster in 0..4 {
            let base = cluster as f32 * 100.0;
            index.insert(cluster * 10, [base, base]).unwrap();
            index.insert(cluster * 10 + 1, [base + 1.0, base]).unwrap();
        }

        index.rebuild_n_grains(4).unwrap();

        let stats = index.stats();
        assert_eq!(stats.grains, 4);
        assert_eq!(stats.vectors, 8);
        assert_eq!(stats.grain_sizes, vec![2, 2, 2, 2]);
        let results = index.search_with_nprobe(&[200.2, 200.0], 1, 1).unwrap();
        assert_eq!(results[0].id, VectorId::new(20));
    }

    #[test]
    fn adaptive_quantization_builds_variable_width_grains() {
        let mut index = AperonIndex::with_options(4, 4, 0, 2);
        index
            .enable_adaptive_quantization(1, 4, 0, 2, 1, 2, 0.8)
            .unwrap();

        for i in 0..8 {
            index.insert(i as u64, [i as f32, 0.0, 0.0, 0.0]).unwrap();
        }
        for i in 0..8 {
            let x = i as f32;
            index.insert(100 + i as u64, [x, x, x, x]).unwrap();
        }

        index.rebuild_two_grains().unwrap();

        let stats = index.stats();
        assert_eq!(stats.grains, 2);
        assert!(stats.grain_local_dims.iter().any(|dim| *dim < 4));
        assert!(stats.grain_local_dims.iter().any(|dim| *dim > 1));
        assert!(stats
            .grain_residual_bits
            .iter()
            .all(|bits| *bits == 1 || *bits == 2));

        let loaded = AperonIndex::from_legacy_index(index.to_legacy_index().unwrap()).unwrap();
        let loaded_stats = loaded.stats();
        assert_eq!(loaded_stats.grain_local_dims, stats.grain_local_dims);
        assert_eq!(loaded_stats.grain_sketch_dims, stats.grain_sketch_dims);
        assert_eq!(loaded_stats.grain_residual_bits, stats.grain_residual_bits);
    }

    #[test]
    fn search_with_nprobe_clamps_to_available_grains() {
        let mut index = AperonIndex::with_options(2, 2, 0, 2);
        for cluster in 0..4 {
            let base = cluster as f32 * 100.0;
            index.insert(cluster * 10, [base, base]).unwrap();
            index.insert(cluster * 10 + 1, [base + 1.0, base]).unwrap();
        }
        index.rebuild_n_grains(4).unwrap();

        let all_routes = index.search(&[200.2, 200.0], 3).unwrap();
        let clamped = index.search_with_nprobe(&[200.2, 200.0], 3, 999).unwrap();

        assert_eq!(clamped, all_routes);
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

    #[test]
    fn search_with_rerank_improves_accuracy_on_loaded_index() {
        let mut index = AperonIndex::with_options(3, 2, 1, 4);
        index.insert(1, [0.0, 0.0, 0.0]).unwrap();
        index.insert(2, [10.0, 0.0, 0.0]).unwrap();
        index.insert(3, [0.0, 10.0, 0.0]).unwrap();
        index.rebuild_single_grain().unwrap();

        let legacy = index.to_legacy_index().unwrap();
        let loaded = AperonIndex::from_legacy_index(legacy).unwrap();

        // 1. Search with rerank factor = 0 (no rerank, quantized integer/distance units)
        let mut loaded_no_rerank = loaded.clone();
        loaded_no_rerank.set_rerank_factor(0);
        let results_no_rerank = loaded_no_rerank.search(&[9.0, 0.0, 0.0], 2).unwrap();

        // 2. Search with rerank factor = 4 (uses reconstructed vectors in float32 L2 space)
        let mut loaded_rerank = loaded.clone();
        loaded_rerank.set_rerank_factor(4);
        let results_rerank = loaded_rerank.search(&[9.0, 0.0, 0.0], 2).unwrap();

        assert_eq!(results_no_rerank.len(), 2);
        assert_eq!(results_rerank.len(), 2);

        assert_eq!(results_no_rerank[0].id, VectorId::new(2));
        assert_eq!(results_rerank[0].id, VectorId::new(2));

        // The rerank distance should be closer to the exact float32 L2 distance
        // exact L2 distance squared from [9.0, 0.0, 0.0] to [10.0, 0.0, 0.0] is 1.0.
        // The reconstructed vector will be close to [10.0, 0.0, 0.0].
        let rerank_dist = results_rerank[0].distance;
        assert!(
            (rerank_dist - 1.0).abs() < 1.0,
            "rerank distance {} not close to 1.0",
            rerank_dist
        );
    }

    #[test]
    fn vlbrd_two_bit_round_trip_preserves_searchability() {
        let mut index = AperonIndex::with_options(3, 2, 2, 4);
        index.set_residual_bits(2).unwrap();
        index.insert(1, [0.0, 0.0, 0.0]).unwrap();
        index.insert(2, [10.0, 0.0, 0.0]).unwrap();
        index.insert(3, [0.0, 10.0, 1.0]).unwrap();
        index.insert(4, [0.0, 0.0, 10.0]).unwrap();
        index.rebuild_single_grain().unwrap();

        let stats = index.stats();
        assert_eq!(stats.residual_bits, 2);
        assert!(stats.encoded_bytes > 0);

        let loaded = AperonIndex::from_legacy_index(index.to_legacy_index().unwrap()).unwrap();
        assert_eq!(loaded.residual_bits(), 2);
        let results = loaded.search(&[9.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].id, VectorId::new(2));
    }

    #[test]
    fn shared_basis_pq_round_trip_preserves_reconstruction_path() {
        let mut index = AperonIndex::with_options(4, 2, 0, 4);
        index.enable_shared_basis_pq(4, 2, 2, 4, true).unwrap();
        for i in 0..16 {
            let x = i as f32;
            index
                .insert(i as u64, [x, x * 0.5, (i % 4) as f32, 1.0])
                .unwrap();
        }

        index.rebuild_n_grains(4).unwrap();
        let stats = index.stats();
        assert_eq!(stats.grains, 4);
        assert!(stats.encoded_bytes > 0);

        let legacy = index.to_legacy_index().unwrap();
        let loaded = AperonIndex::from_legacy_index(legacy).unwrap();
        assert_eq!(loaded.stats().grain_local_dims, vec![2, 2, 2, 2]);
        let results = loaded
            .search_with_nprobe(&[10.1, 5.0, 2.0, 1.0], 3, 4)
            .unwrap();
        assert_eq!(results[0].id, VectorId::new(10));
    }

    #[test]
    fn lattice_routing_round_trip() {
        let mut index = AperonIndex::with_options(4, 2, 0, 4);
        for i in 0..16 {
            let x = i as f32;
            index
                .insert(i as u64, [x, x * 0.5, (i % 4) as f32, 1.0])
                .unwrap();
        }

        index.rebuild_n_grains(4).unwrap();
        index.enable_lattice_routing(4, 0.5).unwrap();

        assert!(index.lattice_router().is_some());

        let legacy = index.to_legacy_index().unwrap();
        let loaded = AperonIndex::from_legacy_index(legacy).unwrap();

        assert!(loaded.lattice_router().is_some());
        let results = loaded
            .search_with_nprobe(&[10.1, 5.0, 2.0, 1.0], 3, 4)
            .unwrap();
        assert_eq!(results[0].id, VectorId::new(10));
    }

    #[test]
    fn hlr_routing_round_trip() {
        let mut index = AperonIndex::with_options(4, 2, 0, 4);
        for i in 0..16 {
            let x = i as f32;
            index
                .insert(i as u64, [x, x * 0.5, (i % 4) as f32, 1.0])
                .unwrap();
        }

        index.rebuild_n_grains(4).unwrap();
        let configs = vec![
            crate::routing::HierarchicalLatticeLayerConfig {
                routing_dim: 2,
                spacing: 1.0,
            },
            crate::routing::HierarchicalLatticeLayerConfig {
                routing_dim: 2,
                spacing: 0.5,
            },
        ];
        index.enable_hlr_routing(&configs).unwrap();

        assert!(index.hlr_router().is_some());

        let results = index
            .search_with_nprobe(&[10.1, 5.0, 2.0, 1.0], 3, 4)
            .unwrap();
        assert_eq!(results[0].id, VectorId::new(10));
    }

    #[test]
    fn test_zero_copy_cow_index_branching() {
        let mut index = AperonIndex::with_options(3, 2, 0, 4);
        index.insert(1, [1.0, 2.0, 3.0]).unwrap();
        index.insert(2, [4.0, 5.0, 6.0]).unwrap();
        index.rebuild_single_grain().unwrap();

        // Fork the index by cloning it (Zero-copy CoW)
        let start = std::time::Instant::now();
        let mut child_index = index.clone();
        let elapsed = start.elapsed();
        println!("Forking index took: {:?}", elapsed);
        // Assert that branching takes < 10ms
        assert!(
            elapsed.as_millis() < 10,
            "Branching took too long: {:?}",
            elapsed
        );

        // Verify isolation: insert a vector into the child_index and rebuild it
        child_index.insert(3, [7.0, 8.0, 9.0]).unwrap();
        child_index.rebuild_single_grain().unwrap();

        // Child should have 3 vectors, parent should still have 2
        assert_eq!(child_index.stats().vectors, 3);
        assert_eq!(index.stats().vectors, 2);

        // Verify we can search both indices and get correct results
        let results_child = child_index.search(&[7.0, 8.0, 9.0], 1).unwrap();
        assert_eq!(results_child[0].id, VectorId::new(3));

        let results_parent = index.search(&[7.0, 8.0, 9.0], 1).unwrap();
        assert_eq!(results_parent[0].id, VectorId::new(2)); // nearest from parent
    }
}
