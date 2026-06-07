use crate::{distance::l2_squared, Distance, GrainId};
use std::{collections::HashMap, mem::size_of};

/// Chosen grain and score for a query route.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Route {
    pub grain_id: GrainId,
    pub distance: Distance,
}

/// Centroid router for assigning queries to local grains.
#[derive(Clone, Debug)]
pub struct CentroidRouter {
    dim: usize,
    centroids: Vec<(GrainId, Vec<f32>)>,
}

impl CentroidRouter {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            centroids: Vec::new(),
        }
    }

    pub fn add_centroid(
        &mut self,
        grain_id: GrainId,
        centroid: impl Into<Vec<f32>>,
    ) -> Result<(), String> {
        let centroid = centroid.into();
        if centroid.len() != self.dim {
            return Err(format!(
                "dimension mismatch: expected {}, got {}",
                self.dim,
                centroid.len()
            ));
        }

        self.centroids.push((grain_id, centroid));
        Ok(())
    }

    pub fn route(&self, query: &[f32]) -> Option<Route> {
        if query.len() != self.dim {
            return None;
        }

        self.centroids
            .iter()
            .filter_map(|(grain_id, centroid)| {
                l2_squared(query, centroid).map(|distance| Route {
                    grain_id: *grain_id,
                    distance,
                })
            })
            .min_by(|a, b| a.distance.total_cmp(&b.distance))
    }
}

/// LatticeRouter for O(1) routing in low-dimensional projected space.
#[derive(Clone, Debug, PartialEq)]
pub struct LatticeRouter {
    pub dim: usize,
    pub routing_dim: usize,
    pub spacing: f32,
    pub projection: Vec<f32>,
    pub lattice_map: HashMap<Vec<i16>, Vec<usize>>,
    pub centroids_coords: Vec<i16>,
    pub num_centroids: usize,
}

impl LatticeRouter {
    pub fn new(
        dim: usize,
        routing_dim: usize,
        spacing: f32,
        centroids: &[Vec<f32>],
    ) -> Result<Self, String> {
        if routing_dim == 0 {
            return Err("routing_dim must be greater than zero".to_string());
        }
        if routing_dim > 6 {
            return Err(
                "routing_dim cannot be greater than 6 to limit boundary search overhead"
                    .to_string(),
            );
        }
        if spacing <= 0.0 {
            return Err("spacing must be positive".to_string());
        }

        // 1. Run PCA on the centroids to get the routing projection matrix.
        let (_, raw_projection) = crate::grain::run_pca(centroids, dim, routing_dim);

        // Transpose raw_projection (dim x routing_dim) to projection (routing_dim x dim)
        let mut projection = vec![0.0; routing_dim * dim];
        for d in 0..dim {
            for j in 0..routing_dim {
                projection[j * dim + d] = raw_projection[d * routing_dim + j];
            }
        }

        let mut router = Self {
            dim,
            routing_dim,
            spacing,
            projection,
            lattice_map: HashMap::new(),
            centroids_coords: Vec::new(),
            num_centroids: centroids.len(),
        };

        // 2. Populate lattice map and centroids_coords
        let mut centroids_coords = vec![0_i16; centroids.len() * routing_dim];
        for (g_idx, centroid) in centroids.iter().enumerate() {
            let coords = router.project_and_quantize(centroid);
            for j in 0..routing_dim {
                centroids_coords[g_idx * routing_dim + j] = coords[j];
            }
            router.lattice_map.entry(coords).or_default().push(g_idx);
        }
        router.centroids_coords = centroids_coords;

        Ok(router)
    }

    pub fn project_and_quantize(&self, vector: &[f32]) -> Vec<i16> {
        let mut coords = vec![0; self.routing_dim];
        for (j, coord) in coords.iter_mut().enumerate().take(self.routing_dim) {
            let mut dot = 0.0;
            for (d, value) in vector.iter().enumerate().take(self.dim) {
                dot += value * self.projection[j * self.dim + d];
            }
            *coord = (dot / self.spacing).round() as i16;
        }
        coords
    }

    pub fn route_nprobe(&self, query: &[f32], nprobe: usize) -> Vec<usize> {
        if query.len() != self.dim || nprobe == 0 {
            return Vec::new();
        }

        // 1. Project query to low-dim routing space
        let mut z_q = [0.0; 6];
        for (j, value) in z_q.iter_mut().enumerate().take(self.routing_dim) {
            let mut dot = 0.0;
            for (d, query_value) in query.iter().enumerate().take(self.dim) {
                dot += query_value * self.projection[j * self.dim + d];
            }
            *value = dot / self.spacing;
        }

        // 2. Scan all centroids in the projected space
        let mut dists = Vec::with_capacity(self.num_centroids);
        for g_idx in 0..self.num_centroids {
            let mut dist_sq = 0.0;
            for (j, query_coord) in z_q.iter().enumerate().take(self.routing_dim) {
                let diff = query_coord - self.centroids_coords[g_idx * self.routing_dim + j] as f32;
                dist_sq += diff * diff;
            }
            dists.push((g_idx, dist_sq));
        }

        // 3. Partition and sort only the top nprobe candidates
        let limit = nprobe.min(self.num_centroids);
        if limit < self.num_centroids {
            dists.select_nth_unstable_by(limit, |a, b| {
                a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0))
            });
            dists.truncate(limit);
        }
        dists.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

        // 4. Return top nprobe candidates
        dists.into_iter().map(|(g_idx, _)| g_idx).collect()
    }
}

/// One residual PCA lattice layer in a hierarchical router.
#[derive(Clone, Debug, PartialEq)]
pub struct HierarchicalLatticeLayer {
    pub routing_dim: usize,
    pub spacing: f32,
    pub mean: Vec<f32>,
    /// Column-major basis with shape dim x routing_dim.
    pub projection: Vec<f32>,
}

impl HierarchicalLatticeLayer {
    pub fn project(&self, residual: &[f32], dim: usize) -> Vec<f32> {
        let mut centered = vec![0.0; dim];
        for d in 0..dim {
            centered[d] = residual[d] - self.mean[d];
        }
        project_with_basis(&centered, dim, &self.projection, self.routing_dim)
    }

    pub fn quantize(&self, coords: &[f32]) -> Vec<i16> {
        coords
            .iter()
            .map(|coord| (coord / self.spacing).round() as i16)
            .collect()
    }

    fn residual_after_projection(&self, residual: &[f32], dim: usize) -> Vec<f32> {
        let mut centered = vec![0.0; dim];
        for d in 0..dim {
            centered[d] = residual[d] - self.mean[d];
        }
        let coords = project_with_basis(&centered, dim, &self.projection, self.routing_dim);
        subtract_reconstruction(&centered, dim, &self.projection, self.routing_dim, &coords)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HierarchicalLatticeLayerConfig {
    pub routing_dim: usize,
    pub spacing: f32,
}

/// Hierarchical lattice router using residual PCA at each layer.
///
/// # Centered Residual Energy Determination
/// The router tracks the remaining variance (energy) at each stage in `residual_energy`:
/// - `residual_energy[0]` is the uncentered mean squared norm of the input vectors.
/// - For each layer `l >= 1`, `residual_energy[l]` represents the **centered residual energy**
///   (variance) of the residuals. This is because PCA projection centers the residuals at each layer
///   (mean subtracted, mean of output residuals is zero).
/// - As a result, the residual energy decreases monotonically after each layer representation is subtracted.
#[derive(Clone, Debug, PartialEq)]
pub struct HierarchicalLatticeRouter {
    pub dim: usize,
    pub layers: Vec<HierarchicalLatticeLayer>,
    pub entries: Vec<HierarchicalLatticeEntry>,
    /// Tracks residual energy: `[0]` is uncentered input energy; `[l]` for `l >= 1` is centered.
    pub residual_energy: Vec<f32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HierarchicalLatticeEntry {
    pub z_order_key: Vec<u8>,
    pub coords: Vec<i16>,
    pub index: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HtlaRouter {
    pub dim: usize,
    pub chart_dim: usize,
    pub levels: usize,
    pub nodes: Vec<HtlaNode>,
    pub child_entries: Vec<HtlaChildEntry>,
    pub centroids: Vec<Vec<f32>>,
    pub diagnostics: HtlaDiagnostics,
    
    // Contiguous arrays for cache locality & memory locality
    pub node_centroids: Vec<f32>,
    pub node_projections: Vec<f32>,
    pub node_projection_biases: Vec<f32>,
    pub node_eigenvalues: Vec<f32>,
    
    // Configurable soft-spill threshold (L1 L-inf squared distance gap)
    pub soft_spill_threshold: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HtlaNode {
    pub centroid_offset: usize,
    pub projection_offset: usize,
    pub bias_offset: usize,
    pub eigenvalues_offset: usize,
    pub child_start: usize,
    pub child_len: usize,
    pub level: usize,
    pub centroid_index: Option<usize>,
    pub radius: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HtlaChildEntry {
    pub key: u128,
    pub coords: Vec<i16>,
    pub child_node: usize,
    pub centroid_index: Option<usize>,
    pub distance_to_parent: f32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct HtlaDiagnostics {
    pub pca_energy: Vec<f32>,
    pub d80: Vec<usize>,
    pub d90: Vec<usize>,
    pub d95: Vec<usize>,
    pub residual_energy: Vec<f32>,
    pub radius_shrink: Vec<f32>,
    pub norm_sep_p10: f32,
    pub norm_sep_p25: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HtlaRoute {
    pub candidates: Vec<usize>,
    pub final_nprobe: Vec<usize>,
    pub fallback: bool,
    pub working_set_bytes: usize,
}

impl HtlaRouter {
    pub fn new(
        dim: usize,
        centroids: &[Vec<f32>],
        levels: usize,
        chart_dim: usize,
    ) -> Result<Self, String> {
        if dim == 0 {
            return Err("dim must be greater than zero".to_string());
        }
        if centroids.is_empty() {
            return Err("at least one centroid is required".to_string());
        }
        if levels < 2 {
            return Err("levels must be at least 2".to_string());
        }
        if chart_dim == 0 || chart_dim > dim {
            return Err(format!("chart_dim must be in 1..={dim}"));
        }
        for centroid in centroids {
            if centroid.len() != dim {
                return Err(format!(
                    "dimension mismatch: expected {}, got {}",
                    dim,
                    centroid.len()
                ));
            }
        }

        let mut router = Self {
            dim,
            chart_dim,
            levels,
            nodes: Vec::new(),
            child_entries: Vec::new(),
            centroids: centroids.to_vec(),
            diagnostics: HtlaDiagnostics::default(),
            node_centroids: Vec::new(),
            node_projections: Vec::new(),
            node_projection_biases: Vec::new(),
            node_eigenvalues: Vec::new(),
            soft_spill_threshold: 3000.0,
        };
        let indices = (0..centroids.len()).collect::<Vec<_>>();
        router.build_node(0, &indices);
        router.compute_diagnostics();
        Ok(router)
    }

    pub fn route(&self, query: &[f32], beam: usize, pool: usize, final_nprobe: usize) -> HtlaRoute {
        if query.len() != self.dim || self.nodes.is_empty() || pool == 0 || final_nprobe == 0 {
            return HtlaRoute {
                candidates: Vec::new(),
                final_nprobe: Vec::new(),
                fallback: true,
                working_set_bytes: 0,
            };
        }

        let beam = beam.max(1);
        let mut fallback = false;
        let mut working_nodes = 0usize;
        let mut frontier = vec![(0usize, 0.0f32)];
        let mut candidates = Vec::new();

        while !frontier.is_empty() && candidates.len() < pool {
            let mut scored = Vec::new();
            let mut leaf_scored = Vec::new();
            for (node_idx, _) in frontier {
                let node = &self.nodes[node_idx];
                working_nodes += 1;
                if let Some(index) = node.centroid_index {
                    candidates.push(index);
                    continue;
                }
                let q_coords = self.project_query_to_node(query, node);
                for entry in
                    &self.child_entries[node.child_start..node.child_start + node.child_len]
                {
                    let score = local_key_distance(&q_coords, &entry.coords);
                    if let Some(index) = entry
                        .centroid_index
                        .or(self.nodes[entry.child_node].centroid_index)
                    {
                        leaf_scored.push((index, score));
                    } else {
                        scored.push((entry.child_node, score));
                    }
                }
            }
            if !leaf_scored.is_empty() {
                leaf_scored.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
                for (index, _) in leaf_scored {
                    if candidates.len() >= pool {
                        break;
                    }
                    candidates.push(index);
                }
            }
            if scored.is_empty() {
                break;
            }
            scored.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
            let mut keep = beam.min(scored.len());
            if scored.len() > 1 {
                let score_gap = scored[1].1 - scored[0].1;
                if score_gap > self.soft_spill_threshold {
                    keep = 1;
                }
            }
            frontier = scored.into_iter().take(keep).collect();
        }

        candidates.sort_unstable();
        candidates.dedup();
        if candidates.len() < final_nprobe.min(self.centroids.len()) {
            fallback = true;
        }
        if candidates.len() > pool {
            candidates.truncate(pool);
        }

        let mut reranked = candidates
            .iter()
            .copied()
            .map(|idx| {
                (
                    idx,
                    l2_squared(query, &self.centroids[idx]).unwrap_or(f32::INFINITY),
                )
            })
            .collect::<Vec<_>>();
        reranked.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        let final_nprobe = reranked
            .into_iter()
            .take(final_nprobe.min(candidates.len()))
            .map(|(idx, _)| idx)
            .collect();

        HtlaRoute {
            candidates,
            final_nprobe,
            fallback,
            working_set_bytes: working_nodes * self.node_bytes()
                + beam * self.chart_dim * size_of::<f32>(),
        }
    }

    pub fn node_centroid(&self, node: &HtlaNode) -> &[f32] {
        &self.node_centroids[node.centroid_offset..node.centroid_offset + self.dim]
    }

    pub fn node_projection(&self, node: &HtlaNode) -> &[f32] {
        &self.node_projections[node.projection_offset..node.projection_offset + self.dim * self.chart_dim]
    }

    pub fn node_projection_bias(&self, node: &HtlaNode) -> &[f32] {
        &self.node_projection_biases[node.bias_offset..node.bias_offset + self.chart_dim]
    }

    pub fn node_eigenvalues(&self, node: &HtlaNode) -> &[f32] {
        &self.node_eigenvalues[node.eigenvalues_offset..node.eigenvalues_offset + self.chart_dim]
    }

    pub fn resident_bytes(&self) -> usize {
        self.centroids.len() * self.dim * size_of::<f32>()
            + self.nodes.len() * self.node_bytes()
            + self.child_entries.len()
                * (size_of::<HtlaChildEntry>() + self.chart_dim * size_of::<i16>())
    }

    fn node_bytes(&self) -> usize {
        size_of::<HtlaNode>()
            + self.dim * size_of::<f32>()
            + self.dim * self.chart_dim * size_of::<f32>()
            + self.chart_dim * size_of::<f32>()
            + self.chart_dim * size_of::<f32>()
    }

    fn build_node(&mut self, level: usize, indices: &[usize]) -> usize {
        let node_idx = self.nodes.len();
        let points = indices
            .iter()
            .map(|&idx| self.centroids[idx].clone())
            .collect::<Vec<_>>();
        let centroid = mean_vector(&points, self.dim);
        let (mean, projection) = crate::grain::run_pca(&points, self.dim, self.chart_dim);
        let eigenvalues = projected_variance(&points, self.dim, &mean, &projection, self.chart_dim);
        let radius = points
            .iter()
            .map(|point| l2_squared(point, &centroid).unwrap_or(0.0).sqrt())
            .fold(0.0, f32::max);

        // Contiguous storage for node data
        let centroid_offset = self.node_centroids.len();
        self.node_centroids.extend_from_slice(&centroid);

        let projection_offset = self.node_projections.len();
        self.node_projections.extend_from_slice(&projection);

        let bias_offset = self.node_projection_biases.len();
        for component in 0..self.chart_dim {
            let mut dot = 0.0;
            for d in 0..self.dim {
                dot += centroid[d] * projection[d * self.chart_dim + component];
            }
            self.node_projection_biases.push(-dot);
        }

        let eigenvalues_offset = self.node_eigenvalues.len();
        self.node_eigenvalues.extend_from_slice(&eigenvalues);

        self.nodes.push(HtlaNode {
            centroid_offset,
            projection_offset,
            bias_offset,
            eigenvalues_offset,
            child_start: self.child_entries.len(),
            child_len: 0,
            level,
            centroid_index: if indices.len() == 1 {
                Some(indices[0])
            } else {
                None
            },
            radius,
        });

        if indices.len() == 1 || level + 1 >= self.levels {
            if indices.len() > 1 {
                let child_start = self.child_entries.len();
                for &idx in indices {
                    let coords = self.project_centroid_to_node(idx, node_idx);
                    let key = morton_u128(&coords);
                    let leaf_idx = self.nodes.len();

                    let c_offset = self.node_centroids.len();
                    self.node_centroids.extend_from_slice(&self.centroids[idx]);

                    let p_offset = self.node_projections.len();
                    self.node_projections.resize(p_offset + self.dim * self.chart_dim, 0.0);

                    let b_offset = self.node_projection_biases.len();
                    self.node_projection_biases.resize(b_offset + self.chart_dim, 0.0);

                    let e_offset = self.node_eigenvalues.len();
                    self.node_eigenvalues.resize(e_offset + self.chart_dim, 0.0);

                    self.nodes.push(HtlaNode {
                        centroid_offset: c_offset,
                        projection_offset: p_offset,
                        bias_offset: b_offset,
                        eigenvalues_offset: e_offset,
                        child_start: self.child_entries.len(),
                        child_len: 0,
                        level: level + 1,
                        centroid_index: Some(idx),
                        radius: 0.0,
                    });
                    
                    let parent_centroid = self.node_centroid(&self.nodes[node_idx]).to_vec();
                    self.child_entries.push(HtlaChildEntry {
                        key,
                        coords,
                        child_node: leaf_idx,
                        centroid_index: Some(idx),
                        distance_to_parent: l2_squared(
                            &self.centroids[idx],
                            &parent_centroid,
                        )
                        .unwrap_or(0.0)
                        .sqrt(),
                    });
                }
                self.child_entries[child_start..].sort_by(|a, b| a.key.cmp(&b.key));
                self.nodes[node_idx].child_start = child_start;
                self.nodes[node_idx].child_len = indices.len();
            }
            return node_idx;
        }

        let fanout = fanout_for(indices.len(), self.levels - level);
        let mut ordered = indices.to_vec();
        ordered.sort_by(|&a, &b| {
            let ca = self.project_centroid_to_node(a, node_idx);
            let cb = self.project_centroid_to_node(b, node_idx);
            ca.cmp(&cb).then_with(|| a.cmp(&b))
        });
        let chunk = ordered.len().div_ceil(fanout);
        let groups = ordered
            .chunks(chunk.max(1))
            .map(|c| c.to_vec())
            .collect::<Vec<_>>();

        let mut pending = Vec::with_capacity(groups.len());
        for group in groups {
            let child_idx = self.build_node(level + 1, &group);
            let child_centroid = self.node_centroid(&self.nodes[child_idx]).to_vec();
            let coords = self.project_vector_to_node(&child_centroid, node_idx);
            let parent_centroid = self.node_centroid(&self.nodes[node_idx]).to_vec();
            pending.push(HtlaChildEntry {
                key: morton_u128(&coords),
                coords,
                child_node: child_idx,
                centroid_index: None,
                distance_to_parent: l2_squared(
                    &child_centroid,
                    &parent_centroid,
                )
                .unwrap_or(0.0)
                .sqrt(),
            });
        }
        pending.sort_by(|a, b| {
            a.key
                .cmp(&b.key)
                .then_with(|| a.child_node.cmp(&b.child_node))
        });
        let child_start = self.child_entries.len();
        self.child_entries.extend(pending);
        self.nodes[node_idx].child_start = child_start;
        self.nodes[node_idx].child_len = self.child_entries.len() - child_start;
        node_idx
    }

    fn project_query_to_node(&self, query: &[f32], node: &HtlaNode) -> Vec<i16> {
        quantized_project(
            query,
            self.dim,
            self.node_projection(node),
            self.node_projection_bias(node),
            self.chart_dim,
        )
    }

    fn project_centroid_to_node(&self, centroid_idx: usize, node_idx: usize) -> Vec<i16> {
        self.project_vector_to_node(&self.centroids[centroid_idx], node_idx)
    }

    fn project_vector_to_node(&self, vector: &[f32], node_idx: usize) -> Vec<i16> {
        let node = &self.nodes[node_idx];
        quantized_project(
            vector,
            self.dim,
            self.node_projection(node),
            self.node_projection_bias(node),
            self.chart_dim,
        )
    }

    fn compute_diagnostics(&mut self) {
        let mut residual_energy = vec![mean_energy(&self.centroids)];
        let mut radius_shrink = Vec::new();
        let mut sep = Vec::new();
        let n_nodes = self.nodes.len();
        for i in 0..n_nodes {
            let node = &self.nodes[i];
            if node.child_len == 0 {
                continue;
            }
            let start = node.eigenvalues_offset;
            let end = start + self.chart_dim;
            let node_ev = &self.node_eigenvalues[start..end];
            let total = node_ev.iter().sum::<f32>().max(1e-9);
            self.diagnostics.pca_energy.push(total);
            self.diagnostics
                .d80
                .push(energy_dim(node_ev, 0.80));
            self.diagnostics
                .d90
                .push(energy_dim(node_ev, 0.90));
            self.diagnostics
                .d95
                .push(energy_dim(node_ev, 0.95));
            let child_radius = self.child_entries
                [node.child_start..node.child_start + node.child_len]
                .iter()
                .map(|entry| self.nodes[entry.child_node].radius)
                .fold(0.0, f32::max);
            if node.radius > 0.0 {
                radius_shrink.push(child_radius / node.radius);
            }
            let children = &self.child_entries[node.child_start..node.child_start + node.child_len];
            for i in 0..children.len() {
                for j in i + 1..children.len() {
                    let dist = local_key_distance(&children[i].coords, &children[j].coords).sqrt();
                    sep.push(dist / (node.radius + 1e-6));
                }
            }
            residual_energy.push(total);
        }
        self.diagnostics.residual_energy = residual_energy;
        self.diagnostics.radius_shrink = radius_shrink;
        sep.sort_by(|a, b| a.total_cmp(b));
        self.diagnostics.norm_sep_p10 = percentile(&sep, 0.10);
        self.diagnostics.norm_sep_p25 = percentile(&sep, 0.25);
    }
}

impl HierarchicalLatticeRouter {
    pub fn new(
        dim: usize,
        layer_configs: &[HierarchicalLatticeLayerConfig],
        vectors: &[Vec<f32>],
    ) -> Result<Self, String> {
        if dim == 0 {
            return Err("dim must be greater than zero".to_string());
        }
        if layer_configs.is_empty() {
            return Err("at least one HLR layer is required".to_string());
        }
        for vector in vectors {
            if vector.len() != dim {
                return Err(format!(
                    "dimension mismatch: expected {}, got {}",
                    dim,
                    vector.len()
                ));
            }
        }

        let mut residuals = vectors.to_vec();
        let mut layers = Vec::with_capacity(layer_configs.len());
        let mut residual_energy = vec![mean_energy(&residuals)];

        for config in layer_configs {
            if config.routing_dim == 0 {
                return Err("HLR routing_dim must be greater than zero".to_string());
            }
            if config.routing_dim > dim {
                return Err(format!(
                    "HLR routing_dim {} cannot exceed dim {}",
                    config.routing_dim, dim
                ));
            }
            if config.spacing <= 0.0 {
                return Err("HLR spacing must be positive".to_string());
            }

            let (mean, projection) = crate::grain::run_pca(&residuals, dim, config.routing_dim);
            let layer = HierarchicalLatticeLayer {
                routing_dim: config.routing_dim,
                spacing: config.spacing,
                mean,
                projection,
            };

            residuals = residuals
                .iter()
                .map(|residual| layer.residual_after_projection(residual, dim))
                .collect();
            residual_energy.push(mean_energy(&residuals));
            layers.push(layer);
        }

        let mut entries = vectors
            .iter()
            .enumerate()
            .map(|(index, vector)| {
                let coords = flatten_key(&layers, dim, vector);
                let z_order_key = encode_z_order(&coords);
                HierarchicalLatticeEntry {
                    z_order_key,
                    coords,
                    index,
                }
            })
            .collect::<Vec<_>>();
        entries.sort_by(|a, b| {
            a.z_order_key
                .cmp(&b.z_order_key)
                .then_with(|| a.index.cmp(&b.index))
        });

        Ok(Self {
            dim,
            layers,
            entries,
            residual_energy,
        })
    }

    pub fn key(&self, vector: &[f32]) -> Result<Vec<i16>, String> {
        if vector.len() != self.dim {
            return Err(format!(
                "dimension mismatch: expected {}, got {}",
                self.dim,
                vector.len()
            ));
        }
        Ok(flatten_key(&self.layers, self.dim, vector))
    }

    pub fn route_nprobe(&self, query: &[f32], nprobe: usize) -> Vec<usize> {
        if query.len() != self.dim || nprobe == 0 || self.entries.is_empty() {
            return Vec::new();
        }
        let query_coords = flatten_key(&self.layers, self.dim, query);
        let query_z_order_key = encode_z_order(&query_coords);

        let pos = match self
            .entries
            .binary_search_by(|entry| entry.z_order_key.cmp(&query_z_order_key))
        {
            Ok(idx) => idx,
            Err(idx) => idx,
        };

        // Select search window around pos of size W = (nprobe * 16).max(128)
        let w_size = (nprobe * 16).max(128).min(self.entries.len());
        let half_w = w_size / 2;
        let start_idx = pos.saturating_sub(half_w);
        let end_idx = (start_idx + w_size).min(self.entries.len());
        let actual_start_idx = if end_idx == self.entries.len() && self.entries.len() > w_size {
            self.entries.len() - w_size
        } else {
            start_idx
        };

        let mut scored = self.entries[actual_start_idx..end_idx]
            .iter()
            .map(|entry| {
                let dist = lattice_key_distance(&query_coords, &entry.coords);
                (entry.index, dist)
            })
            .collect::<Vec<_>>();

        let limit = nprobe.min(scored.len());
        if limit < scored.len() {
            scored.select_nth_unstable_by(limit, |a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
            scored.truncate(limit);
        }
        scored.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        scored.into_iter().map(|(index, _)| index).collect()
    }
}

fn flatten_key(layers: &[HierarchicalLatticeLayer], dim: usize, vector: &[f32]) -> Vec<i16> {
    let total_dim = layers.iter().map(|layer| layer.routing_dim).sum();
    let mut key = Vec::with_capacity(total_dim);
    let mut residual = vector.to_vec();
    for layer in layers {
        let coords = layer.project(&residual, dim);
        key.extend(layer.quantize(&coords));
        residual = layer.residual_after_projection(&residual, dim);
    }
    key
}

pub fn encode_z_order(coords: &[i16]) -> Vec<u8> {
    let d = coords.len();
    if d == 0 {
        return Vec::new();
    }
    let u_coords: Vec<u16> = coords.iter().map(|&x| (x as i32 + 32768) as u16).collect();
    let total_bits = 16 * d;
    let num_bytes = total_bits.div_ceil(8);
    let mut out = vec![0u8; num_bytes];
    for b in 0..total_bits {
        let coord_idx = b % d;
        let bit_in_coord = 15 - (b / d);
        let bit_val = ((u_coords[coord_idx] >> bit_in_coord) & 1) as u8;
        let byte_idx = b / 8;
        let bit_in_byte = 7 - (b % 8);
        out[byte_idx] |= bit_val << bit_in_byte;
    }
    out
}

fn project_with_basis(vector: &[f32], dim: usize, projection: &[f32], k: usize) -> Vec<f32> {
    let mut out = vec![0.0; k];
    for component in 0..k {
        for d in 0..dim {
            out[component] += vector[d] * projection[d * k + component];
        }
    }
    out
}

fn subtract_reconstruction(
    vector: &[f32],
    dim: usize,
    projection: &[f32],
    k: usize,
    coords: &[f32],
) -> Vec<f32> {
    let mut residual = vec![0.0; dim];
    for d in 0..dim {
        let mut recon = 0.0;
        for component in 0..k {
            recon += coords[component] * projection[d * k + component];
        }
        residual[d] = vector[d] - recon;
    }
    residual
}

fn mean_energy(vectors: &[Vec<f32>]) -> f32 {
    if vectors.is_empty() {
        return 0.0;
    }
    vectors
        .iter()
        .map(|vector| vector.iter().map(|value| value * value).sum::<f32>())
        .sum::<f32>()
        / vectors.len() as f32
}

fn lattice_key_distance(a: &[i16], b: &[i16]) -> i64 {
    a.iter()
        .zip(b)
        .map(|(left, right)| {
            let diff = i64::from(*left) - i64::from(*right);
            diff * diff
        })
        .sum()
}

fn local_key_distance(a: &[i16], b: &[i16]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(left, right)| {
            let diff = f32::from(*left) - f32::from(*right);
            diff * diff
        })
        .sum()
}

fn mean_vector(vectors: &[Vec<f32>], dim: usize) -> Vec<f32> {
    let mut mean = vec![0.0; dim];
    if vectors.is_empty() {
        return mean;
    }
    for vector in vectors {
        for d in 0..dim {
            mean[d] += vector[d];
        }
    }
    for value in &mut mean {
        *value /= vectors.len() as f32;
    }
    mean
}

fn projected_variance(
    vectors: &[Vec<f32>],
    dim: usize,
    mean: &[f32],
    projection: &[f32],
    k: usize,
) -> Vec<f32> {
    let mut out = vec![0.0; k];
    if vectors.is_empty() {
        return out;
    }
    for vector in vectors {
        for component in 0..k {
            let mut dot = 0.0;
            for d in 0..dim {
                dot += (vector[d] - mean[d]) * projection[d * k + component];
            }
            out[component] += dot * dot;
        }
    }
    for value in &mut out {
        *value /= vectors.len() as f32;
    }
    out
}

fn quantized_project(
    vector: &[f32],
    dim: usize,
    projection: &[f32],
    bias: &[f32],
    k: usize,
) -> Vec<i16> {
    let mut coords = vec![0_i16; k];
    for component in 0..k {
        let mut dot = bias[component];
        for d in 0..dim {
            dot += vector[d] * projection[d * k + component];
        }
        coords[component] = dot.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
    }
    coords
}

fn morton_u128(coords: &[i16]) -> u128 {
    if coords.is_empty() {
        return 0;
    }
    let dims = coords.len().min(16);
    let values = coords
        .iter()
        .take(dims)
        .map(|&coord| (i32::from(coord) + 32768) as u16)
        .collect::<Vec<_>>();
    let bits_per_dim = (128 / dims).min(16);
    let mut key = 0_u128;
    for bit in (0..bits_per_dim).rev() {
        for value in &values {
            key <<= 1;
            key |= u128::from((value >> bit) & 1);
        }
    }
    key
}

fn fanout_for(count: usize, remaining_levels: usize) -> usize {
    if remaining_levels <= 2 {
        return count.max(1);
    }
    let internal_hops = remaining_levels - 1;
    let fanout = (count as f64).powf(1.0 / internal_hops as f64).ceil() as usize;
    fanout.clamp(2, count.max(2))
}

fn energy_dim(eigenvalues: &[f32], target: f32) -> usize {
    let total = eigenvalues.iter().sum::<f32>();
    if total <= 0.0 {
        return 0;
    }
    let mut acc = 0.0;
    for (idx, value) in eigenvalues.iter().enumerate() {
        acc += *value;
        if acc / total >= target {
            return idx + 1;
        }
    }
    eigenvalues.len()
}

fn percentile(values: &[f32], p: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let idx = ((values.len() - 1) as f32 * p).round() as usize;
    values[idx.min(values.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hlr_trains_two_orthogonal_residual_layers() {
        let vectors = vec![
            vec![-6.0, 0.0, 0.0, 0.0],
            vec![-3.0, 0.0, 0.0, 0.0],
            vec![3.0, 0.0, 0.0, 0.0],
            vec![6.0, 0.0, 0.0, 0.0],
            vec![0.0, -2.0, 0.0, 0.0],
            vec![0.0, 2.0, 0.0, 0.0],
        ];
        let router = HierarchicalLatticeRouter::new(
            4,
            &[
                HierarchicalLatticeLayerConfig {
                    routing_dim: 1,
                    spacing: 1.0,
                },
                HierarchicalLatticeLayerConfig {
                    routing_dim: 1,
                    spacing: 1.0,
                },
            ],
            &vectors,
        )
        .unwrap();

        assert_eq!(router.layers.len(), 2);
        assert!(router.residual_energy[1] < router.residual_energy[0]);
        assert!(router.residual_energy[2] < router.residual_energy[1]);

        let dot = router.layers[0].projection[0] * router.layers[1].projection[0]
            + router.layers[0].projection[1] * router.layers[1].projection[1]
            + router.layers[0].projection[2] * router.layers[1].projection[2]
            + router.layers[0].projection[3] * router.layers[1].projection[3];
        assert!(
            dot.abs() < 1e-3,
            "residual PCA layers are not orthogonal: {dot}"
        );
    }

    #[test]
    fn hlr_supports_three_layers_and_routes_nearby_keys() {
        let vectors = vec![
            vec![-8.0, 0.0, 0.0],
            vec![-4.0, 0.0, 0.0],
            vec![4.0, 0.0, 0.0],
            vec![8.0, 0.0, 0.0],
            vec![0.0, -6.0, 0.0],
            vec![0.0, 6.0, 0.0],
            vec![0.0, 0.0, -3.0],
            vec![0.0, 0.0, 3.0],
        ];
        let router = HierarchicalLatticeRouter::new(
            3,
            &[
                HierarchicalLatticeLayerConfig {
                    routing_dim: 1,
                    spacing: 1.0,
                },
                HierarchicalLatticeLayerConfig {
                    routing_dim: 1,
                    spacing: 1.0,
                },
                HierarchicalLatticeLayerConfig {
                    routing_dim: 1,
                    spacing: 1.0,
                },
            ],
            &vectors,
        )
        .unwrap();

        assert_eq!(router.layers.len(), 3);
        assert_eq!(router.residual_energy.len(), 4);
        assert!(router
            .residual_energy
            .windows(2)
            .all(|pair| pair[1] <= pair[0] + 1e-4));

        let routed = router.route_nprobe(&[7.8, 0.1, 0.0], 2);
        assert!(routed.contains(&3));
    }

    #[test]
    fn z_order_encoding_interleaving() {
        let key = encode_z_order(&[1, 2]);
        assert_eq!(key[0], 0xC0);
        assert_eq!(key[key.len() - 1], 6);
    }

    struct SimpleRng {
        state: u64,
    }
    impl SimpleRng {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }
        fn next_u32(&mut self) -> u32 {
            self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (self.state >> 32) as u32
        }
        fn gen_range(&mut self, min: f32, max: f32) -> f32 {
            let val = self.next_u32() as f32 / u32::MAX as f32;
            min + val * (max - min)
        }
    }

    #[test]
    fn hlr_z_order_has_high_recall_parity() {
        let mut rng = SimpleRng::new(12345);
        let dim = 8;
        let num_vectors = 500;
        let mut vectors = Vec::with_capacity(num_vectors);
        for _ in 0..num_vectors {
            let vec: Vec<f32> = (0..dim).map(|_| rng.gen_range(-10.0, 10.0)).collect();
            vectors.push(vec);
        }

        let configs = vec![
            HierarchicalLatticeLayerConfig {
                routing_dim: 2,
                spacing: 2.0,
            },
            HierarchicalLatticeLayerConfig {
                routing_dim: 2,
                spacing: 1.0,
            },
        ];

        let router = HierarchicalLatticeRouter::new(dim, &configs, &vectors).unwrap();

        let mut exact_matches = 0;
        let queries_count = 50;
        for _ in 0..queries_count {
            let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-10.0, 10.0)).collect();
            let nprobe = 5;

            let query_coords = flatten_key(&router.layers, dim, &query);
            let mut all_scored = router
                .entries
                .iter()
                .map(|entry| {
                    let dist = lattice_key_distance(&query_coords, &entry.coords);
                    (entry.index, dist)
                })
                .collect::<Vec<_>>();
            all_scored.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
            let exact_routed: Vec<usize> =
                all_scored.into_iter().take(nprobe).map(|x| x.0).collect();

            let z_routed = router.route_nprobe(&query, nprobe);

            let intersection: std::collections::HashSet<_> = exact_routed.iter().copied().collect();
            let matched = z_routed.iter().filter(|x| intersection.contains(x)).count();
            exact_matches += matched;
        }

        let total_possible = queries_count * 5;
        let recall = exact_matches as f32 / total_possible as f32;
        println!(
            "Z-order window routing recall vs exact HLR routing: {:.4}",
            recall
        );
        assert!(recall >= 0.70, "Recall was too low: {}", recall);
    }

    #[test]
    fn htla_trains_array_tree_and_routes_deterministically() {
        let vectors = (0..64)
            .map(|i| vec![i as f32, (i % 8) as f32, 0.0, 1.0])
            .collect::<Vec<_>>();
        let router = HtlaRouter::new(4, &vectors, 3, 2).unwrap();

        assert!(!router.nodes.is_empty());
        assert!(!router.child_entries.is_empty());
        assert!(router.child_entries.iter().any(|entry| entry.key > 0));
        assert!(router.resident_bytes() > vectors.len() * 4 * size_of::<f32>());
        assert!(!router.diagnostics.d80.is_empty());

        let first = router.route(&[10.1, 2.0, 0.0, 1.0], 4, 16, 4);
        let second = router.route(&[10.1, 2.0, 0.0, 1.0], 4, 16, 4);
        assert_eq!(first.final_nprobe, second.final_nprobe);
        assert!(first.candidates.len() <= 16);
        assert!(first.final_nprobe.contains(&10) || first.final_nprobe.contains(&11));
    }
}
