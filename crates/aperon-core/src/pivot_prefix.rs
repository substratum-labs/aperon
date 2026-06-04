use crate::distance::l2_squared_unchecked;
use std::{mem::size_of, time::Instant};

pub const DEFAULT_FINAL_NPROBE: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrefixScoreMode {
    Union,
    Weighted,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PivotPrefixConfig {
    pub block_size: usize,
    pub pivot_count: usize,
    pub prefix_len: usize,
    pub top_blocks: usize,
    pub candidate_pool: usize,
    pub mode: PrefixScoreMode,
    pub cluster_iters: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PivotPrefixRouter {
    pub dim: usize,
    pub k_centroids: usize,
    pub block_size: usize,
    pub num_blocks: usize,
    pub num_pivots: usize,
    pub prefix_len: usize,
    pub top_blocks: usize,
    pub candidate_pool: usize,
    pub mode: PrefixScoreMode,
    pub pivots: Vec<f32>,
    pub block_prefix_pivots: Vec<u16>,
    pub posting_offsets: Vec<u32>,
    pub posting_block_ids: Vec<u32>,
    pub posting_positions: Vec<u8>,
    pub idf: Vec<f32>,
    pub block_offsets: Vec<u32>,
    pub block_payload: Vec<u32>,
    pub block_representatives: Vec<f32>,
    pub centroid_vectors: Vec<f32>,
    pub build_time_s: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DensePivotSketch {
    pub dim: usize,
    pub k_centroids: usize,
    pub block_size: usize,
    pub num_blocks: usize,
    pub num_pivots: usize,
    pub top_blocks: usize,
    pub candidate_pool: usize,
    pub pivots: Vec<f32>,
    pub block_pivot_distances: Vec<f32>,
    pub block_offsets: Vec<u32>,
    pub block_payload: Vec<u32>,
    pub block_representatives: Vec<f32>,
    pub centroid_vectors: Vec<f32>,
    pub build_time_s: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PivotRouteScratch {
    pub generation: u32,
    pub query_pivot_dist: Vec<f32>,
    pub query_prefix_pivots: Vec<u16>,
    pivot_order: Vec<usize>,
    pub block_scores: Vec<f32>,
    pub block_score_gen: Vec<u32>,
    pub block_seen_gen: Vec<u32>,
    pub touched_blocks: Vec<u32>,
    pub selected_blocks: Vec<u32>,
    pub candidate_centroids: Vec<u32>,
    pub centroid_scores: Vec<f32>,
    centroid_order: Vec<usize>,
    pub pool_candidates: Vec<u32>,
    pub final_nprobe: Vec<u32>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RouteMetrics {
    pub pivot_evals: usize,
    pub posting_entries_touched: usize,
    pub unique_blocks_touched: usize,
    pub duplicate_blocks: usize,
    pub duplicate_block_rate: f32,
    pub selected_blocks: usize,
    pub centroid_evals: usize,
    pub candidate_count: usize,
    pub working_set_bytes: usize,
    pub fallback: bool,
}

impl PivotPrefixRouter {
    pub fn build(centroids: &[f32], dim: usize, config: PivotPrefixConfig) -> Result<Self, String> {
        validate_matrix(centroids, dim)?;
        if config.block_size == 0 {
            return Err("block_size must be greater than zero".to_string());
        }
        if config.pivot_count == 0 {
            return Err("pivot_count must be greater than zero".to_string());
        }
        if config.prefix_len == 0 {
            return Err("prefix_len must be greater than zero".to_string());
        }

        let started = Instant::now();
        let (block_offsets, block_payload, block_representatives) =
            balanced_blocks(centroids, dim, config.block_size, config.cluster_iters);
        let num_blocks = block_offsets.len() - 1;
        let num_pivots = config.pivot_count.min(num_blocks);
        let prefix_len = config.prefix_len.min(num_pivots);
        let pivots = deterministic_seeds(&block_representatives, dim, num_pivots);
        let mut block_prefix_pivots = vec![0_u16; num_blocks * prefix_len];
        let mut buckets = vec![Vec::<(u32, u8)>::new(); num_pivots];

        for block in 0..num_blocks {
            let rep = row(&block_representatives, dim, block);
            let mut order = (0..num_pivots).collect::<Vec<_>>();
            order.sort_by(|&a, &b| {
                let da = l2_squared_unchecked(rep, row(&pivots, dim, a)).sqrt();
                let db = l2_squared_unchecked(rep, row(&pivots, dim, b)).sqrt();
                da.total_cmp(&db).then_with(|| a.cmp(&b))
            });
            for pos in 0..prefix_len {
                let pivot = order[pos];
                block_prefix_pivots[block * prefix_len + pos] = pivot as u16;
                buckets[pivot].push((block as u32, pos as u8));
            }
        }

        let mut posting_offsets = Vec::with_capacity(num_pivots + 1);
        let mut posting_block_ids = Vec::new();
        let mut posting_positions = Vec::new();
        let mut idf = Vec::with_capacity(num_pivots);
        posting_offsets.push(0);
        for bucket in buckets.iter_mut() {
            bucket.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
            let df = bucket.len() as f32;
            idf.push(((num_blocks as f32 + 1.0) / (df + 1.0)).ln() + 1.0);
            for &(block, pos) in bucket.iter() {
                posting_block_ids.push(block);
                posting_positions.push(pos);
            }
            posting_offsets.push(posting_block_ids.len() as u32);
        }

        Ok(Self {
            dim,
            k_centroids: centroids.len() / dim,
            block_size: config.block_size,
            num_blocks,
            num_pivots,
            prefix_len,
            top_blocks: config.top_blocks,
            candidate_pool: config.candidate_pool,
            mode: config.mode,
            pivots,
            block_prefix_pivots,
            posting_offsets,
            posting_block_ids,
            posting_positions,
            idf,
            block_offsets,
            block_payload,
            block_representatives,
            centroid_vectors: centroids.to_vec(),
            build_time_s: started.elapsed().as_secs_f64(),
        })
    }

    pub fn scratch(&self) -> PivotRouteScratch {
        PivotRouteScratch::new(
            self.num_pivots,
            self.prefix_len,
            self.num_blocks,
            self.posting_block_ids.len(),
            self.top_blocks,
            self.candidate_pool.max(self.top_blocks * self.block_size),
        )
    }

    pub fn route(&self, query: &[f32], scratch: &mut PivotRouteScratch) -> RouteMetrics {
        assert_eq!(query.len(), self.dim);
        scratch.next_generation();
        scratch.query_prefix_pivots.clear();
        scratch.touched_blocks.clear();
        scratch.selected_blocks.clear();
        scratch.candidate_centroids.clear();
        scratch.centroid_scores.clear();
        scratch.centroid_order.clear();
        scratch.pool_candidates.clear();
        scratch.final_nprobe.clear();

        for pivot in 0..self.num_pivots {
            scratch.query_pivot_dist[pivot] =
                l2_squared_unchecked(query, row(&self.pivots, self.dim, pivot)).sqrt();
            scratch.pivot_order[pivot] = pivot;
        }
        scratch.pivot_order[..self.num_pivots].sort_unstable_by(|&a, &b| {
            scratch.query_pivot_dist[a]
                .total_cmp(&scratch.query_pivot_dist[b])
                .then_with(|| a.cmp(&b))
        });
        for &pivot in scratch.pivot_order[..self.prefix_len].iter() {
            scratch.query_prefix_pivots.push(pivot as u16);
        }

        let mut entries = 0_usize;
        let mut duplicates = 0_usize;
        for (qpos, &pivot) in scratch.query_prefix_pivots.iter().enumerate() {
            let pivot = pivot as usize;
            let start = self.posting_offsets[pivot] as usize;
            let end = self.posting_offsets[pivot + 1] as usize;
            entries += end - start;
            for idx in start..end {
                let block = self.posting_block_ids[idx] as usize;
                if scratch.block_seen_gen[block] == scratch.generation {
                    duplicates += 1;
                } else {
                    scratch.block_seen_gen[block] = scratch.generation;
                }
                if scratch.block_score_gen[block] != scratch.generation {
                    scratch.block_score_gen[block] = scratch.generation;
                    scratch.block_scores[block] = 0.0;
                    scratch.touched_blocks.push(block as u32);
                }
                let bpos = self.posting_positions[idx] as usize;
                let weight = match self.mode {
                    PrefixScoreMode::Weighted => {
                        self.idf[pivot] / ((qpos + 1) as f32 * (bpos + 1) as f32)
                    }
                    PrefixScoreMode::Union => 1.0 / (qpos + bpos + 1) as f32,
                };
                scratch.block_scores[block] += weight;
            }
        }

        let unique_blocks = scratch.touched_blocks.len();
        if unique_blocks == 0 {
            let limit = self.top_blocks.min(self.num_blocks);
            for block in 0..limit {
                scratch.selected_blocks.push(block as u32);
            }
        } else {
            scratch.touched_blocks.sort_unstable_by(|&a, &b| {
                let sa = scratch.block_scores[a as usize];
                let sb = scratch.block_scores[b as usize];
                sb.total_cmp(&sa).then_with(|| a.cmp(&b))
            });
            let limit = self.top_blocks.min(unique_blocks);
            scratch
                .selected_blocks
                .extend_from_slice(&scratch.touched_blocks[..limit]);
        }

        scan_and_rerank(
            query,
            self.dim,
            &self.centroid_vectors,
            &self.block_offsets,
            &self.block_payload,
            self.candidate_pool,
            scratch,
        );

        let centroid_evals = scratch.candidate_centroids.len();
        RouteMetrics {
            pivot_evals: self.num_pivots,
            posting_entries_touched: entries,
            unique_blocks_touched: unique_blocks,
            duplicate_blocks: duplicates,
            duplicate_block_rate: if entries == 0 {
                0.0
            } else {
                duplicates as f32 / entries as f32
            },
            selected_blocks: scratch.selected_blocks.len(),
            centroid_evals,
            candidate_count: scratch.pool_candidates.len(),
            working_set_bytes: self.query_working_set_bytes(entries, unique_blocks, centroid_evals),
            fallback: scratch.final_nprobe.len() < DEFAULT_FINAL_NPROBE.min(self.k_centroids),
        }
    }

    pub fn resident_bytes(&self) -> usize {
        self.centroid_vectors.len() * size_of::<f32>()
            + self.block_payload.len() * size_of::<u32>()
            + self.block_offsets.len() * size_of::<u32>()
            + self.block_representatives.len() * size_of::<f32>()
            + self.pivots.len() * size_of::<f32>()
            + self.block_prefix_pivots.len() * size_of::<u16>()
            + self.posting_offsets.len() * size_of::<u32>()
            + self.posting_block_ids.len() * size_of::<u32>()
            + self.posting_positions.len() * size_of::<u8>()
            + self.idf.len() * size_of::<f32>()
    }

    fn query_working_set_bytes(
        &self,
        posting_entries: usize,
        unique_blocks: usize,
        centroid_evals: usize,
    ) -> usize {
        self.num_pivots * self.dim * size_of::<f32>()
            + posting_entries * (size_of::<u32>() + size_of::<u8>())
            + unique_blocks * (size_of::<f32>() + 2 * size_of::<u32>())
            + centroid_evals * self.dim * size_of::<f32>()
    }
}

impl DensePivotSketch {
    pub fn build(
        centroids: &[f32],
        dim: usize,
        block_size: usize,
        pivot_count: usize,
        top_blocks: usize,
        candidate_pool: usize,
        cluster_iters: usize,
    ) -> Result<Self, String> {
        validate_matrix(centroids, dim)?;
        if block_size == 0 || pivot_count == 0 {
            return Err("block_size and pivot_count must be greater than zero".to_string());
        }
        let started = Instant::now();
        let (block_offsets, block_payload, block_representatives) =
            balanced_blocks(centroids, dim, block_size, cluster_iters);
        let num_blocks = block_offsets.len() - 1;
        let num_pivots = pivot_count.min(num_blocks);
        let pivots = deterministic_seeds(&block_representatives, dim, num_pivots);
        let mut block_pivot_distances = vec![0.0; num_blocks * num_pivots];
        for block in 0..num_blocks {
            for pivot in 0..num_pivots {
                block_pivot_distances[block * num_pivots + pivot] = l2_squared_unchecked(
                    row(&block_representatives, dim, block),
                    row(&pivots, dim, pivot),
                )
                .sqrt();
            }
        }
        Ok(Self {
            dim,
            k_centroids: centroids.len() / dim,
            block_size,
            num_blocks,
            num_pivots,
            top_blocks,
            candidate_pool,
            pivots,
            block_pivot_distances,
            block_offsets,
            block_payload,
            block_representatives,
            centroid_vectors: centroids.to_vec(),
            build_time_s: started.elapsed().as_secs_f64(),
        })
    }

    pub fn scratch(&self) -> PivotRouteScratch {
        PivotRouteScratch::new(
            self.num_pivots,
            0,
            self.num_blocks,
            self.num_blocks,
            self.top_blocks,
            self.candidate_pool.max(self.top_blocks * self.block_size),
        )
    }

    pub fn route(&self, query: &[f32], scratch: &mut PivotRouteScratch) -> RouteMetrics {
        assert_eq!(query.len(), self.dim);
        scratch.next_generation();
        scratch.touched_blocks.clear();
        scratch.selected_blocks.clear();
        scratch.candidate_centroids.clear();
        scratch.centroid_scores.clear();
        scratch.centroid_order.clear();
        scratch.pool_candidates.clear();
        scratch.final_nprobe.clear();

        for pivot in 0..self.num_pivots {
            scratch.query_pivot_dist[pivot] =
                l2_squared_unchecked(query, row(&self.pivots, self.dim, pivot)).sqrt();
        }
        for block in 0..self.num_blocks {
            let mut score = 0.0_f32;
            for pivot in 0..self.num_pivots {
                let delta = (self.block_pivot_distances[block * self.num_pivots + pivot]
                    - scratch.query_pivot_dist[pivot])
                    .abs();
                score = score.max(delta);
            }
            scratch.block_scores[block] = score;
            scratch.touched_blocks.push(block as u32);
        }
        scratch.touched_blocks.sort_unstable_by(|&a, &b| {
            scratch.block_scores[a as usize]
                .total_cmp(&scratch.block_scores[b as usize])
                .then_with(|| a.cmp(&b))
        });
        let limit = self.top_blocks.min(self.num_blocks);
        scratch
            .selected_blocks
            .extend_from_slice(&scratch.touched_blocks[..limit]);
        scan_and_rerank(
            query,
            self.dim,
            &self.centroid_vectors,
            &self.block_offsets,
            &self.block_payload,
            self.candidate_pool,
            scratch,
        );
        let centroid_evals = scratch.candidate_centroids.len();
        RouteMetrics {
            pivot_evals: self.num_pivots,
            posting_entries_touched: 0,
            unique_blocks_touched: self.num_blocks,
            duplicate_blocks: 0,
            duplicate_block_rate: 0.0,
            selected_blocks: scratch.selected_blocks.len(),
            centroid_evals,
            candidate_count: scratch.pool_candidates.len(),
            working_set_bytes: self.query_working_set_bytes(centroid_evals),
            fallback: scratch.final_nprobe.len() < DEFAULT_FINAL_NPROBE.min(self.k_centroids),
        }
    }

    pub fn resident_bytes(&self) -> usize {
        self.centroid_vectors.len() * size_of::<f32>()
            + self.block_payload.len() * size_of::<u32>()
            + self.block_offsets.len() * size_of::<u32>()
            + self.block_representatives.len() * size_of::<f32>()
            + self.pivots.len() * size_of::<f32>()
            + self.block_pivot_distances.len() * size_of::<f32>()
    }

    fn query_working_set_bytes(&self, centroid_evals: usize) -> usize {
        self.num_pivots * self.dim * size_of::<f32>()
            + self.num_blocks * self.num_pivots * size_of::<f32>()
            + self.num_blocks * (size_of::<f32>() + size_of::<u32>())
            + centroid_evals * self.dim * size_of::<f32>()
    }
}

impl PivotRouteScratch {
    fn new(
        num_pivots: usize,
        prefix_len: usize,
        num_blocks: usize,
        touched_capacity: usize,
        top_blocks: usize,
        candidate_capacity: usize,
    ) -> Self {
        Self {
            generation: 0,
            query_pivot_dist: vec![0.0; num_pivots],
            query_prefix_pivots: Vec::with_capacity(prefix_len),
            pivot_order: vec![0; num_pivots],
            block_scores: vec![0.0; num_blocks],
            block_score_gen: vec![0; num_blocks],
            block_seen_gen: vec![0; num_blocks],
            touched_blocks: Vec::with_capacity(touched_capacity.max(num_blocks)),
            selected_blocks: Vec::with_capacity(top_blocks),
            candidate_centroids: Vec::with_capacity(candidate_capacity),
            centroid_scores: Vec::with_capacity(candidate_capacity),
            centroid_order: Vec::with_capacity(candidate_capacity),
            pool_candidates: Vec::with_capacity(candidate_capacity),
            final_nprobe: Vec::with_capacity(DEFAULT_FINAL_NPROBE),
        }
    }

    fn next_generation(&mut self) {
        if self.generation == u32::MAX {
            self.block_score_gen.fill(0);
            self.block_seen_gen.fill(0);
            self.generation = 1;
        } else {
            self.generation += 1;
        }
    }
}

fn scan_and_rerank(
    query: &[f32],
    dim: usize,
    centroids: &[f32],
    block_offsets: &[u32],
    block_payload: &[u32],
    candidate_pool: usize,
    scratch: &mut PivotRouteScratch,
) {
    for &block in &scratch.selected_blocks {
        let block = block as usize;
        let start = block_offsets[block] as usize;
        let end = block_offsets[block + 1] as usize;
        for &centroid in &block_payload[start..end] {
            let centroid = centroid as usize;
            scratch.candidate_centroids.push(centroid as u32);
            scratch
                .centroid_scores
                .push(l2_squared_unchecked(query, row(centroids, dim, centroid)));
        }
    }
    scratch.centroid_order.clear();
    scratch
        .centroid_order
        .extend(0..scratch.candidate_centroids.len());
    scratch.centroid_order.sort_unstable_by(|&a, &b| {
        scratch.centroid_scores[a]
            .total_cmp(&scratch.centroid_scores[b])
            .then_with(|| scratch.candidate_centroids[a].cmp(&scratch.candidate_centroids[b]))
    });
    let pool_limit = candidate_pool.min(scratch.centroid_order.len());
    for &idx in &scratch.centroid_order[..pool_limit] {
        scratch
            .pool_candidates
            .push(scratch.candidate_centroids[idx]);
    }
    let final_limit = DEFAULT_FINAL_NPROBE.min(scratch.pool_candidates.len());
    scratch
        .final_nprobe
        .extend_from_slice(&scratch.pool_candidates[..final_limit]);
}

pub fn sample_centroids(xb: &[f32], dim: usize, k: usize) -> Result<Vec<f32>, String> {
    validate_matrix(xb, dim)?;
    let n = xb.len() / dim;
    if k >= n {
        return Ok(xb.to_vec());
    }
    let mut out = Vec::with_capacity(k * dim);
    for i in 0..k {
        let idx = if k == 1 { 0 } else { i * (n - 1) / (k - 1) };
        out.extend_from_slice(row(xb, dim, idx));
    }
    Ok(out)
}

pub fn exact_topk(vectors: &[f32], dim: usize, queries: &[f32], k: usize) -> Vec<Vec<u32>> {
    let n = vectors.len() / dim;
    let mut out = Vec::with_capacity(queries.len() / dim);
    let mut scored = Vec::with_capacity(n);
    for query in queries.chunks_exact(dim) {
        scored.clear();
        for idx in 0..n {
            scored.push((
                idx as u32,
                l2_squared_unchecked(query, row(vectors, dim, idx)),
            ));
        }
        scored.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        out.push(scored.iter().take(k.min(n)).map(|(idx, _)| *idx).collect());
    }
    out
}

pub fn coverage(results: &[Vec<u32>], exact: &[Vec<u32>]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    for (got, want) in results.iter().zip(exact) {
        let mut hits = 0_usize;
        for id in got {
            if want.contains(id) {
                hits += 1;
            }
        }
        sum += hits as f64 / want.len().max(1) as f64;
    }
    sum / results.len() as f64
}

fn balanced_blocks(
    centroids: &[f32],
    dim: usize,
    block_size: usize,
    iters: usize,
) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
    let n = centroids.len() / dim;
    let block_count = n.div_ceil(block_size);
    let mut reps = deterministic_seeds(centroids, dim, block_count);
    let mut assign = vec![0_usize; n];
    let mut dists = vec![0.0_f32; n * block_count];
    let mut order = vec![0_usize; block_count];
    let mut point_order = vec![0_usize; n];
    let mut margins = vec![0.0_f32; n];
    for _ in 0..iters {
        for point in 0..n {
            for block in 0..block_count {
                dists[point * block_count + block] =
                    l2_squared_unchecked(row(centroids, dim, point), row(&reps, dim, block));
            }
            order.iter_mut().enumerate().for_each(|(i, slot)| *slot = i);
            order.sort_by(|&a, &b| {
                dists[point * block_count + a]
                    .total_cmp(&dists[point * block_count + b])
                    .then_with(|| a.cmp(&b))
            });
            let best = order[0];
            let second = if block_count > 1 { order[1] } else { best };
            margins[point] =
                dists[point * block_count + second] - dists[point * block_count + best];
        }
        point_order
            .iter_mut()
            .enumerate()
            .for_each(|(i, slot)| *slot = i);
        point_order.sort_by(|&a, &b| margins[a].total_cmp(&margins[b]).then_with(|| a.cmp(&b)));
        let mut counts = vec![0_usize; block_count];
        for &point in &point_order {
            order.iter_mut().enumerate().for_each(|(i, slot)| *slot = i);
            order.sort_by(|&a, &b| {
                dists[point * block_count + a]
                    .total_cmp(&dists[point * block_count + b])
                    .then_with(|| a.cmp(&b))
            });
            for &choice in &order {
                if counts[choice] < block_size {
                    assign[point] = choice;
                    counts[choice] += 1;
                    break;
                }
            }
        }
        recompute_reps(centroids, dim, &assign, block_count, &mut reps);
    }

    let mut sorted_blocks = (0..block_count).collect::<Vec<_>>();
    sorted_blocks.sort_by(|&a, &b| {
        reps[a * dim]
            .total_cmp(&reps[b * dim])
            .then_with(|| a.cmp(&b))
    });
    let mut remap = vec![0_usize; block_count];
    for (new_id, &old_id) in sorted_blocks.iter().enumerate() {
        remap[old_id] = new_id;
    }
    for value in &mut assign {
        *value = remap[*value];
    }

    let mut offsets = Vec::with_capacity(block_count + 1);
    let mut payload = Vec::with_capacity(n);
    let mut final_reps = vec![0.0_f32; block_count * dim];
    offsets.push(0);
    for block in 0..block_count {
        let members = (0..n)
            .filter(|&idx| assign[idx] == block)
            .collect::<Vec<_>>();
        let mut center = vec![0.0_f32; dim];
        for &member in &members {
            for d in 0..dim {
                center[d] += centroids[member * dim + d];
            }
        }
        if !members.is_empty() {
            let denom = members.len() as f32;
            for d in 0..dim {
                center[d] /= denom;
                final_reps[block * dim + d] = center[d];
            }
        }
        let mut ordered = members;
        ordered.sort_by(|&a, &b| {
            l2_squared_unchecked(row(centroids, dim, a), &center)
                .total_cmp(&l2_squared_unchecked(row(centroids, dim, b), &center))
                .then_with(|| a.cmp(&b))
        });
        payload.extend(ordered.into_iter().map(|idx| idx as u32));
        offsets.push(payload.len() as u32);
    }
    (offsets, payload, final_reps)
}

fn deterministic_seeds(points: &[f32], dim: usize, count: usize) -> Vec<f32> {
    let n = points.len() / dim;
    if count >= n {
        return points.to_vec();
    }
    let mut center = vec![0.0_f32; dim];
    for point in points.chunks_exact(dim) {
        for d in 0..dim {
            center[d] += point[d];
        }
    }
    for value in &mut center {
        *value /= n as f32;
    }
    let mut first = 0_usize;
    let mut first_dist = f32::INFINITY;
    for idx in 0..n {
        let dist = l2_squared_unchecked(row(points, dim, idx), &center);
        if dist < first_dist {
            first = idx;
            first_dist = dist;
        }
    }
    let mut chosen = vec![first];
    let mut nearest = (0..n)
        .map(|idx| l2_squared_unchecked(row(points, dim, idx), row(points, dim, first)))
        .collect::<Vec<_>>();
    while chosen.len() < count {
        for &idx in &chosen {
            nearest[idx] = -1.0;
        }
        let mut next = 0_usize;
        let mut best = f32::NEG_INFINITY;
        for (idx, &dist) in nearest.iter().enumerate() {
            if dist > best {
                next = idx;
                best = dist;
            }
        }
        chosen.push(next);
        for (idx, value) in nearest.iter_mut().enumerate() {
            *value = value.min(l2_squared_unchecked(
                row(points, dim, idx),
                row(points, dim, next),
            ));
        }
    }
    let mut out = Vec::with_capacity(count * dim);
    for idx in chosen {
        out.extend_from_slice(row(points, dim, idx));
    }
    out
}

fn recompute_reps(
    points: &[f32],
    dim: usize,
    assign: &[usize],
    block_count: usize,
    reps: &mut [f32],
) {
    reps.fill(0.0);
    let mut counts = vec![0_usize; block_count];
    for (idx, &block) in assign.iter().enumerate() {
        counts[block] += 1;
        for d in 0..dim {
            reps[block * dim + d] += points[idx * dim + d];
        }
    }
    for block in 0..block_count {
        if counts[block] > 0 {
            let denom = counts[block] as f32;
            for d in 0..dim {
                reps[block * dim + d] /= denom;
            }
        }
    }
}

fn validate_matrix(values: &[f32], dim: usize) -> Result<(), String> {
    if dim == 0 {
        return Err("dim must be greater than zero".to_string());
    }
    if values.is_empty() || values.len() % dim != 0 {
        return Err("matrix payload must be non-empty and divisible by dim".to_string());
    }
    Ok(())
}

fn row(values: &[f32], dim: usize, idx: usize) -> &[f32] {
    &values[idx * dim..(idx + 1) * dim]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_without_growing_scratch_buffers() {
        let centroids = vec![
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 10.0, 10.0, 11.0, 10.0, 10.0, 11.0,
        ];
        let router = PivotPrefixRouter::build(
            &centroids,
            2,
            PivotPrefixConfig {
                block_size: 2,
                pivot_count: 2,
                prefix_len: 1,
                top_blocks: 1,
                candidate_pool: 2,
                mode: PrefixScoreMode::Union,
                cluster_iters: 2,
            },
        )
        .unwrap();
        let mut scratch = router.scratch();
        let capacities = (
            scratch.touched_blocks.capacity(),
            scratch.selected_blocks.capacity(),
            scratch.candidate_centroids.capacity(),
            scratch.pool_candidates.capacity(),
        );
        let metrics = router.route(&[0.2, 0.1], &mut scratch);
        assert_eq!(metrics.selected_blocks, 1);
        assert_eq!(scratch.final_nprobe.len(), 2);
        assert_eq!(
            capacities,
            (
                scratch.touched_blocks.capacity(),
                scratch.selected_blocks.capacity(),
                scratch.candidate_centroids.capacity(),
                scratch.pool_candidates.capacity(),
            )
        );
    }

    #[test]
    fn dense_fallback_routes_nearby_centroids() {
        let centroids = vec![
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 10.0, 10.0, 11.0, 10.0, 10.0, 11.0,
        ];
        let router = DensePivotSketch::build(&centroids, 2, 2, 2, 1, 2, 2).unwrap();
        let mut scratch = router.scratch();
        router.route(&[10.2, 10.1], &mut scratch);
        assert!(scratch.final_nprobe.contains(&3));
    }
}
