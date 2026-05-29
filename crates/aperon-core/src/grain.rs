use crate::binary::LegacySingleGrain;
use crate::layout::{BlockSoaLayout, VectorId};
use crate::quantization::{
    quantize_i16, quantize_i8, quantize_u16, scale_for_i16, scale_for_i8, scale_for_u16,
    scaled_weight,
};
use crate::scan::{scan_block_into, ScanWeights};

/// Stable identifier for a local Aperon grain.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct GrainId(u64);

impl GrainId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Local vector group scanned with pointer-free compact layouts.
#[derive(Clone, Debug)]
pub struct Grain {
    id: GrainId,
    layout: BlockSoaLayout,
    source_dim: usize,
    mean: Vec<f32>,
    projection: Vec<f32>,
    proj_scales: Vec<f32>,
    residual_scale: f32,
    sketch_projection: Vec<f32>,
    sketch_scales: Vec<f32>,
    distance_unit: f64,
    coord_weights: Vec<i64>,
    residual_weight: i64,
    sketch_weights: Vec<i64>,
}

impl Grain {
    pub fn new(id: GrainId, dim: usize) -> Self {
        Self {
            id,
            layout: BlockSoaLayout::new(dim),
            source_dim: dim,
            mean: vec![0.0; dim],
            projection: identity_projection(dim, dim),
            proj_scales: vec![1.0; dim],
            residual_scale: 1.0,
            sketch_projection: Vec::new(),
            sketch_scales: Vec::new(),
            distance_unit: 1.0,
            coord_weights: vec![1; dim],
            residual_weight: 1,
            sketch_weights: Vec::new(),
        }
    }

    pub fn build(
        id: GrainId,
        vectors: &[Vec<f32>],
        ids: &[VectorId],
        source_dim: usize,
        local_dim: usize,
        sketch_dim: usize,
        block_size: usize,
    ) -> Result<Self, String> {
        if vectors.len() != ids.len() {
            return Err("vectors and ids length mismatch".to_string());
        }
        for vector in vectors {
            if vector.len() != source_dim {
                return Err(format!(
                    "dimension mismatch: expected {}, got {}",
                    source_dim,
                    vector.len()
                ));
            }
        }

        let local_dim = local_dim.min(source_dim);
        let (mean, projection) = run_pca(vectors, source_dim, local_dim);
        let centered = center_vectors(vectors, &mean);
        let coords = project_matrix(&centered, source_dim, &projection, local_dim);

        let mut proj_scales = vec![1.0; local_dim];
        let mut qcoords = vec![0; vectors.len() * local_dim];
        for k in 0..local_dim {
            let abs_max = (0..vectors.len())
                .map(|i| coords[i * local_dim + k].abs())
                .fold(0.0_f32, f32::max);
            let scale = scale_for_i16(abs_max);
            proj_scales[k] = scale;
            for i in 0..vectors.len() {
                qcoords[i * local_dim + k] = quantize_i16(coords[i * local_dim + k], scale);
            }
        }

        let residuals = residual_matrix(&centered, source_dim, &coords, &projection, local_dim);
        let residual_norms = residual_norms(&residuals, source_dim);
        let residual_scale = scale_for_u16(residual_norms.iter().copied().fold(0.0_f32, f32::max));
        let qresiduals = residual_norms
            .iter()
            .map(|value| quantize_u16(*value, residual_scale))
            .collect::<Vec<_>>();

        let (_, sketch_projection) = run_pca_flat(
            &residuals,
            residuals.len() / source_dim,
            source_dim,
            sketch_dim,
        );
        let sketches = project_matrix(&residuals, source_dim, &sketch_projection, sketch_dim);
        let mut sketch_scales = vec![1.0; sketch_dim];
        let mut qsketches = vec![0; vectors.len() * sketch_dim];
        for m in 0..sketch_dim {
            let abs_max = (0..vectors.len())
                .map(|i| sketches[i * sketch_dim + m].abs())
                .fold(0.0_f32, f32::max);
            let scale = scale_for_i8(abs_max);
            sketch_scales[m] = scale;
            for i in 0..vectors.len() {
                qsketches[i * sketch_dim + m] = quantize_i8(sketches[i * sketch_dim + m], scale);
            }
        }

        let mut layout = BlockSoaLayout::with_shape(local_dim, sketch_dim, block_size);
        for i in 0..vectors.len() {
            let coord_start = i * local_dim;
            let sketch_start = i * sketch_dim;
            layout.push_quantized(
                ids[i],
                &qcoords[coord_start..coord_start + local_dim],
                qresiduals[i],
                &qsketches[sketch_start..sketch_start + sketch_dim],
            )?;
        }

        let mut grain = Self {
            id,
            layout,
            source_dim,
            mean,
            projection,
            proj_scales,
            residual_scale,
            sketch_projection,
            sketch_scales,
            distance_unit: 1.0,
            coord_weights: Vec::new(),
            residual_weight: 1,
            sketch_weights: Vec::new(),
        };
        grain.build_distance_weights();
        Ok(grain)
    }

    pub fn from_legacy_single(id: GrainId, legacy: LegacySingleGrain) -> Result<Self, String> {
        let source_dim = legacy.dimension as usize;
        let local_dim = legacy.local_dim as usize;
        let sketch_dim = legacy.sketch_dim as usize;
        let block_size = legacy.block_size as usize;
        if source_dim == 0 || local_dim == 0 || block_size == 0 {
            return Err("dimension, local_dim, and block_size must be nonzero".to_string());
        }

        expect_len(&legacy.mean, source_dim, "mean")?;
        expect_len(&legacy.projection, source_dim * local_dim, "projection")?;
        expect_len(&legacy.proj_scales, local_dim, "proj_scales")?;
        expect_len(
            &legacy.sketch_projection,
            source_dim * sketch_dim,
            "sketch_projection",
        )?;
        expect_len(&legacy.sketch_scales, sketch_dim, "sketch_scales")?;

        let layout = BlockSoaLayout::from_raw_block_bytes(
            local_dim,
            sketch_dim,
            block_size,
            legacy.num_vectors as usize,
            &legacy.block_data,
        )?;
        let mut grain = Self {
            id,
            layout,
            source_dim,
            mean: legacy.mean,
            projection: legacy.projection,
            proj_scales: legacy.proj_scales,
            residual_scale: legacy.residual_scale,
            sketch_projection: legacy.sketch_projection,
            sketch_scales: legacy.sketch_scales,
            distance_unit: 1.0,
            coord_weights: Vec::new(),
            residual_weight: 1,
            sketch_weights: Vec::new(),
        };
        grain.build_distance_weights();
        Ok(grain)
    }

    pub const fn id(&self) -> GrainId {
        self.id
    }

    pub fn dim(&self) -> usize {
        self.source_dim
    }

    pub fn local_dim(&self) -> usize {
        self.layout.local_dim()
    }

    pub fn sketch_dim(&self) -> usize {
        self.layout.sketch_dim()
    }

    pub fn len(&self) -> usize {
        self.layout.len()
    }

    pub fn block_size(&self) -> usize {
        self.layout.block_size()
    }

    pub fn centroid(&self) -> &[f32] {
        &self.mean
    }

    pub fn mean(&self) -> &[f32] {
        &self.mean
    }

    pub fn projection(&self) -> &[f32] {
        &self.projection
    }

    pub fn proj_scales(&self) -> &[f32] {
        &self.proj_scales
    }

    pub fn residual_scale(&self) -> f32 {
        self.residual_scale
    }

    pub fn sketch_projection(&self) -> &[f32] {
        &self.sketch_projection
    }

    pub fn sketch_scales(&self) -> &[f32] {
        &self.sketch_scales
    }

    /// Serialize the block data to the on-disk wire format.
    pub fn raw_block_bytes(&self) -> Vec<u8> {
        self.layout.raw_block_bytes()
    }

    pub fn vector_ids(&self) -> &[VectorId] {
        self.layout.ids()
    }

    pub fn is_empty(&self) -> bool {
        self.layout.is_empty()
    }

    pub fn insert(&mut self, id: VectorId, vector: impl Into<Vec<f32>>) -> Result<(), String> {
        let vector = vector.into();
        let projected = self.project_query(&vector)?;
        self.layout.push_quantized(
            id,
            &projected.coords,
            projected.residual,
            &projected.sketches,
        )
    }

    pub fn scan(&self, query: &[f32], top_k: usize) -> Result<Vec<ScoredVector>, String> {
        let projected = self.project_query(query)?;
        let mut out = Vec::new();
        let mut distances = vec![0; self.layout.block_size()];
        for block in 0..self.layout.block_count() {
            let lanes = self.layout.block_len(block);
            scan_block_into(
                &self.layout,
                block,
                lanes,
                &projected.coords,
                projected.residual,
                &projected.sketches,
                ScanWeights {
                    coord: &self.coord_weights,
                    residual: self.residual_weight,
                    sketch: &self.sketch_weights,
                },
                &mut distances,
            );
            for (lane, dist) in distances.iter().copied().take(lanes).enumerate() {
                if let Some(id) = self.layout.id_at(block * self.layout.block_size() + lane) {
                    out.push(ScoredVector {
                        id,
                        distance: dist as f64 * self.distance_unit,
                    });
                }
            }
        }
        out.sort_by(|a, b| {
            a.distance
                .total_cmp(&b.distance)
                .then_with(|| a.id.cmp(&b.id))
        });
        out.truncate(top_k);
        Ok(out)
    }

    fn project_query(&self, query: &[f32]) -> Result<QueryProjection, String> {
        if query.len() != self.source_dim {
            return Err(format!(
                "dimension mismatch: expected {}, got {}",
                self.source_dim,
                query.len()
            ));
        }

        let centered = query
            .iter()
            .zip(&self.mean)
            .map(|(value, mean)| value - mean)
            .collect::<Vec<_>>();
        let z = project_one(&centered, &self.projection, self.local_dim());
        let coords = z
            .iter()
            .zip(&self.proj_scales)
            .map(|(value, scale)| quantize_i16(*value, *scale))
            .collect::<Vec<_>>();

        let mut recon = vec![0.0; self.source_dim];
        for (d, recon_value) in recon.iter_mut().enumerate() {
            for (k, value) in z.iter().enumerate() {
                *recon_value += value * self.projection[d * self.local_dim() + k];
            }
        }

        let residual = centered
            .iter()
            .zip(&recon)
            .map(|(value, recon)| value - recon)
            .collect::<Vec<_>>();
        let residual_norm = residual.iter().map(|value| value * value).sum::<f32>();
        let qresidual = quantize_u16(residual_norm, self.residual_scale);
        let sketch_values = project_one(&residual, &self.sketch_projection, self.sketch_dim());
        let sketches = sketch_values
            .iter()
            .zip(&self.sketch_scales)
            .map(|(value, scale)| quantize_i8(*value, *scale))
            .collect::<Vec<_>>();

        Ok(QueryProjection {
            coords,
            residual: qresidual,
            sketches,
        })
    }

    fn build_distance_weights(&mut self) {
        let mut max_scale_sq = 0.0_f64;
        let mut unit = f64::MAX;

        for scale in &self.proj_scales {
            if *scale > 0.0 {
                let sq = f64::from(*scale) * f64::from(*scale);
                unit = unit.min(sq);
                max_scale_sq = max_scale_sq.max(sq);
            }
        }
        if self.residual_scale > 0.0 {
            unit = unit.min(f64::from(self.residual_scale));
        }
        for scale in &self.sketch_scales {
            if *scale > 0.0 {
                unit = unit.min(f64::from(*scale) * f64::from(*scale));
            }
        }
        if !unit.is_finite() || unit <= 0.0 {
            unit = 1.0;
        }
        if max_scale_sq > 0.0 {
            unit = unit.max(max_scale_sq / f64::from(i32::MAX));
        }

        self.distance_unit = unit;
        self.coord_weights = self
            .proj_scales
            .iter()
            .map(|scale| scaled_weight(f64::from(*scale) * f64::from(*scale), unit, 32767))
            .collect();
        self.residual_weight = scaled_weight(f64::from(self.residual_scale), unit, u32::MAX as i64);
        self.sketch_weights = self
            .sketch_scales
            .iter()
            .map(|scale| scaled_weight(f64::from(*scale) * f64::from(*scale), unit, 8_388_607))
            .collect();
    }
}

fn expect_len<T>(slice: &[T], expected: usize, name: &str) -> Result<(), String> {
    if slice.len() == expected {
        Ok(())
    } else {
        Err(format!(
            "{name} length mismatch: expected {}, got {}",
            expected,
            slice.len()
        ))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScoredVector {
    pub id: VectorId,
    pub distance: f64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct QueryProjection {
    coords: Vec<i16>,
    residual: u16,
    sketches: Vec<i8>,
}

fn center_vectors(vectors: &[Vec<f32>], mean: &[f32]) -> Vec<f32> {
    vectors
        .iter()
        .flat_map(|vector| vector.iter().zip(mean).map(|(value, mean)| value - mean))
        .collect()
}

fn deterministic_projection(dim: usize, k: usize) -> Vec<f32> {
    let mut projection = vec![0.0; dim * k];
    for component in 0..k.min(dim) {
        projection[component * k + component] = 1.0;
    }
    projection
}

fn identity_projection(dim: usize, k: usize) -> Vec<f32> {
    deterministic_projection(dim, k)
}

fn run_pca(vectors: &[Vec<f32>], dim: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    let flat = vectors
        .iter()
        .flat_map(|vector| vector.iter().copied())
        .collect::<Vec<_>>();
    run_pca_flat(&flat, vectors.len(), dim, k)
}

fn run_pca_flat(values: &[f32], rows: usize, dim: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    let mut mean = vec![0.0; dim];
    if rows == 0 || k == 0 {
        return (mean, vec![0.0; dim * k]);
    }

    for row in values.chunks_exact(dim).take(rows) {
        for (slot, value) in mean.iter_mut().zip(row) {
            *slot += value;
        }
    }
    for value in &mut mean {
        *value /= rows as f32;
    }

    let mut centered = Vec::with_capacity(rows * dim);
    for row in values.chunks_exact(dim).take(rows) {
        centered.extend(row.iter().zip(&mean).map(|(value, mean)| value - mean));
    }

    let mut projection = vec![0.0; dim * k];
    for component in 0..k {
        let mut v = deterministic_unit_vector(dim, component);

        for _ in 0..30 {
            let mut y = vec![0.0; rows];
            for row in 0..rows {
                for d in 0..dim {
                    y[row] += centered[row * dim + d] * v[d];
                }
            }

            let mut w = vec![0.0; dim];
            for d in 0..dim {
                for row in 0..rows {
                    w[d] += centered[row * dim + d] * y[row];
                }
            }

            let norm = w.iter().map(|value| value * value).sum::<f32>().sqrt();
            if norm < 1e-9 {
                break;
            }
            for d in 0..dim {
                v[d] = w[d] / norm;
            }
        }

        for d in 0..dim {
            projection[d * k + component] = v[d];
        }

        for row in 0..rows {
            let mut dot = 0.0;
            for d in 0..dim {
                dot += centered[row * dim + d] * v[d];
            }
            for d in 0..dim {
                centered[row * dim + d] -= dot * v[d];
            }
        }
    }

    (mean, projection)
}

fn deterministic_unit_vector(dim: usize, component: usize) -> Vec<f32> {
    let mut state = 42_u64 + component as u64;
    let mut vector = vec![0.0; dim];
    for value in &mut vector {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let unit = ((state >> 32) as u32) as f32 / u32::MAX as f32;
        *value = unit - 0.5;
    }

    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}

fn project_matrix(vectors: &[f32], dim: usize, projection: &[f32], k: usize) -> Vec<f32> {
    if k == 0 {
        return Vec::new();
    }
    let rows = vectors.len() / dim;
    let mut out = vec![0.0; rows * k];
    for row in 0..rows {
        out[row * k..row * k + k].copy_from_slice(&project_one(
            &vectors[row * dim..row * dim + dim],
            projection,
            k,
        ));
    }
    out
}

fn project_one(vector: &[f32], projection: &[f32], k: usize) -> Vec<f32> {
    let mut out = vec![0.0; k];
    for (component, out_value) in out.iter_mut().enumerate() {
        for (d, value) in vector.iter().enumerate() {
            *out_value += value * projection[d * k + component];
        }
    }
    out
}

fn residual_matrix(
    centered: &[f32],
    dim: usize,
    coords: &[f32],
    projection: &[f32],
    local_dim: usize,
) -> Vec<f32> {
    let rows = centered.len() / dim;
    let mut residuals = vec![0.0; centered.len()];
    for row in 0..rows {
        for d in 0..dim {
            let mut recon = 0.0;
            for k in 0..local_dim {
                recon += coords[row * local_dim + k] * projection[d * local_dim + k];
            }
            residuals[row * dim + d] = centered[row * dim + d] - recon;
        }
    }
    residuals
}

fn residual_norms(residuals: &[f32], dim: usize) -> Vec<f32> {
    residuals
        .chunks_exact(dim)
        .map(|row| row.iter().map(|value| value * value).sum())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn built_grain_scans_nearest_quantized_vector() {
        let vectors = vec![
            vec![0.0, 0.0, 0.0],
            vec![10.0, 0.0, 0.0],
            vec![0.0, 10.0, 0.0],
        ];
        let ids = vec![VectorId::new(1), VectorId::new(2), VectorId::new(3)];
        let grain = Grain::build(GrainId::new(0), &vectors, &ids, 3, 2, 1, 4).unwrap();

        let results = grain.scan(&[9.0, 0.0, 0.0], 1).unwrap();

        assert_eq!(results[0].id, VectorId::new(2));
    }

    #[test]
    fn pca_projection_captures_dominant_axis() {
        let vectors = vec![
            vec![-2.0, 0.0],
            vec![-1.0, 0.0],
            vec![1.0, 0.0],
            vec![2.0, 0.0],
        ];
        let ids = vec![
            VectorId::new(0),
            VectorId::new(1),
            VectorId::new(2),
            VectorId::new(3),
        ];
        let grain = Grain::build(GrainId::new(0), &vectors, &ids, 2, 1, 0, 4).unwrap();

        assert!(grain.projection[0].abs() > 0.99);
        assert!(grain.projection[1].abs() < 0.01);
    }
}
