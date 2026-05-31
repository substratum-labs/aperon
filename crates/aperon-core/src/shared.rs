use crate::layout::VectorId;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SharedQuantizationConfig {
    pub basis_cols: usize,
    pub local_dim: usize,
    pub pq_subquantizers: usize,
    pub pq_bits: u8,
    pub opq: bool,
}

impl SharedQuantizationConfig {
    pub fn new(
        dim: usize,
        basis_cols: usize,
        local_dim: usize,
        pq_subquantizers: usize,
        pq_bits: u8,
        opq: bool,
    ) -> Result<Self, String> {
        if basis_cols == 0 || local_dim == 0 {
            return Err("basis_cols and local_dim must be nonzero".to_string());
        }
        if pq_subquantizers == 0 || !dim.is_multiple_of(pq_subquantizers) {
            return Err("pq_subquantizers must be nonzero and divide dimension".to_string());
        }
        if !matches!(pq_bits, 4 | 8) {
            return Err("pq_bits must be 4 or 8".to_string());
        }
        Ok(Self {
            basis_cols: basis_cols.min(dim),
            local_dim: local_dim.min(basis_cols).min(dim),
            pq_subquantizers,
            pq_bits,
            opq,
        })
    }

    pub fn vocabulary(&self) -> usize {
        1_usize << self.pq_bits
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SharedQuantizer {
    pub config: SharedQuantizationConfig,
    pub basis: Vec<f32>,
    pub opq_rotation: Vec<f32>,
    pub pq_codebooks: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct SharedBuildGrain {
    pub ids: Vec<VectorId>,
    pub vectors: Vec<Vec<f32>>,
    pub mean: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SharedGrainEncoding {
    pub column_indices: Vec<usize>,
    pub coord_scales: Vec<f32>,
    pub qcoords: Vec<i16>,
    pub pq_codes: Vec<u8>,
    pub error_scale: f32,
    pub error_norms: Vec<u8>,
}

pub fn train_shared_quantizer(
    _vectors: &[Vec<f32>],
    grains: &[SharedBuildGrain],
    dim: usize,
    config: SharedQuantizationConfig,
) -> Result<(SharedQuantizer, Vec<SharedGrainEncoding>), String> {
    let flat = flatten_centered_grains(grains, dim)?;
    let (_mean, basis) = run_pca_flat(&flat, flat.len() / dim, dim, config.basis_cols);
    let mut encodings = Vec::with_capacity(grains.len());
    let mut all_residuals = Vec::new();

    for grain in grains {
        let encoding = encode_basis_only(grain, dim, &basis, &config)?;
        append_residuals(grain, dim, &basis, &encoding, &mut all_residuals);
        encodings.push(encoding);
    }

    let opq_rotation = if config.opq {
        let (_res_mean, rotation) =
            run_pca_flat(&all_residuals, all_residuals.len() / dim, dim, dim);
        rotation
    } else {
        identity_projection(dim, dim)
    };
    let rotated = rotate_rows(&all_residuals, dim, &opq_rotation);
    let pq_codebooks =
        train_pq_codebooks(&rotated, dim, config.pq_subquantizers, config.vocabulary());

    let quantizer = SharedQuantizer {
        config,
        basis,
        opq_rotation,
        pq_codebooks,
    };

    let mut offset = 0;
    for (grain, encoding) in grains.iter().zip(&mut encodings) {
        let rows = grain.vectors.len();
        let residual_rows = &rotated[offset * dim..(offset + rows) * dim];
        let codes = encode_pq(
            residual_rows,
            dim,
            quantizer.config.pq_subquantizers,
            quantizer.config.vocabulary(),
            &quantizer.pq_codebooks,
        );
        let mut errors = Vec::with_capacity(rows);
        for row in 0..rows {
            let decoded = decode_pq_rotated(
                &quantizer.pq_codebooks,
                dim,
                quantizer.config.pq_subquantizers,
                quantizer.config.vocabulary(),
                &quantizer.opq_rotation,
                &codes[row * quantizer.config.pq_subquantizers
                    ..(row + 1) * quantizer.config.pq_subquantizers],
            );
            let original = &all_residuals[(offset + row) * dim..(offset + row + 1) * dim];
            let err = original
                .iter()
                .zip(decoded)
                .map(|(a, b)| {
                    let diff = a - b;
                    diff * diff
                })
                .sum::<f32>();
            errors.push(err);
        }
        let max_error = errors.iter().copied().fold(0.0_f32, f32::max);
        let error_scale = if max_error > 0.0 {
            max_error / u8::MAX as f32
        } else {
            1.0
        };
        encoding.pq_codes = codes;
        encoding.error_scale = error_scale;
        encoding.error_norms = errors
            .into_iter()
            .map(|value| {
                (value / error_scale)
                    .round_ties_even()
                    .clamp(0.0, u8::MAX as f32) as u8
            })
            .collect();
        offset += rows;
    }

    Ok((quantizer, encodings))
}

#[allow(clippy::too_many_arguments)]
pub fn reconstruct_shared(
    mean: &[f32],
    basis: &[f32],
    dim: usize,
    basis_cols: usize,
    column_indices: &[usize],
    coord_scales: &[f32],
    qcoords: &[i16],
    pq_codebooks: &[f32],
    pq_subquantizers: usize,
    pq_bits: u8,
    opq_rotation: &[f32],
    pq_codes: &[u8],
) -> Vec<f32> {
    let mut recon = mean.to_vec();
    for (k, (&col, &scale)) in column_indices.iter().zip(coord_scales).enumerate() {
        let z = qcoords[k] as f32 * scale;
        for d in 0..dim {
            recon[d] += z * basis[d * basis_cols + col];
        }
    }

    let residual = decode_pq_rotated(
        pq_codebooks,
        dim,
        pq_subquantizers,
        1_usize << pq_bits,
        opq_rotation,
        pq_codes,
    );
    for (value, residual) in recon.iter_mut().zip(residual) {
        *value += residual;
    }
    recon
}

pub fn packed_pq_bytes(codes: &[u8], pq_bits: u8) -> Vec<u8> {
    if pq_bits == 8 {
        return codes.to_vec();
    }
    let mut out = vec![0_u8; codes.len().div_ceil(2)];
    for (idx, code) in codes.iter().enumerate() {
        if idx % 2 == 0 {
            out[idx / 2] |= code & 0x0f;
        } else {
            out[idx / 2] |= (code & 0x0f) << 4;
        }
    }
    out
}

pub fn unpack_pq_bytes(bytes: &[u8], codes: usize, pq_bits: u8) -> Vec<u8> {
    if pq_bits == 8 {
        return bytes.to_vec();
    }
    let mut out = Vec::with_capacity(codes);
    for byte in bytes {
        out.push(byte & 0x0f);
        if out.len() < codes {
            out.push(byte >> 4);
        }
    }
    out
}

fn encode_basis_only(
    grain: &SharedBuildGrain,
    dim: usize,
    basis: &[f32],
    config: &SharedQuantizationConfig,
) -> Result<SharedGrainEncoding, String> {
    let rows = grain.vectors.len();
    let basis_cols = config.basis_cols;
    let mut all_coords = vec![0.0_f32; rows * basis_cols];
    for (row_idx, vector) in grain.vectors.iter().enumerate() {
        if vector.len() != dim {
            return Err("dimension mismatch in shared grain build".to_string());
        }
        for col in 0..basis_cols {
            let mut dot = 0.0;
            for d in 0..dim {
                dot += (vector[d] - grain.mean[d]) * basis[d * basis_cols + col];
            }
            all_coords[row_idx * basis_cols + col] = dot;
        }
    }

    let mut variances = (0..basis_cols)
        .map(|col| {
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            for row in 0..rows {
                let value = all_coords[row * basis_cols + col];
                sum += value;
                sum_sq += value * value;
            }
            let mean = if rows > 0 { sum / rows as f32 } else { 0.0 };
            (col, sum_sq / rows.max(1) as f32 - mean * mean)
        })
        .collect::<Vec<_>>();
    variances.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let mut column_indices = variances
        .into_iter()
        .take(config.local_dim)
        .map(|(col, _)| col)
        .collect::<Vec<_>>();
    column_indices.sort_unstable();

    let mut coord_scales = vec![1.0; column_indices.len()];
    let mut qcoords = vec![0_i16; rows * column_indices.len()];
    for (k, &col) in column_indices.iter().enumerate() {
        let abs_max = (0..rows)
            .map(|row| all_coords[row * basis_cols + col].abs())
            .fold(0.0_f32, f32::max);
        let scale = if abs_max > 0.0 {
            abs_max / i8::MAX as f32
        } else {
            1.0
        };
        coord_scales[k] = scale;
        for row in 0..rows {
            let scaled = all_coords[row * basis_cols + col] / scale;
            qcoords[row * column_indices.len() + k] = scaled
                .round_ties_even()
                .clamp(i8::MIN as f32, i8::MAX as f32)
                as i16;
        }
    }

    Ok(SharedGrainEncoding {
        column_indices,
        coord_scales,
        qcoords,
        pq_codes: Vec::new(),
        error_scale: 1.0,
        error_norms: Vec::new(),
    })
}

fn append_residuals(
    grain: &SharedBuildGrain,
    dim: usize,
    basis: &[f32],
    encoding: &SharedGrainEncoding,
    out: &mut Vec<f32>,
) {
    let basis_cols = basis.len() / dim;
    for (row, vector) in grain.vectors.iter().enumerate() {
        let mut recon = grain.mean.clone();
        for (k, (&col, &scale)) in encoding
            .column_indices
            .iter()
            .zip(&encoding.coord_scales)
            .enumerate()
        {
            let z = encoding.qcoords[row * encoding.column_indices.len() + k] as f32 * scale;
            for d in 0..dim {
                recon[d] += z * basis[d * basis_cols + col];
            }
        }
        out.extend((0..dim).map(|d| vector[d] - recon[d]));
    }
}

fn train_pq_codebooks(
    values: &[f32],
    dim: usize,
    subquantizers: usize,
    vocabulary: usize,
) -> Vec<f32> {
    let rows = values.len() / dim;
    let subdim = dim / subquantizers;
    let mut codebooks = vec![0.0_f32; subquantizers * vocabulary * subdim];
    for m in 0..subquantizers {
        let mut points = Vec::with_capacity(rows * subdim);
        for row in 0..rows {
            points.extend_from_slice(&values[row * dim + m * subdim..row * dim + (m + 1) * subdim]);
        }
        let centroids = kmeans_flat(&points, rows, subdim, vocabulary.min(rows.max(1)));
        let centroid_count = centroids.len() / subdim;
        for code in 0..vocabulary {
            let src = (code % centroid_count.max(1)) * subdim;
            let dst = (m * vocabulary + code) * subdim;
            if centroids.is_empty() {
                continue;
            }
            codebooks[dst..dst + subdim].copy_from_slice(&centroids[src..src + subdim]);
        }
    }
    codebooks
}

fn encode_pq(
    values: &[f32],
    dim: usize,
    subquantizers: usize,
    vocabulary: usize,
    codebooks: &[f32],
) -> Vec<u8> {
    let rows = values.len() / dim;
    let subdim = dim / subquantizers;
    let mut codes = vec![0_u8; rows * subquantizers];
    for row in 0..rows {
        for m in 0..subquantizers {
            let point = &values[row * dim + m * subdim..row * dim + (m + 1) * subdim];
            let mut best = 0;
            let mut best_dist = f32::INFINITY;
            for code in 0..vocabulary {
                let centroid = &codebooks
                    [(m * vocabulary + code) * subdim..(m * vocabulary + code + 1) * subdim];
                let dist = point
                    .iter()
                    .zip(centroid)
                    .map(|(a, b)| {
                        let diff = a - b;
                        diff * diff
                    })
                    .sum::<f32>();
                if dist < best_dist {
                    best_dist = dist;
                    best = code;
                }
            }
            codes[row * subquantizers + m] = best as u8;
        }
    }
    codes
}

fn decode_pq_rotated(
    codebooks: &[f32],
    dim: usize,
    subquantizers: usize,
    vocabulary: usize,
    rotation: &[f32],
    codes: &[u8],
) -> Vec<f32> {
    let subdim = dim / subquantizers;
    let mut rotated = vec![0.0_f32; dim];
    for m in 0..subquantizers {
        let code = codes[m] as usize;
        let src = (m * vocabulary + code) * subdim;
        rotated[m * subdim..(m + 1) * subdim].copy_from_slice(&codebooks[src..src + subdim]);
    }
    let mut out = vec![0.0_f32; dim];
    for d in 0..dim {
        for r in 0..dim {
            out[d] += rotated[r] * rotation[d * dim + r];
        }
    }
    out
}

fn rotate_rows(values: &[f32], dim: usize, rotation: &[f32]) -> Vec<f32> {
    let rows = values.len() / dim;
    let mut out = vec![0.0_f32; values.len()];
    for row in 0..rows {
        for col in 0..dim {
            for d in 0..dim {
                out[row * dim + col] += values[row * dim + d] * rotation[d * dim + col];
            }
        }
    }
    out
}

fn flatten_centered_grains(grains: &[SharedBuildGrain], dim: usize) -> Result<Vec<f32>, String> {
    let rows = grains
        .iter()
        .map(|grain| grain.vectors.len())
        .sum::<usize>();
    let mut flat = Vec::with_capacity(rows * dim);
    for grain in grains {
        for vector in &grain.vectors {
            if vector.len() != dim || grain.mean.len() != dim {
                return Err("dimension mismatch in shared quantizer training".to_string());
            }
            flat.extend((0..dim).map(|d| vector[d] - grain.mean[d]));
        }
    }
    Ok(flat)
}

fn kmeans_flat(points: &[f32], rows: usize, dim: usize, k: usize) -> Vec<f32> {
    if rows == 0 || dim == 0 || k == 0 {
        return Vec::new();
    }
    let mut centroids = vec![0.0_f32; k * dim];
    for c in 0..k {
        let src = c * rows / k;
        centroids[c * dim..(c + 1) * dim].copy_from_slice(&points[src * dim..(src + 1) * dim]);
    }
    let mut assignments = vec![0_usize; rows];
    for _ in 0..20 {
        let mut changed = false;
        for row in 0..rows {
            let point = &points[row * dim..(row + 1) * dim];
            let mut best = 0;
            let mut best_dist = f32::INFINITY;
            for c in 0..k {
                let centroid = &centroids[c * dim..(c + 1) * dim];
                let dist = point
                    .iter()
                    .zip(centroid)
                    .map(|(a, b)| {
                        let diff = a - b;
                        diff * diff
                    })
                    .sum::<f32>();
                if dist < best_dist {
                    best_dist = dist;
                    best = c;
                }
            }
            changed |= assignments[row] != best;
            assignments[row] = best;
        }
        if !changed {
            break;
        }
        let mut sums = vec![0.0_f32; k * dim];
        let mut counts = vec![0_usize; k];
        for row in 0..rows {
            let c = assignments[row];
            counts[c] += 1;
            for d in 0..dim {
                sums[c * dim + d] += points[row * dim + d];
            }
        }
        for c in 0..k {
            if counts[c] == 0 {
                continue;
            }
            for d in 0..dim {
                centroids[c * dim + d] = sums[c * dim + d] / counts[c] as f32;
            }
        }
    }
    centroids
}

fn identity_projection(dim: usize, k: usize) -> Vec<f32> {
    let mut projection = vec![0.0; dim * k];
    for component in 0..k.min(dim) {
        projection[component * k + component] = 1.0;
    }
    projection
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pq_training_wraps_codebook_rows_when_vocabulary_exceeds_points() {
        let values = [
            1.0_f32, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];

        let codebooks = train_pq_codebooks(&values, 4, 2, 256);

        assert_eq!(codebooks.len(), 2 * 256 * 2);
        assert!(codebooks.iter().all(|value| value.is_finite()));
    }
}
