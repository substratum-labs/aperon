use crate::layout::BlockSoaLayout;

#[cfg(target_arch = "x86")]
type M256i = std::arch::x86::__m256i;
#[cfg(target_arch = "x86_64")]
type M256i = std::arch::x86_64::__m256i;

#[derive(Clone, Copy, Debug)]
pub(crate) struct ScanWeights<'a> {
    pub coord: &'a [i64],
    pub residual: i64,
    pub sketch: &'a [i64],
}

#[cfg(test)]
fn scan_block(
    layout: &BlockSoaLayout,
    block: usize,
    lanes: usize,
    query_coords: &[i16],
    query_residual: u16,
    query_sketches: &[i8],
    weights: ScanWeights<'_>,
) -> Vec<i64> {
    let mut scores = vec![0; lanes];
    scan_block_into(
        layout,
        block,
        lanes,
        query_coords,
        query_residual,
        query_sketches,
        weights,
        &mut scores,
    );
    scores
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_block_into(
    layout: &BlockSoaLayout,
    block: usize,
    lanes: usize,
    query_coords: &[i16],
    query_residual: u16,
    query_sketches: &[i8],
    weights: ScanWeights<'_>,
    scores: &mut [i64],
) {
    assert!(scores.len() >= lanes, "score scratch is shorter than lanes");
    if layout.residual_bits() != 8 {
        scalar_scan_block_into(
            layout,
            block,
            lanes,
            query_coords,
            query_residual,
            query_sketches,
            weights,
            scores,
        );
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: AArch64 guarantees NEON availability.
        unsafe {
            aarch64_neon_scan_block(
                layout,
                block,
                lanes,
                query_coords,
                query_residual,
                query_sketches,
                weights,
                scores,
            );
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                // SAFETY: Runtime feature detection above guarantees AVX2 support.
                unsafe {
                    x86_avx2_scan_block(
                        layout,
                        block,
                        lanes,
                        query_coords,
                        query_residual,
                        query_sketches,
                        weights,
                        scores,
                    );
                }
                return;
            }
        }

        scalar_scan_block_into(
            layout,
            block,
            lanes,
            query_coords,
            query_residual,
            query_sketches,
            weights,
            scores,
        );
    }
}

#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn scalar_scan_block(
    layout: &BlockSoaLayout,
    block: usize,
    lanes: usize,
    query_coords: &[i16],
    query_residual: u16,
    query_sketches: &[i8],
    weights: ScanWeights<'_>,
) -> Vec<i64> {
    let mut scores = vec![0; lanes];
    scalar_scan_block_into(
        layout,
        block,
        lanes,
        query_coords,
        query_residual,
        query_sketches,
        weights,
        &mut scores,
    );
    scores
}

#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn scalar_scan_block_into(
    layout: &BlockSoaLayout,
    block: usize,
    lanes: usize,
    query_coords: &[i16],
    query_residual: u16,
    query_sketches: &[i8],
    weights: ScanWeights<'_>,
    scores: &mut [i64],
) {
    residual_scores(
        &layout.residual_block(block)[..lanes],
        query_residual,
        weights.residual,
        scores,
    );
    for (dim, (&query, &weight)) in query_coords.iter().zip(weights.coord).enumerate() {
        let coords = layout.coord_block(block, dim);
        for lane in 0..lanes {
            let diff = i64::from(query) - i64::from(coords[lane]);
            scores[lane] += diff * diff * weight;
        }
    }
    for (dim, (&query, &weight)) in query_sketches.iter().zip(weights.sketch).enumerate() {
        for (lane, score) in scores.iter_mut().enumerate().take(lanes) {
            let diff = i64::from(query) - i64::from(layout.sketch(block, dim, lane));
            *score += diff * diff * weight;
        }
    }
}

fn residual_scores(residuals: &[u16], query_residual: u16, weight: i64, scores: &mut [i64]) {
    for (score, residual) in scores.iter_mut().zip(residuals) {
        // Residual lanes store quantized squared residual norms, not signed coordinates.
        // The norm-only distance term is ||rq||^2 + ||rx||^2; sketch lanes carry the
        // signed low-rank residual interaction when sketch_dim > 0.
        *score = (i64::from(query_residual) + i64::from(*residual)) * weight;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn x86_avx2_scan_block(
    layout: &BlockSoaLayout,
    block: usize,
    lanes: usize,
    query_coords: &[i16],
    query_residual: u16,
    query_sketches: &[i8],
    weights: ScanWeights<'_>,
    scores: &mut [i64],
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    residual_scores(
        &layout.residual_block(block)[..lanes],
        query_residual,
        weights.residual,
        scores,
    );
    let mut lane = 0;
    while lane + 8 <= lanes {
        for (dim, (&query, &weight)) in query_coords.iter().zip(weights.coord).enumerate() {
            let ptr = layout.coord_block(block, dim).as_ptr().add(lane) as *const __m128i;
            let values = _mm256_cvtepi16_epi32(_mm_loadu_si128(ptr));
            add_i32_square_lanes(
                &mut scores[lane..lane + 8],
                values,
                i32::from(query),
                weight,
            );
        }
        lane += 8;
    }
    add_scalar_coord_tail(
        layout,
        block,
        lane,
        lanes,
        query_coords,
        weights.coord,
        scores,
    );

    lane = 0;
    while lane + 8 <= lanes {
        for (dim, (&query, &weight)) in query_sketches.iter().zip(weights.sketch).enumerate() {
            let ptr = layout.sketch_block(block, dim).as_ptr().add(lane) as *const i64;
            let packed = std::ptr::read_unaligned(ptr);
            let values = _mm256_cvtepi8_epi32(_mm_cvtsi64_si128(packed));
            add_i32_square_lanes(
                &mut scores[lane..lane + 8],
                values,
                i32::from(query),
                weight,
            );
        }
        lane += 8;
    }
    add_scalar_sketch_tail(
        layout,
        block,
        lane,
        lanes,
        query_sketches,
        weights.sketch,
        scores,
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn add_i32_square_lanes(scores: &mut [i64], values: M256i, query: i32, weight: i64) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let diff = _mm256_sub_epi32(_mm256_set1_epi32(query), values);
    let even_sq = _mm256_mul_epi32(diff, diff);
    // `mul_epi32` only consumes the even i32 lanes in each 64-bit pair. Shifting
    // each 128-bit half right by one i32 places the original odd lanes into
    // those even positions: [d1, d2, d3, 0 | d5, d6, d7, 0].
    let odd = _mm256_srli_si256(diff, 4);
    let odd_sq = _mm256_mul_epi32(odd, odd);
    let mut even = [0_i64; 4];
    let mut odd = [0_i64; 4];
    _mm256_storeu_si256(even.as_mut_ptr() as *mut __m256i, even_sq);
    _mm256_storeu_si256(odd.as_mut_ptr() as *mut __m256i, odd_sq);
    for idx in 0..4 {
        scores[idx * 2] += even[idx] * weight;
        scores[idx * 2 + 1] += odd[idx] * weight;
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
unsafe fn aarch64_neon_scan_block(
    layout: &BlockSoaLayout,
    block: usize,
    lanes: usize,
    query_coords: &[i16],
    query_residual: u16,
    query_sketches: &[i8],
    weights: ScanWeights<'_>,
    scores: &mut [i64],
) {
    use std::arch::aarch64::*;

    residual_scores(
        &layout.residual_block(block)[..lanes],
        query_residual,
        weights.residual,
        scores,
    );
    let mut lane = 0;
    while lane + 8 <= lanes {
        for (dim, (&query, &weight)) in query_coords.iter().zip(weights.coord).enumerate() {
            let raw = vld1q_s16(layout.coord_block(block, dim).as_ptr().add(lane));
            let lo = vsubq_s32(vdupq_n_s32(i32::from(query)), vmovl_s16(vget_low_s16(raw)));
            add_neon_i32x4_square_lanes(&mut scores[lane..lane + 4], lo, weight);
            let hi = vsubq_s32(vdupq_n_s32(i32::from(query)), vmovl_s16(vget_high_s16(raw)));
            add_neon_i32x4_square_lanes(&mut scores[lane + 4..lane + 8], hi, weight);
        }
        lane += 8;
    }
    add_scalar_coord_tail(
        layout,
        block,
        lane,
        lanes,
        query_coords,
        weights.coord,
        scores,
    );

    lane = 0;
    while lane + 8 <= lanes {
        for (dim, (&query, &weight)) in query_sketches.iter().zip(weights.sketch).enumerate() {
            let raw = vld1_s8(layout.sketch_block(block, dim).as_ptr().add(lane));
            let expanded = vmovl_s8(raw);
            let lo = vsubq_s32(
                vdupq_n_s32(i32::from(query)),
                vmovl_s16(vget_low_s16(expanded)),
            );
            add_neon_i32x4_square_lanes(&mut scores[lane..lane + 4], lo, weight);
            let hi = vsubq_s32(
                vdupq_n_s32(i32::from(query)),
                vmovl_s16(vget_high_s16(expanded)),
            );
            add_neon_i32x4_square_lanes(&mut scores[lane + 4..lane + 8], hi, weight);
        }
        lane += 8;
    }
    add_scalar_sketch_tail(
        layout,
        block,
        lane,
        lanes,
        query_sketches,
        weights.sketch,
        scores,
    );
}

#[cfg(target_arch = "aarch64")]
unsafe fn add_neon_i32x4_square_lanes(
    scores: &mut [i64],
    diff: std::arch::aarch64::int32x4_t,
    weight: i64,
) {
    use std::arch::aarch64::*;

    let lo = vmull_s32(vget_low_s32(diff), vget_low_s32(diff));
    let hi = vmull_s32(vget_high_s32(diff), vget_high_s32(diff));
    let mut lo_lanes = [0_i64; 2];
    let mut hi_lanes = [0_i64; 2];
    vst1q_s64(lo_lanes.as_mut_ptr(), lo);
    vst1q_s64(hi_lanes.as_mut_ptr(), hi);
    scores[0] += lo_lanes[0] * weight;
    scores[1] += lo_lanes[1] * weight;
    scores[2] += hi_lanes[0] * weight;
    scores[3] += hi_lanes[1] * weight;
}

fn add_scalar_coord_tail(
    layout: &BlockSoaLayout,
    block: usize,
    start: usize,
    lanes: usize,
    query_coords: &[i16],
    weights: &[i64],
    scores: &mut [i64],
) {
    for (dim, (&query, &weight)) in query_coords.iter().zip(weights).enumerate() {
        let coords = layout.coord_block(block, dim);
        for lane in start..lanes {
            let diff = i64::from(query) - i64::from(coords[lane]);
            scores[lane] += diff * diff * weight;
        }
    }
}

fn add_scalar_sketch_tail(
    layout: &BlockSoaLayout,
    block: usize,
    start: usize,
    lanes: usize,
    query_sketches: &[i8],
    weights: &[i64],
    scores: &mut [i64],
) {
    for (dim, (&query, &weight)) in query_sketches.iter().zip(weights).enumerate() {
        for (lane, score) in scores.iter_mut().enumerate().take(lanes).skip(start) {
            let diff = i64::from(query) - i64::from(layout.sketch(block, dim, lane));
            *score += diff * diff * weight;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::VectorId;
    use std::{hint::black_box, time::Instant};

    #[test]
    fn scalar_scan_matches_reference_formula() {
        let mut layout = BlockSoaLayout::with_shape(3, 2, 8);
        for lane in 0..7 {
            layout
                .push_quantized(
                    VectorId::new(lane as u64),
                    &[lane as i16 - 3, lane as i16 * 2, 7 - lane as i16],
                    lane as u16 + 5,
                    &[lane as i8 - 2, 3 - lane as i8],
                )
                .unwrap();
        }
        let weights = ScanWeights {
            coord: &[2, 3, 5],
            residual: 7,
            sketch: &[11, 13],
        };

        let scores = scalar_scan_block(&layout, 0, 7, &[4, -2, 1], 9, &[3, -4], weights);

        for (lane, score) in scores.iter().enumerate() {
            let mut expected = (9_i64 + i64::from(layout.residual(0, lane))) * 7;
            for dim in 0..3 {
                let diff = i64::from([4, -2, 1][dim]) - i64::from(layout.coord(0, dim, lane));
                expected += diff * diff * weights.coord[dim];
            }
            for dim in 0..2 {
                let diff = i64::from([3, -4][dim]) - i64::from(layout.sketch(0, dim, lane));
                expected += diff * diff * weights.sketch[dim];
            }
            assert_eq!(*score, expected);
        }
    }

    #[test]
    fn dispatch_matches_scalar_for_full_and_tail_blocks() {
        let mut layout = BlockSoaLayout::with_shape(5, 3, 16);
        for lane in 0..23 {
            layout
                .push_quantized(
                    VectorId::new(lane as u64),
                    &[
                        lane as i16 - 11,
                        17 - lane as i16,
                        lane as i16 * 3 - 9,
                        4,
                        -6,
                    ],
                    lane as u16 * 2 + 1,
                    &[lane as i8 - 5, 9 - lane as i8, lane as i8 / 2],
                )
                .unwrap();
        }
        let weights = ScanWeights {
            coord: &[1, 2, 3, 4, 5],
            residual: 6,
            sketch: &[7, 8, 9],
        };

        for block in 0..layout.block_count() {
            let lanes = layout.block_len(block);
            let scalar = scalar_scan_block(
                &layout,
                block,
                lanes,
                &[3, -7, 11, 13, -17],
                19,
                &[5, -3, 2],
                weights,
            );
            let dispatched = scan_block(
                &layout,
                block,
                lanes,
                &[3, -7, 11, 13, -17],
                19,
                &[5, -3, 2],
                weights,
            );
            assert_eq!(dispatched, scalar);
        }
    }

    #[test]
    #[ignore = "release-mode smoke benchmark for the T-126 scan budget"]
    fn block_scan_stays_under_five_ns_per_vector() {
        let block_size = 64;
        let vectors = 4096;
        let mut layout = BlockSoaLayout::with_shape(8, 4, block_size);
        for lane in 0..vectors {
            layout
                .push_quantized(
                    VectorId::new(lane as u64),
                    &[
                        lane as i16,
                        17 - lane as i16,
                        lane as i16 * 3,
                        -lane as i16,
                        11,
                        -13,
                        19,
                        -23,
                    ],
                    (lane % 1024) as u16,
                    &[(lane % 127) as i8, -((lane % 63) as i8), 7, -9],
                )
                .unwrap();
        }
        let weights = ScanWeights {
            coord: &[1, 1, 1, 1, 1, 1, 1, 1],
            residual: 1,
            sketch: &[1, 1, 1, 1],
        };
        let query_coords = [3, -7, 11, -13, 17, -19, 23, -29];
        let query_sketches = [5, -3, 2, -1];
        let blocks = layout.block_count();

        let mut checksum = 0_i64;
        for block in 0..blocks {
            for score in scan_block(
                &layout,
                block,
                layout.block_len(block),
                &query_coords,
                31,
                &query_sketches,
                weights,
            ) {
                checksum ^= score;
            }
        }
        black_box(checksum);

        let iterations = 5000;
        let started = Instant::now();
        for _ in 0..iterations {
            for block in 0..blocks {
                for score in scan_block(
                    black_box(&layout),
                    block,
                    layout.block_len(block),
                    black_box(&query_coords),
                    31,
                    black_box(&query_sketches),
                    weights,
                ) {
                    checksum ^= score;
                }
            }
        }
        black_box(checksum);

        let scanned = iterations * vectors;
        let ns_per_vector = started.elapsed().as_nanos() as f64 / scanned as f64;
        println!("block scan: {ns_per_vector:.3} ns/vector");
        assert!(
            ns_per_vector <= 5.0,
            "block scan took {ns_per_vector:.3} ns/vector"
        );
    }
}
