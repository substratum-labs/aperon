use crate::layout::BlockSoaLayout;

pub(crate) trait SIMDScanKernel {
    fn scan_block(
        &self,
        layout: &BlockSoaLayout,
        block: usize,
        lanes: usize,
        query_coords: &[i16],
        query_residual: u16,
        query_sketch: &[i8],
        weights: ScanWeights<'_>,
        scores: &mut [i64],
    );

    fn scan_block_multi(
        &self,
        layout: &BlockSoaLayout,
        block: usize,
        lanes: usize,
        queries_coords: &[&[i16]],
        queries_residual: &[u16],
        queries_sketches: &[&[i8]],
        weights: ScanWeights<'_>,
        queries_scores: &mut [&mut [i64]],
    );
}

pub(crate) struct ScalarScanKernel;

impl SIMDScanKernel for ScalarScanKernel {
    fn scan_block(
        &self,
        layout: &BlockSoaLayout,
        block: usize,
        lanes: usize,
        query_coords: &[i16],
        query_residual: u16,
        query_sketch: &[i8],
        weights: ScanWeights<'_>,
        scores: &mut [i64],
    ) {
        scalar_scan_block_into(
            layout,
            block,
            lanes,
            query_coords,
            query_residual,
            query_sketch,
            weights,
            scores,
        );
    }

    fn scan_block_multi(
        &self,
        layout: &BlockSoaLayout,
        block: usize,
        lanes: usize,
        queries_coords: &[&[i16]],
        queries_residual: &[u16],
        queries_sketches: &[&[i8]],
        weights: ScanWeights<'_>,
        queries_scores: &mut [&mut [i64]],
    ) {
        let m = queries_coords.len();
        for q in 0..m {
            scalar_scan_block_into(
                layout,
                block,
                lanes,
                queries_coords[q],
                queries_residual[q],
                queries_sketches[q],
                weights,
                queries_scores[q],
            );
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub(crate) struct NeonScanKernel;

#[cfg(target_arch = "aarch64")]
impl SIMDScanKernel for NeonScanKernel {
    fn scan_block(
        &self,
        layout: &BlockSoaLayout,
        block: usize,
        lanes: usize,
        query_coords: &[i16],
        query_residual: u16,
        query_sketch: &[i8],
        weights: ScanWeights<'_>,
        scores: &mut [i64],
    ) {
        scan_block_into(
            layout,
            block,
            lanes,
            query_coords,
            query_residual,
            query_sketch,
            weights,
            scores,
        );
    }

    fn scan_block_multi(
        &self,
        layout: &BlockSoaLayout,
        block: usize,
        lanes: usize,
        queries_coords: &[&[i16]],
        queries_residual: &[u16],
        queries_sketches: &[&[i8]],
        weights: ScanWeights<'_>,
        queries_scores: &mut [&mut [i64]],
    ) {
        unsafe {
            aarch64_neon_scan_block_multi(
                layout,
                block,
                lanes,
                queries_coords,
                queries_residual,
                queries_sketches,
                weights,
                queries_scores,
            );
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) struct Avx2ScanKernel;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl SIMDScanKernel for Avx2ScanKernel {
    fn scan_block(
        &self,
        layout: &BlockSoaLayout,
        block: usize,
        lanes: usize,
        query_coords: &[i16],
        query_residual: u16,
        query_sketch: &[i8],
        weights: ScanWeights<'_>,
        scores: &mut [i64],
    ) {
        scan_block_into(
            layout,
            block,
            lanes,
            query_coords,
            query_residual,
            query_sketch,
            weights,
            scores,
        );
    }

    fn scan_block_multi(
        &self,
        layout: &BlockSoaLayout,
        block: usize,
        lanes: usize,
        queries_coords: &[&[i16]],
        queries_residual: &[u16],
        queries_sketches: &[&[i8]],
        weights: ScanWeights<'_>,
        queries_scores: &mut [&mut [i64]],
    ) {
        unsafe {
            x86_avx2_scan_block_multi(
                layout,
                block,
                lanes,
                queries_coords,
                queries_residual,
                queries_sketches,
                weights,
                queries_scores,
            );
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) struct Avx512ScanKernel;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl SIMDScanKernel for Avx512ScanKernel {
    fn scan_block(
        &self,
        layout: &BlockSoaLayout,
        block: usize,
        lanes: usize,
        query_coords: &[i16],
        query_residual: u16,
        query_sketch: &[i8],
        weights: ScanWeights<'_>,
        scores: &mut [i64],
    ) {
        scan_block_into(
            layout,
            block,
            lanes,
            query_coords,
            query_residual,
            query_sketch,
            weights,
            scores,
        );
    }

    fn scan_block_multi(
        &self,
        layout: &BlockSoaLayout,
        block: usize,
        lanes: usize,
        queries_coords: &[&[i16]],
        queries_residual: &[u16],
        queries_sketches: &[&[i8]],
        weights: ScanWeights<'_>,
        queries_scores: &mut [&mut [i64]],
    ) {
        unsafe {
            x86_avx512_scan_block_multi(
                layout,
                block,
                lanes,
                queries_coords,
                queries_residual,
                queries_sketches,
                weights,
                queries_scores,
            );
        }
    }
}

pub(crate) fn get_optimal_scan_kernel() -> Box<dyn SIMDScanKernel + Send + Sync> {
    #[cfg(target_arch = "aarch64")]
    {
        Box::new(NeonScanKernel)
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            Box::new(Avx512ScanKernel)
        } else if std::arch::is_x86_feature_detected!("avx2") {
            Box::new(Avx2ScanKernel)
        } else {
            Box::new(ScalarScanKernel)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
    {
        Box::new(ScalarScanKernel)
    }
}

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
            if std::arch::is_x86_feature_detected!("avx512f") {
                unsafe {
                    x86_avx512_scan_block(
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

#[allow(dead_code)]
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
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn x86_avx512_scan_block(
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
        let scores_ptr = scores.as_ptr().add(lane);
        let mut v_scores = _mm512_loadu_si512(scores_ptr as *const _);

        for (dim, (&query, &weight)) in query_coords.iter().zip(weights.coord).enumerate() {
            let coord_ptr = layout.coord_block(block, dim).as_ptr().add(lane) as *const __m128i;
            let raw = _mm_loadu_si128(coord_ptr);
            let widened = _mm512_cvtepi16_epi64(raw);
            let v_query = _mm512_set1_epi64(i64::from(query));
            let diff = _mm512_sub_epi64(v_query, widened);
            let diff_sq = _mm512_mul_epu32(diff, diff);
            let v_weight = _mm512_set1_epi64(weight);
            let prod = _mm512_mul_epu32(diff_sq, v_weight);
            v_scores = _mm512_add_epi64(v_scores, prod);
        }

        for (dim, (&query, &weight)) in query_sketches.iter().zip(weights.sketch).enumerate() {
            let sketch_ptr = layout.sketch_block(block, dim).as_ptr().add(lane) as *const i64;
            let packed = std::ptr::read_unaligned(sketch_ptr);
            let raw = _mm_cvtsi64_si128(packed);
            let widened = _mm512_cvtepi8_epi64(raw);
            let v_query = _mm512_set1_epi64(i64::from(query));
            let diff = _mm512_sub_epi64(v_query, widened);
            let diff_sq = _mm512_mul_epu32(diff, diff);
            let v_weight = _mm512_set1_epi64(weight);
            let prod = _mm512_mul_epu32(diff_sq, v_weight);
            v_scores = _mm512_add_epi64(v_scores, prod);
        }

        _mm512_storeu_si512(scores.as_mut_ptr().add(lane) as *mut _, v_scores);
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
        let mut v_scores_lo = _mm256_loadu_si256(scores.as_ptr().add(lane) as *const __m256i);
        let mut v_scores_hi = _mm256_loadu_si256(scores.as_ptr().add(lane + 4) as *const __m256i);

        for (dim, (&query, &weight)) in query_coords.iter().zip(weights.coord).enumerate() {
            let ptr = layout.coord_block(block, dim).as_ptr().add(lane) as *const __m128i;
            let raw = _mm_loadu_si128(ptr);
            let values = _mm256_cvtepi16_epi32(raw);

            let diff = _mm256_sub_epi32(_mm256_set1_epi32(i32::from(query)), values);
            let even_sq = _mm256_mul_epi32(diff, diff);
            let odd = _mm256_srli_si256(diff, 4);
            let odd_sq = _mm256_mul_epi32(odd, odd);

            let v_weight = _mm256_set1_epi64x(weight);
            let even_prod = _mm256_mul_epi32(even_sq, v_weight);
            let odd_prod = _mm256_mul_epi32(odd_sq, v_weight);

            let unpack_lo = _mm256_unpacklo_epi64(even_prod, odd_prod);
            let unpack_hi = _mm256_unpackhi_epi64(even_prod, odd_prod);

            let prod_lo = _mm256_permute2x128_si256(unpack_lo, unpack_hi, 0x20);
            let prod_hi = _mm256_permute2x128_si256(unpack_lo, unpack_hi, 0x31);

            v_scores_lo = _mm256_add_epi64(v_scores_lo, prod_lo);
            v_scores_hi = _mm256_add_epi64(v_scores_hi, prod_hi);
        }

        for (dim, (&query, &weight)) in query_sketches.iter().zip(weights.sketch).enumerate() {
            let ptr = layout.sketch_block(block, dim).as_ptr().add(lane) as *const i64;
            let packed = std::ptr::read_unaligned(ptr);
            let raw = _mm_cvtsi64_si128(packed);
            let values = _mm256_cvtepi8_epi32(raw);

            let diff = _mm256_sub_epi32(_mm256_set1_epi32(i32::from(query)), values);
            let even_sq = _mm256_mul_epi32(diff, diff);
            let odd = _mm256_srli_si256(diff, 4);
            let odd_sq = _mm256_mul_epi32(odd, odd);

            let v_weight = _mm256_set1_epi64x(weight);
            let even_prod = _mm256_mul_epi32(even_sq, v_weight);
            let odd_prod = _mm256_mul_epi32(odd_sq, v_weight);

            let unpack_lo = _mm256_unpacklo_epi64(even_prod, odd_prod);
            let unpack_hi = _mm256_unpackhi_epi64(even_prod, odd_prod);

            let prod_lo = _mm256_permute2x128_si256(unpack_lo, unpack_hi, 0x20);
            let prod_hi = _mm256_permute2x128_si256(unpack_lo, unpack_hi, 0x31);

            v_scores_lo = _mm256_add_epi64(v_scores_lo, prod_lo);
            v_scores_hi = _mm256_add_epi64(v_scores_hi, prod_hi);
        }

        _mm256_storeu_si256(scores.as_mut_ptr().add(lane) as *mut __m256i, v_scores_lo);
        _mm256_storeu_si256(
            scores.as_mut_ptr().add(lane + 4) as *mut __m256i,
            v_scores_hi,
        );
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
        let mut v_scores_01 = vld1q_s64(scores.as_ptr().add(lane));
        let mut v_scores_23 = vld1q_s64(scores.as_ptr().add(lane + 2));
        let mut v_scores_45 = vld1q_s64(scores.as_ptr().add(lane + 4));
        let mut v_scores_67 = vld1q_s64(scores.as_ptr().add(lane + 6));

        for (dim, (&query, &weight)) in query_coords.iter().zip(weights.coord).enumerate() {
            let raw = vld1q_s16(layout.coord_block(block, dim).as_ptr().add(lane));
            let v_query = vdupq_n_s16(query);
            let diff = vsubq_s16(v_query, raw);

            let diff_lo = vmovl_s16(vget_low_s16(diff));
            let diff_hi = vmovl_s16(vget_high_s16(diff));

            let sq_lo = vmulq_s32(diff_lo, diff_lo);
            let sq_hi = vmulq_s32(diff_hi, diff_hi);

            let v_weight = vdupq_n_s32(weight as i32);
            let prod_lo = vmulq_s32(sq_lo, v_weight);
            let prod_hi = vmulq_s32(sq_hi, v_weight);

            let prod_01 = vmovl_s32(vget_low_s32(prod_lo));
            let prod_23 = vmovl_s32(vget_high_s32(prod_lo));
            let prod_45 = vmovl_s32(vget_low_s32(prod_hi));
            let prod_67 = vmovl_s32(vget_high_s32(prod_hi));

            v_scores_01 = vaddq_s64(v_scores_01, prod_01);
            v_scores_23 = vaddq_s64(v_scores_23, prod_23);
            v_scores_45 = vaddq_s64(v_scores_45, prod_45);
            v_scores_67 = vaddq_s64(v_scores_67, prod_67);
        }

        for (dim, (&query, &weight)) in query_sketches.iter().zip(weights.sketch).enumerate() {
            let raw = vld1_s8(layout.sketch_block(block, dim).as_ptr().add(lane));
            let v_query = vdup_n_s8(query);
            let diff = vsub_s8(v_query, raw);
            let diff_16 = vmovl_s8(diff);

            let diff_lo = vmovl_s16(vget_low_s16(diff_16));
            let diff_hi = vmovl_s16(vget_high_s16(diff_16));

            let sq_lo = vmulq_s32(diff_lo, diff_lo);
            let sq_hi = vmulq_s32(diff_hi, diff_hi);

            let v_weight = vdupq_n_s32(weight as i32);
            let prod_lo = vmulq_s32(sq_lo, v_weight);
            let prod_hi = vmulq_s32(sq_hi, v_weight);

            let prod_01 = vmovl_s32(vget_low_s32(prod_lo));
            let prod_23 = vmovl_s32(vget_high_s32(prod_lo));
            let prod_45 = vmovl_s32(vget_low_s32(prod_hi));
            let prod_67 = vmovl_s32(vget_high_s32(prod_hi));

            v_scores_01 = vaddq_s64(v_scores_01, prod_01);
            v_scores_23 = vaddq_s64(v_scores_23, prod_23);
            v_scores_45 = vaddq_s64(v_scores_45, prod_45);
            v_scores_67 = vaddq_s64(v_scores_67, prod_67);
        }

        vst1q_s64(scores.as_mut_ptr().add(lane), v_scores_01);
        vst1q_s64(scores.as_mut_ptr().add(lane + 2), v_scores_23);
        vst1q_s64(scores.as_mut_ptr().add(lane + 4), v_scores_45);
        vst1q_s64(scores.as_mut_ptr().add(lane + 6), v_scores_67);
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

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub(crate) fn scan_block_pruned_into(
    layout: &BlockSoaLayout,
    block: usize,
    lanes: usize,
    query_coords: &[i16],
    query_residual: u16,
    query_sketches: &[i8],
    weights: ScanWeights<'_>,
    scores: &mut [i64],
    threshold: i64,
) -> bool {
    residual_scores(
        &layout.residual_block(block)[..lanes],
        query_residual,
        weights.residual,
        scores,
    );

    let mut lane = 0;
    while lane + 8 <= lanes {
        for (dim, (&query, &weight)) in query_coords.iter().zip(weights.coord).enumerate() {
            if dim % 4 == 0 {
                let mut all_exceed = true;
                for l in lane..lane + 8 {
                    if scores[l] < threshold {
                        all_exceed = false;
                        break;
                    }
                }
                if all_exceed {
                    break;
                }
            }

            #[cfg(target_arch = "aarch64")]
            unsafe {
                use std::arch::aarch64::*;
                let raw = vld1q_s16(layout.coord_block(block, dim).as_ptr().add(lane));
                let lo = vsubq_s32(vdupq_n_s32(i32::from(query)), vmovl_s16(vget_low_s16(raw)));
                add_neon_i32x4_square_lanes(&mut scores[lane..lane + 4], lo, weight);
                let hi = vsubq_s32(vdupq_n_s32(i32::from(query)), vmovl_s16(vget_high_s16(raw)));
                add_neon_i32x4_square_lanes(&mut scores[lane + 4..lane + 8], hi, weight);
            }

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            unsafe {
                #[cfg(target_arch = "x86")]
                use std::arch::x86::*;
                #[cfg(target_arch = "x86_64")]
                use std::arch::x86_64::*;
                if std::arch::is_x86_feature_detected!("avx2") {
                    let ptr = layout.coord_block(block, dim).as_ptr().add(lane) as *const __m128i;
                    let values = _mm256_cvtepi16_epi32(_mm_loadu_si128(ptr));
                    add_i32_square_lanes(
                        &mut scores[lane..lane + 8],
                        values,
                        i32::from(query),
                        weight,
                    );
                } else {
                    let coords = layout.coord_block(block, dim);
                    for l in lane..lane + 8 {
                        let diff = i64::from(query) - i64::from(coords[l]);
                        scores[l] += diff * diff * weight;
                    }
                }
            }

            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
            {
                let coords = layout.coord_block(block, dim);
                for l in lane..lane + 8 {
                    let diff = i64::from(query) - i64::from(coords[l]);
                    scores[l] += diff * diff * weight;
                }
            }
        }
        lane += 8;
    }

    for (dim, (&query, &weight)) in query_coords.iter().zip(weights.coord).enumerate() {
        let coords = layout.coord_block(block, dim);
        for l in lane..lanes {
            if scores[l] < threshold {
                let diff = i64::from(query) - i64::from(coords[l]);
                scores[l] += diff * diff * weight;
            }
        }
    }

    let mut lane_sk = 0;
    while lane_sk + 8 <= lanes {
        for (dim, (&query, &weight)) in query_sketches.iter().zip(weights.sketch).enumerate() {
            if dim % 2 == 0 {
                let mut all_exceed = true;
                for l in lane_sk..lane_sk + 8 {
                    if scores[l] < threshold {
                        all_exceed = false;
                        break;
                    }
                }
                if all_exceed {
                    break;
                }
            }

            #[cfg(target_arch = "aarch64")]
            unsafe {
                use std::arch::aarch64::*;
                let raw = vld1_s8(layout.sketch_block(block, dim).as_ptr().add(lane_sk));
                let expanded = vmovl_s8(raw);
                let lo = vsubq_s32(
                    vdupq_n_s32(i32::from(query)),
                    vmovl_s16(vget_low_s16(expanded)),
                );
                add_neon_i32x4_square_lanes(&mut scores[lane_sk..lane_sk + 4], lo, weight);
                let hi = vsubq_s32(
                    vdupq_n_s32(i32::from(query)),
                    vmovl_s16(vget_high_s16(expanded)),
                );
                add_neon_i32x4_square_lanes(&mut scores[lane_sk + 4..lane_sk + 8], hi, weight);
            }

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            unsafe {
                #[cfg(target_arch = "x86")]
                use std::arch::x86::*;
                #[cfg(target_arch = "x86_64")]
                use std::arch::x86_64::*;
                if std::arch::is_x86_feature_detected!("avx2") {
                    let ptr = layout.sketch_block(block, dim).as_ptr().add(lane_sk) as *const i64;
                    let packed = std::ptr::read_unaligned(ptr);
                    let values = _mm256_cvtepi8_epi32(_mm_cvtsi64_si128(packed));
                    add_i32_square_lanes(
                        &mut scores[lane_sk..lane_sk + 8],
                        values,
                        i32::from(query),
                        weight,
                    );
                } else {
                    for l in lane_sk..lane_sk + 8 {
                        let diff = i64::from(query) - i64::from(layout.sketch(block, dim, l));
                        scores[l] += diff * diff * weight;
                    }
                }
            }

            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
            {
                for l in lane_sk..lane_sk + 8 {
                    let diff = i64::from(query) - i64::from(layout.sketch(block, dim, l));
                    scores[l] += diff * diff * weight;
                }
            }
        }
        lane_sk += 8;
    }

    for (dim, (&query, &weight)) in query_sketches.iter().zip(weights.sketch).enumerate() {
        for l in lane_sk..lanes {
            if scores[l] < threshold {
                let diff = i64::from(query) - i64::from(layout.sketch(block, dim, l));
                scores[l] += diff * diff * weight;
            }
        }
    }

    let mut final_exceed = true;
    for &score in scores.iter().take(lanes) {
        if score < threshold {
            final_exceed = false;
            break;
        }
    }
    final_exceed
}

#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_block_multi_into(
    layout: &BlockSoaLayout,
    block: usize,
    lanes: usize,
    queries_coords: &[&[i16]],
    queries_residual: &[u16],
    queries_sketches: &[&[i8]],
    weights: ScanWeights<'_>,
    queries_scores: &mut [&mut [i64]],
) {
    let m = queries_coords.len();
    assert!(m <= 8, "maximum supported multi-query batch size is 8");
    for score in queries_scores.iter().take(m) {
        assert!(score.len() >= lanes, "score scratch is shorter than lanes");
    }

    if layout.residual_bits() != 8 {
        for q in 0..m {
            scalar_scan_block_into(
                layout,
                block,
                lanes,
                queries_coords[q],
                queries_residual[q],
                queries_sketches[q],
                weights,
                queries_scores[q],
            );
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            aarch64_neon_scan_block_multi(
                layout,
                block,
                lanes,
                queries_coords,
                queries_residual,
                queries_sketches,
                weights,
                queries_scores,
            );
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::arch::is_x86_feature_detected!("avx512f") {
                unsafe {
                    x86_avx512_scan_block_multi(
                        layout,
                        block,
                        lanes,
                        queries_coords,
                        queries_residual,
                        queries_sketches,
                        weights,
                        queries_scores,
                    );
                }
                return;
            }
            if std::arch::is_x86_feature_detected!("avx2") {
                unsafe {
                    x86_avx2_scan_block_multi(
                        layout,
                        block,
                        lanes,
                        queries_coords,
                        queries_residual,
                        queries_sketches,
                        weights,
                        queries_scores,
                    );
                }
                return;
            }
        }

        // Fallback to scalar
        for q in 0..m {
            scalar_scan_block_into(
                layout,
                block,
                lanes,
                queries_coords[q],
                queries_residual[q],
                queries_sketches[q],
                weights,
                queries_scores[q],
            );
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
unsafe fn aarch64_neon_scan_block_multi(
    layout: &BlockSoaLayout,
    block: usize,
    lanes: usize,
    queries_coords: &[&[i16]],
    queries_residual: &[u16],
    queries_sketches: &[&[i8]],
    weights: ScanWeights<'_>,
    queries_scores: &mut [&mut [i64]],
) {
    use std::arch::aarch64::*;

    let m = queries_coords.len();
    let vector_lanes = (lanes / 8) * 8;
    if m == 4 {
        if vector_lanes < lanes {
            let res_block = layout.residual_block(block);
            for q in 0..4 {
                residual_scores(
                    &res_block[vector_lanes..lanes],
                    queries_residual[q],
                    weights.residual,
                    &mut queries_scores[q][vector_lanes..lanes],
                );
            }
        }
    } else {
        for q in 0..m {
            residual_scores(
                &layout.residual_block(block)[..lanes],
                queries_residual[q],
                weights.residual,
                queries_scores[q],
            );
        }
    }
    let mut lane = 0;
    while lane + 8 <= lanes {
        if m == 4 {
            let q0_coords = queries_coords[0];
            let q1_coords = queries_coords[1];
            let q2_coords = queries_coords[2];
            let q3_coords = queries_coords[3];

            let q0_sketches = queries_sketches[0];
            let q1_sketches = queries_sketches[1];
            let q2_sketches = queries_sketches[2];
            let q3_sketches = queries_sketches[3];

            let s0_ptr = queries_scores[0].as_mut_ptr();
            let s1_ptr = queries_scores[1].as_mut_ptr();
            let s2_ptr = queries_scores[2].as_mut_ptr();
            let s3_ptr = queries_scores[3].as_mut_ptr();

            let coords_ptr = layout.block_coords_ptr(block);
            let sketches_ptr = layout.block_sketches_ptr(block);
            let block_size = layout.block_size();

            let residual_ptr = layout.residual_block(block).as_ptr();
            let v_res_weight = vdupq_n_s32(weights.residual as i32);

            let v_q0_residual = vdupq_n_u32(queries_residual[0] as u32);
            let v_q1_residual = vdupq_n_u32(queries_residual[1] as u32);
            let v_q2_residual = vdupq_n_u32(queries_residual[2] as u32);
            let v_q3_residual = vdupq_n_u32(queries_residual[3] as u32);

            // Hoist weight and query coordinates duplication outside the inner loop
            let mut w_coord_dupped = [vdupq_n_s32(0); 8];
            for (d, item) in w_coord_dupped.iter_mut().enumerate() {
                if d < weights.coord.len() {
                    *item = vdupq_n_s32(weights.coord[d] as i32);
                }
            }

            let mut w_sketch_dupped = [vdupq_n_s32(0); 4];
            for (d, item) in w_sketch_dupped.iter_mut().enumerate() {
                if d < weights.sketch.len() {
                    *item = vdupq_n_s32(weights.sketch[d] as i32);
                }
            }

            let mut q_coord_dupped = [[vdupq_n_s16(0); 8]; 4];
            for d in 0..8 {
                q_coord_dupped[0][d] = vdupq_n_s16(q0_coords[d]);
                q_coord_dupped[1][d] = vdupq_n_s16(q1_coords[d]);
                q_coord_dupped[2][d] = vdupq_n_s16(q2_coords[d]);
                q_coord_dupped[3][d] = vdupq_n_s16(q3_coords[d]);
            }

            let mut q_sketch_dupped = [[vdup_n_s8(0); 4]; 4];
            for d in 0..4 {
                q_sketch_dupped[0][d] = vdup_n_s8(q0_sketches[d]);
                q_sketch_dupped[1][d] = vdup_n_s8(q1_sketches[d]);
                q_sketch_dupped[2][d] = vdup_n_s8(q2_sketches[d]);
                q_sketch_dupped[3][d] = vdup_n_s8(q3_sketches[d]);
            }

            let all_weights_one = weights.residual == 1
                && weights.coord.iter().all(|&w| w == 1)
                && weights.sketch.iter().all(|&w| w == 1);

            let mut inner_lane = lane;
            if all_weights_one {
                while inner_lane + 8 <= lanes {
                    let raw_res = vld1q_u16(residual_ptr.add(inner_lane));
                    let res_lo = vmovl_u16(vget_low_u16(raw_res));
                    let res_hi = vmovl_u16(vget_high_u16(raw_res));

                    // Q0 scores init
                    let sum_q0_lo = vaddq_u32(v_q0_residual, res_lo);
                    let sum_q0_hi = vaddq_u32(v_q0_residual, res_hi);
                    let mut v_scores_q0_0 =
                        vmovl_s32(vget_low_s32(vreinterpretq_s32_u32(sum_q0_lo)));
                    let mut v_scores_q0_1 =
                        vmovl_s32(vget_high_s32(vreinterpretq_s32_u32(sum_q0_lo)));
                    let mut v_scores_q0_2 =
                        vmovl_s32(vget_low_s32(vreinterpretq_s32_u32(sum_q0_hi)));
                    let mut v_scores_q0_3 =
                        vmovl_s32(vget_high_s32(vreinterpretq_s32_u32(sum_q0_hi)));

                    // Q1 scores init
                    let sum_q1_lo = vaddq_u32(v_q1_residual, res_lo);
                    let sum_q1_hi = vaddq_u32(v_q1_residual, res_hi);
                    let mut v_scores_q1_0 =
                        vmovl_s32(vget_low_s32(vreinterpretq_s32_u32(sum_q1_lo)));
                    let mut v_scores_q1_1 =
                        vmovl_s32(vget_high_s32(vreinterpretq_s32_u32(sum_q1_lo)));
                    let mut v_scores_q1_2 =
                        vmovl_s32(vget_low_s32(vreinterpretq_s32_u32(sum_q1_hi)));
                    let mut v_scores_q1_3 =
                        vmovl_s32(vget_high_s32(vreinterpretq_s32_u32(sum_q1_hi)));

                    // Q2 scores init
                    let sum_q2_lo = vaddq_u32(v_q2_residual, res_lo);
                    let sum_q2_hi = vaddq_u32(v_q2_residual, res_hi);
                    let mut v_scores_q2_0 =
                        vmovl_s32(vget_low_s32(vreinterpretq_s32_u32(sum_q2_lo)));
                    let mut v_scores_q2_1 =
                        vmovl_s32(vget_high_s32(vreinterpretq_s32_u32(sum_q2_lo)));
                    let mut v_scores_q2_2 =
                        vmovl_s32(vget_low_s32(vreinterpretq_s32_u32(sum_q2_hi)));
                    let mut v_scores_q2_3 =
                        vmovl_s32(vget_high_s32(vreinterpretq_s32_u32(sum_q2_hi)));

                    // Q3 scores init
                    let sum_q3_lo = vaddq_u32(v_q3_residual, res_lo);
                    let sum_q3_hi = vaddq_u32(v_q3_residual, res_hi);
                    let mut v_scores_q3_0 =
                        vmovl_s32(vget_low_s32(vreinterpretq_s32_u32(sum_q3_lo)));
                    let mut v_scores_q3_1 =
                        vmovl_s32(vget_high_s32(vreinterpretq_s32_u32(sum_q3_lo)));
                    let mut v_scores_q3_2 =
                        vmovl_s32(vget_low_s32(vreinterpretq_s32_u32(sum_q3_hi)));
                    let mut v_scores_q3_3 =
                        vmovl_s32(vget_high_s32(vreinterpretq_s32_u32(sum_q3_hi)));

                    if weights.coord.len() == 8 && weights.sketch.len() == 4 {
                        macro_rules! step_coord_unweighted {
                            ($d:expr) => {
                                let raw = vld1q_s16(coords_ptr.add($d * block_size + inner_lane));

                                // Q0
                                let v_query_q0 = q_coord_dupped[0][$d];
                                let diff0 = vsubq_s16(v_query_q0, raw);
                                let sq0_lo = vmull_s16(vget_low_s16(diff0), vget_low_s16(diff0));
                                let sq0_hi = vmull_s16(vget_high_s16(diff0), vget_high_s16(diff0));
                                v_scores_q0_0 = vaddw_s32(v_scores_q0_0, vget_low_s32(sq0_lo));
                                v_scores_q0_1 = vaddw_high_s32(v_scores_q0_1, sq0_lo);
                                v_scores_q0_2 = vaddw_s32(v_scores_q0_2, vget_low_s32(sq0_hi));
                                v_scores_q0_3 = vaddw_high_s32(v_scores_q0_3, sq0_hi);

                                // Q1
                                let v_query_q1 = q_coord_dupped[1][$d];
                                let diff1 = vsubq_s16(v_query_q1, raw);
                                let sq1_lo = vmull_s16(vget_low_s16(diff1), vget_low_s16(diff1));
                                let sq1_hi = vmull_s16(vget_high_s16(diff1), vget_high_s16(diff1));
                                v_scores_q1_0 = vaddw_s32(v_scores_q1_0, vget_low_s32(sq1_lo));
                                v_scores_q1_1 = vaddw_high_s32(v_scores_q1_1, sq1_lo);
                                v_scores_q1_2 = vaddw_s32(v_scores_q1_2, vget_low_s32(sq1_hi));
                                v_scores_q1_3 = vaddw_high_s32(v_scores_q1_3, sq1_hi);

                                // Q2
                                let v_query_q2 = q_coord_dupped[2][$d];
                                let diff2 = vsubq_s16(v_query_q2, raw);
                                let sq2_lo = vmull_s16(vget_low_s16(diff2), vget_low_s16(diff2));
                                let sq2_hi = vmull_s16(vget_high_s16(diff2), vget_high_s16(diff2));
                                v_scores_q2_0 = vaddw_s32(v_scores_q2_0, vget_low_s32(sq2_lo));
                                v_scores_q2_1 = vaddw_high_s32(v_scores_q2_1, sq2_lo);
                                v_scores_q2_2 = vaddw_s32(v_scores_q2_2, vget_low_s32(sq2_hi));
                                v_scores_q2_3 = vaddw_high_s32(v_scores_q2_3, sq2_hi);

                                // Q3
                                let v_query_q3 = q_coord_dupped[3][$d];
                                let diff3 = vsubq_s16(v_query_q3, raw);
                                let sq3_lo = vmull_s16(vget_low_s16(diff3), vget_low_s16(diff3));
                                let sq3_hi = vmull_s16(vget_high_s16(diff3), vget_high_s16(diff3));
                                v_scores_q3_0 = vaddw_s32(v_scores_q3_0, vget_low_s32(sq3_lo));
                                v_scores_q3_1 = vaddw_high_s32(v_scores_q3_1, sq3_lo);
                                v_scores_q3_2 = vaddw_s32(v_scores_q3_2, vget_low_s32(sq3_hi));
                                v_scores_q3_3 = vaddw_high_s32(v_scores_q3_3, sq3_hi);
                            };
                        }

                        step_coord_unweighted!(0);
                        step_coord_unweighted!(1);
                        step_coord_unweighted!(2);
                        step_coord_unweighted!(3);
                        step_coord_unweighted!(4);
                        step_coord_unweighted!(5);
                        step_coord_unweighted!(6);
                        step_coord_unweighted!(7);

                        macro_rules! step_sketch_unweighted {
                            ($d:expr) => {
                                let raw = vld1_s8(sketches_ptr.add($d * block_size + inner_lane));

                                // Q0
                                let v_query_q0 = q_sketch_dupped[0][$d];
                                let diff0 = vsub_s8(v_query_q0, raw);
                                let diff0_16 = vmovl_s8(diff0);
                                let sq0_lo =
                                    vmull_s16(vget_low_s16(diff0_16), vget_low_s16(diff0_16));
                                let sq0_hi =
                                    vmull_s16(vget_high_s16(diff0_16), vget_high_s16(diff0_16));
                                v_scores_q0_0 = vaddw_s32(v_scores_q0_0, vget_low_s32(sq0_lo));
                                v_scores_q0_1 = vaddw_high_s32(v_scores_q0_1, sq0_lo);
                                v_scores_q0_2 = vaddw_s32(v_scores_q0_2, vget_low_s32(sq0_hi));
                                v_scores_q0_3 = vaddw_high_s32(v_scores_q0_3, sq0_hi);

                                // Q1
                                let v_query_q1 = q_sketch_dupped[1][$d];
                                let diff1 = vsub_s8(v_query_q1, raw);
                                let diff1_16 = vmovl_s8(diff1);
                                let sq1_lo =
                                    vmull_s16(vget_low_s16(diff1_16), vget_low_s16(diff1_16));
                                let sq1_hi =
                                    vmull_s16(vget_high_s16(diff1_16), vget_high_s16(diff1_16));
                                v_scores_q1_0 = vaddw_s32(v_scores_q1_0, vget_low_s32(sq1_lo));
                                v_scores_q1_1 = vaddw_high_s32(v_scores_q1_1, sq1_lo);
                                v_scores_q1_2 = vaddw_s32(v_scores_q1_2, vget_low_s32(sq1_hi));
                                v_scores_q1_3 = vaddw_high_s32(v_scores_q1_3, sq1_hi);

                                // Q2
                                let v_query_q2 = q_sketch_dupped[2][$d];
                                let diff2 = vsub_s8(v_query_q2, raw);
                                let diff2_16 = vmovl_s8(diff2);
                                let sq2_lo =
                                    vmull_s16(vget_low_s16(diff2_16), vget_low_s16(diff2_16));
                                let sq2_hi =
                                    vmull_s16(vget_high_s16(diff2_16), vget_high_s16(diff2_16));
                                v_scores_q2_0 = vaddw_s32(v_scores_q2_0, vget_low_s32(sq2_lo));
                                v_scores_q2_1 = vaddw_high_s32(v_scores_q2_1, sq2_lo);
                                v_scores_q2_2 = vaddw_s32(v_scores_q2_2, vget_low_s32(sq2_hi));
                                v_scores_q2_3 = vaddw_high_s32(v_scores_q2_3, sq2_hi);

                                // Q3
                                let v_query_q3 = q_sketch_dupped[3][$d];
                                let diff3 = vsub_s8(v_query_q3, raw);
                                let diff3_16 = vmovl_s8(diff3);
                                let sq3_lo =
                                    vmull_s16(vget_low_s16(diff3_16), vget_low_s16(diff3_16));
                                let sq3_hi =
                                    vmull_s16(vget_high_s16(diff3_16), vget_high_s16(diff3_16));
                                v_scores_q3_0 = vaddw_s32(v_scores_q3_0, vget_low_s32(sq3_lo));
                                v_scores_q3_1 = vaddw_high_s32(v_scores_q3_1, sq3_lo);
                                v_scores_q3_2 = vaddw_s32(v_scores_q3_2, vget_low_s32(sq3_hi));
                                v_scores_q3_3 = vaddw_high_s32(v_scores_q3_3, sq3_hi);
                            };
                        }

                        step_sketch_unweighted!(0);
                        step_sketch_unweighted!(1);
                        step_sketch_unweighted!(2);
                        step_sketch_unweighted!(3);
                    } else {
                        for (dim, &weight) in weights.coord.iter().enumerate() {
                            let raw = vld1q_s16(coords_ptr.add(dim * block_size + inner_lane));
                            let v_weight = vdupq_n_s32(weight as i32);

                            let v_query_q0 = vdupq_n_s16(*q0_coords.get_unchecked(dim));
                            let v_query_q1 = vdupq_n_s16(*q1_coords.get_unchecked(dim));
                            let v_query_q2 = vdupq_n_s16(*q2_coords.get_unchecked(dim));
                            let v_query_q3 = vdupq_n_s16(*q3_coords.get_unchecked(dim));

                            let diff0 = vsubq_s16(v_query_q0, raw);
                            let diff1 = vsubq_s16(v_query_q1, raw);
                            let diff2 = vsubq_s16(v_query_q2, raw);
                            let diff3 = vsubq_s16(v_query_q3, raw);

                            let sq0_lo = vmull_s16(vget_low_s16(diff0), vget_low_s16(diff0));
                            let sq0_hi = vmull_s16(vget_high_s16(diff0), vget_high_s16(diff0));
                            let sq1_lo = vmull_s16(vget_low_s16(diff1), vget_low_s16(diff1));
                            let sq1_hi = vmull_s16(vget_high_s16(diff1), vget_high_s16(diff1));
                            let sq2_lo = vmull_s16(vget_low_s16(diff2), vget_low_s16(diff2));
                            let sq2_hi = vmull_s16(vget_high_s16(diff2), vget_high_s16(diff2));
                            let sq3_lo = vmull_s16(vget_low_s16(diff3), vget_low_s16(diff3));
                            let sq3_hi = vmull_s16(vget_high_s16(diff3), vget_high_s16(diff3));

                            let prod0_lo = vmulq_s32(sq0_lo, v_weight);
                            let prod0_hi = vmulq_s32(sq0_hi, v_weight);
                            let prod1_lo = vmulq_s32(sq1_lo, v_weight);
                            let prod1_hi = vmulq_s32(sq1_hi, v_weight);
                            let prod2_lo = vmulq_s32(sq2_lo, v_weight);
                            let prod2_hi = vmulq_s32(sq2_hi, v_weight);
                            let prod3_lo = vmulq_s32(sq3_lo, v_weight);
                            let prod3_hi = vmulq_s32(sq3_hi, v_weight);

                            v_scores_q0_0 = vaddw_s32(v_scores_q0_0, vget_low_s32(prod0_lo));
                            v_scores_q0_1 = vaddw_high_s32(v_scores_q0_1, prod0_lo);
                            v_scores_q0_2 = vaddw_s32(v_scores_q0_2, vget_low_s32(prod0_hi));
                            v_scores_q0_3 = vaddw_high_s32(v_scores_q0_3, prod0_hi);

                            v_scores_q1_0 = vaddw_s32(v_scores_q1_0, vget_low_s32(prod1_lo));
                            v_scores_q1_1 = vaddw_high_s32(v_scores_q1_1, prod1_lo);
                            v_scores_q1_2 = vaddw_s32(v_scores_q1_2, vget_low_s32(prod1_hi));
                            v_scores_q1_3 = vaddw_high_s32(v_scores_q1_3, prod1_hi);

                            v_scores_q2_0 = vaddw_s32(v_scores_q2_0, vget_low_s32(prod2_lo));
                            v_scores_q2_1 = vaddw_high_s32(v_scores_q2_1, prod2_lo);
                            v_scores_q2_2 = vaddw_s32(v_scores_q2_2, vget_low_s32(prod2_hi));
                            v_scores_q2_3 = vaddw_high_s32(v_scores_q2_3, prod2_hi);

                            v_scores_q3_0 = vaddw_s32(v_scores_q3_0, vget_low_s32(prod3_lo));
                            v_scores_q3_1 = vaddw_high_s32(v_scores_q3_1, prod3_lo);
                            v_scores_q3_2 = vaddw_s32(v_scores_q3_2, vget_low_s32(prod3_hi));
                            v_scores_q3_3 = vaddw_high_s32(v_scores_q3_3, prod3_hi);
                        }

                        for (dim, &weight) in weights.sketch.iter().enumerate() {
                            let raw = vld1_s8(sketches_ptr.add(dim * block_size + inner_lane));
                            let v_weight = vdupq_n_s32(weight as i32);

                            let v_query_q0 = vdup_n_s8(*q0_sketches.get_unchecked(dim));
                            let v_query_q1 = vdup_n_s8(*q1_sketches.get_unchecked(dim));
                            let v_query_q2 = vdup_n_s8(*q2_sketches.get_unchecked(dim));
                            let v_query_q3 = vdup_n_s8(*q3_sketches.get_unchecked(dim));

                            let diff0 = vsub_s8(v_query_q0, raw);
                            let diff1 = vsub_s8(v_query_q1, raw);
                            let diff2 = vsub_s8(v_query_q2, raw);
                            let diff3 = vsub_s8(v_query_q3, raw);

                            let diff0_16 = vmovl_s8(diff0);
                            let diff1_16 = vmovl_s8(diff1);
                            let diff2_16 = vmovl_s8(diff2);
                            let diff3_16 = vmovl_s8(diff3);

                            let sq0_lo = vmull_s16(vget_low_s16(diff0_16), vget_low_s16(diff0_16));
                            let sq0_hi =
                                vmull_s16(vget_high_s16(diff0_16), vget_high_s16(diff0_16));
                            let sq1_lo = vmull_s16(vget_low_s16(diff1_16), vget_low_s16(diff1_16));
                            let sq1_hi =
                                vmull_s16(vget_high_s16(diff1_16), vget_high_s16(diff1_16));
                            let sq2_lo = vmull_s16(vget_low_s16(diff2_16), vget_low_s16(diff2_16));
                            let sq2_hi =
                                vmull_s16(vget_high_s16(diff2_16), vget_high_s16(diff2_16));
                            let sq3_lo = vmull_s16(vget_low_s16(diff3_16), vget_low_s16(diff3_16));
                            let sq3_hi =
                                vmull_s16(vget_high_s16(diff3_16), vget_high_s16(diff3_16));

                            let prod0_lo = vmulq_s32(sq0_lo, v_weight);
                            let prod0_hi = vmulq_s32(sq0_hi, v_weight);
                            let prod1_lo = vmulq_s32(sq1_lo, v_weight);
                            let prod1_hi = vmulq_s32(sq1_hi, v_weight);
                            let prod2_lo = vmulq_s32(sq2_lo, v_weight);
                            let prod2_hi = vmulq_s32(sq2_hi, v_weight);
                            let prod3_lo = vmulq_s32(sq3_lo, v_weight);
                            let prod3_hi = vmulq_s32(sq3_hi, v_weight);

                            v_scores_q0_0 = vaddw_s32(v_scores_q0_0, vget_low_s32(prod0_lo));
                            v_scores_q0_1 = vaddw_high_s32(v_scores_q0_1, prod0_lo);
                            v_scores_q0_2 = vaddw_s32(v_scores_q0_2, vget_low_s32(prod0_hi));
                            v_scores_q0_3 = vaddw_high_s32(v_scores_q0_3, prod0_hi);

                            v_scores_q1_0 = vaddw_s32(v_scores_q1_0, vget_low_s32(prod1_lo));
                            v_scores_q1_1 = vaddw_high_s32(v_scores_q1_1, prod1_lo);
                            v_scores_q1_2 = vaddw_s32(v_scores_q1_2, vget_low_s32(prod1_hi));
                            v_scores_q1_3 = vaddw_high_s32(v_scores_q1_3, prod1_hi);

                            v_scores_q2_0 = vaddw_s32(v_scores_q2_0, vget_low_s32(prod2_lo));
                            v_scores_q2_1 = vaddw_high_s32(v_scores_q2_1, prod2_lo);
                            v_scores_q2_2 = vaddw_s32(v_scores_q2_2, vget_low_s32(prod2_hi));
                            v_scores_q2_3 = vaddw_high_s32(v_scores_q2_3, prod2_hi);

                            v_scores_q3_0 = vaddw_s32(v_scores_q3_0, vget_low_s32(prod3_lo));
                            v_scores_q3_1 = vaddw_high_s32(v_scores_q3_1, prod3_lo);
                            v_scores_q3_2 = vaddw_s32(v_scores_q3_2, vget_low_s32(prod3_hi));
                            v_scores_q3_3 = vaddw_high_s32(v_scores_q3_3, prod3_hi);
                        }
                    }

                    vst1q_s64(s0_ptr.add(inner_lane), v_scores_q0_0);
                    vst1q_s64(s0_ptr.add(inner_lane + 2), v_scores_q0_1);
                    vst1q_s64(s0_ptr.add(inner_lane + 4), v_scores_q0_2);
                    vst1q_s64(s0_ptr.add(inner_lane + 6), v_scores_q0_3);

                    vst1q_s64(s1_ptr.add(inner_lane), v_scores_q1_0);
                    vst1q_s64(s1_ptr.add(inner_lane + 2), v_scores_q1_1);
                    vst1q_s64(s1_ptr.add(inner_lane + 4), v_scores_q1_2);
                    vst1q_s64(s1_ptr.add(inner_lane + 6), v_scores_q1_3);

                    vst1q_s64(s2_ptr.add(inner_lane), v_scores_q2_0);
                    vst1q_s64(s2_ptr.add(inner_lane + 2), v_scores_q2_1);
                    vst1q_s64(s2_ptr.add(inner_lane + 4), v_scores_q2_2);
                    vst1q_s64(s2_ptr.add(inner_lane + 6), v_scores_q2_3);

                    vst1q_s64(s3_ptr.add(inner_lane), v_scores_q3_0);
                    vst1q_s64(s3_ptr.add(inner_lane + 2), v_scores_q3_1);
                    vst1q_s64(s3_ptr.add(inner_lane + 4), v_scores_q3_2);
                    vst1q_s64(s3_ptr.add(inner_lane + 6), v_scores_q3_3);

                    inner_lane += 8;
                }
            } else {
                while inner_lane + 8 <= lanes {
                    let raw_res = vld1q_u16(residual_ptr.add(inner_lane));
                    let res_lo = vmovl_u16(vget_low_u16(raw_res));
                    let res_hi = vmovl_u16(vget_high_u16(raw_res));

                    // Q0 scores init
                    let sum_q0_lo = vaddq_u32(v_q0_residual, res_lo);
                    let sum_q0_hi = vaddq_u32(v_q0_residual, res_hi);
                    let prod_q0_lo = vmulq_s32(vreinterpretq_s32_u32(sum_q0_lo), v_res_weight);
                    let prod_q0_hi = vmulq_s32(vreinterpretq_s32_u32(sum_q0_hi), v_res_weight);
                    let mut v_scores_q0_0 = vmovl_s32(vget_low_s32(prod_q0_lo));
                    let mut v_scores_q0_1 = vmovl_s32(vget_high_s32(prod_q0_lo));
                    let mut v_scores_q0_2 = vmovl_s32(vget_low_s32(prod_q0_hi));
                    let mut v_scores_q0_3 = vmovl_s32(vget_high_s32(prod_q0_hi));

                    // Q1 scores init
                    let sum_q1_lo = vaddq_u32(v_q1_residual, res_lo);
                    let sum_q1_hi = vaddq_u32(v_q1_residual, res_hi);
                    let prod_q1_lo = vmulq_s32(vreinterpretq_s32_u32(sum_q1_lo), v_res_weight);
                    let prod_q1_hi = vmulq_s32(vreinterpretq_s32_u32(sum_q1_hi), v_res_weight);
                    let mut v_scores_q1_0 = vmovl_s32(vget_low_s32(prod_q1_lo));
                    let mut v_scores_q1_1 = vmovl_s32(vget_high_s32(prod_q1_lo));
                    let mut v_scores_q1_2 = vmovl_s32(vget_low_s32(prod_q1_hi));
                    let mut v_scores_q1_3 = vmovl_s32(vget_high_s32(prod_q1_hi));

                    // Q2 scores init
                    let sum_q2_lo = vaddq_u32(v_q2_residual, res_lo);
                    let sum_q2_hi = vaddq_u32(v_q2_residual, res_hi);
                    let prod_q2_lo = vmulq_s32(vreinterpretq_s32_u32(sum_q2_lo), v_res_weight);
                    let prod_q2_hi = vmulq_s32(vreinterpretq_s32_u32(sum_q2_hi), v_res_weight);
                    let mut v_scores_q2_0 = vmovl_s32(vget_low_s32(prod_q2_lo));
                    let mut v_scores_q2_1 = vmovl_s32(vget_high_s32(prod_q2_lo));
                    let mut v_scores_q2_2 = vmovl_s32(vget_low_s32(prod_q2_hi));
                    let mut v_scores_q2_3 = vmovl_s32(vget_high_s32(prod_q2_hi));

                    // Q3 scores init
                    let sum_q3_lo = vaddq_u32(v_q3_residual, res_lo);
                    let sum_q3_hi = vaddq_u32(v_q3_residual, res_hi);
                    let prod_q3_lo = vmulq_s32(vreinterpretq_s32_u32(sum_q3_lo), v_res_weight);
                    let prod_q3_hi = vmulq_s32(vreinterpretq_s32_u32(sum_q3_hi), v_res_weight);
                    let mut v_scores_q3_0 = vmovl_s32(vget_low_s32(prod_q3_lo));
                    let mut v_scores_q3_1 = vmovl_s32(vget_high_s32(prod_q3_lo));
                    let mut v_scores_q3_2 = vmovl_s32(vget_low_s32(prod_q3_hi));
                    let mut v_scores_q3_3 = vmovl_s32(vget_high_s32(prod_q3_hi));

                    if weights.coord.len() == 8 && weights.sketch.len() == 4 {
                        macro_rules! step_coord {
                            ($d:expr) => {
                                let raw = vld1q_s16(coords_ptr.add($d * block_size + inner_lane));
                                let v_weight = w_coord_dupped[$d];

                                // Q0
                                let v_query_q0 = q_coord_dupped[0][$d];
                                let diff0 = vsubq_s16(v_query_q0, raw);
                                let sq0_lo = vmull_s16(vget_low_s16(diff0), vget_low_s16(diff0));
                                let sq0_hi = vmull_s16(vget_high_s16(diff0), vget_high_s16(diff0));
                                let prod0_lo = vmulq_s32(sq0_lo, v_weight);
                                let prod0_hi = vmulq_s32(sq0_hi, v_weight);
                                v_scores_q0_0 = vaddw_s32(v_scores_q0_0, vget_low_s32(prod0_lo));
                                v_scores_q0_1 = vaddw_high_s32(v_scores_q0_1, prod0_lo);
                                v_scores_q0_2 = vaddw_s32(v_scores_q0_2, vget_low_s32(prod0_hi));
                                v_scores_q0_3 = vaddw_high_s32(v_scores_q0_3, prod0_hi);

                                // Q1
                                let v_query_q1 = q_coord_dupped[1][$d];
                                let diff1 = vsubq_s16(v_query_q1, raw);
                                let sq1_lo = vmull_s16(vget_low_s16(diff1), vget_low_s16(diff1));
                                let sq1_hi = vmull_s16(vget_high_s16(diff1), vget_high_s16(diff1));
                                let prod1_lo = vmulq_s32(sq1_lo, v_weight);
                                let prod1_hi = vmulq_s32(sq1_hi, v_weight);
                                v_scores_q1_0 = vaddw_s32(v_scores_q1_0, vget_low_s32(prod1_lo));
                                v_scores_q1_1 = vaddw_high_s32(v_scores_q1_1, prod1_lo);
                                v_scores_q1_2 = vaddw_s32(v_scores_q1_2, vget_low_s32(prod1_hi));
                                v_scores_q1_3 = vaddw_high_s32(v_scores_q1_3, prod1_hi);

                                // Q2
                                let v_query_q2 = q_coord_dupped[2][$d];
                                let diff2 = vsubq_s16(v_query_q2, raw);
                                let sq2_lo = vmull_s16(vget_low_s16(diff2), vget_low_s16(diff2));
                                let sq2_hi = vmull_s16(vget_high_s16(diff2), vget_high_s16(diff2));
                                let prod2_lo = vmulq_s32(sq2_lo, v_weight);
                                let prod2_hi = vmulq_s32(sq2_hi, v_weight);
                                v_scores_q2_0 = vaddw_s32(v_scores_q2_0, vget_low_s32(prod2_lo));
                                v_scores_q2_1 = vaddw_high_s32(v_scores_q2_1, prod2_lo);
                                v_scores_q2_2 = vaddw_s32(v_scores_q2_2, vget_low_s32(prod2_hi));
                                v_scores_q2_3 = vaddw_high_s32(v_scores_q2_3, prod2_hi);

                                // Q3
                                let v_query_q3 = q_coord_dupped[3][$d];
                                let diff3 = vsubq_s16(v_query_q3, raw);
                                let sq3_lo = vmull_s16(vget_low_s16(diff3), vget_low_s16(diff3));
                                let sq3_hi = vmull_s16(vget_high_s16(diff3), vget_high_s16(diff3));
                                let prod3_lo = vmulq_s32(sq3_lo, v_weight);
                                let prod3_hi = vmulq_s32(sq3_hi, v_weight);
                                v_scores_q3_0 = vaddw_s32(v_scores_q3_0, vget_low_s32(prod3_lo));
                                v_scores_q3_1 = vaddw_high_s32(v_scores_q3_1, prod3_lo);
                                v_scores_q3_2 = vaddw_s32(v_scores_q3_2, vget_low_s32(prod3_hi));
                                v_scores_q3_3 = vaddw_high_s32(v_scores_q3_3, prod3_hi);
                            };
                        }

                        step_coord!(0);
                        step_coord!(1);
                        step_coord!(2);
                        step_coord!(3);
                        step_coord!(4);
                        step_coord!(5);
                        step_coord!(6);
                        step_coord!(7);

                        macro_rules! step_sketch {
                            ($d:expr) => {
                                let raw = vld1_s8(sketches_ptr.add($d * block_size + inner_lane));
                                let v_weight = w_sketch_dupped[$d];

                                // Q0
                                let v_query_q0 = q_sketch_dupped[0][$d];
                                let diff0 = vsub_s8(v_query_q0, raw);
                                let diff0_16 = vmovl_s8(diff0);
                                let sq0_lo =
                                    vmull_s16(vget_low_s16(diff0_16), vget_low_s16(diff0_16));
                                let sq0_hi =
                                    vmull_s16(vget_high_s16(diff0_16), vget_high_s16(diff0_16));
                                let prod0_lo = vmulq_s32(sq0_lo, v_weight);
                                let prod0_hi = vmulq_s32(sq0_hi, v_weight);
                                v_scores_q0_0 = vaddw_s32(v_scores_q0_0, vget_low_s32(prod0_lo));
                                v_scores_q0_1 = vaddw_high_s32(v_scores_q0_1, prod0_lo);
                                v_scores_q0_2 = vaddw_s32(v_scores_q0_2, vget_low_s32(prod0_hi));
                                v_scores_q0_3 = vaddw_high_s32(v_scores_q0_3, prod0_hi);

                                // Q1
                                let v_query_q1 = q_sketch_dupped[1][$d];
                                let diff1 = vsub_s8(v_query_q1, raw);
                                let diff1_16 = vmovl_s8(diff1);
                                let sq1_lo =
                                    vmull_s16(vget_low_s16(diff1_16), vget_low_s16(diff1_16));
                                let sq1_hi =
                                    vmull_s16(vget_high_s16(diff1_16), vget_high_s16(diff1_16));
                                let prod1_lo = vmulq_s32(sq1_lo, v_weight);
                                let prod1_hi = vmulq_s32(sq1_hi, v_weight);
                                v_scores_q1_0 = vaddw_s32(v_scores_q1_0, vget_low_s32(prod1_lo));
                                v_scores_q1_1 = vaddw_high_s32(v_scores_q1_1, prod1_lo);
                                v_scores_q1_2 = vaddw_s32(v_scores_q1_2, vget_low_s32(prod1_hi));
                                v_scores_q1_3 = vaddw_high_s32(v_scores_q1_3, prod1_hi);

                                // Q2
                                let v_query_q2 = q_sketch_dupped[2][$d];
                                let diff2 = vsub_s8(v_query_q2, raw);
                                let diff2_16 = vmovl_s8(diff2);
                                let sq2_lo =
                                    vmull_s16(vget_low_s16(diff2_16), vget_low_s16(diff2_16));
                                let sq2_hi =
                                    vmull_s16(vget_high_s16(diff2_16), vget_high_s16(diff2_16));
                                let prod2_lo = vmulq_s32(sq2_lo, v_weight);
                                let prod2_hi = vmulq_s32(sq2_hi, v_weight);
                                v_scores_q2_0 = vaddw_s32(v_scores_q2_0, vget_low_s32(prod2_lo));
                                v_scores_q2_1 = vaddw_high_s32(v_scores_q2_1, prod2_lo);
                                v_scores_q2_2 = vaddw_s32(v_scores_q2_2, vget_low_s32(prod2_hi));
                                v_scores_q2_3 = vaddw_high_s32(v_scores_q2_3, prod2_hi);

                                // Q3
                                let v_query_q3 = q_sketch_dupped[3][$d];
                                let diff3 = vsub_s8(v_query_q3, raw);
                                let diff3_16 = vmovl_s8(diff3);
                                let sq3_lo =
                                    vmull_s16(vget_low_s16(diff3_16), vget_low_s16(diff3_16));
                                let sq3_hi =
                                    vmull_s16(vget_high_s16(diff3_16), vget_high_s16(diff3_16));
                                let prod3_lo = vmulq_s32(sq3_lo, v_weight);
                                let prod3_hi = vmulq_s32(sq3_hi, v_weight);
                                v_scores_q3_0 = vaddw_s32(v_scores_q3_0, vget_low_s32(prod3_lo));
                                v_scores_q3_1 = vaddw_high_s32(v_scores_q3_1, prod3_lo);
                                v_scores_q3_2 = vaddw_s32(v_scores_q3_2, vget_low_s32(prod3_hi));
                                v_scores_q3_3 = vaddw_high_s32(v_scores_q3_3, prod3_hi);
                            };
                        }

                        step_sketch!(0);
                        step_sketch!(1);
                        step_sketch!(2);
                        step_sketch!(3);
                    } else {
                        for (dim, &weight) in weights.coord.iter().enumerate() {
                            let raw = vld1q_s16(coords_ptr.add(dim * block_size + inner_lane));
                            let v_weight = vdupq_n_s32(weight as i32);

                            let v_query_q0 = vdupq_n_s16(*q0_coords.get_unchecked(dim));
                            let v_query_q1 = vdupq_n_s16(*q1_coords.get_unchecked(dim));
                            let v_query_q2 = vdupq_n_s16(*q2_coords.get_unchecked(dim));
                            let v_query_q3 = vdupq_n_s16(*q3_coords.get_unchecked(dim));

                            let diff0 = vsubq_s16(v_query_q0, raw);
                            let diff1 = vsubq_s16(v_query_q1, raw);
                            let diff2 = vsubq_s16(v_query_q2, raw);
                            let diff3 = vsubq_s16(v_query_q3, raw);

                            let sq0_lo = vmull_s16(vget_low_s16(diff0), vget_low_s16(diff0));
                            let sq0_hi = vmull_s16(vget_high_s16(diff0), vget_high_s16(diff0));
                            let sq1_lo = vmull_s16(vget_low_s16(diff1), vget_low_s16(diff1));
                            let sq1_hi = vmull_s16(vget_high_s16(diff1), vget_high_s16(diff1));
                            let sq2_lo = vmull_s16(vget_low_s16(diff2), vget_low_s16(diff2));
                            let sq2_hi = vmull_s16(vget_high_s16(diff2), vget_high_s16(diff2));
                            let sq3_lo = vmull_s16(vget_low_s16(diff3), vget_low_s16(diff3));
                            let sq3_hi = vmull_s16(vget_high_s16(diff3), vget_high_s16(diff3));

                            let prod0_lo = vmulq_s32(sq0_lo, v_weight);
                            let prod0_hi = vmulq_s32(sq0_hi, v_weight);
                            let prod1_lo = vmulq_s32(sq1_lo, v_weight);
                            let prod1_hi = vmulq_s32(sq1_hi, v_weight);
                            let prod2_lo = vmulq_s32(sq2_lo, v_weight);
                            let prod2_hi = vmulq_s32(sq2_hi, v_weight);
                            let prod3_lo = vmulq_s32(sq3_lo, v_weight);
                            let prod3_hi = vmulq_s32(sq3_hi, v_weight);

                            v_scores_q0_0 = vaddw_s32(v_scores_q0_0, vget_low_s32(prod0_lo));
                            v_scores_q0_1 = vaddw_high_s32(v_scores_q0_1, prod0_lo);
                            v_scores_q0_2 = vaddw_s32(v_scores_q0_2, vget_low_s32(prod0_hi));
                            v_scores_q0_3 = vaddw_high_s32(v_scores_q0_3, prod0_hi);

                            v_scores_q1_0 = vaddw_s32(v_scores_q1_0, vget_low_s32(prod1_lo));
                            v_scores_q1_1 = vaddw_high_s32(v_scores_q1_1, prod1_lo);
                            v_scores_q1_2 = vaddw_s32(v_scores_q1_2, vget_low_s32(prod1_hi));
                            v_scores_q1_3 = vaddw_high_s32(v_scores_q1_3, prod1_hi);

                            v_scores_q2_0 = vaddw_s32(v_scores_q2_0, vget_low_s32(prod2_lo));
                            v_scores_q2_1 = vaddw_high_s32(v_scores_q2_1, prod2_lo);
                            v_scores_q2_2 = vaddw_s32(v_scores_q2_2, vget_low_s32(prod2_hi));
                            v_scores_q2_3 = vaddw_high_s32(v_scores_q2_3, prod2_hi);

                            v_scores_q3_0 = vaddw_s32(v_scores_q3_0, vget_low_s32(prod3_lo));
                            v_scores_q3_1 = vaddw_high_s32(v_scores_q3_1, prod3_lo);
                            v_scores_q3_2 = vaddw_s32(v_scores_q3_2, vget_low_s32(prod3_hi));
                            v_scores_q3_3 = vaddw_high_s32(v_scores_q3_3, prod3_hi);
                        }

                        for (dim, &weight) in weights.sketch.iter().enumerate() {
                            let raw = vld1_s8(sketches_ptr.add(dim * block_size + inner_lane));
                            let v_weight = vdupq_n_s32(weight as i32);

                            let v_query_q0 = vdup_n_s8(*q0_sketches.get_unchecked(dim));
                            let v_query_q1 = vdup_n_s8(*q1_sketches.get_unchecked(dim));
                            let v_query_q2 = vdup_n_s8(*q2_sketches.get_unchecked(dim));
                            let v_query_q3 = vdup_n_s8(*q3_sketches.get_unchecked(dim));

                            let diff0 = vsub_s8(v_query_q0, raw);
                            let diff1 = vsub_s8(v_query_q1, raw);
                            let diff2 = vsub_s8(v_query_q2, raw);
                            let diff3 = vsub_s8(v_query_q3, raw);

                            let diff0_16 = vmovl_s8(diff0);
                            let diff1_16 = vmovl_s8(diff1);
                            let diff2_16 = vmovl_s8(diff2);
                            let diff3_16 = vmovl_s8(diff3);

                            let sq0_lo = vmull_s16(vget_low_s16(diff0_16), vget_low_s16(diff0_16));
                            let sq0_hi =
                                vmull_s16(vget_high_s16(diff0_16), vget_high_s16(diff0_16));
                            let sq1_lo = vmull_s16(vget_low_s16(diff1_16), vget_low_s16(diff1_16));
                            let sq1_hi =
                                vmull_s16(vget_high_s16(diff1_16), vget_high_s16(diff1_16));
                            let sq2_lo = vmull_s16(vget_low_s16(diff2_16), vget_low_s16(diff2_16));
                            let sq2_hi =
                                vmull_s16(vget_high_s16(diff2_16), vget_high_s16(diff2_16));
                            let sq3_lo = vmull_s16(vget_low_s16(diff3_16), vget_low_s16(diff3_16));
                            let sq3_hi =
                                vmull_s16(vget_high_s16(diff3_16), vget_high_s16(diff3_16));

                            let prod0_lo = vmulq_s32(sq0_lo, v_weight);
                            let prod0_hi = vmulq_s32(sq0_hi, v_weight);
                            let prod1_lo = vmulq_s32(sq1_lo, v_weight);
                            let prod1_hi = vmulq_s32(sq1_hi, v_weight);
                            let prod2_lo = vmulq_s32(sq2_lo, v_weight);
                            let prod2_hi = vmulq_s32(sq2_hi, v_weight);
                            let prod3_lo = vmulq_s32(sq3_lo, v_weight);
                            let prod3_hi = vmulq_s32(sq3_hi, v_weight);

                            v_scores_q0_0 = vaddw_s32(v_scores_q0_0, vget_low_s32(prod0_lo));
                            v_scores_q0_1 = vaddw_high_s32(v_scores_q0_1, prod0_lo);
                            v_scores_q0_2 = vaddw_s32(v_scores_q0_2, vget_low_s32(prod0_hi));
                            v_scores_q0_3 = vaddw_high_s32(v_scores_q0_3, prod0_hi);

                            v_scores_q1_0 = vaddw_s32(v_scores_q1_0, vget_low_s32(prod1_lo));
                            v_scores_q1_1 = vaddw_high_s32(v_scores_q1_1, prod1_lo);
                            v_scores_q1_2 = vaddw_s32(v_scores_q1_2, vget_low_s32(prod1_hi));
                            v_scores_q1_3 = vaddw_high_s32(v_scores_q1_3, prod1_hi);

                            v_scores_q2_0 = vaddw_s32(v_scores_q2_0, vget_low_s32(prod2_lo));
                            v_scores_q2_1 = vaddw_high_s32(v_scores_q2_1, prod2_lo);
                            v_scores_q2_2 = vaddw_s32(v_scores_q2_2, vget_low_s32(prod2_hi));
                            v_scores_q2_3 = vaddw_high_s32(v_scores_q2_3, prod2_hi);

                            v_scores_q3_0 = vaddw_s32(v_scores_q3_0, vget_low_s32(prod3_lo));
                            v_scores_q3_1 = vaddw_high_s32(v_scores_q3_1, prod3_lo);
                            v_scores_q3_2 = vaddw_s32(v_scores_q3_2, vget_low_s32(prod3_hi));
                            v_scores_q3_3 = vaddw_high_s32(v_scores_q3_3, prod3_hi);
                        }
                    }

                    vst1q_s64(s0_ptr.add(inner_lane), v_scores_q0_0);
                    vst1q_s64(s0_ptr.add(inner_lane + 2), v_scores_q0_1);
                    vst1q_s64(s0_ptr.add(inner_lane + 4), v_scores_q0_2);
                    vst1q_s64(s0_ptr.add(inner_lane + 6), v_scores_q0_3);

                    vst1q_s64(s1_ptr.add(inner_lane), v_scores_q1_0);
                    vst1q_s64(s1_ptr.add(inner_lane + 2), v_scores_q1_1);
                    vst1q_s64(s1_ptr.add(inner_lane + 4), v_scores_q1_2);
                    vst1q_s64(s1_ptr.add(inner_lane + 6), v_scores_q1_3);

                    vst1q_s64(s2_ptr.add(inner_lane), v_scores_q2_0);
                    vst1q_s64(s2_ptr.add(inner_lane + 2), v_scores_q2_1);
                    vst1q_s64(s2_ptr.add(inner_lane + 4), v_scores_q2_2);
                    vst1q_s64(s2_ptr.add(inner_lane + 6), v_scores_q2_3);

                    vst1q_s64(s3_ptr.add(inner_lane), v_scores_q3_0);
                    vst1q_s64(s3_ptr.add(inner_lane + 2), v_scores_q3_1);
                    vst1q_s64(s3_ptr.add(inner_lane + 4), v_scores_q3_2);
                    vst1q_s64(s3_ptr.add(inner_lane + 6), v_scores_q3_3);

                    inner_lane += 8;
                }
            }
            lane = inner_lane;
        } else if m <= 4 {
            let mut v_scores = [[vdupq_n_s64(0); 4]; 4];
            for q in 0..m {
                v_scores[q][0] = vld1q_s64(queries_scores[q].as_ptr().add(lane));
                v_scores[q][1] = vld1q_s64(queries_scores[q].as_ptr().add(lane + 2));
                v_scores[q][2] = vld1q_s64(queries_scores[q].as_ptr().add(lane + 4));
                v_scores[q][3] = vld1q_s64(queries_scores[q].as_ptr().add(lane + 6));
            }

            for (dim, &weight) in weights.coord.iter().enumerate() {
                let raw = vld1q_s16(layout.coord_block(block, dim).as_ptr().add(lane));
                let v_weight = vdupq_n_s32(weight as i32);

                for q in 0..m {
                    let v_query = vdupq_n_s16(queries_coords[q][dim]);
                    let diff = vsubq_s16(v_query, raw);

                    let diff_lo = vmovl_s16(vget_low_s16(diff));
                    let diff_hi = vmovl_s16(vget_high_s16(diff));

                    let sq_lo = vmulq_s32(diff_lo, diff_lo);
                    let sq_hi = vmulq_s32(diff_hi, diff_hi);

                    let prod_lo = vmulq_s32(sq_lo, v_weight);
                    let prod_hi = vmulq_s32(sq_hi, v_weight);

                    let prod_01 = vmovl_s32(vget_low_s32(prod_lo));
                    let prod_23 = vmovl_s32(vget_high_s32(prod_lo));
                    let prod_45 = vmovl_s32(vget_low_s32(prod_hi));
                    let prod_67 = vmovl_s32(vget_high_s32(prod_hi));

                    v_scores[q][0] = vaddq_s64(v_scores[q][0], prod_01);
                    v_scores[q][1] = vaddq_s64(v_scores[q][1], prod_23);
                    v_scores[q][2] = vaddq_s64(v_scores[q][2], prod_45);
                    v_scores[q][3] = vaddq_s64(v_scores[q][3], prod_67);
                }
            }

            for (dim, &weight) in weights.sketch.iter().enumerate() {
                let raw = vld1_s8(layout.sketch_block(block, dim).as_ptr().add(lane));
                let v_weight = vdupq_n_s32(weight as i32);

                for q in 0..m {
                    let v_query = vdup_n_s8(queries_sketches[q][dim]);
                    let diff = vsub_s8(v_query, raw);
                    let diff_16 = vmovl_s8(diff);

                    let diff_lo = vmovl_s16(vget_low_s16(diff_16));
                    let diff_hi = vmovl_s16(vget_high_s16(diff_16));

                    let sq_lo = vmulq_s32(diff_lo, diff_lo);
                    let sq_hi = vmulq_s32(diff_hi, diff_hi);

                    let prod_lo = vmulq_s32(sq_lo, v_weight);
                    let prod_hi = vmulq_s32(sq_hi, v_weight);

                    let prod_01 = vmovl_s32(vget_low_s32(prod_lo));
                    let prod_23 = vmovl_s32(vget_high_s32(prod_lo));
                    let prod_45 = vmovl_s32(vget_low_s32(prod_hi));
                    let prod_67 = vmovl_s32(vget_high_s32(prod_hi));

                    v_scores[q][0] = vaddq_s64(v_scores[q][0], prod_01);
                    v_scores[q][1] = vaddq_s64(v_scores[q][1], prod_23);
                    v_scores[q][2] = vaddq_s64(v_scores[q][2], prod_45);
                    v_scores[q][3] = vaddq_s64(v_scores[q][3], prod_67);
                }
            }

            for q in 0..m {
                vst1q_s64(queries_scores[q].as_mut_ptr().add(lane), v_scores[q][0]);
                vst1q_s64(queries_scores[q].as_mut_ptr().add(lane + 2), v_scores[q][1]);
                vst1q_s64(queries_scores[q].as_mut_ptr().add(lane + 4), v_scores[q][2]);
                vst1q_s64(queries_scores[q].as_mut_ptr().add(lane + 6), v_scores[q][3]);
            }
        } else {
            let mut v_scores = [[vdupq_n_s64(0); 4]; 8];
            for q in 0..m {
                v_scores[q][0] = vld1q_s64(queries_scores[q].as_ptr().add(lane));
                v_scores[q][1] = vld1q_s64(queries_scores[q].as_ptr().add(lane + 2));
                v_scores[q][2] = vld1q_s64(queries_scores[q].as_ptr().add(lane + 4));
                v_scores[q][3] = vld1q_s64(queries_scores[q].as_ptr().add(lane + 6));
            }

            for (dim, &weight) in weights.coord.iter().enumerate() {
                let raw = vld1q_s16(layout.coord_block(block, dim).as_ptr().add(lane));
                let v_weight = vdupq_n_s32(weight as i32);

                for q in 0..m {
                    let v_query = vdupq_n_s16(queries_coords[q][dim]);
                    let diff = vsubq_s16(v_query, raw);

                    let diff_lo = vmovl_s16(vget_low_s16(diff));
                    let diff_hi = vmovl_s16(vget_high_s16(diff));

                    let sq_lo = vmulq_s32(diff_lo, diff_lo);
                    let sq_hi = vmulq_s32(diff_hi, diff_hi);

                    let prod_lo = vmulq_s32(sq_lo, v_weight);
                    let prod_hi = vmulq_s32(sq_hi, v_weight);

                    let prod_01 = vmovl_s32(vget_low_s32(prod_lo));
                    let prod_23 = vmovl_s32(vget_high_s32(prod_lo));
                    let prod_45 = vmovl_s32(vget_low_s32(prod_hi));
                    let prod_67 = vmovl_s32(vget_high_s32(prod_hi));

                    v_scores[q][0] = vaddq_s64(v_scores[q][0], prod_01);
                    v_scores[q][1] = vaddq_s64(v_scores[q][1], prod_23);
                    v_scores[q][2] = vaddq_s64(v_scores[q][2], prod_45);
                    v_scores[q][3] = vaddq_s64(v_scores[q][3], prod_67);
                }
            }

            for (dim, &weight) in weights.sketch.iter().enumerate() {
                let raw = vld1_s8(layout.sketch_block(block, dim).as_ptr().add(lane));
                let v_weight = vdupq_n_s32(weight as i32);

                for q in 0..m {
                    let v_query = vdup_n_s8(queries_sketches[q][dim]);
                    let diff = vsub_s8(v_query, raw);
                    let diff_16 = vmovl_s8(diff);

                    let diff_lo = vmovl_s16(vget_low_s16(diff_16));
                    let diff_hi = vmovl_s16(vget_high_s16(diff_16));

                    let sq_lo = vmulq_s32(diff_lo, diff_lo);
                    let sq_hi = vmulq_s32(diff_hi, diff_hi);

                    let prod_lo = vmulq_s32(sq_lo, v_weight);
                    let prod_hi = vmulq_s32(sq_hi, v_weight);

                    let prod_01 = vmovl_s32(vget_low_s32(prod_lo));
                    let prod_23 = vmovl_s32(vget_high_s32(prod_lo));
                    let prod_45 = vmovl_s32(vget_low_s32(prod_hi));
                    let prod_67 = vmovl_s32(vget_high_s32(prod_hi));

                    v_scores[q][0] = vaddq_s64(v_scores[q][0], prod_01);
                    v_scores[q][1] = vaddq_s64(v_scores[q][1], prod_23);
                    v_scores[q][2] = vaddq_s64(v_scores[q][2], prod_45);
                    v_scores[q][3] = vaddq_s64(v_scores[q][3], prod_67);
                }
            }

            for q in 0..m {
                vst1q_s64(queries_scores[q].as_mut_ptr().add(lane), v_scores[q][0]);
                vst1q_s64(queries_scores[q].as_mut_ptr().add(lane + 2), v_scores[q][1]);
                vst1q_s64(queries_scores[q].as_mut_ptr().add(lane + 4), v_scores[q][2]);
                vst1q_s64(queries_scores[q].as_mut_ptr().add(lane + 6), v_scores[q][3]);
            }
        }
        lane += 8;
    }
    for q in 0..m {
        add_scalar_coord_tail(
            layout,
            block,
            lane,
            lanes,
            queries_coords[q],
            weights.coord,
            queries_scores[q],
        );
        add_scalar_sketch_tail(
            layout,
            block,
            lane,
            lanes,
            queries_sketches[q],
            weights.sketch,
            queries_scores[q],
        );
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
unsafe fn x86_avx2_scan_block_multi(
    layout: &BlockSoaLayout,
    block: usize,
    lanes: usize,
    queries_coords: &[&[i16]],
    queries_residual: &[u16],
    queries_sketches: &[&[i8]],
    weights: ScanWeights<'_>,
    queries_scores: &mut [&mut [i64]],
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let m = queries_coords.len();
    let vector_lanes = (lanes / 8) * 8;
    if m == 4 {
        if vector_lanes < lanes {
            let res_block = layout.residual_block(block);
            for q in 0..4 {
                residual_scores(
                    &res_block[vector_lanes..lanes],
                    queries_residual[q],
                    weights.residual,
                    &mut queries_scores[q][vector_lanes..lanes],
                );
            }
        }
    } else {
        for q in 0..m {
            residual_scores(
                &layout.residual_block(block)[..lanes],
                queries_residual[q],
                weights.residual,
                queries_scores[q],
            );
        }
    }

    let mut w_coord_v = [_mm256_setzero_si256(); 8];
    let mut q_coord_v = [[_mm256_setzero_si256(); 8]; 8];
    if weights.coord.len() == 8 {
        for (d, w) in w_coord_v.iter_mut().enumerate() {
            *w = _mm256_set1_epi64x(weights.coord[d]);
        }
        for q in 0..m {
            for d in 0..8 {
                q_coord_v[q][d] = _mm256_set1_epi32(i32::from(queries_coords[q][d]));
            }
        }
    }

    let mut w_sketch_v = [_mm256_setzero_si256(); 4];
    let mut q_sketch_v = [[_mm256_setzero_si256(); 4]; 8];
    if weights.sketch.len() == 4 {
        for (d, w) in w_sketch_v.iter_mut().enumerate() {
            *w = _mm256_set1_epi64x(weights.sketch[d]);
        }
        for q in 0..m {
            for d in 0..4 {
                q_sketch_v[q][d] = _mm256_set1_epi32(i32::from(queries_sketches[q][d]));
            }
        }
    }

    let mut lane = 0;
    while lane + 8 <= lanes {
        if m == 4 && weights.coord.len() == 8 && weights.sketch.len() == 4 {
            let residual_ptr = layout.residual_block(block).as_ptr();
            let raw_res = _mm_loadu_si128(residual_ptr.add(lane) as *const __m128i);
            let res_32 = _mm256_cvtepu16_epi32(raw_res);
            let v_res_weight = _mm256_set1_epi32(weights.residual as i32);

            // Q0 scores init
            let sum0 = _mm256_add_epi32(_mm256_set1_epi32(queries_residual[0] as i32), res_32);
            let prod0 = _mm256_mullo_epi32(sum0, v_res_weight);
            let mut v_scores_lo_0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(prod0));
            let mut v_scores_hi_0 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(prod0, 1));

            // Q1 scores init
            let sum1 = _mm256_add_epi32(_mm256_set1_epi32(queries_residual[1] as i32), res_32);
            let prod1 = _mm256_mullo_epi32(sum1, v_res_weight);
            let mut v_scores_lo_1 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(prod1));
            let mut v_scores_hi_1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(prod1, 1));

            // Q2 scores init
            let sum2 = _mm256_add_epi32(_mm256_set1_epi32(queries_residual[2] as i32), res_32);
            let prod2 = _mm256_mullo_epi32(sum2, v_res_weight);
            let mut v_scores_lo_2 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(prod2));
            let mut v_scores_hi_2 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(prod2, 1));

            // Q3 scores init
            let sum3 = _mm256_add_epi32(_mm256_set1_epi32(queries_residual[3] as i32), res_32);
            let prod3 = _mm256_mullo_epi32(sum3, v_res_weight);
            let mut v_scores_lo_3 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(prod3));
            let mut v_scores_hi_3 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(prod3, 1));

            macro_rules! step_coord {
                ($d:expr) => {
                    let ptr = layout.coord_block(block, $d).as_ptr().add(lane) as *const __m128i;
                    let raw = _mm_loadu_si128(ptr);
                    let values = _mm256_cvtepi16_epi32(raw);
                    let v_weight = w_coord_v[$d];

                    let diff0 = _mm256_sub_epi32(q_coord_v[0][$d], values);
                    let diff1 = _mm256_sub_epi32(q_coord_v[1][$d], values);
                    let diff2 = _mm256_sub_epi32(q_coord_v[2][$d], values);
                    let diff3 = _mm256_sub_epi32(q_coord_v[3][$d], values);

                    let even_sq0 = _mm256_mul_epi32(diff0, diff0);
                    let odd0 = _mm256_srli_si256(diff0, 4);
                    let odd_sq0 = _mm256_mul_epi32(odd0, odd0);

                    let even_sq1 = _mm256_mul_epi32(diff1, diff1);
                    let odd1 = _mm256_srli_si256(diff1, 4);
                    let odd_sq1 = _mm256_mul_epi32(odd1, odd1);

                    let even_sq2 = _mm256_mul_epi32(diff2, diff2);
                    let odd2 = _mm256_srli_si256(diff2, 4);
                    let odd_sq2 = _mm256_mul_epi32(odd2, odd2);

                    let even_sq3 = _mm256_mul_epi32(diff3, diff3);
                    let odd3 = _mm256_srli_si256(diff3, 4);
                    let odd_sq3 = _mm256_mul_epi32(odd3, odd3);

                    let (even_prod0, odd_prod0) = if weights.coord[$d] == 1 {
                        (even_sq0, odd_sq0)
                    } else {
                        (
                            _mm256_mul_epi32(even_sq0, v_weight),
                            _mm256_mul_epi32(odd_sq0, v_weight),
                        )
                    };

                    let (even_prod1, odd_prod1) = if weights.coord[$d] == 1 {
                        (even_sq1, odd_sq1)
                    } else {
                        (
                            _mm256_mul_epi32(even_sq1, v_weight),
                            _mm256_mul_epi32(odd_sq1, v_weight),
                        )
                    };

                    let (even_prod2, odd_prod2) = if weights.coord[$d] == 1 {
                        (even_sq2, odd_sq2)
                    } else {
                        (
                            _mm256_mul_epi32(even_sq2, v_weight),
                            _mm256_mul_epi32(odd_sq2, v_weight),
                        )
                    };

                    let (even_prod3, odd_prod3) = if weights.coord[$d] == 1 {
                        (even_sq3, odd_sq3)
                    } else {
                        (
                            _mm256_mul_epi32(even_sq3, v_weight),
                            _mm256_mul_epi32(odd_sq3, v_weight),
                        )
                    };

                    let unpack_lo0 = _mm256_unpacklo_epi64(even_prod0, odd_prod0);
                    let unpack_hi0 = _mm256_unpackhi_epi64(even_prod0, odd_prod0);
                    let prod_lo0 = _mm256_permute2x128_si256(unpack_lo0, unpack_hi0, 0x20);
                    let prod_hi0 = _mm256_permute2x128_si256(unpack_lo0, unpack_hi0, 0x31);
                    v_scores_lo_0 = _mm256_add_epi64(v_scores_lo_0, prod_lo0);
                    v_scores_hi_0 = _mm256_add_epi64(v_scores_hi_0, prod_hi0);

                    let unpack_lo1 = _mm256_unpacklo_epi64(even_prod1, odd_prod1);
                    let unpack_hi1 = _mm256_unpackhi_epi64(even_prod1, odd_prod1);
                    let prod_lo1 = _mm256_permute2x128_si256(unpack_lo1, unpack_hi1, 0x20);
                    let prod_hi1 = _mm256_permute2x128_si256(unpack_lo1, unpack_hi1, 0x31);
                    v_scores_lo_1 = _mm256_add_epi64(v_scores_lo_1, prod_lo1);
                    v_scores_hi_1 = _mm256_add_epi64(v_scores_hi_1, prod_hi1);

                    let unpack_lo2 = _mm256_unpacklo_epi64(even_prod2, odd_prod2);
                    let unpack_hi2 = _mm256_unpackhi_epi64(even_prod2, odd_prod2);
                    let prod_lo2 = _mm256_permute2x128_si256(unpack_lo2, unpack_hi2, 0x20);
                    let prod_hi2 = _mm256_permute2x128_si256(unpack_lo2, unpack_hi2, 0x31);
                    v_scores_lo_2 = _mm256_add_epi64(v_scores_lo_2, prod_lo2);
                    v_scores_hi_2 = _mm256_add_epi64(v_scores_hi_2, prod_hi2);

                    let unpack_lo3 = _mm256_unpacklo_epi64(even_prod3, odd_prod3);
                    let unpack_hi3 = _mm256_unpackhi_epi64(even_prod3, odd_prod3);
                    let prod_lo3 = _mm256_permute2x128_si256(unpack_lo3, unpack_hi3, 0x20);
                    let prod_hi3 = _mm256_permute2x128_si256(unpack_lo3, unpack_hi3, 0x31);
                    v_scores_lo_3 = _mm256_add_epi64(v_scores_lo_3, prod_lo3);
                    v_scores_hi_3 = _mm256_add_epi64(v_scores_hi_3, prod_hi3);
                };
            }

            step_coord!(0);
            step_coord!(1);
            step_coord!(2);
            step_coord!(3);
            step_coord!(4);
            step_coord!(5);
            step_coord!(6);
            step_coord!(7);

            macro_rules! step_sketch {
                ($d:expr) => {
                    let ptr = layout.sketch_block(block, $d).as_ptr().add(lane) as *const i64;
                    let packed = std::ptr::read_unaligned(ptr);
                    let raw = _mm_cvtsi64_si128(packed);
                    let values = _mm256_cvtepi8_epi32(raw);
                    let v_weight = w_sketch_v[$d];

                    let diff0 = _mm256_sub_epi32(q_sketch_v[0][$d], values);
                    let diff1 = _mm256_sub_epi32(q_sketch_v[1][$d], values);
                    let diff2 = _mm256_sub_epi32(q_sketch_v[2][$d], values);
                    let diff3 = _mm256_sub_epi32(q_sketch_v[3][$d], values);

                    let even_sq0 = _mm256_mul_epi32(diff0, diff0);
                    let odd0 = _mm256_srli_si256(diff0, 4);
                    let odd_sq0 = _mm256_mul_epi32(odd0, odd0);

                    let even_sq1 = _mm256_mul_epi32(diff1, diff1);
                    let odd1 = _mm256_srli_si256(diff1, 4);
                    let odd_sq1 = _mm256_mul_epi32(odd1, odd1);

                    let even_sq2 = _mm256_mul_epi32(diff2, diff2);
                    let odd2 = _mm256_srli_si256(diff2, 4);
                    let odd_sq2 = _mm256_mul_epi32(odd2, odd2);

                    let even_sq3 = _mm256_mul_epi32(diff3, diff3);
                    let odd3 = _mm256_srli_si256(diff3, 4);
                    let odd_sq3 = _mm256_mul_epi32(odd3, odd3);

                    let (even_prod0, odd_prod0) = if weights.sketch[$d] == 1 {
                        (even_sq0, odd_sq0)
                    } else {
                        (
                            _mm256_mul_epi32(even_sq0, v_weight),
                            _mm256_mul_epi32(odd_sq0, v_weight),
                        )
                    };

                    let (even_prod1, odd_prod1) = if weights.sketch[$d] == 1 {
                        (even_sq1, odd_sq1)
                    } else {
                        (
                            _mm256_mul_epi32(even_sq1, v_weight),
                            _mm256_mul_epi32(odd_sq1, v_weight),
                        )
                    };

                    let (even_prod2, odd_prod2) = if weights.sketch[$d] == 1 {
                        (even_sq2, odd_sq2)
                    } else {
                        (
                            _mm256_mul_epi32(even_sq2, v_weight),
                            _mm256_mul_epi32(odd_sq2, v_weight),
                        )
                    };

                    let (even_prod3, odd_prod3) = if weights.sketch[$d] == 1 {
                        (even_sq3, odd_sq3)
                    } else {
                        (
                            _mm256_mul_epi32(even_sq3, v_weight),
                            _mm256_mul_epi32(odd_sq3, v_weight),
                        )
                    };

                    let unpack_lo0 = _mm256_unpacklo_epi64(even_prod0, odd_prod0);
                    let unpack_hi0 = _mm256_unpackhi_epi64(even_prod0, odd_prod0);
                    let prod_lo0 = _mm256_permute2x128_si256(unpack_lo0, unpack_hi0, 0x20);
                    let prod_hi0 = _mm256_permute2x128_si256(unpack_lo0, unpack_hi0, 0x31);
                    v_scores_lo_0 = _mm256_add_epi64(v_scores_lo_0, prod_lo0);
                    v_scores_hi_0 = _mm256_add_epi64(v_scores_hi_0, prod_hi0);

                    let unpack_lo1 = _mm256_unpacklo_epi64(even_prod1, odd_prod1);
                    let unpack_hi1 = _mm256_unpackhi_epi64(even_prod1, odd_prod1);
                    let prod_lo1 = _mm256_permute2x128_si256(unpack_lo1, unpack_hi1, 0x20);
                    let prod_hi1 = _mm256_permute2x128_si256(unpack_lo1, unpack_hi1, 0x31);
                    v_scores_lo_1 = _mm256_add_epi64(v_scores_lo_1, prod_lo1);
                    v_scores_hi_1 = _mm256_add_epi64(v_scores_hi_1, prod_hi1);

                    let unpack_lo2 = _mm256_unpacklo_epi64(even_prod2, odd_prod2);
                    let unpack_hi2 = _mm256_unpackhi_epi64(even_prod2, odd_prod2);
                    let prod_lo2 = _mm256_permute2x128_si256(unpack_lo2, unpack_hi2, 0x20);
                    let prod_hi2 = _mm256_permute2x128_si256(unpack_lo2, unpack_hi2, 0x31);
                    v_scores_lo_2 = _mm256_add_epi64(v_scores_lo_2, prod_lo2);
                    v_scores_hi_2 = _mm256_add_epi64(v_scores_hi_2, prod_hi2);

                    let unpack_lo3 = _mm256_unpacklo_epi64(even_prod3, odd_prod3);
                    let unpack_hi3 = _mm256_unpackhi_epi64(even_prod3, odd_prod3);
                    let prod_lo3 = _mm256_permute2x128_si256(unpack_lo3, unpack_hi3, 0x20);
                    let prod_hi3 = _mm256_permute2x128_si256(unpack_lo3, unpack_hi3, 0x31);
                    v_scores_lo_3 = _mm256_add_epi64(v_scores_lo_3, prod_lo3);
                    v_scores_hi_3 = _mm256_add_epi64(v_scores_hi_3, prod_hi3);
                };
            }

            step_sketch!(0);
            step_sketch!(1);
            step_sketch!(2);
            step_sketch!(3);

            _mm256_storeu_si256(
                queries_scores[0].as_mut_ptr().add(lane) as *mut __m256i,
                v_scores_lo_0,
            );
            _mm256_storeu_si256(
                queries_scores[0].as_mut_ptr().add(lane + 4) as *mut __m256i,
                v_scores_hi_0,
            );
            _mm256_storeu_si256(
                queries_scores[1].as_mut_ptr().add(lane) as *mut __m256i,
                v_scores_lo_1,
            );
            _mm256_storeu_si256(
                queries_scores[1].as_mut_ptr().add(lane + 4) as *mut __m256i,
                v_scores_hi_1,
            );
            _mm256_storeu_si256(
                queries_scores[2].as_mut_ptr().add(lane) as *mut __m256i,
                v_scores_lo_2,
            );
            _mm256_storeu_si256(
                queries_scores[2].as_mut_ptr().add(lane + 4) as *mut __m256i,
                v_scores_hi_2,
            );
            _mm256_storeu_si256(
                queries_scores[3].as_mut_ptr().add(lane) as *mut __m256i,
                v_scores_lo_3,
            );
            _mm256_storeu_si256(
                queries_scores[3].as_mut_ptr().add(lane + 4) as *mut __m256i,
                v_scores_hi_3,
            );
        } else {
            let mut v_scores_lo = [_mm256_setzero_si256(); 8];
            let mut v_scores_hi = [_mm256_setzero_si256(); 8];
            for q in 0..m {
                v_scores_lo[q] =
                    _mm256_loadu_si256(queries_scores[q].as_ptr().add(lane) as *const __m256i);
                v_scores_hi[q] =
                    _mm256_loadu_si256(queries_scores[q].as_ptr().add(lane + 4) as *const __m256i);
            }

            for (dim, &weight) in weights.coord.iter().enumerate() {
                let ptr = layout.coord_block(block, dim).as_ptr().add(lane) as *const __m128i;
                let raw = _mm_loadu_si128(ptr);
                let values = _mm256_cvtepi16_epi32(raw);
                let v_weight = w_coord_v[dim];

                for q in 0..m {
                    let diff = _mm256_sub_epi32(q_coord_v[q][dim], values);
                    let even_sq = _mm256_mul_epi32(diff, diff);
                    let odd = _mm256_srli_si256(diff, 4);
                    let odd_sq = _mm256_mul_epi32(odd, odd);

                    let even_prod = if weight == 1 {
                        even_sq
                    } else {
                        _mm256_mul_epi32(even_sq, v_weight)
                    };
                    let odd_prod = if weight == 1 {
                        odd_sq
                    } else {
                        _mm256_mul_epi32(odd_sq, v_weight)
                    };

                    let unpack_lo = _mm256_unpacklo_epi64(even_prod, odd_prod);
                    let unpack_hi = _mm256_unpackhi_epi64(even_prod, odd_prod);

                    let prod_lo = _mm256_permute2x128_si256(unpack_lo, unpack_hi, 0x20);
                    let prod_hi = _mm256_permute2x128_si256(unpack_lo, unpack_hi, 0x31);

                    v_scores_lo[q] = _mm256_add_epi64(v_scores_lo[q], prod_lo);
                    v_scores_hi[q] = _mm256_add_epi64(v_scores_hi[q], prod_hi);
                }
            }

            for (dim, &weight) in weights.sketch.iter().enumerate() {
                let ptr = layout.sketch_block(block, dim).as_ptr().add(lane) as *const i64;
                let packed = std::ptr::read_unaligned(ptr);
                let raw = _mm_cvtsi64_si128(packed);
                let values = _mm256_cvtepi8_epi32(raw);
                let v_weight = w_sketch_v[dim];

                for q in 0..m {
                    let diff = _mm256_sub_epi32(q_sketch_v[q][dim], values);
                    let even_sq = _mm256_mul_epi32(diff, diff);
                    let odd = _mm256_srli_si256(diff, 4);
                    let odd_sq = _mm256_mul_epi32(odd, odd);

                    let even_prod = if weight == 1 {
                        even_sq
                    } else {
                        _mm256_mul_epi32(even_sq, v_weight)
                    };
                    let odd_prod = if weight == 1 {
                        odd_sq
                    } else {
                        _mm256_mul_epi32(odd_sq, v_weight)
                    };

                    let unpack_lo = _mm256_unpacklo_epi64(even_prod, odd_prod);
                    let unpack_hi = _mm256_unpackhi_epi64(even_prod, odd_prod);

                    let prod_lo = _mm256_permute2x128_si256(unpack_lo, unpack_hi, 0x20);
                    let prod_hi = _mm256_permute2x128_si256(unpack_lo, unpack_hi, 0x31);

                    v_scores_lo[q] = _mm256_add_epi64(v_scores_lo[q], prod_lo);
                    v_scores_hi[q] = _mm256_add_epi64(v_scores_hi[q], prod_hi);
                }
            }

            for q in 0..m {
                _mm256_storeu_si256(
                    queries_scores[q].as_mut_ptr().add(lane) as *mut __m256i,
                    v_scores_lo[q],
                );
                _mm256_storeu_si256(
                    queries_scores[q].as_mut_ptr().add(lane + 4) as *mut __m256i,
                    v_scores_hi[q],
                );
            }
        }
        lane += 8;
    }

    for q in 0..m {
        add_scalar_coord_tail(
            layout,
            block,
            lane,
            lanes,
            queries_coords[q],
            weights.coord,
            queries_scores[q],
        );
        add_scalar_sketch_tail(
            layout,
            block,
            lane,
            lanes,
            queries_sketches[q],
            weights.sketch,
            queries_scores[q],
        );
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
unsafe fn x86_avx512_scan_block_multi(
    layout: &BlockSoaLayout,
    block: usize,
    lanes: usize,
    queries_coords: &[&[i16]],
    queries_residual: &[u16],
    queries_sketches: &[&[i8]],
    weights: ScanWeights<'_>,
    queries_scores: &mut [&mut [i64]],
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let m = queries_coords.len();
    let vector_lanes = (lanes / 8) * 8;
    if m == 8 {
        if vector_lanes < lanes {
            let res_block = layout.residual_block(block);
            for q in 0..8 {
                residual_scores(
                    &res_block[vector_lanes..lanes],
                    queries_residual[q],
                    weights.residual,
                    &mut queries_scores[q][vector_lanes..lanes],
                );
            }
        }
    } else if m == 4 {
        if vector_lanes < lanes {
            let res_block = layout.residual_block(block);
            for q in 0..4 {
                residual_scores(
                    &res_block[vector_lanes..lanes],
                    queries_residual[q],
                    weights.residual,
                    &mut queries_scores[q][vector_lanes..lanes],
                );
            }
        }
    } else {
        for q in 0..m {
            residual_scores(
                &layout.residual_block(block)[..lanes],
                queries_residual[q],
                weights.residual,
                queries_scores[q],
            );
        }
    }

    let mut w_coord_v = [_mm512_setzero_si512(); 8];
    let mut q_coord_v = [[_mm512_setzero_si512(); 8]; 8];
    if weights.coord.len() == 8 {
        for (d, w) in w_coord_v.iter_mut().enumerate() {
            *w = _mm512_set1_epi64(weights.coord[d]);
        }
        for q in 0..m {
            for d in 0..8 {
                q_coord_v[q][d] = _mm512_set1_epi64(i64::from(queries_coords[q][d]));
            }
        }
    }

    let mut w_sketch_v = [_mm512_setzero_si512(); 4];
    let mut q_sketch_v = [[_mm512_setzero_si512(); 4]; 8];
    if weights.sketch.len() == 4 {
        for (d, w) in w_sketch_v.iter_mut().enumerate() {
            *w = _mm512_set1_epi64(weights.sketch[d]);
        }
        for q in 0..m {
            for d in 0..4 {
                q_sketch_v[q][d] = _mm512_set1_epi64(i64::from(queries_sketches[q][d]));
            }
        }
    }

    let mut lane = 0;
    while lane + 8 <= lanes {
        if m == 8 && weights.coord.len() == 8 && weights.sketch.len() == 4 {
            let residual_ptr = layout.residual_block(block).as_ptr();
            let raw_res = _mm_loadu_si128(residual_ptr.add(lane) as *const __m128i);
            let res_64 = _mm512_cvtepu16_epi64(raw_res);
            let v_res_weight = _mm512_set1_epi64(weights.residual);

            let mut v_scores_0 = _mm512_mul_epu32(
                _mm512_add_epi64(_mm512_set1_epi64(queries_residual[0] as i64), res_64),
                v_res_weight,
            );
            let mut v_scores_1 = _mm512_mul_epu32(
                _mm512_add_epi64(_mm512_set1_epi64(queries_residual[1] as i64), res_64),
                v_res_weight,
            );
            let mut v_scores_2 = _mm512_mul_epu32(
                _mm512_add_epi64(_mm512_set1_epi64(queries_residual[2] as i64), res_64),
                v_res_weight,
            );
            let mut v_scores_3 = _mm512_mul_epu32(
                _mm512_add_epi64(_mm512_set1_epi64(queries_residual[3] as i64), res_64),
                v_res_weight,
            );
            let mut v_scores_4 = _mm512_mul_epu32(
                _mm512_add_epi64(_mm512_set1_epi64(queries_residual[4] as i64), res_64),
                v_res_weight,
            );
            let mut v_scores_5 = _mm512_mul_epu32(
                _mm512_add_epi64(_mm512_set1_epi64(queries_residual[5] as i64), res_64),
                v_res_weight,
            );
            let mut v_scores_6 = _mm512_mul_epu32(
                _mm512_add_epi64(_mm512_set1_epi64(queries_residual[6] as i64), res_64),
                v_res_weight,
            );
            let mut v_scores_7 = _mm512_mul_epu32(
                _mm512_add_epi64(_mm512_set1_epi64(queries_residual[7] as i64), res_64),
                v_res_weight,
            );

            macro_rules! step_coord {
                ($d:expr) => {
                    let coord_ptr =
                        layout.coord_block(block, $d).as_ptr().add(lane) as *const __m128i;
                    let raw = _mm_loadu_si128(coord_ptr);
                    let widened = _mm512_cvtepi16_epi64(raw);
                    let v_weight = w_coord_v[$d];

                    let diff0 = _mm512_sub_epi64(q_coord_v[0][$d], widened);
                    let diff_sq0 = _mm512_mul_epu32(diff0, diff0);
                    let prod0 = if weights.coord[$d] == 1 {
                        diff_sq0
                    } else {
                        _mm512_mul_epu32(diff_sq0, v_weight)
                    };
                    v_scores_0 = _mm512_add_epi64(v_scores_0, prod0);

                    let diff1 = _mm512_sub_epi64(q_coord_v[1][$d], widened);
                    let diff_sq1 = _mm512_mul_epu32(diff1, diff1);
                    let prod1 = if weights.coord[$d] == 1 {
                        diff_sq1
                    } else {
                        _mm512_mul_epu32(diff_sq1, v_weight)
                    };
                    v_scores_1 = _mm512_add_epi64(v_scores_1, prod1);

                    let diff2 = _mm512_sub_epi64(q_coord_v[2][$d], widened);
                    let diff_sq2 = _mm512_mul_epu32(diff2, diff2);
                    let prod2 = if weights.coord[$d] == 1 {
                        diff_sq2
                    } else {
                        _mm512_mul_epu32(diff_sq2, v_weight)
                    };
                    v_scores_2 = _mm512_add_epi64(v_scores_2, prod2);

                    let diff3 = _mm512_sub_epi64(q_coord_v[3][$d], widened);
                    let diff_sq3 = _mm512_mul_epu32(diff3, diff3);
                    let prod3 = if weights.coord[$d] == 1 {
                        diff_sq3
                    } else {
                        _mm512_mul_epu32(diff_sq3, v_weight)
                    };
                    v_scores_3 = _mm512_add_epi64(v_scores_3, prod3);

                    let diff4 = _mm512_sub_epi64(q_coord_v[4][$d], widened);
                    let diff_sq4 = _mm512_mul_epu32(diff4, diff4);
                    let prod4 = if weights.coord[$d] == 1 {
                        diff_sq4
                    } else {
                        _mm512_mul_epu32(diff_sq4, v_weight)
                    };
                    v_scores_4 = _mm512_add_epi64(v_scores_4, prod4);

                    let diff5 = _mm512_sub_epi64(q_coord_v[5][$d], widened);
                    let diff_sq5 = _mm512_mul_epu32(diff5, diff5);
                    let prod5 = if weights.coord[$d] == 1 {
                        diff_sq5
                    } else {
                        _mm512_mul_epu32(diff_sq5, v_weight)
                    };
                    v_scores_5 = _mm512_add_epi64(v_scores_5, prod5);

                    let diff6 = _mm512_sub_epi64(q_coord_v[6][$d], widened);
                    let diff_sq6 = _mm512_mul_epu32(diff6, diff6);
                    let prod6 = if weights.coord[$d] == 1 {
                        diff_sq6
                    } else {
                        _mm512_mul_epu32(diff_sq6, v_weight)
                    };
                    v_scores_6 = _mm512_add_epi64(v_scores_6, prod6);

                    let diff7 = _mm512_sub_epi64(q_coord_v[7][$d], widened);
                    let diff_sq7 = _mm512_mul_epu32(diff7, diff7);
                    let prod7 = if weights.coord[$d] == 1 {
                        diff_sq7
                    } else {
                        _mm512_mul_epu32(diff_sq7, v_weight)
                    };
                    v_scores_7 = _mm512_add_epi64(v_scores_7, prod7);
                };
            }

            step_coord!(0);
            step_coord!(1);
            step_coord!(2);
            step_coord!(3);
            step_coord!(4);
            step_coord!(5);
            step_coord!(6);
            step_coord!(7);

            macro_rules! step_sketch {
                ($d:expr) => {
                    let sketch_ptr =
                        layout.sketch_block(block, $d).as_ptr().add(lane) as *const i64;
                    let packed = std::ptr::read_unaligned(sketch_ptr);
                    let raw = _mm_cvtsi64_si128(packed);
                    let widened = _mm512_cvtepi8_epi64(raw);
                    let v_weight = w_sketch_v[$d];

                    let diff0 = _mm512_sub_epi64(q_sketch_v[0][$d], widened);
                    let diff_sq0 = _mm512_mul_epu32(diff0, diff0);
                    let prod0 = if weights.sketch[$d] == 1 {
                        diff_sq0
                    } else {
                        _mm512_mul_epu32(diff_sq0, v_weight)
                    };
                    v_scores_0 = _mm512_add_epi64(v_scores_0, prod0);

                    let diff1 = _mm512_sub_epi64(q_sketch_v[1][$d], widened);
                    let diff_sq1 = _mm512_mul_epu32(diff1, diff1);
                    let prod1 = if weights.sketch[$d] == 1 {
                        diff_sq1
                    } else {
                        _mm512_mul_epu32(diff_sq1, v_weight)
                    };
                    v_scores_1 = _mm512_add_epi64(v_scores_1, prod1);

                    let diff2 = _mm512_sub_epi64(q_sketch_v[2][$d], widened);
                    let diff_sq2 = _mm512_mul_epu32(diff2, diff2);
                    let prod2 = if weights.sketch[$d] == 1 {
                        diff_sq2
                    } else {
                        _mm512_mul_epu32(diff_sq2, v_weight)
                    };
                    v_scores_2 = _mm512_add_epi64(v_scores_2, prod2);

                    let diff3 = _mm512_sub_epi64(q_sketch_v[3][$d], widened);
                    let diff_sq3 = _mm512_mul_epu32(diff3, diff3);
                    let prod3 = if weights.sketch[$d] == 1 {
                        diff_sq3
                    } else {
                        _mm512_mul_epu32(diff_sq3, v_weight)
                    };
                    v_scores_3 = _mm512_add_epi64(v_scores_3, prod3);

                    let diff4 = _mm512_sub_epi64(q_sketch_v[4][$d], widened);
                    let diff_sq4 = _mm512_mul_epu32(diff4, diff4);
                    let prod4 = if weights.sketch[$d] == 1 {
                        diff_sq4
                    } else {
                        _mm512_mul_epu32(diff_sq4, v_weight)
                    };
                    v_scores_4 = _mm512_add_epi64(v_scores_4, prod4);

                    let diff5 = _mm512_sub_epi64(q_sketch_v[5][$d], widened);
                    let diff_sq5 = _mm512_mul_epu32(diff5, diff5);
                    let prod5 = if weights.sketch[$d] == 1 {
                        diff_sq5
                    } else {
                        _mm512_mul_epu32(diff_sq5, v_weight)
                    };
                    v_scores_5 = _mm512_add_epi64(v_scores_5, prod5);

                    let diff6 = _mm512_sub_epi64(q_sketch_v[6][$d], widened);
                    let diff_sq6 = _mm512_mul_epu32(diff6, diff6);
                    let prod6 = if weights.sketch[$d] == 1 {
                        diff_sq6
                    } else {
                        _mm512_mul_epu32(diff_sq6, v_weight)
                    };
                    v_scores_6 = _mm512_add_epi64(v_scores_6, prod6);

                    let diff7 = _mm512_sub_epi64(q_sketch_v[7][$d], widened);
                    let diff_sq7 = _mm512_mul_epu32(diff7, diff7);
                    let prod7 = if weights.sketch[$d] == 1 {
                        diff_sq7
                    } else {
                        _mm512_mul_epu32(diff_sq7, v_weight)
                    };
                    v_scores_7 = _mm512_add_epi64(v_scores_7, prod7);
                };
            }

            step_sketch!(0);
            step_sketch!(1);
            step_sketch!(2);
            step_sketch!(3);

            _mm512_storeu_si512(
                queries_scores[0].as_mut_ptr().add(lane) as *mut _,
                v_scores_0,
            );
            _mm512_storeu_si512(
                queries_scores[1].as_mut_ptr().add(lane) as *mut _,
                v_scores_1,
            );
            _mm512_storeu_si512(
                queries_scores[2].as_mut_ptr().add(lane) as *mut _,
                v_scores_2,
            );
            _mm512_storeu_si512(
                queries_scores[3].as_mut_ptr().add(lane) as *mut _,
                v_scores_3,
            );
            _mm512_storeu_si512(
                queries_scores[4].as_mut_ptr().add(lane) as *mut _,
                v_scores_4,
            );
            _mm512_storeu_si512(
                queries_scores[5].as_mut_ptr().add(lane) as *mut _,
                v_scores_5,
            );
            _mm512_storeu_si512(
                queries_scores[6].as_mut_ptr().add(lane) as *mut _,
                v_scores_6,
            );
            _mm512_storeu_si512(
                queries_scores[7].as_mut_ptr().add(lane) as *mut _,
                v_scores_7,
            );
        } else if m == 4 && weights.coord.len() == 8 && weights.sketch.len() == 4 {
            let residual_ptr = layout.residual_block(block).as_ptr();
            let raw_res = _mm_loadu_si128(residual_ptr.add(lane) as *const __m128i);
            let res_64 = _mm512_cvtepu16_epi64(raw_res);
            let v_res_weight = _mm512_set1_epi64(weights.residual);

            // Q0 scores init
            let sum0 = _mm512_add_epi64(_mm512_set1_epi64(queries_residual[0] as i64), res_64);
            let mut v_scores_0 = _mm512_mul_epu32(sum0, v_res_weight);

            // Q1 scores init
            let sum1 = _mm512_add_epi64(_mm512_set1_epi64(queries_residual[1] as i64), res_64);
            let mut v_scores_1 = _mm512_mul_epu32(sum1, v_res_weight);

            // Q2 scores init
            let sum2 = _mm512_add_epi64(_mm512_set1_epi64(queries_residual[2] as i64), res_64);
            let mut v_scores_2 = _mm512_mul_epu32(sum2, v_res_weight);

            // Q3 scores init
            let sum3 = _mm512_add_epi64(_mm512_set1_epi64(queries_residual[3] as i64), res_64);
            let mut v_scores_3 = _mm512_mul_epu32(sum3, v_res_weight);

            macro_rules! step_coord {
                ($d:expr) => {
                    let coord_ptr =
                        layout.coord_block(block, $d).as_ptr().add(lane) as *const __m128i;
                    let raw = _mm_loadu_si128(coord_ptr);
                    let widened = _mm512_cvtepi16_epi64(raw);
                    let v_weight = w_coord_v[$d];

                    let diff0 = _mm512_sub_epi64(q_coord_v[0][$d], widened);
                    let diff_sq0 = _mm512_mul_epu32(diff0, diff0);
                    let prod0 = if weights.coord[$d] == 1 {
                        diff_sq0
                    } else {
                        _mm512_mul_epu32(diff_sq0, v_weight)
                    };
                    v_scores_0 = _mm512_add_epi64(v_scores_0, prod0);

                    let diff1 = _mm512_sub_epi64(q_coord_v[1][$d], widened);
                    let diff_sq1 = _mm512_mul_epu32(diff1, diff1);
                    let prod1 = if weights.coord[$d] == 1 {
                        diff_sq1
                    } else {
                        _mm512_mul_epu32(diff_sq1, v_weight)
                    };
                    v_scores_1 = _mm512_add_epi64(v_scores_1, prod1);

                    let diff2 = _mm512_sub_epi64(q_coord_v[2][$d], widened);
                    let diff_sq2 = _mm512_mul_epu32(diff2, diff2);
                    let prod2 = if weights.coord[$d] == 1 {
                        diff_sq2
                    } else {
                        _mm512_mul_epu32(diff_sq2, v_weight)
                    };
                    v_scores_2 = _mm512_add_epi64(v_scores_2, prod2);

                    let diff3 = _mm512_sub_epi64(q_coord_v[3][$d], widened);
                    let diff_sq3 = _mm512_mul_epu32(diff3, diff3);
                    let prod3 = if weights.coord[$d] == 1 {
                        diff_sq3
                    } else {
                        _mm512_mul_epu32(diff_sq3, v_weight)
                    };
                    v_scores_3 = _mm512_add_epi64(v_scores_3, prod3);
                };
            }

            step_coord!(0);
            step_coord!(1);
            step_coord!(2);
            step_coord!(3);
            step_coord!(4);
            step_coord!(5);
            step_coord!(6);
            step_coord!(7);

            macro_rules! step_sketch {
                ($d:expr) => {
                    let sketch_ptr =
                        layout.sketch_block(block, $d).as_ptr().add(lane) as *const i64;
                    let packed = std::ptr::read_unaligned(sketch_ptr);
                    let raw = _mm_cvtsi64_si128(packed);
                    let widened = _mm512_cvtepi8_epi64(raw);
                    let v_weight = w_sketch_v[$d];

                    let diff0 = _mm512_sub_epi64(q_sketch_v[0][$d], widened);
                    let diff_sq0 = _mm512_mul_epu32(diff0, diff0);
                    let prod0 = if weights.sketch[$d] == 1 {
                        diff_sq0
                    } else {
                        _mm512_mul_epu32(diff_sq0, v_weight)
                    };
                    v_scores_0 = _mm512_add_epi64(v_scores_0, prod0);

                    let diff1 = _mm512_sub_epi64(q_sketch_v[1][$d], widened);
                    let diff_sq1 = _mm512_mul_epu32(diff1, diff1);
                    let prod1 = if weights.sketch[$d] == 1 {
                        diff_sq1
                    } else {
                        _mm512_mul_epu32(diff_sq1, v_weight)
                    };
                    v_scores_1 = _mm512_add_epi64(v_scores_1, prod1);

                    let diff2 = _mm512_sub_epi64(q_sketch_v[2][$d], widened);
                    let diff_sq2 = _mm512_mul_epu32(diff2, diff2);
                    let prod2 = if weights.sketch[$d] == 1 {
                        diff_sq2
                    } else {
                        _mm512_mul_epu32(diff_sq2, v_weight)
                    };
                    v_scores_2 = _mm512_add_epi64(v_scores_2, prod2);

                    let diff3 = _mm512_sub_epi64(q_sketch_v[3][$d], widened);
                    let diff_sq3 = _mm512_mul_epu32(diff3, diff3);
                    let prod3 = if weights.sketch[$d] == 1 {
                        diff_sq3
                    } else {
                        _mm512_mul_epu32(diff_sq3, v_weight)
                    };
                    v_scores_3 = _mm512_add_epi64(v_scores_3, prod3);
                };
            }

            step_sketch!(0);
            step_sketch!(1);
            step_sketch!(2);
            step_sketch!(3);

            _mm512_storeu_si512(
                queries_scores[0].as_mut_ptr().add(lane) as *mut _,
                v_scores_0,
            );
            _mm512_storeu_si512(
                queries_scores[1].as_mut_ptr().add(lane) as *mut _,
                v_scores_1,
            );
            _mm512_storeu_si512(
                queries_scores[2].as_mut_ptr().add(lane) as *mut _,
                v_scores_2,
            );
            _mm512_storeu_si512(
                queries_scores[3].as_mut_ptr().add(lane) as *mut _,
                v_scores_3,
            );
        } else {
            let mut v_scores = [_mm512_setzero_si512(); 8];
            for q in 0..m {
                v_scores[q] = _mm512_loadu_si512(queries_scores[q].as_ptr().add(lane) as *const _);
            }

            for (dim, &weight) in weights.coord.iter().enumerate() {
                let coord_ptr = layout.coord_block(block, dim).as_ptr().add(lane) as *const __m128i;
                let raw = _mm_loadu_si128(coord_ptr);
                let widened = _mm512_cvtepi16_epi64(raw);
                let v_weight = w_coord_v[dim];

                for q in 0..m {
                    let diff = _mm512_sub_epi64(q_coord_v[q][dim], widened);
                    let diff_sq = _mm512_mul_epu32(diff, diff);
                    let prod = if weight == 1 {
                        diff_sq
                    } else {
                        _mm512_mul_epu32(diff_sq, v_weight)
                    };
                    v_scores[q] = _mm512_add_epi64(v_scores[q], prod);
                }
            }

            for (dim, &weight) in weights.sketch.iter().enumerate() {
                let sketch_ptr = layout.sketch_block(block, dim).as_ptr().add(lane) as *const i64;
                let packed = std::ptr::read_unaligned(sketch_ptr);
                let raw = _mm_cvtsi64_si128(packed);
                let widened = _mm512_cvtepi8_epi64(raw);
                let v_weight = w_sketch_v[dim];

                for q in 0..m {
                    let diff = _mm512_sub_epi64(q_sketch_v[q][dim], widened);
                    let diff_sq = _mm512_mul_epu32(diff, diff);
                    let prod = if weight == 1 {
                        diff_sq
                    } else {
                        _mm512_mul_epu32(diff_sq, v_weight)
                    };
                    v_scores[q] = _mm512_add_epi64(v_scores[q], prod);
                }
            }

            for q in 0..m {
                _mm512_storeu_si512(
                    queries_scores[q].as_mut_ptr().add(lane) as *mut _,
                    v_scores[q],
                );
            }
        }
        lane += 8;
    }

    for q in 0..m {
        add_scalar_coord_tail(
            layout,
            block,
            lane,
            lanes,
            queries_coords[q],
            weights.coord,
            queries_scores[q],
        );
        add_scalar_sketch_tail(
            layout,
            block,
            lane,
            lanes,
            queries_sketches[q],
            weights.sketch,
            queries_scores[q],
        );
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

    #[test]
    fn test_scan_block_pruning() {
        let mut layout = BlockSoaLayout::with_shape(3, 2, 8);
        for lane in 0..8 {
            layout
                .push_quantized(
                    VectorId::new(lane as u64),
                    &[lane as i16, lane as i16 * 2, 8 - lane as i16],
                    lane as u16 + 5,
                    &[lane as i8, 4 - lane as i8],
                )
                .unwrap();
        }
        let weights = ScanWeights {
            coord: &[2, 3, 5],
            residual: 7,
            sketch: &[11, 13],
        };

        // When threshold is very high (no pruning should happen)
        let mut scores_normal = vec![0; 8];
        let pruned = scan_block_pruned_into(
            &layout,
            0,
            8,
            &[4, 2, 1],
            9,
            &[3, -4],
            weights,
            &mut scores_normal,
            9999999, // high threshold
        );
        assert!(!pruned);

        // Verify matches scalar calculation
        let mut scores_ref = vec![0; 8];
        scalar_scan_block_into(
            &layout,
            0,
            8,
            &[4, 2, 1],
            9,
            &[3, -4],
            weights,
            &mut scores_ref,
        );
        assert_eq!(scores_normal, scores_ref);

        // When threshold is very low (e.g. 50, all lanes in reference scores are > 100)
        let mut scores_pruned = vec![0; 8];
        let pruned_active = scan_block_pruned_into(
            &layout,
            0,
            8,
            &[4, 2, 1],
            9,
            &[3, -4],
            weights,
            &mut scores_pruned,
            10, // very low threshold, will trigger early pruning
        );
        // Residual score alone is (9 + 5) * 7 = 98 for lane 0, so it will exceed 10 instantly!
        assert!(pruned_active);
    }

    #[test]
    fn test_scan_block_multi() {
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

        let q_coords = [
            vec![3i16, -7, 11, 13, -17],
            vec![1i16, 2, 3, 4, 5],
            vec![-5i16, 10, -15, 20, -25],
        ];
        let q_residuals = vec![19u16, 5, 23];
        let q_sketches = [vec![5i8, -3, 2], vec![1i8, 1, 1], vec![-2i8, 4, 0]];

        let q_coords_refs: Vec<&[i16]> = q_coords.iter().map(|v| v.as_slice()).collect();
        let q_sketches_refs: Vec<&[i8]> = q_sketches.iter().map(|v| v.as_slice()).collect();

        // Run single query scans for baseline
        let mut expected_scores = vec![vec![0i64; 23]; 3];
        for q in 0..3 {
            for block in 0..layout.block_count() {
                let lanes = layout.block_len(block);
                let start_idx = block * 16;
                let mut scratch = vec![0i64; lanes];
                scan_block_into(
                    &layout,
                    block,
                    lanes,
                    q_coords_refs[q],
                    q_residuals[q],
                    q_sketches_refs[q],
                    weights,
                    &mut scratch,
                );
                expected_scores[q][start_idx..start_idx + lanes].copy_from_slice(&scratch);
            }
        }

        // Run multi-query scans
        let mut actual_scores = vec![vec![0i64; 23]; 3];
        for block in 0..layout.block_count() {
            let lanes = layout.block_len(block);
            let start_idx = block * 16;
            let mut s0 = vec![0i64; lanes];
            let mut s1 = vec![0i64; lanes];
            let mut s2 = vec![0i64; lanes];
            {
                let mut scratch_refs = [&mut s0[..], &mut s1[..], &mut s2[..]];
                scan_block_multi_into(
                    &layout,
                    block,
                    lanes,
                    &q_coords_refs,
                    &q_residuals,
                    &q_sketches_refs,
                    weights,
                    &mut scratch_refs,
                );
            }
            actual_scores[0][start_idx..start_idx + lanes].copy_from_slice(&s0);
            actual_scores[1][start_idx..start_idx + lanes].copy_from_slice(&s1);
            actual_scores[2][start_idx..start_idx + lanes].copy_from_slice(&s2);
        }

        assert_eq!(actual_scores, expected_scores);
    }

    #[test]
    fn test_simd_scan_kernel_trait() {
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

        let q_coords = [
            vec![3i16, -7, 11, 13, -17],
            vec![1i16, 2, 3, 4, 5],
            vec![-5i16, 10, -15, 20, -25],
        ];
        let q_residuals = vec![19u16, 5, 23];
        let q_sketches = [vec![5i8, -3, 2], vec![1i8, 1, 1], vec![-2i8, 4, 0]];

        let q_coords_refs: Vec<&[i16]> = q_coords.iter().map(|v| v.as_slice()).collect();
        let q_sketches_refs: Vec<&[i8]> = q_sketches.iter().map(|v| v.as_slice()).collect();

        let kernel = get_optimal_scan_kernel();

        // Run multi-query scans using the trait dispatch
        let mut actual_scores = vec![vec![0i64; 23]; 3];
        for block in 0..layout.block_count() {
            let lanes = layout.block_len(block);
            let start_idx = block * 16;
            let mut s0 = vec![0i64; lanes];
            let mut s1 = vec![0i64; lanes];
            let mut s2 = vec![0i64; lanes];
            {
                let mut scratch_refs = [&mut s0[..], &mut s1[..], &mut s2[..]];
                kernel.scan_block_multi(
                    &layout,
                    block,
                    lanes,
                    &q_coords_refs,
                    &q_residuals,
                    &q_sketches_refs,
                    weights,
                    &mut scratch_refs,
                );
            }
            actual_scores[0][start_idx..start_idx + lanes].copy_from_slice(&s0);
            actual_scores[1][start_idx..start_idx + lanes].copy_from_slice(&s1);
            actual_scores[2][start_idx..start_idx + lanes].copy_from_slice(&s2);
        }

        // Run single query scan using the trait dispatch to verify parity
        let mut expected_scores = vec![vec![0i64; 23]; 3];
        for q in 0..3 {
            for block in 0..layout.block_count() {
                let lanes = layout.block_len(block);
                let start_idx = block * 16;
                let mut scratch = vec![0i64; lanes];
                kernel.scan_block(
                    &layout,
                    block,
                    lanes,
                    q_coords_refs[q],
                    q_residuals[q],
                    q_sketches_refs[q],
                    weights,
                    &mut scratch,
                );
                expected_scores[q][start_idx..start_idx + lanes].copy_from_slice(&scratch);
            }
        }

        assert_eq!(actual_scores, expected_scores);
    }

    #[test]
    #[ignore = "release-mode smoke benchmark for the T-217 multi-query scaling"]
    fn bench_scan_block_multi() {
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

        // Create 4 queries
        let mut q_coords = Vec::new();
        let mut q_residuals = Vec::new();
        let mut q_sketches = Vec::new();
        for i in 0..4 {
            q_coords.push(vec![i as i16, -7, 11, -13, 17, -19, 23, -29]);
            q_residuals.push(31u16);
            q_sketches.push(vec![5, -3, 2, -1]);
        }

        let q_coords_refs: Vec<&[i16]> = q_coords.iter().map(|v| v.as_slice()).collect();
        let q_sketches_refs: Vec<&[i8]> = q_sketches.iter().map(|v| v.as_slice()).collect();

        // 1. Benchmark Single Query (run 4 queries sequentially)
        let iterations = 20000;
        let mut scratch_seq = vec![0i64; block_size];
        let started_seq = Instant::now();
        let mut checksum_seq = 0i64;
        for _ in 0..iterations {
            for q in 0..4 {
                for block in 0..layout.block_count() {
                    let lanes = layout.block_len(block);
                    scan_block_into(
                        black_box(&layout),
                        block,
                        lanes,
                        black_box(q_coords_refs[q]),
                        q_residuals[q],
                        black_box(q_sketches_refs[q]),
                        weights,
                        &mut scratch_seq[..],
                    );
                    checksum_seq ^= scratch_seq[0];
                }
            }
        }
        let elapsed_seq = started_seq.elapsed();
        black_box(checksum_seq);

        // 2. Benchmark Multi-Query Batch (run 4 queries in a single batched call)
        let mut s0 = vec![0i64; block_size];
        let mut s1 = vec![0i64; block_size];
        let mut s2 = vec![0i64; block_size];
        let mut s3 = vec![0i64; block_size];
        let mut scratch_refs = [&mut s0[..], &mut s1[..], &mut s2[..], &mut s3[..]];
        let started_multi = Instant::now();
        let mut checksum_multi = 0i64;
        for _ in 0..iterations {
            for block in 0..layout.block_count() {
                let lanes = layout.block_len(block);
                scan_block_multi_into(
                    black_box(&layout),
                    block,
                    lanes,
                    black_box(&q_coords_refs),
                    black_box(&q_residuals),
                    black_box(&q_sketches_refs),
                    weights,
                    &mut scratch_refs,
                );
                checksum_multi ^= scratch_refs[0][0];
            }
        }
        let elapsed_multi = started_multi.elapsed();
        black_box(checksum_multi);

        println!("Sequential 4 queries took: {:?}", elapsed_seq);
        println!("Multi-query batch of 4 took: {:?}", elapsed_multi);
        let speedup = elapsed_seq.as_secs_f64() / elapsed_multi.as_secs_f64();
        println!("Multi-query speedup factor: {:.2}x", speedup);
        assert!(
            speedup > 1.5,
            "Multi-query speedup is too low: {:.2}x",
            speedup
        );
    }
}
