/// Distance type used by Aperon scoring paths.
pub type Distance = f32;

/// Computes squared L2 distance between two equally sized vectors.
pub fn l2_squared(lhs: &[f32], rhs: &[f32]) -> Option<Distance> {
    if lhs.len() != rhs.len() {
        return None;
    }

    Some(l2_squared_unchecked(lhs, rhs))
}

pub fn l2_squared_unchecked(lhs: &[f32], rhs: &[f32]) -> Distance {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: Runtime feature detection above guarantees AVX2/FMA support.
            return unsafe { x86_avx2_l2_squared(lhs, rhs) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: AArch64 guarantees NEON availability.
        unsafe { aarch64_neon_l2_squared(lhs, rhs) }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_l2_squared(lhs, rhs)
    }
}

fn scalar_l2_squared(lhs: &[f32], rhs: &[f32]) -> Distance {
    lhs.iter()
        .zip(rhs)
        .map(|(a, b)| {
            let delta = a - b;
            delta * delta
        })
        .sum()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn x86_avx2_l2_squared(lhs: &[f32], rhs: &[f32]) -> Distance {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut i = 0;
    let mut sum = _mm256_setzero_ps();
    while i + 8 <= lhs.len() {
        let a = _mm256_loadu_ps(lhs.as_ptr().add(i));
        let b = _mm256_loadu_ps(rhs.as_ptr().add(i));
        let diff = _mm256_sub_ps(a, b);
        sum = _mm256_fmadd_ps(diff, diff, sum);
        i += 8;
    }

    let mut lanes = [0.0_f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), sum);
    lanes.into_iter().sum::<f32>() + scalar_l2_squared(&lhs[i..], &rhs[i..])
}

#[cfg(target_arch = "aarch64")]
unsafe fn aarch64_neon_l2_squared(lhs: &[f32], rhs: &[f32]) -> Distance {
    use std::arch::aarch64::*;

    let mut i = 0;
    let mut sum = vdupq_n_f32(0.0);
    while i + 4 <= lhs.len() {
        let a = vld1q_f32(lhs.as_ptr().add(i));
        let b = vld1q_f32(rhs.as_ptr().add(i));
        let diff = vsubq_f32(a, b);
        sum = vfmaq_f32(sum, diff, diff);
        i += 4;
    }

    vaddvq_f32(sum) + scalar_l2_squared(&lhs[i..], &rhs[i..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_dimension_mismatch() {
        assert_eq!(l2_squared(&[1.0], &[1.0, 2.0]), None);
    }

    #[test]
    fn computes_squared_l2() {
        assert_eq!(l2_squared(&[1.0, 3.0], &[4.0, -1.0]), Some(25.0));
    }

    #[test]
    fn simd_dispatch_matches_scalar() {
        let lhs = (0..33).map(|x| x as f32 * 0.5).collect::<Vec<_>>();
        let rhs = (0..33).map(|x| x as f32 * -0.25).collect::<Vec<_>>();
        assert_eq!(l2_squared(&lhs, &rhs), Some(scalar_l2_squared(&lhs, &rhs)));
    }
}
