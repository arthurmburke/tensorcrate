//! x86_64 SIMD kernels.
//!
//! Public entry points perform runtime selection once per operation. AVX2 with
//! FMA is used when available; SSE2 is the x86_64 baseline and keeps the SIMD
//! feature useful on older processors.

#![allow(unsafe_op_in_unsafe_fn)]

macro_rules! x86_kernels {
    (
        mod $modname:ident, ty = $t:ty,
        avx_vec = $avx_vec:ty, avx_lanes = $avx_lanes:expr,
        avx_load = $avx_load:ident, avx_store = $avx_store:ident,
        avx_set1 = $avx_set1:ident, avx_zero = $avx_zero:ident,
        avx_add = $avx_add:ident, avx_sub = $avx_sub:ident,
        avx_mul = $avx_mul:ident, avx_div = $avx_div:ident,
        avx_fma = $avx_fma:ident,
        sse_vec = $sse_vec:ty, sse_lanes = $sse_lanes:expr,
        sse_load = $sse_load:ident, sse_store = $sse_store:ident,
        sse_set1 = $sse_set1:ident, sse_zero = $sse_zero:ident,
        sse_add = $sse_add:ident, sse_sub = $sse_sub:ident,
        sse_mul = $sse_mul:ident, sse_div = $sse_div:ident
    ) => {
        pub mod $modname {
            use core::arch::x86_64::*;
            use std::arch::is_x86_feature_detected;

            use crate::tensors::BinaryOp;

            /// Lanes used by the preferred AVX2 implementation.
            pub const LANES: usize = $avx_lanes;

            #[inline]
            fn has_avx2_fma() -> bool {
                is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
            }

            #[inline]
            fn has_avx2() -> bool {
                is_x86_feature_detected!("avx2")
            }

            #[inline]
            pub fn dot(a: &[$t], b: &[$t]) -> $t {
                debug_assert_eq!(a.len(), b.len());
                if has_avx2_fma() {
                    unsafe { dot_avx(a, b) }
                } else {
                    unsafe { dot_sse(a, b) }
                }
            }

            #[target_feature(enable = "avx2,fma")]
            unsafe fn dot_avx(a: &[$t], b: &[$t]) -> $t {
                let mut i = 0;
                let mut acc: [$avx_vec; 4] = [$avx_zero(); 4];
                while i + 4 * $avx_lanes <= a.len() {
                    for lane in 0..4 {
                        let offset = i + lane * $avx_lanes;
                        let va = $avx_load(a.as_ptr().add(offset));
                        let vb = $avx_load(b.as_ptr().add(offset));
                        acc[lane] = $avx_fma(va, vb, acc[lane]);
                    }
                    i += 4 * $avx_lanes;
                }
                let total = $avx_add($avx_add(acc[0], acc[1]), $avx_add(acc[2], acc[3]));
                let mut lanes = [0 as $t; $avx_lanes];
                $avx_store(lanes.as_mut_ptr(), total);
                let mut sum: $t = lanes.into_iter().sum();
                while i + $avx_lanes <= a.len() {
                    let product =
                        $avx_mul($avx_load(a.as_ptr().add(i)), $avx_load(b.as_ptr().add(i)));
                    $avx_store(lanes.as_mut_ptr(), product);
                    sum += lanes.into_iter().sum::<$t>();
                    i += $avx_lanes;
                }
                while i < a.len() {
                    sum += *a.get_unchecked(i) * *b.get_unchecked(i);
                    i += 1;
                }
                sum
            }

            #[target_feature(enable = "sse2")]
            unsafe fn dot_sse(a: &[$t], b: &[$t]) -> $t {
                let mut i = 0;
                let mut acc: [$sse_vec; 4] = [$sse_zero(); 4];
                while i + 4 * $sse_lanes <= a.len() {
                    for lane in 0..4 {
                        let offset = i + lane * $sse_lanes;
                        let product = $sse_mul(
                            $sse_load(a.as_ptr().add(offset)),
                            $sse_load(b.as_ptr().add(offset)),
                        );
                        acc[lane] = $sse_add(acc[lane], product);
                    }
                    i += 4 * $sse_lanes;
                }
                let total = $sse_add($sse_add(acc[0], acc[1]), $sse_add(acc[2], acc[3]));
                let mut lanes = [0 as $t; $sse_lanes];
                $sse_store(lanes.as_mut_ptr(), total);
                let mut sum: $t = lanes.into_iter().sum();
                while i + $sse_lanes <= a.len() {
                    let product =
                        $sse_mul($sse_load(a.as_ptr().add(i)), $sse_load(b.as_ptr().add(i)));
                    $sse_store(lanes.as_mut_ptr(), product);
                    sum += lanes.into_iter().sum::<$t>();
                    i += $sse_lanes;
                }
                while i < a.len() {
                    sum += *a.get_unchecked(i) * *b.get_unchecked(i);
                    i += 1;
                }
                sum
            }

            #[inline]
            pub fn elementwise(a: &[$t], b: &[$t], op: BinaryOp, out: &mut [$t]) {
                debug_assert!(a.len() == b.len() && a.len() == out.len());
                if op == BinaryOp::Rem {
                    for i in 0..a.len() {
                        out[i] = a[i] % b[i];
                    }
                } else if has_avx2() {
                    unsafe { elementwise_avx(a, b, op, out) };
                } else {
                    unsafe { elementwise_sse(a, b, op, out) };
                }
            }

            #[target_feature(enable = "avx2")]
            unsafe fn elementwise_avx(a: &[$t], b: &[$t], op: BinaryOp, out: &mut [$t]) {
                let mut i = 0;
                while i + $avx_lanes <= a.len() {
                    let va = $avx_load(a.as_ptr().add(i));
                    let vb = $avx_load(b.as_ptr().add(i));
                    let result = match op {
                        BinaryOp::Add => $avx_add(va, vb),
                        BinaryOp::Sub => $avx_sub(va, vb),
                        BinaryOp::Mul => $avx_mul(va, vb),
                        BinaryOp::Div => $avx_div(va, vb),
                        BinaryOp::Rem => unreachable!(),
                    };
                    $avx_store(out.as_mut_ptr().add(i), result);
                    i += $avx_lanes;
                }
                scalar_elementwise(a, b, op, out, i);
            }

            #[target_feature(enable = "sse2")]
            unsafe fn elementwise_sse(a: &[$t], b: &[$t], op: BinaryOp, out: &mut [$t]) {
                let mut i = 0;
                while i + $sse_lanes <= a.len() {
                    let va = $sse_load(a.as_ptr().add(i));
                    let vb = $sse_load(b.as_ptr().add(i));
                    let result = match op {
                        BinaryOp::Add => $sse_add(va, vb),
                        BinaryOp::Sub => $sse_sub(va, vb),
                        BinaryOp::Mul => $sse_mul(va, vb),
                        BinaryOp::Div => $sse_div(va, vb),
                        BinaryOp::Rem => unreachable!(),
                    };
                    $sse_store(out.as_mut_ptr().add(i), result);
                    i += $sse_lanes;
                }
                scalar_elementwise(a, b, op, out, i);
            }

            fn scalar_elementwise(a: &[$t], b: &[$t], op: BinaryOp, out: &mut [$t], start: usize) {
                for i in start..a.len() {
                    out[i] = match op {
                        BinaryOp::Add => a[i] + b[i],
                        BinaryOp::Sub => a[i] - b[i],
                        BinaryOp::Mul => a[i] * b[i],
                        BinaryOp::Div => a[i] / b[i],
                        BinaryOp::Rem => a[i] % b[i],
                    };
                }
            }

            #[inline]
            pub fn broadcast(
                values: &[$t],
                scalar: $t,
                op: BinaryOp,
                scalar_left: bool,
                out: &mut [$t],
            ) {
                debug_assert_eq!(values.len(), out.len());
                if op == BinaryOp::Rem {
                    scalar_broadcast(values, scalar, op, scalar_left, out, 0);
                } else if has_avx2() {
                    unsafe { broadcast_avx(values, scalar, op, scalar_left, out) };
                } else {
                    unsafe { broadcast_sse(values, scalar, op, scalar_left, out) };
                }
            }

            #[target_feature(enable = "avx2")]
            unsafe fn broadcast_avx(
                values: &[$t],
                scalar: $t,
                op: BinaryOp,
                scalar_left: bool,
                out: &mut [$t],
            ) {
                let scalar = $avx_set1(scalar);
                let mut i = 0;
                while i + $avx_lanes <= values.len() {
                    let value = $avx_load(values.as_ptr().add(i));
                    let (lhs, rhs) = if scalar_left {
                        (scalar, value)
                    } else {
                        (value, scalar)
                    };
                    let result = match op {
                        BinaryOp::Add => $avx_add(lhs, rhs),
                        BinaryOp::Sub => $avx_sub(lhs, rhs),
                        BinaryOp::Mul => $avx_mul(lhs, rhs),
                        BinaryOp::Div => $avx_div(lhs, rhs),
                        BinaryOp::Rem => unreachable!(),
                    };
                    $avx_store(out.as_mut_ptr().add(i), result);
                    i += $avx_lanes;
                }
                scalar_broadcast(values, scalar_value(scalar), op, scalar_left, out, i);
            }

            #[target_feature(enable = "sse2")]
            unsafe fn broadcast_sse(
                values: &[$t],
                scalar_value_: $t,
                op: BinaryOp,
                scalar_left: bool,
                out: &mut [$t],
            ) {
                let scalar = $sse_set1(scalar_value_);
                let mut i = 0;
                while i + $sse_lanes <= values.len() {
                    let value = $sse_load(values.as_ptr().add(i));
                    let (lhs, rhs) = if scalar_left {
                        (scalar, value)
                    } else {
                        (value, scalar)
                    };
                    let result = match op {
                        BinaryOp::Add => $sse_add(lhs, rhs),
                        BinaryOp::Sub => $sse_sub(lhs, rhs),
                        BinaryOp::Mul => $sse_mul(lhs, rhs),
                        BinaryOp::Div => $sse_div(lhs, rhs),
                        BinaryOp::Rem => unreachable!(),
                    };
                    $sse_store(out.as_mut_ptr().add(i), result);
                    i += $sse_lanes;
                }
                scalar_broadcast(values, scalar_value_, op, scalar_left, out, i);
            }

            #[target_feature(enable = "avx2")]
            unsafe fn scalar_value(value: $avx_vec) -> $t {
                let mut lanes = [0 as $t; $avx_lanes];
                $avx_store(lanes.as_mut_ptr(), value);
                lanes[0]
            }

            fn scalar_broadcast(
                values: &[$t],
                scalar: $t,
                op: BinaryOp,
                scalar_left: bool,
                out: &mut [$t],
                start: usize,
            ) {
                for i in start..values.len() {
                    let (lhs, rhs) = if scalar_left {
                        (scalar, values[i])
                    } else {
                        (values[i], scalar)
                    };
                    out[i] = match op {
                        BinaryOp::Add => lhs + rhs,
                        BinaryOp::Sub => lhs - rhs,
                        BinaryOp::Mul => lhs * rhs,
                        BinaryOp::Div => lhs / rhs,
                        BinaryOp::Rem => lhs % rhs,
                    };
                }
            }

            #[inline]
            pub fn matmul(a: &[$t], b: &[$t], m: usize, k: usize, n: usize, out: &mut [$t]) {
                out.fill(0 as $t);
                matmul_accumulate(a, b, m, k, n, out);
            }

            #[inline]
            pub fn matmul_add(
                a: &[$t],
                b: &[$t],
                addend: &[$t],
                m: usize,
                k: usize,
                n: usize,
                out: &mut [$t],
            ) {
                out.copy_from_slice(addend);
                matmul_accumulate(a, b, m, k, n, out);
            }

            #[inline]
            pub fn matmul_accumulate(
                a: &[$t],
                b: &[$t],
                m: usize,
                k: usize,
                n: usize,
                out: &mut [$t],
            ) {
                debug_assert!(a.len() == m * k && b.len() == k * n && out.len() == m * n);
                if has_avx2_fma() {
                    unsafe { matmul_avx(a, b, m, k, n, out) };
                } else {
                    unsafe { matmul_sse(a, b, m, k, n, out) };
                }
            }

            #[target_feature(enable = "avx2,fma")]
            unsafe fn matmul_avx(a: &[$t], b: &[$t], m: usize, k: usize, n: usize, out: &mut [$t]) {
                for i in 0..m {
                    let arow = i * k;
                    let orow = i * n;
                    for p in 0..k {
                        let value = *a.get_unchecked(arow + p);
                        let broadcast = $avx_set1(value);
                        let brow = p * n;
                        let mut j = 0;
                        while j + $avx_lanes <= n {
                            let rhs = $avx_load(b.as_ptr().add(brow + j));
                            let current = $avx_load(out.as_ptr().add(orow + j));
                            $avx_store(
                                out.as_mut_ptr().add(orow + j),
                                $avx_fma(broadcast, rhs, current),
                            );
                            j += $avx_lanes;
                        }
                        while j < n {
                            *out.get_unchecked_mut(orow + j) += value * *b.get_unchecked(brow + j);
                            j += 1;
                        }
                    }
                }
            }

            #[target_feature(enable = "sse2")]
            unsafe fn matmul_sse(a: &[$t], b: &[$t], m: usize, k: usize, n: usize, out: &mut [$t]) {
                for i in 0..m {
                    let arow = i * k;
                    let orow = i * n;
                    for p in 0..k {
                        let value = *a.get_unchecked(arow + p);
                        let broadcast = $sse_set1(value);
                        let brow = p * n;
                        let mut j = 0;
                        while j + $sse_lanes <= n {
                            let rhs = $sse_load(b.as_ptr().add(brow + j));
                            let current = $sse_load(out.as_ptr().add(orow + j));
                            $sse_store(
                                out.as_mut_ptr().add(orow + j),
                                $sse_add(current, $sse_mul(broadcast, rhs)),
                            );
                            j += $sse_lanes;
                        }
                        while j < n {
                            *out.get_unchecked_mut(orow + j) += value * *b.get_unchecked(brow + j);
                            j += 1;
                        }
                    }
                }
            }

            #[cfg(test)]
            mod architecture_tests {
                use super::*;

                #[test]
                fn forced_instruction_paths_match_scalar() {
                    let a = (0..37)
                        .map(|i| i as $t * 0.125 as $t - 2.0 as $t)
                        .collect::<Vec<_>>();
                    let b = (0..37)
                        .map(|i| (i % 9) as $t * 0.2 as $t + 0.5 as $t)
                        .collect::<Vec<_>>();
                    let expected: $t = a.iter().zip(&b).map(|(x, y)| *x * *y).sum();
                    let tolerance = 1e-4 as $t * (1.0 as $t + expected.abs());

                    let sse = unsafe { dot_sse(&a, &b) };
                    assert!((sse - expected).abs() <= tolerance);

                    if has_avx2_fma() {
                        let avx = unsafe { dot_avx(&a, &b) };
                        assert!((avx - expected).abs() <= tolerance);
                    }
                }
            }
        }
    };
}

x86_kernels! {
    mod f32k, ty = f32,
    avx_vec = __m256, avx_lanes = 8,
    avx_load = _mm256_loadu_ps, avx_store = _mm256_storeu_ps,
    avx_set1 = _mm256_set1_ps, avx_zero = _mm256_setzero_ps,
    avx_add = _mm256_add_ps, avx_sub = _mm256_sub_ps,
    avx_mul = _mm256_mul_ps, avx_div = _mm256_div_ps,
    avx_fma = _mm256_fmadd_ps,
    sse_vec = __m128, sse_lanes = 4,
    sse_load = _mm_loadu_ps, sse_store = _mm_storeu_ps,
    sse_set1 = _mm_set1_ps, sse_zero = _mm_setzero_ps,
    sse_add = _mm_add_ps, sse_sub = _mm_sub_ps,
    sse_mul = _mm_mul_ps, sse_div = _mm_div_ps
}

x86_kernels! {
    mod f64k, ty = f64,
    avx_vec = __m256d, avx_lanes = 4,
    avx_load = _mm256_loadu_pd, avx_store = _mm256_storeu_pd,
    avx_set1 = _mm256_set1_pd, avx_zero = _mm256_setzero_pd,
    avx_add = _mm256_add_pd, avx_sub = _mm256_sub_pd,
    avx_mul = _mm256_mul_pd, avx_div = _mm256_div_pd,
    avx_fma = _mm256_fmadd_pd,
    sse_vec = __m128d, sse_lanes = 2,
    sse_load = _mm_loadu_pd, sse_store = _mm_storeu_pd,
    sse_set1 = _mm_set1_pd, sse_zero = _mm_setzero_pd,
    sse_add = _mm_add_pd, sse_sub = _mm_sub_pd,
    sse_mul = _mm_mul_pd, sse_div = _mm_div_pd
}

pub mod fft_f32 {
    use core::arch::x86_64::*;
    use std::arch::is_x86_feature_detected;

    /// Radix-2 FFT over an interleaved complex buffer.
    pub fn radix2(buf: &mut [f32], n: usize, direction: f32) {
        debug_assert_eq!(buf.len(), 2 * n);
        if n <= 1 {
            return;
        }

        let mut reversed = 0usize;
        for i in 1..n {
            let mut bit = n >> 1;
            while reversed & bit != 0 {
                reversed ^= bit;
                bit >>= 1;
            }
            reversed ^= bit;
            if i < reversed {
                buf.swap(2 * i, 2 * reversed);
                buf.swap(2 * i + 1, 2 * reversed + 1);
            }
        }

        let use_avx = is_x86_feature_detected!("avx");
        // Twiddle scratch for the widest stage, allocated once rather than once
        // per stage; see the aarch64 path for the same change.
        let mut twiddle_buf = vec![0.0f32; n];
        let mut len = 2usize;
        while len <= n {
            let half = len / 2;
            let angle = direction * std::f32::consts::TAU / len as f32;
            let step = (angle.cos(), angle.sin());
            let twiddles = &mut twiddle_buf[..2 * half];
            let (mut real, mut imaginary) = (1.0f32, 0.0f32);
            for offset in 0..half {
                twiddles[2 * offset] = real;
                twiddles[2 * offset + 1] = imaginary;
                (real, imaginary) = (
                    real * step.0 - imaginary * step.1,
                    real * step.1 + imaginary * step.0,
                );
            }

            for start in (0..n).step_by(len) {
                let mut offset = if use_avx {
                    unsafe { butterflies_avx(buf, &twiddles, start, half) }
                } else {
                    0
                };
                while offset < half {
                    let even = 2 * (start + offset);
                    let odd = 2 * (start + offset + half);
                    let (odd_real, odd_imaginary) = (buf[odd], buf[odd + 1]);
                    let twiddle_real = twiddles[2 * offset];
                    let twiddle_imaginary = twiddles[2 * offset + 1];
                    let product_real = odd_real * twiddle_real - odd_imaginary * twiddle_imaginary;
                    let product_imaginary =
                        odd_real * twiddle_imaginary + odd_imaginary * twiddle_real;
                    let (even_real, even_imaginary) = (buf[even], buf[even + 1]);
                    buf[even] = even_real + product_real;
                    buf[even + 1] = even_imaginary + product_imaginary;
                    buf[odd] = even_real - product_real;
                    buf[odd + 1] = even_imaginary - product_imaginary;
                    offset += 1;
                }
            }
            len *= 2;
        }
    }

    #[target_feature(enable = "avx")]
    unsafe fn butterflies_avx(
        buf: &mut [f32],
        twiddles: &[f32],
        start: usize,
        half: usize,
    ) -> usize {
        let mut offset = 0;
        while offset + 4 <= half {
            let even_ptr = buf.as_ptr().add(2 * (start + offset));
            let odd_ptr = buf.as_ptr().add(2 * (start + offset + half));
            let even = _mm256_loadu_ps(even_ptr);
            let odd = _mm256_loadu_ps(odd_ptr);
            let twiddle = _mm256_loadu_ps(twiddles.as_ptr().add(2 * offset));

            let odd_real = _mm256_moveldup_ps(odd);
            let odd_imaginary = _mm256_movehdup_ps(odd);
            let twiddle_swapped = _mm256_permute_ps::<0b1011_0001>(twiddle);
            let product = _mm256_addsub_ps(
                _mm256_mul_ps(odd_real, twiddle),
                _mm256_mul_ps(odd_imaginary, twiddle_swapped),
            );

            _mm256_storeu_ps(
                buf.as_mut_ptr().add(2 * (start + offset)),
                _mm256_add_ps(even, product),
            );
            _mm256_storeu_ps(
                buf.as_mut_ptr().add(2 * (start + offset + half)),
                _mm256_sub_ps(even, product),
            );
            offset += 4;
        }
        offset
    }
}
