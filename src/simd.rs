//! NEON SIMD kernels for the CPU math paths.
//!
//! These are the *middle tier* between the naive generic scalar loops in
//! [`tensors`](crate::tensors) and the Metal GPU path. Metal only pays off once
//! a problem is large enough to amortize command-buffer setup (see the
//! `MIN_*` thresholds in `tensors`); below those sizes — which is most real
//! matmuls and every short FFT — the generic path runs one scalar multiply at a
//! time. These kernels vectorize that gap.
//!
//! Everything here is `aarch64`-only and hand-written against
//! [`core::arch::aarch64`]. NEON is part of the aarch64 baseline (it is always
//! present in `target_feature`), so no runtime feature detection is needed. The
//! kernels operate on concrete `f32`/`f64` slices; the generic-to-concrete
//! bridge (via `TypeId`) lives in `tensors::simd_dispatch`, which keeps the
//! scalar path as the correctness oracle for every non-float element type
//! (integers, `Complex`, `Dual`).

#![cfg(target_arch = "aarch64")]

/// Generates the elementwise / reduction / matmul kernels for one float type.
///
/// The four floating intrinsics differ only in name between `f32` (4-lane
/// `float32x4_t`) and `f64` (2-lane `float64x2_t`), so the bodies are shared and
/// the intrinsic set is passed in.
macro_rules! neon_kernels {
    (
        mod $modname:ident, ty = $t:ty, vec = $v:ty, lanes = $lanes:expr,
        load = $load:ident, store = $store:ident, dup = $dup:ident,
        add = $add:ident, sub = $sub:ident, mul = $mul:ident, div = $div:ident,
        fma = $fma:ident, addv = $addv:ident
    ) => {
        pub mod $modname {
            use core::arch::aarch64::*;

            /// Lanes per NEON register for this element type.
            pub const LANES: usize = $lanes;

            /// Dot product of two equal-length slices.
            ///
            /// Four independent accumulators keep several FMAs in flight so the
            /// (multi-cycle) FMA latency is hidden, then a single horizontal add
            /// combines them. The naive scalar reduction cannot do this because
            /// its `sum` carries a serial dependency across every element.
            #[inline]
            pub fn dot(a: &[$t], b: &[$t]) -> $t {
                debug_assert_eq!(a.len(), b.len());
                let n = a.len();
                let mut i = 0;
                let mut total: $t;
                unsafe {
                    let mut acc = [$dup(0 as $t); 4];
                    while i + 4 * LANES <= n {
                        let mut k = 0;
                        while k < 4 {
                            let off = i + k * LANES;
                            let va = $load(a.as_ptr().add(off));
                            let vb = $load(b.as_ptr().add(off));
                            acc[k] = $fma(acc[k], va, vb);
                            k += 1;
                        }
                        i += 4 * LANES;
                    }
                    let partial = $add($add(acc[0], acc[1]), $add(acc[2], acc[3]));
                    total = $addv(partial);
                    while i + LANES <= n {
                        let va = $load(a.as_ptr().add(i));
                        let vb = $load(b.as_ptr().add(i));
                        total += $addv($mul(va, vb));
                        i += LANES;
                    }
                }
                while i < n {
                    total += a[i] * b[i];
                    i += 1;
                }
                total
            }

            /// Elementwise binary op. `op` matches the encoding used by
            /// `broadcast_right`/`elementwise` in `tensors`:
            /// 0 = add, 1 = sub, 2 = mul, 3 = div, 4 = rem.
            ///
            /// Remainder has no NEON floating-point instruction, so op 4 runs the
            /// scalar loop; the caller gates it out before reaching SIMD anyway.
            #[inline]
            pub fn elementwise(a: &[$t], b: &[$t], op: u32, out: &mut [$t]) {
                let n = a.len();
                debug_assert!(b.len() == n && out.len() == n);
                if op == 4 {
                    for i in 0..n {
                        out[i] = a[i] % b[i];
                    }
                    return;
                }
                let mut i = 0;
                unsafe {
                    while i + LANES <= n {
                        let va = $load(a.as_ptr().add(i));
                        let vb = $load(b.as_ptr().add(i));
                        let vr = match op {
                            0 => $add(va, vb),
                            1 => $sub(va, vb),
                            2 => $mul(va, vb),
                            _ => $div(va, vb),
                        };
                        $store(out.as_mut_ptr().add(i), vr);
                        i += LANES;
                    }
                }
                while i < n {
                    out[i] = match op {
                        0 => a[i] + b[i],
                        1 => a[i] - b[i],
                        2 => a[i] * b[i],
                        _ => a[i] / b[i],
                    };
                    i += 1;
                }
            }

            /// Tensor/scalar broadcast. `scalar_left` selects operand order for
            /// the non-commutative ops (`scalar - x` vs `x - scalar`, etc.).
            #[inline]
            pub fn broadcast(values: &[$t], scalar: $t, op: u32, scalar_left: bool, out: &mut [$t]) {
                let n = values.len();
                debug_assert_eq!(out.len(), n);
                if op == 4 {
                    for i in 0..n {
                        out[i] = if scalar_left { scalar % values[i] } else { values[i] % scalar };
                    }
                    return;
                }
                let mut i = 0;
                unsafe {
                    let vs = $dup(scalar);
                    while i + LANES <= n {
                        let vx = $load(values.as_ptr().add(i));
                        let (lhs, rhs) = if scalar_left { (vs, vx) } else { (vx, vs) };
                        let vr = match op {
                            0 => $add(lhs, rhs),
                            1 => $sub(lhs, rhs),
                            2 => $mul(lhs, rhs),
                            _ => $div(lhs, rhs),
                        };
                        $store(out.as_mut_ptr().add(i), vr);
                        i += LANES;
                    }
                }
                while i < n {
                    let x = values[i];
                    let (l, r) = if scalar_left { (scalar, x) } else { (x, scalar) };
                    out[i] = match op {
                        0 => l + r,
                        1 => l - r,
                        2 => l * r,
                        _ => l / r,
                    };
                    i += 1;
                }
            }

            /// Row-major matrix multiply: `a` is `m×k`, `b` is `k×n`, `out` is
            /// `m×n` (must be pre-sized; it is overwritten).
            ///
            /// The loop order is `i, p, j` with `j` innermost — the *broadcast-A,
            /// stream-B-rows* microkernel. `a[i][p]` is splat across a register
            /// and fused-multiply-added into a contiguous run of the output row
            /// using a contiguous run of `b`'s `p`-th row. Both `b` and `out` are
            /// walked with unit stride, which is what makes the vectorization (and
            /// the cache behaviour) pay off; the textbook `i, j, p` order strides
            /// down a column of `b` and defeats both.
            #[inline]
            pub fn matmul(a: &[$t], b: &[$t], m: usize, k: usize, n: usize, out: &mut [$t]) {
                debug_assert!(a.len() == m * k && b.len() == k * n && out.len() == m * n);
                for v in out.iter_mut() {
                    *v = 0 as $t;
                }
                unsafe {
                    for i in 0..m {
                        let arow = i * k;
                        let orow = i * n;
                        for p in 0..k {
                            let aip = *a.get_unchecked(arow + p);
                            let va = $dup(aip);
                            let brow = p * n;
                            let mut j = 0;
                            while j + LANES <= n {
                                let vb = $load(b.as_ptr().add(brow + j));
                                let vo = $load(out.as_ptr().add(orow + j));
                                $store(out.as_mut_ptr().add(orow + j), $fma(vo, va, vb));
                                j += LANES;
                            }
                            while j < n {
                                *out.get_unchecked_mut(orow + j) += aip * *b.get_unchecked(brow + j);
                                j += 1;
                            }
                        }
                    }
                }
            }
        }
    };
}

neon_kernels! {
    mod f32k, ty = f32, vec = float32x4_t, lanes = 4,
    load = vld1q_f32, store = vst1q_f32, dup = vdupq_n_f32,
    add = vaddq_f32, sub = vsubq_f32, mul = vmulq_f32, div = vdivq_f32,
    fma = vfmaq_f32, addv = vaddvq_f32
}

neon_kernels! {
    mod f64k, ty = f64, vec = float64x2_t, lanes = 2,
    load = vld1q_f64, store = vst1q_f64, dup = vdupq_n_f64,
    add = vaddq_f64, sub = vsubq_f64, mul = vmulq_f64, div = vdivq_f64,
    fma = vfmaq_f64, addv = vaddvq_f64
}

/// Radix-2 Cooley–Tukey FFT over an interleaved `[re, im, re, im, …]` buffer of
/// `n` complex numbers (`buf.len() == 2 * n`, `n` a power of two).
///
/// `f32`-only, mirroring the Metal path. The butterfly stages are vectorized
/// four complex points at a time with `vld2q`/`vst2q`, which deinterleave the
/// real and imaginary lanes in a single instruction — the layout NEON wants for
/// the complex multiply. Per-stage twiddles are precomputed once (as split
/// real/imag arrays) rather than iterated with a running complex product, so the
/// vector lanes stay independent. Stages too short to fill a vector (len 2 and 4)
/// and any ragged tail fall back to scalar butterflies.
pub mod fft_f32 {
    use core::arch::aarch64::*;

    /// `direction` is `-1.0` for the forward transform, `+1.0` for the inverse
    /// (matching `radix2_fft`'s sign convention; normalization is the caller's
    /// job, exactly as in the scalar path).
    pub fn radix2(buf: &mut [f32], n: usize, direction: f32) {
        debug_assert_eq!(buf.len(), 2 * n);
        if n <= 1 {
            return;
        }

        // Bit-reversal permutation on complex indices (swap interleaved pairs).
        let mut j = 0usize;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            if i < j {
                buf.swap(2 * i, 2 * j);
                buf.swap(2 * i + 1, 2 * j + 1);
            }
        }

        let tau = std::f32::consts::TAU;
        let mut len = 2usize;
        while len <= n {
            let half = len / 2;
            let angle = direction * tau / len as f32;
            let step = (angle.cos(), angle.sin());

            // Precompute this stage's twiddles: tw[o] = exp(i·angle·o).
            let mut tw_re = vec![0.0f32; half];
            let mut tw_im = vec![0.0f32; half];
            let (mut wr, mut wi) = (1.0f32, 0.0f32);
            for o in 0..half {
                tw_re[o] = wr;
                tw_im[o] = wi;
                let nr = wr * step.0 - wi * step.1;
                let ni = wr * step.1 + wi * step.0;
                wr = nr;
                wi = ni;
            }

            let mut start = 0;
            while start < n {
                let mut o = 0;
                unsafe {
                    while o + 4 <= half {
                        let eptr = buf.as_ptr().add(2 * (start + o));
                        let optr = buf.as_ptr().add(2 * (start + o + half));
                        let even = vld2q_f32(eptr); // .0 = re, .1 = im
                        let odd = vld2q_f32(optr);
                        let twr = vld1q_f32(tw_re.as_ptr().add(o));
                        let twi = vld1q_f32(tw_im.as_ptr().add(o));

                        // p = odd * tw  (complex)
                        let pr = vsubq_f32(vmulq_f32(odd.0, twr), vmulq_f32(odd.1, twi));
                        let pi = vaddq_f32(vmulq_f32(odd.0, twi), vmulq_f32(odd.1, twr));

                        let sum = float32x4x2_t(vaddq_f32(even.0, pr), vaddq_f32(even.1, pi));
                        let dif = float32x4x2_t(vsubq_f32(even.0, pr), vsubq_f32(even.1, pi));
                        vst2q_f32(buf.as_mut_ptr().add(2 * (start + o)), sum);
                        vst2q_f32(buf.as_mut_ptr().add(2 * (start + o + half)), dif);
                        o += 4;
                    }
                }
                // Scalar tail (and the whole of len 2 / len 4 stages).
                while o < half {
                    let ei = 2 * (start + o);
                    let oi = 2 * (start + o + half);
                    let (er, eim) = (buf[ei], buf[ei + 1]);
                    let (or_, oim) = (buf[oi], buf[oi + 1]);
                    let pr = or_ * tw_re[o] - oim * tw_im[o];
                    let pi = or_ * tw_im[o] + oim * tw_re[o];
                    buf[ei] = er + pr;
                    buf[ei + 1] = eim + pi;
                    buf[oi] = er - pr;
                    buf[oi + 1] = eim - pi;
                    o += 1;
                }
                start += len;
            }
            len *= 2;
        }
    }
}
