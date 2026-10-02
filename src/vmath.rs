//! The `f32` elementary functions, written so a whole slice vectorizes.
//!
//! Every [`Analytic`] function of an `f32` — and of an `f16` or `bf16`, which
//! compute in `f32` — is defined here, once, as a scalar function built only
//! from IEEE arithmetic, fused multiply-adds, square roots, comparisons and
//! bit manipulation. Each of those is exactly specified, and Rust never
//! contracts or reassociates floating-point arithmetic, so the same function
//! evaluated one element at a time or by LLVM's vectorizer gives the same bits.
//! That is what lets every host path agree exactly: the unfused kernels, the
//! fused interpreter, the closures `math!` generates and the scalar
//! [`numbers`](crate::numbers) traits all call these functions, and the slice
//! kernels below are loops over them that LLVM turns into SIMD.
//!
//! A fused multiply-add is one instruction on `aarch64` and on `x86_64` with
//! FMA; without it `mul_add` is a correctly rounded library call, slower but
//! with identical results.
//!
//! # Accuracy
//!
//! Within one ulp of the correctly rounded result for `exp`, `ln`, `sin`, `cos`,
//! `arccos`, `cosh` and `tanh`, two for the rest and three for `tan`, where the
//! platform `libm` these replace for `f32` is within one. The tests measure
//! each function against `f64` (`vmath::tests::sweep` over a dense sample of
//! every `f32` below 9000). In exchange a slice evaluates two to six times
//! faster than through `libm`.
//!
//! `sin`, `cos` and `tan` reduce by `π/2` in three `f32` parts, which is
//! accurate for `|x| < 8192`; beyond that, and for infinities and NaN, they
//! call `libm`, in a second pass over just those elements so the first pass
//! keeps no branch.

// The fdlibm and Cephes constants are quoted as published.
#![allow(clippy::excessive_precision)]

use std::any::TypeId;
use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

use half::{bf16, f16};

use crate::numbers::Real;
use crate::tensors::Analytic;

/// `1.5·2²³`: adding it to an `f32` below `2²²` in magnitude rounds to an
/// integer, which then sits in the low bits of the sum's representation.
const SHIFT: f32 = 12_582_912.0;

/// `y` rounded to the nearest integer, as an `f32` and as an `i32`.
#[inline(always)]
fn round(y: f32) -> (f32, i32) {
    let t = y + SHIFT;
    (t - SHIFT, t.to_bits() as i32 - SHIFT.to_bits() as i32)
}

/// `2ⁿ` for `n` in `[-126, 127]`.
#[inline(always)]
fn pow2(n: i32) -> f32 {
    f32::from_bits(((n + 127) as u32) << 23)
}

/// `p·2ⁿ` for `n` in `[-152, 130]`, rounded once: two exact steps, so a
/// subnormal result is not rounded twice.
#[inline(always)]
fn scale(p: f32, n: i32) -> f32 {
    let half = n >> 1;
    p * pow2(half) * pow2(n - half)
}

// ---- exp ---------------------------------------------------------------------------

const LOG2_E: f32 = std::f32::consts::LOG2_E;
/// `ln 2` in two parts (fdlibm's `expf` split).
const LN2_HI: f32 = 6.931_457_519_5e-1;
const LN2_LO: f32 = 1.428_606_765_3e-6;

/// `eˣ = p·2ⁿ` with `p` in about `[0.7, 1.42]`, for `x` already clamped to
/// `[-104, 90]`, which covers every finite non-zero `f32` result of `eˣ` and
/// of `eˣ/2`.
#[inline(always)]
fn exp_parts(x: f32) -> (f32, i32) {
    let (n, k) = round(x * LOG2_E);
    let r = (-n).mul_add(LN2_HI, x);
    let r = (-n).mul_add(LN2_LO, r);
    // Taylor series to r⁷; |r| ≤ ln2/2 bounds the omitted terms by 0.1 ulp.
    let p = 1.0 / 5_040.0f32;
    let p = p.mul_add(r, 1.0 / 720.0);
    let p = p.mul_add(r, 1.0 / 120.0);
    let p = p.mul_add(r, 1.0 / 24.0);
    let p = p.mul_add(r, 1.0 / 6.0);
    let p = p.mul_add(r, 0.5);
    let p = p.mul_add(r * r, r);
    (p + 1.0, k)
}

#[inline(always)]
fn clamp_exp(x: f32) -> f32 {
    x.clamp(-104.0, 90.0)
}

#[inline(always)]
pub(crate) fn exp(x: f32) -> f32 {
    let (p, n) = exp_parts(clamp_exp(x));
    let y = scale(p, n);
    if x.is_nan() { x } else { y }
}

// ---- ln ----------------------------------------------------------------------------

const LN2_HI_LOG: f32 = 6.931_381_225_6e-1;
const LN2_LO_LOG: f32 = 9.058_000_614_5e-6;

/// musl's `logf`: `ln x = k·ln 2 + ln(1 + f)`, with `1 + f` in `[√½, √2)`.
#[inline(always)]
pub(crate) fn ln(x: f32) -> f32 {
    // Subnormals scaled into the normal range first.
    let tiny = x < f32::MIN_POSITIVE;
    let scaled = if tiny { x * 33_554_432.0 } else { x };
    let k0: i32 = if tiny { -25 } else { 0 };
    let ix = scaled.to_bits().wrapping_add(0x3F80_0000 - 0x3F35_04F3);
    let k = k0 + (ix >> 23) as i32 - 0x7F;
    let m = f32::from_bits((ix & 0x007F_FFFF) + 0x3F35_04F3);
    let f = m - 1.0;
    let s = f / (2.0 + f);
    let z = s * s;
    let w = z * z;
    let t1 = w * (0.400_009_72 + w * 0.242_790_79);
    let t2 = z * (0.666_666_63 + w * 0.284_987_87);
    let r = t2 + t1;
    let hfsq = 0.5 * f * f;
    let dk = k as f32;
    let y = s * (hfsq + r) + dk * LN2_LO_LOG - hfsq + f + dk * LN2_HI_LOG;
    if x > 0.0 && x < f32::INFINITY {
        y
    } else if x == 0.0 {
        f32::NEG_INFINITY
    } else if x == f32::INFINITY {
        x
    } else {
        f32::NAN
    }
}

// ---- sin, cos, tan -----------------------------------------------------------------

/// Arguments the three-part reduction handles accurately.
const TRIG_LIMIT: f32 = 8192.0;

/// `π/2` in three `f32` parts.
const PIO2_1: f32 = 1.570_796_4;
const PIO2_2: f32 = -4.371_139e-8;
const PIO2_3: f32 = -1.715_124_5e-15;

/// `(sin r, cos r, quadrant)` where `x = r + quadrant·π/2` and `|r| ≤ π/4`.
#[inline(always)]
fn sincos_core(x: f32) -> (f32, f32, i32) {
    let (k, q) = round(x * std::f32::consts::FRAC_2_PI);
    let r = (-k).mul_add(PIO2_1, x);
    let r = (-k).mul_add(PIO2_2, r);
    let r = (-k).mul_add(PIO2_3, r);
    // With k = 0 the steps only add zeros, which can flip the sign of a zero.
    let r = if k == 0.0 { x } else { r };
    let z = r * r;
    // Taylor series; |r| ≤ π/4 bounds the omitted terms by 0.05 ulp.
    let s = 1.0 / 362_880.0f32;
    let s = s.mul_add(z, -1.0 / 5_040.0);
    let s = s.mul_add(z, 1.0 / 120.0);
    let s = s.mul_add(z, -1.0 / 6.0);
    // `r` itself where it is zero, which keeps the sign of a zero.
    let sin = if r == 0.0 { r } else { (r * z).mul_add(s, r) };
    let c = -1.0 / 3_628_800.0f32;
    let c = c.mul_add(z, 1.0 / 40_320.0);
    let c = c.mul_add(z, -1.0 / 720.0);
    let c = c.mul_add(z, 1.0 / 24.0);
    let c = c.mul_add(z, -0.5);
    let cos = c.mul_add(z, 1.0);
    (sin, cos, q & 3)
}

#[inline(always)]
fn sin_core(x: f32) -> f32 {
    let (s, c, q) = sincos_core(x);
    let v = if q & 1 == 0 { s } else { c };
    if q & 2 == 0 { v } else { -v }
}

#[inline(always)]
fn cos_core(x: f32) -> f32 {
    let (s, c, q) = sincos_core(x);
    let v = if q & 1 == 0 { c } else { s };
    if (q + 1) & 2 == 0 { v } else { -v }
}

#[inline(always)]
fn tan_core(x: f32) -> f32 {
    let (k, q) = round(x * std::f32::consts::FRAC_2_PI);
    let r = (-k).mul_add(PIO2_1, x);
    let r = (-k).mul_add(PIO2_2, r);
    let r = (-k).mul_add(PIO2_3, r);
    let r = if k == 0.0 { x } else { r };
    // Cephes' tanf polynomial on |r| ≤ π/4; an odd quadrant is −1/tan r.
    let z = r * r;
    let p = 9.385_402e-3f32;
    let p = p.mul_add(z, 3.119_922_3e-3);
    let p = p.mul_add(z, 2.443_013_5e-2);
    let p = p.mul_add(z, 5.341_128e-2);
    let p = p.mul_add(z, 1.333_88e-1);
    let p = p.mul_add(z, 3.333_315_7e-1);
    let t = if r == 0.0 { r } else { (r * z).mul_add(p, r) };
    if q & 1 == 0 { t } else { -1.0 / t }
}

fn in_trig_range(x: f32) -> bool {
    x.abs() < TRIG_LIMIT
}

#[inline]
pub(crate) fn sin(x: f32) -> f32 {
    if in_trig_range(x) {
        sin_core(x)
    } else {
        x.sin()
    }
}

#[inline]
pub(crate) fn cos(x: f32) -> f32 {
    if in_trig_range(x) {
        cos_core(x)
    } else {
        x.cos()
    }
}

#[inline]
pub(crate) fn tan(x: f32) -> f32 {
    if in_trig_range(x) {
        tan_core(x)
    } else {
        x.tan()
    }
}

// ---- arctan, arcsin, arccos --------------------------------------------------------
//
// Cephes' single-precision reductions and polynomials, with every branch
// computed and the result selected. The constants they add are rounded, so
// each comes with its rounding error (`PIO2_2` for `π/2`) to add back first.

/// `π/4 − FRAC_PI_4` and `π − PI`, as `f32`.
const PIO4_LO: f32 = -2.185_569_5e-8;
const PI_LO: f32 = -8.742_278e-8;

#[inline(always)]
pub(crate) fn arctan(x: f32) -> f32 {
    let a = x.abs();
    // atan a = base + atan t, with |t| ≤ tan(π/8).
    let far = a > 2.414_213_6;
    let mid = a > 0.414_213_57;
    let t = if far {
        -1.0 / a
    } else if mid {
        (a - 1.0) / (a + 1.0)
    } else {
        a
    };
    let (base, base_lo) = if far {
        (FRAC_PI_2, PIO2_2)
    } else if mid {
        (FRAC_PI_4, PIO4_LO)
    } else {
        (0.0, 0.0)
    };
    let z = t * t;
    let p = 8.053_744_5e-2f32;
    let p = p.mul_add(z, -1.387_768_6e-1);
    let p = p.mul_add(z, 1.997_771_1e-1);
    let p = p.mul_add(z, -3.333_295e-1);
    // The base's rounding error goes back in before the base itself.
    let y = base + ((t * z).mul_add(p, t) + base_lo);
    y.copysign(x)
}

/// `asin a` for `a` in `[0, ½]`: Cephes' polynomial.
#[inline(always)]
fn asin_small(a: f32) -> f32 {
    let z = a * a;
    let p = 4.216_32e-2f32;
    let p = p.mul_add(z, 2.418_131_1e-2);
    let p = p.mul_add(z, 4.547_002_6e-2);
    let p = p.mul_add(z, 7.495_300_3e-2);
    let p = p.mul_add(z, 1.666_675_2e-1);
    if a == 0.0 { a } else { (a * z).mul_add(p, a) }
}

#[inline(always)]
pub(crate) fn arcsin(x: f32) -> f32 {
    let a = x.abs();
    // Above ½, asin a = π/2 − 2·asin √((1 − a)/2).
    let large = a > 0.5;
    let w = (0.5 * (1.0 - a)).sqrt();
    let y = if large {
        FRAC_PI_2 - (2.0 * asin_small(w) - PIO2_2)
    } else {
        asin_small(a)
    };
    let y = y.copysign(x);
    if a > 1.0 || x.is_nan() { f32::NAN } else { y }
}

#[inline(always)]
pub(crate) fn arccos(x: f32) -> f32 {
    let a = x.abs();
    // Beyond ½ either way, through the half-angle form; otherwise π/2 − asin.
    let w = (0.5 * (1.0 - a)).sqrt();
    let half = 2.0 * asin_small(w);
    let y = if x > 0.5 {
        half
    } else if x < -0.5 {
        PI - (half - PI_LO)
    } else {
        FRAC_PI_2 - (asin_small(a).copysign(x) - PIO2_2)
    };
    if a > 1.0 || x.is_nan() { f32::NAN } else { y }
}

// ---- sinh, cosh, tanh --------------------------------------------------------------

/// Beyond this, `e⁻ˣ` is below half an ulp of `eˣ` and the hyperbolic
/// functions are `eˣ/2`, built from [`exp_parts`] so they do not overflow
/// early.
const HUGE: f32 = 9.0;

#[inline(always)]
pub(crate) fn sinh(x: f32) -> f32 {
    let a = x.abs();
    // Taylor series below 1, where the exponential form would cancel.
    let z = a * a;
    let p = 1.0 / 39_916_800.0f32;
    let p = p.mul_add(z, 1.0 / 362_880.0);
    let p = p.mul_add(z, 1.0 / 5_040.0);
    let p = p.mul_add(z, 1.0 / 120.0);
    let p = p.mul_add(z, 1.0 / 6.0);
    let series = (a * z).mul_add(p, a);
    let e = exp(a);
    let moderate = (e - 1.0 / e) * 0.5;
    let (p, n) = exp_parts(clamp_exp(a));
    let huge = scale(p, n - 1);
    let y = if a < 1.0 {
        series
    } else if a < HUGE {
        moderate
    } else {
        huge
    };
    let y = y.copysign(x);
    if x.is_nan() { x } else { y }
}

#[inline(always)]
pub(crate) fn cosh(x: f32) -> f32 {
    let a = x.abs();
    let e = exp(a);
    let moderate = (e + 1.0 / e) * 0.5;
    let (p, n) = exp_parts(clamp_exp(a));
    let huge = scale(p, n - 1);
    let y = if a < HUGE { moderate } else { huge };
    if x.is_nan() { x } else { y }
}

#[inline(always)]
pub(crate) fn tanh(x: f32) -> f32 {
    let a = x.abs();
    // Cephes' polynomial below 0.625, 1 − 2/(e²ᵃ + 1) above.
    let z = a * a;
    let p = -5.704_988_7e-3f32;
    let p = p.mul_add(z, 2.063_908_9e-2);
    let p = p.mul_add(z, -5.373_971_6e-2);
    let p = p.mul_add(z, 1.333_144_2e-1);
    let p = p.mul_add(z, -3.333_328_2e-1);
    let small = (a * z).mul_add(p, a);
    let large = 1.0 - 2.0 / (exp(2.0 * a) + 1.0);
    let y = if a < 0.625 { small } else { large };
    let y = y.copysign(x);
    if x.is_nan() { x } else { y }
}

// ---- dispatch ----------------------------------------------------------------------

/// `op(x)`: the definition of every analytic function of an `f32` on the host.
pub(crate) fn value(op: Analytic, x: f32) -> f32 {
    match op {
        Analytic::Sin => sin(x),
        Analytic::Cos => cos(x),
        Analytic::Tan => tan(x),
        Analytic::Sec => cos(x).recip(),
        Analytic::Csc => sin(x).recip(),
        Analytic::Arcsin => arcsin(x),
        Analytic::Arccos => arccos(x),
        Analytic::Arctan => arctan(x),
        Analytic::Exp => exp(x),
        Analytic::Ln => ln(x),
        Analytic::Sinh => sinh(x),
        Analytic::Cosh => cosh(x),
        Analytic::Tanh => tanh(x),
        Analytic::Sqrt => x.sqrt(),
    }
}

/// `out[i] = op(values[i])`, bit for bit what [`value`] gives element by
/// element.
pub(crate) fn unary(op: Analytic, values: &[f32], out: &mut [f32]) {
    assert_eq!(values.len(), out.len(), "unary: lengths differ");
    // Without FMA in the baseline, `mul_add` is a library call per lane; with
    // it, the same loops compile to vector FMAs. Same results either way.
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        #[target_feature(enable = "avx2,fma")]
        unsafe fn unary_fma(op: Analytic, values: &[f32], out: &mut [f32]) {
            unary_loops(op, values, out)
        }
        // SAFETY: the features were just detected.
        return unsafe { unary_fma(op, values, out) };
    }
    unary_loops(op, values, out)
}

#[inline(always)]
fn unary_loops(op: Analytic, values: &[f32], out: &mut [f32]) {
    // One loop per function, so each body is a straight line LLVM vectorizes.
    #[inline(always)]
    fn each(values: &[f32], out: &mut [f32], f: impl Fn(f32) -> f32) {
        for (out, &x) in out.iter_mut().zip(values) {
            *out = f(x);
        }
    }
    // The trigonometric loops run without their range check, then patch the
    // elements outside the range with the same `libm` call `value` makes.
    #[inline(always)]
    fn trig(
        values: &[f32],
        out: &mut [f32],
        core: impl Fn(f32) -> f32,
        exact: impl Fn(f32) -> f32,
    ) {
        each(values, out, core);
        for (out, &x) in out.iter_mut().zip(values) {
            if !in_trig_range(x) {
                *out = exact(x);
            }
        }
    }
    match op {
        Analytic::Sin => trig(values, out, sin_core, f32::sin),
        Analytic::Cos => trig(values, out, cos_core, f32::cos),
        Analytic::Tan => trig(values, out, tan_core, f32::tan),
        Analytic::Sec => {
            trig(values, out, cos_core, f32::cos);
            out.iter_mut().for_each(|y| *y = y.recip());
        }
        Analytic::Csc => {
            trig(values, out, sin_core, f32::sin);
            out.iter_mut().for_each(|y| *y = y.recip());
        }
        Analytic::Arcsin => each(values, out, arcsin),
        Analytic::Arccos => each(values, out, arccos),
        Analytic::Arctan => each(values, out, arctan),
        Analytic::Exp => each(values, out, exp),
        Analytic::Ln => each(values, out, ln),
        Analytic::Sinh => each(values, out, sinh),
        Analytic::Cosh => each(values, out, cosh),
        Analytic::Tanh => each(values, out, tanh),
        Analytic::Sqrt => each(values, out, f32::sqrt),
    }
}

/// `op(x)` for the element types that compute in `f32` — `f32` itself, `f16`
/// and `bf16` — and `None` for every other type, which keeps its own
/// definitions.
///
/// The compact types widen exactly, evaluate in `f32` and round once, as the
/// `half` crate's own functions do. `sec` and `csc` are the reciprocal of the
/// rounded cosine or sine, as [`Analytic::value`] defines them for every type.
#[inline]
pub(crate) fn compact_value<T: Real>(op: Analytic, x: T) -> Option<T> {
    let x32 = if let Some(x) = as_type::<T, f32>(x) {
        x
    } else if let Some(x) = as_type::<T, f16>(x) {
        x.to_f32()
    } else if let Some(x) = as_type::<T, bf16>(x) {
        x.to_f32()
    } else {
        return None;
    };
    let round = |y: f32| -> T {
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            as_type::<f32, T>(y).unwrap()
        } else if TypeId::of::<T>() == TypeId::of::<f16>() {
            as_type::<f16, T>(f16::from_f32(y)).unwrap()
        } else {
            as_type::<bf16, T>(bf16::from_f32(y)).unwrap()
        }
    };
    Some(match op {
        Analytic::Sec => round(cos(x32)).recip(),
        Analytic::Csc => round(sin(x32)).recip(),
        Analytic::Sqrt => x.sqrt(),
        _ => round(value(op, x32)),
    })
}

/// `out[i] = op.value(values[i])` for any element type, vectorized for the
/// types [`compact_value`] covers.
pub(crate) fn unary_slice<T: Real>(op: Analytic, values: &[T], out: &mut [T]) {
    assert_eq!(values.len(), out.len(), "unary: lengths differ");
    if let (Some(values), Some(out)) = (slice_as::<T, f32>(values), slice_as_mut::<T, f32>(out)) {
        return unary(op, values, out);
    }
    let compact =
        TypeId::of::<T>() == TypeId::of::<f16>() || TypeId::of::<T>() == TypeId::of::<bf16>();
    if compact && !matches!(op, Analytic::Sec | Analytic::Csc | Analytic::Sqrt) {
        // Widen a block, evaluate it in f32, round it back: what `value` does
        // one element at a time.
        const BLOCK: usize = 1024;
        let mut wide = [0.0f32; BLOCK];
        let mut result = [0.0f32; BLOCK];
        for (values, out) in values.chunks(BLOCK).zip(out.chunks_mut(BLOCK)) {
            let n = values.len();
            for (w, &x) in wide.iter_mut().zip(values) {
                *w = x.into_f64() as f32;
            }
            unary(op, &wide[..n], &mut result[..n]);
            if let Some(out) = slice_as_mut::<T, f16>(out) {
                for (o, &y) in out.iter_mut().zip(&result[..n]) {
                    *o = f16::from_f32(y);
                }
            } else if let Some(out) = slice_as_mut::<T, bf16>(out) {
                for (o, &y) in out.iter_mut().zip(&result[..n]) {
                    *o = bf16::from_f32(y);
                }
            }
        }
        return;
    }
    for (out, &x) in out.iter_mut().zip(values) {
        *out = op.value(x);
    }
}

/// Elements per thread for [`unary_parallel`]: the functions cost a few
/// nanoseconds an element, so even short tensors are worth splitting.
const PARALLEL_GRAIN: usize = 8 * 1024;

/// [`unary_slice`], on several threads when `values` is long enough. Each
/// element's result is the same however the work is split.
pub(crate) fn unary_parallel<T: Real>(op: Analytic, values: &[T], out: &mut [T]) {
    assert_eq!(values.len(), out.len(), "unary: lengths differ");
    if !crate::parallel::plain_float::<T>() || values.len() < 2 * PARALLEL_GRAIN {
        return unary_slice(op, values, out);
    }
    // SAFETY: `T` is a plain float, which threads may share.
    unsafe {
        crate::parallel::for_slices_unchecked(out, PARALLEL_GRAIN, |start, window| {
            unary_slice(op, &values[start..start + window.len()], window);
        });
    }
}

fn as_type<T: 'static, U: Copy + 'static>(x: T) -> Option<U> {
    (TypeId::of::<T>() == TypeId::of::<U>()).then(|| {
        // SAFETY: one type.
        unsafe { std::mem::transmute_copy::<T, U>(&x) }
    })
}

fn slice_as<T: 'static, U: 'static>(values: &[T]) -> Option<&[U]> {
    (TypeId::of::<T>() == TypeId::of::<U>()).then(|| {
        // SAFETY: one type, so one layout.
        unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<U>(), values.len()) }
    })
}

fn slice_as_mut<T: 'static, U: 'static>(values: &mut [T]) -> Option<&mut [U]> {
    (TypeId::of::<T>() == TypeId::of::<U>()).then(|| {
        // SAFETY: one type, so one layout.
        unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<U>(), values.len()) }
    })
}

/// Elementary functions by element type: the `f32` ones here, `f64`'s from
/// `libm`. The scalar [`numbers`](crate::numbers) traits are written over
/// this, so `f32` and the compact types that compute in it agree with the
/// tensor kernels.
pub(crate) trait Elementary: Copy {
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sinh(self) -> Self;
    fn cosh(self) -> Self;
    fn tanh(self) -> Self;
}

impl Elementary for f32 {
    fn sin(self) -> f32 {
        sin(self)
    }
    fn cos(self) -> f32 {
        cos(self)
    }
    fn tan(self) -> f32 {
        tan(self)
    }
    fn asin(self) -> f32 {
        arcsin(self)
    }
    fn acos(self) -> f32 {
        arccos(self)
    }
    fn atan(self) -> f32 {
        arctan(self)
    }
    fn exp(self) -> f32 {
        exp(self)
    }
    fn ln(self) -> f32 {
        ln(self)
    }
    fn sinh(self) -> f32 {
        sinh(self)
    }
    fn cosh(self) -> f32 {
        cosh(self)
    }
    fn tanh(self) -> f32 {
        tanh(self)
    }
}

impl Elementary for f64 {
    fn sin(self) -> f64 {
        f64::sin(self)
    }
    fn cos(self) -> f64 {
        f64::cos(self)
    }
    fn tan(self) -> f64 {
        f64::tan(self)
    }
    fn asin(self) -> f64 {
        f64::asin(self)
    }
    fn acos(self) -> f64 {
        f64::acos(self)
    }
    fn atan(self) -> f64 {
        f64::atan(self)
    }
    fn exp(self) -> f64 {
        f64::exp(self)
    }
    fn ln(self) -> f64 {
        f64::ln(self)
    }
    fn sinh(self) -> f64 {
        f64::sinh(self)
    }
    fn cosh(self) -> f64 {
        f64::cosh(self)
    }
    fn tanh(self) -> f64 {
        f64::tanh(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Distance in ulps between an `f32` and the `f64` reference rounded.
    fn ulps(got: f32, want: f64) -> u32 {
        let want = want as f32;
        if got.is_nan() || want.is_nan() {
            return if got.is_nan() && want.is_nan() {
                0
            } else {
                u32::MAX
            };
        }
        if got == want {
            return 0;
        }
        let key = |v: f32| {
            let b = v.to_bits() as i64;
            if b < 0x8000_0000 { b } else { 0x8000_0000 - b }
        };
        (key(got) - key(want)).unsigned_abs() as u32
    }

    fn sample() -> Vec<f32> {
        let mut xs: Vec<f32> = Vec::new();
        // Dense near zero, then logarithmically spaced out to 1e7.
        for i in -20_000..=20_000 {
            xs.push(i as f32 * 1.0e-4);
        }
        let mut v = 1.0e-30f32;
        while v < 1.0e7 {
            xs.push(v);
            xs.push(-v);
            v *= 1.0137;
        }
        xs.extend([
            0.0,
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::MIN_POSITIVE,
            1.0e-45,
            f32::MAX,
            f32::MIN,
            88.72,
            88.73,
            -103.0,
            -87.33,
            std::f32::consts::FRAC_PI_2,
            std::f32::consts::PI,
            1.0,
            -1.0,
            0.999_999_9,
            -0.999_999_9,
        ]);
        xs
    }

    fn check(op: Analytic, reference: fn(f64) -> f64, tolerance: u32) {
        let xs = sample();
        let mut batch = vec![0.0f32; xs.len()];
        unary(op, &xs, &mut batch);
        let mut worst = (0, 0.0f32);
        for (&x, &b) in xs.iter().zip(&batch) {
            let scalar = value(op, x);
            assert_eq!(
                scalar.to_bits(),
                b.to_bits(),
                "{op:?}({x}): batch {b} against scalar {scalar}"
            );
            let gap = ulps(scalar, reference(f64::from(x)));
            if gap > worst.0 {
                worst = (gap, x);
            }
        }
        assert!(
            worst.0 <= tolerance,
            "{op:?}: {} ulps at {}",
            worst.0,
            worst.1
        );
    }

    #[test]
    fn every_function_is_accurate_and_the_batch_matches_the_scalar() {
        for (op, reference, tolerance) in REFERENCES {
            check(op, reference, tolerance);
        }
    }

    /// Each function, its `f64` reference and its tolerance in ulps, as
    /// measured by `sweep` over every 61st `f32` in the trigonometric range.
    type Reference = (Analytic, fn(f64) -> f64, u32);

    const REFERENCES: [Reference; 14] = [
        (Analytic::Exp, f64::exp, 1),
        (Analytic::Ln, f64::ln, 1),
        (Analytic::Sin, f64::sin, 1),
        (Analytic::Cos, f64::cos, 1),
        (Analytic::Tan, f64::tan, 3),
        (Analytic::Arctan, f64::atan, 2),
        (Analytic::Arcsin, f64::asin, 2),
        (Analytic::Arccos, f64::acos, 1),
        (Analytic::Sinh, f64::sinh, 2),
        (Analytic::Cosh, f64::cosh, 1),
        (Analytic::Tanh, f64::tanh, 1),
        // Two roundings: the f32 cosine or sine, then its reciprocal.
        (Analytic::Sec, |x| 1.0 / x.cos(), 2),
        (Analytic::Csc, |x| 1.0 / x.sin(), 2),
        (Analytic::Sqrt, f64::sqrt, 0),
    ];

    #[test]
    #[ignore = "slow: about 70M evaluations per function"]
    fn sweep() {
        for (op, reference, tolerance) in REFERENCES {
            let mut worst = (0, 0.0f32);
            let mut bits = 0u32;
            while bits < u32::MAX - 61 {
                let x = f32::from_bits(bits);
                bits += 61;
                if x.is_nan() || x.abs() >= 9000.0 {
                    continue;
                }
                let gap = ulps(value(op, x), reference(f64::from(x)));
                if gap > worst.0 {
                    worst = (gap, x);
                }
            }
            println!("{op:?}: worst {} ulps at {:e}", worst.0, worst.1);
            assert!(worst.0 <= tolerance, "{op:?}");
        }
    }

    #[test]
    fn signed_zeros_and_limits_follow_libm() {
        for op in Analytic::ALL {
            let at = |x: f32| value(op, x);
            match op {
                Analytic::Sin
                | Analytic::Tan
                | Analytic::Arcsin
                | Analytic::Arctan
                | Analytic::Sinh
                | Analytic::Tanh
                | Analytic::Sqrt => {
                    assert_eq!(at(-0.0).to_bits(), (-0.0f32).to_bits(), "{op:?}(-0)");
                }
                _ => {}
            }
        }
        assert_eq!(exp(f32::NEG_INFINITY), 0.0);
        assert_eq!(exp(f32::INFINITY), f32::INFINITY);
        assert_eq!(ln(0.0), f32::NEG_INFINITY);
        assert!(ln(-1.0).is_nan());
        assert_eq!(tanh(f32::INFINITY), 1.0);
        assert_eq!(tanh(f32::NEG_INFINITY), -1.0);
        assert_eq!(arctan(f32::INFINITY), std::f32::consts::FRAC_PI_2);
        assert_eq!(arccos(-1.0), std::f32::consts::PI);
        assert_eq!(arccos(1.0).to_bits(), 0.0f32.to_bits());
        assert!(arcsin(1.5).is_nan() && arccos(-1.5).is_nan());
        assert!(sin(f32::INFINITY).is_nan());
    }
}

#[cfg(test)]
mod speed {
    use super::*;

    #[test]
    #[ignore = "timing, run by hand"]
    fn against_libm() {
        let xs: Vec<f32> = (0..1 << 20)
            .map(|i| (i as f32 * 0.37).sin() * 20.0)
            .collect();
        let mut out = vec![0.0f32; xs.len()];
        for op in Analytic::ALL {
            let start = std::time::Instant::now();
            for _ in 0..10 {
                unary(op, &xs, &mut out);
            }
            let fast = start.elapsed() / 10;
            let libm = |x: f32| match op {
                Analytic::Sin => x.sin(),
                Analytic::Cos => x.cos(),
                Analytic::Tan => x.tan(),
                Analytic::Sec => x.cos().recip(),
                Analytic::Csc => x.sin().recip(),
                Analytic::Arcsin => (x * 0.05).asin(),
                Analytic::Arccos => (x * 0.05).acos(),
                Analytic::Arctan => x.atan(),
                Analytic::Exp => x.exp(),
                Analytic::Ln => x.abs().ln(),
                Analytic::Sinh => x.sinh(),
                Analytic::Cosh => x.cosh(),
                Analytic::Tanh => x.tanh(),
                Analytic::Sqrt => x.abs().sqrt(),
            };
            let start = std::time::Instant::now();
            for _ in 0..10 {
                for (o, &x) in out.iter_mut().zip(&xs) {
                    *o = libm(x);
                }
                std::hint::black_box(&mut out);
            }
            let slow = start.elapsed() / 10;
            println!(
                "{op:?}: {fast:?} vs libm {slow:?} ({:.1}x)",
                slow.as_secs_f64() / fast.as_secs_f64()
            );
        }
    }
}
