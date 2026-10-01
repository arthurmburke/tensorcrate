//! The compact floats, `f16` and `bf16`, on the CPU.
//!
//! Two rules decide what an operation on a compact tensor computes:
//!
//! - **Elementwise operations round to the element type every time.** `a + b`
//!   on `f16` is the correctly rounded `f16` sum — what an FP16 instruction
//!   produces, and what the `half` crate's scalar operators produce too. On
//!   AArch64 with the `simd` feature the `f16` operations here *are* those
//!   instructions (`fadd v.8h`, `fsqrt v.8h`, …), eight lanes at a time.
//!   `bf16` has no arithmetic instructions on any CPU, only conversions and
//!   widening multiply-adds, so a `bf16` operation widens to `f32` (exactly),
//!   computes, and rounds back once with `BFCVTN` where the CPU has it — which
//!   is the correctly rounded `bf16` result.
//!
//! - **Accumulation runs in `f32` and rounds once.** A sum, a dot product, a
//!   matrix product, a prefix sum or a variance folds its terms in `f32` and
//!   rounds the total to the element type at the end. Every product of two
//!   compact values is exact in `f32`, so this is the arithmetic of the
//!   hardware's widening multiply-accumulate (`FMLAL`, `BFMLALB`/`BFMLALT`,
//!   which the dot products use directly) and of the GPU kernels — and it is
//!   the difference between a sum of ten thousand ones being `10000` rather
//!   than the `2048` an `f16` running total gets stuck at.
//!
//! Every entry point takes a generic slice and answers `None`/`false` for any
//! element type other than `f16` and `bf16`, so the tensor code can try it
//! first and fall through to its usual paths.

use std::any::TypeId;

use half::{bf16, f16};

use crate::tensors::{BinaryOp, Compare, Host, Matrix, Reduce};

/// One of the two compact types.
trait Compact: Copy + 'static {
    fn widen(self) -> f32;
    fn narrow(value: f32) -> Self;
}

impl Compact for f16 {
    fn widen(self) -> f32 {
        f32::from(self)
    }

    fn narrow(value: f32) -> Self {
        f16::from_f32(value)
    }
}

impl Compact for bf16 {
    fn widen(self) -> f32 {
        f32::from(self)
    }

    fn narrow(value: f32) -> Self {
        bf16::from_f32(value)
    }
}

fn same<T: 'static, U: 'static>() -> bool {
    TypeId::of::<T>() == TypeId::of::<U>()
}

/// `&[T]` as `&[U]`, once [`dispatch!`] has established they are one type.
fn cast<T: 'static, U: 'static>(values: &[T]) -> &[U] {
    assert!(same::<T, U>());
    // SAFETY: one type, so one layout.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<U>(), values.len()) }
}

/// The mutable form of [`cast`].
fn cast_mut<T: 'static, U: 'static>(values: &mut [T]) -> &mut [U] {
    assert!(same::<T, U>());
    // SAFETY: one type, so one layout.
    unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<U>(), values.len()) }
}

/// A value of `T` as `U`, once they are known to be one type.
fn cast_value<T: Copy + 'static, U: Copy + 'static>(value: T) -> U {
    cast::<T, U>(std::slice::from_ref(&value))[0]
}

/// A `Vec<T>` as a `Vec<U>`, once they are known to be one type.
fn cast_vec<T: 'static, U: 'static>(values: Vec<T>) -> Vec<U> {
    assert!(same::<T, U>());
    let mut values = std::mem::ManuallyDrop::new(values);
    // SAFETY: one type, so the allocation and its layout carry over unchanged.
    unsafe { Vec::from_raw_parts(values.as_mut_ptr().cast::<U>(), values.len(), values.capacity()) }
}

/// Evaluate `$body` with `$C` bound to the compact type `$T` is, or give
/// `$otherwise` when it is neither.
macro_rules! dispatch {
    ($T:ty, $C:ident => $body:expr, otherwise $otherwise:expr) => {
        if same::<$T, f16>() {
            type $C = f16;
            $body
        } else if same::<$T, bf16>() {
            type $C = bf16;
            $body
        } else {
            $otherwise
        }
    };
}

// ---- conversions --------------------------------------------------------------

/// The values widened to `f32`, exactly.
pub(crate) fn widen<T: 'static>(values: &[T]) -> Option<Vec<f32>> {
    dispatch!(T, C => {
        let mut out = vec![0.0f32; values.len()];
        widen_into::<C>(cast(values), &mut out);
        Some(out)
    }, otherwise None)
}

/// `values` rounded to `T`, once each.
pub(crate) fn narrow<T: 'static>(values: &[f32]) -> Option<Vec<T>> {
    dispatch!(T, C => {
        let mut out = vec![C::narrow(0.0); values.len()];
        narrow_into::<C>(values, &mut out);
        Some(cast_vec(out))
    }, otherwise None)
}

fn widen_into<C: Compact>(values: &[C], out: &mut [f32]) {
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    if neon::widen(values, out) {
        return;
    }
    for (out, &x) in out.iter_mut().zip(values) {
        *out = x.widen();
    }
}

fn narrow_into<C: Compact>(values: &[f32], out: &mut [C]) {
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    if neon::narrow(values, out) {
        return;
    }
    for (out, &x) in out.iter_mut().zip(values) {
        *out = C::narrow(x);
    }
}

// ---- accumulations: f32, rounded once ------------------------------------------

/// `Σ aᵢbᵢ`, accumulated in `f32`.
pub(crate) fn dot<T: Copy + 'static>(a: &[T], b: &[T]) -> Option<T> {
    dispatch!(T, C => {
        let (a, b) = (cast::<T, C>(a), cast::<T, C>(b));
        Some(cast_value(C::narrow(dot_f32(a, b))))
    }, otherwise None)
}

fn dot_f32<C: Compact>(a: &[C], b: &[C]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    if let Some(total) = neon::dot(a, b) {
        return total;
    }
    // Each product of two compact values is exact in `f32`, so this is one
    // rounding per term, as a fused multiply-add would be.
    a.iter()
        .zip(b)
        .fold(0.0f32, |total, (&x, &y)| total + x.widen() * y.widen())
}

/// A whole-slice fold, the `f32` total rounded once. `Min` and `Max` are exact
/// either way; `Sum` is where the `f32` accumulator matters.
pub(crate) fn reduce<T: Copy + 'static>(values: &[T], op: Reduce) -> Option<T> {
    dispatch!(T, C => {
        let values = cast::<T, C>(values);
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        if let Some(total) = neon::reduce(values, op) {
            return Some(cast_value(C::narrow(total)));
        }
        Some(cast_value(C::narrow(fold_f32(values, op))))
    }, otherwise None)
}

/// [`reduce`] left to right, one term at a time — the definition the
/// vectorized fold is tested against, and what
/// [`Reduce::fold`](crate::tensors::Reduce::fold) is for the compact types.
pub(crate) fn fold<T: Copy + 'static>(values: &[T], op: Reduce) -> Option<T> {
    dispatch!(T, C => Some(cast_value(C::narrow(fold_f32(cast::<T, C>(values), op)))), otherwise None)
}

fn fold_f32<C: Compact>(values: &[C], op: Reduce) -> f32 {
    values
        .iter()
        .fold(op.identity::<f32>(), |total, &x| op.combine(total, x.widen()))
}

/// Inclusive prefix sum with an `f32` running total, each output rounded once.
pub(crate) fn prefix_sum<T: 'static>(values: &[T]) -> Option<Vec<T>> {
    dispatch!(T, C => {
        let mut running = 0.0f32;
        let out: Vec<C> = cast::<T, C>(values)
            .iter()
            .map(|&x| {
                running += x.widen();
                C::narrow(running)
            })
            .collect();
        Some(cast_vec(out))
    }, otherwise None)
}

/// `out = a·b` (or `out += a·b` when `accumulate`), for a `rows × inner` by
/// `inner × cols` product in row-major order.
///
/// The operands are widened once, the product runs through the `f32` matrix
/// path — Accelerate's SGEMM on macOS, the SIMD kernels elsewhere — and every
/// output element rounds once. Widening is exact and every product of compact
/// values is exact in `f32`, so this is the arithmetic of a widening
/// multiply-accumulate instruction, at the speed of the best `f32` GEMM
/// available.
pub(crate) fn matmul<T: 'static>(
    a: &[T],
    b: &[T],
    rows: usize,
    inner: usize,
    cols: usize,
    out: &mut [T],
    accumulate: bool,
) -> bool {
    dispatch!(T, C => {
        let left = Matrix::<f32, Host>::from_flat(rows, inner, widen(a).expect("compact"));
        let right = Matrix::<f32, Host>::from_flat(inner, cols, widen(b).expect("compact"));
        let product = if accumulate {
            let addend = Matrix::<f32, Host>::from_flat(rows, cols, widen(&*out).expect("compact"));
            left.matmul_add(&right, addend)
        } else {
            left.matmul(&right)
        };
        narrow_into::<C>(product.data(), cast_mut(out));
        true
    }, otherwise false)
}

// ---- elementwise: rounded every operation -------------------------------------
//
// These change nothing about the values — the scalar `half` operators are
// already correctly rounded — so they only run where there are vector
// instructions to run them with, and otherwise report `false` and leave the
// caller's scalar loop to it.

/// `out = a op b`, elementwise.
pub(crate) fn elementwise<T: 'static>(a: &[T], b: &[T], op: BinaryOp, out: &mut [T]) -> bool {
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    return dispatch!(T, C => neon::binary::<C>(cast(a), cast(b), op, cast_mut(out)), otherwise false);
    #[allow(unreachable_code)]
    {
        let _ = (a, b, op, out);
        false
    }
}

/// `out = values op scalar`, or `scalar op values` when `scalar_left`.
pub(crate) fn broadcast<T: Copy + 'static>(
    values: &[T],
    scalar: T,
    op: BinaryOp,
    scalar_left: bool,
    out: &mut [T],
) -> bool {
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    return dispatch!(T, C => neon::broadcast::<C>(
        cast(values),
        cast_value(scalar),
        op,
        scalar_left,
        cast_mut(out),
    ), otherwise false);
    #[allow(unreachable_code)]
    {
        let _ = (values, scalar, op, scalar_left, out);
        false
    }
}

/// `out = op(a, b)`, elementwise, including the `1`/`0` predicates.
pub(crate) fn compare<T: 'static>(a: &[T], b: &[T], op: Compare, out: &mut [T]) -> bool {
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    return dispatch!(T, C => neon::compare::<C>(cast(a), cast(b), op, cast_mut(out)), otherwise false);
    #[allow(unreachable_code)]
    {
        let _ = (a, b, op, out);
        false
    }
}

/// `out = op(values, scalar)`, or `op(scalar, values)` when `scalar_left`.
pub(crate) fn compare_scalar<T: Copy + 'static>(
    values: &[T],
    scalar: T,
    op: Compare,
    scalar_left: bool,
    out: &mut [T],
) -> bool {
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    return dispatch!(T, C => neon::compare_scalar::<C>(
        cast(values),
        cast_value(scalar),
        op,
        scalar_left,
        cast_mut(out),
    ), otherwise false);
    #[allow(unreachable_code)]
    {
        let _ = (values, scalar, op, scalar_left, out);
        false
    }
}

/// `out = min(max(values, low), high)`.
#[cfg_attr(not(feature = "simd"), allow(dead_code))]
pub(crate) fn clamp<T: Copy + 'static>(values: &[T], low: T, high: T, out: &mut [T]) -> bool {
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    return dispatch!(T, C => neon::clamp::<C>(
        cast(values),
        cast_value(low),
        cast_value(high),
        cast_mut(out),
    ), otherwise false);
    #[allow(unreachable_code)]
    {
        let _ = (values, low, high, out);
        false
    }
}

/// `out = √values`.
pub(crate) fn sqrt<T: 'static>(values: &[T], out: &mut [T]) -> bool {
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    return dispatch!(T, C => neon::sqrt::<C>(cast(values), cast_mut(out)), otherwise false);
    #[allow(unreachable_code)]
    {
        let _ = (values, out);
        false
    }
}

/// AArch64 kernels.
///
/// Stable Rust has no `f16` vector type, so `f16` and `bf16` lanes travel as
/// `uint16x8_t` and the half-precision instructions are written as inline
/// assembly; everything on `f32` lanes uses the ordinary intrinsics. `FP16` is
/// part of the Apple-silicon baseline, while `BF16` (M2 and later) and `FHM`
/// are detected at runtime, with an exact fallback for each.
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
mod neon {
    use std::arch::aarch64::*;
    use std::arch::asm;
    use std::arch::is_aarch64_feature_detected;

    use half::f16;

    use super::{BinaryOp, Compact, Compare, Reduce, same};

    /// Lanes per vector of a 16-bit type.
    const LANES: usize = 8;

    /// A compact slice as its raw 16-bit encodings.
    fn lanes<C: Compact>(values: &[C]) -> &[u16] {
        assert_eq!(size_of::<C>(), size_of::<u16>());
        // SAFETY: both compact types are `repr(transparent)` over a `u16`, so
        // every value is a valid `u16` and the layouts match.
        unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u16>(), values.len()) }
    }

    /// The mutable form of [`lanes`]. Any `u16` is a valid compact value.
    fn lanes_mut<C: Compact>(values: &mut [C]) -> &mut [u16] {
        assert_eq!(size_of::<C>(), size_of::<u16>());
        // SAFETY: as in `lanes`, and every bit pattern is a valid `f16`/`bf16`.
        unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u16>(), values.len()) }
    }

    fn bits<C: Compact>(value: C) -> u16 {
        lanes(std::slice::from_ref(&value))[0]
    }

    fn fp16() -> bool {
        is_aarch64_feature_detected!("fp16")
    }

    fn bf16_instructions() -> bool {
        is_aarch64_feature_detected!("bf16")
    }

    fn fhm() -> bool {
        is_aarch64_feature_detected!("fhm")
    }

    // ---- single-vector operations ------------------------------------------------

    macro_rules! f16_binary_asm {
        ($name:ident, $instruction:literal) => {
            #[target_feature(enable = "neon,fp16")]
            #[inline]
            unsafe fn $name(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
                let out: uint16x8_t;
                unsafe {
                    asm!(
                        concat!($instruction, " {o:v}.8h, {a:v}.8h, {b:v}.8h"),
                        o = lateout(vreg) out,
                        a = in(vreg) a,
                        b = in(vreg) b,
                        options(pure, nomem, nostack, preserves_flags),
                    );
                }
                out
            }
        };
    }

    f16_binary_asm!(f16_add, "fadd");
    f16_binary_asm!(f16_sub, "fsub");
    f16_binary_asm!(f16_mul, "fmul");
    f16_binary_asm!(f16_div, "fdiv");
    // `minNum`/`maxNum`: a number beats a NaN, as `f32::min` and the shaders do.
    f16_binary_asm!(f16_min, "fminnm");
    f16_binary_asm!(f16_max, "fmaxnm");
    // Masks: all ones where the comparison holds, which a NaN never does.
    f16_binary_asm!(f16_greater, "fcmgt");
    f16_binary_asm!(f16_greater_equal, "fcmge");

    #[target_feature(enable = "neon,fp16")]
    #[inline]
    unsafe fn f16_sqrt(a: uint16x8_t) -> uint16x8_t {
        let out: uint16x8_t;
        unsafe {
            asm!(
                "fsqrt {o:v}.8h, {a:v}.8h",
                o = lateout(vreg) out,
                a = in(vreg) a,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
        out
    }

    #[target_feature(enable = "neon,fp16")]
    #[inline]
    unsafe fn f16_widen(a: uint16x8_t) -> (float32x4_t, float32x4_t) {
        let (lo, hi): (float32x4_t, float32x4_t);
        unsafe {
            asm!(
                "fcvtl {lo:v}.4s, {a:v}.4h",
                "fcvtl2 {hi:v}.4s, {a:v}.8h",
                // `lo` is written before `a` is read again, so it must not
                // share `a`'s register: `out`, not `lateout`.
                lo = out(vreg) lo,
                hi = lateout(vreg) hi,
                a = in(vreg) a,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
        (lo, hi)
    }

    #[target_feature(enable = "neon,fp16")]
    #[inline]
    unsafe fn f16_narrow(lo: float32x4_t, hi: float32x4_t) -> uint16x8_t {
        let out: uint16x8_t;
        unsafe {
            asm!(
                "fcvtn {o:v}.4h, {lo:v}.4s",
                "fcvtn2 {o:v}.8h, {hi:v}.4s",
                o = out(vreg) out,
                lo = in(vreg) lo,
                hi = in(vreg) hi,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
        out
    }

    /// `acc_lo += a[0..4]·b[0..4]`, `acc_hi += a[4..8]·b[4..8]`, widening and
    /// fused — `FMLAL`/`FMLAL2`.
    #[target_feature(enable = "neon,fp16,fhm")]
    #[inline]
    unsafe fn f16_fmlal(
        acc_lo: float32x4_t,
        acc_hi: float32x4_t,
        a: uint16x8_t,
        b: uint16x8_t,
    ) -> (float32x4_t, float32x4_t) {
        let (mut lo, mut hi) = (acc_lo, acc_hi);
        unsafe {
            asm!(
                "fmlal {lo:v}.4s, {a:v}.4h, {b:v}.4h",
                "fmlal2 {hi:v}.4s, {a:v}.4h, {b:v}.4h",
                lo = inout(vreg) lo,
                hi = inout(vreg) hi,
                a = in(vreg) a,
                b = in(vreg) b,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
        (lo, hi)
    }

    /// A `bf16` vector widened to two `f32` ones: the top half of an `f32` is a
    /// `bf16`, so this is a shift.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn bf16_widen(a: uint16x8_t) -> (float32x4_t, float32x4_t) {
        (
            vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(a))),
            vreinterpretq_f32_u32(vshll_high_n_u16::<16>(a)),
        )
    }

    /// Round two `f32` vectors to one `bf16` vector — `BFCVTN`/`BFCVTN2`.
    #[target_feature(enable = "neon,bf16")]
    #[inline]
    unsafe fn bf16_narrow_instruction(lo: float32x4_t, hi: float32x4_t) -> uint16x8_t {
        let out: uint16x8_t;
        unsafe {
            asm!(
                "bfcvtn {o:v}.4h, {lo:v}.4s",
                "bfcvtn2 {o:v}.8h, {hi:v}.4s",
                o = out(vreg) out,
                lo = in(vreg) lo,
                hi = in(vreg) hi,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
        out
    }

    /// The same rounding for a CPU without `BF16`: round to nearest, ties to
    /// even, by adding `0x7FFF` plus the result's low bit and truncating, with
    /// NaNs quieted rather than rounded — bit for bit `half::bf16::from_f32`.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn bf16_narrow_software(lo: float32x4_t, hi: float32x4_t) -> uint16x8_t {
        let round = |x: float32x4_t| {
            let bits = vreinterpretq_u32_f32(x);
            let lsb = vandq_u32(vshrq_n_u32::<16>(bits), vdupq_n_u32(1));
            let rounded = vshrq_n_u32::<16>(vaddq_u32(bits, vaddq_u32(vdupq_n_u32(0x7FFF), lsb)));
            let quiet = vorrq_u32(vshrq_n_u32::<16>(bits), vdupq_n_u32(0x0040));
            let is_number = vceqq_f32(x, x);
            vmovn_u32(vbslq_u32(is_number, rounded, quiet))
        };
        vcombine_u16(round(lo), round(hi))
    }

    /// `acc_even += a[even]·b[even]`, `acc_odd += a[odd]·b[odd]`, widening and
    /// fused — `BFMLALB`/`BFMLALT`.
    #[target_feature(enable = "neon,bf16")]
    #[inline]
    unsafe fn bf16_fmlal(
        acc_even: float32x4_t,
        acc_odd: float32x4_t,
        a: uint16x8_t,
        b: uint16x8_t,
    ) -> (float32x4_t, float32x4_t) {
        let (mut even, mut odd) = (acc_even, acc_odd);
        unsafe {
            asm!(
                "bfmlalb {even:v}.4s, {a:v}.8h, {b:v}.8h",
                "bfmlalt {odd:v}.4s, {a:v}.8h, {b:v}.8h",
                even = inout(vreg) even,
                odd = inout(vreg) odd,
                a = in(vreg) a,
                b = in(vreg) b,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
        (even, odd)
    }

    /// The `bf16` loops, twice over: once with `BFCVTN` for the rounding, for a
    /// CPU with `BF16`, and once with the integer rounding for one without.
    /// Each copy is compiled with the features it uses, so the rounding inlines
    /// into the loop; [`bf16!`] picks the copy once per call.
    macro_rules! bf16_loops {
        ($module:ident, $features:literal, $narrow:ident) => {
            mod $module {
                use super::*;

        #[target_feature(enable = $features)]
        pub(super) unsafe fn narrow_bf16(values: &[f32], out: &mut [u16]) {
            for i in (0..values.len()).step_by(LANES) {
                let (lo, hi) = unsafe { load_f32(values, i) };
                unsafe { store(out, i, $narrow(lo, hi)) };
            }
        }

        #[target_feature(enable = $features)]
        pub(super) unsafe fn binary_bf16(a: &[u16], b: &[u16], op: BinaryOp, out: &mut [u16]) {
            for i in (0..out.len()).step_by(LANES) {
                let ((x_lo, x_hi), (y_lo, y_hi)) =
                    unsafe { (bf16_widen(load(a, i)), bf16_widen(load(b, i))) };
                let z = unsafe { $narrow(f32_op(op, x_lo, y_lo), f32_op(op, x_hi, y_hi)) };
                unsafe { store(out, i, z) };
            }
        }

        #[target_feature(enable = $features)]
        pub(super) unsafe fn broadcast_bf16(
            values: &[u16],
            scalar: u16,
            op: BinaryOp,
            scalar_left: bool,
            out: &mut [u16],
        ) {
            let s = vreinterpretq_f32_u32(vdupq_n_u32(u32::from(scalar) << 16));
            for i in (0..out.len()).step_by(LANES) {
                let (x_lo, x_hi) = unsafe { bf16_widen(load(values, i)) };
                let (lo, hi) = if scalar_left {
                    (f32_op(op, s, x_lo), f32_op(op, s, x_hi))
                } else {
                    (f32_op(op, x_lo, s), f32_op(op, x_hi, s))
                };
                unsafe { store(out, i, $narrow(lo, hi)) };
            }
        }

        #[target_feature(enable = $features)]
        pub(super) unsafe fn compare_bf16(
            a: &[u16],
            b: &[u16],
            scalar: Option<u16>,
            op: Compare,
            scalar_left: bool,
            out: &mut [u16],
        ) {
            let s = scalar.map(|s| vreinterpretq_f32_u32(vdupq_n_u32(u32::from(s) << 16)));
            for i in (0..out.len()).step_by(LANES) {
                let (x_lo, x_hi) = unsafe { bf16_widen(load(a, i)) };
                let (lo, hi) = match s {
                    None => {
                        let (y_lo, y_hi) = unsafe { bf16_widen(load(b, i)) };
                        (f32_compare(op, x_lo, y_lo), f32_compare(op, x_hi, y_hi))
                    }
                    Some(s) if scalar_left => (f32_compare(op, s, x_lo), f32_compare(op, s, x_hi)),
                    Some(s) => (f32_compare(op, x_lo, s), f32_compare(op, x_hi, s)),
                };
                unsafe { store(out, i, $narrow(lo, hi)) };
            }
        }

        #[target_feature(enable = $features)]
        pub(super) unsafe fn clamp_bf16(values: &[u16], low: u16, high: u16, out: &mut [u16]) {
            let widen = |s: u16| vreinterpretq_f32_u32(vdupq_n_u32(u32::from(s) << 16));
            let (low, high) = (widen(low), widen(high));
            for i in (0..out.len()).step_by(LANES) {
                let (lo, hi) = unsafe { bf16_widen(load(values, i)) };
                let lo = vminnmq_f32(vmaxnmq_f32(lo, low), high);
                let hi = vminnmq_f32(vmaxnmq_f32(hi, low), high);
                unsafe { store(out, i, $narrow(lo, hi)) };
            }
        }

        #[target_feature(enable = $features)]
        pub(super) unsafe fn sqrt_bf16(values: &[u16], out: &mut [u16]) {
            for i in (0..out.len()).step_by(LANES) {
                let (lo, hi) = unsafe { bf16_widen(load(values, i)) };
                unsafe { store(out, i, $narrow(vsqrtq_f32(lo), vsqrtq_f32(hi))) };
            }
        }

            }
        };
    }

    bf16_loops!(bf16_hardware, "neon,bf16", bf16_narrow_instruction);
    bf16_loops!(bf16_software, "neon", bf16_narrow_software);

    /// Call the `bf16` loop `$name` from whichever copy this CPU can run.
    macro_rules! bf16 {
        ($name:ident($($argument:expr),* $(,)?)) => {
            if bf16_instructions() {
                unsafe { bf16_hardware::$name($($argument),*) }
            } else {
                unsafe { bf16_software::$name($($argument),*) }
            }
        };
    }

    // ---- loads and stores, with the ragged tail through a padded vector -------------

    /// Load lanes `i..i + 8` of `values`, zero-filling past the end.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn load(values: &[u16], i: usize) -> uint16x8_t {
        if i + LANES <= values.len() {
            unsafe { vld1q_u16(values.as_ptr().add(i)) }
        } else {
            let mut padded = [0u16; LANES];
            padded[..values.len() - i].copy_from_slice(&values[i..]);
            unsafe { vld1q_u16(padded.as_ptr()) }
        }
    }

    /// Store lanes into `out[i..i + 8]`, dropping those past the end.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn store(out: &mut [u16], i: usize, value: uint16x8_t) {
        if i + LANES <= out.len() {
            unsafe { vst1q_u16(out.as_mut_ptr().add(i), value) };
        } else {
            let mut padded = [0u16; LANES];
            unsafe { vst1q_u16(padded.as_mut_ptr(), value) };
            let n = out.len() - i;
            out[i..].copy_from_slice(&padded[..n]);
        }
    }

    /// Eight `f32`s from `values[i..]`, zero-filling past the end.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn load_f32(values: &[f32], i: usize) -> (float32x4_t, float32x4_t) {
        if i + LANES <= values.len() {
            unsafe { (vld1q_f32(values.as_ptr().add(i)), vld1q_f32(values.as_ptr().add(i + 4))) }
        } else {
            let mut padded = [0.0f32; LANES];
            padded[..values.len() - i].copy_from_slice(&values[i..]);
            unsafe { (vld1q_f32(padded.as_ptr()), vld1q_f32(padded.as_ptr().add(4))) }
        }
    }

    /// Store eight `f32`s into `out[i..]`, dropping those past the end.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn store_f32(out: &mut [f32], i: usize, (lo, hi): (float32x4_t, float32x4_t)) {
        let mut padded = [0.0f32; LANES];
        unsafe {
            vst1q_f32(padded.as_mut_ptr(), lo);
            vst1q_f32(padded.as_mut_ptr().add(4), hi);
        }
        let n = (out.len() - i).min(LANES);
        out[i..i + n].copy_from_slice(&padded[..n]);
    }

    // ---- conversions ---------------------------------------------------------------

    pub(super) fn widen<C: Compact>(values: &[C], out: &mut [f32]) -> bool {
        let values = lanes(values);
        if same::<C, f16>() {
            if !fp16() {
                return false;
            }
            unsafe { widen_f16(values, out) };
        } else {
            unsafe { widen_bf16(values, out) };
        }
        true
    }

    #[target_feature(enable = "neon,fp16")]
    unsafe fn widen_f16(values: &[u16], out: &mut [f32]) {
        for i in (0..values.len()).step_by(LANES) {
            unsafe { store_f32(out, i, f16_widen(load(values, i))) };
        }
    }

    #[target_feature(enable = "neon")]
    unsafe fn widen_bf16(values: &[u16], out: &mut [f32]) {
        for i in (0..values.len()).step_by(LANES) {
            unsafe { store_f32(out, i, bf16_widen(load(values, i))) };
        }
    }

    pub(super) fn narrow<C: Compact>(values: &[f32], out: &mut [C]) -> bool {
        let out = lanes_mut(out);
        if same::<C, f16>() {
            if !fp16() {
                return false;
            }
            unsafe { narrow_f16(values, out) };
        } else {
            bf16!(narrow_bf16(values, out));
        }
        true
    }

    #[target_feature(enable = "neon,fp16")]
    unsafe fn narrow_f16(values: &[f32], out: &mut [u16]) {
        for i in (0..values.len()).step_by(LANES) {
            let (lo, hi) = unsafe { load_f32(values, i) };
            unsafe { store(out, i, f16_narrow(lo, hi)) };
        }
    }


    // ---- accumulations -------------------------------------------------------------

    pub(super) fn dot<C: Compact>(a: &[C], b: &[C]) -> Option<f32> {
        let (a, b) = (lanes(a), lanes(b));
        if same::<C, f16>() {
            if !fp16() {
                return None;
            }
            Some(if fhm() {
                unsafe { dot_f16_fmlal(a, b) }
            } else {
                unsafe { dot_f16_widened(a, b) }
            })
        } else if bf16_instructions() {
            Some(unsafe { dot_bf16_fmlal(a, b) })
        } else {
            Some(unsafe { dot_bf16_widened(a, b) })
        }
    }

    // The dot products keep four independent accumulator pairs, so the chain
    // of dependent multiply-adds is a quarter as long; a vector's worth of
    // tail goes through the zero-padded load.
    #[target_feature(enable = "neon,fp16,fhm")]
    unsafe fn dot_f16_fmlal(a: &[u16], b: &[u16]) -> f32 {
        let zero = vdupq_n_f32(0.0);
        let mut acc = [(zero, zero); 4];
        let whole = a.len() / (4 * LANES) * (4 * LANES);
        for i in (0..whole).step_by(4 * LANES) {
            for (k, (lo, hi)) in acc.iter_mut().enumerate() {
                let j = i + k * LANES;
                (*lo, *hi) = unsafe { f16_fmlal(*lo, *hi, load(a, j), load(b, j)) };
            }
        }
        for i in (whole..a.len()).step_by(LANES) {
            let (lo, hi) = &mut acc[0];
            (*lo, *hi) = unsafe { f16_fmlal(*lo, *hi, load(a, i), load(b, i)) };
        }
        sum_pairs(acc)
    }

    #[target_feature(enable = "neon,fp16")]
    unsafe fn dot_f16_widened(a: &[u16], b: &[u16]) -> f32 {
        let zero = vdupq_n_f32(0.0);
        let mut acc = [(zero, zero); 4];
        for (n, i) in (0..a.len()).step_by(LANES).enumerate() {
            let ((a_lo, a_hi), (b_lo, b_hi)) =
                unsafe { (f16_widen(load(a, i)), f16_widen(load(b, i))) };
            let (lo, hi) = &mut acc[n % 4];
            *lo = vfmaq_f32(*lo, a_lo, b_lo);
            *hi = vfmaq_f32(*hi, a_hi, b_hi);
        }
        sum_pairs(acc)
    }

    /// The horizontal sum of the accumulator pairs.
    #[target_feature(enable = "neon")]
    #[inline]
    fn sum_pairs(acc: [(float32x4_t, float32x4_t); 4]) -> f32 {
        let pairs = acc.map(|(lo, hi)| vaddq_f32(lo, hi));
        vaddvq_f32(vaddq_f32(vaddq_f32(pairs[0], pairs[1]), vaddq_f32(pairs[2], pairs[3])))
    }

    #[target_feature(enable = "neon,bf16")]
    unsafe fn dot_bf16_fmlal(a: &[u16], b: &[u16]) -> f32 {
        let zero = vdupq_n_f32(0.0);
        let mut acc = [(zero, zero); 4];
        let whole = a.len() / (4 * LANES) * (4 * LANES);
        for i in (0..whole).step_by(4 * LANES) {
            for (k, (even, odd)) in acc.iter_mut().enumerate() {
                let j = i + k * LANES;
                (*even, *odd) = unsafe { bf16_fmlal(*even, *odd, load(a, j), load(b, j)) };
            }
        }
        for i in (whole..a.len()).step_by(LANES) {
            let (even, odd) = &mut acc[0];
            (*even, *odd) = unsafe { bf16_fmlal(*even, *odd, load(a, i), load(b, i)) };
        }
        sum_pairs(acc)
    }

    #[target_feature(enable = "neon")]
    unsafe fn dot_bf16_widened(a: &[u16], b: &[u16]) -> f32 {
        let zero = vdupq_n_f32(0.0);
        let mut acc = [(zero, zero); 4];
        for (n, i) in (0..a.len()).step_by(LANES).enumerate() {
            let ((a_lo, a_hi), (b_lo, b_hi)) =
                unsafe { (bf16_widen(load(a, i)), bf16_widen(load(b, i))) };
            let (lo, hi) = &mut acc[n % 4];
            *lo = vfmaq_f32(*lo, a_lo, b_lo);
            *hi = vfmaq_f32(*hi, a_hi, b_hi);
        }
        sum_pairs(acc)
    }

    /// The `f32` fold of a whole slice. The zero-filled tail of the last vector
    /// would be wrong for `Min` and `Max`, so only whole vectors go through the
    /// lanes and the tail is folded one element at a time.
    pub(super) fn reduce<C: Compact>(values: &[C], op: Reduce) -> Option<f32> {
        if same::<C, f16>() && !fp16() {
            return None;
        }
        let whole = values.len() / LANES * LANES;
        let head = lanes(&values[..whole]);
        let lanes = unsafe {
            if same::<C, f16>() {
                reduce_f16(head, op)
            } else {
                reduce_bf16(head, op)
            }
        };
        Some(
            values[whole..]
                .iter()
                .fold(lanes, |total, &x| op.combine(total, x.widen())),
        )
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn reduce_lanes(
        op: Reduce,
        values: &[u16],
        widen: impl Fn(uint16x8_t) -> (float32x4_t, float32x4_t),
    ) -> f32 {
        let identity = vdupq_n_f32(op.identity::<f32>());
        let (mut lo, mut hi) = (identity, identity);
        for i in (0..values.len()).step_by(LANES) {
            let (x_lo, x_hi) = widen(unsafe { load(values, i) });
            (lo, hi) = match op {
                Reduce::Sum => (vaddq_f32(lo, x_lo), vaddq_f32(hi, x_hi)),
                Reduce::Min => (vminnmq_f32(lo, x_lo), vminnmq_f32(hi, x_hi)),
                Reduce::Max => (vmaxnmq_f32(lo, x_lo), vmaxnmq_f32(hi, x_hi)),
            };
        }
        match op {
            Reduce::Sum => vaddvq_f32(vaddq_f32(lo, hi)),
            Reduce::Min => vminnmvq_f32(vminnmq_f32(lo, hi)),
            Reduce::Max => vmaxnmvq_f32(vmaxnmq_f32(lo, hi)),
        }
    }

    #[target_feature(enable = "neon,fp16")]
    unsafe fn reduce_f16(values: &[u16], op: Reduce) -> f32 {
        unsafe { reduce_lanes(op, values, |x| f16_widen(x)) }
    }

    #[target_feature(enable = "neon")]
    unsafe fn reduce_bf16(values: &[u16], op: Reduce) -> f32 {
        unsafe { reduce_lanes(op, values, |x| bf16_widen(x)) }
    }

    // ---- elementwise -----------------------------------------------------------------

    pub(super) fn binary<C: Compact>(a: &[C], b: &[C], op: BinaryOp, out: &mut [C]) -> bool {
        if op == BinaryOp::Rem {
            return false;
        }
        let (a, b, out) = (lanes(a), lanes(b), lanes_mut(out));
        if same::<C, f16>() {
            if !fp16() {
                return false;
            }
            unsafe { binary_f16(a, b, op, out) };
        } else {
            bf16!(binary_bf16(a, b, op, out));
        }
        true
    }

    #[target_feature(enable = "neon,fp16")]
    unsafe fn binary_f16(a: &[u16], b: &[u16], op: BinaryOp, out: &mut [u16]) {
        for i in (0..out.len()).step_by(LANES) {
            let (x, y) = unsafe { (load(a, i), load(b, i)) };
            let z = unsafe { f16_op(op, x, y) };
            unsafe { store(out, i, z) };
        }
    }

    #[target_feature(enable = "neon,fp16")]
    #[inline]
    unsafe fn f16_op(op: BinaryOp, x: uint16x8_t, y: uint16x8_t) -> uint16x8_t {
        unsafe {
            match op {
                BinaryOp::Add => f16_add(x, y),
                BinaryOp::Sub => f16_sub(x, y),
                BinaryOp::Mul => f16_mul(x, y),
                _ => f16_div(x, y),
            }
        }
    }

    #[target_feature(enable = "neon")]
    #[inline]
    fn f32_op(op: BinaryOp, x: float32x4_t, y: float32x4_t) -> float32x4_t {
        match op {
            BinaryOp::Add => vaddq_f32(x, y),
            BinaryOp::Sub => vsubq_f32(x, y),
            BinaryOp::Mul => vmulq_f32(x, y),
            _ => vdivq_f32(x, y),
        }
    }


    pub(super) fn broadcast<C: Compact>(
        values: &[C],
        scalar: C,
        op: BinaryOp,
        scalar_left: bool,
        out: &mut [C],
    ) -> bool {
        if op == BinaryOp::Rem {
            return false;
        }
        let scalar = bits(scalar);
        let (values, out) = (lanes(values), lanes_mut(out));
        if same::<C, f16>() {
            if !fp16() {
                return false;
            }
            unsafe { broadcast_f16(values, scalar, op, scalar_left, out) };
        } else {
            bf16!(broadcast_bf16(values, scalar, op, scalar_left, out));
        }
        true
    }

    #[target_feature(enable = "neon,fp16")]
    unsafe fn broadcast_f16(
        values: &[u16],
        scalar: u16,
        op: BinaryOp,
        scalar_left: bool,
        out: &mut [u16],
    ) {
        let s = vdupq_n_u16(scalar);
        for i in (0..out.len()).step_by(LANES) {
            let x = unsafe { load(values, i) };
            let z = unsafe {
                if scalar_left {
                    f16_op(op, s, x)
                } else {
                    f16_op(op, x, s)
                }
            };
            unsafe { store(out, i, z) };
        }
    }


    pub(super) fn compare<C: Compact>(a: &[C], b: &[C], op: Compare, out: &mut [C]) -> bool {
        let (a, b, out) = (lanes(a), lanes(b), lanes_mut(out));
        if same::<C, f16>() {
            if !fp16() {
                return false;
            }
            unsafe { compare_f16(a, b, None, op, false, out) };
        } else {
            bf16!(compare_bf16(a, b, None, op, false, out));
        }
        true
    }

    pub(super) fn compare_scalar<C: Compact>(
        values: &[C],
        scalar: C,
        op: Compare,
        scalar_left: bool,
        out: &mut [C],
    ) -> bool {
        let scalar = bits(scalar);
        let (values, out) = (lanes(values), lanes_mut(out));
        if same::<C, f16>() {
            if !fp16() {
                return false;
            }
            unsafe { compare_f16(values, &[], Some(scalar), op, scalar_left, out) };
        } else {
            bf16!(compare_bf16(values, &[], Some(scalar), op, scalar_left, out));
        }
        true
    }

    /// `op(x, y)` on `f16` lanes. The predicates select `1.0`/`0.0` (and `0.5`
    /// for a `MaxShare` tie) through the comparison masks.
    #[target_feature(enable = "neon,fp16")]
    #[inline]
    unsafe fn f16_compare(op: Compare, x: uint16x8_t, y: uint16x8_t) -> uint16x8_t {
        let one = vdupq_n_u16(0x3C00);
        unsafe {
            match op {
                Compare::Min => f16_min(x, y),
                Compare::Max => f16_max(x, y),
                Compare::Less => vandq_u16(f16_greater(y, x), one),
                Compare::LessEqual => vandq_u16(f16_greater_equal(y, x), one),
                Compare::Greater => vandq_u16(f16_greater(x, y), one),
                Compare::GreaterEqual => vandq_u16(f16_greater_equal(x, y), one),
                Compare::MaxShare => {
                    let above = f16_greater(x, y);
                    let below = f16_greater(y, x);
                    let tie = vmvnq_u16(vorrq_u16(above, below));
                    vorrq_u16(vandq_u16(above, one), vandq_u16(tie, vdupq_n_u16(0x3800)))
                }
            }
        }
    }

    #[target_feature(enable = "neon,fp16")]
    unsafe fn compare_f16(
        a: &[u16],
        b: &[u16],
        scalar: Option<u16>,
        op: Compare,
        scalar_left: bool,
        out: &mut [u16],
    ) {
        for i in (0..out.len()).step_by(LANES) {
            let x = unsafe { load(a, i) };
            let z = unsafe {
                match scalar {
                    None => f16_compare(op, x, load(b, i)),
                    Some(s) if scalar_left => f16_compare(op, vdupq_n_u16(s), x),
                    Some(s) => f16_compare(op, x, vdupq_n_u16(s)),
                }
            };
            unsafe { store(out, i, z) };
        }
    }

    /// `op(x, y)` on widened `bf16` lanes, as `f32` values that narrow exactly.
    #[target_feature(enable = "neon")]
    #[inline]
    fn f32_compare(op: Compare, x: float32x4_t, y: float32x4_t) -> float32x4_t {
        let one = vdupq_n_u32(1.0f32.to_bits());
        let select = |mask: uint32x4_t| vreinterpretq_f32_u32(vandq_u32(mask, one));
        match op {
            Compare::Min => vminnmq_f32(x, y),
            Compare::Max => vmaxnmq_f32(x, y),
            Compare::Less => select(vcltq_f32(x, y)),
            Compare::LessEqual => select(vcleq_f32(x, y)),
            Compare::Greater => select(vcgtq_f32(x, y)),
            Compare::GreaterEqual => select(vcgeq_f32(x, y)),
            Compare::MaxShare => {
                let above = vcgtq_f32(x, y);
                let below = vcltq_f32(x, y);
                let tie = vmvnq_u32(vorrq_u32(above, below));
                vreinterpretq_f32_u32(vorrq_u32(
                    vandq_u32(above, one),
                    vandq_u32(tie, vdupq_n_u32(0.5f32.to_bits())),
                ))
            }
        }
    }


    pub(super) fn clamp<C: Compact>(values: &[C], low: C, high: C, out: &mut [C]) -> bool {
        let (low, high) = (bits(low), bits(high));
        let (values, out) = (lanes(values), lanes_mut(out));
        if same::<C, f16>() {
            if !fp16() {
                return false;
            }
            unsafe { clamp_f16(values, low, high, out) };
        } else {
            bf16!(clamp_bf16(values, low, high, out));
        }
        true
    }

    #[target_feature(enable = "neon,fp16")]
    unsafe fn clamp_f16(values: &[u16], low: u16, high: u16, out: &mut [u16]) {
        let (low, high) = (vdupq_n_u16(low), vdupq_n_u16(high));
        for i in (0..out.len()).step_by(LANES) {
            let x = unsafe { load(values, i) };
            unsafe { store(out, i, f16_min(f16_max(x, low), high)) };
        }
    }


    pub(super) fn sqrt<C: Compact>(values: &[C], out: &mut [C]) -> bool {
        let (values, out) = (lanes(values), lanes_mut(out));
        if same::<C, f16>() {
            if !fp16() {
                return false;
            }
            unsafe { sqrt_f16(values, out) };
        } else {
            bf16!(sqrt_bf16(values, out));
        }
        true
    }

    #[target_feature(enable = "neon,fp16")]
    unsafe fn sqrt_f16(values: &[u16], out: &mut [u16]) {
        for i in (0..out.len()).step_by(LANES) {
            unsafe { store(out, i, f16_sqrt(load(values, i))) };
        }
    }


    #[cfg(test)]
    mod tests {
        use half::bf16;

        use super::*;

        /// The vector `bf16` rounding — either one — must agree with the
        /// scalar `half::bf16::from_f32` on every class of input: ties both
        /// ways, carries into the exponent, overflow to infinity, subnormals and
        /// NaNs.
        #[test]
        fn bf16_narrowing_matches_the_scalar_rounding() {
            let mut inputs: Vec<f32> = vec![
                0.0,
                -0.0,
                1.0,
                f32::from_bits(0x3F80_8000),
                f32::from_bits(0x3F81_8000),
                f32::from_bits(0x3F80_8001),
                f32::from_bits(0x7F7F_FFFF),
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NAN,
                f32::from_bits(0x7F80_0001),
                f32::from_bits(0x0000_8000),
                f32::from_bits(0x0001_8000),
                f32::MIN_POSITIVE,
            ];
            let mut state = 0x9E37_79B9u32;
            for _ in 0..4096 {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                inputs.push(f32::from_bits(state));
            }
            let expected: Vec<u16> = inputs.iter().map(|&x| bf16::from_f32(x).to_bits()).collect();
            for software in [true, false] {
                if !software && !bf16_instructions() {
                    continue;
                }
                let mut out = vec![0u16; inputs.len()];
                for i in (0..inputs.len()).step_by(LANES) {
                    let (lo, hi) = unsafe { load_f32(&inputs, i) };
                    let narrowed = unsafe {
                        if software {
                            bf16_narrow_software(lo, hi)
                        } else {
                            bf16_narrow_instruction(lo, hi)
                        }
                    };
                    unsafe { store(&mut out, i, narrowed) };
                }
                assert_eq!(out, expected, "software rounding: {software}");
            }
        }

        /// `FCVTN` must agree with `half::f16::from_f32` on everything but NaN
        /// payloads.
        #[test]
        fn f16_narrowing_matches_the_scalar_rounding() {
            if !fp16() {
                return;
            }
            let mut state = 0x1234_5678u32;
            let mut inputs: Vec<f32> = vec![65504.0, 65520.0, 1e-8, -6.0e-8, 2049.0, 2051.0];
            for _ in 0..4096 {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                inputs.push(f32::from_bits(state));
            }
            let mut out = vec![0u16; inputs.len()];
            unsafe { narrow_f16(&inputs, &mut out) };
            for (&x, &got) in inputs.iter().zip(&out) {
                let want = f16::from_f32(x);
                if want.is_nan() {
                    assert!(f16::from_bits(got).is_nan());
                } else {
                    assert_eq!(got, want.to_bits(), "{x:e}");
                }
            }
        }
    }
}
