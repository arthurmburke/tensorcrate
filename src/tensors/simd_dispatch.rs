//! CPU SIMD tier, above the generic scalar
//! loops. Each entry point downcasts the generic element type to a concrete
//! float via `TypeId` (returning `None` — i.e. defer to scalar — for every
//! other type), then calls the architecture-specific kernels in
//! [`crate::simd`]. The size gates are deliberately small: CPU SIMD has
//! almost no fixed cost.
//!
//! Every entry point takes flat slices and runtime extents, and writes into
//! storage the caller owns, so a dispatched operation allocates nothing
//! beyond its result.

use std::any::TypeId;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};

use super::{BinaryOp, Compare, Reduce};
use crate::numbers::{Coefficient, Complex};

// Below these lengths the generic scalar loop is already fine (and the
// reinterpret/dispatch bookkeeping is not worth it).
const MIN_ELEMENTS: usize = 16;
const MIN_MATMUL_OPS: usize = super::HOST_MATMUL_DISPATCH_OPS;
const MIN_FFT_LENGTH: usize = 8;

fn is<T: 'static, U: 'static>() -> bool {
    TypeId::of::<T>() == TypeId::of::<U>()
}

/// Reinterpret `&[T]` as `&[U]` when `T` is exactly `U`. SAFETY rests on the
/// `TypeId` equality: identical type ⇒ identical layout and lifetime.
unsafe fn as_slice<T: 'static, U: 'static>(values: &[T]) -> Option<&[U]> {
    if !is::<T, U>() {
        return None;
    }
    Some(unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<U>(), values.len()) })
}

unsafe fn as_slice_mut<T: 'static, U: 'static>(values: &mut [T]) -> Option<&mut [U]> {
    if !is::<T, U>() {
        return None;
    }
    Some(unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<U>(), values.len()) })
}

fn from_f32<T: Copy + 'static>(v: f32) -> T {
    unsafe { std::ptr::read((&v as *const f32).cast::<T>()) }
}
fn from_f64<T: Copy + 'static>(v: f64) -> T {
    unsafe { std::ptr::read((&v as *const f64).cast::<T>()) }
}

/// Elements per thread for the elementwise kernels, which are bound by memory
/// bandwidth: below twice this a kernel runs on its calling thread alone.
const PARALLEL_GRAIN: usize = 32 * 1024;

/// Run `kernel` over windows of `out` and the same windows of its operands, on
/// several threads when `out` is long enough. Whether the SIMD path ran, which
/// depends only on the element type and the operation, so is the same for
/// every window.
fn split<T: 'static>(out: &mut [T], kernel: impl Fn(Range<usize>, &mut [T]) -> bool) -> bool {
    if !crate::parallel::plain_float::<T>() || out.len() < 2 * PARALLEL_GRAIN {
        return kernel(0..out.len(), out);
    }
    let ran = AtomicBool::new(true);
    // SAFETY: `T` is a plain float, and the operands the kernel captures are
    // slices of it.
    unsafe {
        crate::parallel::for_slices_unchecked(out, PARALLEL_GRAIN, |start, window| {
            if !kernel(start..start + window.len(), window) {
                ran.store(false, Ordering::Relaxed);
            }
        });
    }
    ran.into_inner()
}

pub fn dot<T: Coefficient>(a: &[T], b: &[T]) -> Option<T> {
    if a.len() < MIN_ELEMENTS {
        return None;
    }
    debug_assert_eq!(a.len(), b.len());
    unsafe {
        if let (Some(a), Some(b)) = (as_slice::<T, f32>(a), as_slice::<T, f32>(b)) {
            return Some(from_f32(crate::simd::f32k::dot(a, b)));
        }
        if let (Some(a), Some(b)) = (as_slice::<T, f64>(a), as_slice::<T, f64>(b)) {
            return Some(from_f64(crate::simd::f64k::dot(a, b)));
        }
    }
    None
}

/// Writes `a op b` into `out`, returning whether the SIMD path ran.
pub fn elementwise<T: Coefficient>(a: &[T], b: &[T], op: BinaryOp, out: &mut [T]) -> bool {
    if a.len() < MIN_ELEMENTS || op == BinaryOp::Rem {
        return false;
    }
    debug_assert!(b.len() == a.len() && out.len() == a.len());
    split(out, |range, out| {
        elementwise_window(&a[range.clone()], &b[range], op, out)
    })
}

fn elementwise_window<T: Coefficient>(a: &[T], b: &[T], op: BinaryOp, out: &mut [T]) -> bool {
    // The compact floats have kernels of their own: native FP16 lanes, and
    // BF16 through `f32` with one rounding.
    if crate::compact::elementwise(a, b, op, out) {
        return true;
    }
    unsafe {
        if let (Some(a), Some(b), Some(out)) = (
            as_slice::<T, f32>(a),
            as_slice::<T, f32>(b),
            as_slice_mut::<T, f32>(out),
        ) {
            crate::simd::f32k::elementwise(a, b, op, out);
            return true;
        }
        if let (Some(a), Some(b), Some(out)) = (
            as_slice::<T, f64>(a),
            as_slice::<T, f64>(b),
            as_slice_mut::<T, f64>(out),
        ) {
            crate::simd::f64k::elementwise(a, b, op, out);
            return true;
        }
    }
    false
}

/// Writes the tensor/scalar broadcast into `out`, returning whether the SIMD
/// path ran. Allocation-free for the same reason as [`elementwise`].
pub fn broadcast<T: Coefficient>(
    values: &[T],
    scalar: T,
    op: BinaryOp,
    scalar_left: bool,
    out: &mut [T],
) -> bool {
    if values.len() < MIN_ELEMENTS || op == BinaryOp::Rem {
        return false;
    }
    debug_assert_eq!(out.len(), values.len());
    split(out, |range, out| {
        broadcast_window(&values[range], scalar, op, scalar_left, out)
    })
}

fn broadcast_window<T: Coefficient>(
    values: &[T],
    scalar: T,
    op: BinaryOp,
    scalar_left: bool,
    out: &mut [T],
) -> bool {
    if crate::compact::broadcast(values, scalar, op, scalar_left, out) {
        return true;
    }
    unsafe {
        if let (Some(v), Some(s), Some(out)) = (
            as_slice::<T, f32>(values),
            as_slice::<T, f32>(std::slice::from_ref(&scalar)),
            as_slice_mut::<T, f32>(out),
        ) {
            crate::simd::f32k::broadcast(v, s[0], op, scalar_left, out);
            return true;
        }
        if let (Some(v), Some(s), Some(out)) = (
            as_slice::<T, f64>(values),
            as_slice::<T, f64>(std::slice::from_ref(&scalar)),
            as_slice_mut::<T, f64>(out),
        ) {
            crate::simd::f64k::broadcast(v, s[0], op, scalar_left, out);
            return true;
        }
    }
    false
}

/// Writes the elementwise comparison into `out`, returning whether the SIMD
/// path ran.
pub fn compare<T: Coefficient>(a: &[T], b: &[T], op: Compare, out: &mut [T]) -> bool {
    if a.len() < MIN_ELEMENTS {
        return false;
    }
    debug_assert!(b.len() == a.len() && out.len() == a.len());
    split(out, |range, out| {
        compare_window(&a[range.clone()], &b[range], op, out)
    })
}

fn compare_window<T: Coefficient>(a: &[T], b: &[T], op: Compare, out: &mut [T]) -> bool {
    if crate::compact::compare(a, b, op, out) {
        return true;
    }
    unsafe {
        if let (Some(a), Some(b), Some(out)) = (
            as_slice::<T, f32>(a),
            as_slice::<T, f32>(b),
            as_slice_mut::<T, f32>(out),
        ) {
            crate::simd::f32k::compare(a, b, op, out);
            return true;
        }
        if let (Some(a), Some(b), Some(out)) = (
            as_slice::<T, f64>(a),
            as_slice::<T, f64>(b),
            as_slice_mut::<T, f64>(out),
        ) {
            crate::simd::f64k::compare(a, b, op, out);
            return true;
        }
    }
    false
}

/// Writes the comparison against a splatted scalar into `out`, returning
/// whether the SIMD path ran.
pub fn compare_scalar<T: Coefficient>(
    values: &[T],
    scalar: T,
    op: Compare,
    scalar_left: bool,
    out: &mut [T],
) -> bool {
    if values.len() < MIN_ELEMENTS {
        return false;
    }
    debug_assert_eq!(out.len(), values.len());
    split(out, |range, out| {
        compare_scalar_window(&values[range], scalar, op, scalar_left, out)
    })
}

fn compare_scalar_window<T: Coefficient>(
    values: &[T],
    scalar: T,
    op: Compare,
    scalar_left: bool,
    out: &mut [T],
) -> bool {
    if crate::compact::compare_scalar(values, scalar, op, scalar_left, out) {
        return true;
    }
    unsafe {
        if let (Some(v), Some(s), Some(out)) = (
            as_slice::<T, f32>(values),
            as_slice::<T, f32>(std::slice::from_ref(&scalar)),
            as_slice_mut::<T, f32>(out),
        ) {
            crate::simd::f32k::compare_scalar(v, s[0], op, scalar_left, out);
            return true;
        }
        if let (Some(v), Some(s), Some(out)) = (
            as_slice::<T, f64>(values),
            as_slice::<T, f64>(std::slice::from_ref(&scalar)),
            as_slice_mut::<T, f64>(out),
        ) {
            crate::simd::f64k::compare_scalar(v, s[0], op, scalar_left, out);
            return true;
        }
    }
    false
}

/// Writes the clamped values into `out`, returning whether the SIMD path
/// ran.
pub fn clamp<T: Coefficient>(values: &[T], low: T, high: T, out: &mut [T]) -> bool {
    if values.len() < MIN_ELEMENTS {
        return false;
    }
    debug_assert_eq!(out.len(), values.len());
    split(out, |range, out| {
        clamp_window(&values[range], low, high, out)
    })
}

fn clamp_window<T: Coefficient>(values: &[T], low: T, high: T, out: &mut [T]) -> bool {
    if crate::compact::clamp(values, low, high, out) {
        return true;
    }
    unsafe {
        if let (Some(v), Some(low), Some(high), Some(out)) = (
            as_slice::<T, f32>(values),
            as_slice::<T, f32>(std::slice::from_ref(&low)),
            as_slice::<T, f32>(std::slice::from_ref(&high)),
            as_slice_mut::<T, f32>(out),
        ) {
            crate::simd::f32k::clamp(v, low[0], high[0], out);
            return true;
        }
        if let (Some(v), Some(low), Some(high), Some(out)) = (
            as_slice::<T, f64>(values),
            as_slice::<T, f64>(std::slice::from_ref(&low)),
            as_slice::<T, f64>(std::slice::from_ref(&high)),
            as_slice_mut::<T, f64>(out),
        ) {
            crate::simd::f64k::clamp(v, low[0], high[0], out);
            return true;
        }
    }
    false
}

// ---- in place -----------------------------------------------------------------
//
// The same operations with the result written over the left operand, so a
// caller that owns it never allocates. Each returns whether the SIMD path ran,
// and touches nothing when it did not. The compact floats have no in-place
// kernels: they answer `false`, and the caller decides what to do about it.

/// `a = a op b`, returning whether the SIMD path ran.
pub fn elementwise_assign<T: Coefficient>(a: &mut [T], b: &[T], op: BinaryOp) -> bool {
    if a.len() < MIN_ELEMENTS || op == BinaryOp::Rem {
        return false;
    }
    debug_assert_eq!(a.len(), b.len());
    split(a, |range, a| {
        // SAFETY: the `TypeId` check inside `as_slice_mut` makes `T` the float.
        unsafe {
            if let (Some(a), Some(b)) = (
                as_slice_mut::<T, f32>(a),
                as_slice::<T, f32>(&b[range.clone()]),
            ) {
                crate::simd::f32k::elementwise_assign(a, b, op);
                return true;
            }
            if let (Some(a), Some(b)) = (as_slice_mut::<T, f64>(a), as_slice::<T, f64>(&b[range])) {
                crate::simd::f64k::elementwise_assign(a, b, op);
                return true;
            }
        }
        false
    })
}

/// `values = values op scalar` (or `scalar op values`), returning whether the
/// SIMD path ran.
pub fn broadcast_assign<T: Coefficient>(
    values: &mut [T],
    scalar: T,
    op: BinaryOp,
    scalar_left: bool,
) -> bool {
    if values.len() < MIN_ELEMENTS || op == BinaryOp::Rem {
        return false;
    }
    split(values, |_, values| {
        // SAFETY: as in `elementwise_assign`.
        unsafe {
            if let (Some(v), Some(s)) = (
                as_slice_mut::<T, f32>(values),
                as_slice::<T, f32>(std::slice::from_ref(&scalar)),
            ) {
                crate::simd::f32k::broadcast_assign(v, s[0], op, scalar_left);
                return true;
            }
            if let (Some(v), Some(s)) = (
                as_slice_mut::<T, f64>(values),
                as_slice::<T, f64>(std::slice::from_ref(&scalar)),
            ) {
                crate::simd::f64k::broadcast_assign(v, s[0], op, scalar_left);
                return true;
            }
        }
        false
    })
}

/// `a = compare(a, b)`, returning whether the SIMD path ran.
pub fn compare_assign<T: Coefficient>(a: &mut [T], b: &[T], op: Compare) -> bool {
    if a.len() < MIN_ELEMENTS {
        return false;
    }
    debug_assert_eq!(a.len(), b.len());
    split(a, |range, a| {
        // SAFETY: as in `elementwise_assign`.
        unsafe {
            if let (Some(a), Some(b)) = (
                as_slice_mut::<T, f32>(a),
                as_slice::<T, f32>(&b[range.clone()]),
            ) {
                crate::simd::f32k::compare_assign(a, b, op);
                return true;
            }
            if let (Some(a), Some(b)) = (as_slice_mut::<T, f64>(a), as_slice::<T, f64>(&b[range])) {
                crate::simd::f64k::compare_assign(a, b, op);
                return true;
            }
        }
        false
    })
}

/// `values = compare(values, scalar)`, returning whether the SIMD path ran.
pub fn compare_scalar_assign<T: Coefficient>(
    values: &mut [T],
    scalar: T,
    op: Compare,
    scalar_left: bool,
) -> bool {
    if values.len() < MIN_ELEMENTS {
        return false;
    }
    split(values, |_, values| {
        // SAFETY: as in `elementwise_assign`.
        unsafe {
            if let (Some(v), Some(s)) = (
                as_slice_mut::<T, f32>(values),
                as_slice::<T, f32>(std::slice::from_ref(&scalar)),
            ) {
                crate::simd::f32k::compare_scalar_assign(v, s[0], op, scalar_left);
                return true;
            }
            if let (Some(v), Some(s)) = (
                as_slice_mut::<T, f64>(values),
                as_slice::<T, f64>(std::slice::from_ref(&scalar)),
            ) {
                crate::simd::f64k::compare_scalar_assign(v, s[0], op, scalar_left);
                return true;
            }
        }
        false
    })
}

/// `values = min(max(values, low), high)`, returning whether the SIMD path ran.
pub fn clamp_assign<T: Coefficient>(values: &mut [T], low: T, high: T) -> bool {
    if values.len() < MIN_ELEMENTS {
        return false;
    }
    split(values, |_, values| {
        // SAFETY: as in `elementwise_assign`.
        unsafe {
            if let (Some(v), Some(low), Some(high)) = (
                as_slice_mut::<T, f32>(values),
                as_slice::<T, f32>(std::slice::from_ref(&low)),
                as_slice::<T, f32>(std::slice::from_ref(&high)),
            ) {
                crate::simd::f32k::clamp_assign(v, low[0], high[0]);
                return true;
            }
            if let (Some(v), Some(low), Some(high)) = (
                as_slice_mut::<T, f64>(values),
                as_slice::<T, f64>(std::slice::from_ref(&low)),
                as_slice::<T, f64>(std::slice::from_ref(&high)),
            ) {
                crate::simd::f64k::clamp_assign(v, low[0], high[0]);
                return true;
            }
        }
        false
    })
}

/// The whole-slice fold, or `None` for an element type the kernels do not
/// cover (and for slices too short to be worth the dispatch).
pub fn reduce<T: Coefficient>(values: &[T], op: Reduce) -> Option<T> {
    if values.len() < MIN_ELEMENTS {
        return None;
    }
    unsafe {
        if let Some(v) = as_slice::<T, f32>(values) {
            return Some(from_f32(crate::simd::f32k::reduce(v, op)));
        }
        if let Some(v) = as_slice::<T, f64>(values) {
            return Some(from_f64(crate::simd::f64k::reduce(v, op)));
        }
    }
    None
}

/// `Σ(xᵢ − mean)²`, or `None` for an element type the kernels do not cover.
pub fn sum_squared_deviations<T: Coefficient>(values: &[T], mean: T) -> Option<T> {
    if values.len() < MIN_ELEMENTS {
        return None;
    }
    unsafe {
        if let (Some(v), Some(mean)) = (
            as_slice::<T, f32>(values),
            as_slice::<T, f32>(std::slice::from_ref(&mean)),
        ) {
            return Some(from_f32(crate::simd::f32k::sum_squared_deviations(
                v, mean[0],
            )));
        }
        if let (Some(v), Some(mean)) = (
            as_slice::<T, f64>(values),
            as_slice::<T, f64>(std::slice::from_ref(&mean)),
        ) {
            return Some(from_f64(crate::simd::f64k::sum_squared_deviations(
                v, mean[0],
            )));
        }
    }
    None
}

/// `totals += values`, returning whether the SIMD path ran.
pub fn accumulate<T: Coefficient>(totals: &mut [T], values: &[T]) -> bool {
    if totals.len() < MIN_ELEMENTS {
        return false;
    }
    debug_assert_eq!(values.len(), totals.len());
    unsafe {
        if let (Some(v), Some(totals)) =
            (as_slice::<T, f32>(values), as_slice_mut::<T, f32>(totals))
        {
            crate::simd::f32k::accumulate(totals, v);
            return true;
        }
        if let (Some(v), Some(totals)) =
            (as_slice::<T, f64>(values), as_slice_mut::<T, f64>(totals))
        {
            crate::simd::f64k::accumulate(totals, v);
            return true;
        }
    }
    false
}

/// `totals += (values − means)²`, returning whether the SIMD path ran.
pub fn accumulate_squared_deviations<T: Coefficient>(
    totals: &mut [T],
    values: &[T],
    means: &[T],
) -> bool {
    if totals.len() < MIN_ELEMENTS {
        return false;
    }
    debug_assert!(values.len() == totals.len() && means.len() == totals.len());
    unsafe {
        if let (Some(v), Some(m), Some(totals)) = (
            as_slice::<T, f32>(values),
            as_slice::<T, f32>(means),
            as_slice_mut::<T, f32>(totals),
        ) {
            crate::simd::f32k::accumulate_squared_deviations(totals, v, m);
            return true;
        }
        if let (Some(v), Some(m), Some(totals)) = (
            as_slice::<T, f64>(values),
            as_slice::<T, f64>(means),
            as_slice_mut::<T, f64>(totals),
        ) {
            crate::simd::f64k::accumulate_squared_deviations(totals, v, m);
            return true;
        }
    }
    false
}

/// Writes `a·b` into `out`, returning whether the SIMD path ran.
///
/// `out` is the caller's final storage, so the product lands in its
/// destination directly: a scratch buffer and a copy out would cost about
/// as much as the arithmetic itself for small products.
pub fn matmul<T: Coefficient>(
    a: &[T],
    b: &[T],
    rows: usize,
    inner: usize,
    cols: usize,
    out: &mut [T],
) -> bool {
    if rows.saturating_mul(inner).saturating_mul(cols) < MIN_MATMUL_OPS {
        return false;
    }
    unsafe {
        if let (Some(a), Some(b), Some(out)) = (
            as_slice::<T, f32>(a),
            as_slice::<T, f32>(b),
            as_slice_mut::<T, f32>(out),
        ) {
            crate::simd::f32k::matmul(a, b, rows, inner, cols, out);
            return true;
        }
        if let (Some(a), Some(b), Some(out)) = (
            as_slice::<T, f64>(a),
            as_slice::<T, f64>(b),
            as_slice_mut::<T, f64>(out),
        ) {
            crate::simd::f64k::matmul(a, b, rows, inner, cols, out);
            return true;
        }
    }
    false
}

pub fn matmul_add<T: Coefficient>(
    a: &[T],
    b: &[T],
    rows: usize,
    inner: usize,
    cols: usize,
    addend: &mut [T],
) -> bool {
    if rows.saturating_mul(inner).saturating_mul(cols) < MIN_MATMUL_OPS {
        return false;
    }
    unsafe {
        if let (Some(a), Some(b), Some(addend)) = (
            as_slice::<T, f32>(a),
            as_slice::<T, f32>(b),
            as_slice_mut::<T, f32>(addend),
        ) {
            crate::simd::f32k::matmul_accumulate(a, b, rows, inner, cols, addend);
            return true;
        }
        if let (Some(a), Some(b), Some(addend)) = (
            as_slice::<T, f64>(a),
            as_slice::<T, f64>(b),
            as_slice_mut::<T, f64>(addend),
        ) {
            crate::simd::f64k::matmul_accumulate(a, b, rows, inner, cols, addend);
            return true;
        }
    }
    false
}

/// Writes `matrix·vector` into `out`, returning whether the SIMD path ran.
///
/// Row-times-vector: each output is a dot product of a matrix row with the
/// vector, so the per-row reduction kernel is the right shape here.
pub fn matvec<T: Coefficient>(
    matrix: &[T],
    vector: &[T],
    rows: usize,
    cols: usize,
    out: &mut [T],
) -> bool {
    if rows.saturating_mul(cols) < MIN_MATMUL_OPS {
        return false;
    }
    unsafe {
        if let (Some(m), Some(v), Some(out)) = (
            as_slice::<T, f32>(matrix),
            as_slice::<T, f32>(vector),
            as_slice_mut::<T, f32>(out),
        ) {
            for (row, slot) in out.iter_mut().enumerate() {
                *slot = crate::simd::f32k::dot(&m[row * cols..row * cols + cols], v);
            }
            return true;
        }
        if let (Some(m), Some(v), Some(out)) = (
            as_slice::<T, f64>(matrix),
            as_slice::<T, f64>(vector),
            as_slice_mut::<T, f64>(out),
        ) {
            for (row, slot) in out.iter_mut().enumerate() {
                *slot = crate::simd::f64k::dot(&m[row * cols..row * cols + cols], v);
            }
            return true;
        }
    }
    false
}

/// Adds `matrix·vector` into `addend`, returning whether the SIMD path ran.
pub fn matvec_add<T: Coefficient>(
    matrix: &[T],
    vector: &[T],
    rows: usize,
    cols: usize,
    addend: &mut [T],
) -> bool {
    if rows.saturating_mul(cols) < MIN_MATMUL_OPS {
        return false;
    }
    unsafe {
        if let (Some(m), Some(v), Some(addend)) = (
            as_slice::<T, f32>(matrix),
            as_slice::<T, f32>(vector),
            as_slice_mut::<T, f32>(addend),
        ) {
            for (row, slot) in addend.iter_mut().enumerate() {
                *slot += crate::simd::f32k::dot(&m[row * cols..row * cols + cols], v);
            }
            return true;
        }
        if let (Some(m), Some(v), Some(addend)) = (
            as_slice::<T, f64>(matrix),
            as_slice::<T, f64>(vector),
            as_slice_mut::<T, f64>(addend),
        ) {
            for (row, slot) in addend.iter_mut().enumerate() {
                *slot += crate::simd::f64k::dot(&m[row * cols..row * cols + cols], v);
            }
            return true;
        }
    }
    false
}

/// Writes `vectorᵀ·matrix` into `out`, returning whether the SIMD path ran.
pub fn vecmat<T: Coefficient>(
    vector: &[T],
    matrix: &[T],
    rows: usize,
    cols: usize,
    out: &mut [T],
) -> bool {
    if rows.saturating_mul(cols) < MIN_MATMUL_OPS {
        return false;
    }
    // (1×R)·(R×C): the broadcast-A matmul vectorizes across the C columns.
    matmul(vector, matrix, 1, rows, cols, out)
}

/// In-place radix-2 FFT for power-of-two `f32` lengths. Returns `false` (so
/// the caller keeps the generic path) for every other element type or shape.
/// `direction` is `-1.0` forward, `+1.0` inverse; normalization stays with
/// the caller.
pub fn radix2_fft<T: Coefficient>(output: &mut [Complex<T>], direction: f64) -> bool {
    let n = output.len();
    if n < MIN_FFT_LENGTH || !n.is_power_of_two() || !is::<T, f32>() {
        return false;
    }
    // SAFETY: T is f32 and `Complex` is `#[repr(C)]`, so the buffer is
    // exactly `[re, im, …]` — `2*n` contiguous f32.
    let buf = unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr().cast::<f32>(), 2 * n) };
    crate::simd::fft_f32::radix2(buf, n, direction as f32);
    true
}
