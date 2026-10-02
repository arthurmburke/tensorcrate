//! The host kernels that read their operands through strided layouts: a
//! broadcast binary operation, an axis reduction and an axis arg-reduction.
//!
//! They are the [`Host`](super::Host) implementations of the matching
//! [`Kernels`](super::Kernels) methods, and the CPU path the `Metal` ones fall
//! back to over shared memory. Each walks its layout with
//! [`for_each_run`](super::layout::for_each_run), so an operand is read in
//! place whatever its strides — a zero stride, which is how a broadcast
//! repeats an element, included — and nothing is copied into order first.

use std::any::TypeId;

use super::kernels::{AxisReduction, Pairwise};
use super::layout::{contiguous_strides, for_each_run};
use super::{BinaryOp, Compare, Reduce};
use crate::numbers::Real;

/// A strided layout over a slice: element `[i₀, …]` is
/// `values[offset + Σ iₖ·strides[k]]`.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Layout<'a, T> {
    pub(crate) values: &'a [T],
    pub(crate) shape: &'a [usize],
    pub(crate) strides: &'a [usize],
    pub(crate) offset: usize,
}

/// `op(a, b)` over the two layouts' common shape, in row-major order.
pub(crate) fn binary<T: Real>(a: Layout<'_, T>, b: Layout<'_, T>, op: Pairwise) -> Vec<T> {
    assert_eq!(
        a.shape, b.shape,
        "strided binary: the operands' shapes differ"
    );
    // One instantiation per operation, so each inner loop is a plain one.
    match op {
        Pairwise::Arithmetic(BinaryOp::Add) => zip(a, b, |x, y| x + y),
        Pairwise::Arithmetic(BinaryOp::Sub) => zip(a, b, |x, y| x - y),
        Pairwise::Arithmetic(BinaryOp::Mul) => zip(a, b, |x, y| x * y),
        Pairwise::Arithmetic(BinaryOp::Div) => zip(a, b, |x, y| x / y),
        Pairwise::Arithmetic(BinaryOp::Rem) => zip(a, b, |x, y| x % y),
        Pairwise::Compare(Compare::Min) => zip(a, b, |x, y| Compare::Min.value(x, y)),
        Pairwise::Compare(Compare::Max) => zip(a, b, |x, y| Compare::Max.value(x, y)),
        Pairwise::Compare(compare) => zip(a, b, |x, y| compare.value(x, y)),
        Pairwise::Power => zip(a, b, |x, y| Pairwise::Power.value(x, y)),
    }
}

fn zip<T: Copy>(a: Layout<'_, T>, b: Layout<'_, T>, f: impl Fn(T, T) -> T) -> Vec<T> {
    let mut out = Vec::with_capacity(a.shape.iter().product());
    for_each_run(
        a.shape,
        [a.offset, b.offset],
        [a.strides, b.strides],
        |[x, y], len, steps| {
            let (left, right) = (a.values, b.values);
            match steps {
                [1, 1] => out.extend(
                    left[x..x + len]
                        .iter()
                        .zip(&right[y..y + len])
                        .map(|(&l, &r)| f(l, r)),
                ),
                [1, 0] => {
                    let r = right[y];
                    out.extend(left[x..x + len].iter().map(|&l| f(l, r)));
                }
                [0, 1] => {
                    let l = left[x];
                    out.extend(right[y..y + len].iter().map(|&r| f(l, r)));
                }
                [step_x, step_y] => {
                    out.extend((0..len).map(|k| f(left[x + k * step_x], right[y + k * step_y])))
                }
            }
        },
    );
    out
}

/// The strides that place each element of `shape` at its result when `axes`
/// are folded away: the row-major strides of the kept axes, and zero along a
/// folded one.
fn result_strides(shape: &[usize], axes: &[usize]) -> Vec<usize> {
    let kept = (0..shape.len())
        .filter(|axis| !axes.contains(axis))
        .map(|axis| shape[axis])
        .collect::<Vec<_>>();
    let kept_strides = contiguous_strides(&kept);
    let mut next = 0;
    (0..shape.len())
        .map(|axis| {
            if axes.contains(&axis) {
                0
            } else {
                next += 1;
                kept_strides[next - 1]
            }
        })
        .collect()
}

/// `input` folded over `axes` by `op`, one result per index of the other
/// axes. `f64` accumulates in `f64`, every other type in `f32`, and each
/// result rounds to `T` once.
pub(crate) fn reduce<T: Real>(input: Layout<'_, T>, axes: &[usize], op: AxisReduction) -> Vec<T> {
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        reduce_in::<T, f64>(input, axes, op)
    } else {
        reduce_in::<T, f32>(input, axes, op)
    }
}

/// [`reduce`] with accumulator type `A`.
///
/// The input is walked once in its own row-major order, each element folded
/// into the accumulator of the result it belongs to, so every result folds
/// its slice in the slice's row-major order and the input is read in the
/// order it lies whichever axes fold.
fn reduce_in<T: Real, A: Real>(input: Layout<'_, T>, axes: &[usize], op: AxisReduction) -> Vec<T> {
    let to = result_strides(input.shape, axes);
    let results = (0..input.shape.len())
        .filter(|axis| !axes.contains(axis))
        .map(|axis| input.shape[axis])
        .product::<usize>();
    let depth = axes
        .iter()
        .map(|&axis| input.shape[axis])
        .product::<usize>();
    let count = A::from_f64(depth as f64);
    let means = || {
        let mut totals = vec![A::zero(); results];
        fold(input, &to, &mut totals, |total, x, _| total + x);
        totals
            .into_iter()
            .map(|total| total / count)
            .collect::<Vec<_>>()
    };
    let totals = match op {
        AxisReduction::Fold(reduce) => {
            let mut totals = vec![reduce.identity(); results];
            fold(input, &to, &mut totals, |total, x, _| {
                reduce.combine(total, x)
            });
            totals
        }
        AxisReduction::Mean => means(),
        AxisReduction::Variance { divisor } => {
            let means = means();
            let mut totals = vec![A::zero(); results];
            fold(input, &to, &mut totals, |total, x, slot| {
                let deviation = x - means[slot];
                total + deviation * deviation
            });
            let divisor = A::from_f64(divisor as f64);
            totals.into_iter().map(|total| total / divisor).collect()
        }
    };
    totals
        .into_iter()
        .map(|total| T::from_f64(total.into_f64()))
        .collect()
}

/// Fold every element of `input`, widened to `A`, into the accumulator `to`
/// places it at: `accumulators[slot] = combine(accumulators[slot], x, slot)`.
fn fold<T: Real, A: Real>(
    input: Layout<'_, T>,
    to: &[usize],
    accumulators: &mut [A],
    combine: impl Fn(A, A, usize) -> A,
) {
    for_each_run(
        input.shape,
        [input.offset, 0],
        [input.strides, to],
        |[from, at], len, [step, result_step]| {
            for k in 0..len {
                let slot = at + k * result_step;
                let x = A::from_f64(input.values[from + k * step].into_f64());
                accumulators[slot] = combine(accumulators[slot], x, slot);
            }
        },
    );
}

/// Whether `x`, at a later position, replaces `best` as the extreme `op`
/// looks for: it is a number, and `best` is a NaN or `x` is strictly past it.
fn replaces<T: Real>(op: Reduce, x: T, best: T) -> bool {
    if x.is_nan() {
        return false;
    }
    best.is_nan()
        || match op {
            Reduce::Min => x < best,
            _ => x > best,
        }
}

/// The position along `axis` of each slice's extreme element — see
/// [`Kernels::arg_reduce`](super::Kernels::arg_reduce).
pub(crate) fn arg_reduce<T: Real>(input: Layout<'_, T>, axis: usize, op: Reduce) -> Vec<u32> {
    let to = result_strides(input.shape, &[axis]);
    // A third operand whose only stride is along `axis` walks the coordinate.
    let mut along = vec![0; input.shape.len()];
    along[axis] = 1;
    let results = input.shape.iter().product::<usize>() / input.shape[axis];
    let mut best = vec![T::nan(); results];
    let mut positions = vec![0u32; results];
    for_each_run(
        input.shape,
        [input.offset, 0, 0],
        [input.strides, &to, &along],
        |[from, at, coordinate], len, [step, result_step, coordinate_step]| {
            for k in 0..len {
                let slot = at + k * result_step;
                let x = input.values[from + k * step];
                if replaces(op, x, best[slot]) {
                    best[slot] = x;
                    positions[slot] = (coordinate + k * coordinate_step) as u32;
                }
            }
        },
    );
    positions
}
