//! Host elementwise operations that consume their input.
//!
//! The borrowing operations allocate and zero an output the kernel then
//! overwrites. At a few thousand elements that costs about as much as the
//! kernel. These take the left operand by value, overwrite its allocation, and
//! return it, so only one state is ever materialized.
//!
//! Where the SIMD tier has no in-place kernel, the result is the same as the
//! borrowing operation's:
//!
//! - `f16` and `bf16` keep their out-of-place vector kernels.
//! - Every other element type, and short inputs, run a scalar loop in place.

#[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
use super::simd_dispatch;
use super::{BinaryOp, Compare};
use crate::numbers::Coefficient;

/// The scalar definition of `op`.
fn binary<T: Coefficient>(op: BinaryOp, a: T, b: T) -> T {
    match op {
        BinaryOp::Add => a + b,
        BinaryOp::Sub => a - b,
        BinaryOp::Mul => a * b,
        BinaryOp::Div => a / b,
        BinaryOp::Rem => a % b,
    }
}

/// `a op b`, element by element, over `a`'s allocation. `f` is the scalar
/// definition of `op`.
pub(super) fn zip<T: Coefficient>(
    mut a: Vec<T>,
    b: &[T],
    op: BinaryOp,
    f: impl Fn(T, T) -> T,
) -> Vec<T> {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        if simd_dispatch::elementwise_assign(&mut a, b, op) {
            return a;
        }
        if crate::compact::is_compact::<T>() {
            let mut out = vec![T::zero(); a.len()];
            if simd_dispatch::elementwise(&a, b, op, &mut out) {
                return out;
            }
        }
    }
    let _ = op;
    for (x, &y) in a.iter_mut().zip(b) {
        *x = f(*x, y);
    }
    a
}

/// `values op scalar`, or `scalar op values` when `scalar_left`.
pub(super) fn broadcast<T: Coefficient>(
    mut values: Vec<T>,
    scalar: T,
    op: BinaryOp,
    scalar_left: bool,
) -> Vec<T> {
    #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        if simd_dispatch::broadcast_assign(&mut values, scalar, op, scalar_left) {
            return values;
        }
        if crate::compact::is_compact::<T>() {
            let mut out = vec![T::zero(); values.len()];
            if simd_dispatch::broadcast(&values, scalar, op, scalar_left, &mut out) {
                return out;
            }
        }
    }
    for x in &mut values {
        *x = if scalar_left {
            binary(op, scalar, *x)
        } else {
            binary(op, *x, scalar)
        };
    }
    values
}

/// `op(a, b)`, element by element, over `a`'s allocation. `f` is the scalar
/// definition of `op`.
pub(super) fn compare<T: Coefficient>(
    mut a: Vec<T>,
    b: &[T],
    op: Compare,
    f: impl Fn(T, T) -> T,
) -> Vec<T> {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        if simd_dispatch::compare_assign(&mut a, b, op) {
            return a;
        }
        if crate::compact::is_compact::<T>() {
            let mut out = vec![T::zero(); a.len()];
            if simd_dispatch::compare(&a, b, op, &mut out) {
                return out;
            }
        }
    }
    let _ = op;
    for (x, &y) in a.iter_mut().zip(b) {
        *x = f(*x, y);
    }
    a
}

/// `op(values, scalar)`, or `op(scalar, values)` when `scalar_left`. `f` is
/// the scalar definition of `op`, given the operands in that order.
pub(super) fn compare_scalar<T: Coefficient>(
    mut values: Vec<T>,
    scalar: T,
    op: Compare,
    scalar_left: bool,
    f: impl Fn(T, T) -> T,
) -> Vec<T> {
    #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        if simd_dispatch::compare_scalar_assign(&mut values, scalar, op, scalar_left) {
            return values;
        }
        if crate::compact::is_compact::<T>() {
            let mut out = vec![T::zero(); values.len()];
            if simd_dispatch::compare_scalar(&values, scalar, op, scalar_left, &mut out) {
                return out;
            }
        }
    }
    let _ = op;
    for x in &mut values {
        *x = if scalar_left {
            f(scalar, *x)
        } else {
            f(*x, scalar)
        };
    }
    values
}

/// Confine every element to `[low, high]`. `f` is the scalar definition.
pub(super) fn clamp<T: Coefficient>(
    mut values: Vec<T>,
    low: T,
    high: T,
    f: impl Fn(T) -> T,
) -> Vec<T> {
    #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        if simd_dispatch::clamp_assign(&mut values, low, high) {
            return values;
        }
        if crate::compact::is_compact::<T>() {
            let mut out = vec![T::zero(); values.len()];
            if simd_dispatch::clamp(&values, low, high, &mut out) {
                return out;
            }
        }
    }
    let _ = (low, high);
    for x in &mut values {
        *x = f(*x);
    }
    values
}

/// `f` applied to every element, over the same allocation.
pub(super) fn map<T: Copy>(mut values: Vec<T>, f: impl Fn(T) -> T) -> Vec<T> {
    for x in &mut values {
        *x = f(*x);
    }
    values
}
