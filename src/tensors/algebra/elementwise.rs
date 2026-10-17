//! Scaling, minima and maxima, clamping and comparison, element by element.
//!
//! Each trait takes its other operand as a type parameter. A tensor reference
//! as `Rhs` pairs elements; a scalar as `Rhs` applies it to every element. A
//! scalar on the left is the scalar as `Self` with the tensor reference as
//! `Rhs`.
//!
//! The method names match the primitive types' own `min`, `max` and `clamp`.
//! The primitives therefore do not implement these traits: their inherent
//! methods and [`Ord`] already cover them, and a second impl would make
//! `3.min(4)` ambiguous.

use crate::tensors::Compare;

/// Multiplication of every element by one scalar.
pub trait Scale<S> {
    /// The scaled result.
    type Output;

    /// Multiply every element by `factor`.
    fn scale(self, factor: S) -> Self::Output;
}

/// The elementwise minimum and maximum.
pub trait MinMax<Rhs = Self> {
    /// The result, with the tensor operand's shape.
    type Output;

    /// The lesser of each pair.
    ///
    /// # Panics
    ///
    /// If two tensor operands have different shapes.
    fn min(self, rhs: Rhs) -> Self::Output;

    /// The greater of each pair. `max(0.0)` is a relu.
    ///
    /// # Panics
    ///
    /// If two tensor operands have different shapes.
    fn max(self, rhs: Rhs) -> Self::Output;
}

/// Confinement of every element to a closed interval, in one pass.
pub trait Clamp<S> {
    /// The clamped result.
    type Output;

    /// Confine every element to `[low, high]`.
    ///
    /// # Panics
    ///
    /// If `low > high`.
    fn clamp(self, low: S, high: S) -> Self::Output;
}

/// Elementwise comparison under one of the [`Compare`] predicates.
///
/// The predicates give `1` where they hold and `0` where they do not, in the
/// element type.
pub trait Comparison<Rhs = Self> {
    /// The result, with the tensor operand's shape.
    type Output;

    /// Apply `op` to each pair, with `self` as the left operand.
    ///
    /// # Panics
    ///
    /// If two tensor operands have different shapes.
    fn compare(self, rhs: Rhs, op: Compare) -> Self::Output;
}
