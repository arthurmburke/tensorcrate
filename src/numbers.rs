//! The numeric types the language is built on.
//!
//! Everything here is a concrete Rust type — there is no runtime type tag. The
//! [`Coefficient`] trait is what a numeric element must satisfy, and it is
//! implemented for the primitive integers and floats, for [`Complex`], and for
//! [`Dual`]. That closure is deliberate: it lets the extensions nest, so
//! `Dual<Complex<f64>>` (a dual number with complex coefficients) and
//! `Matrix<Dual<f64>>` both work.

mod complex;
mod dual;
mod primitive;

use std::fmt::{self, Display};

use num_traits::{Float, Num};

pub use complex::Complex;
pub use dual::Dual;
pub use half::{bf16, f16};

/// What a numeric element must provide: the primitive integers and floats,
/// [`Complex`], and [`Dual`] all implement it.
///
/// Because `Complex` and `Dual` are themselves coefficients, the extensions
/// nest — `Dual<Complex<f64>>` and `Vector<Dual<f64>>` are ordinary types.
pub trait Coefficient: Num + Copy + 'static {
    /// Whether ordinary division belongs to a field-like coefficient domain.
    ///
    /// Matrix inversion needs fractional intermediate values. Primitive integer
    /// division truncates them, so it cannot implement Gauss–Jordan elimination
    /// correctly even when the final inverse happens to contain only integers.
    fn supports_fractional_division() -> bool {
        true
    }

    /// The quotient truncated toward zero, `trunc(self / rhs)`. This is the
    /// (locally constant) integer quotient underlying `%`.
    fn trunc_div(self, rhs: Self) -> Self;

    /// Round to the nearest whole value (the identity for integer types).
    fn round(self) -> Self;

    /// A real, comparable size for this value. Used to choose pivots in
    /// [`Matrix`](crate::tensors::Matrix) elimination, where the element type
    /// itself may have no ordering (complex numbers, for instance).
    fn magnitude(self) -> f64;
}

/// A floating-point element the generic tensor algebra runs on: `f32`, `f64`,
/// and the compact `f16` and `bf16`.
///
/// This is [`Float`], [`Coefficient`] and [`Power`] together, plus the one
/// thing none of them provides: the IEEE 754 *total* order, which sorting needs
/// and `PartialOrd` cannot give. [`Kernels`](crate::tensors::Kernels), the
/// optimizers, the automatic-differentiation types and the statistics are
/// written once over `Real`, so a `f64` model uses exactly the same code as a
/// `f32` one.
///
/// Every `Real` type computes in its own precision: elementwise operations
/// round to the type each time, while sums and products of the compact types
/// accumulate in `f32` and round once (see the `compact` CPU kernels and the
/// Metal shaders). The Metal backend computes in `f32`, `f16` and `bf16`;
/// `f64` stays on the host.
pub trait Real: Float + Coefficient + Power<Output = Self> + fmt::Debug + Display {
    /// IEEE 754 `totalOrder`: `−0.0` precedes `+0.0` and NaNs sort to the ends
    /// by sign instead of comparing unordered.
    fn total_order(&self, other: &Self) -> std::cmp::Ordering;

    /// Round an `f64` to this type. Out-of-range values saturate to infinity,
    /// as an `as` cast does.
    fn from_f64(value: f64) -> Self {
        <Self as num_traits::NumCast>::from(value).unwrap_or_else(|| {
            if value.is_nan() {
                Self::nan()
            } else if value > 0.0 {
                Self::infinity()
            } else {
                Self::neg_infinity()
            }
        })
    }

    /// Widen to an `f64`, exactly.
    ///
    /// Named `into_f64` because `to_f64` is already [`ToPrimitive`](num_traits::ToPrimitive)'s,
    /// which every [`Float`] has, and the two would be ambiguous.
    fn into_f64(self) -> f64 {
        <Self as num_traits::ToPrimitive>::to_f64(&self).unwrap_or(f64::NAN)
    }
}

/// A numeric type usable for computing the result of `sin`.
pub trait Sin {
    type Output;

    /// The result of sin(x).
    fn sin(self) -> <Self as Sin>::Output;
}

/// A numeric type usable for computing the result of `cos`.
pub trait Cos {
    type Output;

    /// The result of cos(x).
    fn cos(self) -> <Self as Cos>::Output;
}

/// A numeric type usable for computing the result of `tan`.
pub trait Tan {
    type Output;

    fn tan(self) -> <Self as Tan>::Output;
}

/// A numeric type usable for computing the result of `sec`.
pub trait Sec {
    type Output;

    fn sec(self) -> <Self as Sec>::Output;
}

/// A numeric type usable for computing the result of `csc`.
pub trait Csc {
    type Output;

    fn csc(self) -> <Self as Csc>::Output;
}

/// A numeric type usable for computing the result of `exp(x)`
pub trait Exp {
    type Output;

    fn exp(self) -> <Self as Exp>::Output;
}

/// A numeric type usable for computing the result of `arctan(x)`
pub trait Arctan {
    type Output;

    fn arctan(self) -> <Self as Arctan>::Output;
}

/// A numeric type usable for computing the result of `arcsin(x)`.
pub trait Arcsin {
    type Output;

    fn arcsin(self) -> <Self as Arcsin>::Output;
}

/// A numeric type usable for computing the result of `arccos(x)`.
pub trait Arccos {
    type Output;

    fn arccos(self) -> <Self as Arccos>::Output;
}

/// A numeric type usable for computing the natural logarithm `ln(x)`.
pub trait Ln {
    type Output;

    fn ln(self) -> <Self as Ln>::Output;
}

/// A numeric type usable for computing the hyperbolic sine `sinh(x)`.
pub trait Sinh {
    type Output;

    fn sinh(self) -> <Self as Sinh>::Output;
}

/// A numeric type usable for computing the hyperbolic cosine `cosh(x)`.
pub trait Cosh {
    type Output;

    fn cosh(self) -> <Self as Cosh>::Output;
}

/// A numeric type usable for computing the hyperbolic tangent `tanh(x)`.
pub trait Tanh {
    type Output;

    fn tanh(self) -> <Self as Tanh>::Output;
}

/// A numeric type usable for computing the principal square root `√x`.
pub trait Sqrt {
    type Output;

    fn sqrt(self) -> <Self as Sqrt>::Output;
}

/// A numeric type usable for raising a base to a power, `x^y`.
pub trait Power<Rhs = Self> {
    type Output;

    fn power(self, exponent: Rhs) -> <Self as Power<Rhs>>::Output;
}

/// A numeric type usable for taking the reciprocal of a value
pub trait Recip {
    type Output;

    fn recip(self) -> <Self as Recip>::Output;
}
