//! Conjugation and the conjugate products.
//!
//! [`Conj`] itself is defined beside the numeric types in [`crate::numbers`],
//! because [`Field`](crate::numbers::Field) needs it. It follows the same
//! conventions as the traits here.

pub use crate::numbers::Conj;

/// The conjugate inner product, `Σ conj(selfᵢ) · rhsᵢ`.
///
/// [`Dot`](super::Dot) keeps its bilinear meaning. On real values the two agree.
pub trait Dotc<Rhs = Self> {
    /// The scalar-like result.
    type Output;

    /// Compute the conjugate inner product.
    ///
    /// # Panics
    ///
    /// If the shapes differ.
    fn dotc(self, rhs: Rhs) -> Self::Output;
}

/// The conjugate transpose.
///
/// A [`Tensor`](crate::tensors::Tensor) exchanges its last two axes. A scalar's
/// adjoint is its conjugate.
pub trait Adjoint {
    /// The conjugate-transposed result.
    type Output;

    /// Conjugate every element and exchange the rows and columns.
    fn adjoint(self) -> Self::Output;
}
