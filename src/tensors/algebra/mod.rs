//! The algebraic traits: the entry point for every mathematical operation.
//!
//! Each operation is one trait. A scalar, a dense or sparse tensor, a
//! forward-mode dual tensor and a reverse-mode tape variable all implement the
//! same trait, so generic code names the operation once and keeps the concrete
//! output type.
//!
//! # Conventions
//!
//! The traits follow the shape of [`std::ops`]:
//!
//! * The receiver is `self`, taken by value.
//! * The right operand is a type parameter, `Rhs`, defaulting to `Self`.
//! * The result is the associated type `Output`.
//!
//! Tensors implement each trait for references only, `&Matrix` with
//! `Rhs = &Matrix`. Method syntax therefore borrows: `a.matmul(&b)` leaves `a`
//! usable. Copy scalars implement the traits by value.
//!
//! A scalar operand on the right is the `Rhs` parameter. A scalar operand on
//! the left is `Self`, with the tensor reference as `Rhs`. Both orders are
//! impls of one trait rather than two methods.
//!
//! # Generic code
//!
//! A bound names the reference type, as it would for `&a + &b`:
//!
//! ```
//! use tensorcrate::prelude::*;
//! use tensorcrate::tensors::Matrix;
//!
//! fn gram<M>(matrix: &M) -> M
//! where
//!     for<'a> &'a M: MatMul<&'a M, Output = M> + Transpose<Output = M>,
//! {
//!     matrix.transpose().matmul(matrix)
//! }
//!
//! let a = Matrix::from_rows([[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]]);
//! assert_eq!(gram(&a), Matrix::from_rows([[35.0, 44.0], [44.0, 56.0]]));
//! ```
//!
//! # Families
//!
//! * [`linear`]: products, transposition, inversion and determinants.
//! * [`conjugate`]: conjugation and the conjugate products.
//! * [`elementwise`]: scaling, minima and maxima, clamping and comparison.
//! * [`reduce`]: whole-value reductions, prefix sums and sorting.
//! * The analytic functions (`Sin`, `Exp`, `Power`, ...) live in
//!   [`crate::numbers`] and are re-exported by [`crate::prelude`].
//!
//! Elementwise `+ - * / %` and negation stay on the [`std::ops`] traits.

pub mod conjugate;
pub mod elementwise;
pub mod linear;
pub mod reduce;

pub use conjugate::{Adjoint, Dotc};
pub use elementwise::{Clamp, Comparison, MinMax, Scale};
pub use linear::{
    Determinant, Dot, Inverse, MatMul, MatMulAdd, MatVec, MatVecAdd, Transpose, VecMat,
};
pub use reduce::{PrefixSum, Reduction, Sort};
