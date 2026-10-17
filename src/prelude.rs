//! The mathematical traits, for a glob import.
//!
//! Every mathematical operation in the crate is a trait method. Importing this
//! module brings them all into scope:
//!
//! ```
//! use tensorcrate::prelude::*;
//! use tensorcrate::tensors::{Matrix, Vector};
//!
//! let a = Matrix::from_rows([[1.0, 2.0], [3.0, 4.0]]);
//! let v = Vector::new([1.0, 1.0]);
//! assert_eq!(MatVec::matvec(&a, &v), Vector::new([3.0, 7.0]));
//! assert_eq!(Dot::dot(&v, &v), 2.0);
//! ```
//!
//! The types themselves are not re-exported. Import them from
//! [`numbers`](crate::numbers) and [`tensors`](crate::tensors).

pub use crate::numbers::{
    Arccos, Arcsin, Arctan, Conj, Cos, Cosh, Csc, Exp, Ln, Power, Recip, Sec, Sin, Sinh, Sqrt, Tan,
    Tanh,
};
pub use crate::tensors::algebra::{
    Adjoint, Clamp, Comparison, Cross, Determinant, Dot, Dotc, Inverse, MatMul, MatMulAdd, MatVec,
    MatVecAdd, MinMax, PrefixSum, Reduction, Scale, Sort, Transpose, VecMat,
};
pub use crate::tensors::{Ordered, Transcendental};
