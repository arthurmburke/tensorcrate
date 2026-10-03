//! Traits for the named linear-algebra operations.
//!
//! The standard operator traits describe elementwise arithmetic in this crate.
//! These traits describe the contractions whose result depends on the ranks of
//! their operands.  In particular, they make code generic over an ordinary
//! tensor, a forward-mode dual tensor, or a reverse-mode tape variable without
//! erasing the concrete output type.
//!
//! ```
//! use tensorcrate::tensors::{MatMul, Matrix, Transpose};
//!
//! fn gram<M>(matrix: &M) -> <M as MatMul>::Output
//! where
//!     M: MatMul + Transpose<Output = M>,
//! {
//!     matrix.transpose().matmul(matrix)
//! }
//!
//! let a = Matrix::from_rows([[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]]);
//! assert_eq!(gram(&a), Matrix::from_rows([[35.0, 44.0], [44.0, 56.0]]));
//! ```

use super::{
    DualMatrix, DualVector, Host, Kernels, Matrix, MatrixVar, SparseMatrix, Vector, VectorVar,
};
use crate::numbers::{Coefficient, Real};

/// Matrix multiplication, `self · rhs`.
///
/// `Rhs` and `Output` are associated with the implementation, so the same
/// bound works for dense matrices and their forward- and reverse-mode forms.
pub trait MatMul<Rhs: ?Sized = Self> {
    /// The matrix-like result.
    type Output;

    /// Multiply two matrices.
    ///
    /// # Panics
    ///
    /// If the left column count differs from the right row count.
    fn matmul(&self, rhs: &Rhs) -> Self::Output;
}

/// Matrix–column-vector multiplication, `self · rhs`.
pub trait MatVec<Rhs: ?Sized> {
    /// The vector-like result.
    type Output;

    /// Multiply a matrix by a column vector.
    ///
    /// # Panics
    ///
    /// If the matrix column count differs from the vector length.
    fn matvec(&self, rhs: &Rhs) -> Self::Output;
}

/// Row-vector–matrix multiplication, `self · rhs`.
pub trait VecMat<Rhs: ?Sized> {
    /// The vector-like result.
    type Output;

    /// Multiply a row vector by a matrix.
    ///
    /// # Panics
    ///
    /// If the vector length differs from the matrix row count.
    fn vecmat(&self, rhs: &Rhs) -> Self::Output;
}

/// The inner (dot) product of two vectors.
pub trait Dot<Rhs: ?Sized = Self> {
    /// The scalar-like result.
    type Output;

    /// Compute the inner product.
    ///
    /// # Panics
    ///
    /// If the vector lengths differ.
    fn dot(&self, rhs: &Rhs) -> Self::Output;
}

/// Transposition of a two-dimensional value.
///
/// N-dimensional [`Tensor`](super::Tensor)s use `transpose(a, b)` because the
/// two axes are part of the operation.  This trait intentionally represents
/// the unambiguous matrix operation only.
pub trait Transpose {
    /// The transposed matrix-like result.
    type Output;

    /// Exchange the rows and columns.
    fn transpose(&self) -> Self::Output;
}

// Dense host tensors retain the full `Coefficient` surface of their inherent
// methods, including integers, complex values, and scalar dual numbers.
impl<T: Coefficient> MatMul for Matrix<T, Host> {
    type Output = Self;

    fn matmul(&self, rhs: &Self) -> Self {
        Matrix::<T, Host>::matmul(self, rhs)
    }
}

impl<T: Coefficient> MatVec<Vector<T, Host>> for Matrix<T, Host> {
    type Output = Vector<T, Host>;

    fn matvec(&self, rhs: &Vector<T, Host>) -> Self::Output {
        Matrix::<T, Host>::matvec(self, rhs)
    }
}

impl<T: Coefficient> VecMat<Matrix<T, Host>> for Vector<T, Host> {
    type Output = Self;

    fn vecmat(&self, rhs: &Matrix<T, Host>) -> Self {
        Vector::<T, Host>::vecmat(self, rhs)
    }
}

impl<T: Coefficient> Dot for Vector<T, Host> {
    type Output = T;

    fn dot(&self, rhs: &Self) -> T {
        Vector::<T, Host>::dot(self, rhs)
    }
}

impl<T: Coefficient> Transpose for Matrix<T, Host> {
    type Output = Self;

    fn transpose(&self) -> Self {
        Matrix::<T, Host>::transpose(self)
    }
}

// Resident dense tensors have the same traits; each forwarding call remains
// on the GPU just like the corresponding inherent method.
#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal {
    use super::{Dot, MatMul, MatVec, Transpose, VecMat};
    use crate::metal::MetalElement;
    use crate::tensors::{Matrix, Metal, Vector};

    impl<T: MetalElement> MatMul for Matrix<T, Metal> {
        type Output = Self;

        fn matmul(&self, rhs: &Self) -> Self {
            Matrix::<T, Metal>::matmul(self, rhs)
        }
    }

    impl<T: MetalElement> MatVec<Vector<T, Metal>> for Matrix<T, Metal> {
        type Output = Vector<T, Metal>;

        fn matvec(&self, rhs: &Vector<T, Metal>) -> Self::Output {
            Matrix::<T, Metal>::matvec(self, rhs)
        }
    }

    impl<T: MetalElement> VecMat<Matrix<T, Metal>> for Vector<T, Metal> {
        type Output = Self;

        fn vecmat(&self, rhs: &Matrix<T, Metal>) -> Self {
            Vector::<T, Metal>::vecmat(self, rhs)
        }
    }

    impl<T: MetalElement> Dot for Vector<T, Metal> {
        type Output = T;

        fn dot(&self, rhs: &Self) -> T {
            Vector::<T, Metal>::dot(self, rhs)
        }
    }

    impl<T: MetalElement> Transpose for Matrix<T, Metal> {
        type Output = Self;

        fn transpose(&self) -> Self {
            Matrix::<T, Metal>::transpose(self)
        }
    }
}

// Forward-mode values preserve their value/tangent representation.
impl<B: Kernels<T>, T: Real> MatMul for DualMatrix<B, T> {
    type Output = Self;

    fn matmul(&self, rhs: &Self) -> Self {
        DualMatrix::matmul(self, rhs)
    }
}

impl<B: Kernels<T>, T: Real> MatVec<DualVector<B, T>> for DualMatrix<B, T> {
    type Output = DualVector<B, T>;

    fn matvec(&self, rhs: &DualVector<B, T>) -> Self::Output {
        DualMatrix::matvec(self, rhs)
    }
}

impl<B: Kernels<T>, T: Real> VecMat<DualMatrix<B, T>> for DualVector<B, T> {
    type Output = Self;

    fn vecmat(&self, rhs: &DualMatrix<B, T>) -> Self {
        DualVector::vecmat(self, rhs)
    }
}

impl<B: Kernels<T>, T: Real> Dot for DualVector<B, T> {
    type Output = crate::numbers::Dual<T>;

    fn dot(&self, rhs: &Self) -> Self::Output {
        DualVector::dot(self, rhs)
    }
}

impl<B: Kernels<T>, T: Real> Transpose for DualMatrix<B, T> {
    type Output = Self;

    fn transpose(&self) -> Self {
        DualMatrix::transpose(self)
    }
}

// Reverse-mode values preserve their tape lifetime in both input and output.
impl<'t, B: Kernels<T>, T: Real> MatMul for MatrixVar<'t, B, T> {
    type Output = Self;

    fn matmul(&self, rhs: &Self) -> Self {
        MatrixVar::matmul(self, rhs)
    }
}

impl<'t, B: Kernels<T>, T: Real> MatVec<VectorVar<'t, B, T>> for MatrixVar<'t, B, T> {
    type Output = VectorVar<'t, B, T>;

    fn matvec(&self, rhs: &VectorVar<'t, B, T>) -> Self::Output {
        MatrixVar::matvec(self, rhs)
    }
}

impl<'t, B: Kernels<T>, T: Real> VecMat<MatrixVar<'t, B, T>> for VectorVar<'t, B, T> {
    type Output = Self;

    fn vecmat(&self, rhs: &MatrixVar<'t, B, T>) -> Self {
        VectorVar::vecmat(self, rhs)
    }
}

impl<'t, B: Kernels<T>, T: Real> Dot for VectorVar<'t, B, T> {
    type Output = super::ScalarVar<'t, B, T>;

    fn dot(&self, rhs: &Self) -> Self::Output {
        VectorVar::dot(self, rhs)
    }
}

impl<'t, B: Kernels<T>, T: Real> Transpose for MatrixVar<'t, B, T> {
    type Output = Self;

    fn transpose(&self) -> Self {
        MatrixVar::transpose(self)
    }
}

// Sparse matrices participate where their representation already supplies an
// operation.  A slice is deliberately the RHS, matching `SparseMatrix::matvec`
// and allowing both arrays and vectors through normal slice coercion.
impl<T: Coefficient> MatVec<[T]> for SparseMatrix<T> {
    type Output = Vec<T>;

    fn matvec(&self, rhs: &[T]) -> Vec<T> {
        SparseMatrix::matvec(self, rhs)
    }
}

impl<T: Coefficient> Transpose for SparseMatrix<T> {
    type Output = Self;

    fn transpose(&self) -> Self {
        SparseMatrix::transpose(self)
    }
}
