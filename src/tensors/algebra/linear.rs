//! Products, transposition, inversion and determinants.
//!
//! The standard operator traits describe elementwise arithmetic in this crate.
//! These traits describe the contractions whose result depends on the ranks of
//! their operands. See the [module conventions](super) for receivers and
//! generic bounds.

use crate::numbers::{Coefficient, Complex, Conj, Real};
use crate::tensors::{
    DualMatrix, DualVector, Host, Kernels, Matrix, MatrixVar, ScalarVar, SparseMatrix, Vector,
    VectorVar,
};

/// Matrix multiplication, `self · rhs`.
pub trait MatMul<Rhs = Self> {
    /// The matrix-like result.
    type Output;

    /// Multiply two matrices.
    ///
    /// # Panics
    ///
    /// If the left column count differs from the right row count.
    fn matmul(self, rhs: Rhs) -> Self::Output;
}

/// Fused matrix multiply and add, `self · rhs + addend`.
///
/// The addend is taken by value so its storage can hold the result.
pub trait MatMulAdd<Rhs, Addend> {
    /// The matrix-like result.
    type Output;

    /// Multiply two matrices and add `addend` in the same pass.
    ///
    /// # Panics
    ///
    /// If the product is undefined or `addend` does not have its shape.
    fn matmul_add(self, rhs: Rhs, addend: Addend) -> Self::Output;
}

/// Matrix by column vector multiplication, `self · rhs`.
pub trait MatVec<Rhs> {
    /// The vector-like result.
    type Output;

    /// Multiply a matrix by a column vector.
    ///
    /// # Panics
    ///
    /// If the matrix column count differs from the vector length.
    fn matvec(self, rhs: Rhs) -> Self::Output;
}

/// Fused matrix by vector multiply and add, `self · rhs + addend`.
///
/// The addend is taken by value so its storage can hold the result.
pub trait MatVecAdd<Rhs, Addend> {
    /// The vector-like result.
    type Output;

    /// Multiply a matrix by a vector and add `addend` in the same pass.
    ///
    /// # Panics
    ///
    /// If the product is undefined or `addend` does not have its length.
    fn matvec_add(self, rhs: Rhs, addend: Addend) -> Self::Output;
}

/// Row vector by matrix multiplication, `self · rhs`.
pub trait VecMat<Rhs> {
    /// The vector-like result.
    type Output;

    /// Multiply a row vector by a matrix.
    ///
    /// # Panics
    ///
    /// If the vector length differs from the matrix row count.
    fn vecmat(self, rhs: Rhs) -> Self::Output;
}

/// The bilinear inner product. [`Dotc`](super::Dotc) is the conjugate one.
pub trait Dot<Rhs = Self> {
    /// The scalar-like result.
    type Output;

    /// Compute the inner product.
    ///
    /// # Panics
    ///
    /// If the shapes differ.
    fn dot(self, rhs: Rhs) -> Self::Output;
}

/// Transposition of a matrix-like value.
///
/// A [`Tensor`](crate::tensors::Tensor) exchanges its last two axes, as
/// [`Adjoint`](super::Adjoint) does. Exchanging any other pair of axes is
/// `Tensor::transpose(a, b)`, which is a layout operation rather than this one.
pub trait Transpose {
    /// The transposed result.
    type Output;

    /// Exchange the rows and columns.
    fn transpose(self) -> Self::Output;
}

/// The multiplicative inverse of a square matrix.
pub trait Inverse {
    /// The result, usually a `Result` because the matrix can be singular.
    type Output;

    /// Invert the matrix.
    fn inverse(self) -> Self::Output;
}

/// The determinant of a square matrix.
pub trait Determinant {
    /// The scalar-like result.
    type Output;

    /// Compute the determinant.
    ///
    /// # Panics
    ///
    /// If the matrix is not square.
    fn determinant(self) -> Self::Output;
}

// ---- dense host tensors ------------------------------------------------------
//
// These keep the full `Coefficient` surface of the inherent methods, including
// integers, complex values and scalar dual numbers.

impl<'b, T: Coefficient> MatMul<&'b Matrix<T, Host>> for &Matrix<T, Host> {
    type Output = Matrix<T, Host>;

    fn matmul(self, rhs: &'b Matrix<T, Host>) -> Self::Output {
        Matrix::<T, Host>::matmul(self, rhs)
    }
}

impl<'b, T: Coefficient> MatVec<&'b Vector<T, Host>> for &Matrix<T, Host> {
    type Output = Vector<T, Host>;

    fn matvec(self, rhs: &'b Vector<T, Host>) -> Self::Output {
        Matrix::<T, Host>::matvec(self, rhs)
    }
}

impl<'b, T: Coefficient> VecMat<&'b Matrix<T, Host>> for &Vector<T, Host> {
    type Output = Vector<T, Host>;

    fn vecmat(self, rhs: &'b Matrix<T, Host>) -> Self::Output {
        Vector::<T, Host>::vecmat(self, rhs)
    }
}

impl<'b, T: Coefficient> Dot<&'b Vector<T, Host>> for &Vector<T, Host> {
    type Output = T;

    fn dot(self, rhs: &'b Vector<T, Host>) -> T {
        Vector::<T, Host>::dot(self, rhs)
    }
}

impl<T: Coefficient> Transpose for &Matrix<T, Host> {
    type Output = Matrix<T, Host>;

    fn transpose(self) -> Self::Output {
        Matrix::<T, Host>::transpose(self)
    }
}

impl<T: Real> Conj for &Matrix<Complex<T>, Host> {
    type Output = Matrix<Complex<T>, Host>;

    fn conj(self) -> Self::Output {
        self.map(|value| value.conj())
    }
}

impl<T: Real> Conj for &Vector<Complex<T>, Host> {
    type Output = Vector<Complex<T>, Host>;

    fn conj(self) -> Self::Output {
        self.map(|value| value.conj())
    }
}

// ---- resident dense tensors --------------------------------------------------
//
// Each forwarding call stays on the GPU, as the inherent method does.

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal {
    use super::{Dot, MatMul, MatVec, Transpose, VecMat};
    use crate::metal::MetalElement;
    use crate::tensors::{Matrix, Metal, Vector};

    impl<'b, T: MetalElement> MatMul<&'b Matrix<T, Metal>> for &Matrix<T, Metal> {
        type Output = Matrix<T, Metal>;

        fn matmul(self, rhs: &'b Matrix<T, Metal>) -> Self::Output {
            Matrix::<T, Metal>::matmul(self, rhs)
        }
    }

    impl<'b, T: MetalElement> MatVec<&'b Vector<T, Metal>> for &Matrix<T, Metal> {
        type Output = Vector<T, Metal>;

        fn matvec(self, rhs: &'b Vector<T, Metal>) -> Self::Output {
            Matrix::<T, Metal>::matvec(self, rhs)
        }
    }

    impl<'b, T: MetalElement> VecMat<&'b Matrix<T, Metal>> for &Vector<T, Metal> {
        type Output = Vector<T, Metal>;

        fn vecmat(self, rhs: &'b Matrix<T, Metal>) -> Self::Output {
            Vector::<T, Metal>::vecmat(self, rhs)
        }
    }

    impl<'b, T: MetalElement> Dot<&'b Vector<T, Metal>> for &Vector<T, Metal> {
        type Output = T;

        fn dot(self, rhs: &'b Vector<T, Metal>) -> T {
            Vector::<T, Metal>::dot(self, rhs)
        }
    }

    impl<T: MetalElement> Transpose for &Matrix<T, Metal> {
        type Output = Matrix<T, Metal>;

        fn transpose(self) -> Self::Output {
            Matrix::<T, Metal>::transpose(self)
        }
    }
}

// ---- forward mode ------------------------------------------------------------
//
// Dual values keep their value and tangent representation.

impl<'b, B: Kernels<T>, T: Real> MatMul<&'b DualMatrix<B, T>> for &DualMatrix<B, T> {
    type Output = DualMatrix<B, T>;

    fn matmul(self, rhs: &'b DualMatrix<B, T>) -> Self::Output {
        DualMatrix::matmul(self, rhs)
    }
}

impl<'b, B: Kernels<T>, T: Real> MatVec<&'b DualVector<B, T>> for &DualMatrix<B, T> {
    type Output = DualVector<B, T>;

    fn matvec(self, rhs: &'b DualVector<B, T>) -> Self::Output {
        DualMatrix::matvec(self, rhs)
    }
}

impl<'b, B: Kernels<T>, T: Real> VecMat<&'b DualMatrix<B, T>> for &DualVector<B, T> {
    type Output = DualVector<B, T>;

    fn vecmat(self, rhs: &'b DualMatrix<B, T>) -> Self::Output {
        DualVector::vecmat(self, rhs)
    }
}

impl<'b, B: Kernels<T>, T: Real> Dot<&'b DualVector<B, T>> for &DualVector<B, T> {
    type Output = crate::numbers::Dual<T>;

    fn dot(self, rhs: &'b DualVector<B, T>) -> Self::Output {
        DualVector::dot(self, rhs)
    }
}

impl<B: Kernels<T>, T: Real> Transpose for &DualMatrix<B, T> {
    type Output = DualMatrix<B, T>;

    fn transpose(self) -> Self::Output {
        DualMatrix::transpose(self)
    }
}

// ---- reverse mode ------------------------------------------------------------
//
// Tape variables keep their tape lifetime in both operands and the output.

impl<'b, 't, B: Kernels<T>, T: Real> MatMul<&'b MatrixVar<'t, B, T>> for &MatrixVar<'t, B, T> {
    type Output = MatrixVar<'t, B, T>;

    fn matmul(self, rhs: &'b MatrixVar<'t, B, T>) -> Self::Output {
        MatrixVar::matmul(self, rhs)
    }
}

impl<'b, 't, B: Kernels<T>, T: Real> MatVec<&'b VectorVar<'t, B, T>> for &MatrixVar<'t, B, T> {
    type Output = VectorVar<'t, B, T>;

    fn matvec(self, rhs: &'b VectorVar<'t, B, T>) -> Self::Output {
        MatrixVar::matvec(self, rhs)
    }
}

impl<'b, 't, B: Kernels<T>, T: Real> VecMat<&'b MatrixVar<'t, B, T>> for &VectorVar<'t, B, T> {
    type Output = VectorVar<'t, B, T>;

    fn vecmat(self, rhs: &'b MatrixVar<'t, B, T>) -> Self::Output {
        VectorVar::vecmat(self, rhs)
    }
}

impl<'b, 't, B: Kernels<T>, T: Real> Dot<&'b VectorVar<'t, B, T>> for &VectorVar<'t, B, T> {
    type Output = ScalarVar<'t, B, T>;

    fn dot(self, rhs: &'b VectorVar<'t, B, T>) -> Self::Output {
        VectorVar::dot(self, rhs)
    }
}

impl<'t, B: Kernels<T>, T: Real> Transpose for &MatrixVar<'t, B, T> {
    type Output = MatrixVar<'t, B, T>;

    fn transpose(self) -> Self::Output {
        MatrixVar::transpose(self)
    }
}

// ---- sparse ------------------------------------------------------------------
//
// A slice is the right operand, matching `SparseMatrix::matvec`.

impl<'b, T: Coefficient> MatVec<&'b [T]> for &SparseMatrix<T> {
    type Output = Vec<T>;

    fn matvec(self, rhs: &'b [T]) -> Vec<T> {
        SparseMatrix::matvec(self, rhs)
    }
}

impl<T: Coefficient> Transpose for &SparseMatrix<T> {
    type Output = SparseMatrix<T>;

    fn transpose(self) -> Self::Output {
        SparseMatrix::transpose(self)
    }
}
