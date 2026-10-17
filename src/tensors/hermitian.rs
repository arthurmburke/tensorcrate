//! Conjugate products fused into one pass over the input values.

use super::algebra::{Adjoint, Dotc};
use super::{Matrix, SparseMatrix, Tensor, Vector};
use crate::numbers::{Coefficient, Conj};

// Scalars implement every ownership combination, as `std` does for `Add`, so
// `math!` can borrow every operand without knowing its rank. The impls are per
// type because a blanket impl over `T` would overlap the one over `&T`.
macro_rules! scalar_hermitian {
    ($([$($generics:tt)*] $t:ty),* $(,)?) => {$(
        impl<$($generics)*> Dotc for $t
        where
            $t: Coefficient + Conj<Output = $t>,
        {
            type Output = $t;
            fn dotc(self, rhs: $t) -> $t {
                self.conj() * rhs
            }
        }

        impl<'b, $($generics)*> Dotc<&'b $t> for $t
        where
            $t: Coefficient + Conj<Output = $t>,
        {
            type Output = $t;
            fn dotc(self, rhs: &'b $t) -> $t {
                self.conj() * *rhs
            }
        }

        impl<$($generics)*> Dotc<$t> for &$t
        where
            $t: Coefficient + Conj<Output = $t>,
        {
            type Output = $t;
            fn dotc(self, rhs: $t) -> $t {
                self.conj() * rhs
            }
        }

        impl<'b, $($generics)*> Dotc<&'b $t> for &$t
        where
            $t: Coefficient + Conj<Output = $t>,
        {
            type Output = $t;
            fn dotc(self, rhs: &'b $t) -> $t {
                self.conj() * *rhs
            }
        }

        impl<$($generics)*> Adjoint for $t
        where
            $t: Coefficient + Conj<Output = $t>,
        {
            type Output = $t;
            fn adjoint(self) -> $t {
                self.conj()
            }
        }

        impl<$($generics)*> Adjoint for &$t
        where
            $t: Coefficient + Conj<Output = $t>,
        {
            type Output = $t;
            fn adjoint(self) -> $t {
                self.conj()
            }
        }
    )*};
}

scalar_hermitian!(
    [] i8, [] u8, [] i16, [] u16, [] i32, [] u32, [] i64, [] u64,
    [] f32, [] f64, [] crate::numbers::f16, [] crate::numbers::bf16,
    [T] crate::numbers::Complex<T>, [T] crate::numbers::Dual<T>,
);

fn product<T: Coefficient + Conj<Output = T>>(a: &[T], b: &[T]) -> T {
    assert_eq!(a.len(), b.len(), "dotc: incompatible lengths");
    if let Some(sum) = crate::compact::dot(a, b) {
        return sum;
    }
    #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
    if let Some(sum) = super::simd_dispatch::dot(a, b) {
        return sum;
    }
    a.iter()
        .zip(b)
        .fold(T::zero(), |sum, (a, b)| sum + a.conj() * *b)
}

impl<'b, T: Coefficient + Conj<Output = T>> Dotc<&'b Vector<T>> for &Vector<T> {
    type Output = T;
    fn dotc(self, rhs: &'b Vector<T>) -> T {
        product(self.as_slice(), rhs.as_slice())
    }
}

impl<'b, T: Coefficient + Conj<Output = T>> Dotc<&'b Matrix<T>> for &Matrix<T> {
    type Output = T;
    fn dotc(self, rhs: &'b Matrix<T>) -> T {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "dotc: incompatible matrix shapes"
        );
        product(self.as_slice(), rhs.as_slice())
    }
}

impl<'b, T: Coefficient + Conj<Output = T>> Dotc<&'b Tensor<T>> for &Tensor<T> {
    type Output = T;
    fn dotc(self, rhs: &'b Tensor<T>) -> T {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "dotc: incompatible tensor shapes"
        );
        product(self.as_slice(), rhs.as_slice())
    }
}

impl<T: Coefficient + Conj<Output = T>> Adjoint for &Matrix<T> {
    type Output = Matrix<T>;
    fn adjoint(self) -> Matrix<T> {
        let mut values = vec![T::zero(); self.rows() * self.cols()];
        for i in 0..self.rows() {
            for j in 0..self.cols() {
                values[j * self.rows() + i] = self[(i, j)].conj();
            }
        }
        Matrix::from_flat(self.cols(), self.rows(), values)
    }
}

impl<T: Coefficient + Conj<Output = T>> Adjoint for &Tensor<T> {
    type Output = Tensor<T>;
    fn adjoint(self) -> Tensor<T> {
        assert!(
            self.rank() >= 2,
            "adjoint: tensor requires at least two axes"
        );
        let mut shape = self.shape().to_vec();
        let n = shape.len();
        let (rows, cols) = (shape[n - 2], shape[n - 1]);
        shape.swap(n - 2, n - 1);
        let mut values = vec![T::zero(); self.len()];
        if !self.is_empty() {
            for (input, output) in self
                .as_slice()
                .chunks(rows * cols)
                .zip(values.chunks_mut(rows * cols))
            {
                for i in 0..rows {
                    for j in 0..cols {
                        output[j * rows + i] = input[i * cols + j].conj();
                    }
                }
            }
        }
        Tensor::from_vec(&shape, values)
    }
}

impl<T: Coefficient + Conj<Output = T>> Adjoint for &super::CsrMatrix<T> {
    type Output = super::CsrMatrix<T>;
    fn adjoint(self) -> super::CsrMatrix<T> {
        super::CsrMatrix::adjoint(self)
    }
}

impl<T: Coefficient + Conj<Output = T>> Adjoint for &SparseMatrix<T> {
    type Output = SparseMatrix<T>;
    fn adjoint(self) -> SparseMatrix<T> {
        SparseMatrix::from_triplets(
            self.cols(),
            self.rows(),
            self.triplets().map(|(r, c, v)| (c, r, v.conj())),
        )
    }
}

impl<T: Coefficient + Conj<Output = T>> Conj for &Tensor<T> {
    type Output = Tensor<T>;
    fn conj(self) -> Tensor<T> {
        Tensor::from_vec(
            self.shape(),
            self.as_slice().iter().map(|v| v.conj()).collect::<Vec<_>>(),
        )
    }
}

impl<'b, T: crate::numbers::Real, B: super::Kernels<T>> Dotc<&'b super::DualVector<B, T>>
    for &super::DualVector<B, T>
{
    type Output = crate::numbers::Dual<T>;
    fn dotc(self, rhs: &'b super::DualVector<B, T>) -> Self::Output {
        super::DualVector::dot(self, rhs)
    }
}

impl<T: crate::numbers::Real, B: super::Kernels<T>> Adjoint for &super::DualMatrix<B, T> {
    type Output = super::DualMatrix<B, T>;
    fn adjoint(self) -> Self::Output {
        super::DualMatrix::transpose(self)
    }
}

impl<'b, 't, T: crate::numbers::Real, B: super::Kernels<T>> Dotc<&'b super::VectorVar<'t, B, T>>
    for &super::VectorVar<'t, B, T>
{
    type Output = super::ScalarVar<'t, B, T>;
    fn dotc(self, rhs: &'b super::VectorVar<'t, B, T>) -> Self::Output {
        super::VectorVar::dot(self, rhs)
    }
}

impl<'t, T: crate::numbers::Real, B: super::Kernels<T>> Adjoint for &super::MatrixVar<'t, B, T> {
    type Output = super::MatrixVar<'t, B, T>;
    fn adjoint(self) -> Self::Output {
        super::MatrixVar::transpose(self)
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal {
    use crate::metal::OperatorElement;
    use crate::numbers::Field;
    use crate::tensors::algebra::{Adjoint, Dotc};
    use crate::tensors::{Matrix, Metal, Tensor, Vector};

    macro_rules! compact_hermitian {
        ($($t:ty),* $(,)?) => {$(
            impl<'b> Dotc<&'b Vector<$t, Metal>> for &Vector<$t, Metal> {
                type Output = $t;
                fn dotc(self, rhs: &'b Vector<$t, Metal>) -> $t {
                    Vector::<$t, Metal>::dot(self, rhs)
                }
            }
            impl Adjoint for &Matrix<$t, Metal> {
                type Output = Matrix<$t, Metal>;
                fn adjoint(self) -> Self::Output {
                    Matrix::<$t, Metal>::transpose(self)
                }
            }
        )*};
    }
    compact_hermitian!(crate::numbers::f16, crate::numbers::bf16);

    impl<'b, T: Field + OperatorElement> Dotc<&'b Vector<T, Metal>> for &Vector<T, Metal> {
        type Output = T;
        fn dotc(self, rhs: &'b Vector<T, Metal>) -> T {
            assert_eq!(self.len(), rhs.len(), "dotc: incompatible lengths");
            crate::metal::operator_dotc(self.storage(), rhs.storage())
        }
    }

    impl<T: Field + OperatorElement> Adjoint for &Matrix<T, Metal> {
        type Output = Matrix<T, Metal>;
        fn adjoint(self) -> Self::Output {
            Matrix::from_storage(
                self.cols(),
                self.rows(),
                crate::metal::operator_adjoint(self.storage(), self.shape()),
            )
        }
    }

    impl<'b, T: Field + OperatorElement> Dotc<&'b Matrix<T, Metal>> for &Matrix<T, Metal> {
        type Output = T;
        fn dotc(self, rhs: &'b Matrix<T, Metal>) -> T {
            assert_eq!(
                self.shape(),
                rhs.shape(),
                "dotc: incompatible matrix shapes"
            );
            crate::metal::operator_dotc(self.storage(), rhs.storage())
        }
    }

    impl<'b, T: Field + OperatorElement> Dotc<&'b Tensor<T, Metal>> for &Tensor<T, Metal> {
        type Output = T;
        fn dotc(self, rhs: &'b Tensor<T, Metal>) -> T {
            assert_eq!(
                self.shape(),
                rhs.shape(),
                "dotc: incompatible tensor shapes"
            );
            self.as_vector().dotc(rhs.as_vector())
        }
    }

    impl<T: Field + OperatorElement> Adjoint for &Tensor<T, Metal> {
        type Output = Tensor<T, Metal>;
        fn adjoint(self) -> Self::Output {
            assert!(
                self.rank() >= 2,
                "adjoint: tensor requires at least two axes"
            );
            let mut shape = self.shape().to_vec();
            let n = shape.len();
            let local = (shape[n - 2], shape[n - 1]);
            shape.swap(n - 2, n - 1);
            let values = crate::metal::operator_adjoint(self.as_vector().storage(), local);
            Tensor::from_vector(&shape, Vector::from_storage(self.len(), values))
        }
    }
}
