//! Conjugate products fused into one pass over the input values.

use super::{Matrix, SparseMatrix, Tensor, Vector};
use crate::numbers::{Coefficient, Conj};

/// Conjugate inner product. `dot` retains its existing bilinear semantics.
pub trait Dotc<Rhs: ?Sized = Self> {
    type Output;
    fn dotc(&self, rhs: &Rhs) -> Self::Output;
}

/// Conjugate transpose. Tensors exchange their last two axes.
pub trait Adjoint {
    type Output;
    fn adjoint(&self) -> Self::Output;
}

impl<T: Coefficient + Conj<Output = T>> Dotc for T {
    type Output = T;
    fn dotc(&self, rhs: &Self) -> T {
        self.conj() * *rhs
    }
}

impl<T: Coefficient + Conj<Output = T>> Adjoint for T {
    type Output = T;
    fn adjoint(&self) -> T {
        self.conj()
    }
}

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

impl<T: Coefficient + Conj<Output = T>> Dotc for Vector<T> {
    type Output = T;
    fn dotc(&self, rhs: &Self) -> T {
        product(self.as_slice(), rhs.as_slice())
    }
}

impl<T: Coefficient + Conj<Output = T>> Dotc for Matrix<T> {
    type Output = T;
    fn dotc(&self, rhs: &Self) -> T {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "dotc: incompatible matrix shapes"
        );
        product(self.as_slice(), rhs.as_slice())
    }
}

impl<T: Coefficient + Conj<Output = T>> Dotc for Tensor<T> {
    type Output = T;
    fn dotc(&self, rhs: &Self) -> T {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "dotc: incompatible tensor shapes"
        );
        product(self.as_slice(), rhs.as_slice())
    }
}

impl<T: Coefficient + Conj<Output = T>> Adjoint for Matrix<T> {
    type Output = Self;
    fn adjoint(&self) -> Self {
        let mut values = vec![T::zero(); self.rows() * self.cols()];
        for i in 0..self.rows() {
            for j in 0..self.cols() {
                values[j * self.rows() + i] = self[(i, j)].conj();
            }
        }
        Matrix::from_flat(self.cols(), self.rows(), values)
    }
}

impl<T: Coefficient + Conj<Output = T>> Adjoint for Tensor<T> {
    type Output = Self;
    fn adjoint(&self) -> Self {
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

impl<T: Coefficient + Conj<Output = T>> Adjoint for super::CsrMatrix<T> {
    type Output = Self;
    fn adjoint(&self) -> Self {
        self.adjoint()
    }
}

impl<T: Coefficient + Conj<Output = T>> Adjoint for SparseMatrix<T> {
    type Output = Self;
    fn adjoint(&self) -> Self {
        Self::from_triplets(
            self.cols(),
            self.rows(),
            self.triplets().map(|(r, c, v)| (c, r, v.conj())),
        )
    }
}

impl<T: Coefficient + Conj<Output = T>> Conj for Tensor<T> {
    type Output = Self;
    fn conj(&self) -> Self {
        Self::from_vec(
            self.shape(),
            self.as_slice().iter().map(Conj::conj).collect::<Vec<_>>(),
        )
    }
}

impl<T: crate::numbers::Real, B: super::Kernels<T>> Dotc for super::DualVector<B, T> {
    type Output = crate::numbers::Dual<T>;
    fn dotc(&self, rhs: &Self) -> Self::Output {
        self.dot(rhs)
    }
}

impl<T: crate::numbers::Real, B: super::Kernels<T>> Adjoint for super::DualMatrix<B, T> {
    type Output = Self;
    fn adjoint(&self) -> Self {
        self.transpose()
    }
}

impl<'t, T: crate::numbers::Real, B: super::Kernels<T>> Dotc for super::VectorVar<'t, B, T> {
    type Output = super::ScalarVar<'t, B, T>;
    fn dotc(&self, rhs: &Self) -> Self::Output {
        self.dot(rhs)
    }
}

impl<'t, T: crate::numbers::Real, B: super::Kernels<T>> Adjoint for super::MatrixVar<'t, B, T> {
    type Output = Self;
    fn adjoint(&self) -> Self {
        self.transpose()
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal {
    use super::{Adjoint, Dotc};
    use crate::metal::OperatorElement;
    use crate::numbers::Field;
    use crate::tensors::{Matrix, Metal, Tensor, Vector};

    macro_rules! compact_hermitian {
        ($($t:ty),* $(,)?) => {$(
            impl Dotc for Vector<$t, Metal> {
                type Output = $t;
                fn dotc(&self, rhs: &Self) -> $t { self.dot(rhs) }
            }
            impl Adjoint for Matrix<$t, Metal> {
                type Output = Self;
                fn adjoint(&self) -> Self { self.transpose() }
            }
        )*};
    }
    compact_hermitian!(crate::numbers::f16, crate::numbers::bf16);

    impl<T: Field + OperatorElement> Dotc for Vector<T, Metal> {
        type Output = T;
        fn dotc(&self, rhs: &Self) -> T {
            assert_eq!(self.len(), rhs.len(), "dotc: incompatible lengths");
            crate::metal::operator_dotc(self.storage(), rhs.storage())
        }
    }

    impl<T: Field + OperatorElement> Adjoint for Matrix<T, Metal> {
        type Output = Self;
        fn adjoint(&self) -> Self {
            Self::from_storage(
                self.cols(),
                self.rows(),
                crate::metal::operator_adjoint(self.storage(), self.shape()),
            )
        }
    }

    impl<T: Field + OperatorElement> Dotc for Matrix<T, Metal> {
        type Output = T;
        fn dotc(&self, rhs: &Self) -> T {
            assert_eq!(
                self.shape(),
                rhs.shape(),
                "dotc: incompatible matrix shapes"
            );
            crate::metal::operator_dotc(self.storage(), rhs.storage())
        }
    }

    impl<T: Field + OperatorElement> Dotc for Tensor<T, Metal> {
        type Output = T;
        fn dotc(&self, rhs: &Self) -> T {
            assert_eq!(
                self.shape(),
                rhs.shape(),
                "dotc: incompatible tensor shapes"
            );
            self.as_vector().dotc(rhs.as_vector())
        }
    }

    impl<T: Field + OperatorElement> Adjoint for Tensor<T, Metal> {
        type Output = Self;
        fn adjoint(&self) -> Self {
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
