use super::{Cross, Host, Matrix, Tensor, Vector};
use crate::numbers::Coefficient;

fn cross<T: Coefficient>(a: &[T], b: &[T]) -> Vec<T> {
    a.chunks_exact(3)
        .zip(b.chunks_exact(3))
        .flat_map(|(a, b)| {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        })
        .collect()
}
impl<'b, T: Coefficient> Cross<&'b Vector<T, Host>> for &Vector<T, Host> {
    type Output = Vector<T, Host>;
    fn cross(self, rhs: &'b Vector<T, Host>) -> Self::Output {
        assert_eq!((self.len(), rhs.len()), (3, 3), "cross requires 3-vectors");
        Vector::new(cross(self.as_slice(), rhs.as_slice()))
    }
}
impl<'b, T: Coefficient> Cross<&'b Matrix<T, Host>> for &Matrix<T, Host> {
    type Output = Matrix<T, Host>;
    fn cross(self, rhs: &'b Matrix<T, Host>) -> Self::Output {
        assert_eq!(self.shape(), rhs.shape(), "cross shape mismatch");
        assert_eq!(self.cols(), 3, "cross requires a last axis of three");
        Matrix::from_flat(self.rows(), 3, cross(self.as_slice(), rhs.as_slice()))
    }
}
impl<'b, T: Coefficient> Cross<&'b Tensor<T, Host>> for &Tensor<T, Host> {
    type Output = Tensor<T, Host>;
    fn cross(self, rhs: &'b Tensor<T, Host>) -> Self::Output {
        assert_eq!(self.shape(), rhs.shape(), "cross shape mismatch");
        assert_eq!(
            self.shape().last(),
            Some(&3),
            "cross requires a last axis of three"
        );
        Tensor::from_vec(self.shape(), cross(self.as_slice(), rhs.as_slice()))
    }
}
#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal {
    use super::*;
    use crate::{
        metal::{CrossElement, cross_values},
        tensors::Metal,
    };
    impl<'b, T: CrossElement> Cross<&'b Vector<T, Metal>> for &Vector<T, Metal> {
        type Output = Vector<T, Metal>;
        fn cross(self, rhs: &'b Vector<T, Metal>) -> Self::Output {
            assert_eq!((self.len(), rhs.len()), (3, 3), "cross requires 3-vectors");
            Vector::from_storage(3, cross_values(self.storage(), rhs.storage()))
        }
    }
    impl<'b, T: CrossElement> Cross<&'b Matrix<T, Metal>> for &Matrix<T, Metal> {
        type Output = Matrix<T, Metal>;
        fn cross(self, rhs: &'b Matrix<T, Metal>) -> Self::Output {
            assert_eq!(self.shape(), rhs.shape(), "cross shape mismatch");
            assert_eq!(self.cols(), 3, "cross requires a last axis of three");
            Matrix::from_storage(self.rows(), 3, cross_values(self.storage(), rhs.storage()))
        }
    }
    impl<'b, T: CrossElement> Cross<&'b Tensor<T, Metal>> for &Tensor<T, Metal> {
        type Output = Tensor<T, Metal>;
        fn cross(self, rhs: &'b Tensor<T, Metal>) -> Self::Output {
            assert_eq!(self.shape(), rhs.shape(), "cross shape mismatch");
            assert_eq!(
                self.shape().last(),
                Some(&3),
                "cross requires a last axis of three"
            );
            Tensor::from_vector(
                self.shape(),
                Vector::from_storage(
                    self.len(),
                    cross_values(self.as_vector().storage(), rhs.as_vector().storage()),
                ),
            )
        }
    }
}
