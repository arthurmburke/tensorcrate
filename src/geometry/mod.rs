//! Stack-allocated geometry and reference-cell integration for finite elements.

mod mapping;
mod quadrature;
mod small;

pub use mapping::{AffineSimplex, Jacobian, Triangle3};
pub use quadrature::QuadratureRule;
pub use small::{SmallLu, SmallMatrix, SmallVector};

pub type Vec2<T = f64> = SmallVector<T, 2>;
pub type Vec3<T = f64> = SmallVector<T, 3>;
pub type Mat2<T = f64> = SmallMatrix<T, 2, 2>;
pub type Mat3<T = f64> = SmallMatrix<T, 3, 3>;
pub type AffineTriangle<T = f64> = AffineSimplex<T, 2>;
pub type AffineTetrahedron<T = f64> = AffineSimplex<T, 3>;

pub use crate::tensors::algebra::Cross;
