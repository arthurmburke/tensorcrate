use super::{QuadratureRule, SmallMatrix, SmallVector};
use crate::{errors::Error, numbers::Real};

/// A constant nonsingular Jacobian. Differential mappings preserve its signed
/// determinant; integration uses the absolute determinant.
#[derive(Clone, Copy, Debug)]
pub struct Jacobian<T, const N: usize> {
    matrix: SmallMatrix<T, N, N>,
    inverse: SmallMatrix<T, N, N>,
    determinant: T,
}

impl<T: Real, const N: usize> Jacobian<T, N> {
    pub fn new(matrix: SmallMatrix<T, N, N>) -> Result<Self, Error> {
        let lu = matrix.lu()?;
        let determinant = lu.determinant();
        if determinant.is_zero() || !determinant.is_finite() {
            return Err(Error::InvalidArgument(
                "Jacobian determinant is outside the representable nonzero range".into(),
            ));
        }
        Ok(Self {
            matrix,
            inverse: lu.inverse()?,
            determinant,
        })
    }
    pub fn matrix(&self) -> &SmallMatrix<T, N, N> {
        &self.matrix
    }
    pub fn inverse(&self) -> &SmallMatrix<T, N, N> {
        &self.inverse
    }
    pub fn determinant(&self) -> T {
        self.determinant
    }
    pub fn measure_scale(&self) -> T {
        self.determinant.abs()
    }
    pub fn map_vector(&self, v: &SmallVector<T, N>) -> SmallVector<T, N> {
        self.matrix.matvec(v)
    }
    pub fn map_gradient(&self, v: &SmallVector<T, N>) -> SmallVector<T, N> {
        self.inverse.transpose().matvec(v)
    }
    /// Covariant Piola map for H(curl) vector fields.
    pub fn covariant_piola(&self, v: &SmallVector<T, N>) -> SmallVector<T, N> {
        self.map_gradient(v)
    }
    /// Contravariant Piola map for H(div) vector fields.
    pub fn contravariant_piola(&self, v: &SmallVector<T, N>) -> SmallVector<T, N> {
        self.map_vector(v) / self.determinant
    }
    /// Oriented cofactor map of an area normal.
    pub fn map_area_normal(&self, v: &SmallVector<T, N>) -> SmallVector<T, N> {
        self.map_gradient(v) * self.determinant
    }
}
impl<T: Real> Jacobian<T, 3> {
    /// Curl of the covariantly mapped field on an affine cell.
    pub fn map_curl(&self, curl: &SmallVector<T, 3>) -> SmallVector<T, 3> {
        self.map_vector(curl) / self.determinant
    }
}

/// Affine map of the reference simplex (nonnegative coordinates summing to at
/// most one). Negative cell orientations are supported explicitly.
#[derive(Clone, Copy, Debug)]
pub struct AffineSimplex<T, const N: usize> {
    origin: SmallVector<T, N>,
    jacobian: Jacobian<T, N>,
}
impl<T: Real, const N: usize> AffineSimplex<T, N> {
    pub fn new(
        origin: SmallVector<T, N>,
        endpoints: [SmallVector<T, N>; N],
    ) -> Result<Self, Error> {
        if origin.0.iter().any(|v| !v.is_finite()) {
            return Err(Error::InvalidArgument("nonfinite simplex origin".into()));
        }
        let jacobian = Jacobian::new(SmallMatrix::from_columns(endpoints.map(|v| v - origin)))?;
        Ok(Self { origin, jacobian })
    }
    pub fn jacobian(&self) -> &Jacobian<T, N> {
        &self.jacobian
    }
    pub fn map_point(&self, reference: SmallVector<T, N>) -> SmallVector<T, N> {
        self.origin + self.jacobian.map_vector(&reference)
    }
    pub fn reference_point(&self, physical: SmallVector<T, N>) -> SmallVector<T, N> {
        self.jacobian.inverse.matvec(&(physical - self.origin))
    }
    pub fn measure(&self) -> T {
        self.jacobian.measure_scale() / (1..=N).fold(T::one(), |s, i| s * T::from_f64(i as f64))
    }
}
impl<T: Real> AffineSimplex<T, 2> {
    pub fn from_vertices(v: [SmallVector<T, 2>; 3]) -> Result<Self, Error> {
        Self::new(v[0], [v[1], v[2]])
    }
    pub fn barycentric(&self, physical: SmallVector<T, 2>) -> [T; 3] {
        let r = self.reference_point(physical);
        [T::one() - r[0] - r[1], r[0], r[1]]
    }
}
impl<T: Real> AffineSimplex<T, 3> {
    pub fn from_vertices(v: [SmallVector<T, 3>; 4]) -> Result<Self, Error> {
        Self::new(v[0], [v[1], v[2], v[3]])
    }
    pub fn barycentric(&self, physical: SmallVector<T, 3>) -> [T; 4] {
        let r = self.reference_point(physical);
        [T::one() - r[0] - r[1] - r[2], r[0], r[1], r[2]]
    }
}

/// A planar triangle embedded in 3D, for boundary-face integration.
#[derive(Clone, Copy, Debug)]
pub struct Triangle3<T> {
    origin: SmallVector<T, 3>,
    edges: [SmallVector<T, 3>; 2],
    normal: SmallVector<T, 3>,
    scale: T,
}
impl<T: Real> Triangle3<T> {
    pub fn new(v: [SmallVector<T, 3>; 3]) -> Result<Self, Error> {
        if v.iter().flat_map(|v| v.0).any(|x| !x.is_finite()) {
            return Err(Error::InvalidArgument("nonfinite triangle vertex".into()));
        }
        let edges = [v[1] - v[0], v[2] - v[0]];
        let cross = edges[0].cross(&edges[1]);
        let scale = cross.norm();
        if scale.is_zero() {
            return Err(Error::Singular);
        }
        if !scale.is_finite() {
            return Err(Error::InvalidArgument("triangle area overflow".into()));
        }
        Ok(Self {
            origin: v[0],
            edges,
            normal: cross / scale,
            scale,
        })
    }
    pub fn normal(&self) -> SmallVector<T, 3> {
        self.normal
    }
    pub fn area(&self) -> T {
        self.scale / T::from_f64(2.0)
    }
    pub fn map_point(&self, r: SmallVector<T, 2>) -> SmallVector<T, 3> {
        self.origin + self.edges[0] * r[0] + self.edges[1] * r[1]
    }
    pub fn integrate<V>(&self, rule: &QuadratureRule<T, 2>, f: impl Fn(SmallVector<T, 3>) -> V) -> V
    where
        V: num_traits::Zero + std::ops::Add<Output = V> + std::ops::Mul<T, Output = V>,
    {
        rule.integrate(|r| f(self.map_point(r)) * self.scale)
    }
}
