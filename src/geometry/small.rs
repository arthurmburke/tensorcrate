use super::Cross;
use crate::{
    errors::Error,
    numbers::{Coefficient, Conj, Real},
    tensors::{Adjoint, Determinant, Dot, Dotc, Inverse, MatMul, MatVec, Transpose},
};
use std::ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub};

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SmallVector<T, const N: usize>(pub [T; N]);

impl<T, const N: usize> SmallVector<T, N> {
    pub const fn new(values: [T; N]) -> Self {
        Self(values)
    }
    pub fn as_array(&self) -> &[T; N] {
        &self.0
    }
    pub fn map<U>(self, f: impl FnMut(T) -> U) -> SmallVector<U, N> {
        SmallVector(self.0.map(f))
    }
}
impl<T, const N: usize> From<[T; N]> for SmallVector<T, N> {
    fn from(values: [T; N]) -> Self {
        Self(values)
    }
}
impl<T, const N: usize> Index<usize> for SmallVector<T, N> {
    type Output = T;
    fn index(&self, i: usize) -> &T {
        &self.0[i]
    }
}
impl<T, const N: usize> IndexMut<usize> for SmallVector<T, N> {
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.0[i]
    }
}
impl<T: Coefficient, const N: usize> SmallVector<T, N> {
    pub fn zeros() -> Self {
        Self([T::zero(); N])
    }
    pub fn dot(&self, rhs: &Self) -> T {
        self.0
            .iter()
            .zip(&rhs.0)
            .fold(T::zero(), |s, (&a, &b)| s + a * b)
    }
}
impl<T: Coefficient> SmallVector<T, 3> {
    pub fn cross(&self, rhs: &Self) -> Self {
        Self([
            self[1] * rhs[2] - self[2] * rhs[1],
            self[2] * rhs[0] - self[0] * rhs[2],
            self[0] * rhs[1] - self[1] * rhs[0],
        ])
    }
}
impl<T: Real, const N: usize> SmallVector<T, N> {
    /// Euclidean norm with scaling to avoid intermediate squared overflow.
    pub fn norm(&self) -> T {
        if self.0.iter().any(|v| v.is_nan()) {
            return T::nan();
        }
        let scale = self.0.iter().fold(T::zero(), |s, v| s.max(v.abs()));
        if scale.is_zero() || !scale.is_finite() {
            return scale;
        }
        scale
            * self
                .0
                .iter()
                .fold(T::zero(), |s, &v| {
                    let v = v / scale;
                    s + v * v
                })
                .sqrt()
    }
    pub fn normalized(&self) -> Result<Self, Error> {
        let scale = self.0.iter().fold(T::zero(), |s, v| s.max(v.abs()));
        if scale.is_zero() || self.0.iter().any(|v| !v.is_finite()) {
            return Err(Error::InvalidArgument(
                "normalization requires a finite nonzero vector".into(),
            ));
        }
        let scaled = *self / scale;
        Ok(scaled / scaled.norm())
    }
}
impl<T: Coefficient, const N: usize> Add for SmallVector<T, N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(std::array::from_fn(|i| self[i] + rhs[i]))
    }
}
impl<T: Coefficient, const N: usize> Sub for SmallVector<T, N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(std::array::from_fn(|i| self[i] - rhs[i]))
    }
}
impl<T: Coefficient, const N: usize> Add<&SmallVector<T, N>> for &SmallVector<T, N> {
    type Output = SmallVector<T, N>;
    fn add(self, rhs: &SmallVector<T, N>) -> Self::Output {
        *self + *rhs
    }
}
impl<T: Coefficient, const N: usize> Sub<&SmallVector<T, N>> for &SmallVector<T, N> {
    type Output = SmallVector<T, N>;
    fn sub(self, rhs: &SmallVector<T, N>) -> Self::Output {
        *self - *rhs
    }
}
impl<T: Coefficient, const N: usize> Mul<T> for SmallVector<T, N> {
    type Output = Self;
    fn mul(self, rhs: T) -> Self {
        self.map(|v| v * rhs)
    }
}
impl<T: Coefficient, const N: usize> Div<T> for SmallVector<T, N> {
    type Output = Self;
    fn div(self, rhs: T) -> Self {
        self.map(|v| v / rhs)
    }
}
impl<T: Coefficient + Neg<Output = T>, const N: usize> Neg for SmallVector<T, N> {
    type Output = Self;
    fn neg(self) -> Self {
        self.map(|v| -v)
    }
}
impl<'b, T: Coefficient, const N: usize> Dot<&'b SmallVector<T, N>> for &SmallVector<T, N> {
    type Output = T;
    fn dot(self, rhs: &'b SmallVector<T, N>) -> T {
        SmallVector::dot(self, rhs)
    }
}
impl<'b, T: Coefficient + Conj<Output = T>, const N: usize> Dotc<&'b SmallVector<T, N>>
    for &SmallVector<T, N>
{
    type Output = T;
    fn dotc(self, rhs: &'b SmallVector<T, N>) -> T {
        self.0
            .iter()
            .zip(&rhs.0)
            .fold(T::zero(), |s, (a, b)| s + a.conj() * *b)
    }
}
impl<'b, T: Coefficient> Cross<&'b SmallVector<T, 3>> for &SmallVector<T, 3> {
    type Output = SmallVector<T, 3>;
    fn cross(self, rhs: &'b SmallVector<T, 3>) -> Self::Output {
        SmallVector::cross(self, rhs)
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SmallMatrix<T, const R: usize, const C: usize>(pub [[T; C]; R]);

impl<T, const R: usize, const C: usize> SmallMatrix<T, R, C> {
    pub const fn new(rows: [[T; C]; R]) -> Self {
        Self(rows)
    }
    pub fn as_array(&self) -> &[[T; C]; R] {
        &self.0
    }
}
impl<T, const R: usize, const C: usize> From<[[T; C]; R]> for SmallMatrix<T, R, C> {
    fn from(rows: [[T; C]; R]) -> Self {
        Self(rows)
    }
}
impl<T, const R: usize, const C: usize> Index<(usize, usize)> for SmallMatrix<T, R, C> {
    type Output = T;
    fn index(&self, (i, j): (usize, usize)) -> &T {
        &self.0[i][j]
    }
}
impl<T, const R: usize, const C: usize> IndexMut<(usize, usize)> for SmallMatrix<T, R, C> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut T {
        &mut self.0[i][j]
    }
}
impl<T: Coefficient, const R: usize, const C: usize> SmallMatrix<T, R, C> {
    pub fn zeros() -> Self {
        Self([[T::zero(); C]; R])
    }
    pub fn from_columns(columns: [SmallVector<T, R>; C]) -> Self {
        Self(std::array::from_fn(|i| {
            std::array::from_fn(|j| columns[j][i])
        }))
    }
    pub fn transpose(&self) -> SmallMatrix<T, C, R> {
        SmallMatrix(std::array::from_fn(|i| {
            std::array::from_fn(|j| self[(j, i)])
        }))
    }
    pub fn matvec(&self, x: &SmallVector<T, C>) -> SmallVector<T, R> {
        SmallVector(std::array::from_fn(|i| {
            (0..C).fold(T::zero(), |s, j| s + self[(i, j)] * x[j])
        }))
    }
    pub fn matmul<const K: usize>(&self, rhs: &SmallMatrix<T, C, K>) -> SmallMatrix<T, R, K> {
        SmallMatrix(std::array::from_fn(|i| {
            std::array::from_fn(|j| (0..C).fold(T::zero(), |s, k| s + self[(i, k)] * rhs[(k, j)]))
        }))
    }
}
impl<T: Coefficient, const N: usize> SmallMatrix<T, N, N> {
    pub fn identity() -> Self {
        Self(std::array::from_fn(|i| {
            std::array::from_fn(|j| if i == j { T::one() } else { T::zero() })
        }))
    }
}
impl<T: Coefficient> SmallMatrix<T, 2, 2> {
    pub fn determinant(&self) -> T {
        self[(0, 0)] * self[(1, 1)] - self[(0, 1)] * self[(1, 0)]
    }
}
impl<T: Coefficient> SmallMatrix<T, 3, 3> {
    pub fn determinant(&self) -> T {
        self[(0, 0)] * (self[(1, 1)] * self[(2, 2)] - self[(1, 2)] * self[(2, 1)])
            - self[(0, 1)] * (self[(1, 0)] * self[(2, 2)] - self[(1, 2)] * self[(2, 0)])
            + self[(0, 2)] * (self[(1, 0)] * self[(2, 1)] - self[(1, 1)] * self[(2, 0)])
    }
}
impl<T: Coefficient, const R: usize, const C: usize> Transpose for &SmallMatrix<T, R, C> {
    type Output = SmallMatrix<T, C, R>;
    fn transpose(self) -> Self::Output {
        SmallMatrix::transpose(self)
    }
}
impl<'b, T: Coefficient, const R: usize, const C: usize, const K: usize>
    MatMul<&'b SmallMatrix<T, C, K>> for &SmallMatrix<T, R, C>
{
    type Output = SmallMatrix<T, R, K>;
    fn matmul(self, rhs: &'b SmallMatrix<T, C, K>) -> Self::Output {
        SmallMatrix::matmul(self, rhs)
    }
}
impl<'b, T: Coefficient, const R: usize, const C: usize> MatVec<&'b SmallVector<T, C>>
    for &SmallMatrix<T, R, C>
{
    type Output = SmallVector<T, R>;
    fn matvec(self, rhs: &'b SmallVector<T, C>) -> Self::Output {
        SmallMatrix::matvec(self, rhs)
    }
}
impl<T: Coefficient + Conj<Output = T>, const R: usize, const C: usize> Adjoint
    for &SmallMatrix<T, R, C>
{
    type Output = SmallMatrix<T, C, R>;
    fn adjoint(self) -> Self::Output {
        SmallMatrix(std::array::from_fn(|i| {
            std::array::from_fn(|j| self[(j, i)].conj())
        }))
    }
}

impl<T: Coefficient> Determinant for &SmallMatrix<T, 2, 2> {
    type Output = T;
    fn determinant(self) -> T {
        SmallMatrix::<T, 2, 2>::determinant(self)
    }
}
impl<T: Coefficient> Determinant for &SmallMatrix<T, 3, 3> {
    type Output = T;
    fn determinant(self) -> T {
        SmallMatrix::<T, 3, 3>::determinant(self)
    }
}
impl<T: Real, const N: usize> Inverse for &SmallMatrix<T, N, N> {
    type Output = Result<SmallMatrix<T, N, N>, Error>;
    fn inverse(self) -> Self::Output {
        SmallMatrix::inverse(self)
    }
}

/// Reusable scaled-partial-pivoting LU with relative pivot rejection.
#[derive(Clone, Copy, Debug)]
pub struct SmallLu<T, const N: usize> {
    lu: SmallMatrix<T, N, N>,
    permutation: [usize; N],
    odd: bool,
}

impl<T: Real, const N: usize> SmallMatrix<T, N, N> {
    pub fn lu(&self) -> Result<SmallLu<T, N>, Error> {
        SmallLu::factor(*self, T::epsilon() * T::from_f64(N.max(1) as f64))
    }
    pub fn solve(&self, rhs: &SmallVector<T, N>) -> Result<SmallVector<T, N>, Error> {
        self.lu()?.solve(rhs)
    }
    pub fn inverse(&self) -> Result<Self, Error> {
        self.lu()?.inverse()
    }
}
impl<T: Real, const N: usize> SmallLu<T, N> {
    pub fn factor(mut lu: SmallMatrix<T, N, N>, relative_tolerance: T) -> Result<Self, Error> {
        if !relative_tolerance.is_finite()
            || relative_tolerance < T::zero()
            || lu.0.iter().flatten().any(|v| !v.is_finite())
        {
            return Err(Error::InvalidArgument(
                "LU requires finite coefficients and a nonnegative finite tolerance".into(),
            ));
        }
        let mut scales: [T; N] =
            std::array::from_fn(|i| lu.0[i].iter().fold(T::zero(), |s, v| s.max(v.abs())));
        let mut permutation = std::array::from_fn(|i| i);
        let mut odd = false;
        for k in 0..N {
            let mut pivot = k;
            let mut best = T::zero();
            for i in k..N {
                let score = if scales[i].is_zero() {
                    T::zero()
                } else {
                    lu[(i, k)].abs() / scales[i]
                };
                if score > best {
                    best = score;
                    pivot = i;
                }
            }
            if best <= relative_tolerance || lu[(pivot, k)].is_zero() {
                return Err(Error::Singular);
            }
            if pivot != k {
                lu.0.swap(k, pivot);
                scales.swap(k, pivot);
                permutation.swap(k, pivot);
                odd = !odd;
            }
            for i in k + 1..N {
                lu[(i, k)] = lu[(i, k)] / lu[(k, k)];
                for j in k + 1..N {
                    lu[(i, j)] = lu[(i, j)] - lu[(i, k)] * lu[(k, j)];
                }
            }
            if lu.0.iter().flatten().any(|v| !v.is_finite()) {
                return Err(Error::InvalidArgument("LU arithmetic overflow".into()));
            }
        }
        Ok(Self {
            lu,
            permutation,
            odd,
        })
    }
    pub fn solve(&self, rhs: &SmallVector<T, N>) -> Result<SmallVector<T, N>, Error> {
        if rhs.0.iter().any(|v| !v.is_finite()) {
            return Err(Error::InvalidArgument("nonfinite right hand side".into()));
        }
        let mut x = SmallVector(std::array::from_fn(|i| rhs[self.permutation[i]]));
        for i in 0..N {
            for j in 0..i {
                x[i] = x[i] - self.lu[(i, j)] * x[j];
            }
        }
        for i in (0..N).rev() {
            for j in i + 1..N {
                x[i] = x[i] - self.lu[(i, j)] * x[j];
            }
            x[i] = x[i] / self.lu[(i, i)];
        }
        if x.0.iter().any(|v| !v.is_finite()) {
            return Err(Error::InvalidArgument("LU solution overflow".into()));
        }
        Ok(x)
    }
    pub fn inverse(&self) -> Result<SmallMatrix<T, N, N>, Error> {
        let mut inverse = SmallMatrix::zeros();
        for j in 0..N {
            let mut e = SmallVector::zeros();
            e[j] = T::one();
            let column = self.solve(&e)?;
            for i in 0..N {
                inverse[(i, j)] = column[i];
            }
        }
        Ok(inverse)
    }
    pub fn determinant(&self) -> T {
        (0..N).fold(if self.odd { -T::one() } else { T::one() }, |s, i| {
            s * self.lu[(i, i)]
        })
    }
}
