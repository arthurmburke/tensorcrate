//! Statically-shaped vectors and matrices.
//!
//! [`Vector<T, N>`] and [`Matrix<T, R, C>`] carry their dimensions as const
//! generic parameters, so the shapes are part of the type and the compiler
//! checks them. A matrix product `Matrix<R, K> · Matrix<K, C>` only type-checks
//! when the inner dimensions agree; adding a `Matrix<2, 3>` to a `Matrix<3, 2>`
//! is a compile error, not a runtime one. The only operation that can still fail
//! at runtime is [`Matrix::inverse`], because singularity is a property of the
//! values, not the shape.
//!
//! Storage is a fixed-size array (`[T; N]` / `[[T; C]; R]`), so these live on the
//! stack and are `Copy` when `T` is. `+ - * /` are elementwise; the
//! linear-algebra products are the named methods.

use std::cmp::Ordering;
use std::fmt::{self, Display};
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

use crate::errors::Error;
use crate::numbers::Coefficient;

// ---- vectors ----------------------------------------------------------------

/// A length-`N` vector, backed by `[T; N]`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Vector<T, const N: usize> {
    data: [T; N],
}

impl<T, const N: usize> Vector<T, N> {
    /// A vector from its elements.
    pub const fn new(data: [T; N]) -> Self {
        Vector { data }
    }

    pub fn data(&self) -> &[T; N] {
        &self.data
    }

    pub fn get(&self, i: usize) -> Option<&T> {
        self.data.get(i)
    }

    pub const fn len(&self) -> usize {
        N
    }

    pub const fn is_empty(&self) -> bool {
        N == 0
    }

    /// Apply `f` to every element, producing a vector of the new element type —
    /// e.g. lifting a `Vector<f64, N>` into a `Vector<Complex<f64>, N>`.
    pub fn map<U>(&self, f: impl Fn(&T) -> U) -> Vector<U, N> {
        Vector {
            data: std::array::from_fn(|i| f(&self.data[i])),
        }
    }
}

impl<T: Coefficient, const N: usize> Vector<T, N> {
    pub fn zeros() -> Self {
        Vector {
            data: std::array::from_fn(|_| T::zero()),
        }
    }

    /// Multiply every element by `scalar`.
    pub fn scale(&self, scalar: T) -> Self {
        self.map(|&x| x * scalar)
    }

    /// Dot product with a vector of the same length — the length match is
    /// enforced by the type.
    pub fn dot(&self, other: &Vector<T, N>) -> T {
        let mut sum = T::zero();
        for i in 0..N {
            sum = sum + self.data[i] * other.data[i];
        }
        sum
    }

    /// Row vector times matrix: `(1×N)·(N×C) = (1×C)`.
    pub fn vecmat<const C: usize>(&self, m: &Matrix<T, N, C>) -> Vector<T, C> {
        Vector {
            data: std::array::from_fn(|j| {
                let mut sum = T::zero();
                for p in 0..N {
                    sum = sum + self.data[p] * m.data[p][j];
                }
                sum
            }),
        }
    }
}

// ---- matrices ---------------------------------------------------------------

/// An `R × C` matrix, backed by `[[T; C]; R]` in row-major order.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Matrix<T, const R: usize, const C: usize> {
    data: [[T; C]; R],
}

impl<T, const R: usize, const C: usize> Matrix<T, R, C> {
    /// A matrix from its rows.
    pub const fn from_rows(data: [[T; C]; R]) -> Self {
        Matrix { data }
    }

    pub fn data(&self) -> &[[T; C]; R] {
        &self.data
    }

    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        self.data.get(row)?.get(col)
    }

    pub const fn shape(&self) -> (usize, usize) {
        (R, C)
    }

    /// Apply `f` to every element, producing a matrix of the new element type.
    pub fn map<U>(&self, f: impl Fn(&T) -> U) -> Matrix<U, R, C> {
        Matrix {
            data: std::array::from_fn(|i| std::array::from_fn(|j| f(&self.data[i][j]))),
        }
    }
}

impl<T: Coefficient, const R: usize, const C: usize> Matrix<T, R, C> {
    pub fn zeros() -> Self {
        Matrix {
            data: std::array::from_fn(|_| std::array::from_fn(|_| T::zero())),
        }
    }

    /// Multiply every element by `scalar`.
    pub fn scale(&self, scalar: T) -> Self {
        self.map(|&x| x * scalar)
    }

    /// Matrix product `(R×C)·(C×C2) = (R×C2)`. The shared inner dimension `C` is
    /// enforced by the type: a mismatch does not compile.
    pub fn matmul<const C2: usize>(&self, other: &Matrix<T, C, C2>) -> Matrix<T, R, C2> {
        Matrix {
            data: std::array::from_fn(|i| {
                std::array::from_fn(|j| {
                    let mut sum = T::zero();
                    for p in 0..C {
                        sum = sum + self.data[i][p] * other.data[p][j];
                    }
                    sum
                })
            }),
        }
    }

    /// Matrix times column vector: `(R×C)·(C×1) = (R×1)`.
    pub fn matvec(&self, v: &Vector<T, C>) -> Vector<T, R> {
        Vector {
            data: std::array::from_fn(|i| {
                let mut sum = T::zero();
                for p in 0..C {
                    sum = sum + self.data[i][p] * v.data[p];
                }
                sum
            }),
        }
    }

    /// Transpose: an `R×C` matrix becomes `C×R`.
    pub fn transpose(&self) -> Matrix<T, C, R> {
        Matrix {
            data: std::array::from_fn(|i| std::array::from_fn(|j| self.data[j][i])),
        }
    }
}

impl<T: Coefficient, const N: usize> Matrix<T, N, N> {
    /// The `N × N` identity matrix.
    pub fn identity() -> Self {
        Matrix {
            data: std::array::from_fn(|i| {
                std::array::from_fn(|j| if i == j { T::one() } else { T::zero() })
            }),
        }
    }

    /// Determinant, by fraction-free (Bareiss) elimination — every division is
    /// exact, so an integer matrix keeps an exact integer determinant.
    pub fn determinant(&self) -> T {
        if N == 0 {
            return T::one();
        }
        let mut m = self.data;
        let mut prev = T::one();
        let mut negate = false;

        for k in 0..N - 1 {
            if m[k][k].is_zero() {
                match (k + 1..N).find(|&p| !m[p][k].is_zero()) {
                    Some(p) => {
                        m.swap(k, p);
                        negate = !negate;
                    }
                    None => return T::zero(),
                }
            }
            for i in k + 1..N {
                for j in k + 1..N {
                    let value = m[i][j] * m[k][k] - m[i][k] * m[k][j];
                    m[i][j] = value / prev;
                }
            }
            prev = m[k][k];
        }

        let det = m[N - 1][N - 1];
        if negate { T::zero() - det } else { det }
    }

    /// Inverse, by Gauss–Jordan elimination with partial pivoting on
    /// [`Coefficient::magnitude`] (so complex and dual elements, which have no
    /// ordering, still pivot sensibly). Returns [`Error::Singular`] when the
    /// matrix has no inverse.
    pub fn inverse(&self) -> Result<Self, Error> {
        let mut a = self.data;
        let mut inv = Self::identity().data;

        for col in 0..N {
            let pivot = (col..N)
                .max_by(|&x, &y| {
                    a[x][col]
                        .magnitude()
                        .partial_cmp(&a[y][col].magnitude())
                        .unwrap_or(Ordering::Equal)
                })
                .expect("col < N, so the range is non-empty");
            if a[pivot][col].is_zero() {
                return Err(Error::Singular);
            }
            if pivot != col {
                a.swap(col, pivot);
                inv.swap(col, pivot);
            }

            let scale = a[col][col];
            for j in 0..N {
                a[col][j] = a[col][j] / scale;
                inv[col][j] = inv[col][j] / scale;
            }
            for r in 0..N {
                if r == col {
                    continue;
                }
                let factor = a[r][col];
                if factor.is_zero() {
                    continue;
                }
                for j in 0..N {
                    a[r][j] = a[r][j] - factor * a[col][j];
                    inv[r][j] = inv[r][j] - factor * inv[col][j];
                }
            }
        }
        Ok(Matrix { data: inv })
    }
}

// ---- elementwise operators --------------------------------------------------

macro_rules! elementwise {
    ($Type:ident < $($dim:ident),+ >, $Trait:ident, $method:ident, $op:tt) => {
        impl<T: Coefficient, $(const $dim: usize),+> $Trait for $Type<T, $($dim),+> {
            type Output = $Type<T, $($dim),+>;
            fn $method(self, rhs: Self) -> Self::Output {
                self.zip_with(&rhs, |a, b| a $op b)
            }
        }
    };
}

impl<T: Coefficient, const N: usize> Vector<T, N> {
    fn zip_with(&self, rhs: &Self, f: impl Fn(T, T) -> T) -> Self {
        Vector {
            data: std::array::from_fn(|i| f(self.data[i], rhs.data[i])),
        }
    }
}

impl<T: Coefficient, const R: usize, const C: usize> Matrix<T, R, C> {
    fn zip_with(&self, rhs: &Self, f: impl Fn(T, T) -> T) -> Self {
        Matrix {
            data: std::array::from_fn(|i| {
                std::array::from_fn(|j| f(self.data[i][j], rhs.data[i][j]))
            }),
        }
    }
}

elementwise!(Vector<N>, Add, add, +);
elementwise!(Vector<N>, Sub, sub, -);
elementwise!(Vector<N>, Mul, mul, *);
elementwise!(Vector<N>, Div, div, /);
elementwise!(Vector<N>, Rem, rem, %);
elementwise!(Matrix<R, C>, Add, add, +);
elementwise!(Matrix<R, C>, Sub, sub, -);
elementwise!(Matrix<R, C>, Mul, mul, *);
elementwise!(Matrix<R, C>, Div, div, /);
elementwise!(Matrix<R, C>, Rem, rem, %);

impl<T: Coefficient + Neg<Output = T>, const N: usize> Neg for Vector<T, N> {
    type Output = Vector<T, N>;
    fn neg(self) -> Self {
        self.map(|&x| -x)
    }
}

impl<T: Coefficient + Neg<Output = T>, const R: usize, const C: usize> Neg for Matrix<T, R, C> {
    type Output = Matrix<T, R, C>;
    fn neg(self) -> Self {
        self.map(|&x| -x)
    }
}

// ---- display ----------------------------------------------------------------

impl<T: Display, const N: usize> Display for Vector<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for x in &self.data {
            write!(f, " {x}")?;
        }
        write!(f, " ]")
    }
}

impl<T: Display, const R: usize, const C: usize> Display for Matrix<T, R, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (r, row) in self.data.iter().enumerate() {
            if r > 0 {
                writeln!(f)?;
            }
            write!(f, "[")?;
            for x in row {
                write!(f, " {x}")?;
            }
            write!(f, " ]")?;
        }
        Ok(())
    }
}
