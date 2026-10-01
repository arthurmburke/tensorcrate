//! Forward-mode automatic differentiation over tensors.
//!
//! A [`Dual`] scalar carries `value + tangent·ε` in one struct. A dual *tensor*
//! could do the same — `Vector<Dual<f64>>` already works on the host — but that
//! layout is wrong for a GPU: the Metal shaders work on plain floats, so dual
//! elements would need a second, parallel set of kernels, and the interleaved
//! `[v, d, v, d, …]` access pattern coalesces worse than two flat buffers.
//!
//! So [`DualVector`] and [`DualMatrix`] keep the two parts in *separate* tensors
//! on the same backend, and every rule is expressed with the [`Kernels`] that
//! already exist, over any [`Real`] element type (`f32` unless you say
//! otherwise — [`DualVector<B, T>`] names it):
//!
//! | operation | value | tangent |
//! |---|---|---|
//! | `a + b` | `a + b` | `ȧ + ḃ` |
//! | `a * b` (elementwise) | `a ⊙ b` | `ȧ ⊙ b + a ⊙ ḃ` |
//! | `a / b` | `a / b` | `(ȧ ⊙ b − a ⊙ ḃ) / b²` |
//! | [`matmul`](DualMatrix::matmul) | `AB` | `ȦB + AḂ` |
//! | [`dot`](DualVector::dot) | `u·v` | `u̇·v + u·v̇` |
//! | [`analytic`](DualVector::analytic) | `f(a)` | `f'(a) ⊙ ȧ` |
//! | [`maximum`](DualVector::maximum) | `max(a, b)` | the winner's tangent, split at a tie |
//! | [`row_sums`](DualMatrix::row_sums) | `A·1` | `Ȧ·1` |
//!
//! Nothing here names a backend, because it is written against
//! [`Kernels`]: the same code runs on `Host` vectors and in
//! `Metal` shared memory, so a resident chain stays resident for the whole
//! derivative computation — and the `Host` instantiation is the oracle the GPU
//! one is tested against.
//!
//! Nothing here names a *dimension* either. The two parts of a dual tensor must
//! agree in shape, which is checked when they are paired, and every rule below
//! then takes its shapes from the operands.
//!
//! The tangent is a *direction*, not "the" derivative: one pass computes the
//! directional derivative `J·v`. Seeding a one-hot direction
//! ([`DualVector::seed`]) gives one column of the Jacobian, which is what
//! [`jacobian`] and [`gradient`] do — one pass per input. That cost is precisely
//! what reverse mode exists to avoid.
//!
//! ```
//! use tensorcrate::tensors::{DualMatrix, Matrix, Vector, gradient};
//!
//! // ∇‖x‖² = 2x
//! let x = Vector::new([1.0f32, 2.0, 3.0]);
//! assert_eq!(gradient(&x, |v| v.dot(v)).to_vec(), [2.0, 4.0, 6.0]);
//!
//! // d/dt (A·A) with A = tI is 2tI, so at t = 3 the tangent is 6I.
//! let a = Matrix::<f32>::identity(2).scale(3.0);
//! let squared = DualMatrix::new(a, Matrix::identity(2)).squared();
//! assert_eq!(squared.value().to_rows(), [[9.0, 0.0], [0.0, 9.0]]);
//! assert_eq!(squared.tangent().to_rows(), [[6.0, 0.0], [0.0, 6.0]]);
//! ```

use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::numbers::{Dual, Real};

use super::{Analytic, BinaryOp, Compare, Host, Kernels, Matrix, Vector};

/// A vector and its tangent, for forward-mode differentiation.
pub struct DualVector<B: Kernels<T> = Host, T: Real = f32> {
    value: Vector<T, B>,
    tangent: Vector<T, B>,
}

/// A matrix and its tangent, for forward-mode differentiation.
pub struct DualMatrix<B: Kernels<T> = Host, T: Real = f32> {
    value: Matrix<T, B>,
    tangent: Matrix<T, B>,
}

impl<B: Kernels<T>, T: Real> Clone for DualVector<B, T>
where
    B::Vector<T>: Clone,
{
    fn clone(&self) -> Self {
        DualVector {
            value: self.value.clone(),
            tangent: self.tangent.clone(),
        }
    }
}

impl<B: Kernels<T>, T: Real> Clone for DualMatrix<B, T>
where
    B::Matrix<T>: Clone,
{
    fn clone(&self) -> Self {
        DualMatrix {
            value: self.value.clone(),
            tangent: self.tangent.clone(),
        }
    }
}

impl<B: Kernels<T>, T: Real> DualVector<B, T> {
    /// A vector paired with the direction to differentiate along.
    ///
    /// # Panics
    ///
    /// If the two lengths differ.
    #[track_caller]
    pub fn new(value: Vector<T, B>, tangent: Vector<T, B>) -> Self {
        assert!(
            value.len() == tangent.len(),
            "DualVector::new: value has {} elements but the tangent has {}",
            value.len(),
            tangent.len()
        );
        DualVector { value, tangent }
    }

    /// A vector held constant: its tangent is zero.
    pub fn constant(value: Vector<T, B>) -> Self {
        let tangent = Vector::filled(value.len(), T::zero());
        DualVector { value, tangent }
    }

    /// A vector seeded to differentiate with respect to element `index`, giving
    /// one column of a Jacobian. An out-of-range index seeds nothing.
    pub fn seed(value: Vector<T, B>, index: usize) -> Self {
        let mut direction = vec![T::zero(); value.len()];
        if let Some(slot) = direction.get_mut(index) {
            *slot = T::one();
        }
        DualVector {
            value,
            tangent: Vector::build(&direction),
        }
    }

    /// The number of elements.
    pub fn len(&self) -> usize {
        self.value.len()
    }

    /// Whether this vector holds no elements.
    pub fn is_empty(&self) -> bool {
        self.value.is_empty()
    }

    pub fn value(&self) -> &Vector<T, B> {
        &self.value
    }

    pub fn tangent(&self) -> &Vector<T, B> {
        &self.tangent
    }

    /// The two parts, by value.
    pub fn into_parts(self) -> (Vector<T, B>, Vector<T, B>) {
        (self.value, self.tangent)
    }

    /// Dot product with another dual vector: `u·v + (u̇·v + u·v̇)ε`.
    pub fn dot(&self, other: &Self) -> Dual<T> {
        Dual::new(
            B::dot(&self.value, &other.value),
            B::dot(&self.tangent, &other.value) + B::dot(&self.value, &other.tangent),
        )
    }

    /// The sum of the elements, differentiated — the usual way to reduce to a
    /// scalar loss.
    pub fn sum(&self) -> Dual<T> {
        Dual::new(sum(self.value.as_slice()), sum(self.tangent.as_slice()))
    }

    /// Row vector times matrix, `(1×N)·(N×C)`, differentiated.
    pub fn vecmat(&self, m: &DualMatrix<B, T>) -> DualVector<B, T> {
        DualVector {
            value: B::vecmat(&self.value, &m.value),
            tangent: B::vector_elementwise(
                &B::vecmat(&self.tangent, &m.value),
                &B::vecmat(&self.value, &m.tangent),
                BinaryOp::Add,
            ),
        }
    }

    /// Multiply by a constant: both parts scale.
    pub fn scale(&self, scalar: T) -> Self {
        DualVector {
            value: B::vector_broadcast(&self.value, scalar, BinaryOp::Mul, false),
            tangent: B::vector_broadcast(&self.tangent, scalar, BinaryOp::Mul, false),
        }
    }

    /// Add a constant: the tangent is unchanged, since `d/dx (a + c) = ȧ`.
    pub fn shift(&self, scalar: T) -> Self {
        DualVector {
            value: B::vector_broadcast(&self.value, scalar, BinaryOp::Add, false),
            tangent: duplicate_vector(&self.tangent),
        }
    }

    /// Combine with a *dual* scalar — a differentiable parameter — elementwise.
    /// `scalar_left` puts the scalar on the left of noncommutative operations.
    ///
    /// The scalar is expanded into a filled tensor and the elementwise rules do
    /// the rest, so this trades some bandwidth for having exactly one derivation
    /// of each product rule. When the scalar is a constant, prefer [`scale`] and
    /// [`shift`], which use the broadcast kernels directly.
    ///
    /// [`scale`]: Self::scale
    /// [`shift`]: Self::shift
    pub fn broadcast(&self, scalar: Dual<T>, op: BinaryOp, scalar_left: bool) -> Self {
        let expanded = DualVector {
            value: Vector::filled(self.len(), scalar.real),
            tangent: Vector::filled(self.len(), scalar.dual),
        };
        if scalar_left {
            elementwise_vector(&expanded, self, op)
        } else {
            elementwise_vector(self, &expanded, op)
        }
    }

    /// Elementwise larger of two dual vectors.
    ///
    /// The value picks the larger operand and the tangent follows it, split
    /// evenly where they tie — see [`Compare`] for that convention.
    pub fn maximum(&self, other: &Self) -> Self {
        self.select(other, true)
    }

    /// Elementwise smaller of two dual vectors.
    pub fn minimum(&self, other: &Self) -> Self {
        self.select(other, false)
    }

    /// The shared body of [`maximum`](Self::maximum) and
    /// [`minimum`](Self::minimum): the tangent is a convex combination of the two
    /// operands' tangents, weighted by which one the value came from.
    fn select(&self, other: &Self, largest: bool) -> Self {
        let op = if largest { Compare::Max } else { Compare::Min };
        let value = B::vector_compare(&self.value, &other.value, op);
        let share = B::vector_compare(&self.value, &other.value, Compare::MaxShare);
        let complement = B::vector_broadcast(&share, T::one(), BinaryOp::Sub, true);
        let (mine, theirs) = if largest {
            (&share, &complement)
        } else {
            (&complement, &share)
        };
        DualVector {
            value,
            tangent: B::vector_elementwise(
                &B::vector_elementwise(mine, &self.tangent, BinaryOp::Mul),
                &B::vector_elementwise(theirs, &other.tangent, BinaryOp::Mul),
                BinaryOp::Add,
            ),
        }
    }

    /// Elementwise maximum against a constant — `clamp_min(0.0)` is a rectifier.
    pub fn clamp_min(&self, floor: T) -> Self {
        self.select_scalar(floor, true)
    }

    /// Elementwise minimum against a constant.
    pub fn clamp_max(&self, ceiling: T) -> Self {
        self.select_scalar(ceiling, false)
    }

    /// Confine every element to `[floor, ceiling]`.
    pub fn clamp(&self, floor: T, ceiling: T) -> Self {
        self.clamp_min(floor).clamp_max(ceiling)
    }

    /// The rectifier `max(a, 0)`, whose slope is `½` exactly at the kink.
    pub fn relu(&self) -> Self {
        self.clamp_min(T::zero())
    }

    /// Elementwise absolute value, as `max(a, −a)`.
    ///
    /// The tangent works out to `sign(a) ⊙ ȧ`, with `sign(0) = 0` because the tie
    /// splits the subgradient between `+1` and `−1`.
    pub fn abs(&self) -> Self {
        self.maximum(&self.scale(-T::one()))
    }

    fn select_scalar(&self, scalar: T, largest: bool) -> Self {
        let op = if largest { Compare::Max } else { Compare::Min };
        let value = B::vector_compare_scalar(&self.value, scalar, op, false);
        let share = B::vector_compare_scalar(&self.value, scalar, Compare::MaxShare, false);
        // A constant contributes no tangent, so only this operand's share counts.
        let weight = if largest {
            share
        } else {
            B::vector_broadcast(&share, T::one(), BinaryOp::Sub, true)
        };
        DualVector {
            value,
            tangent: B::vector_elementwise(&weight, &self.tangent, BinaryOp::Mul),
        }
    }

    /// Apply an analytic function elementwise: `f(a) + f'(a)⊙ȧ·ε`.
    pub fn analytic(&self, f: Analytic) -> Self {
        let (value, tangent) = B::vector_unary_dual(&self.value, &self.tangent, f);
        DualVector { value, tangent }
    }

    /// Elementwise reciprocal, with `d(1/a) = −ȧ/a²`.
    pub fn recip(&self) -> Self {
        let squared = B::vector_elementwise(&self.value, &self.value, BinaryOp::Mul);
        DualVector {
            value: B::vector_broadcast(&self.value, T::one(), BinaryOp::Div, true),
            tangent: B::vector_broadcast(
                &B::vector_elementwise(&self.tangent, &squared, BinaryOp::Div),
                -T::one(),
                BinaryOp::Mul,
                false,
            ),
        }
    }
}

impl<B: Kernels<T>, T: Real> DualMatrix<B, T> {
    /// A matrix paired with the direction to differentiate along.
    ///
    /// # Panics
    ///
    /// If the two shapes differ.
    #[track_caller]
    pub fn new(value: Matrix<T, B>, tangent: Matrix<T, B>) -> Self {
        assert!(
            value.shape() == tangent.shape(),
            "DualMatrix::new: value is {}×{} but the tangent is {}×{}",
            value.rows(),
            value.cols(),
            tangent.rows(),
            tangent.cols()
        );
        DualMatrix { value, tangent }
    }

    /// A matrix held constant: its tangent is zero.
    pub fn constant(value: Matrix<T, B>) -> Self {
        let (rows, cols) = value.shape();
        DualMatrix {
            value,
            tangent: Matrix::filled(rows, cols, T::zero()),
        }
    }

    /// A matrix seeded to differentiate with respect to element `(row, col)`,
    /// giving one column of a Jacobian. Out-of-range indices seed nothing.
    pub fn seed(value: Matrix<T, B>, row: usize, col: usize) -> Self {
        let (rows, cols) = value.shape();
        let mut direction = vec![T::zero(); rows * cols];
        if row < rows && col < cols {
            direction[row * cols + col] = T::one();
        }
        DualMatrix {
            value,
            tangent: Matrix::build(rows, cols, &direction),
        }
    }

    /// The `(rows, columns)` extents.
    pub fn shape(&self) -> (usize, usize) {
        self.value.shape()
    }

    /// The number of rows.
    pub fn rows(&self) -> usize {
        self.value.rows()
    }

    /// The number of columns.
    pub fn cols(&self) -> usize {
        self.value.cols()
    }

    pub fn value(&self) -> &Matrix<T, B> {
        &self.value
    }

    pub fn tangent(&self) -> &Matrix<T, B> {
        &self.tangent
    }

    /// The two parts, by value.
    pub fn into_parts(self) -> (Matrix<T, B>, Matrix<T, B>) {
        (self.value, self.tangent)
    }

    /// Matrix product, differentiated: `AB + (ȦB + AḂ)ε`.
    ///
    /// Three products and a sum in principle; on a backend that can accumulate
    /// inside the product kernel the sum rides along in the second product, so
    /// the tangent costs two dispatches rather than three.
    pub fn matmul(&self, other: &DualMatrix<B, T>) -> DualMatrix<B, T> {
        DualMatrix {
            value: B::matmul(&self.value, &other.value),
            tangent: B::matmul_add(
                &self.value,
                &other.tangent,
                B::matmul(&self.tangent, &other.value),
            ),
        }
    }

    /// Fused matrix multiply-add, differentiated:
    /// `AB + D + (ȦB + AḂ + Ḋ)ε`.
    pub fn matmul_add(
        &self,
        other: &DualMatrix<B, T>,
        addend: &DualMatrix<B, T>,
    ) -> DualMatrix<B, T> {
        DualMatrix {
            value: B::matmul_add(&self.value, &other.value, duplicate_matrix(&addend.value)),
            tangent: B::matmul_add(
                &self.value,
                &other.tangent,
                B::matmul_add(
                    &self.tangent,
                    &other.value,
                    duplicate_matrix(&addend.tangent),
                ),
            ),
        }
    }

    /// Matrix times column vector, differentiated: `Av + (Ȧv + Av̇)ε`.
    pub fn matvec(&self, v: &DualVector<B, T>) -> DualVector<B, T> {
        DualVector {
            value: B::matvec(&self.value, &v.value),
            tangent: B::vector_elementwise(
                &B::matvec(&self.tangent, &v.value),
                &B::matvec(&self.value, &v.tangent),
                BinaryOp::Add,
            ),
        }
    }

    /// Fused matrix-vector multiply-add, differentiated:
    /// `Av + b + (Ȧv + Av̇ + ḃ)ε`.
    pub fn matvec_add(&self, v: &DualVector<B, T>, addend: &DualVector<B, T>) -> DualVector<B, T> {
        DualVector {
            value: B::matvec_add(&self.value, &v.value, duplicate_vector(&addend.value)),
            tangent: B::matvec_add(
                &self.value,
                &v.tangent,
                B::matvec_add(&self.tangent, &v.value, duplicate_vector(&addend.tangent)),
            ),
        }
    }

    /// Transpose, differentiated: both parts transpose.
    pub fn transpose(&self) -> DualMatrix<B, T> {
        DualMatrix {
            value: B::transpose(&self.value),
            tangent: B::transpose(&self.tangent),
        }
    }

    /// Multiply by a constant: both parts scale.
    pub fn scale(&self, scalar: T) -> Self {
        DualMatrix {
            value: B::matrix_broadcast(&self.value, scalar, BinaryOp::Mul, false),
            tangent: B::matrix_broadcast(&self.tangent, scalar, BinaryOp::Mul, false),
        }
    }

    /// Add a constant: the tangent is unchanged.
    pub fn shift(&self, scalar: T) -> Self {
        DualMatrix {
            value: B::matrix_broadcast(&self.value, scalar, BinaryOp::Add, false),
            tangent: duplicate_matrix(&self.tangent),
        }
    }

    /// Combine with a *dual* scalar elementwise; see
    /// [`DualVector::broadcast`].
    pub fn broadcast(&self, scalar: Dual<T>, op: BinaryOp, scalar_left: bool) -> Self {
        let (rows, cols) = self.shape();
        let expanded = DualMatrix {
            value: Matrix::filled(rows, cols, scalar.real),
            tangent: Matrix::filled(rows, cols, scalar.dual),
        };
        if scalar_left {
            elementwise_matrix(&expanded, self, op)
        } else {
            elementwise_matrix(self, &expanded, op)
        }
    }

    /// Elementwise larger of two dual matrices; see [`DualVector::maximum`].
    pub fn maximum(&self, other: &Self) -> Self {
        self.select(other, true)
    }

    /// Elementwise smaller of two dual matrices.
    pub fn minimum(&self, other: &Self) -> Self {
        self.select(other, false)
    }

    fn select(&self, other: &Self, largest: bool) -> Self {
        let op = if largest { Compare::Max } else { Compare::Min };
        let value = B::matrix_compare(&self.value, &other.value, op);
        let share = B::matrix_compare(&self.value, &other.value, Compare::MaxShare);
        let complement = B::matrix_broadcast(&share, T::one(), BinaryOp::Sub, true);
        let (mine, theirs) = if largest {
            (&share, &complement)
        } else {
            (&complement, &share)
        };
        DualMatrix {
            value,
            tangent: B::matrix_elementwise(
                &B::matrix_elementwise(mine, &self.tangent, BinaryOp::Mul),
                &B::matrix_elementwise(theirs, &other.tangent, BinaryOp::Mul),
                BinaryOp::Add,
            ),
        }
    }

    /// Elementwise maximum against a constant.
    pub fn clamp_min(&self, floor: T) -> Self {
        self.select_scalar(floor, true)
    }

    /// Elementwise minimum against a constant.
    pub fn clamp_max(&self, ceiling: T) -> Self {
        self.select_scalar(ceiling, false)
    }

    /// Confine every element to `[floor, ceiling]`.
    pub fn clamp(&self, floor: T, ceiling: T) -> Self {
        self.clamp_min(floor).clamp_max(ceiling)
    }

    /// The rectifier `max(A, 0)`.
    pub fn relu(&self) -> Self {
        self.clamp_min(T::zero())
    }

    /// Elementwise absolute value; see [`DualVector::abs`].
    pub fn abs(&self) -> Self {
        self.maximum(&self.scale(-T::one()))
    }

    fn select_scalar(&self, scalar: T, largest: bool) -> Self {
        let op = if largest { Compare::Max } else { Compare::Min };
        let value = B::matrix_compare_scalar(&self.value, scalar, op, false);
        let share = B::matrix_compare_scalar(&self.value, scalar, Compare::MaxShare, false);
        let weight = if largest {
            share
        } else {
            B::matrix_broadcast(&share, T::one(), BinaryOp::Sub, true)
        };
        DualMatrix {
            value,
            tangent: B::matrix_elementwise(&weight, &self.tangent, BinaryOp::Mul),
        }
    }

    /// Valid cross-correlation with a dual window, giving
    /// `(R−KR+1) × (C−KC+1)`.
    ///
    /// Correlation is bilinear, exactly like a matrix product, so the tangent is
    /// the same two-term rule: `Ẋ ⋆ K + X ⋆ K̇`.
    pub fn correlate(&self, window: &DualMatrix<B, T>) -> DualMatrix<B, T> {
        self.correlate_with(window, false)
    }

    /// Convolution proper, with the window reversed; see
    /// [`MatrixVar::convolve`](super::tape::MatrixVar::convolve).
    pub fn convolve(&self, window: &DualMatrix<B, T>) -> DualMatrix<B, T> {
        self.correlate_with(window, true)
    }

    fn correlate_with(&self, window: &DualMatrix<B, T>, flip: bool) -> DualMatrix<B, T> {
        DualMatrix {
            value: B::correlate(&self.value, &window.value, flip),
            tangent: B::matrix_elementwise(
                &B::correlate(&self.tangent, &window.value, flip),
                &B::correlate(&self.value, &window.tangent, flip),
                BinaryOp::Add,
            ),
        }
    }

    /// Surround both parts with zeros.
    pub fn pad(&self, pad_rows: usize, pad_cols: usize) -> DualMatrix<B, T> {
        DualMatrix {
            value: B::pad(&self.value, pad_rows, pad_cols),
            tangent: B::pad(&self.tangent, pad_rows, pad_cols),
        }
    }

    /// Reverse both axes of both parts.
    pub fn flipped(&self) -> Self {
        DualMatrix {
            value: B::flip(&self.value),
            tangent: B::flip(&self.tangent),
        }
    }

    /// Sum along each row, giving one entry per row.
    ///
    /// This is `A·1`, so it needs no reduction kernel of its own — and on the
    /// `Metal` backend it stays resident like any other product.
    pub fn row_sums(&self) -> DualVector<B, T> {
        let ones = Vector::<T, B>::filled(self.cols(), T::one());
        DualVector {
            value: B::matvec(&self.value, &ones),
            tangent: B::matvec(&self.tangent, &ones),
        }
    }

    /// Sum along each column, giving one entry per column: `1ᵀ·A`.
    pub fn column_sums(&self) -> DualVector<B, T> {
        let ones = Vector::<T, B>::filled(self.rows(), T::one());
        DualVector {
            value: B::vecmat(&ones, &self.value),
            tangent: B::vecmat(&ones, &self.tangent),
        }
    }

    /// Apply an analytic function elementwise: `f(A) + f'(A)⊙Ȧ·ε`.
    pub fn analytic(&self, f: Analytic) -> Self {
        let (value, tangent) = B::matrix_unary_dual(&self.value, &self.tangent, f);
        DualMatrix { value, tangent }
    }

    /// Elementwise reciprocal, with `d(1/A) = −Ȧ/A²` (not the matrix inverse).
    pub fn recip(&self) -> Self {
        let squared = B::matrix_elementwise(&self.value, &self.value, BinaryOp::Mul);
        DualMatrix {
            value: B::matrix_broadcast(&self.value, T::one(), BinaryOp::Div, true),
            tangent: B::matrix_broadcast(
                &B::matrix_elementwise(&self.tangent, &squared, BinaryOp::Div),
                -T::one(),
                BinaryOp::Mul,
                false,
            ),
        }
    }

    /// The sum of the elements, differentiated.
    pub fn sum(&self) -> Dual<T> {
        Dual::new(sum(self.value.as_slice()), sum(self.tangent.as_slice()))
    }

    /// The Frobenius inner product `Σᵢⱼ aᵢⱼbᵢⱼ = tr(AᵀB)`, differentiated.
    pub fn frobenius_dot(&self, other: &Self) -> Dual<T> {
        elementwise_matrix(self, other, BinaryOp::Mul).sum()
    }

    /// `A·A`, differentiated — the smallest case where the product rule shows up
    /// on both sides.
    ///
    /// # Panics
    ///
    /// If the matrix is not square.
    #[track_caller]
    pub fn squared(&self) -> Self {
        assert!(
            self.rows() == self.cols(),
            "squared: matrix is {}×{}, not square",
            self.rows(),
            self.cols()
        );
        self.matmul(self)
    }
}

/// Sum of a slice, read in place — on a resident tensor this reads shared memory
/// directly, exactly like [`Vector::dot`](super::Vector::dot) on that backend.
fn sum<T: Real>(values: &[T]) -> T {
    if let Some(total) = crate::compact::reduce(values, super::Reduce::Sum) {
        return total;
    }
    values.iter().fold(T::zero(), |total, &value| total + value)
}

/// A second copy of a tensor's storage on the same backend. Moving to the backend
/// it is already on is just that copy — within shared memory, when resident.
fn duplicate_vector<B: Kernels<T>, T: Real>(v: &Vector<T, B>) -> Vector<T, B> {
    v.to_backend::<B>()
}

fn duplicate_matrix<B: Kernels<T>, T: Real>(m: &Matrix<T, B>) -> Matrix<T, B> {
    m.to_backend::<B>()
}

/// The full set of analytic functions, as methods, for both dual tensor types.
macro_rules! analytic_methods {
    ($($method:ident => $variant:ident),+ $(,)?) => {
        impl<B: Kernels<T>, T: Real> DualVector<B, T> {
            $(
                #[doc = concat!("Elementwise `", stringify!($method), "`, differentiated.")]
                pub fn $method(&self) -> Self {
                    self.analytic(Analytic::$variant)
                }
            )+
        }

        impl<B: Kernels<T>, T: Real> DualMatrix<B, T> {
            $(
                #[doc = concat!("Elementwise `", stringify!($method), "`, differentiated.")]
                pub fn $method(&self) -> Self {
                    self.analytic(Analytic::$variant)
                }
            )+
        }
    };
}

analytic_methods!(
    sin => Sin,
    cos => Cos,
    tan => Tan,
    sec => Sec,
    csc => Csc,
    arcsin => Arcsin,
    arccos => Arccos,
    arctan => Arctan,
    exp => Exp,
    ln => Ln,
    sinh => Sinh,
    cosh => Cosh,
    tanh => Tanh,
    sqrt => Sqrt,
);

// ---- interoperating with `Dual` elements ------------------------------------

impl<B: Kernels<T>, T: Real> DualVector<B, T> {
    /// Split a host vector of dual numbers into value and tangent parts on
    /// backend `B`.
    pub fn from_dual_vector(duals: &Vector<Dual<T>, Host>) -> Self {
        let elements = duals.data();
        let value = elements.iter().map(|dual| dual.real).collect::<Vec<_>>();
        let tangent = elements.iter().map(|dual| dual.dual).collect::<Vec<_>>();
        DualVector {
            value: Vector::build(&value),
            tangent: Vector::build(&tangent),
        }
    }

    /// Recombine the two parts into a host vector of dual numbers — the form the
    /// scalar [`Dual`] operations in [`crate::numbers`] work on.
    pub fn to_dual_vector(&self) -> Vector<Dual<T>, Host> {
        let (value, tangent) = (self.value.as_slice(), self.tangent.as_slice());
        Vector::new(
            value
                .iter()
                .zip(tangent)
                .map(|(&v, &t)| Dual::new(v, t))
                .collect::<Vec<_>>(),
        )
    }
}

impl<B: Kernels<T>, T: Real> DualMatrix<B, T> {
    /// Split a host matrix of dual numbers into value and tangent parts on
    /// backend `B`.
    pub fn from_dual_matrix(duals: &Matrix<Dual<T>, Host>) -> Self {
        let (rows, cols) = duals.shape();
        let elements = duals.data();
        let value = elements.iter().map(|dual| dual.real).collect::<Vec<_>>();
        let tangent = elements.iter().map(|dual| dual.dual).collect::<Vec<_>>();
        DualMatrix {
            value: Matrix::build(rows, cols, &value),
            tangent: Matrix::build(rows, cols, &tangent),
        }
    }

    /// Recombine the two parts into a host matrix of dual numbers.
    pub fn to_dual_matrix(&self) -> Matrix<Dual<T>, Host> {
        let (rows, cols) = self.shape();
        let (value, tangent) = (self.value.as_slice(), self.tangent.as_slice());
        Matrix::from_flat(
            rows,
            cols,
            value
                .iter()
                .zip(tangent)
                .map(|(&v, &t)| Dual::new(v, t))
                .collect::<Vec<_>>(),
        )
    }
}

// ---- reshaping --------------------------------------------------------------

/// Row-major flattening, in both directions.
///
/// This is the adapter that makes matrix inputs and outputs work with the vector
/// machinery: a matrix is `rows * cols` scalars, and once they are named as a
/// vector, [`jacobian`] and [`gradient`] apply unchanged. A vector and a matrix
/// are the same run of elements on either backend, so the by-value forms move
/// rather than copy.
impl<B: Kernels<T>, T: Real> DualMatrix<B, T> {
    /// Flatten both parts into a dual vector, consuming this matrix.
    pub fn into_flattened(self) -> DualVector<B, T> {
        let len = self.rows() * self.cols();
        let (value, tangent) = self.into_parts();
        DualVector {
            value: Vector::from_storage(len, B::matrix_into_flattened(value.into_storage())),
            tangent: Vector::from_storage(len, B::matrix_into_flattened(tangent.into_storage())),
        }
    }

    /// Flatten both parts into a dual vector, leaving this matrix intact.
    pub fn flattened(&self) -> DualVector<B, T> {
        self.clone_parts().into_flattened()
    }

    /// Rebuild a `rows × cols` dual matrix from its row-major flattening.
    ///
    /// # Panics
    ///
    /// If the vector's length is not `rows * cols`.
    #[track_caller]
    pub fn from_flattened(rows: usize, cols: usize, vector: DualVector<B, T>) -> Self {
        assert!(
            vector.len() == rows * cols,
            "from_flattened: {} elements cannot fill a {rows}×{cols} matrix",
            vector.len()
        );
        let (value, tangent) = vector.into_parts();
        DualMatrix {
            value: Matrix::from_storage(rows, cols, B::vector_into_matrix(value.into_storage())),
            tangent: Matrix::from_storage(
                rows,
                cols,
                B::vector_into_matrix(tangent.into_storage()),
            ),
        }
    }

    /// A copy of both parts, without requiring the storage to be `Clone`.
    fn clone_parts(&self) -> Self {
        DualMatrix {
            value: duplicate_matrix(&self.value),
            tangent: duplicate_matrix(&self.tangent),
        }
    }
}

impl<B: Kernels<T>, T: Real> DualVector<B, T> {
    /// View as a `1 × N` dual matrix, consuming this vector.
    pub fn into_row(self) -> DualMatrix<B, T> {
        let len = self.len();
        let (value, tangent) = self.into_parts();
        DualMatrix {
            value: Matrix::from_storage(1, len, B::vector_into_matrix(value.into_storage())),
            tangent: Matrix::from_storage(1, len, B::vector_into_matrix(tangent.into_storage())),
        }
    }

    /// View as an `N × 1` dual matrix, consuming this vector.
    pub fn into_column(self) -> DualMatrix<B, T> {
        let len = self.len();
        let (value, tangent) = self.into_parts();
        DualMatrix {
            value: Matrix::from_storage(len, 1, B::vector_into_matrix(value.into_storage())),
            tangent: Matrix::from_storage(len, 1, B::vector_into_matrix(tangent.into_storage())),
        }
    }
}

// ---- Jacobians and gradients ------------------------------------------------

/// The outer product of the gradient operator with the matrix.
///
/// With `f` producing an `OUT1 × OUT2` matrix from a length-`IN` input, the
/// result is `OUT1 × (IN · OUT2)`: columns `[j·OUT2 .. (j+1)·OUT2]` are the
/// derivatives with respect to `xⱼ`.
pub fn matrix_gradient<B: Kernels<T>, T: Real>(
    at: &Vector<T, B>,
    f: impl Fn(&DualVector<B, T>) -> DualMatrix<B, T>,
) -> Matrix<T, B> {
    let mut shape = (0, 0);
    let tangents = (0..at.len())
        .map(|input| {
            let tangent = f(&DualVector::seed(duplicate_vector(at), input)).tangent;
            shape = tangent.shape();
            tangent.into_storage()
        })
        .collect::<Vec<_>>();

    let (rows, cols) = shape;
    Matrix::from_storage(
        rows,
        cols * tangents.len(),
        B::hmerge(&tangents, rows, cols),
    )
}

/// The Jacobian of `f` at `at`, by one forward pass per input element.
///
/// Column `j` is the tangent of `f` seeded along input `j`, so this costs one
/// evaluation of `f` per input and comes out `OUT × IN`. For many inputs and few
/// outputs — a scalar loss over a parameter vector, say — that is the wrong way
/// around, and reverse mode is the answer.
pub fn jacobian<B: Kernels<T>, T: Real>(
    at: &Vector<T, B>,
    f: impl Fn(&DualVector<B, T>) -> DualVector<B, T>,
) -> Matrix<T, B> {
    let mut outputs = 0;
    let tangents = (0..at.len())
        .map(|input| {
            let tangent = f(&DualVector::seed(duplicate_vector(at), input)).tangent;
            outputs = tangent.len();
            tangent.into_storage()
        })
        .collect::<Vec<_>>();

    Matrix::from_storage(outputs, tangents.len(), B::hstack(&tangents, outputs))
}

/// The gradient of a scalar-valued `f` with respect to a *matrix* input.
///
/// The result has the shape of the input, so `∂f/∂Aᵢⱼ` sits at `(i, j)` — this is
/// the derivative meant by "differentiate the loss with respect to the weights".
/// It costs `rows * cols` forward passes, one per element, since each pass
/// extracts a single partial. A `256 × 256` input therefore means 65,536
/// evaluations of `f`; that asymmetry is the argument for reverse mode, not a
/// defect of this implementation.
///
/// ```
/// use tensorcrate::tensors::{DualVector, Matrix, Vector, gradient_wrt_matrix};
///
/// // f(A) = ‖A·x‖², whose gradient is 2(Ax)xᵀ.
/// let a = Matrix::<f32>::from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
/// let x = Vector::new([1.0f32, 0.5, -1.0]);
///
/// let gradient = gradient_wrt_matrix(&a, |m| {
///     let mapped = m.matvec(&DualVector::constant(x.clone()));
///     mapped.dot(&mapped)
/// });
///
/// let projected = a.matvec(&x);
/// for row in 0..2 {
///     for col in 0..3 {
///         let expected = 2.0 * projected[row] * x[col];
///         assert!((gradient[(row, col)] - expected).abs() < 1e-4);
///     }
/// }
/// ```
pub fn gradient_wrt_matrix<B: Kernels<T>, T: Real>(
    at: &Matrix<T, B>,
    f: impl Fn(&DualMatrix<B, T>) -> Dual<T>,
) -> Matrix<T, B> {
    let (rows, cols) = at.shape();
    let stacked = (0..rows)
        .map(|row| {
            let partials = (0..cols)
                .map(|col| f(&DualMatrix::seed(duplicate_matrix(at), row, col)).dual)
                .collect::<Vec<_>>();
            Vector::<T, B>::build(&partials).into_storage()
        })
        .collect::<Vec<_>>();
    Matrix::from_storage(rows, cols, B::vstack(&stacked, cols))
}

/// The Jacobian of a vector-valued `f` with respect to a *matrix* input.
///
/// Column `i * cols + j` holds the derivatives with respect to `Aᵢⱼ`, matching
/// the row-major flattening of the input, so the result is `OUT × (rows·cols)`.
/// Like [`gradient_wrt_matrix`], it runs `rows * cols` passes.
///
/// A matrix-valued `f` needs no separate driver — flatten its output with
/// [`DualMatrix::into_flattened`] and the result is the standard Jacobian,
/// `(OR·OC) × (rows·cols)`:
///
/// ```
/// # use tensorcrate::tensors::{DualMatrix, Matrix, jacobian_wrt_matrix};
/// # let a = Matrix::<f32>::identity(2);
/// # let b = Matrix::<f32>::from_rows([[1.0, 2.0], [3.0, 4.0]]);
/// let jacobian = jacobian_wrt_matrix(&a, |m| {
///     m.matmul(&DualMatrix::constant(b.clone())).into_flattened()
/// });
/// assert_eq!(jacobian.shape(), (4, 4));
/// ```
pub fn jacobian_wrt_matrix<B: Kernels<T>, T: Real>(
    at: &Matrix<T, B>,
    f: impl Fn(&DualMatrix<B, T>) -> DualVector<B, T>,
) -> Matrix<T, B> {
    let (rows, cols) = at.shape();
    let mut outputs = 0;
    let columns = (0..rows * cols)
        .map(|input| {
            let tangent = f(&DualMatrix::seed(
                duplicate_matrix(at),
                input / cols,
                input % cols,
            ))
            .tangent;
            outputs = tangent.len();
            tangent.into_storage()
        })
        .collect::<Vec<_>>();
    Matrix::from_storage(outputs, rows * cols, B::hstack(&columns, outputs))
}

/// The gradient of a scalar-valued `f` at `at`, by one forward pass per input —
/// the single-output case of [`jacobian`].
pub fn gradient<B: Kernels<T>, T: Real>(
    at: &Vector<T, B>,
    f: impl Fn(&DualVector<B, T>) -> Dual<T>,
) -> Vector<T, B> {
    let mut derivatives = vec![T::zero(); at.len()];
    for (input, slot) in derivatives.iter_mut().enumerate() {
        *slot = f(&DualVector::seed(duplicate_vector(at), input)).dual;
    }
    Vector::build(&derivatives)
}

// ---- operators --------------------------------------------------------------

fn elementwise_vector<B: Kernels<T>, T: Real>(
    a: &DualVector<B, T>,
    b: &DualVector<B, T>,
    op: BinaryOp,
) -> DualVector<B, T> {
    let value = B::vector_elementwise(&a.value, &b.value, op);
    let tangent = match op {
        // d(a ± b) = ȧ ± ḃ
        BinaryOp::Add | BinaryOp::Sub => B::vector_elementwise(&a.tangent, &b.tangent, op),
        // d(a⊙b) = ȧ⊙b + a⊙ḃ
        BinaryOp::Mul => B::vector_elementwise(
            &B::vector_elementwise(&a.tangent, &b.value, BinaryOp::Mul),
            &B::vector_elementwise(&a.value, &b.tangent, BinaryOp::Mul),
            BinaryOp::Add,
        ),
        // d(a/b) = (ȧ⊙b − a⊙ḃ)/b²
        BinaryOp::Div => {
            let numerator = B::vector_elementwise(
                &B::vector_elementwise(&a.tangent, &b.value, BinaryOp::Mul),
                &B::vector_elementwise(&a.value, &b.tangent, BinaryOp::Mul),
                BinaryOp::Sub,
            );
            B::vector_elementwise(
                &numerator,
                &B::vector_elementwise(&b.value, &b.value, BinaryOp::Mul),
                BinaryOp::Div,
            )
        }
        BinaryOp::Rem => panic!("dual tensors do not differentiate remainder"),
    };
    DualVector { value, tangent }
}

fn elementwise_matrix<B: Kernels<T>, T: Real>(
    a: &DualMatrix<B, T>,
    b: &DualMatrix<B, T>,
    op: BinaryOp,
) -> DualMatrix<B, T> {
    let value = B::matrix_elementwise(&a.value, &b.value, op);
    let tangent = match op {
        BinaryOp::Add | BinaryOp::Sub => B::matrix_elementwise(&a.tangent, &b.tangent, op),
        BinaryOp::Mul => B::matrix_elementwise(
            &B::matrix_elementwise(&a.tangent, &b.value, BinaryOp::Mul),
            &B::matrix_elementwise(&a.value, &b.tangent, BinaryOp::Mul),
            BinaryOp::Add,
        ),
        BinaryOp::Div => {
            let numerator = B::matrix_elementwise(
                &B::matrix_elementwise(&a.tangent, &b.value, BinaryOp::Mul),
                &B::matrix_elementwise(&a.value, &b.tangent, BinaryOp::Mul),
                BinaryOp::Sub,
            );
            B::matrix_elementwise(
                &numerator,
                &B::matrix_elementwise(&b.value, &b.value, BinaryOp::Mul),
                BinaryOp::Div,
            )
        }
        BinaryOp::Rem => panic!("dual tensors do not differentiate remainder"),
    };
    DualMatrix { value, tangent }
}

/// One operator for a dual tensor, by value and by reference. Dual tensors own
/// heap or shared allocations, so the by-reference forms are what most code
/// wants.
macro_rules! dual_operator {
    ($Type:ident, $Trait:ident, $method:ident, $op:expr, $apply:ident) => {
        impl<B: Kernels<T>, T: Real> $Trait for $Type<B, T> {
            type Output = Self;
            fn $method(self, rhs: Self) -> Self::Output {
                $apply(&self, &rhs, $op)
            }
        }

        impl<B: Kernels<T>, T: Real> $Trait<&$Type<B, T>> for &$Type<B, T> {
            type Output = $Type<B, T>;
            fn $method(self, rhs: &$Type<B, T>) -> Self::Output {
                $apply(self, rhs, $op)
            }
        }
    };
}

dual_operator!(DualVector, Add, add, BinaryOp::Add, elementwise_vector);
dual_operator!(DualVector, Sub, sub, BinaryOp::Sub, elementwise_vector);
dual_operator!(DualVector, Mul, mul, BinaryOp::Mul, elementwise_vector);
dual_operator!(DualVector, Div, div, BinaryOp::Div, elementwise_vector);
dual_operator!(DualMatrix, Add, add, BinaryOp::Add, elementwise_matrix);
dual_operator!(DualMatrix, Sub, sub, BinaryOp::Sub, elementwise_matrix);
dual_operator!(DualMatrix, Mul, mul, BinaryOp::Mul, elementwise_matrix);
dual_operator!(DualMatrix, Div, div, BinaryOp::Div, elementwise_matrix);

impl<B: Kernels<T>, T: Real> Neg for DualVector<B, T> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        self.scale(-T::one())
    }
}

impl<B: Kernels<T>, T: Real> Neg for &DualVector<B, T> {
    type Output = DualVector<B, T>;
    fn neg(self) -> Self::Output {
        self.scale(-T::one())
    }
}

impl<B: Kernels<T>, T: Real> Neg for DualMatrix<B, T> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        self.scale(-T::one())
    }
}

impl<B: Kernels<T>, T: Real> Neg for &DualMatrix<B, T> {
    type Output = DualMatrix<B, T>;
    fn neg(self) -> Self::Output {
        self.scale(-T::one())
    }
}
