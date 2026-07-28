//! Forward-mode automatic differentiation over tensors.
//!
//! A [`Dual`] scalar carries `value + tangent·ε` in one struct. A dual *tensor*
//! could do the same — `Vector<Dual<f32>, N>` already works on the host — but
//! that layout is wrong for a GPU: the Metal shaders are 32-bit, so dual
//! elements would need a second, parallel set of kernels, and the interleaved
//! `[v, d, v, d, …]` access pattern coalesces worse than two flat buffers.
//!
//! So [`DualVector`] and [`DualMatrix`] keep the two parts in *separate* tensors
//! on the same backend, and every rule is expressed with the `f32` kernels that
//! already exist:
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
//! [`Kernels`]: the same code runs on `Host` arrays and in
//! `Metal` shared memory, so a resident chain stays resident for the whole
//! derivative computation — and the `Host` instantiation is the oracle the GPU
//! one is tested against.
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
//! assert_eq!(gradient(&x, |v| v.dot(v)).to_array(), [2.0, 4.0, 6.0]);
//!
//! // d/dt (A·A) with A = tI is 2tI, so at t = 3 the tangent is 6I.
//! let a = Matrix::<f32, 2, 2>::identity().scale(3.0);
//! let squared = DualMatrix::new(a, Matrix::identity()).squared();
//! assert_eq!(squared.value().to_rows(), [[9.0, 0.0], [0.0, 9.0]]);
//! assert_eq!(squared.tangent().to_rows(), [[6.0, 0.0], [0.0, 6.0]]);
//! ```

use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::numbers::Dual;

use super::{Analytic, BinaryOp, Compare, Host, Kernels, Matrix, Vector};

/// A length-`N` vector and its tangent, for forward-mode differentiation.
pub struct DualVector<const N: usize, B: Kernels = Host> {
    value: Vector<f32, N, B>,
    tangent: Vector<f32, N, B>,
}

/// An `R × C` matrix and its tangent, for forward-mode differentiation.
pub struct DualMatrix<const R: usize, const C: usize, B: Kernels = Host> {
    value: Matrix<f32, R, C, B>,
    tangent: Matrix<f32, R, C, B>,
}

// As elsewhere, these follow the storage: host-backed dual tensors are `Copy`,
// resident ones are not.
impl<const N: usize, B: Kernels> Copy for DualVector<N, B> where B::Vector<f32, N>: Copy {}

impl<const N: usize, B: Kernels> Clone for DualVector<N, B>
where
    B::Vector<f32, N>: Clone,
{
    fn clone(&self) -> Self {
        DualVector {
            value: self.value.clone(),
            tangent: self.tangent.clone(),
        }
    }
}

impl<const R: usize, const C: usize, B: Kernels> Copy for DualMatrix<R, C, B> where
    B::Matrix<f32, R, C>: Copy
{
}

impl<const R: usize, const C: usize, B: Kernels> Clone for DualMatrix<R, C, B>
where
    B::Matrix<f32, R, C>: Clone,
{
    fn clone(&self) -> Self {
        DualMatrix {
            value: self.value.clone(),
            tangent: self.tangent.clone(),
        }
    }
}

impl<const N: usize, B: Kernels> DualVector<N, B> {
    /// A vector paired with the direction to differentiate along.
    pub fn new(value: Vector<f32, N, B>, tangent: Vector<f32, N, B>) -> Self {
        DualVector { value, tangent }
    }

    /// A vector held constant: its tangent is zero.
    pub fn constant(value: Vector<f32, N, B>) -> Self {
        DualVector {
            value,
            tangent: Vector::filled(0.0),
        }
    }

    /// A vector seeded to differentiate with respect to element `index`, giving
    /// one column of a Jacobian. An out-of-range index seeds nothing.
    pub fn seed(value: Vector<f32, N, B>, index: usize) -> Self {
        let mut direction = vec![0.0f32; N];
        if let Some(slot) = direction.get_mut(index) {
            *slot = 1.0;
        }
        DualVector {
            value,
            tangent: Vector {
                data: B::store_vector::<N>(&direction),
            },
        }
    }

    pub fn value(&self) -> &Vector<f32, N, B> {
        &self.value
    }

    pub fn tangent(&self) -> &Vector<f32, N, B> {
        &self.tangent
    }

    /// The two parts, by value.
    pub fn into_parts(self) -> (Vector<f32, N, B>, Vector<f32, N, B>) {
        (self.value, self.tangent)
    }

    /// Dot product with another dual vector: `u·v + (u̇·v + u·v̇)ε`.
    pub fn dot(&self, other: &Self) -> Dual<f32> {
        Dual::new(
            B::dot(&self.value, &other.value),
            B::dot(&self.tangent, &other.value) + B::dot(&self.value, &other.tangent),
        )
    }

    /// The sum of the elements, differentiated — the usual way to reduce to a
    /// scalar loss.
    pub fn sum(&self) -> Dual<f32> {
        Dual::new(sum(self.value.as_slice()), sum(self.tangent.as_slice()))
    }

    /// Row vector times matrix, `(1×N)·(N×C)`, differentiated.
    pub fn vecmat<const C: usize>(&self, m: &DualMatrix<N, C, B>) -> DualVector<C, B> {
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
    pub fn scale(&self, scalar: f32) -> Self {
        DualVector {
            value: B::vector_broadcast(&self.value, scalar, BinaryOp::Mul, false),
            tangent: B::vector_broadcast(&self.tangent, scalar, BinaryOp::Mul, false),
        }
    }

    /// Add a constant: the tangent is unchanged, since `d/dx (a + c) = ȧ`.
    pub fn shift(&self, scalar: f32) -> Self {
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
    pub fn broadcast(&self, scalar: Dual<f32>, op: BinaryOp, scalar_left: bool) -> Self {
        let expanded = DualVector {
            value: Vector::filled(scalar.real),
            tangent: Vector::filled(scalar.dual),
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
        let complement = B::vector_broadcast(&share, 1.0, BinaryOp::Sub, true);
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
    pub fn clamp_min(&self, floor: f32) -> Self {
        self.select_scalar(floor, true)
    }

    /// Elementwise minimum against a constant.
    pub fn clamp_max(&self, ceiling: f32) -> Self {
        self.select_scalar(ceiling, false)
    }

    /// Confine every element to `[floor, ceiling]`.
    pub fn clamp(&self, floor: f32, ceiling: f32) -> Self {
        self.clamp_min(floor).clamp_max(ceiling)
    }

    /// The rectifier `max(a, 0)`, whose slope is `½` exactly at the kink.
    pub fn relu(&self) -> Self {
        self.clamp_min(0.0)
    }

    /// Elementwise absolute value, as `max(a, −a)`.
    ///
    /// The tangent works out to `sign(a) ⊙ ȧ`, with `sign(0) = 0` because the tie
    /// splits the subgradient between `+1` and `−1`.
    pub fn abs(&self) -> Self {
        self.maximum(&self.scale(-1.0))
    }

    fn select_scalar(&self, scalar: f32, largest: bool) -> Self {
        let op = if largest { Compare::Max } else { Compare::Min };
        let value = B::vector_compare_scalar(&self.value, scalar, op, false);
        let share = B::vector_compare_scalar(&self.value, scalar, Compare::MaxShare, false);
        // A constant contributes no tangent, so only this operand's share counts.
        let weight = if largest {
            share
        } else {
            B::vector_broadcast(&share, 1.0, BinaryOp::Sub, true)
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
            value: B::vector_broadcast(&self.value, 1.0, BinaryOp::Div, true),
            tangent: B::vector_broadcast(
                &B::vector_elementwise(&self.tangent, &squared, BinaryOp::Div),
                -1.0,
                BinaryOp::Mul,
                false,
            ),
        }
    }
}

impl<const R: usize, const C: usize, B: Kernels> DualMatrix<R, C, B> {
    /// A matrix paired with the direction to differentiate along.
    pub fn new(value: Matrix<f32, R, C, B>, tangent: Matrix<f32, R, C, B>) -> Self {
        DualMatrix { value, tangent }
    }

    /// A matrix held constant: its tangent is zero.
    pub fn constant(value: Matrix<f32, R, C, B>) -> Self {
        DualMatrix {
            value,
            tangent: Matrix::filled(0.0),
        }
    }

    /// A matrix seeded to differentiate with respect to element `(row, col)`,
    /// giving one column of a Jacobian. Out-of-range indices seed nothing.
    pub fn seed(value: Matrix<f32, R, C, B>, row: usize, col: usize) -> Self {
        let mut direction = vec![0.0f32; R * C];
        if row < R && col < C {
            direction[row * C + col] = 1.0;
        }
        DualMatrix {
            value,
            tangent: Matrix {
                data: B::store_matrix::<R, C>(&direction),
            },
        }
    }

    pub fn value(&self) -> &Matrix<f32, R, C, B> {
        &self.value
    }

    pub fn tangent(&self) -> &Matrix<f32, R, C, B> {
        &self.tangent
    }

    /// The two parts, by value.
    pub fn into_parts(self) -> (Matrix<f32, R, C, B>, Matrix<f32, R, C, B>) {
        (self.value, self.tangent)
    }

    /// Matrix product, differentiated: `AB + (ȦB + AḂ)ε`.
    ///
    /// Three products and a sum in principle; on a backend that can accumulate
    /// inside the product kernel the sum rides along in the second product, so
    /// the tangent costs two dispatches rather than three.
    pub fn matmul<const C2: usize>(&self, other: &DualMatrix<C, C2, B>) -> DualMatrix<R, C2, B> {
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
    pub fn matmul_add<const C2: usize>(
        &self,
        other: &DualMatrix<C, C2, B>,
        addend: &DualMatrix<R, C2, B>,
    ) -> DualMatrix<R, C2, B> {
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
    pub fn matvec(&self, v: &DualVector<C, B>) -> DualVector<R, B> {
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
    pub fn matvec_add(&self, v: &DualVector<C, B>, addend: &DualVector<R, B>) -> DualVector<R, B> {
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
    pub fn transpose(&self) -> DualMatrix<C, R, B> {
        DualMatrix {
            value: B::transpose(&self.value),
            tangent: B::transpose(&self.tangent),
        }
    }

    /// Multiply by a constant: both parts scale.
    pub fn scale(&self, scalar: f32) -> Self {
        DualMatrix {
            value: B::matrix_broadcast(&self.value, scalar, BinaryOp::Mul, false),
            tangent: B::matrix_broadcast(&self.tangent, scalar, BinaryOp::Mul, false),
        }
    }

    /// Add a constant: the tangent is unchanged.
    pub fn shift(&self, scalar: f32) -> Self {
        DualMatrix {
            value: B::matrix_broadcast(&self.value, scalar, BinaryOp::Add, false),
            tangent: duplicate_matrix(&self.tangent),
        }
    }

    /// Combine with a *dual* scalar elementwise; see
    /// [`DualVector::broadcast`].
    pub fn broadcast(&self, scalar: Dual<f32>, op: BinaryOp, scalar_left: bool) -> Self {
        let expanded = DualMatrix {
            value: Matrix::filled(scalar.real),
            tangent: Matrix::filled(scalar.dual),
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
        let complement = B::matrix_broadcast(&share, 1.0, BinaryOp::Sub, true);
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
    pub fn clamp_min(&self, floor: f32) -> Self {
        self.select_scalar(floor, true)
    }

    /// Elementwise minimum against a constant.
    pub fn clamp_max(&self, ceiling: f32) -> Self {
        self.select_scalar(ceiling, false)
    }

    /// Confine every element to `[floor, ceiling]`.
    pub fn clamp(&self, floor: f32, ceiling: f32) -> Self {
        self.clamp_min(floor).clamp_max(ceiling)
    }

    /// The rectifier `max(A, 0)`.
    pub fn relu(&self) -> Self {
        self.clamp_min(0.0)
    }

    /// Elementwise absolute value; see [`DualVector::abs`].
    pub fn abs(&self) -> Self {
        self.maximum(&self.scale(-1.0))
    }

    fn select_scalar(&self, scalar: f32, largest: bool) -> Self {
        let op = if largest { Compare::Max } else { Compare::Min };
        let value = B::matrix_compare_scalar(&self.value, scalar, op, false);
        let share = B::matrix_compare_scalar(&self.value, scalar, Compare::MaxShare, false);
        let weight = if largest {
            share
        } else {
            B::matrix_broadcast(&share, 1.0, BinaryOp::Sub, true)
        };
        DualMatrix {
            value,
            tangent: B::matrix_elementwise(&weight, &self.tangent, BinaryOp::Mul),
        }
    }

    /// Sum along each row, giving one entry per row.
    ///
    /// This is `A·1`, so it needs no reduction kernel of its own — and on the
    /// `Metal` backend it stays resident like any other product.
    pub fn row_sums(&self) -> DualVector<R, B> {
        let ones = Vector::<f32, C, B>::filled(1.0);
        DualVector {
            value: B::matvec(&self.value, &ones),
            tangent: B::matvec(&self.tangent, &Vector::<f32, C, B>::filled(1.0)),
        }
    }

    /// Sum along each column, giving one entry per column: `1ᵀ·A`.
    pub fn column_sums(&self) -> DualVector<C, B> {
        let ones = Vector::<f32, R, B>::filled(1.0);
        DualVector {
            value: B::vecmat(&ones, &self.value),
            tangent: B::vecmat(&Vector::<f32, R, B>::filled(1.0), &self.tangent),
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
            value: B::matrix_broadcast(&self.value, 1.0, BinaryOp::Div, true),
            tangent: B::matrix_broadcast(
                &B::matrix_elementwise(&self.tangent, &squared, BinaryOp::Div),
                -1.0,
                BinaryOp::Mul,
                false,
            ),
        }
    }

    /// The sum of the elements, differentiated.
    pub fn sum(&self) -> Dual<f32> {
        Dual::new(sum(self.value.as_slice()), sum(self.tangent.as_slice()))
    }

    /// The Frobenius inner product `Σᵢⱼ aᵢⱼbᵢⱼ = tr(AᵀB)`, differentiated.
    pub fn frobenius_dot(&self, other: &Self) -> Dual<f32> {
        elementwise_matrix(self, other, BinaryOp::Mul).sum()
    }
}

impl<const N: usize, B: Kernels> DualMatrix<N, N, B> {
    /// `A·A`, differentiated — the smallest case where the product rule shows up
    /// on both sides.
    pub fn squared(&self) -> Self {
        self.matmul(self)
    }
}

/// Sum of a slice, read in place — on a resident tensor this reads shared memory
/// directly, exactly like [`Vector::dot`](super::Vector::dot) on that backend.
fn sum(values: &[f32]) -> f32 {
    values.iter().sum()
}

/// A second copy of a tensor's storage on the same backend. Moving to the backend
/// it is already on is just that copy — within shared memory, when resident.
fn duplicate_vector<const N: usize, B: Kernels>(v: &Vector<f32, N, B>) -> Vector<f32, N, B> {
    v.to_backend::<B>()
}

fn duplicate_matrix<const R: usize, const C: usize, B: Kernels>(
    m: &Matrix<f32, R, C, B>,
) -> Matrix<f32, R, C, B> {
    m.to_backend::<B>()
}

/// The full set of analytic functions, as methods, for both dual tensor types.
macro_rules! analytic_methods {
    ($($method:ident => $variant:ident),+ $(,)?) => {
        impl<const N: usize, B: Kernels> DualVector<N, B> {
            $(
                #[doc = concat!("Elementwise `", stringify!($method), "`, differentiated.")]
                pub fn $method(&self) -> Self {
                    self.analytic(Analytic::$variant)
                }
            )+
        }

        impl<const R: usize, const C: usize, B: Kernels> DualMatrix<R, C, B> {
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
);

// ---- interoperating with `Dual` elements ------------------------------------

impl<const N: usize, B: Kernels> DualVector<N, B> {
    /// Split a host vector of dual numbers into value and tangent parts on
    /// backend `B`.
    pub fn from_dual_vector(duals: &Vector<Dual<f32>, N, Host>) -> Self {
        let elements = *duals.data();
        DualVector {
            value: Vector {
                data: B::store_vector::<N>(&elements.map(|dual| dual.real)),
            },
            tangent: Vector {
                data: B::store_vector::<N>(&elements.map(|dual| dual.dual)),
            },
        }
    }

    /// Recombine the two parts into a host vector of dual numbers — the form the
    /// scalar [`Dual`] operations in [`crate::numbers`] work on.
    pub fn to_dual_vector(&self) -> Vector<Dual<f32>, N, Host> {
        let (value, tangent) = (self.value.as_slice(), self.tangent.as_slice());
        Vector::new(std::array::from_fn(|i| Dual::new(value[i], tangent[i])))
    }
}

impl<const R: usize, const C: usize, B: Kernels> DualMatrix<R, C, B> {
    /// Split a host matrix of dual numbers into value and tangent parts on
    /// backend `B`.
    pub fn from_dual_matrix(duals: &Matrix<Dual<f32>, R, C, Host>) -> Self {
        let mut value = Vec::with_capacity(R * C);
        let mut tangent = Vec::with_capacity(R * C);
        for row in duals.data() {
            for dual in row {
                value.push(dual.real);
                tangent.push(dual.dual);
            }
        }
        DualMatrix {
            value: Matrix {
                data: B::store_matrix::<R, C>(&value),
            },
            tangent: Matrix {
                data: B::store_matrix::<R, C>(&tangent),
            },
        }
    }

    /// Recombine the two parts into a host matrix of dual numbers.
    pub fn to_dual_matrix(&self) -> Matrix<Dual<f32>, R, C, Host> {
        let (value, tangent) = (self.value.as_slice(), self.tangent.as_slice());
        Matrix::from_rows(std::array::from_fn(|row| {
            std::array::from_fn(|col| Dual::new(value[row * C + col], tangent[row * C + col]))
        }))
    }
}

// ---- reshaping --------------------------------------------------------------

/// Row-major flattening, in both directions.
///
/// This is the adapter that makes matrix inputs and outputs work with the vector
/// machinery: a matrix is `R * C` scalars, and once they are named as a vector,
/// [`jacobian`] and [`gradient`] apply unchanged. On the `Metal` backend a vector
/// and a matrix are the same allocation, so the by-value forms move rather than
/// copy.
impl<const R: usize, const C: usize, B: Kernels> DualMatrix<R, C, B> {
    /// Flatten both parts into a dual vector, consuming this matrix.
    pub fn into_flattened(self) -> DualVector<{ R * C }, B> {
        DualVector {
            value: Vector {
                data: B::matrix_into_flattened::<R, C>(self.value.data),
            },
            tangent: Vector {
                data: B::matrix_into_flattened::<R, C>(self.tangent.data),
            },
        }
    }

    /// Flatten both parts into a dual vector, leaving this matrix intact.
    pub fn flattened(&self) -> DualVector<{ R * C }, B> {
        self.clone_parts().into_flattened()
    }

    /// Rebuild an `R × C` dual matrix from its row-major flattening.
    pub fn from_flattened(vector: DualVector<{ R * C }, B>) -> Self {
        DualMatrix {
            value: Matrix {
                data: B::vector_into_matrix::<R, C>(vector.value.data),
            },
            tangent: Matrix {
                data: B::vector_into_matrix::<R, C>(vector.tangent.data),
            },
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

impl<const N: usize, B: Kernels> DualVector<N, B> {
    /// View as a `1 × N` dual matrix, consuming this vector.
    pub fn into_row(self) -> DualMatrix<1, N, B> {
        DualMatrix {
            value: Matrix {
                data: B::vector_into_row::<N>(self.value.data),
            },
            tangent: Matrix {
                data: B::vector_into_row::<N>(self.tangent.data),
            },
        }
    }

    /// View as an `N × 1` dual matrix, consuming this vector.
    pub fn into_column(self) -> DualMatrix<N, 1, B> {
        DualMatrix {
            value: Matrix {
                data: B::vector_into_column::<N>(self.value.data),
            },
            tangent: Matrix {
                data: B::vector_into_column::<N>(self.tangent.data),
            },
        }
    }
}

// ---- Jacobians and gradients ------------------------------------------------

/// The outer product of the gradient operator with the matrix.
///
/// Columns [j..j+N] are the derivatives w.r.t. xj.
pub fn matrix_gradient<const IN: usize, const OUT1: usize, const OUT2: usize, B: Kernels>(
    at: &Vector<f32, IN, B>,
    f: impl Fn(&DualVector<IN, B>) -> DualMatrix<OUT1, OUT2, B>,
) -> Matrix<f32, OUT1, { IN * OUT2 }, B> {
    let tangents = std::array::from_fn(|input| {
        f(&DualVector::seed(duplicate_vector(at), input))
            .tangent
            .data
    });

    Matrix {
        data: B::hmerge(tangents),
    }
}

/// The Jacobian of `f` at `at`, by one forward pass per input element.
///
/// Column `j` is the tangent of `f` seeded along input `j`, so this costs `IN`
/// evaluations of `f`. For many inputs and few outputs — a scalar loss over a
/// parameter vector, say — that is the wrong way around, and reverse mode is the
/// answer.
pub fn jacobian<const IN: usize, const OUT: usize, B: Kernels>(
    at: &Vector<f32, IN, B>,
    f: impl Fn(&DualVector<IN, B>) -> DualVector<OUT, B>,
) -> Matrix<f32, OUT, IN, B> {
    let tangents = std::array::from_fn(|input| {
        f(&DualVector::seed(duplicate_vector(at), input))
            .tangent
            .data
    });

    Matrix {
        data: B::hstack::<OUT, IN>(tangents),
    }
}

/// The gradient of a scalar-valued `f` with respect to a *matrix* input.
///
/// The result has the shape of the input, so `∂f/∂Aᵢⱼ` sits at `(i, j)` — this is
/// the derivative meant by "differentiate the loss with respect to the weights".
/// It costs `R * C` forward passes, one per element, since each pass extracts a
/// single partial. A `256 × 256` input therefore means 65,536 evaluations of `f`;
/// that asymmetry is the argument for reverse mode, not a defect of this
/// implementation.
///
/// ```
/// use tensorcrate::tensors::{DualVector, Matrix, Vector, gradient_wrt_matrix};
///
/// // f(A) = ‖A·x‖², whose gradient is 2(Ax)xᵀ.
/// let a = Matrix::<f32, 2, 3>::from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
/// let x = Vector::new([1.0f32, 0.5, -1.0]);
///
/// let gradient = gradient_wrt_matrix(&a, |m| {
///     let mapped = m.matvec(&DualVector::constant(x));
///     mapped.dot(&mapped)
/// });
///
/// let projected = a.matvec(&x);
/// for row in 0..2 {
///     for col in 0..3 {
///         let expected = 2.0 * projected.data()[row] * x.data()[col];
///         assert!((gradient.data()[row][col] - expected).abs() < 1e-4);
///     }
/// }
/// ```
pub fn gradient_wrt_matrix<const R: usize, const C: usize, B: Kernels>(
    at: &Matrix<f32, R, C, B>,
    f: impl Fn(&DualMatrix<R, C, B>) -> Dual<f32>,
) -> Matrix<f32, R, C, B> {
    let rows = std::array::from_fn(|row| {
        let partials: [f32; C] =
            std::array::from_fn(|col| f(&DualMatrix::seed(duplicate_matrix(at), row, col)).dual);
        B::store_vector::<C>(&partials)
    });
    Matrix {
        data: B::vstack::<R, C>(rows),
    }
}

/// The Jacobian of a vector-valued `f` with respect to a *matrix* input.
///
/// Column `i * C + j` holds the derivatives with respect to `Aᵢⱼ`, matching the
/// row-major flattening of the input, so the result is `OUT × (R·C)`. Like
/// [`gradient_wrt_matrix`], it runs `R * C` passes.
///
/// A matrix-valued `f` needs no separate driver — flatten its output with
/// [`DualMatrix::into_flattened`] and the result is the standard Jacobian,
/// `(OR·OC) × (R·C)`:
///
/// ```
/// # use tensorcrate::tensors::{DualMatrix, Matrix, jacobian_wrt_matrix};
/// # let a = Matrix::<f32, 2, 2>::identity();
/// # let b = Matrix::<f32, 2, 2>::from_rows([[1.0, 2.0], [3.0, 4.0]]);
/// let jacobian: Matrix<f32, 4, 4> = jacobian_wrt_matrix::<2, 2, 4, _>(&a, |m| {
///     m.matmul(&DualMatrix::constant(b)).into_flattened()
/// });
/// ```
///
/// Writing that type down needs the turbofish: `{ R * C }` in the return type
/// only normalizes to a literal once the parameters are given explicitly, which
/// is a rough edge of `generic_const_exprs`. Code that lets inference name the
/// type — `let j = jacobian_wrt_matrix(&a, f);` — needs no annotation and no
/// turbofish.
pub fn jacobian_wrt_matrix<const R: usize, const C: usize, const OUT: usize, B: Kernels>(
    at: &Matrix<f32, R, C, B>,
    f: impl Fn(&DualMatrix<R, C, B>) -> DualVector<OUT, B>,
) -> Matrix<f32, OUT, { R * C }, B> {
    let columns = std::array::from_fn(|input| {
        f(&DualMatrix::seed(
            duplicate_matrix(at),
            input / C,
            input % C,
        ))
        .tangent
        .data
    });
    Matrix {
        data: B::hstack::<OUT, { R * C }>(columns),
    }
}

/// The gradient of a scalar-valued `f` at `at`, by one forward pass per input —
/// the single-output case of [`jacobian`].
pub fn gradient<const IN: usize, B: Kernels>(
    at: &Vector<f32, IN, B>,
    f: impl Fn(&DualVector<IN, B>) -> Dual<f32>,
) -> Vector<f32, IN, B> {
    let mut derivatives = vec![0.0f32; IN];
    for (input, slot) in derivatives.iter_mut().enumerate() {
        *slot = f(&DualVector::seed(duplicate_vector(at), input)).dual;
    }
    Vector {
        data: B::store_vector::<IN>(&derivatives),
    }
}

// ---- operators --------------------------------------------------------------

fn elementwise_vector<const N: usize, B: Kernels>(
    a: &DualVector<N, B>,
    b: &DualVector<N, B>,
    op: BinaryOp,
) -> DualVector<N, B> {
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

fn elementwise_matrix<const R: usize, const C: usize, B: Kernels>(
    a: &DualMatrix<R, C, B>,
    b: &DualMatrix<R, C, B>,
    op: BinaryOp,
) -> DualMatrix<R, C, B> {
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

/// One operator for a dual tensor, by value and by reference. The by-reference
/// forms matter on the `Metal` backend, where the tensors are not `Copy`.
macro_rules! dual_operator {
    ($Type:ident < $($dim:ident),+ >, $Trait:ident, $method:ident, $op:expr, $apply:ident) => {
        impl<$(const $dim: usize),+, B: Kernels> $Trait for $Type<$($dim),+, B> {
            type Output = Self;
            fn $method(self, rhs: Self) -> Self::Output {
                $apply(&self, &rhs, $op)
            }
        }

        impl<$(const $dim: usize),+, B: Kernels> $Trait<&$Type<$($dim),+, B>>
            for &$Type<$($dim),+, B>
        {
            type Output = $Type<$($dim),+, B>;
            fn $method(self, rhs: &$Type<$($dim),+, B>) -> Self::Output {
                $apply(self, rhs, $op)
            }
        }
    };
}

dual_operator!(DualVector<N>, Add, add, BinaryOp::Add, elementwise_vector);
dual_operator!(DualVector<N>, Sub, sub, BinaryOp::Sub, elementwise_vector);
dual_operator!(DualVector<N>, Mul, mul, BinaryOp::Mul, elementwise_vector);
dual_operator!(DualVector<N>, Div, div, BinaryOp::Div, elementwise_vector);
dual_operator!(DualMatrix<R, C>, Add, add, BinaryOp::Add, elementwise_matrix);
dual_operator!(DualMatrix<R, C>, Sub, sub, BinaryOp::Sub, elementwise_matrix);
dual_operator!(DualMatrix<R, C>, Mul, mul, BinaryOp::Mul, elementwise_matrix);
dual_operator!(DualMatrix<R, C>, Div, div, BinaryOp::Div, elementwise_matrix);

impl<const N: usize, B: Kernels> Neg for DualVector<N, B> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        self.scale(-1.0)
    }
}

impl<const N: usize, B: Kernels> Neg for &DualVector<N, B> {
    type Output = DualVector<N, B>;
    fn neg(self) -> Self::Output {
        self.scale(-1.0)
    }
}

impl<const R: usize, const C: usize, B: Kernels> Neg for DualMatrix<R, C, B> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        self.scale(-1.0)
    }
}

impl<const R: usize, const C: usize, B: Kernels> Neg for &DualMatrix<R, C, B> {
    type Output = DualMatrix<R, C, B>;
    fn neg(self) -> Self::Output {
        self.scale(-1.0)
    }
}
