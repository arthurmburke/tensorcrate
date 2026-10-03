//! Tensor operations for the [`Metal`] backend.
//!
//! Everything here keeps its operands *and* its result in GPU-shared memory:
//! [`matmul`](Matrix::matmul), [`matvec`](Matrix::matvec),
//! [`vecmat`](Vector::vecmat), the elementwise operators and scalar
//! broadcasting all encode straight over the existing allocations, so a chain of
//! operations costs one upload at the start and one download at the end rather
//! than a pair per call.
//!
//! When a dispatch cannot run — no Metal device, allocation or compilation
//! failure, or an unsupported operation such as `%` — the operation panics.
//! The Metal backend never silently executes a Host kernel. Move tensors to
//! [`Host`] explicitly when CPU execution is wanted.
//!
//! Shape checking is the same here as on the host: the extents live in the
//! tensor rather than the allocation, so a mismatched product panics before any
//! dispatch is encoded.
//!
//! Operations with no GPU kernel at all — such as [`Matrix::determinant`] and
//! [`Matrix::inverse`] — are not implemented for this backend. Reach them
//! through [`to_backend::<Host>()`](Matrix::to_backend).
//!
//! Everything here is defined for every [`MetalElement`] — `f32`, `f16` and
//! `bf16` — and runs that type's own kernels: a `Vector<f16, Metal>` adds in
//! `half`. Sums, products and moments accumulate in `f32` and round to the
//! element type once. `f16` and `bf16` also convert to and from `f32`, and
//! have a matrix product with an `f32` result.
//!
//! [`Matrix::determinant`]: super::Matrix::determinant
//! [`Matrix::inverse`]: super::Matrix::inverse

use std::cmp::Ordering;
use std::fmt::{self, Display};
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

use half::{bf16, f16};
use num_traits::PrimInt;

use super::shape::{assert_inner, assert_ordered_bounds, assert_same_len, assert_same_shape};
use super::{
    Analytic, Backend, BinaryOp, Compare, Host, Matrix, Metal, Reduce, SortOrder, Vector,
    require_metal,
};
use crate::metal::MetalElement;

macro_rules! low_precision_tensors {
    ($ty:ty) => {
        impl Vector<$ty, Metal> {
            /// Convert FP32 values into compact resident storage.
            pub fn from_f32<B: Backend>(values: &Vector<f32, B>) -> Self {
                let converted = values
                    .as_slice()
                    .iter()
                    .copied()
                    .map(<$ty>::from_f32)
                    .collect::<Vec<_>>();
                Vector::build(&converted)
            }

            /// Convert the compact values to FP32 on backend `B`.
            pub fn to_f32<B: Backend>(&self) -> Vector<f32, B> {
                let converted = self
                    .as_slice()
                    .iter()
                    .copied()
                    .map(f32::from)
                    .collect::<Vec<_>>();
                Vector::build(&converted)
            }
        }

        impl Matrix<$ty, Metal> {
            /// Convert an FP32 matrix into compact resident row-major storage.
            pub fn from_f32<B: Backend>(values: &Matrix<f32, B>) -> Self {
                let converted = values
                    .as_slice()
                    .iter()
                    .copied()
                    .map(<$ty>::from_f32)
                    .collect::<Vec<_>>();
                Matrix::build(values.rows(), values.cols(), &converted)
            }

            /// Convert the compact matrix to FP32 on backend `B`.
            pub fn to_f32<B: Backend>(&self) -> Matrix<f32, B> {
                let converted = self
                    .as_slice()
                    .iter()
                    .copied()
                    .map(f32::from)
                    .collect::<Vec<_>>();
                Matrix::build(self.rows(), self.cols(), &converted)
            }

            /// TensorOps product with widened FP32 accumulation and output.
            #[track_caller]
            pub fn matmul_f32(&self, other: &Self) -> Matrix<f32, Metal> {
                assert_inner(self.shape(), other.shape(), "matmul_f32");
                let (rows, inner, cols) = (self.rows(), self.cols(), other.cols());
                let data = require_metal(
                    "widened matrix multiplication",
                    self.storage()
                        .matmul_f32(other.storage(), rows, inner, cols),
                );
                Matrix::from_storage(rows, cols, data)
            }
        }
    };
}

low_precision_tensors!(f16);
low_precision_tensors!(bf16);

impl<T: Copy + 'static> Vector<T, Metal> {
    /// Whether the elements really are in GPU-shared memory — for any element
    /// type, so an index vector answers too.
    ///
    /// `false` means the process has no Metal device. Operations will panic
    /// rather than run Host kernels implicitly.
    pub fn is_device_resident(&self) -> bool {
        self.storage().is_device_resident()
    }
}

impl<T: Copy + 'static> Matrix<T, Metal> {
    /// Whether the elements really are in GPU-shared memory; see
    /// [`Vector::is_device_resident`].
    pub fn is_device_resident(&self) -> bool {
        self.storage().is_device_resident()
    }
}

impl<T: MetalElement> Vector<T, Metal> {
    /// Dot product with a vector of the same length.
    ///
    /// Multiplication and reduction both run on the GPU. The final scalar is a
    /// synchronization point and is rounded to `T` once.
    ///
    /// # Panics
    ///
    /// If the two lengths differ.
    #[track_caller]
    pub fn dot(&self, other: &Self) -> T {
        assert_same_len(self.len(), other.len(), "dot");
        if self.is_empty() {
            return T::zero();
        }
        let products = require_metal(
            "dot product multiplication",
            self.storage().elementwise(other.storage(), BinaryOp::Mul),
        );
        let total = require_metal("dot product reduction", products.reduce(Reduce::Sum));
        T::from_f64(f64::from(total))
    }

    /// Row vector times matrix: `(1×N)·(N×C) = (1×C)`, on the GPU.
    ///
    /// # Panics
    ///
    /// If this vector's length is not the matrix's row count.
    #[track_caller]
    pub fn vecmat(&self, m: &Matrix<T, Metal>) -> Vector<T, Metal> {
        assert_inner((1, self.len()), m.shape(), "vecmat");
        let (rows, cols) = m.shape();
        let data = require_metal(
            "vector-matrix multiplication",
            self.storage().matmul(m.storage(), 1, rows, cols),
        );
        Vector::from_storage(cols, data)
    }

    /// Apply an analytic function elementwise, on the GPU.
    ///
    /// The host tensor of the same name maps the scalar trait from
    /// [`crate::numbers`] over its elements; resident tensors run the unary
    /// kernel compiled for their element type instead.
    /// Both spellings — this and the named methods in
    /// [`analytic`](crate::tensors::analytic) — are the same operation.
    pub fn analytic(&self, f: Analytic) -> Self {
        let data = require_metal("vector analytic operation", self.storage().unary(f));
        Vector::from_storage(self.len(), data)
    }

    /// Elementwise comparison with another resident vector, on the GPU.
    #[track_caller]
    pub fn compare(&self, other: &Self, op: Compare) -> Self {
        assert_same_len(self.len(), other.len(), "compare");
        let data = require_metal(
            "vector comparison",
            self.storage().compare(other.storage(), op),
        );
        Vector::from_storage(self.len(), data)
    }

    /// Elementwise comparison against a scalar, on the GPU.
    pub fn compare_scalar(&self, scalar: T, op: Compare, scalar_left: bool) -> Self {
        let data = require_metal(
            "vector-scalar comparison",
            self.storage().compare_scalar(scalar, op, scalar_left),
        );
        Vector::from_storage(self.len(), data)
    }

    /// Elementwise minimum with another resident vector, on the GPU.
    ///
    /// # Panics
    ///
    /// If the two lengths differ.
    #[track_caller]
    pub fn min(&self, other: &Self) -> Self {
        self.compare(other, Compare::Min)
    }

    /// Elementwise maximum with another resident vector, on the GPU.
    ///
    /// # Panics
    ///
    /// If the two lengths differ.
    #[track_caller]
    pub fn max(&self, other: &Self) -> Self {
        self.compare(other, Compare::Max)
    }

    /// The lesser of each element and `scalar`, on the GPU.
    pub fn min_scalar(&self, scalar: T) -> Self {
        self.compare_scalar(scalar, Compare::Min, false)
    }

    /// The greater of each element and `scalar`, on the GPU — `max_scalar(0.0)`
    /// is a relu.
    pub fn max_scalar(&self, scalar: T) -> Self {
        self.compare_scalar(scalar, Compare::Max, false)
    }

    /// Confine every element to `[low, high]`, in one dispatch.
    ///
    /// # Panics
    ///
    /// If `low > high`.
    #[track_caller]
    pub fn clamp(&self, low: T, high: T) -> Self {
        assert_ordered_bounds(&low, &high);
        let data = require_metal("vector clamp", self.storage().clamp(low, high));
        Vector::from_storage(self.len(), data)
    }

    /// Fold the whole vector to one value with a GPU tree reduction. An empty
    /// vector gives [`op.identity()`](Reduce::identity).
    ///
    /// The answer is a number rather than a tensor, so this necessarily comes
    /// back to the CPU: it is a synchronization point, unlike the operations
    /// that leave their result resident.
    pub fn reduce(&self, op: Reduce) -> T {
        let value = require_metal("vector reduction", self.storage().reduce(op));
        // The fold ran in `f32`; this is its one rounding to `T`.
        T::from_f64(f64::from(value))
    }

    /// The sum of every element.
    pub fn sum(&self) -> T {
        self.reduce(Reduce::Sum)
    }

    /// The smallest element, or `None` when there are none.
    pub fn minimum(&self) -> Option<T> {
        (!self.is_empty()).then(|| self.reduce(Reduce::Min))
    }

    /// The largest element, or `None` when there are none.
    pub fn maximum(&self) -> Option<T> {
        (!self.is_empty()).then(|| self.reduce(Reduce::Max))
    }

    /// Inclusive prefix sum, staying resident.
    ///
    /// `log2(len)` dispatches of a Hillis–Steele scan. Its additions associate
    /// differently from the host's running total, so the two agree to a
    /// rounding error rather than bit for bit.
    pub fn prefix_sum(&self) -> Self {
        let data = require_metal("vector prefix sum", self.storage().prefix_sum());
        Vector::from_storage(self.len(), data)
    }

    /// Sort in [`SortOrder`]'s total order, staying resident.
    ///
    /// A bitonic sort over integer sort keys, so the result matches a host
    /// total-order sort exactly, NaNs included.
    pub fn sorted(&self, order: SortOrder) -> Self {
        let data = require_metal("vector sort", self.storage().sort(order));
        Vector::from_storage(self.len(), data)
    }

    /// Sort under an arbitrary comparator.
    ///
    /// A closure cannot cross to the GPU, so this operation is unsupported on
    /// Metal. Transfer explicitly to [`Host`] to use an arbitrary comparator.
    pub fn sorted_by(&self, _compare: impl FnMut(&T, &T) -> Ordering) -> Self {
        require_metal("sort with an arbitrary comparator", None)
    }

    /// Multiply every element by `scalar`, on the GPU.
    pub fn scale(&self, scalar: T) -> Self {
        self.broadcast_right(scalar, BinaryOp::Mul)
    }

    /// Implementation hook used by `math!` for tensor/scalar broadcasting.
    #[doc(hidden)]
    pub fn broadcast_right(&self, scalar: T, op: BinaryOp) -> Self {
        let data = require_metal(
            "vector-scalar broadcast",
            self.storage().broadcast(scalar, op, false),
        );
        Vector::from_storage(self.len(), data)
    }

    /// Implementation hook used by `math!` for scalar/tensor broadcasting.
    #[doc(hidden)]
    pub fn broadcast_left(&self, scalar: T, op: BinaryOp) -> Self {
        let data = require_metal(
            "scalar-vector broadcast",
            self.storage().broadcast(scalar, op, true),
        );
        Vector::from_storage(self.len(), data)
    }
}

impl<T: MetalElement> Matrix<T, Metal> {
    /// Matrix product `(R×K)·(K×C) = (R×C)`, on the GPU, with both operands
    /// and the result staying in shared memory.
    ///
    /// # Panics
    ///
    /// If this matrix's column count is not the other's row count.
    #[track_caller]
    pub fn matmul(&self, other: &Matrix<T, Metal>) -> Matrix<T, Metal> {
        assert_inner(self.shape(), other.shape(), "matmul");
        let (rows, inner, cols) = (self.rows(), self.cols(), other.cols());
        let data = require_metal(
            "matrix multiplication",
            self.storage().matmul(other.storage(), rows, inner, cols),
        );
        Matrix::from_storage(rows, cols, data)
    }

    /// Matrix exponentiation using efficient integer powers.
    ///
    /// # Panics
    ///
    /// If this matrix is non-square.
    #[track_caller]
    pub fn powi<I: PrimInt>(&self, power: I) -> Self {
        let (r, c) = self.shape();
        assert!(r == c, "matrix powers must be square");

        let mut power = power.to_i32().expect("expected a valid integer");

        // Requires a single memcpy of memory on the host to metal.
        let mut result = Matrix::<T, Host>::identity(r).to_backend::<Metal>();
        let mut base = self.clone();

        while power > 0 {
            if power & 1 == 1 {
                result = result.matmul(&base);
            }

            base = base.matmul(&base);
            power >>= 1;
        }

        result
    }

    /// Fused matrix multiply-add: `self·other + addend`.
    ///
    /// The owned addend is the Metal kernel's accumulator, so the operation
    /// needs neither an intermediate product buffer nor a second dispatch.
    ///
    /// # Panics
    ///
    /// If the inner dimensions disagree, or `addend` is the wrong shape.
    #[track_caller]
    pub fn matmul_add(
        &self,
        other: &Matrix<T, Metal>,
        mut addend: Matrix<T, Metal>,
    ) -> Matrix<T, Metal> {
        assert_inner(self.shape(), other.shape(), "matmul_add");
        assert_same_shape(
            addend.shape(),
            (self.rows(), other.cols()),
            "matmul_add addend",
        );
        let (rows, inner, cols) = (self.rows(), self.cols(), other.cols());
        require_metal(
            "matrix multiply-add",
            self.storage().matmul_accumulate(
                other.storage(),
                addend.storage_mut(),
                rows,
                inner,
                cols,
            ),
        );
        addend
    }

    /// Matrix times column vector: `(R×C)·(C×1) = (R×1)`, on the GPU.
    ///
    /// # Panics
    ///
    /// If the vector's length is not this matrix's column count.
    #[track_caller]
    pub fn matvec(&self, v: &Vector<T, Metal>) -> Vector<T, Metal> {
        assert_inner(self.shape(), (v.len(), 1), "matvec");
        let (rows, cols) = self.shape();
        let data = require_metal(
            "matrix-vector multiplication",
            self.storage().matmul(v.storage(), rows, cols, 1),
        );
        Vector::from_storage(rows, data)
    }

    /// Fused matrix-vector multiply-add: `self·v + addend`.
    ///
    /// This is the same accumulating Metal matmul with a single output column.
    ///
    /// # Panics
    ///
    /// If the vector's length is not this matrix's column count, or the addend's
    /// length is not its row count.
    #[track_caller]
    pub fn matvec_add(
        &self,
        v: &Vector<T, Metal>,
        mut addend: Vector<T, Metal>,
    ) -> Vector<T, Metal> {
        assert_inner(self.shape(), (v.len(), 1), "matvec_add");
        assert_same_len(addend.len(), self.rows(), "matvec_add addend");
        let (rows, cols) = self.shape();
        require_metal(
            "matrix-vector multiply-add",
            self.storage()
                .matmul_accumulate(v.storage(), addend.storage_mut(), rows, cols, 1),
        );
        addend
    }

    /// Transpose: an `R×C` matrix becomes `C×R`.
    pub fn transpose(&self) -> Matrix<T, Metal> {
        let (rows, cols) = self.shape();
        let data = require_metal("matrix transpose", self.storage().transpose(rows, cols));
        Matrix::from_storage(cols, rows, data)
    }

    /// Apply an analytic function elementwise, on the GPU; see
    /// [`Vector::analytic`].
    pub fn analytic(&self, f: Analytic) -> Self {
        let (rows, cols) = self.shape();
        let data = require_metal("matrix analytic operation", self.storage().unary(f));
        Matrix::from_storage(rows, cols, data)
    }

    /// Elementwise comparison with another resident matrix, on the GPU.
    #[track_caller]
    pub fn compare(&self, other: &Self, op: Compare) -> Self {
        assert_same_shape(self.shape(), other.shape(), "compare");
        let (rows, cols) = self.shape();
        let data = require_metal(
            "matrix comparison",
            self.storage().compare(other.storage(), op),
        );
        Matrix::from_storage(rows, cols, data)
    }

    /// Elementwise comparison against a scalar, on the GPU.
    pub fn compare_scalar(&self, scalar: T, op: Compare, scalar_left: bool) -> Self {
        let (rows, cols) = self.shape();
        let data = require_metal(
            "matrix-scalar comparison",
            self.storage().compare_scalar(scalar, op, scalar_left),
        );
        Matrix::from_storage(rows, cols, data)
    }

    /// Elementwise minimum with another resident matrix.
    ///
    /// # Panics
    ///
    /// If the two shapes differ.
    #[track_caller]
    pub fn min(&self, other: &Self) -> Self {
        self.compare(other, Compare::Min)
    }

    /// Elementwise maximum with another resident matrix.
    ///
    /// # Panics
    ///
    /// If the two shapes differ.
    #[track_caller]
    pub fn max(&self, other: &Self) -> Self {
        self.compare(other, Compare::Max)
    }

    /// The lesser of each element and `scalar`.
    pub fn min_scalar(&self, scalar: T) -> Self {
        self.compare_scalar(scalar, Compare::Min, false)
    }

    /// The greater of each element and `scalar`.
    pub fn max_scalar(&self, scalar: T) -> Self {
        self.compare_scalar(scalar, Compare::Max, false)
    }

    /// Confine every element to `[low, high]`, in one dispatch.
    ///
    /// # Panics
    ///
    /// If `low > high`.
    #[track_caller]
    pub fn clamp(&self, low: T, high: T) -> Self {
        assert_ordered_bounds(&low, &high);
        let (rows, cols) = self.shape();
        let data = require_metal("matrix clamp", self.storage().clamp(low, high));
        Matrix::from_storage(rows, cols, data)
    }

    /// Multiply every element by `scalar`, on the GPU.
    pub fn scale(&self, scalar: T) -> Self {
        self.broadcast_right(scalar, BinaryOp::Mul)
    }

    /// Implementation hook used by `math!` for tensor/scalar broadcasting.
    #[doc(hidden)]
    pub fn broadcast_right(&self, scalar: T, op: BinaryOp) -> Self {
        let (rows, cols) = self.shape();
        let data = require_metal(
            "matrix-scalar broadcast",
            self.storage().broadcast(scalar, op, false),
        );
        Matrix::from_storage(rows, cols, data)
    }

    /// Implementation hook used by `math!` for scalar/tensor broadcasting.
    #[doc(hidden)]
    pub fn broadcast_left(&self, scalar: T, op: BinaryOp) -> Self {
        let (rows, cols) = self.shape();
        let data = require_metal(
            "scalar-matrix broadcast",
            self.storage().broadcast(scalar, op, true),
        );
        Matrix::from_storage(rows, cols, data)
    }
}

/// Elementwise `op` over two resident vectors, in shared memory.
#[track_caller]
pub(super) fn vector_elementwise<T: MetalElement>(
    a: &Vector<T, Metal>,
    b: &Vector<T, Metal>,
    op: BinaryOp,
) -> Vector<T, Metal> {
    assert_same_len(a.len(), b.len(), op.name());
    let data = require_metal(
        "vector elementwise operation",
        a.storage().elementwise(b.storage(), op),
    );
    Vector::from_storage(a.len(), data)
}

/// Elementwise `op` over two resident matrices, in shared memory.
#[track_caller]
pub(super) fn matrix_elementwise<T: MetalElement>(
    a: &Matrix<T, Metal>,
    b: &Matrix<T, Metal>,
    op: BinaryOp,
) -> Matrix<T, Metal> {
    assert_same_shape(a.shape(), b.shape(), op.name());
    let (rows, cols) = a.shape();
    let data = require_metal(
        "matrix elementwise operation",
        a.storage().elementwise(b.storage(), op),
    );
    Matrix::from_storage(rows, cols, data)
}

/// Implements one operator for a resident tensor, by value and by reference.
///
/// `Metal`-backed tensors own a shared allocation, so the by-reference forms are
/// what most code wants: `&a * &b` leaves both operands usable.
macro_rules! resident_operator {
    ($Type:ident, $Trait:ident, $method:ident, $op:expr, $apply:ident) => {
        impl<T: MetalElement> $Trait for $Type<T, Metal> {
            type Output = Self;
            #[track_caller]
            fn $method(self, rhs: Self) -> Self::Output {
                $apply(&self, &rhs, $op)
            }
        }

        impl<T: MetalElement> $Trait<&$Type<T, Metal>> for &$Type<T, Metal> {
            type Output = $Type<T, Metal>;
            #[track_caller]
            fn $method(self, rhs: &$Type<T, Metal>) -> Self::Output {
                $apply(self, rhs, $op)
            }
        }
    };
}

resident_operator!(Vector, Add, add, BinaryOp::Add, vector_elementwise);
resident_operator!(Vector, Sub, sub, BinaryOp::Sub, vector_elementwise);
resident_operator!(Vector, Mul, mul, BinaryOp::Mul, vector_elementwise);
resident_operator!(Vector, Div, div, BinaryOp::Div, vector_elementwise);
resident_operator!(Vector, Rem, rem, BinaryOp::Rem, vector_elementwise);
resident_operator!(Matrix, Add, add, BinaryOp::Add, matrix_elementwise);
resident_operator!(Matrix, Sub, sub, BinaryOp::Sub, matrix_elementwise);
resident_operator!(Matrix, Mul, mul, BinaryOp::Mul, matrix_elementwise);
resident_operator!(Matrix, Div, div, BinaryOp::Div, matrix_elementwise);
resident_operator!(Matrix, Rem, rem, BinaryOp::Rem, matrix_elementwise);

impl<T: MetalElement> Neg for Vector<T, Metal> {
    type Output = Self;
    fn neg(self) -> Self {
        self.broadcast_right(-T::one(), BinaryOp::Mul)
    }
}

impl<T: MetalElement> Neg for &Vector<T, Metal> {
    type Output = Vector<T, Metal>;
    fn neg(self) -> Self::Output {
        self.broadcast_right(-T::one(), BinaryOp::Mul)
    }
}

impl<T: MetalElement> Neg for Matrix<T, Metal> {
    type Output = Self;
    fn neg(self) -> Self {
        self.broadcast_right(-T::one(), BinaryOp::Mul)
    }
}

impl<T: MetalElement> Neg for &Matrix<T, Metal> {
    type Output = Matrix<T, Metal>;
    fn neg(self) -> Self::Output {
        self.broadcast_right(-T::one(), BinaryOp::Mul)
    }
}

// The same formatting as the host-backed tensors, so a backend switch does not
// change what a printed tensor looks like.
impl<T: MetalElement> Display for Vector<T, Metal> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for x in self.as_slice() {
            write!(f, " {x}")?;
        }
        write!(f, " ]")
    }
}

impl<T: MetalElement> Display for Matrix<T, Metal> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let values = self.as_slice();
        let (rows, cols) = self.shape();
        for r in 0..rows {
            if r > 0 {
                writeln!(f)?;
            }
            write!(f, "[")?;
            for x in &values[r * cols..r * cols + cols] {
                write!(f, " {x}")?;
            }
            write!(f, " ]")?;
        }
        Ok(())
    }
}
