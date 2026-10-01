//! Tensor operations for the [`Metal`] backend.
//!
//! Everything here keeps its operands *and* its result in GPU-shared memory:
//! [`matmul`](Matrix::matmul), [`matvec`](Matrix::matvec),
//! [`vecmat`](Vector::vecmat), the elementwise operators and scalar
//! broadcasting all encode straight over the existing allocations, so a chain of
//! operations costs one upload at the start and one download at the end rather
//! than a pair per call.
//!
//! [`Vector::dot`] deliberately is not a GPU dispatch. It reduces on the CPU
//! over the shared allocations — still copy-free — because a `1×N·N×1` matmul
//! would put the entire reduction on one GPU thread.
//!
//! When a dispatch cannot run at all — no Metal device, or an operation the
//! shaders do not implement, like `%` — the operands move to the [`Host`]
//! backend, its kernels produce the result, and the result moves back. That
//! keeps the answers identical everywhere; it costs the copies this backend
//! exists to avoid, but only on a machine that could not have avoided them.
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
    Analytic, Backend, BinaryOp, Compare, Host, Kernels, Matrix, Metal, Reduce, SortOrder, Vector,
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
                match self
                    .storage()
                    .matmul_f32(other.storage(), rows, inner, cols)
                {
                    Some(data) => Matrix::from_storage(rows, cols, data),
                    None => {
                        let left = self.to_f32::<Host>();
                        let right = other.to_f32::<Host>();
                        left.matmul(&right).to_backend()
                    }
                }
            }
        }
    };
}

low_precision_tensors!(f16);
low_precision_tensors!(bf16);

impl<T: MetalElement> Vector<T, Metal> {
    /// Whether the elements really are in GPU-shared memory.
    ///
    /// `false` means the process has no Metal device, so this vector fell back
    /// to CPU storage and CPU kernels. Results are unaffected.
    pub fn is_device_resident(&self) -> bool {
        self.storage().is_device_resident()
    }

    /// Dot product with a vector of the same length.
    ///
    /// The reduction runs on the CPU, reading both shared allocations in place —
    /// shared storage is ordinary cached memory from the CPU's side, so this
    /// still copies nothing. The GPU alternative available here, a `1×N` by
    /// `N×1` matmul, would run the whole sum on a single thread. The sum
    /// accumulates in `f32` and rounds to `T` once.
    ///
    /// # Panics
    ///
    /// If the two lengths differ.
    #[track_caller]
    pub fn dot(&self, other: &Self) -> T {
        assert_same_len(self.len(), other.len(), "dot");
        reduce_dot(self.as_slice(), other.as_slice())
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
        match self.storage().matmul(m.storage(), 1, rows, cols) {
            Some(data) => Vector::from_storage(cols, data),
            None => self
                .to_backend::<Host>()
                .vecmat(&m.to_backend::<Host>())
                .to_backend(),
        }
    }

    /// Apply an analytic function elementwise, on the GPU.
    ///
    /// The host tensor of the same name maps the scalar trait from
    /// [`crate::numbers`] over its elements; resident tensors run the unary
    /// kernel compiled for their element type instead.
    /// Both spellings — this and the named methods in
    /// [`analytic`](crate::tensors::analytic) — are the same operation.
    pub fn analytic(&self, f: Analytic) -> Self {
        match self.storage().unary(f) {
            Some(data) => Vector::from_storage(self.len(), data),
            None => Host::vector_unary(&self.to_backend::<Host>(), f).to_backend(),
        }
    }

    /// Elementwise comparison with another resident vector, on the GPU.
    #[track_caller]
    pub fn compare(&self, other: &Self, op: Compare) -> Self {
        assert_same_len(self.len(), other.len(), "compare");
        match self.storage().compare(other.storage(), op) {
            Some(data) => Vector::from_storage(self.len(), data),
            None => {
                Host::vector_compare(&self.to_backend::<Host>(), &other.to_backend::<Host>(), op)
                    .to_backend()
            }
        }
    }

    /// Elementwise comparison against a scalar, on the GPU.
    pub fn compare_scalar(&self, scalar: T, op: Compare, scalar_left: bool) -> Self {
        match self.storage().compare_scalar(scalar, op, scalar_left) {
            Some(data) => Vector::from_storage(self.len(), data),
            None => {
                Host::vector_compare_scalar(&self.to_backend::<Host>(), scalar, op, scalar_left)
                    .to_backend()
            }
        }
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
        match self.storage().clamp(low, high) {
            Some(data) => Vector::from_storage(self.len(), data),
            None => Host::vector_clamp(&self.to_backend::<Host>(), low, high).to_backend(),
        }
    }

    /// Fold the whole vector to one value with a GPU tree reduction. An empty
    /// vector gives [`op.identity()`](Reduce::identity).
    ///
    /// The answer is a number rather than a tensor, so this necessarily comes
    /// back to the CPU: it is a synchronization point, unlike the operations
    /// that leave their result resident.
    pub fn reduce(&self, op: Reduce) -> T {
        match self.storage().reduce(op) {
            // The fold ran in `f32`; this is its one rounding to `T`.
            Some(value) => T::from_f64(f64::from(value)),
            None => Host::vector_reduce(&self.to_backend::<Host>(), op),
        }
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
        match self.storage().prefix_sum() {
            Some(data) => Vector::from_storage(self.len(), data),
            None => Host::vector_prefix_sum(&self.to_backend::<Host>()).to_backend(),
        }
    }

    /// Sort in [`SortOrder`]'s total order, staying resident.
    ///
    /// A bitonic sort over integer sort keys, so the result matches a host
    /// total-order sort exactly, NaNs included.
    pub fn sorted(&self, order: SortOrder) -> Self {
        match self.storage().sort(order) {
            Some(data) => Vector::from_storage(self.len(), data),
            None => Host::vector_sort(&self.to_backend::<Host>(), order).to_backend(),
        }
    }

    /// Sort under an arbitrary comparator.
    ///
    /// A closure cannot cross to the GPU, so this is the one operation here that
    /// leaves the device: the elements are read out of shared memory, sorted on
    /// the CPU, and stored back. [`sorted`](Self::sorted) is the resident
    /// version, and covers everything a total order can express.
    pub fn sorted_by(&self, compare: impl FnMut(&T, &T) -> Ordering) -> Self {
        let mut values = self.to_vec();
        values.sort_by(compare);
        Vector::new(values).to_backend()
    }

    /// Multiply every element by `scalar`, on the GPU.
    pub fn scale(&self, scalar: T) -> Self {
        self.broadcast_right(scalar, BinaryOp::Mul)
    }

    /// Implementation hook used by `math!` for tensor/scalar broadcasting.
    #[doc(hidden)]
    pub fn broadcast_right(&self, scalar: T, op: BinaryOp) -> Self {
        match self.storage().broadcast(scalar, op, false) {
            Some(data) => Vector::from_storage(self.len(), data),
            None => self
                .to_backend::<Host>()
                .broadcast_right(scalar, op)
                .to_backend(),
        }
    }

    /// Implementation hook used by `math!` for scalar/tensor broadcasting.
    #[doc(hidden)]
    pub fn broadcast_left(&self, scalar: T, op: BinaryOp) -> Self {
        match self.storage().broadcast(scalar, op, true) {
            Some(data) => Vector::from_storage(self.len(), data),
            None => self
                .to_backend::<Host>()
                .broadcast_left(scalar, op)
                .to_backend(),
        }
    }
}

impl<T: MetalElement> Matrix<T, Metal> {
    /// Whether the elements really are in GPU-shared memory; see
    /// [`Vector::is_device_resident`].
    pub fn is_device_resident(&self) -> bool {
        self.storage().is_device_resident()
    }

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
        match self.storage().matmul(other.storage(), rows, inner, cols) {
            Some(data) => Matrix::from_storage(rows, cols, data),
            None => self
                .to_backend::<Host>()
                .matmul(&other.to_backend::<Host>())
                .to_backend(),
        }
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
        if self
            .storage()
            .matmul_accumulate(other.storage(), addend.storage_mut(), rows, inner, cols)
            .is_some()
        {
            return addend;
        }
        self.to_backend::<Host>()
            .matmul_add(&other.to_backend::<Host>(), addend.to_backend::<Host>())
            .to_backend()
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
        match self.storage().matmul(v.storage(), rows, cols, 1) {
            Some(data) => Vector::from_storage(rows, data),
            None => self
                .to_backend::<Host>()
                .matvec(&v.to_backend::<Host>())
                .to_backend(),
        }
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
        if self
            .storage()
            .matmul_accumulate(v.storage(), addend.storage_mut(), rows, cols, 1)
            .is_some()
        {
            return addend;
        }
        self.to_backend::<Host>()
            .matvec_add(&v.to_backend::<Host>(), addend.to_backend::<Host>())
            .to_backend()
    }

    /// Transpose: an `R×C` matrix becomes `C×R`.
    pub fn transpose(&self) -> Matrix<T, Metal> {
        let (rows, cols) = self.shape();
        match self.storage().transpose(rows, cols) {
            Some(data) => Matrix::from_storage(cols, rows, data),
            None => self.to_backend::<Host>().transpose().to_backend(),
        }
    }

    /// Apply an analytic function elementwise, on the GPU; see
    /// [`Vector::analytic`].
    pub fn analytic(&self, f: Analytic) -> Self {
        let (rows, cols) = self.shape();
        match self.storage().unary(f) {
            Some(data) => Matrix::from_storage(rows, cols, data),
            None => Host::matrix_unary(&self.to_backend::<Host>(), f).to_backend(),
        }
    }

    /// Elementwise comparison with another resident matrix, on the GPU.
    #[track_caller]
    pub fn compare(&self, other: &Self, op: Compare) -> Self {
        assert_same_shape(self.shape(), other.shape(), "compare");
        let (rows, cols) = self.shape();
        match self.storage().compare(other.storage(), op) {
            Some(data) => Matrix::from_storage(rows, cols, data),
            None => {
                Host::matrix_compare(&self.to_backend::<Host>(), &other.to_backend::<Host>(), op)
                    .to_backend()
            }
        }
    }

    /// Elementwise comparison against a scalar, on the GPU.
    pub fn compare_scalar(&self, scalar: T, op: Compare, scalar_left: bool) -> Self {
        let (rows, cols) = self.shape();
        match self.storage().compare_scalar(scalar, op, scalar_left) {
            Some(data) => Matrix::from_storage(rows, cols, data),
            None => {
                Host::matrix_compare_scalar(&self.to_backend::<Host>(), scalar, op, scalar_left)
                    .to_backend()
            }
        }
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
        match self.storage().clamp(low, high) {
            Some(data) => Matrix::from_storage(rows, cols, data),
            None => Host::matrix_clamp(&self.to_backend::<Host>(), low, high).to_backend(),
        }
    }

    /// Multiply every element by `scalar`, on the GPU.
    pub fn scale(&self, scalar: T) -> Self {
        self.broadcast_right(scalar, BinaryOp::Mul)
    }

    /// Implementation hook used by `math!` for tensor/scalar broadcasting.
    #[doc(hidden)]
    pub fn broadcast_right(&self, scalar: T, op: BinaryOp) -> Self {
        let (rows, cols) = self.shape();
        match self.storage().broadcast(scalar, op, false) {
            Some(data) => Matrix::from_storage(rows, cols, data),
            None => self
                .to_backend::<Host>()
                .broadcast_right(scalar, op)
                .to_backend(),
        }
    }

    /// Implementation hook used by `math!` for scalar/tensor broadcasting.
    #[doc(hidden)]
    pub fn broadcast_left(&self, scalar: T, op: BinaryOp) -> Self {
        let (rows, cols) = self.shape();
        match self.storage().broadcast(scalar, op, true) {
            Some(data) => Matrix::from_storage(rows, cols, data),
            None => self
                .to_backend::<Host>()
                .broadcast_left(scalar, op)
                .to_backend(),
        }
    }
}

/// Sum of products over two CPU-readable slices, vectorized where possible,
/// accumulated in `f32` and rounded to `T` once.
fn reduce_dot<T: MetalElement>(a: &[T], b: &[T]) -> T {
    if let Some(total) = crate::compact::dot(a, b) {
        return total;
    }
    #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
    if let (Some(a), Some(b)) = (as_f32(a), as_f32(b)) {
        return T::from_f64(f64::from(crate::simd::f32k::dot(a, b)));
    }
    let total: f32 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| x.into_f64() as f32 * y.into_f64() as f32)
        .sum();
    T::from_f64(f64::from(total))
}

/// `values` as `f32`s, when that is what they are.
#[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
fn as_f32<T: 'static>(values: &[T]) -> Option<&[f32]> {
    // SAFETY: equal `TypeId`s mean one type, so the layouts are identical.
    (std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>())
        .then(|| unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<f32>(), values.len()) })
}

/// Elementwise `op` over two resident vectors, in shared memory.
#[track_caller]
pub(super) fn vector_elementwise<T: MetalElement>(
    a: &Vector<T, Metal>,
    b: &Vector<T, Metal>,
    op: BinaryOp,
) -> Vector<T, Metal> {
    assert_same_len(a.len(), b.len(), op.name());
    match a.storage().elementwise(b.storage(), op) {
        Some(data) => Vector::from_storage(a.len(), data),
        None => Host::vector_elementwise(&a.to_backend::<Host>(), &b.to_backend::<Host>(), op)
            .to_backend(),
    }
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
    match a.storage().elementwise(b.storage(), op) {
        Some(data) => Matrix::from_storage(rows, cols, data),
        None => Host::matrix_elementwise(&a.to_backend::<Host>(), &b.to_backend::<Host>(), op)
            .to_backend(),
    }
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
