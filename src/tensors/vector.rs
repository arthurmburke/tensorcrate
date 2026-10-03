//! [`Vector`]: a one-dimensional tensor whose length is a runtime value.

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::fmt::{self, Display};
use std::ops::Index;

use super::matrix::{Block, assemble_blocks};
use super::order::{ordered_max, ordered_min};
use super::shape::{assert_inner, assert_ordered_bounds, assert_same_len};
#[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
use super::simd_dispatch;
use super::{Backend, BinaryOp, Compare, Host, Matrix, Reduce, SortOrder};
use crate::numbers::{Coefficient, Real};

/// A vector whose length is fixed when it is built, backed by a `Vec<T>` on the
/// default [`Host`] backend.
pub struct Vector<T, B: Backend = Host> {
    len: usize,
    data: B::Storage<T>,
}

// The storage type varies with the backend, so these are the derives written by
// hand: which of them a tensor gets depends on what its storage supports.
impl<T, B: Backend> Clone for Vector<T, B>
where
    B::Storage<T>: Clone,
{
    fn clone(&self) -> Self {
        Vector {
            len: self.len,
            data: self.data.clone(),
        }
    }
}

impl<T, B: Backend> PartialEq for Vector<T, B>
where
    B::Storage<T>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.data == other.data
    }
}

impl<T, B: Backend> Eq for Vector<T, B> where B::Storage<T>: Eq {}

impl<T, B: Backend> fmt::Debug for Vector<T, B>
where
    B::Storage<T>: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Vector")
            .field("len", &self.len)
            .field("data", &self.data)
            .finish()
    }
}

/// The shape queries, which read a field and so need nothing of the backend or
/// the element type.
impl<T, B: Backend> Vector<T, B> {
    /// The number of elements.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether this vector holds no elements.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Copyable vectors on any backend. These are storage operations; arithmetic
/// remains implemented for the element/backend combinations with kernels.
impl<T: Copy + 'static, B: Backend> Vector<T, B> {
    /// Move this vector's elements onto backend `B2`.
    ///
    /// This is the operation that copies: onto [`Metal`] it is an upload into
    /// shared memory, off it a download. Everything in between stays put.
    ///
    /// ```
    /// # #[cfg(all(feature = "metal", target_os = "macos"))] {
    /// use tensorcrate::tensors::{Host, Metal, Vector};
    ///
    /// let v = Vector::new([1.0f32, 2.0, 3.0]);
    /// let resident = v.to_backend::<Metal>();
    /// assert_eq!(resident.to_backend::<Host>(), v);
    /// # }
    /// ```
    pub fn to_backend<B2: Backend>(&self) -> Vector<T, B2> {
        Vector {
            len: self.len,
            data: super::backend::transfer::<T, B, B2>(&self.data),
        }
    }

    /// A vector of `len` elements, every one set to `value`, allocated directly
    /// on backend `B`.
    pub fn filled(len: usize, value: T) -> Self {
        Vector {
            len,
            data: B::store(&vec![value; len]),
        }
    }

    /// Borrow the elements as a slice, without copying.
    pub fn as_slice(&self) -> &[T] {
        B::as_slice(&self.data)
    }

    /// Build from values on backend `B`. The length is the slice's.
    pub(crate) fn build(values: &[T]) -> Self {
        Vector {
            len: values.len(),
            data: B::store(values),
        }
    }

    /// Attach a length to storage that already holds exactly that many values —
    /// how a tensor is rebuilt from the result of a kernel dispatch.
    pub(crate) fn from_storage(len: usize, data: B::Storage<T>) -> Self {
        Vector { len, data }
    }

    /// The backend storage itself, which the kernels hand straight to a
    /// dispatch.
    pub(crate) fn storage(&self) -> &B::Storage<T> {
        &self.data
    }

    /// The backend storage itself, for a dispatch that accumulates in place.
    pub(crate) fn storage_mut(&mut self) -> &mut B::Storage<T> {
        &mut self.data
    }

    /// Consume this vector and take its storage, which is how a reshape moves
    /// the elements instead of copying them.
    pub(crate) fn into_storage(self) -> B::Storage<T> {
        self.data
    }

    /// Copy the elements into a `Vec`.
    pub fn to_vec(&self) -> Vec<T> {
        self.as_slice().to_vec()
    }

    /// Vectors as the rows of a matrix, in order. Every vector must have the
    /// same length.
    ///
    /// ```
    /// use tensorcrate::tensors::{Matrix, Vector};
    ///
    /// let a = Vector::new([1.0f32, 2.0]);
    /// let b = Vector::new([3.0f32, 4.0]);
    /// assert_eq!(Vector::vstack([&a, &b]), Matrix::from_rows([[1.0, 2.0], [3.0, 4.0]]));
    /// ```
    ///
    /// # Panics
    ///
    /// If there are no vectors, or their lengths differ.
    #[track_caller]
    pub fn vstack<V: Borrow<Self>>(parts: impl IntoIterator<Item = V>) -> Matrix<T, B> {
        let parts = parts.into_iter().collect::<Vec<_>>();
        let first = parts
            .first()
            .unwrap_or_else(|| panic!("vstack: there are no vectors to stack"))
            .borrow();
        Self::stack_rows(first.len, &parts)
    }

    /// Vectors as the columns of a matrix, in order. Every vector must have
    /// the same length.
    ///
    /// ```
    /// use tensorcrate::tensors::{Matrix, Vector};
    ///
    /// let a = Vector::new([1.0f32, 2.0]);
    /// let b = Vector::new([3.0f32, 4.0]);
    /// assert_eq!(Vector::hstack([&a, &b]), Matrix::from_rows([[1.0, 3.0], [2.0, 4.0]]));
    /// ```
    ///
    /// # Panics
    ///
    /// If there are no vectors, or their lengths differ.
    #[track_caller]
    pub fn hstack<V: Borrow<Self>>(parts: impl IntoIterator<Item = V>) -> Matrix<T, B> {
        let parts = parts.into_iter().collect::<Vec<_>>();
        let first = parts
            .first()
            .unwrap_or_else(|| panic!("hstack: there are no vectors to stack"))
            .borrow();
        Self::stack_columns(first.len, &parts)
    }

    /// [`vstack`](Self::vstack) of vectors of `len` elements, of which there
    /// may be none.
    #[track_caller]
    pub(crate) fn stack_rows<V: Borrow<Self>>(len: usize, parts: &[V]) -> Matrix<T, B> {
        let blocks = parts
            .iter()
            .enumerate()
            .map(|(row, part)| {
                let part = part.borrow();
                assert_eq!(
                    part.len, len,
                    "vstack: vectors of {len} and {} elements differ in length",
                    part.len
                );
                Block {
                    source: &part.data,
                    shape: (1, len),
                    at: (row, 0),
                }
            })
            .collect::<Vec<_>>();
        let rows = parts.len();
        Matrix::from_storage(rows, len, assemble_blocks::<T, B>((rows, len), &blocks))
    }

    /// [`hstack`](Self::hstack) of vectors of `len` elements, of which there
    /// may be none.
    #[track_caller]
    pub(crate) fn stack_columns<V: Borrow<Self>>(len: usize, parts: &[V]) -> Matrix<T, B> {
        let blocks = parts
            .iter()
            .enumerate()
            .map(|(column, part)| {
                let part = part.borrow();
                assert_eq!(
                    part.len, len,
                    "hstack: vectors of {len} and {} elements differ in length",
                    part.len
                );
                Block {
                    source: &part.data,
                    shape: (len, 1),
                    at: (0, column),
                }
            })
            .collect::<Vec<_>>();
        let cols = parts.len();
        Matrix::from_storage(len, cols, assemble_blocks::<T, B>((len, cols), &blocks))
    }

    /// Consume this vector and view its elements as a `1 × len` row matrix.
    ///
    /// On the Metal backend this only changes the recorded shape; the existing
    /// allocation is reused without a copy or kernel dispatch.
    pub fn into_row_matrix(self) -> Matrix<T, B> {
        Matrix::from_storage(1, self.len, self.data)
    }

    /// Consume this vector and view its elements as a `len × 1` column matrix.
    ///
    /// As with [`into_row_matrix`](Self::into_row_matrix), Metal reuses the
    /// existing allocation.
    pub fn into_column_matrix(self) -> Matrix<T, B> {
        Matrix::from_storage(self.len, 1, self.data)
    }
}

impl<T: Real, B: Backend> Vector<T, B> {
    /// An arithmetic progression: `start`, `start + step`, `start + 2·step`, …
    pub fn ramp(len: usize, start: T, step: T) -> Self {
        let values = (0..len)
            .map(|index| start + step * T::from_f64(index as f64))
            .collect::<Vec<_>>();
        Vector {
            len,
            data: B::store(&values),
        }
    }
}

impl<T> Vector<T, Host> {
    /// A vector from its elements.
    ///
    /// Both a fixed-size array and a `Vec` are accepted, so a length known in
    /// source and one computed at runtime are written the same way:
    ///
    /// ```
    /// use tensorcrate::tensors::Vector;
    ///
    /// let literal = Vector::new([1.0, 2.0, 3.0]);
    /// let computed = Vector::new((0..3).map(|i| i as f64 + 1.0).collect::<Vec<_>>());
    /// assert_eq!(literal, computed);
    /// ```
    pub fn new(data: impl Into<Vec<T>>) -> Self {
        let data = data.into();
        Vector {
            len: data.len(),
            data,
        }
    }

    /// Borrow the elements as a slice.
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Borrow the elements as a mutable slice.
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Consume this vector and take its elements.
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    pub fn get(&self, i: usize) -> Option<&T> {
        self.data.get(i)
    }

    /// Apply `f` to every element, producing a vector of the new element type —
    /// e.g. lifting a `Vector<f64>` into a `Vector<Complex<f64>>`.
    pub fn map<U>(&self, f: impl Fn(&T) -> U) -> Vector<U, Host> {
        Vector {
            len: self.len,
            data: self.data.iter().map(f).collect(),
        }
    }

    /// Combine two vectors element by element with `f`, producing a vector of
    /// the new element type.
    ///
    /// # Panics
    ///
    /// If the two lengths differ.
    #[track_caller]
    pub fn zip_map<U, V>(
        &self,
        other: &Vector<U, Host>,
        f: impl Fn(&T, &U) -> V,
    ) -> Vector<V, Host> {
        assert_same_len(self.len, other.len, "zip_map");
        Vector {
            len: self.len,
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| f(a, b))
                .collect(),
        }
    }
}

impl<T> Index<usize> for Vector<T, Host> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        &self.data[index]
    }
}

impl<T: Coefficient> Vector<T, Host> {
    /// A vector of `len` zeros.
    pub fn zeros(len: usize) -> Self {
        Vector {
            len,
            data: vec![T::zero(); len],
        }
    }

    /// A vector of `len` elements, every one set to `value`.
    pub fn repeat(len: usize, value: T) -> Self {
        Vector {
            len,
            data: vec![value; len],
        }
    }

    /// Multiply every element by `scalar`.
    pub fn scale(&self, scalar: T) -> Self {
        self.broadcast_right(scalar, BinaryOp::Mul)
    }

    /// Implementation hook used by `math!` for tensor/scalar broadcasting.
    #[doc(hidden)]
    pub fn broadcast_right(&self, scalar: T, op: BinaryOp) -> Self {
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let mut out = vec![T::zero(); self.len];
            if simd_dispatch::broadcast(&self.data, scalar, op, false, &mut out) {
                return Vector::new(out);
            }
        }
        self.map(|&value| match op {
            BinaryOp::Add => value + scalar,
            BinaryOp::Sub => value - scalar,
            BinaryOp::Mul => value * scalar,
            BinaryOp::Div => value / scalar,
            BinaryOp::Rem => value % scalar,
        })
    }

    /// Implementation hook used by `math!` for scalar/tensor broadcasting.
    #[doc(hidden)]
    pub fn broadcast_left(&self, scalar: T, op: BinaryOp) -> Self {
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let mut out = vec![T::zero(); self.len];
            if simd_dispatch::broadcast(&self.data, scalar, op, true, &mut out) {
                return Vector::new(out);
            }
        }
        self.map(|&value| match op {
            BinaryOp::Add => scalar + value,
            BinaryOp::Sub => scalar - value,
            BinaryOp::Mul => scalar * value,
            BinaryOp::Div => scalar / value,
            BinaryOp::Rem => scalar % value,
        })
    }

    /// Dot product.
    ///
    /// # Panics
    ///
    /// If the two lengths differ.
    #[track_caller]
    pub fn dot(&self, other: &Vector<T, Host>) -> T {
        assert_same_len(self.len, other.len, "dot");
        // `f16` and `bf16` accumulate in `f32` and round once.
        if let Some(output) = crate::compact::dot(&self.data, &other.data) {
            return output;
        }
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        if let Some(output) = simd_dispatch::dot(&self.data, &other.data) {
            return output;
        }
        let mut sum = T::zero();
        for i in 0..self.len {
            sum = sum + self.data[i] * other.data[i];
        }
        sum
    }

    /// Row vector times matrix: `(1×N)·(N×C) = (1×C)`.
    ///
    /// # Panics
    ///
    /// If this vector's length is not the matrix's row count.
    #[track_caller]
    pub fn vecmat(&self, m: &Matrix<T, Host>) -> Vector<T, Host> {
        assert_inner((1, self.len), m.shape(), "vecmat");
        let (rows, cols) = m.shape();
        let matrix = m.as_slice();
        let mut out = vec![T::zero(); cols];

        if crate::compact::matmul(&self.data, matrix, 1, rows, cols, &mut out, false) {
            return Vector::new(out);
        }

        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        if simd_dispatch::vecmat(&self.data, matrix, rows, cols, &mut out) {
            return Vector::new(out);
        }

        for (j, slot) in out.iter_mut().enumerate() {
            let mut sum = T::zero();
            for p in 0..rows {
                sum = sum + self.data[p] * matrix[p * cols + j];
            }
            *slot = sum;
        }
        Vector::new(out)
    }
}

impl<T> Vector<T, Host> {
    /// Sort in place under an arbitrary comparator.
    ///
    /// This is [`slice::sort_by`], so the sort is stable and the comparator may
    /// be anything — a key projection, a reversed order, a tie-break across two
    /// fields. Floats have no total [`Ord`], which is exactly why the ordering
    /// is a parameter here; [`f32::total_cmp`] is the usual argument, and
    /// [`sort`](Vector::sort) is the name for it.
    ///
    /// ```
    /// use tensorcrate::tensors::Vector;
    ///
    /// let mut v = Vector::new([3.0f32, -1.0, 2.0]);
    /// v.sort_by(|left, right| right.total_cmp(left)); // descending
    /// assert_eq!(v.data(), [3.0, 2.0, -1.0]);
    /// ```
    pub fn sort_by(&mut self, compare: impl FnMut(&T, &T) -> Ordering) {
        self.data.sort_by(compare);
    }

    /// [`sort_by`](Vector::sort_by) into a new vector, leaving this one alone.
    pub fn sorted_by(&self, compare: impl FnMut(&T, &T) -> Ordering) -> Self
    where
        T: Clone,
    {
        let mut sorted = self.clone();
        sorted.sort_by(compare);
        sorted
    }
}

impl<T: Coefficient> Vector<T, Host> {
    /// The sum of every element; `0` for an empty vector.
    ///
    /// `f16` and `bf16` sum in `f32` and round once.
    pub fn sum(&self) -> T {
        if let Some(total) = crate::compact::reduce(&self.data, Reduce::Sum) {
            return total;
        }
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        if let Some(total) = simd_dispatch::reduce(&self.data, Reduce::Sum) {
            return total;
        }
        self.data
            .iter()
            .fold(T::zero(), |total, &value| total + value)
    }

    /// Inclusive prefix sum: element `i` of the result is the sum of elements
    /// `0..=i`.
    ///
    /// The scan is the one operation here with a serial dependency — each output
    /// needs the one before it — so the CPU path is an ordinary running total,
    /// which is already about as fast as the memory it streams. The parallel
    /// version is the GPU one: see
    /// [`Kernels::vector_prefix_sum`](crate::tensors::Kernels::vector_prefix_sum),
    /// where `log n` sweeps beat the serial chain because there are thousands of
    /// threads to spend on it.
    ///
    /// `f16` and `bf16` keep the running total in `f32`, rounding each output
    /// once.
    pub fn prefix_sum(&self) -> Self {
        if let Some(out) = crate::compact::prefix_sum(&self.data) {
            return Vector::new(out);
        }
        let mut running = T::zero();
        let mut out = Vec::with_capacity(self.len);
        for &value in &self.data {
            running = running + value;
            out.push(running);
        }
        Vector::new(out)
    }
}

impl<T: Coefficient + PartialOrd> Vector<T, Host> {
    /// Elementwise minimum with another vector.
    ///
    /// # Panics
    ///
    /// If the two lengths differ.
    #[track_caller]
    pub fn min(&self, other: &Self) -> Self {
        assert_same_len(self.len, other.len, "min");
        self.pairwise(other, Compare::Min, ordered_min)
    }

    /// Elementwise maximum with another vector.
    ///
    /// # Panics
    ///
    /// If the two lengths differ.
    #[track_caller]
    pub fn max(&self, other: &Self) -> Self {
        assert_same_len(self.len, other.len, "max");
        self.pairwise(other, Compare::Max, ordered_max)
    }

    /// The lesser of each element and `scalar`.
    pub fn min_scalar(&self, scalar: T) -> Self {
        self.against_scalar(scalar, Compare::Min, ordered_min)
    }

    /// The greater of each element and `scalar` — `max_scalar(0)` is a relu.
    pub fn max_scalar(&self, scalar: T) -> Self {
        self.against_scalar(scalar, Compare::Max, ordered_max)
    }

    /// Confine every element to `[low, high]`.
    ///
    /// One pass, rather than the two a `max_scalar` followed by a `min_scalar`
    /// would make.
    ///
    /// ```
    /// use tensorcrate::tensors::Vector;
    ///
    /// let v = Vector::new([-2.0f32, 0.5, 7.0]);
    /// assert_eq!(v.clamp(0.0, 1.0).data(), [0.0, 0.5, 1.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// If `low > high`, which has no answer to give. A NaN element clamps to
    /// `low`, following `x.max(low).min(high)`.
    #[track_caller]
    pub fn clamp(&self, low: T, high: T) -> Self {
        assert_ordered_bounds(&low, &high);
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let mut out = vec![T::zero(); self.len];
            if simd_dispatch::clamp(&self.data, low, high, &mut out) {
                return Vector::new(out);
            }
        }
        self.map(|&value| ordered_min(ordered_max(value, low), high))
    }

    /// The smallest element, or `None` when there are none.
    pub fn minimum(&self) -> Option<T> {
        self.fold_extreme(Reduce::Min, ordered_min)
    }

    /// The largest element, or `None` when there are none.
    pub fn maximum(&self) -> Option<T> {
        self.fold_extreme(Reduce::Max, ordered_max)
    }

    fn pairwise(&self, other: &Self, op: Compare, scalar: fn(T, T) -> T) -> Self {
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let mut out = vec![T::zero(); self.len];
            if simd_dispatch::compare(&self.data, &other.data, op, &mut out) {
                return Vector::new(out);
            }
        }
        let _ = op;
        Vector::new(
            self.data
                .iter()
                .zip(&other.data)
                .map(|(&left, &right)| scalar(left, right))
                .collect::<Vec<_>>(),
        )
    }

    fn against_scalar(&self, value: T, op: Compare, scalar: fn(T, T) -> T) -> Self {
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let mut out = vec![T::zero(); self.len];
            if simd_dispatch::compare_scalar(&self.data, value, op, false, &mut out) {
                return Vector::new(out);
            }
        }
        let _ = op;
        self.map(|&element| scalar(element, value))
    }

    fn fold_extreme(&self, op: Reduce, scalar: fn(T, T) -> T) -> Option<T> {
        if self.data.is_empty() {
            return None;
        }
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        if let Some(extreme) = simd_dispatch::reduce(&self.data, op) {
            return Some(extreme);
        }
        let _ = op;
        let (&first, rest) = self.data.split_first()?;
        Some(rest.iter().fold(first, |best, &value| scalar(best, value)))
    }
}

impl<T: Real> Vector<T, Host> {
    /// Elementwise comparison with another vector.
    ///
    /// The same operation the [`Metal`] backend runs as one dispatch, so code
    /// written against [`Kernels`](super::Kernels) means the same thing on either.
    ///
    /// # Panics
    ///
    /// If the two lengths differ.
    #[track_caller]
    pub fn compare(&self, other: &Self, op: Compare) -> Self {
        assert_same_len(self.len, other.len, "compare");
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let mut out = vec![T::zero(); self.len];
            if simd_dispatch::compare(&self.data, &other.data, op, &mut out) {
                return Vector::new(out);
            }
        }
        Vector::new(
            self.data
                .iter()
                .zip(&other.data)
                .map(|(&left, &right)| op.value(left, right))
                .collect::<Vec<_>>(),
        )
    }

    /// Elementwise comparison against a scalar; `scalar_left` puts the scalar on
    /// the left, which matters for every op but `Min` and `Max`.
    pub fn compare_scalar(&self, scalar: T, op: Compare, scalar_left: bool) -> Self {
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let mut out = vec![T::zero(); self.len];
            if simd_dispatch::compare_scalar(&self.data, scalar, op, scalar_left, &mut out) {
                return Vector::new(out);
            }
        }
        self.map(|&value| {
            if scalar_left {
                op.value(scalar, value)
            } else {
                op.value(value, scalar)
            }
        })
    }

    /// Fold the whole vector to one value. An empty vector gives
    /// [`op.identity()`](Reduce::identity).
    pub fn reduce(&self, op: Reduce) -> T {
        if let Some(total) = crate::compact::reduce(&self.data, op) {
            return total;
        }
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        if let Some(total) = simd_dispatch::reduce(&self.data, op) {
            return total;
        }
        op.fold(&self.data)
    }

    /// Sort in place, in [`SortOrder`]'s total order.
    ///
    /// The comparator is [`f32::total_cmp`], so `−0.0` precedes `+0.0` and NaNs
    /// sort to the ends by sign instead of landing wherever the partial order
    /// left them. That is the same order the GPU sort produces.
    pub fn sort(&mut self, order: SortOrder) {
        self.data.sort_unstable_by(order.comparator());
    }

    /// [`sort`](Vector::sort) into a new vector.
    pub fn sorted(&self, order: SortOrder) -> Self {
        let mut sorted = self.clone();
        sorted.sort(order);
        sorted
    }
}

impl<T: Display> Display for Vector<T, Host> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for x in &self.data {
            write!(f, " {x}")?;
        }
        write!(f, " ]")
    }
}

impl<T: Coefficient> Vector<T, Host> {
    #[track_caller]
    pub(super) fn zip_with(&self, rhs: &Self, op: BinaryOp, f: impl Fn(T, T) -> T) -> Self {
        assert_same_len(self.len, rhs.len, op.name());

        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let mut out = vec![T::zero(); self.len];
            if simd_dispatch::elementwise(&self.data, &rhs.data, op, &mut out) {
                return Vector::new(out);
            }
        }

        Vector::new(
            self.data
                .iter()
                .zip(&rhs.data)
                .map(|(&a, &b)| f(a, b))
                .collect::<Vec<_>>(),
        )
    }
}
