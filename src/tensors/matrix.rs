//! [`Matrix`]: a two-dimensional, row-major tensor whose shape is a runtime
//! value.

use std::cmp::Ordering;
use std::fmt::{self, Display};
use std::ops::Index;

#[cfg(target_os = "macos")]
use super::accelerate_dispatch;
use super::order::{ordered_max, ordered_min};
use super::shape::{assert_inner, assert_ordered_bounds, assert_same_len, assert_same_shape};
#[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
use super::simd_dispatch;
use super::{Backend, BinaryOp, Compare, Host, Vector};
use crate::errors::Error;
use crate::numbers::{Coefficient, Real};

/// A matrix whose shape is fixed when it is built, backed by a flat row-major
/// `Vec<T>` on the default [`Host`] backend.
pub struct Matrix<T, B: Backend = Host> {
    rows: usize,
    cols: usize,
    data: B::Matrix<T>,
}

// As for `Vector`: hand-written derives, because the storage type — and so which
// of these a tensor gets — depends on the backend.
impl<T, B: Backend> Clone for Matrix<T, B>
where
    B::Matrix<T>: Clone,
{
    fn clone(&self) -> Self {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.clone(),
        }
    }
}

impl<T, B: Backend> PartialEq for Matrix<T, B>
where
    B::Matrix<T>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.shape() == other.shape() && self.data == other.data
    }
}

impl<T, B: Backend> Eq for Matrix<T, B> where B::Matrix<T>: Eq {}

impl<T, B: Backend> fmt::Debug for Matrix<T, B>
where
    B::Matrix<T>: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Matrix")
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("data", &self.data)
            .finish()
    }
}

/// The shape queries, as for [`Vector`].
impl<T, B: Backend> Matrix<T, B> {
    /// The `(rows, columns)` extents.
    pub const fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// The number of rows.
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// The number of columns.
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Whether this matrix holds no elements.
    pub const fn is_empty(&self) -> bool {
        self.rows == 0 || self.cols == 0
    }

    /// Whether this matrix is square, which the determinant and inverse need.
    pub const fn is_square(&self) -> bool {
        self.rows == self.cols
    }
}

/// Copyable matrix storage operations on any backend, as for [`Vector`] above.
impl<T: Copy + 'static, B: Backend> Matrix<T, B> {
    /// Move this matrix's elements onto backend `B2`.
    ///
    /// The counterpart to [`Vector::to_backend`], and the only place a
    /// `Metal`-backed chain copies between CPU and GPU memory.
    pub fn to_backend<B2: Backend>(&self) -> Matrix<T, B2> {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: B2::store_matrix(B::matrix_slice(&self.data)),
        }
    }

    /// A `rows × cols` matrix with every element set to `value`, allocated
    /// directly on backend `B`.
    pub fn filled(rows: usize, cols: usize, value: T) -> Self {
        Matrix {
            rows,
            cols,
            data: B::store_matrix(&vec![value; rows * cols]),
        }
    }

    /// Borrow the elements as one flat row-major slice, without copying.
    pub fn as_slice(&self) -> &[T] {
        B::matrix_slice(&self.data)
    }

    /// Build from typed row-major values on backend `B`.
    pub(crate) fn build(rows: usize, cols: usize, values: &[T]) -> Self {
        debug_assert_eq!(values.len(), rows * cols);
        Matrix {
            rows,
            cols,
            data: B::store_matrix(values),
        }
    }

    /// Attach a shape to storage that already holds exactly `rows * cols`
    /// values — how a tensor is rebuilt from the result of a kernel dispatch.
    pub(crate) fn from_storage(rows: usize, cols: usize, data: B::Matrix<T>) -> Self {
        Matrix { rows, cols, data }
    }

    /// The backend storage itself, which the kernels hand straight to a
    /// dispatch.
    pub(crate) fn storage(&self) -> &B::Matrix<T> {
        &self.data
    }

    /// The backend storage itself, for a dispatch that accumulates in place.
    pub(crate) fn storage_mut(&mut self) -> &mut B::Matrix<T> {
        &mut self.data
    }

    /// Consume this matrix and take its storage, which is how a reshape moves
    /// the elements instead of copying them.
    pub(crate) fn into_storage(self) -> B::Matrix<T> {
        self.data
    }
}

impl<T> Matrix<T, Host> {
    /// A matrix from its rows.
    ///
    /// Accepts anything that iterates over row-like values, so a literal and a
    /// runtime-built `Vec<Vec<T>>` are written the same way:
    ///
    /// ```
    /// use tensorcrate::tensors::Matrix;
    ///
    /// let literal = Matrix::from_rows([[1.0, 2.0], [3.0, 4.0]]);
    /// let computed = Matrix::from_rows((0..2).map(|r| vec![2.0 * r as f64 + 1.0, 2.0 * r as f64 + 2.0]));
    /// assert_eq!(literal, computed);
    /// ```
    ///
    /// # Panics
    ///
    /// If the rows are not all the same length.
    #[track_caller]
    pub fn from_rows<I, R>(rows: I) -> Self
    where
        I: IntoIterator<Item = R>,
        R: Into<Vec<T>>,
    {
        let mut data = Vec::new();
        let mut count = 0;
        let mut cols = 0;
        for row in rows {
            let mut row = row.into();
            if count == 0 {
                cols = row.len();
            } else {
                assert!(
                    row.len() == cols,
                    "from_rows: row {count} has {} elements, but row 0 has {cols}",
                    row.len()
                );
            }
            data.append(&mut row);
            count += 1;
        }
        Matrix {
            rows: count,
            cols,
            data,
        }
    }

    /// A matrix from its row-major elements and an explicit shape.
    ///
    /// # Panics
    ///
    /// If the element count is not `rows * cols`.
    #[track_caller]
    pub fn from_flat(rows: usize, cols: usize, data: impl Into<Vec<T>>) -> Self {
        let data = data.into();
        assert!(
            data.len() == rows * cols,
            "from_flat: {} elements cannot fill a {rows}×{cols} matrix",
            data.len()
        );
        Matrix { rows, cols, data }
    }

    /// Borrow the elements as one flat row-major slice.
    ///
    /// Row `i`, column `j` is at `i * cols + j`; [`row`](Self::row) and the
    /// `(row, column)` index do that arithmetic for you.
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Borrow the elements as one flat row-major mutable slice.
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Consume this matrix and take its row-major elements.
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    /// Borrow one row.
    #[track_caller]
    pub fn row(&self, row: usize) -> &[T] {
        assert!(
            row < self.rows,
            "row {row} is out of range for {} rows",
            self.rows
        );
        &self.data[row * self.cols..(row + 1) * self.cols]
    }

    /// Iterate over the rows.
    pub fn row_iter(&self) -> impl Iterator<Item = &[T]> {
        (0..self.rows).map(move |row| self.row(row))
    }

    /// Copy the elements into a vector of rows.
    ///
    /// Host-only, since it hands back owned elements; a resident tensor reaches
    /// it through [`to_backend::<Host>()`](Self::to_backend).
    pub fn to_rows(&self) -> Vec<Vec<T>>
    where
        T: Clone,
    {
        self.row_iter().map(<[T]>::to_vec).collect()
    }

    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row >= self.rows || col >= self.cols {
            return None;
        }
        self.data.get(row * self.cols + col)
    }

    /// Apply `f` to every element, producing a matrix of the new element type.
    pub fn map<U>(&self, f: impl Fn(&T) -> U) -> Matrix<U, Host> {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(f).collect(),
        }
    }

    /// Combine two matrices element by element with `f`, producing a matrix of
    /// the new element type.
    ///
    /// # Panics
    ///
    /// If the two shapes differ.
    #[track_caller]
    pub fn zip_map<U, V>(
        &self,
        other: &Matrix<U, Host>,
        f: impl Fn(&T, &U) -> V,
    ) -> Matrix<V, Host> {
        assert_same_shape(self.shape(), other.shape(), "zip_map");
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| f(a, b))
                .collect(),
        }
    }
}

impl<T> Index<(usize, usize)> for Matrix<T, Host> {
    type Output = T;

    #[track_caller]
    fn index(&self, (row, col): (usize, usize)) -> &T {
        assert!(
            row < self.rows && col < self.cols,
            "index ({row}, {col}) is out of range for a {}×{} matrix",
            self.rows,
            self.cols
        );
        &self.data[row * self.cols + col]
    }
}

impl<T: Coefficient> Matrix<T, Host> {
    /// A `rows × cols` matrix of zeros.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![T::zero(); rows * cols],
        }
    }

    /// A `rows × cols` matrix with every element set to `value`.
    pub fn repeat(rows: usize, cols: usize, value: T) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![value; rows * cols],
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
            let mut out = vec![T::zero(); self.data.len()];
            if simd_dispatch::broadcast(&self.data, scalar, op, false, &mut out) {
                return Matrix::from_flat(self.rows, self.cols, out);
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
            let mut out = vec![T::zero(); self.data.len()];
            if simd_dispatch::broadcast(&self.data, scalar, op, true, &mut out) {
                return Matrix::from_flat(self.rows, self.cols, out);
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

    /// Matrix product `(R×K)·(K×C) = (R×C)`.
    ///
    /// # Panics
    ///
    /// If this matrix's column count is not the other's row count.
    #[track_caller]
    pub fn matmul(&self, other: &Matrix<T, Host>) -> Matrix<T, Host> {
        assert_inner(self.shape(), other.shape(), "matmul");
        let (rows, inner, cols) = (self.rows, self.cols, other.cols);
        let mut out = vec![T::zero(); rows * cols];

        // `f16` and `bf16` widen once, multiply in `f32`, and round once.
        if crate::compact::matmul(&self.data, &other.data, rows, inner, cols, &mut out, false) {
            return Matrix::from_flat(rows, cols, out);
        }

        #[cfg(target_os = "macos")]
        if accelerate_dispatch::matmul(&self.data, &other.data, rows, inner, cols, &mut out, false)
        {
            return Matrix::from_flat(rows, cols, out);
        }

        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        if simd_dispatch::matmul(&self.data, &other.data, rows, inner, cols, &mut out) {
            return Matrix::from_flat(rows, cols, out);
        }

        for i in 0..rows {
            for j in 0..cols {
                let mut sum = T::zero();
                for p in 0..inner {
                    sum = sum + self.data[i * inner + p] * other.data[p * cols + j];
                }
                out[i * cols + j] = sum;
            }
        }
        Matrix::from_flat(rows, cols, out)
    }

    /// Fused matrix multiply-add: `self·other + addend`.
    ///
    /// `addend` is consumed and used as the accumulator, avoiding a separate
    /// product allocation and elementwise addition.
    ///
    /// # Panics
    ///
    /// If the inner dimensions disagree, or `addend` is not `rows × other.cols`.
    #[track_caller]
    pub fn matmul_add(&self, other: &Matrix<T, Host>, addend: Matrix<T, Host>) -> Matrix<T, Host> {
        assert_inner(self.shape(), other.shape(), "matmul_add");
        assert_same_shape(addend.shape(), (self.rows, other.cols), "matmul_add addend");
        let (rows, inner, cols) = (self.rows, self.cols, other.cols);
        let mut output = addend;

        if crate::compact::matmul(
            &self.data,
            &other.data,
            rows,
            inner,
            cols,
            &mut output.data,
            true,
        ) {
            return output;
        }

        #[cfg(target_os = "macos")]
        if accelerate_dispatch::matmul(
            &self.data,
            &other.data,
            rows,
            inner,
            cols,
            &mut output.data,
            true,
        ) {
            return output;
        }

        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        if simd_dispatch::matmul_add(&self.data, &other.data, rows, inner, cols, &mut output.data) {
            return output;
        }

        for i in 0..rows {
            for p in 0..inner {
                for j in 0..cols {
                    output.data[i * cols + j] = output.data[i * cols + j]
                        + self.data[i * inner + p] * other.data[p * cols + j];
                }
            }
        }
        output
    }

    /// Matrix times column vector: `(R×C)·(C×1) = (R×1)`.
    ///
    /// # Panics
    ///
    /// If the vector's length is not this matrix's column count.
    #[track_caller]
    pub fn matvec(&self, v: &Vector<T, Host>) -> Vector<T, Host> {
        assert_inner(self.shape(), (v.len(), 1), "matvec");
        let (rows, cols) = (self.rows, self.cols);
        let vector = v.as_slice();
        let mut out = vec![T::zero(); rows];

        if crate::compact::matmul(&self.data, vector, rows, cols, 1, &mut out, false) {
            return Vector::new(out);
        }

        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        if simd_dispatch::matvec(&self.data, vector, rows, cols, &mut out) {
            return Vector::new(out);
        }

        for (i, slot) in out.iter_mut().enumerate() {
            let row = &self.data[i * cols..(i + 1) * cols];
            let mut sum = T::zero();
            for (&a, &b) in row.iter().zip(vector) {
                sum = sum + a * b;
            }
            *slot = sum;
        }
        Vector::new(out)
    }

    /// Fused matrix-vector multiply-add: `self·v + addend`.
    ///
    /// Each row uses the SIMD dot-product kernel when available, and the owned
    /// addend is updated in place.
    ///
    /// # Panics
    ///
    /// If the vector's length is not this matrix's column count, or the addend's
    /// length is not its row count.
    #[track_caller]
    pub fn matvec_add(&self, v: &Vector<T, Host>, addend: Vector<T, Host>) -> Vector<T, Host> {
        assert_inner(self.shape(), (v.len(), 1), "matvec_add");
        assert_same_len(addend.len(), self.rows, "matvec_add addend");
        let (rows, cols) = (self.rows, self.cols);
        let vector = v.as_slice();
        let mut output = addend;
        let out = output.storage_mut();

        if crate::compact::matmul(&self.data, vector, rows, cols, 1, out, true) {
            return output;
        }

        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        if simd_dispatch::matvec_add(&self.data, vector, rows, cols, out) {
            return output;
        }

        for (i, slot) in out.iter_mut().enumerate() {
            let row = &self.data[i * cols..(i + 1) * cols];
            let mut sum = T::zero();
            for (&a, &b) in row.iter().zip(vector) {
                sum = sum + a * b;
            }
            *slot = *slot + sum;
        }
        output
    }

    /// Transpose: an `R×C` matrix becomes `C×R`.
    pub fn transpose(&self) -> Matrix<T, Host> {
        let (rows, cols) = (self.rows, self.cols);
        let mut out = vec![T::zero(); rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                out[j * rows + i] = self.data[i * cols + j];
            }
        }
        Matrix::from_flat(cols, rows, out)
    }

    /// The `n × n` identity matrix.
    pub fn identity(n: usize) -> Self {
        let mut data = vec![T::zero(); n * n];
        for i in 0..n {
            data[i * n + i] = T::one();
        }
        Matrix::from_flat(n, n, data)
    }

    /// Determinant, by fraction-free (Bareiss) elimination — every division is
    /// exact, so an integer matrix keeps an exact integer determinant.
    ///
    /// # Panics
    ///
    /// If the matrix is not square.
    #[track_caller]
    pub fn determinant(&self) -> T {
        assert!(
            self.is_square(),
            "determinant: matrix is {}×{}, not square",
            self.rows,
            self.cols
        );
        let n = self.rows;
        if n == 0 {
            return T::one();
        }
        let mut m = self.data.clone();
        let at = |i: usize, j: usize| i * n + j;
        let mut prev = T::one();
        let mut negate = false;

        for k in 0..n - 1 {
            if m[at(k, k)].is_zero() {
                match (k + 1..n).find(|&p| !m[at(p, k)].is_zero()) {
                    Some(p) => {
                        for j in 0..n {
                            m.swap(at(k, j), at(p, j));
                        }
                        negate = !negate;
                    }
                    None => return T::zero(),
                }
            }
            for i in k + 1..n {
                for j in k + 1..n {
                    let value = m[at(i, j)] * m[at(k, k)] - m[at(i, k)] * m[at(k, j)];
                    m[at(i, j)] = value / prev;
                }
            }
            prev = m[at(k, k)];
        }

        let det = m[at(n - 1, n - 1)];
        if negate { T::zero() - det } else { det }
    }

    /// Inverse, by Gauss–Jordan elimination with partial pivoting on
    /// [`Coefficient::magnitude`] (so complex and dual elements, which have no
    /// ordering, still pivot sensibly). Returns [`Error::Singular`] when the
    /// matrix has no inverse, and [`Error::Shape`] when it is not square.
    /// Coefficient domains with truncating division, such as primitive integers
    /// and integer-based complex or dual numbers, return
    /// [`Error::InvalidArgument`] because Gauss–Jordan requires fractions.
    pub fn inverse(&self) -> Result<Self, Error> {
        if !self.is_square() {
            return Err(Error::shape(format!(
                "matrix is {}×{}, so it has no inverse",
                self.rows, self.cols
            )));
        }
        if !T::supports_fractional_division() {
            return Err(Error::InvalidArgument(
                "matrix inversion requires coefficients with fractional division".to_string(),
            ));
        }
        let n = self.rows;
        let at = |i: usize, j: usize| i * n + j;
        let mut a = self.data.clone();
        let mut inv = Self::identity(n).data;

        for col in 0..n {
            let pivot = (col..n)
                .max_by(|&x, &y| {
                    a[at(x, col)]
                        .magnitude()
                        .partial_cmp(&a[at(y, col)].magnitude())
                        .unwrap_or(Ordering::Equal)
                })
                .expect("col < n, so the range is non-empty");
            if a[at(pivot, col)].is_zero() {
                return Err(Error::Singular);
            }
            if pivot != col {
                for j in 0..n {
                    a.swap(at(col, j), at(pivot, j));
                    inv.swap(at(col, j), at(pivot, j));
                }
            }

            let scale = a[at(col, col)];
            for j in 0..n {
                a[at(col, j)] = a[at(col, j)] / scale;
                inv[at(col, j)] = inv[at(col, j)] / scale;
            }
            for r in 0..n {
                if r == col {
                    continue;
                }
                let factor = a[at(r, col)];
                if factor.is_zero() {
                    continue;
                }
                for j in 0..n {
                    a[at(r, j)] = a[at(r, j)] - factor * a[at(col, j)];
                    inv[at(r, j)] = inv[at(r, j)] - factor * inv[at(col, j)];
                }
            }
        }
        Ok(Matrix::from_flat(n, n, inv))
    }
}

impl<T: Coefficient + PartialOrd> Matrix<T, Host> {
    /// Elementwise minimum with another matrix.
    ///
    /// # Panics
    ///
    /// If the two shapes differ.
    #[track_caller]
    pub fn min(&self, other: &Self) -> Self {
        assert_same_shape(self.shape(), other.shape(), "min");
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let mut out = vec![T::zero(); self.data.len()];
            if simd_dispatch::compare(&self.data, &other.data, Compare::Min, &mut out) {
                return Matrix::from_flat(self.rows, self.cols, out);
            }
        }
        Matrix::from_flat(
            self.rows,
            self.cols,
            self.data
                .iter()
                .zip(&other.data)
                .map(|(&left, &right)| ordered_min(left, right))
                .collect::<Vec<_>>(),
        )
    }

    /// Elementwise maximum with another matrix.
    ///
    /// # Panics
    ///
    /// If the two shapes differ.
    #[track_caller]
    pub fn max(&self, other: &Self) -> Self {
        assert_same_shape(self.shape(), other.shape(), "max");
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let mut out = vec![T::zero(); self.data.len()];
            if simd_dispatch::compare(&self.data, &other.data, Compare::Max, &mut out) {
                return Matrix::from_flat(self.rows, self.cols, out);
            }
        }
        Matrix::from_flat(
            self.rows,
            self.cols,
            self.data
                .iter()
                .zip(&other.data)
                .map(|(&left, &right)| ordered_max(left, right))
                .collect::<Vec<_>>(),
        )
    }

    /// The lesser of each element and `scalar`.
    pub fn min_scalar(&self, scalar: T) -> Self {
        self.against_scalar(scalar, Compare::Min, ordered_min)
    }

    /// The greater of each element and `scalar`.
    pub fn max_scalar(&self, scalar: T) -> Self {
        self.against_scalar(scalar, Compare::Max, ordered_max)
    }

    /// Confine every element to `[low, high]`; see
    /// [`Vector::clamp`](Vector::clamp).
    ///
    /// # Panics
    ///
    /// If `low > high`.
    #[track_caller]
    pub fn clamp(&self, low: T, high: T) -> Self {
        assert_ordered_bounds(&low, &high);
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let mut out = vec![T::zero(); self.data.len()];
            if simd_dispatch::clamp(&self.data, low, high, &mut out) {
                return Matrix::from_flat(self.rows, self.cols, out);
            }
        }
        self.map(|&value| ordered_min(ordered_max(value, low), high))
    }

    fn against_scalar(&self, value: T, op: Compare, scalar: fn(T, T) -> T) -> Self {
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let mut out = vec![T::zero(); self.data.len()];
            if simd_dispatch::compare_scalar(&self.data, value, op, false, &mut out) {
                return Matrix::from_flat(self.rows, self.cols, out);
            }
        }
        let _ = op;
        self.map(|&element| scalar(element, value))
    }
}

impl<T: Real> Matrix<T, Host> {
    /// Elementwise comparison with another matrix.
    ///
    /// # Panics
    ///
    /// If the two shapes differ.
    #[track_caller]
    pub fn compare(&self, other: &Self, op: Compare) -> Self {
        assert_same_shape(self.shape(), other.shape(), "compare");
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let mut out = vec![T::zero(); self.data.len()];
            if simd_dispatch::compare(&self.data, &other.data, op, &mut out) {
                return Matrix::from_flat(self.rows, self.cols, out);
            }
        }
        let values = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(&left, &right)| op.value(left, right))
            .collect::<Vec<_>>();
        Matrix::from_flat(self.rows, self.cols, values)
    }

    /// Elementwise comparison against a scalar.
    pub fn compare_scalar(&self, scalar: T, op: Compare, scalar_left: bool) -> Self {
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let mut out = vec![T::zero(); self.data.len()];
            if simd_dispatch::compare_scalar(&self.data, scalar, op, scalar_left, &mut out) {
                return Matrix::from_flat(self.rows, self.cols, out);
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
}

impl<T: Display> Display for Matrix<T, Host> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (r, row) in self.row_iter().enumerate() {
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

impl<T: Coefficient> Matrix<T, Host> {
    #[track_caller]
    pub(super) fn zip_with(&self, rhs: &Self, op: BinaryOp, f: impl Fn(T, T) -> T) -> Self {
        assert_same_shape(self.shape(), rhs.shape(), op.name());

        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let mut out = vec![T::zero(); self.data.len()];
            if simd_dispatch::elementwise(&self.data, &rhs.data, op, &mut out) {
                return Matrix::from_flat(self.rows, self.cols, out);
            }
        }

        Matrix::from_flat(
            self.rows,
            self.cols,
            self.data
                .iter()
                .zip(&rhs.data)
                .map(|(&a, &b)| f(a, b))
                .collect::<Vec<_>>(),
        )
    }
}
