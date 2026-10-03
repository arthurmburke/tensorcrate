//! A coordinate-list sparse matrix.
//!
//! [`SparseMatrix`] stores one entry for every non-zero element. Entries are
//! kept in row-major order, which makes conversion to a dense matrix and
//! iteration deterministic while keeping the representation small for
//! matrices with many zeroes.

use std::{
    fmt,
    ops::{Index, IndexMut},
};

use super::Matrix;
use crate::{
    numbers::Coefficient,
    tensors::{Backend, Host},
};

/// A fixed-shape sparse matrix backed by host vectors.
///
/// The three public vectors are parallel: `values[i]` is at
/// `(row_indices[i], col_indices[i])`. Values are normally kept in row-major
/// order and zero values are omitted. Use [`Self::from_parts`] when building
/// a value from raw vectors so those invariants are checked.
pub struct SparseMatrix<T> {
    /// The number of rows in the matrix.
    pub nrows: usize,
    /// The number of columns in the matrix.
    pub ncols: usize,
    /// The non-zero values of the matrix, stored in row-major order.
    pub values: Vec<T>,
    /// The row indices corresponding to each non-zero value.
    pub row_indices: Vec<usize>,
    /// The column indices corresponding to each non-zero value.
    pub col_indices: Vec<usize>,
    // `Index` must return a reference even when an element is absent. This is
    // the per-matrix zero used for that read-only case.
    zero: T,
}

impl<T> Clone for SparseMatrix<T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Self {
            nrows: self.nrows,
            ncols: self.ncols,
            values: self.values.clone(),
            row_indices: self.row_indices.clone(),
            col_indices: self.col_indices.clone(),
            zero: self.zero.clone(),
        }
    }
}

impl<T> PartialEq for SparseMatrix<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.nrows == other.nrows
            && self.ncols == other.ncols
            && self.values == other.values
            && self.row_indices == other.row_indices
            && self.col_indices == other.col_indices
    }
}

impl<T> Eq for SparseMatrix<T> where T: Eq {}

impl<T: fmt::Debug> fmt::Debug for SparseMatrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SparseMatrix")
            .field("nrows", &self.nrows)
            .field("ncols", &self.ncols)
            .field("values", &self.values)
            .field("row_indices", &self.row_indices)
            .field("col_indices", &self.col_indices)
            .finish()
    }
}

impl<T> SparseMatrix<T> {
    /// The `(rows, columns)` extents.
    pub const fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// The number of rows.
    pub const fn rows(&self) -> usize {
        self.nrows
    }

    /// The number of columns.
    pub const fn cols(&self) -> usize {
        self.ncols
    }

    /// Whether this matrix has no stored non-zero values.
    pub const fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The number of stored non-zero values.
    pub const fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Whether this matrix is square.
    pub const fn is_square(&self) -> bool {
        self.nrows == self.ncols
    }

    /// The capacity of the sparse storage.
    pub fn capacity(&self) -> usize {
        self.values.capacity()
    }

    /// Reserve space for at least `additional` more entries.
    pub fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional);
        self.row_indices.reserve(additional);
        self.col_indices.reserve(additional);
    }

    /// Return a slice of the stored non-zero values.
    pub fn data(&self) -> &[T] {
        &self.values
    }
}

impl<T> SparseMatrix<T>
where
    T: Coefficient + 'static,
{
    /// A `rows × cols` matrix with no stored values.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            nrows: rows,
            ncols: cols,
            values: Vec::new(),
            row_indices: Vec::new(),
            col_indices: Vec::new(),
            zero: T::zero(),
        }
    }

    /// Alias for [`Self::zeros`].
    pub fn new(rows: usize, cols: usize) -> Self {
        Self::zeros(rows, cols)
    }

    /// Build a sparse matrix from parallel coordinate and value vectors.
    ///
    /// The vectors must have equal lengths, coordinates must be in bounds and
    /// must be strictly increasing in row-major order. Zero values and
    /// duplicate coordinates are rejected because they make the sparse
    /// representation ambiguous.
    #[track_caller]
    pub fn from_parts(
        nrows: usize,
        ncols: usize,
        values: Vec<T>,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
    ) -> Self {
        assert_eq!(
            values.len(),
            row_indices.len(),
            "from_parts: values and row_indices have different lengths"
        );
        assert_eq!(
            values.len(),
            col_indices.len(),
            "from_parts: values and col_indices have different lengths"
        );

        let mut previous = None;
        for i in 0..values.len() {
            let row = row_indices[i];
            let col = col_indices[i];
            assert!(
                row < nrows && col < ncols,
                "from_parts: entry ({row}, {col}) is out of bounds for a {nrows}×{ncols} matrix"
            );
            assert!(
                !values[i].is_zero(),
                "from_parts: entry ({row}, {col}) has a zero value"
            );
            let coordinate = (row, col);
            assert!(
                previous.is_none_or(|previous| previous < coordinate),
                "from_parts: coordinates must be strictly increasing in row-major order"
            );
            previous = Some(coordinate);
        }

        Self {
            nrows,
            ncols,
            values,
            row_indices,
            col_indices,
            zero: T::zero(),
        }
    }

    /// Return an iterator over the triplets `(row, column, value)` of the stored entries.
    pub fn triplets(&self) -> impl Iterator<Item = (usize, usize, &T)> {
        self.row_indices
            .iter()
            .zip(&self.col_indices)
            .zip(&self.values)
            .map(|((&row, &col), value)| (row, col, value))
    }

    /// Build a sparse matrix from `(row, column, value)` entries.
    ///
    /// Entries may arrive in any order. Duplicate coordinates are summed; if
    /// their sum is zero, the coordinate is omitted.
    pub fn from_triplets(
        rows: usize,
        cols: usize,
        entries: impl IntoIterator<Item = (usize, usize, T)>,
    ) -> Self {
        let mut entries = entries.into_iter().collect::<Vec<_>>();
        for &(row, col, _) in &entries {
            assert!(
                row < rows && col < cols,
                "from_triplets: entry ({row}, {col}) is out of bounds for a {rows}×{cols} matrix"
            );
        }
        entries.sort_unstable_by_key(|&(row, col, _)| (row, col));

        let mut values: Vec<T> = Vec::with_capacity(entries.len());
        let mut row_indices = Vec::with_capacity(entries.len());
        let mut col_indices = Vec::with_capacity(entries.len());
        for (row, col, value) in entries {
            if let (Some(last_row), Some(last_col)) = (row_indices.last(), col_indices.last())
                && *last_row == row
                && *last_col == col
            {
                let last = values.last_mut().expect("parallel sparse vectors");
                *last = *last + value;
                if last.is_zero() {
                    values.pop();
                    row_indices.pop();
                    col_indices.pop();
                }
            } else if !value.is_zero() {
                values.push(value);
                row_indices.push(row);
                col_indices.push(col);
            }
        }

        Self::from_parts(rows, cols, values, row_indices, col_indices)
    }

    /// Insert or replace an element. Assigning zero removes the entry.
    #[track_caller]
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        self.assert_in_bounds(row, col);
        match self.position(row, col) {
            Ok(index) if value.is_zero() => {
                self.values.remove(index);
                self.row_indices.remove(index);
                self.col_indices.remove(index);
            }
            Ok(index) => self.values[index] = value,
            Err(index) if !value.is_zero() => {
                self.values.insert(index, value);
                self.row_indices.insert(index, row);
                self.col_indices.insert(index, col);
            }
            Err(_) => {}
        }
    }

    /// Remove an element and return its value, or `None` if it was zero.
    #[track_caller]
    pub fn remove(&mut self, row: usize, col: usize) -> Option<T> {
        self.assert_in_bounds(row, col);
        let index = self.position(row, col).ok().unwrap_or(0);
        if index >= self.nnz() || self.row_indices[index] != row || self.col_indices[index] != col {
            return None;
        }
        self.row_indices.remove(index);
        self.col_indices.remove(index);
        Some(self.values.remove(index))
    }

    /// Get an element by coordinate. An in-bounds zero is returned as
    /// `Some(&0)`, matching dense matrix indexing; out-of-bounds coordinates
    /// return `None`.
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row >= self.nrows || col >= self.ncols {
            return None;
        }
        Some(self.index((row, col)))
    }

    /// Return an element by coordinate, panicking if it is out of bounds.
    #[track_caller]
    pub fn value(&self, row: usize, col: usize) -> T {
        self.assert_in_bounds(row, col);
        self[(row, col)]
    }

    /// Iterate over stored entries in row-major order.
    pub fn iter(&self) -> impl Iterator<Item = ((usize, usize), &T)> {
        self.row_indices
            .iter()
            .zip(&self.col_indices)
            .zip(&self.values)
            .map(|((&row, &col), value)| ((row, col), value))
    }

    /// Iterate mutably over stored values in row-major order.
    ///
    /// Mutating a value to zero leaves an explicit zero in the public parallel
    /// vectors. Use [`Self::set`] when zero-elision is required.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = ((usize, usize), &mut T)> {
        self.row_indices
            .iter()
            .zip(&self.col_indices)
            .zip(&mut self.values)
            .map(|((&row, &col), value)| ((row, col), value))
    }

    /// Remove every stored entry.
    pub fn clear(&mut self) {
        self.values.clear();
        self.row_indices.clear();
        self.col_indices.clear();
    }

    /// Return the transpose of this matrix.
    pub fn transpose(&self) -> Self
    where
        T: Clone,
    {
        let entries = self
            .iter()
            .map(|((row, col), value)| (col, row, value.clone()));
        Self::from_triplets(self.ncols, self.nrows, entries)
    }

    /// Multiply this matrix by a dense column vector.
    pub fn matvec(&self, vector: &[T]) -> Vec<T> {
        assert_eq!(
            vector.len(),
            self.ncols,
            "matvec: vector has length {}, but matrix has {} columns",
            vector.len(),
            self.ncols
        );
        let mut result = vec![T::zero(); self.nrows];
        for ((row, col), value) in self.iter() {
            result[row] = result[row] + *value * vector[col];
        }
        result
    }

    /// Apply `f` to the stored values, preserving the sparse coordinates.
    /// Values mapped to zero are omitted from the result.
    pub fn map<U>(&self, f: impl Fn(&T) -> U) -> SparseMatrix<U>
    where
        U: Coefficient + 'static,
    {
        SparseMatrix::from_triplets(
            self.nrows,
            self.ncols,
            self.iter().map(|((row, col), value)| (row, col, f(value))),
        )
    }

    /// Converts the matrix to a dense backend.
    pub fn to_dense<B: Backend>(&self) -> Matrix<T, B> {
        let mut data = vec![T::zero(); self.nrows * self.ncols];
        for ((row, col), value) in self.iter() {
            data[row * self.ncols + col] = *value;
        }

        Matrix::<T, Host>::from_flat(self.nrows, self.ncols, data).to_backend::<B>()
    }

    /// Converts a dense matrix to a sparse matrix.
    pub fn from_dense<B: Backend>(matrix: &Matrix<T, B>) -> Self {
        let nrows = matrix.rows();
        let ncols = matrix.cols();
        let data = matrix.as_slice();
        let entries = (0..nrows).flat_map(|row| {
            (0..ncols).filter_map(move |col| {
                let value = data[row * ncols + col];
                (!value.is_zero()).then_some((row, col, value))
            })
        });
        Self::from_triplets(nrows, ncols, entries)
    }

    fn assert_in_bounds(&self, row: usize, col: usize) {
        assert!(
            row < self.nrows && col < self.ncols,
            "index ({row}, {col}) is out of range for a {}×{} matrix",
            self.nrows,
            self.ncols
        );
    }

    fn position(&self, row: usize, col: usize) -> Result<usize, usize> {
        self.row_indices
            .iter()
            .zip(&self.col_indices)
            .enumerate()
            .find_map(|(index, (&stored_row, &stored_col))| {
                let coordinate = (stored_row, stored_col);
                match coordinate.cmp(&(row, col)) {
                    std::cmp::Ordering::Less => None,
                    std::cmp::Ordering::Equal => Some(Ok(index)),
                    std::cmp::Ordering::Greater => Some(Err(index)),
                }
            })
            .unwrap_or(Err(self.nnz()))
    }
}

impl<T: Coefficient + 'static> Index<(usize, usize)> for SparseMatrix<T> {
    type Output = T;

    #[track_caller]
    fn index(&self, (row, col): (usize, usize)) -> &T {
        self.assert_in_bounds(row, col);
        match self.position(row, col) {
            Ok(index) => &self.values[index],
            Err(_) => &self.zero,
        }
    }
}

impl<T: Coefficient + 'static> IndexMut<(usize, usize)> for SparseMatrix<T> {
    #[track_caller]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut T {
        self.assert_in_bounds(row, col);
        let index = match self.position(row, col) {
            Ok(index) => index,
            Err(index) => {
                self.row_indices.insert(index, row);
                self.col_indices.insert(index, col);
                self.values.insert(index, T::zero());
                index
            }
        };
        &mut self.values[index]
    }
}
