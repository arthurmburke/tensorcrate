//! [`SparseMatrix`]: a two-deimensional sparse matrix type that can be used to represent matrices with a large number of zero elements.

use std::{fmt, ops::{Index, IndexMut}};

use super::Matrix;
use crate::{numbers::Coefficient, tensors::{Backend, Host}};

/// A matrix whose shape is fixed when it is built, backed
/// by host tensors.
pub struct SparseMatrix<T> {
    /// The number of rows in the matrix.
    pub nrows: usize,
    /// The number of columns in the matrix.
    pub ncols: usize,
    /// The non-zero values of the matrix, stored in a flat vector.
    pub values: Vec<T>,
    /// The row indices corresponding to each non-zero value.
    pub row_indices: Vec<usize>,
    /// The column indices corresponding to each non-zero value.
    pub col_indices: Vec<usize>,
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

    /// Whether this matrix contains no non-zero values.
    pub const fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The number of non-zero values.
    pub const fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Whether this matrix is square.
    pub const fn is_square(&self) -> bool {
        self.nrows == self.ncols
    }

}

impl<T> SparseMatrix<T>
where
    T: Coefficient + 'static,
{
    /// Converts the matrix to a dense backend.
    pub fn to_dense<B: Backend>(&self) -> Matrix<T, B>{
        let mut data = vec![T::zero(); self.nrows * self.ncols];
        for i in 0..self.nnz() {
            let row = self.row_indices[i];
            let col = self.col_indices[i];
            data[row * self.ncols + col] = self.values[i];
        }

        Matrix::<T, Host>::from_flat(self.nrows, self.ncols, data)
            .to_backend::<B>()
    }

    /// Converts a dense matrix to a sparse matrix.
    pub fn from_dense<B: Backend>(matrix: &Matrix<T, B>) -> Self {
        let mut values = Vec::new();
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();

        let nrows = matrix.rows();
        let ncols = matrix.cols();
        let data = matrix.as_slice();
        
        for row in 0..nrows {
            for col in 0..ncols {
                let value = data[row * ncols + col];
                if !value.is_zero() {
                    values.push(value);
                    row_indices.push(row);
                    col_indices.push(col);
                }
            }
        }

        Self {
            nrows: matrix.rows(),
            ncols: matrix.cols(),
            values,
            row_indices,
            col_indices,
        }

    }
}

impl<T: Coefficient + 'static> Index<(usize, usize)> for SparseMatrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> T {
        let (row, col) = index;
        for i in 0..self.nnz() {
            if self.row_indices[i] == row && self.col_indices[i] == col {
                return self.values[i];
            }
        }
        T::zero()
    }
}

impl<T: Coefficient + 'static> IndexMut<(usize, usize)> for SparseMatrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut T {
        let (row, col) = index;
        for i in 0..self.nnz() {
            if self.row_indices[i] == row && self.col_indices[i] == col {
                return &mut self.values[i];
            }
        }
        
        self.row_indices.push(row);
        self.col_indices.push(col);
        self.values.push(T::zero());
        let nnz = self.nnz();

        &mut self.values[nnz - 1]
    }
}

