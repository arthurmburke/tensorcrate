//! Compressed sparse rows with validated, immutable structure.

use super::operator::{Apply, LinearOperator, OperatorBackend};
use super::{Backend, Host, MatVec, Matrix, SparseMatrix, Transpose, Vector};
use crate::numbers::{Coefficient, Conj, Field};

/// Sparse rows with sorted unique columns. Values can change without rebuilding
/// the pattern. A transposed index view supports adjoint products in O(nnz),
/// storing positions rather than a second copy of the coefficients.
pub struct CsrMatrix<T, B: Backend = Host> {
    rows: usize,
    cols: usize,
    nnz: usize,
    pub(crate) values: B::Storage<T>,
    pub(crate) offsets: B::Storage<u32>,
    pub(crate) columns: B::Storage<u32>,
    pub(crate) adjoint_offsets: B::Storage<u32>,
    pub(crate) adjoint_columns: B::Storage<u32>,
    pub(crate) adjoint_positions: B::Storage<u32>,
}

impl<T, B: Backend> CsrMatrix<T, B> {
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn nnz(&self) -> usize {
        self.nnz
    }
}

impl<T: Coefficient> CsrMatrix<T> {
    /// Construct from canonical CSR arrays. Explicit zero values are allowed
    /// for patterns reused across frequency sweeps and repeated assembly.
    #[track_caller]
    pub fn from_parts(
        rows: usize,
        cols: usize,
        offsets: Vec<u32>,
        columns: Vec<u32>,
        values: Vec<T>,
    ) -> Self {
        assert!(
            rows <= u32::MAX as usize
                && cols <= u32::MAX as usize
                && values.len() <= u32::MAX as usize,
            "CSR exceeds 32-bit indexing"
        );
        assert_eq!(offsets.len(), rows + 1, "CSR needs rows + 1 offsets");
        assert_eq!(
            columns.len(),
            values.len(),
            "CSR columns and values differ in length"
        );
        assert_eq!(offsets[0], 0, "CSR first offset must be zero");
        assert_eq!(
            offsets[rows] as usize,
            values.len(),
            "CSR last offset must equal nnz"
        );
        for row in 0..rows {
            let (start, end) = (offsets[row] as usize, offsets[row + 1] as usize);
            assert!(
                start <= end && end <= values.len(),
                "CSR offsets must be monotone and in bounds"
            );
            let c = &columns[start..end];
            assert!(
                c.iter().all(|&c| (c as usize) < cols) && c.windows(2).all(|w| w[0] < w[1]),
                "CSR columns must be sorted, unique and in bounds"
            );
        }
        let mut adjoint_offsets = vec![0u32; cols + 1];
        for &col in &columns {
            adjoint_offsets[col as usize + 1] += 1;
        }
        for i in 0..cols {
            adjoint_offsets[i + 1] += adjoint_offsets[i];
        }
        let mut next = adjoint_offsets.clone();
        let mut adjoint_columns = vec![0; values.len()];
        let mut adjoint_positions = vec![0; values.len()];
        for row in 0..rows {
            let start = offsets[row] as usize;
            let end = offsets[row + 1] as usize;
            for (local, &col) in columns[start..end].iter().enumerate() {
                let k = start + local;
                let col = col as usize;
                let slot = next[col] as usize;
                adjoint_columns[slot] = row as u32;
                adjoint_positions[slot] = k as u32;
                next[col] += 1;
            }
        }
        Self {
            rows,
            cols,
            nnz: values.len(),
            offsets,
            columns,
            values,
            adjoint_offsets,
            adjoint_columns,
            adjoint_positions,
        }
    }

    pub fn from_triplets(
        rows: usize,
        cols: usize,
        entries: impl IntoIterator<Item = (usize, usize, T)>,
    ) -> Self {
        assert!(
            rows <= u32::MAX as usize && cols <= u32::MAX as usize,
            "CSR exceeds 32-bit indexing"
        );
        let coo = SparseMatrix::from_triplets(rows, cols, entries);
        assert!(
            coo.nnz() <= u32::MAX as usize,
            "CSR exceeds 32-bit indexing"
        );
        let mut offsets = vec![0; rows + 1];
        for &row in &coo.row_indices {
            offsets[row + 1] += 1;
        }
        for row in 0..rows {
            offsets[row + 1] += offsets[row];
        }
        Self::from_parts(
            rows,
            cols,
            offsets,
            coo.col_indices.into_iter().map(|c| c as u32).collect(),
            coo.values,
        )
    }

    pub fn values_mut(&mut self) -> &mut [T] {
        &mut self.values
    }
    pub fn to_coo(&self) -> SparseMatrix<T> {
        SparseMatrix::from_triplets(
            self.rows,
            self.cols,
            (0..self.rows).flat_map(|row| {
                (self.offsets[row] as usize..self.offsets[row + 1] as usize)
                    .map(move |k| (row, self.columns[k] as usize, self.values[k]))
            }),
        )
    }
    pub fn to_dense(&self) -> Matrix<T> {
        self.to_coo().to_dense::<Host>()
    }
    pub fn transpose(&self) -> Self {
        Self::from_parts(
            self.cols,
            self.rows,
            self.adjoint_offsets.clone(),
            self.adjoint_columns.clone(),
            self.adjoint_positions
                .iter()
                .map(|&k| self.values[k as usize])
                .collect(),
        )
    }
    pub fn adjoint(&self) -> Self
    where
        T: Conj<Output = T>,
    {
        let mut a = self.transpose();
        for v in &mut a.values {
            *v = v.conj();
        }
        a
    }
    pub fn diagonal(&self) -> Vec<T> {
        (0..self.rows.min(self.cols))
            .map(|row| {
                let start = self.offsets[row] as usize;
                let columns = &self.columns[start..self.offsets[row + 1] as usize];
                columns
                    .binary_search(&(row as u32))
                    .map_or(T::zero(), |k| self.values[start + k])
            })
            .collect()
    }
}

impl<T: Coefficient, B: Backend> CsrMatrix<T, B> {
    pub fn values(&self) -> &[T] {
        B::as_slice(&self.values)
    }
    pub fn row_offsets(&self) -> &[u32] {
        B::as_slice(&self.offsets)
    }
    pub fn column_indices(&self) -> &[u32] {
        B::as_slice(&self.columns)
    }
    pub fn to_backend<B2: Backend>(&self) -> CsrMatrix<T, B2> {
        use super::backend::transfer;
        CsrMatrix {
            rows: self.rows,
            cols: self.cols,
            nnz: self.nnz,
            values: transfer::<T, B, B2>(&self.values),
            offsets: transfer::<u32, B, B2>(&self.offsets),
            columns: transfer::<u32, B, B2>(&self.columns),
            adjoint_offsets: transfer::<u32, B, B2>(&self.adjoint_offsets),
            adjoint_columns: transfer::<u32, B, B2>(&self.adjoint_columns),
            adjoint_positions: transfer::<u32, B, B2>(&self.adjoint_positions),
        }
    }
}

impl<T: Field, B: OperatorBackend<T>> LinearOperator<T, B> for CsrMatrix<T, B> {
    fn shape(&self) -> (usize, usize) {
        self.shape()
    }
    fn apply_window(&self, x: &B::Storage<T>, y: &mut B::Storage<T>, p: Apply<T>) {
        B::csr(self, x, y, p);
    }
}

impl<'b, T: Field, B: OperatorBackend<T>> MatVec<&'b Vector<T, B>> for &CsrMatrix<T, B> {
    type Output = Vector<T, B>;
    fn matvec(self, x: &'b Vector<T, B>) -> Self::Output {
        let mut y = Vector::filled(self.rows, T::zero());
        self.apply(x, &mut y);
        y
    }
}

impl<T: Coefficient> Transpose for &CsrMatrix<T> {
    type Output = CsrMatrix<T>;
    fn transpose(self) -> CsrMatrix<T> {
        CsrMatrix::transpose(self)
    }
}

/// Append-only finite-element assembly. Duplicate coordinates are summed when
/// compressed; exact cancellations are omitted.
pub struct CooBuilder<T> {
    rows: usize,
    cols: usize,
    entries: Vec<(usize, usize, T)>,
}

impl<T: Coefficient> CooBuilder<T> {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            entries: Vec::new(),
        }
    }
    pub fn reserve(&mut self, additional: usize) {
        self.entries.reserve(additional);
    }
    pub fn push(&mut self, row: usize, col: usize, value: T) {
        assert!(
            row < self.rows && col < self.cols,
            "assembly entry out of bounds"
        );
        self.entries.push((row, col, value));
    }
    /// Add a row-major local matrix, including rectangular coupling blocks.
    pub fn add_local(&mut self, rows: &[usize], cols: &[usize], values: &[T]) {
        assert_eq!(
            rows.len().checked_mul(cols.len()),
            Some(values.len()),
            "local matrix shape mismatch"
        );
        assert!(
            rows.iter().all(|&r| r < self.rows) && cols.iter().all(|&c| c < self.cols),
            "local DOF out of bounds"
        );
        for (i, &row) in rows.iter().enumerate() {
            for (j, &col) in cols.iter().enumerate() {
                self.push(row, col, values[i * cols.len() + j]);
            }
        }
    }
    pub fn finish(self) -> CsrMatrix<T> {
        CsrMatrix::from_triplets(self.rows, self.cols, self.entries)
    }
}
