//! Triangular inverse operators with precomputed dependency levels.

use super::operator::Apply;
use super::{Backend, CsrMatrix, Host, LinearOperator, OperatorBackend};
use crate::{errors::Error, numbers::Field};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Triangle {
    Lower,
    Upper,
}

/// Applies the inverse of a triangular CSR matrix. Setup validates the shape,
/// coefficients and pivots. Independent rows share one GPU dispatch per level.
/// The structure is immutable, so dependency schedules stay valid.
pub struct TriangularOperator<T, B: Backend = Host> {
    pub(crate) matrix: CsrMatrix<T, B>,
    pub(crate) unit_diagonal: bool,
    pub(crate) levels: Vec<u32>,
    pub(crate) rows: B::Storage<u32>,
    pub(crate) adjoint_levels: Vec<u32>,
    pub(crate) adjoint_rows: B::Storage<u32>,
}

fn schedule(offsets: &[u32], columns: &[u32], lower: bool) -> (Vec<u32>, Vec<u32>) {
    let n = offsets.len() - 1;
    let mut levels = vec![0usize; n];
    for i in 0..n {
        let row = if lower { i } else { n - 1 - i };
        for &col in &columns[offsets[row] as usize..offsets[row + 1] as usize] {
            if col as usize != row {
                levels[row] = levels[row].max(levels[col as usize] + 1);
            }
        }
    }
    let mut grouped = vec![Vec::new(); levels.iter().max().map_or(0, |v| v + 1)];
    for (row, level) in levels.into_iter().enumerate() {
        grouped[level].push(row as u32);
    }
    let mut offsets = vec![0];
    let mut rows = Vec::with_capacity(n);
    for group in grouped {
        rows.extend(group);
        offsets.push(rows.len() as u32);
    }
    (offsets, rows)
}

impl<T: Field> TriangularOperator<T> {
    pub fn new(
        matrix: CsrMatrix<T>,
        triangle: Triangle,
        unit_diagonal: bool,
    ) -> Result<Self, Error> {
        if matrix.rows() != matrix.cols() {
            return Err(Error::shape("triangular solve requires a square matrix"));
        }
        for row in 0..matrix.rows() {
            let mut diagonal = unit_diagonal;
            for k in matrix.offsets[row] as usize..matrix.offsets[row + 1] as usize {
                let col = matrix.columns[k] as usize;
                let value = matrix.values[k];
                if !value.magnitude().is_finite() {
                    return Err(Error::InvalidArgument(
                        "nonfinite triangular coefficient".into(),
                    ));
                }
                if (triangle == Triangle::Lower && col > row)
                    || (triangle == Triangle::Upper && col < row)
                {
                    return Err(Error::InvalidArgument(
                        "entry outside triangular part".into(),
                    ));
                }
                if col == row && !unit_diagonal {
                    diagonal = !value.is_zero();
                }
            }
            if !diagonal {
                return Err(Error::Singular);
            }
        }
        let lower = triangle == Triangle::Lower;
        let (levels, rows) = schedule(&matrix.offsets, &matrix.columns, lower);
        let (adjoint_levels, adjoint_rows) =
            schedule(&matrix.adjoint_offsets, &matrix.adjoint_columns, !lower);
        Ok(Self {
            matrix,
            unit_diagonal,
            levels,
            rows,
            adjoint_levels,
            adjoint_rows,
        })
    }
}

impl<T: Field, B: Backend> TriangularOperator<T, B> {
    pub fn level_count(&self) -> usize {
        self.levels.len() - 1
    }
    pub fn to_backend<B2: Backend>(&self) -> TriangularOperator<T, B2> {
        use super::backend::transfer;
        TriangularOperator {
            matrix: self.matrix.to_backend::<B2>(),
            unit_diagonal: self.unit_diagonal,
            levels: self.levels.clone(),
            rows: transfer::<u32, B, B2>(&self.rows),
            adjoint_levels: self.adjoint_levels.clone(),
            adjoint_rows: transfer::<u32, B, B2>(&self.adjoint_rows),
        }
    }
}

impl<T: Field, B: OperatorBackend<T>> LinearOperator<T, B> for TriangularOperator<T, B> {
    fn shape(&self) -> (usize, usize) {
        self.matrix.shape()
    }
    fn apply_window(&self, x: &B::Storage<T>, y: &mut B::Storage<T>, p: Apply<T>) {
        B::triangular(self, x, y, p);
    }
}

/// Inverse of `L U`, including the reverse order for an adjoint application.
pub struct TriangularFactors<T, B: Backend = Host> {
    lower: TriangularOperator<T, B>,
    upper: TriangularOperator<T, B>,
}

impl<T: Field> TriangularFactors<T> {
    pub fn new(lower: TriangularOperator<T>, upper: TriangularOperator<T>) -> Result<Self, Error> {
        if lower.matrix.shape() != upper.matrix.shape() {
            return Err(Error::shape("triangular factors have different shapes"));
        }
        Ok(Self { lower, upper })
    }
}
impl<T: Field, B: Backend> TriangularFactors<T, B> {
    pub fn to_backend<B2: Backend>(&self) -> TriangularFactors<T, B2> {
        TriangularFactors {
            lower: self.lower.to_backend(),
            upper: self.upper.to_backend(),
        }
    }
}
impl<T: Field, B: OperatorBackend<T>> LinearOperator<T, B> for TriangularFactors<T, B> {
    fn shape(&self) -> (usize, usize) {
        self.lower.matrix.shape()
    }
    fn apply_window(&self, x: &B::Storage<T>, y: &mut B::Storage<T>, p: Apply<T>) {
        let mut intermediate = B::from_vec(vec![T::zero(); self.shape().0]);
        let (first, second) = if p.adjoint {
            (&self.upper, &self.lower)
        } else {
            (&self.lower, &self.upper)
        };
        first.apply_window(
            x,
            &mut intermediate,
            Apply {
                output: 0,
                alpha: T::one(),
                beta: T::zero(),
                ..p
            },
        );
        second.apply_window(&intermediate, y, Apply { input: 0, ..p });
    }
}
