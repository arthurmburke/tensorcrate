//! Composable linear operators with explicit storage backends.
//!
//! Operators accumulate directly into an output window. Blocks share resident
//! vectors and do not download or materialize their operands during application.

use crate::errors::Error;
use crate::numbers::Field;

use super::{Backend, Host, Matrix, Vector};

/// Parameters for an operator applied to windows of two storage buffers.
#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct Apply<T> {
    pub input: usize,
    pub output: usize,
    pub alpha: T,
    pub beta: T,
    pub adjoint: bool,
}

/// Backend kernels used by dense, sparse, diagonal and element operators.
///
/// This is separate from the dense tensor kernel trait so complex operators do
/// not require ordered arithmetic or analytic functions. Metal supports `f32`
/// and `Complex<f32>`; Host also supports their `f64` counterparts.
pub trait OperatorBackend<T: Field>: Backend {
    /// Whether the backend benefits from packed, batched orthogonalization.
    #[doc(hidden)]
    const BATCHED_ARNOLDI: bool = false;
    #[doc(hidden)]
    fn dense(
        a: &Self::Storage<T>,
        shape: (usize, usize),
        x: &Self::Storage<T>,
        y: &mut Self::Storage<T>,
        p: Apply<T>,
    );
    #[doc(hidden)]
    fn csr(
        a: &super::CsrMatrix<T, Self>,
        x: &Self::Storage<T>,
        y: &mut Self::Storage<T>,
        p: Apply<T>,
    );
    #[doc(hidden)]
    fn diagonal(
        a: &Self::Storage<T>,
        n: usize,
        x: &Self::Storage<T>,
        y: &mut Self::Storage<T>,
        p: Apply<T>,
    );
    #[doc(hidden)]
    fn elements(
        a: &ElementOperator<T, Self>,
        x: &Self::Storage<T>,
        y: &mut Self::Storage<T>,
        p: Apply<T>,
    );
    #[doc(hidden)]
    fn scale(y: &mut Self::Storage<T>, offset: usize, len: usize, beta: T);
    #[doc(hidden)]
    fn axpby(x: &Self::Storage<T>, y: &mut Self::Storage<T>, len: usize, alpha: T, beta: T);
    #[doc(hidden)]
    fn inner(x: &Self::Storage<T>, y: &Self::Storage<T>, len: usize) -> T;
    #[doc(hidden)]
    fn norm(x: &Self::Storage<T>, len: usize) -> f64;
    /// Two passes of classical Gram-Schmidt. Basis vectors are contiguous rows.
    /// Returns accumulated projections, the original norm, and the final norm.
    #[doc(hidden)]
    fn orthogonalize(
        basis: &Self::Storage<T>,
        n: usize,
        count: usize,
        work: &mut Self::Storage<T>,
    ) -> (Vec<T>, f64, f64);
    #[doc(hidden)]
    fn triangular(
        a: &super::TriangularOperator<T, Self>,
        x: &Self::Storage<T>,
        y: &mut Self::Storage<T>,
        p: Apply<T>,
    );
}

/// An object-safe linear map, including rectangular and matrix-free maps.
pub trait LinearOperator<T: Field, B: Backend = Host> {
    fn shape(&self) -> (usize, usize);

    /// Implementors must write `alpha A x + beta y` to the output window.
    /// When `adjoint` is set the map is `A*`. The windows are in bounds and
    /// buffers do not alias. A zero beta must not read the old output.
    #[doc(hidden)]
    fn apply_window(&self, x: &B::Storage<T>, y: &mut B::Storage<T>, p: Apply<T>);

    fn apply(&self, x: &Vector<T, B>, y: &mut Vector<T, B>) {
        self.apply_add(T::one(), x, T::zero(), y);
    }

    /// Fused multiply and accumulation, without allocating a product vector.
    fn apply_add(&self, alpha: T, x: &Vector<T, B>, beta: T, y: &mut Vector<T, B>) {
        let (rows, cols) = self.shape();
        assert_eq!(
            (y.len(), x.len()),
            (rows, cols),
            "operator apply: incompatible vector lengths"
        );
        self.apply_window(
            x.storage(),
            y.storage_mut(),
            Apply {
                input: 0,
                output: 0,
                alpha,
                beta,
                adjoint: false,
            },
        );
    }

    fn apply_adjoint(&self, x: &Vector<T, B>, y: &mut Vector<T, B>) {
        let (rows, cols) = self.shape();
        assert_eq!(
            (x.len(), y.len()),
            (rows, cols),
            "operator adjoint: incompatible vector lengths"
        );
        self.apply_window(
            x.storage(),
            y.storage_mut(),
            Apply {
                input: 0,
                output: 0,
                alpha: T::one(),
                beta: T::zero(),
                adjoint: true,
            },
        );
    }
}

impl<T: Field, B: OperatorBackend<T>> LinearOperator<T, B> for Matrix<T, B> {
    fn shape(&self) -> (usize, usize) {
        self.shape()
    }
    fn apply_window(&self, x: &B::Storage<T>, y: &mut B::Storage<T>, p: Apply<T>) {
        B::dense(self.storage(), self.shape(), x, y, p);
    }
}

/// A diagonal map. An all-zero block is represented by an absent block instead.
pub struct DiagonalOperator<T, B: Backend = Host> {
    diagonal: Vector<T, B>,
}

impl<T: Field, B: Backend> DiagonalOperator<T, B> {
    pub fn new(diagonal: Vector<T, B>) -> Self {
        Self { diagonal }
    }
    pub fn diagonal(&self) -> &Vector<T, B> {
        &self.diagonal
    }
}

impl<T: Field, B: OperatorBackend<T>> LinearOperator<T, B> for DiagonalOperator<T, B> {
    fn shape(&self) -> (usize, usize) {
        (self.diagonal.len(), self.diagonal.len())
    }
    fn apply_window(&self, x: &B::Storage<T>, y: &mut B::Storage<T>, p: Apply<T>) {
        B::diagonal(self.diagonal.storage(), self.diagonal.len(), x, y, p);
    }
}

/// A rectangular partition of heterogeneous operators on one backend.
///
/// Blocks can themselves be block operators. Missing blocks represent zero.
/// All child shapes are checked when inserted, before applying any kernels.
pub struct BlockOperator<T: Field, B: Backend = Host> {
    row_offsets: Vec<usize>,
    col_offsets: Vec<usize>,
    blocks: Vec<Option<Box<dyn LinearOperator<T, B>>>>,
}

fn partition(sizes: &[usize]) -> Result<Vec<usize>, Error> {
    let mut offsets = vec![0usize];
    for &size in sizes {
        offsets.push(
            offsets
                .last()
                .unwrap()
                .checked_add(size)
                .ok_or_else(|| Error::shape("block extents overflow usize"))?,
        );
    }
    Ok(offsets)
}

impl<T: Field, B: Backend> BlockOperator<T, B> {
    pub fn new(row_sizes: &[usize], col_sizes: &[usize]) -> Result<Self, Error> {
        let count = row_sizes
            .len()
            .checked_mul(col_sizes.len())
            .ok_or_else(|| Error::shape("block count overflows usize"))?;
        Ok(Self {
            row_offsets: partition(row_sizes)?,
            col_offsets: partition(col_sizes)?,
            blocks: (0..count).map(|_| None).collect(),
        })
    }

    pub fn set_block<O: LinearOperator<T, B> + 'static>(
        &mut self,
        row: usize,
        col: usize,
        block: O,
    ) -> Result<(), Error> {
        if row >= self.row_offsets.len() - 1 || col >= self.col_offsets.len() - 1 {
            return Err(Error::shape("block index out of bounds"));
        }
        let shape = (
            self.row_offsets[row + 1] - self.row_offsets[row],
            self.col_offsets[col + 1] - self.col_offsets[col],
        );
        if block.shape() != shape {
            return Err(Error::shape(format!(
                "block {:?} does not fit partition {shape:?}",
                block.shape()
            )));
        }
        self.blocks[row * (self.col_offsets.len() - 1) + col] = Some(Box::new(block));
        Ok(())
    }

    pub fn row_offsets(&self) -> &[usize] {
        &self.row_offsets
    }
    pub fn col_offsets(&self) -> &[usize] {
        &self.col_offsets
    }
}

impl<T: Field, B: OperatorBackend<T>> LinearOperator<T, B> for BlockOperator<T, B> {
    fn shape(&self) -> (usize, usize) {
        (
            *self.row_offsets.last().unwrap(),
            *self.col_offsets.last().unwrap(),
        )
    }
    fn apply_window(&self, x: &B::Storage<T>, y: &mut B::Storage<T>, p: Apply<T>) {
        let (rows, cols) = self.shape();
        B::scale(y, p.output, if p.adjoint { cols } else { rows }, p.beta);
        for row in 0..self.row_offsets.len() - 1 {
            for col in 0..self.col_offsets.len() - 1 {
                if let Some(block) = &self.blocks[row * (self.col_offsets.len() - 1) + col] {
                    let (input, output) = if p.adjoint {
                        (self.row_offsets[row], self.col_offsets[col])
                    } else {
                        (self.col_offsets[col], self.row_offsets[row])
                    };
                    block.apply_window(
                        x,
                        y,
                        Apply {
                            input: p.input + input,
                            output: p.output + output,
                            beta: T::one(),
                            ..p
                        },
                    );
                }
            }
        }
    }
}

/// Dense local matrices and global DOF maps, applied without global assembly.
///
/// Each local matrix is row-major with `width * width` values. Shared DOFs
/// accumulate all element contributions. Orient edge/face bases in the local
/// matrices before constructing this operator. A reverse adjacency groups
/// contributions by output DOF, avoiding GPU scatter races and atomics.
pub struct ElementOperator<T, B: Backend = Host> {
    pub(crate) size: usize,
    pub(crate) width: usize,
    pub(crate) values: B::Storage<T>,
    pub(crate) dofs: B::Storage<u32>,
    pub(crate) offsets: B::Storage<u32>,
    pub(crate) contributors: B::Storage<u32>,
}

impl<T: Field> ElementOperator<T, Host> {
    pub fn new(size: usize, width: usize, dofs: Vec<u32>, values: Vec<T>) -> Result<Self, Error> {
        if width == 0
            || !dofs.len().is_multiple_of(width)
            || dofs.len().checked_mul(width) != Some(values.len())
        {
            return Err(Error::shape(
                "element matrices and DOF maps have incompatible sizes",
            ));
        }
        if size > u32::MAX as usize
            || values.len() > u32::MAX as usize
            || dofs.iter().any(|&d| d as usize >= size)
        {
            return Err(Error::shape(
                "element DOFs are out of range or exceed 32-bit indexing",
            ));
        }
        let mut grouped = vec![Vec::new(); size];
        for (local, &global) in dofs.iter().enumerate() {
            grouped[global as usize].push(local as u32);
        }
        let mut offsets = vec![0];
        let mut contributors = Vec::with_capacity(dofs.len());
        for group in grouped {
            contributors.extend(group);
            offsets.push(contributors.len() as u32);
        }
        Ok(Self {
            size,
            width,
            values,
            dofs,
            offsets,
            contributors,
        })
    }
}

impl<T: Field, B: Backend> ElementOperator<T, B> {
    pub fn to_backend<B2: Backend>(&self) -> ElementOperator<T, B2> {
        use super::backend::transfer;
        ElementOperator {
            size: self.size,
            width: self.width,
            values: transfer::<T, B, B2>(&self.values),
            dofs: transfer::<u32, B, B2>(&self.dofs),
            offsets: transfer::<u32, B, B2>(&self.offsets),
            contributors: transfer::<u32, B, B2>(&self.contributors),
        }
    }
}

impl<T: Field, B: OperatorBackend<T>> LinearOperator<T, B> for ElementOperator<T, B> {
    fn shape(&self) -> (usize, usize) {
        (self.size, self.size)
    }
    fn apply_window(&self, x: &B::Storage<T>, y: &mut B::Storage<T>, p: Apply<T>) {
        B::elements(self, x, y, p);
    }
}

fn store<T: Field>(y: &mut [T], row: usize, sum: T, p: Apply<T>) {
    let old = if p.beta.is_zero() {
        T::zero()
    } else {
        p.beta * y[p.output + row]
    };
    y[p.output + row] = p.alpha * sum + old;
}

impl<T: Field> OperatorBackend<T> for Host {
    fn dense(a: &Vec<T>, (rows, cols): (usize, usize), x: &Vec<T>, y: &mut Vec<T>, p: Apply<T>) {
        let (out, inner) = if p.adjoint {
            (cols, rows)
        } else {
            (rows, cols)
        };
        for i in 0..out {
            let mut sum = T::zero();
            for j in 0..inner {
                sum = sum
                    + (if p.adjoint {
                        a[j * cols + i].conj()
                    } else {
                        a[i * cols + j]
                    }) * x[p.input + j];
            }
            store(y, i, sum, p);
        }
    }
    fn csr(a: &super::CsrMatrix<T>, x: &Vec<T>, y: &mut Vec<T>, p: Apply<T>) {
        let (offsets, columns) = if p.adjoint {
            (&a.adjoint_offsets, &a.adjoint_columns)
        } else {
            (&a.offsets, &a.columns)
        };
        for row in 0..offsets.len() - 1 {
            let mut sum = T::zero();
            for k in offsets[row] as usize..offsets[row + 1] as usize {
                let value = if p.adjoint {
                    a.values[a.adjoint_positions[k] as usize].conj()
                } else {
                    a.values[k]
                };
                sum = sum + value * x[p.input + columns[k] as usize];
            }
            store(y, row, sum, p);
        }
    }
    fn diagonal(a: &Vec<T>, n: usize, x: &Vec<T>, y: &mut Vec<T>, p: Apply<T>) {
        for i in 0..n {
            store(
                y,
                i,
                (if p.adjoint { a[i].conj() } else { a[i] }) * x[p.input + i],
                p,
            );
        }
    }
    fn elements(a: &ElementOperator<T>, x: &Vec<T>, y: &mut Vec<T>, p: Apply<T>) {
        for row in 0..a.size {
            let mut sum = T::zero();
            for &local in &a.contributors[a.offsets[row] as usize..a.offsets[row + 1] as usize] {
                let (element, i) = (local as usize / a.width, local as usize % a.width);
                for j in 0..a.width {
                    let base = element * a.width * a.width;
                    let value = if p.adjoint {
                        a.values[base + j * a.width + i].conj()
                    } else {
                        a.values[base + i * a.width + j]
                    };
                    sum = sum + value * x[p.input + a.dofs[element * a.width + j] as usize];
                }
            }
            store(y, row, sum, p);
        }
    }
    fn scale(y: &mut Vec<T>, offset: usize, len: usize, beta: T) {
        for v in &mut y[offset..offset + len] {
            *v = if beta.is_zero() { T::zero() } else { beta * *v };
        }
    }
    fn axpby(x: &Vec<T>, y: &mut Vec<T>, len: usize, alpha: T, beta: T) {
        for (i, &value) in x[..len].iter().enumerate() {
            store(
                y,
                i,
                value,
                Apply {
                    input: 0,
                    output: 0,
                    alpha,
                    beta,
                    adjoint: false,
                },
            );
        }
    }
    fn inner(x: &Vec<T>, y: &Vec<T>, len: usize) -> T {
        x[..len]
            .iter()
            .zip(&y[..len])
            .fold(T::zero(), |sum, (a, b)| sum + a.conj() * *b)
    }
    fn norm(x: &Vec<T>, len: usize) -> f64 {
        x[..len]
            .iter()
            .fold(0.0, |n: f64, x| n.hypot(x.magnitude()))
    }
    fn orthogonalize(
        basis: &Vec<T>,
        n: usize,
        count: usize,
        work: &mut Vec<T>,
    ) -> (Vec<T>, f64, f64) {
        assert!(count.checked_mul(n).is_some_and(|size| size <= basis.len()));
        assert_eq!(work.len(), n);
        let before = Self::norm(work, n);
        let mut total = vec![T::zero(); count];
        for _ in 0..2 {
            let projections: Vec<T> = (0..count)
                .map(|k| {
                    basis[k * n..(k + 1) * n]
                        .iter()
                        .zip(&*work)
                        .fold(T::zero(), |s, (v, w)| s + v.conj() * *w)
                })
                .collect();
            for (sum, &value) in total.iter_mut().zip(&projections) {
                *sum = *sum + value;
            }
            for (i, w) in work.iter_mut().enumerate() {
                let sum = (0..count).fold(T::zero(), |s, k| s + basis[k * n + i] * projections[k]);
                *w = *w - sum;
            }
        }
        let after = Self::norm(work, n);
        (total, before, after)
    }
    fn triangular(a: &super::TriangularOperator<T>, x: &Vec<T>, y: &mut Vec<T>, p: Apply<T>) {
        let mut result = vec![T::zero(); a.matrix.rows()];
        let (offsets, columns, order) = if p.adjoint {
            (
                &a.matrix.adjoint_offsets,
                &a.matrix.adjoint_columns,
                &a.adjoint_rows,
            )
        } else {
            (&a.matrix.offsets, &a.matrix.columns, &a.rows)
        };
        for &row in order {
            let row = row as usize;
            let mut rhs = x[p.input + row];
            let mut diagonal = T::one();
            let start = offsets[row] as usize;
            let end = offsets[row + 1] as usize;
            for (local, &col) in columns[start..end].iter().enumerate() {
                let k = start + local;
                let col = col as usize;
                let value = if p.adjoint {
                    a.matrix.values[a.matrix.adjoint_positions[k] as usize].conj()
                } else {
                    a.matrix.values[k]
                };
                if col == row {
                    if !a.unit_diagonal {
                        diagonal = value;
                    }
                } else if !value.is_zero() {
                    rhs = rhs - value * result[col];
                }
            }
            result[row] = rhs / diagonal;
        }
        for (i, &value) in result.iter().enumerate() {
            store(y, i, value, p);
        }
    }
}
