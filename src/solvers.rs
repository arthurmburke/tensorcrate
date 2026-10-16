//! Restarted GMRES with right preconditioning for real and complex systems.
//!
//! The convergence test always checks the true residual `b - A x`. Arnoldi
//! uses two passes of modified Gram-Schmidt and complex unitary rotations.
//! Krylov vectors stay on their backend. Metal reductions return small partial
//! sums to the CPU; Hessenberg matrices and convergence decisions run there.
//! ILU(0) executes on Host; diagonal and block inverse operators can run on GPU.

use crate::errors::Error;
use crate::numbers::Field;
use crate::tensors::{
    Backend, CsrMatrix, DiagonalOperator, LinearOperator, OperatorBackend, Vector,
};

#[derive(Clone, Copy, Debug)]
pub struct GmresOptions {
    pub restart: usize,
    pub max_iterations: usize,
    pub relative_tolerance: f64,
    pub absolute_tolerance: f64,
}

impl Default for GmresOptions {
    fn default() -> Self {
        Self {
            restart: 30,
            max_iterations: 1000,
            relative_tolerance: 1e-8,
            absolute_tolerance: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    Converged,
    IterationLimit,
    Breakdown,
    NonFinite,
}

#[derive(Clone, Debug)]
pub struct SolveReport {
    pub reason: StopReason,
    pub iterations: usize,
    pub residual_norm: f64,
    /// True residual norms at initialization and after each solution update.
    pub residual_history: Vec<f64>,
}

/// Application of the inverse preconditioning approximation, `z = M^-1 r`.
pub trait Preconditioner<T: Field> {
    fn size(&self) -> usize;
    fn apply(&self, rhs: &[T], result: &mut [T]);
}

pub struct IdentityPreconditioner(pub usize);
impl<T: Field> Preconditioner<T> for IdentityPreconditioner {
    fn size(&self) -> usize {
        self.0
    }
    fn apply(&self, rhs: &[T], result: &mut [T]) {
        assert_eq!(rhs.len(), self.0);
        assert_eq!(result.len(), self.0);
        result.copy_from_slice(rhs);
    }
}

/// Inverse diagonal scaling. Missing, zero or nonfinite pivots are rejected.
pub struct Jacobi<T> {
    inverse: Vec<T>,
}
impl<T: Field> Jacobi<T> {
    pub fn new(matrix: &CsrMatrix<T>) -> Result<Self, Error> {
        if matrix.rows() != matrix.cols() {
            return Err(Error::shape("Jacobi requires a square matrix"));
        }
        let mut inverse = matrix.diagonal();
        for value in &mut inverse {
            check_pivot(*value)?;
            *value = T::one() / *value;
            if !value.magnitude().is_finite() {
                return Err(Error::InvalidArgument("Jacobi inverse overflows".into()));
            }
        }
        Ok(Self { inverse })
    }

    /// Upload the inverse diagonal once for use with `gmres_preconditioned_operator`.
    pub fn operator<B: Backend>(&self) -> DiagonalOperator<T, B> {
        DiagonalOperator::new(Vector::new(self.inverse.clone()).to_backend::<B>())
    }
}
impl<T: Field> Preconditioner<T> for Jacobi<T> {
    fn size(&self) -> usize {
        self.inverse.len()
    }
    fn apply(&self, rhs: &[T], result: &mut [T]) {
        assert_eq!(rhs.len(), self.size());
        assert_eq!(result.len(), self.size());
        for ((out, &rhs), &inverse) in result.iter_mut().zip(rhs).zip(&self.inverse) {
            *out = rhs * inverse;
        }
    }
}

/// ILU(0) with the original CSR pattern. No pivoting or fill is performed.
/// Setup rejects absent or zero pivots rather than silently perturbing them.
pub struct Ilu0<T> {
    offsets: Vec<u32>,
    columns: Vec<u32>,
    factors: Vec<T>,
    diagonal: Vec<usize>,
}

fn check_pivot<T: Field>(value: T) -> Result<(), Error> {
    if value.is_zero() {
        return Err(Error::Singular);
    }
    if !value.magnitude().is_finite() {
        return Err(Error::InvalidArgument(
            "nonfinite preconditioner pivot".into(),
        ));
    }
    Ok(())
}

impl<T: Field> Ilu0<T> {
    pub fn new(matrix: &CsrMatrix<T>) -> Result<Self, Error> {
        if matrix.rows() != matrix.cols() {
            return Err(Error::shape("ILU requires a square matrix"));
        }
        let offsets = matrix.row_offsets().to_vec();
        let columns = matrix.column_indices().to_vec();
        let mut factors = matrix.values().to_vec();
        let mut diagonal = Vec::with_capacity(matrix.rows());
        for row in 0..matrix.rows() {
            let start = offsets[row] as usize;
            let end = offsets[row + 1] as usize;
            let local = columns[start..end]
                .binary_search(&(row as u32))
                .map_err(|_| Error::Singular)?;
            diagonal.push(start + local);
        }
        for row in 0..matrix.rows() {
            let end = offsets[row + 1] as usize;
            for p in offsets[row] as usize..diagonal[row] {
                let lower_col = columns[p] as usize;
                check_pivot(factors[diagonal[lower_col]])?;
                let multiplier = factors[p] / factors[diagonal[lower_col]];
                factors[p] = multiplier;
                for q in diagonal[lower_col] + 1..offsets[lower_col + 1] as usize {
                    if let Ok(k) = columns[p + 1..end].binary_search(&columns[q]) {
                        let slot = p + 1 + k;
                        factors[slot] = factors[slot] - multiplier * factors[q];
                    }
                }
            }
            check_pivot(factors[diagonal[row]])?;
        }
        if factors.iter().any(|v| !v.magnitude().is_finite()) {
            return Err(Error::InvalidArgument("ILU factors overflow".into()));
        }
        Ok(Self {
            offsets,
            columns,
            factors,
            diagonal,
        })
    }
}

impl<T: Field> Preconditioner<T> for Ilu0<T> {
    fn size(&self) -> usize {
        self.diagonal.len()
    }
    fn apply(&self, rhs: &[T], result: &mut [T]) {
        assert_eq!(rhs.len(), self.size());
        assert_eq!(result.len(), self.size());
        result.copy_from_slice(rhs);
        for row in 0..self.size() {
            for k in self.offsets[row] as usize..self.diagonal[row] {
                result[row] = result[row] - self.factors[k] * result[self.columns[k] as usize];
            }
        }
        for row in (0..self.size()).rev() {
            for k in self.diagonal[row] + 1..self.offsets[row + 1] as usize {
                result[row] = result[row] - self.factors[k] * result[self.columns[k] as usize];
            }
            result[row] = result[row] / self.factors[self.diagonal[row]];
        }
    }
}

fn rotation<T: Field>(a: T, b: T) -> (f64, T) {
    let (aa, bb) = (a.magnitude(), b.magnitude());
    if bb == 0.0 {
        (1.0, T::zero())
    } else if aa == 0.0 {
        (0.0, b.conj() / T::from_real(bb))
    } else {
        let length = aa.hypot(bb);
        (
            aa / length,
            (a / T::from_real(aa)) * (b.conj() / T::from_real(length)),
        )
    }
}

fn rotate<T: Field>(a: T, b: T, c: f64, s: T) -> (T, T) {
    let c = T::from_real(c);
    (c * a + s * b, -s.conj() * a + c * b)
}

/// Solve `A x = b` starting from the supplied `x`. Iteration exhaustion and
/// breakdown are returned in the report, never reported as convergence.
pub fn gmres<T: Field, B: OperatorBackend<T>, O: LinearOperator<T, B> + ?Sized>(
    a: &O,
    b: &Vector<T, B>,
    x: &mut Vector<T, B>,
    options: GmresOptions,
) -> Result<SolveReport, Error> {
    solve(a, b, x, options, |input, output| {
        B::axpby(
            input.storage(),
            output.storage_mut(),
            input.len(),
            T::one(),
            T::zero(),
        )
    })
}

/// GMRES with right preconditioning. Only the operator and preconditioner
/// interfaces are required; the operator may be a nested block composition.
pub fn gmres_preconditioned<
    T: Field,
    O: LinearOperator<T> + ?Sized,
    P: Preconditioner<T> + ?Sized,
>(
    a: &O,
    b: &Vector<T>,
    x: &mut Vector<T>,
    preconditioner: &P,
    options: GmresOptions,
) -> Result<SolveReport, Error> {
    if preconditioner.size() != b.len() {
        return Err(Error::shape("GMRES preconditioner has incompatible size"));
    }
    solve(a, b, x, options, |input, output| {
        preconditioner.apply(input.as_slice(), output.data_mut())
    })
}

/// Solve using a backend-resident inverse preconditioning operator.
///
/// `inverse` applies `M^-1`, not `M`. For Metal this keeps all full-sized
/// Krylov vectors resident, including preconditioning and orthogonalization.
pub fn gmres_preconditioned_operator<
    T: Field,
    B: OperatorBackend<T>,
    O: LinearOperator<T, B> + ?Sized,
    P: LinearOperator<T, B> + ?Sized,
>(
    a: &O,
    b: &Vector<T, B>,
    x: &mut Vector<T, B>,
    inverse: &P,
    options: GmresOptions,
) -> Result<SolveReport, Error> {
    if inverse.shape() != (b.len(), b.len()) {
        return Err(Error::shape(
            "GMRES inverse preconditioner has incompatible shape",
        ));
    }
    solve(a, b, x, options, |input, output| {
        inverse.apply(input, output)
    })
}

fn solve<T: Field, B: OperatorBackend<T>, O: LinearOperator<T, B> + ?Sized>(
    a: &O,
    b: &Vector<T, B>,
    x: &mut Vector<T, B>,
    options: GmresOptions,
    precondition: impl Fn(&Vector<T, B>, &mut Vector<T, B>),
) -> Result<SolveReport, Error> {
    let n = b.len();
    if a.shape() != (n, n) || x.len() != n {
        return Err(Error::shape(
            "GMRES requires square compatible operator, vectors and preconditioner",
        ));
    }
    if options.restart == 0
        || !options.relative_tolerance.is_finite()
        || !options.absolute_tolerance.is_finite()
        || options.relative_tolerance < 0.0
        || options.absolute_tolerance < 0.0
    {
        return Err(Error::InvalidArgument(
            "GMRES requires a positive restart and finite nonnegative tolerances".into(),
        ));
    }
    let rhs_norm = B::norm(b.storage(), n);
    let tolerance = options
        .absolute_tolerance
        .max(options.relative_tolerance * rhs_norm);
    let mut work = Vector::<T, B>::filled(n, T::zero());
    let mut report = SolveReport {
        reason: StopReason::IterationLimit,
        iterations: 0,
        residual_norm: f64::INFINITY,
        residual_history: Vec::new(),
    };
    if !rhs_norm.is_finite() || !B::norm(x.storage(), n).is_finite() {
        report.reason = StopReason::NonFinite;
        return Ok(report);
    }
    let mut stopped = false;
    loop {
        a.apply(x, &mut work);
        B::axpby(b.storage(), work.storage_mut(), n, T::one(), -T::one());
        let length = B::norm(work.storage(), n);
        report.residual_norm = length;
        report.residual_history.push(length);
        if !length.is_finite() || !tolerance.is_finite() {
            report.reason = StopReason::NonFinite;
            return Ok(report);
        }
        if length <= tolerance {
            report.reason = StopReason::Converged;
            return Ok(report);
        }
        if stopped {
            report.reason = StopReason::Breakdown;
            return Ok(report);
        }
        if report.iterations == options.max_iterations {
            return Ok(report);
        }
        let m = options
            .restart
            .min(options.max_iterations - report.iterations)
            .min(n);
        let mut basis = Vec::with_capacity(m + 1);
        let mut first = work.to_backend::<B>();
        B::scale(first.storage_mut(), 0, n, T::one() / T::from_real(length));
        basis.push(first);
        let mut preconditioned = Vec::with_capacity(m);
        let mut h = vec![vec![T::zero(); m + 1]; m];
        let mut c = vec![0.0; m];
        let mut s = vec![T::zero(); m];
        let mut g = vec![T::zero(); m + 1];
        g[0] = T::from_real(length);
        let mut input = Vector::filled(n, T::zero());
        let mut count = 0;
        let mut breakdown = false;
        for j in 0..m {
            precondition(&basis[j], &mut input);
            preconditioned.push(input.to_backend::<B>());
            a.apply(&input, &mut work);
            let before = B::norm(work.storage(), n);
            for _ in 0..2 {
                for i in 0..=j {
                    let projection = B::inner(basis[i].storage(), work.storage(), n);
                    h[j][i] = h[j][i] + projection;
                    B::axpby(
                        basis[i].storage(),
                        work.storage_mut(),
                        n,
                        -projection,
                        T::one(),
                    );
                }
            }
            let next = B::norm(work.storage(), n);
            if !before.is_finite() || !next.is_finite() {
                report.reason = StopReason::NonFinite;
                return Ok(report);
            }
            h[j][j + 1] = T::from_real(next);
            breakdown = next <= 10.0 * T::epsilon() * before;
            if !breakdown {
                let mut next_basis = work.to_backend::<B>();
                B::scale(
                    next_basis.storage_mut(),
                    0,
                    n,
                    T::one() / T::from_real(next),
                );
                basis.push(next_basis);
            }
            for i in 0..j {
                (h[j][i], h[j][i + 1]) = rotate(h[j][i], h[j][i + 1], c[i], s[i]);
            }
            (c[j], s[j]) = rotation(h[j][j], h[j][j + 1]);
            (h[j][j], h[j][j + 1]) = rotate(h[j][j], h[j][j + 1], c[j], s[j]);
            (g[j], g[j + 1]) = rotate(g[j], g[j + 1], c[j], s[j]);
            report.iterations += 1;
            count = j + 1;
            if breakdown || g[j + 1].magnitude() <= tolerance {
                break;
            }
        }
        let mut coefficients = vec![T::zero(); count];
        for i in (0..count).rev() {
            let mut value = g[i];
            for j in i + 1..count {
                value = value - h[j][i] * coefficients[j];
            }
            if h[i][i].is_zero() {
                report.reason = StopReason::Breakdown;
                return Ok(report);
            }
            coefficients[i] = value / h[i][i];
        }
        let mut candidate = x.to_backend::<B>();
        for (coefficient, z) in coefficients.into_iter().zip(preconditioned) {
            B::axpby(
                z.storage(),
                candidate.storage_mut(),
                n,
                coefficient,
                T::one(),
            );
        }
        if !B::norm(candidate.storage(), n).is_finite() {
            report.reason = StopReason::NonFinite;
            return Ok(report);
        }
        *x = candidate;
        stopped = breakdown;
    }
}
