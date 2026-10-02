//! Dense Host products through Apple's Accelerate BLAS. Accelerate owns the
//! processor-specific choice (including Apple-silicon matrix hardware), so
//! this stays on public APIs rather than binding undocumented instructions.

use std::any::TypeId;
use std::ffi::{c_double, c_float, c_int};

use super::HOST_MATMUL_DISPATCH_OPS;
use crate::numbers::Coefficient;

const CBLAS_ROW_MAJOR: c_int = 101;
const CBLAS_NO_TRANS: c_int = 111;
const CBLAS_TRANS: c_int = 112;

#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: c_int,
        transpose_a: c_int,
        transpose_b: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: c_float,
        a: *const c_float,
        leading_a: c_int,
        b: *const c_float,
        leading_b: c_int,
        beta: c_float,
        c: *mut c_float,
        leading_c: c_int,
    );

    fn cblas_dgemm(
        order: c_int,
        transpose_a: c_int,
        transpose_b: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: c_double,
        a: *const c_double,
        leading_a: c_int,
        b: *const c_double,
        leading_b: c_int,
        beta: c_double,
        c: *mut c_double,
        leading_c: c_int,
    );
}

pub fn matmul<T: Coefficient>(
    a: &[T],
    b: &[T],
    rows: usize,
    inner: usize,
    cols: usize,
    output: &mut [T],
    accumulate: bool,
) -> bool {
    gemm(a, false, b, false, (rows, inner, cols), output, accumulate)
}

/// `output = op(a)·op(b)`, plus `output` itself if `accumulate`, where `op`
/// transposes an operand whose flag is set: `op(a)` is `rows × inner` and
/// `op(b)` is `inner × cols`, each stored row-major as it is before `op`.
/// Whether Accelerate ran it.
pub fn gemm<T: Coefficient>(
    a: &[T],
    transpose_a: bool,
    b: &[T],
    transpose_b: bool,
    (rows, inner, cols): (usize, usize, usize),
    output: &mut [T],
    accumulate: bool,
) -> bool {
    if rows.saturating_mul(inner).saturating_mul(cols) < HOST_MATMUL_DISPATCH_OPS {
        return false;
    }
    let Ok(m) = c_int::try_from(rows) else {
        return false;
    };
    let Ok(k) = c_int::try_from(inner) else {
        return false;
    };
    let Ok(n) = c_int::try_from(cols) else {
        return false;
    };
    assert!(a.len() == rows * inner && b.len() == inner * cols && output.len() == rows * cols);
    // A row-major operand's leading dimension is its stored row length.
    let (op_a, leading_a) = if transpose_a {
        (CBLAS_TRANS, m)
    } else {
        (CBLAS_NO_TRANS, k)
    };
    let (op_b, leading_b) = if transpose_b {
        (CBLAS_TRANS, k)
    } else {
        (CBLAS_NO_TRANS, n)
    };

    if TypeId::of::<T>() == TypeId::of::<f32>() {
        // SAFETY: TypeId equality proves the element layouts, the lengths were
        // checked above, and BLAS writes exactly m*n row-major values into
        // `output`.
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                op_a,
                op_b,
                m,
                n,
                k,
                1.0,
                a.as_ptr().cast(),
                leading_a,
                b.as_ptr().cast(),
                leading_b,
                f32::from(accumulate),
                output.as_mut_ptr().cast(),
                n,
            );
        }
        return true;
    }

    if TypeId::of::<T>() == TypeId::of::<f64>() {
        // SAFETY: As above, with f64 established by TypeId equality.
        unsafe {
            cblas_dgemm(
                CBLAS_ROW_MAJOR,
                op_a,
                op_b,
                m,
                n,
                k,
                1.0,
                a.as_ptr().cast(),
                leading_a,
                b.as_ptr().cast(),
                leading_b,
                f64::from(accumulate),
                output.as_mut_ptr().cast(),
                n,
            );
        }
        return true;
    }

    false
}

#[cfg(test)]
mod tests {
    use crate::tensors::accelerate_dispatch;

    #[test]
    fn sgemm_and_dgemm_dispatch_at_the_host_threshold() {
        const SIDE: usize = 8;

        let a32 = vec![1.0f32; SIDE * SIDE];
        let b32 = vec![2.0f32; SIDE * SIDE];
        let mut c32 = vec![3.0f32; SIDE * SIDE];
        assert!(accelerate_dispatch::matmul(
            &a32, &b32, SIDE, SIDE, SIDE, &mut c32, true,
        ));
        assert_eq!(c32, vec![19.0f32; SIDE * SIDE]);

        let a64 = vec![1.0f64; SIDE * SIDE];
        let b64 = vec![2.0f64; SIDE * SIDE];
        let mut c64 = vec![0.0f64; SIDE * SIDE];
        assert!(accelerate_dispatch::matmul(
            &a64, &b64, SIDE, SIDE, SIDE, &mut c64, false,
        ));
        assert_eq!(c64, vec![16.0f64; SIDE * SIDE]);

        let mut small = [0.0f32; 1];
        assert!(!accelerate_dispatch::matmul(
            &[1.0],
            &[2.0],
            1,
            1,
            1,
            &mut small,
            false,
        ));
    }
}
