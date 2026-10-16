use tensorcrate::numbers::{Coefficient, Complex, Field};
use tensorcrate::solvers::{
    GmresOptions, IdentityPreconditioner, Ilu0, Jacobi, Preconditioner, StopReason, gmres,
    gmres_preconditioned,
};
use tensorcrate::tensors::{
    BlockOperator, CsrMatrix, DiagonalOperator, LinearOperator, Matrix, Vector,
};

fn options() -> GmresOptions {
    GmresOptions {
        relative_tolerance: 1e-11,
        absolute_tolerance: 1e-13,
        ..Default::default()
    }
}

fn check<T: Field>(a: &impl LinearOperator<T>, expected: Vector<T>, opts: GmresOptions) {
    let mut b = Vector::zeros(expected.len());
    a.apply(&expected, &mut b);
    let mut x = Vector::zeros(expected.len());
    let report = gmres(a, &b, &mut x, opts).unwrap();
    assert_eq!(report.reason, StopReason::Converged);
    let mut ax = Vector::zeros(b.len());
    a.apply(&x, &mut ax);
    let residual = b
        .as_slice()
        .iter()
        .zip(ax.as_slice())
        .fold(0.0f64, |n, (&b, &ax)| n.hypot((b - ax).magnitude()));
    assert!((residual - report.residual_norm).abs() < 1e-10);
    for (&x, &expected) in x.as_slice().iter().zip(expected.as_slice()) {
        assert!((x - expected).magnitude() < 1e-8);
    }
}

#[test]
fn nonsymmetric_real_and_complex_systems_converge() {
    check(
        &Matrix::from_rows([[4.0, 1.0, -2.0], [3.0, 7.0, 1.0], [0.0, -1.0, 5.0]]),
        Vector::new([1.0, -2.0, 3.0]),
        options(),
    );
    type C = Complex<f64>;
    let a = Matrix::from_rows([
        [C::new(4.0, 2.0), C::new(1.0, -3.0), C::new(-2.0, 1.0)],
        [C::new(0.0, 0.0), C::new(-1.0, 4.0), C::new(3.0, 2.0)],
        [C::new(2.0, -1.0), C::new(0.0, 2.0), C::new(5.0, -3.0)],
    ]);
    check(
        &a,
        Vector::new([C::new(1.0, 2.0), C::new(-2.0, 3.0), C::new(4.0, -1.0)]),
        options(),
    );
}

#[test]
fn restart_initial_guess_and_right_preconditioning_work() {
    let n = 12;
    let a = CsrMatrix::from_triplets(
        n,
        n,
        (0..n).flat_map(|i| {
            let mut row = vec![(i, i, (i + 1) as f64 + 3.0)];
            if i > 0 {
                row.push((i, i - 1, -1.0));
            }
            if i + 1 < n {
                row.push((i, i + 1, 0.5));
            }
            row
        }),
    );
    let expected = Vector::new((0..n).map(|i| i as f64 - 2.0).collect::<Vec<_>>());
    let mut b = Vector::zeros(n);
    a.apply(&expected, &mut b);
    let opts = GmresOptions {
        restart: 3,
        ..options()
    };
    check(&a, expected, opts);
    let mut x = Vector::filled(n, 0.25);
    let report = gmres_preconditioned(&a, &b, &mut x, &Jacobi::new(&a).unwrap(), opts).unwrap();
    assert_eq!(report.reason, StopReason::Converged);
    assert!(report.residual_history.len() > 2);
    let ilu = Ilu0::new(&a).unwrap();
    let mut x = Vector::zeros(n);
    let report = gmres_preconditioned(&a, &b, &mut x, &ilu, options()).unwrap();
    assert_eq!(report.reason, StopReason::Converged);
    assert_eq!(report.iterations, 1); // ILU(0) is exact for this tridiagonal.
}

#[test]
fn solver_accepts_heterogeneous_blocks() {
    let mut a = BlockOperator::new(&[2, 1], &[2, 1]).unwrap();
    a.set_block(
        0,
        0,
        CsrMatrix::from_triplets(2, 2, [(0, 0, 3.0), (1, 1, 4.0)]),
    )
    .unwrap();
    a.set_block(0, 1, Matrix::from_rows([[1.0], [2.0]]))
        .unwrap();
    a.set_block(1, 0, Matrix::from_rows([[2.0, -1.0]])).unwrap();
    a.set_block(1, 1, DiagonalOperator::new(Vector::new([5.0])))
        .unwrap();
    check(&a, Vector::new([1.0, -2.0, 3.0]), options());
}

#[test]
fn convergence_limit_breakdown_and_nonfinite_are_distinct() {
    let a = Matrix::from_rows([[1.0, 0.0], [0.0, 2.0]]);
    let b = Vector::new([1.0, 1.0]);
    let mut x = Vector::zeros(2);
    let report = gmres(
        &a,
        &b,
        &mut x,
        GmresOptions {
            max_iterations: 0,
            ..options()
        },
    )
    .unwrap();
    assert_eq!(report.reason, StopReason::IterationLimit);
    assert_eq!(report.iterations, 0);
    let report = gmres(
        &a,
        &b,
        &mut x,
        GmresOptions {
            max_iterations: 1,
            ..options()
        },
    )
    .unwrap();
    assert_eq!(report.reason, StopReason::IterationLimit);
    assert_eq!(report.iterations, 1);
    let report = gmres(&Matrix::<f64>::zeros(2, 2), &b, &mut x, options()).unwrap();
    assert_eq!(report.reason, StopReason::Breakdown);
    let report = gmres(&a, &Vector::new([f64::NAN, 1.0]), &mut x, options()).unwrap();
    assert_eq!(report.reason, StopReason::NonFinite);
    let mut bad_guess = Vector::new([f64::NAN; 2]);
    let report = gmres(
        &Matrix::<f64>::zeros(2, 2),
        &Vector::zeros(2),
        &mut bad_guess,
        options(),
    )
    .unwrap();
    assert_eq!(report.reason, StopReason::NonFinite);
    let mut x = Vector::new([1.0, 0.5]);
    let report = gmres(&a, &b, &mut x, options()).unwrap();
    assert_eq!(report.reason, StopReason::Converged);
    assert_eq!(report.iterations, 0);
    let report = gmres(
        &Matrix::<f64>::zeros(0, 0),
        &Vector::zeros(0),
        &mut Vector::zeros(0),
        options(),
    )
    .unwrap();
    assert_eq!(report.reason, StopReason::Converged);
}

#[test]
fn invalid_shapes_tolerances_and_pivots_are_rejected() {
    let a = Matrix::<f64>::identity(2);
    assert!(gmres(&a, &Vector::zeros(3), &mut Vector::zeros(2), options()).is_err());
    for opts in [
        GmresOptions {
            restart: 0,
            ..options()
        },
        GmresOptions {
            absolute_tolerance: -1.0,
            ..options()
        },
        GmresOptions {
            relative_tolerance: f64::NAN,
            ..options()
        },
    ] {
        assert!(gmres(&a, &Vector::zeros(2), &mut Vector::zeros(2), opts).is_err());
    }
    let missing = CsrMatrix::from_triplets(2, 2, [(0, 1, 1.0)]);
    assert!(Jacobi::new(&missing).is_err());
    assert!(Ilu0::new(&missing).is_err());
    let zero_pivot =
        CsrMatrix::from_triplets(2, 2, [(0, 0, 1.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.0)]);
    assert!(Ilu0::new(&zero_pivot).is_err());
    let mut out = [0.0; 2];
    IdentityPreconditioner(2).apply(&[2.0, 3.0], &mut out);
    assert_eq!(out, [2.0, 3.0]);
}

#[test]
fn complex_ilu_solves_without_losing_phase() {
    type C = Complex<f64>;
    let a = CsrMatrix::from_triplets(
        2,
        2,
        [
            (0, 0, C::new(2.0, 3.0)),
            (0, 1, C::new(-1.0, 2.0)),
            (1, 0, C::new(3.0, -1.0)),
            (1, 1, C::new(4.0, 5.0)),
        ],
    );
    let expected = Vector::new([C::new(2.0, -1.0), C::new(-3.0, 2.0)]);
    let mut b = Vector::zeros(2);
    a.apply(&expected, &mut b);
    let mut x = Vector::zeros(2);
    let report = gmres_preconditioned(&a, &b, &mut x, &Ilu0::new(&a).unwrap(), options()).unwrap();
    assert_eq!(report.reason, StopReason::Converged);
    assert_eq!(report.iterations, 1);
    for (&x, &e) in x.as_slice().iter().zip(expected.as_slice()) {
        assert!((x - e).magnitude() < 1e-10);
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_gmres_keeps_real_and_complex_vectors_resident() {
    use tensorcrate::solvers::gmres_preconditioned_operator;
    use tensorcrate::tensors::Metal;
    let a = CsrMatrix::from_triplets(
        3,
        3,
        [
            (0, 0, 4.0f32),
            (0, 1, 1.0),
            (1, 0, -1.0),
            (1, 1, 5.0),
            (1, 2, 1.0),
            (2, 1, 1.0),
            (2, 2, 6.0),
        ],
    );
    let gpu = a.to_backend::<Metal>();
    let expected = Vector::new([1.0f32, -2.0, 3.0]);
    let mut b = Vector::zeros(3);
    a.apply(&expected, &mut b);
    let mut x = Vector::<f32>::filled(3, 0.0f32).to_backend::<Metal>();
    let opts = GmresOptions {
        relative_tolerance: 2e-5,
        ..Default::default()
    };
    let report = gmres_preconditioned_operator(
        &gpu,
        &b.to_backend::<Metal>(),
        &mut x,
        &Jacobi::new(&a).unwrap().operator::<Metal>(),
        opts,
    )
    .unwrap();
    assert_eq!(report.reason, StopReason::Converged);
    assert!(x.is_device_resident());
    for (&x, &e) in x.as_slice().iter().zip(expected.as_slice()) {
        assert!((x - e).abs() < 1e-4);
    }

    type C = Complex<f32>;
    let a = Matrix::from_rows([
        [C::new(3.0, 1.0), C::new(1.0, -2.0)],
        [C::new(-2.0, 1.0), C::new(4.0, 3.0)],
    ]);
    let expected = Vector::new([C::new(1.0, 2.0), C::new(-3.0, 1.0)]);
    let b = a.matvec(&expected).to_backend::<Metal>();
    let mut x = Vector::<C>::filled(2, C::new(0.0, 0.0)).to_backend::<Metal>();
    let report = gmres(&a.to_backend::<Metal>(), &b, &mut x, opts).unwrap();
    assert_eq!(report.reason, StopReason::Converged);
    for (&x, &e) in x.as_slice().iter().zip(expected.as_slice()) {
        assert!((x - e).magnitude() < 1e-4);
    }
    let report = gmres(
        &a.to_backend::<Metal>(),
        &Vector::new([C::new(f32::NAN, 0.0); 2]).to_backend::<Metal>(),
        &mut x,
        opts,
    )
    .unwrap();
    assert_eq!(report.reason, StopReason::NonFinite);
}
