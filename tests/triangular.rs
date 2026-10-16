use tensorcrate::solvers::{GmresOptions, Ilu0, StopReason, gmres_preconditioned_operator};
use tensorcrate::tensors::{CsrMatrix, LinearOperator, Triangle, TriangularOperator, Vector};

#[test]
fn lower_upper_and_unit_triangular_inverses_match_dense() {
    for triangle in [Triangle::Lower, Triangle::Upper] {
        for unit in [false, true] {
            let entries = (0..5).flat_map(|i| {
                (0..5).filter_map(move |j| {
                    let keep = match triangle {
                        Triangle::Lower => j <= i,
                        Triangle::Upper => j >= i,
                    };
                    keep.then_some((
                        i,
                        j,
                        if i == j {
                            if unit { 1.0 } else { 3.0 }
                        } else {
                            (i + j + 1) as f64 * 0.1
                        },
                    ))
                })
            });
            let matrix = CsrMatrix::from_triplets(5, 5, entries);
            let inverse = matrix.to_dense().inverse().unwrap();
            let op = TriangularOperator::new(matrix, triangle, unit).unwrap();
            let x = Vector::new([1.0, 2.0, 3.0, 4.0, 5.0]);
            let mut y = Vector::zeros(5);
            op.apply(&x, &mut y);
            for (&a, &b) in y.as_slice().iter().zip(inverse.matvec(&x).as_slice()) {
                assert!((a - b).abs() < 1e-12);
            }
            op.apply_adjoint(&x, &mut y);
            for (&a, &b) in y
                .as_slice()
                .iter()
                .zip(inverse.transpose().matvec(&x).as_slice())
            {
                assert!((a - b).abs() < 1e-12);
            }
        }
    }
}

#[test]
fn triangular_setup_rejects_invalid_structure_and_pivots() {
    assert!(
        TriangularOperator::new(
            CsrMatrix::from_triplets(2, 2, [(0, 0, 1.0), (0, 1, 2.0), (1, 1, 1.0)]),
            Triangle::Lower,
            false
        )
        .is_err()
    );
    assert!(
        TriangularOperator::new(
            CsrMatrix::<f64>::from_triplets(2, 2, []),
            Triangle::Lower,
            false
        )
        .is_err()
    );
    assert!(
        TriangularOperator::new(
            CsrMatrix::from_triplets(1, 1, [(0, 0, f64::NAN)]),
            Triangle::Lower,
            false
        )
        .is_err()
    );
    let empty = TriangularOperator::new(
        CsrMatrix::<f64>::from_triplets(0, 0, []),
        Triangle::Lower,
        false,
    )
    .unwrap();
    assert_eq!(empty.level_count(), 0);
    empty.apply(&Vector::zeros(0), &mut Vector::zeros(0));
}

#[test]
fn ilu_inverse_operator_matches_host_preconditioner_and_adjoint() {
    use tensorcrate::solvers::Preconditioner;
    let a = CsrMatrix::<f64>::from_triplets(
        3,
        3,
        [
            (0, 0, 4.0),
            (0, 1, 1.0),
            (1, 0, -1.0),
            (1, 1, 5.0),
            (1, 2, 2.0),
            (2, 1, 1.0),
            (2, 2, 3.0),
        ],
    );
    let ilu = Ilu0::new(&a).unwrap();
    let inverse = ilu.operator::<tensorcrate::tensors::Host>().unwrap();
    let x = Vector::new([1.0, 2.0, 3.0]);
    let mut y = Vector::zeros(3);
    inverse.apply(&x, &mut y);
    let mut expected = vec![0.0; 3];
    ilu.apply(x.as_slice(), &mut expected);
    for (&a, &b) in y.as_slice().iter().zip(&expected) {
        assert!((a - b).abs() < 1e-12);
    }
    inverse.apply_adjoint(&x, &mut y);
    let expected = a.to_dense().inverse().unwrap().transpose().matvec(&x);
    for (&a, &b) in y.as_slice().iter().zip(expected.as_slice()) {
        assert!((a - b).abs() < 1e-12);
    }
    let report = gmres_preconditioned_operator(
        &a,
        &x,
        &mut Vector::zeros(3),
        &inverse,
        GmresOptions::default(),
    )
    .unwrap();
    assert_eq!(report.reason, StopReason::Converged);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_triangular_complex_windows_and_ilu_solve_match_host() {
    use tensorcrate::numbers::{Coefficient, Complex};
    use tensorcrate::tensors::{Adjoint, Matrix};
    use tensorcrate::tensors::{BlockOperator, Metal};
    type C = Complex<f32>;
    let matrix = Matrix::from_rows([
        [C::new(2.0, 1.0), C::new(0.0, 0.0), C::new(0.0, 0.0)],
        [C::new(1.0, -2.0), C::new(3.0, 1.0), C::new(0.0, 0.0)],
        [C::new(-1.0, 1.0), C::new(2.0, 1.0), C::new(4.0, -1.0)],
    ]);
    let sparse = CsrMatrix::from_triplets(
        3,
        3,
        (0..3).flat_map(|i| {
            let m = &matrix;
            (0..=i).map(move |j| (i, j, m[(i, j)]))
        }),
    );
    let a = TriangularOperator::new(sparse, Triangle::Lower, false).unwrap();
    let gpu = a.to_backend::<Metal>();
    let mut block = BlockOperator::new(&[1, 3], &[1, 3]).unwrap();
    block.set_block(1, 1, gpu).unwrap();
    let x = Vector::new([
        C::new(9.0, 0.0),
        C::new(1.0, 2.0),
        C::new(3.0, 1.0),
        C::new(-1.0, 2.0),
    ])
    .to_backend::<Metal>();
    let mut y = Vector::<C>::filled(4, C::new(1.0, 1.0)).to_backend::<Metal>();
    block.apply_add(C::new(2.0, 1.0), &x, C::new(-1.0, 2.0), &mut y);
    let rhs = Vector::new(x.as_slice()[1..].to_vec());
    let expected = matrix.inverse().unwrap().matvec(&rhs);
    for (&actual, &v) in y.as_slice()[1..].iter().zip(expected.as_slice()) {
        assert!(
            (actual - (C::new(2.0, 1.0) * v + C::new(-1.0, 2.0) * C::new(1.0, 1.0))).magnitude()
                < 1e-5
        );
    }
    block.apply_adjoint(&x, &mut y);
    let expected = matrix.inverse().unwrap().adjoint().matvec(&rhs);
    for (&actual, &v) in y.as_slice()[1..].iter().zip(expected.as_slice()) {
        assert!((actual - v).magnitude() < 1e-5);
    }
    let a = CsrMatrix::from_triplets(
        3,
        3,
        [
            (0, 0, 4.0f32),
            (0, 1, -1.0),
            (1, 0, -1.0),
            (1, 1, 4.0),
            (1, 2, -1.0),
            (2, 1, -1.0),
            (2, 2, 4.0),
        ],
    );
    let inverse = Ilu0::new(&a).unwrap().operator::<Metal>().unwrap();
    let mut x = Vector::<f32>::zeros(3).to_backend::<Metal>();
    let report = gmres_preconditioned_operator(
        &a.to_backend::<Metal>(),
        &Vector::new([1.0f32, 2.0, 3.0]).to_backend::<Metal>(),
        &mut x,
        &inverse,
        GmresOptions {
            relative_tolerance: 1e-5,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(report.reason, StopReason::Converged);
    assert_eq!(report.iterations, 1);
}
