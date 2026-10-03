use tensorcrate::numbers::{Exp, Power, Sin};
use tensorcrate::tensors::{Analytic, Host, Matrix, SparseMatrix};

#[test]
fn triplets_are_sorted_and_duplicate_values_are_combined() {
    let sparse =
        SparseMatrix::from_triplets(3, 4, [(2, 3, 5i32), (0, 1, 2), (0, 1, -2), (1, 0, 3)]);

    assert_eq!(sparse.shape(), (3, 4));
    assert_eq!(sparse.nnz(), 2);
    assert_eq!(
        sparse
            .iter()
            .map(|(coordinate, &value)| (coordinate, value))
            .collect::<Vec<_>>(),
        vec![((1, 0), 3), ((2, 3), 5)]
    );
    assert_eq!(sparse[(0, 0)], 0);
    assert_eq!(sparse[(2, 3)], 5);
}

#[test]
fn mutation_helpers_keep_zeroes_out_of_the_storage() {
    let mut sparse = SparseMatrix::<i32>::zeros(2, 3);
    sparse.set(1, 2, 7);
    sparse[(0, 1)] = 4;
    assert_eq!(sparse.nnz(), 2);

    sparse.set(1, 2, 0);
    assert_eq!(sparse.remove(0, 1), Some(4));
    assert!(sparse.is_empty());
    assert_eq!(sparse.get(0, 0), Some(&0));
    assert_eq!(sparse.get(2, 0), None);
}

#[test]
fn dense_round_trip_transpose_and_matvec_agree() {
    let dense = Matrix::from_rows([[0i32, 2, 0], [3, 0, 4]]);
    let sparse = SparseMatrix::from_dense(&dense);

    assert_eq!(sparse.to_dense::<Host>(), dense);
    assert_eq!(
        sparse.transpose().to_dense::<Host>(),
        Matrix::from_rows([[0, 3], [2, 0], [0, 4]])
    );
    assert_eq!(sparse.matvec(&[10, 20, 30]), vec![40, 150]);
}

#[test]
fn analytic_operations_cover_every_sparse_entry() {
    let sparse = SparseMatrix::from_triplets(3, 4, [(0, 1, 0.25f64), (2, 3, 0.5)]);

    for operation in Analytic::ALL {
        let actual = sparse.analytic(operation);
        let expected = SparseMatrix::from_triplets(
            3,
            4,
            sparse
                .iter()
                .map(|((row, col), &value)| (row, col, operation.value(value))),
        );
        assert_eq!(actual, expected, "{operation:?}");
        assert_eq!(actual.shape(), sparse.shape());
    }

    assert_eq!(sparse.sin(), sparse.analytic(Analytic::Sin));
    assert_eq!(sparse.exp(), sparse.analytic(Analytic::Exp));
    assert_eq!((&sparse).sin(), sparse.analytic(Analytic::Sin));
}

#[test]
fn analytic_operations_remove_values_that_become_zero() {
    let sparse = SparseMatrix::from_triplets(2, 2, [(0, 0, 1.0f64)]);

    assert!(sparse.ln().is_empty());
    assert_eq!(sparse.ln().shape(), sparse.shape());
}

#[test]
fn sparse_power_operations_preserve_coordinates() {
    let sparse = SparseMatrix::from_triplets(2, 3, [(0, 1, 2.0f64), (1, 2, 3.0)]);
    let exponents = SparseMatrix::from_triplets(2, 3, [(0, 1, 3.0f64), (1, 2, 2.0)]);

    assert_eq!(
        sparse.pow(2.0),
        SparseMatrix::from_triplets(2, 3, [(0, 1, 4.0), (1, 2, 9.0)])
    );
    assert_eq!(
        sparse.pow_elementwise(&exponents),
        SparseMatrix::from_triplets(2, 3, [(0, 1, 8.0), (1, 2, 9.0)])
    );
    assert_eq!((&sparse).power(2.0), sparse.pow(2.0));
}

#[test]
fn sparse_analytic_scalar_traits_keep_the_output_sparse() {
    let sparse = SparseMatrix::from_triplets(2, 2, [(0, 0, 1i32), (1, 1, 2)]);

    let exponential: SparseMatrix<f64> = (&sparse).exp();
    assert_eq!(exponential.shape(), sparse.shape());
    assert_eq!(exponential.nnz(), sparse.nnz());
    assert_eq!(exponential[(0, 0)], 1.0f64.exp());

    let sine: SparseMatrix<f64> = (&sparse).sin();
    assert_eq!(sine[(1, 1)], (2.0f64).sin());
    let inverse_cosine: SparseMatrix<f64> = (&sparse).arccos();
    assert_eq!(inverse_cosine[(0, 0)], (1.0f64).acos());

    let _: fn(&SparseMatrix<f64>) -> SparseMatrix<f64> = |matrix| matrix.sin();
    let _: fn(&SparseMatrix<f64>) -> SparseMatrix<f64> = |matrix| matrix.power(2.0);
    let _: fn(&SparseMatrix<f64>) -> SparseMatrix<f64> = |matrix| matrix.analytic(Analytic::Sin);

    let _ = (
        Exp::exp(&sparse),
        Sin::sin(&sparse),
        Power::power(&sparse, 2),
    );
}
