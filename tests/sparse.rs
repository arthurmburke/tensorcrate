use tensorcrate::tensors::{Host, Matrix, SparseMatrix};

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
