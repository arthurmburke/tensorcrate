use tensorcrate::numbers::Dual;
use tensorcrate::tensors::{
    Dot, DualMatrix, DualVector, MatMul, MatVec, Matrix, SparseMatrix, Tape, Transpose, VecMat,
    Vector,
};

fn multiply<M>(left: &M, right: &M) -> M::Output
where
    M: MatMul,
{
    MatMul::matmul(left, right)
}

fn transpose<M: Transpose>(matrix: &M) -> M::Output {
    Transpose::transpose(matrix)
}

fn dot<V: Dot>(left: &V, right: &V) -> V::Output {
    Dot::dot(left, right)
}

fn matvec<M, V: ?Sized>(matrix: &M, vector: &V) -> M::Output
where
    M: MatVec<V>,
{
    MatVec::matvec(matrix, vector)
}

fn vecmat<V, M>(vector: &V, matrix: &M) -> V::Output
where
    V: VecMat<M>,
{
    VecMat::vecmat(vector, matrix)
}

#[test]
fn dense_values_implement_linear_algebra_traits() {
    let a = Matrix::from_rows([[1, 2], [3, 4]]);
    let b = Matrix::from_rows([[5, 6], [7, 8]]);
    let v = Vector::new([2, 3]);

    assert_eq!(multiply(&a, &b), Matrix::from_rows([[19, 22], [43, 50]]));
    assert_eq!(transpose(&a), Matrix::from_rows([[1, 3], [2, 4]]));
    assert_eq!(dot(&v, &v), 13);
    assert_eq!(matvec(&a, &v), Vector::new([8, 18]));
    assert_eq!(vecmat(&v, &a), Vector::new([11, 16]));
}

#[test]
fn differentiation_types_keep_their_output_types() {
    let a = Matrix::from_rows([[1.0f32, 2.0], [3.0, 4.0]]);
    let da = Matrix::from_rows([[1.0f32, 0.0], [0.0, 1.0]]);
    let dual = DualMatrix::new(a.clone(), da);
    let dual_product = multiply(&dual, &dual);
    assert_eq!(dual_product.value(), &a.matmul(&a));

    let v = DualVector::new(Vector::new([1.0f32, 2.0]), Vector::new([1.0, 1.0]));
    assert_eq!(dot(&v, &v), Dual::new(5.0, 6.0));

    let tape = Tape::new();
    let recorded = tape.matrix(a.clone());
    let product = multiply(&recorded, &recorded);
    assert_eq!(product.value(), &a.matmul(&a));
    assert_eq!(transpose(&recorded).value(), &a.transpose());
}

#[test]
fn sparse_values_share_matvec_and_transpose_traits() {
    let sparse = SparseMatrix::from_triplets(2, 3, [(0, 1, 2), (1, 2, 4)]);
    assert_eq!(matvec(&sparse, [1, 3, 5].as_slice()), vec![6, 20]);
    assert_eq!(transpose(&sparse).shape(), (3, 2));
}
