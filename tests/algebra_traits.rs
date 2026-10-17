use tensorcrate::numbers::{Complex, Dual};
use tensorcrate::prelude::*;
use tensorcrate::tensors::{
    CsrMatrix, DualMatrix, DualVector, Matrix, SparseMatrix, Tape, Tensor, Vector,
};

// Generic helpers name the reference types, as `&a + &b` bounds do. A helper
// whose output names the trait carries the lifetime; one with a fixed output
// type can use a higher-ranked bound, as `gram` does.

fn multiply<'a, M>(left: &'a M, right: &'a M) -> <&'a M as MatMul<&'a M>>::Output
where
    &'a M: MatMul<&'a M>,
{
    left.matmul(right)
}

fn transpose<'a, M>(matrix: &'a M) -> <&'a M as Transpose>::Output
where
    &'a M: Transpose,
{
    matrix.transpose()
}

fn dot<'a, V>(left: &'a V, right: &'a V) -> <&'a V as Dot<&'a V>>::Output
where
    &'a V: Dot<&'a V>,
{
    left.dot(right)
}

fn matvec<'a, M, V: ?Sized>(matrix: &'a M, vector: &'a V) -> <&'a M as MatVec<&'a V>>::Output
where
    &'a M: MatVec<&'a V>,
{
    matrix.matvec(vector)
}

fn vecmat<'a, V, M>(vector: &'a V, matrix: &'a M) -> <&'a V as VecMat<&'a M>>::Output
where
    &'a V: VecMat<&'a M>,
{
    vector.vecmat(matrix)
}

fn gram<M>(matrix: &M) -> M
where
    for<'a> &'a M: MatMul<&'a M, Output = M> + Transpose<Output = M>,
{
    matrix.transpose().matmul(matrix)
}

#[test]
fn geometry_uses_the_shared_algebraic_entry_points() {
    use tensorcrate::geometry::{Mat2, Vec2, Vec3};

    fn cross<'a, V>(a: &'a V, b: &'a V) -> <&'a V as Cross<&'a V>>::Output
    where
        &'a V: Cross<&'a V>,
    {
        a.cross(b)
    }

    let a = Vec3::new([1.0, 0.0, 0.0]);
    let b = Vec3::new([0.0, 1.0, 0.0]);
    assert_eq!(cross(&a, &b), Vec3::new([0.0, 0.0, 1.0]));
    assert_eq!(
        cross(&Vector::new(a.0), &Vector::new(b.0)).as_slice(),
        [0.0, 0.0, 1.0]
    );
    let rows = Matrix::from_rows([a.0, b.0]);
    let rhs = Matrix::from_rows([b.0, a.0]);
    assert_eq!(
        cross(&rows, &rhs).as_slice(),
        [0.0, 0.0, 1.0, 0.0, 0.0, -1.0]
    );
    let tensor = Tensor::from_vec(&[1, 2, 3], rows.as_slice().to_vec());
    let rhs = Tensor::from_vec(&[1, 2, 3], rhs.as_slice().to_vec());
    assert_eq!(
        cross(&tensor, &rhs).as_slice(),
        [0.0, 0.0, 1.0, 0.0, 0.0, -1.0]
    );
    assert_eq!(dot(&a, &b), 0.0);
    assert_eq!(Dotc::dotc(&a, &a), 1.0);

    let matrix = Mat2::new([[2.0, 1.0], [0.0, 4.0]]);
    let vector = Vec2::new([3.0, 2.0]);
    assert_eq!(matvec(&matrix, &vector), Vec2::new([8.0, 8.0]));
    assert_eq!(
        multiply(&matrix, &matrix),
        Mat2::new([[4.0, 6.0], [0.0, 16.0]])
    );
    assert_eq!(transpose(&matrix), Mat2::new([[2.0, 0.0], [1.0, 4.0]]));
    assert_eq!(Adjoint::adjoint(&matrix), transpose(&matrix));
    assert_eq!(Determinant::determinant(&matrix), 8.0);
    assert_eq!(
        multiply(&matrix, &Inverse::inverse(&matrix).unwrap()),
        Mat2::identity()
    );
    assert_eq!(matrix.0, [[2.0, 1.0], [0.0, 4.0]]);
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
    assert_eq!(gram(&a), Matrix::from_rows([[10, 14], [14, 20]]));
}

#[test]
fn method_syntax_borrows_its_receiver() {
    let a = Matrix::from_rows([[1.0, 2.0], [3.0, 4.0]]);
    let v = Vector::new([1.0, 1.0]);

    // Each call goes through the trait, and `a` and `v` stay usable after it.
    let product = MatMul::matmul(&a, &a);
    let image = MatVec::matvec(&a, &v);
    let flipped = Transpose::transpose(&a);
    assert_eq!(product, a.matmul(&a));
    assert_eq!(image, a.matvec(&v));
    assert_eq!(flipped, a.transpose());
    assert_eq!(Dot::dot(&v, &v), 2.0);
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
    assert_eq!(Dotc::dotc(&v, &v), Dual::new(5.0, 6.0));

    let tape = Tape::new();
    let recorded = tape.matrix(a.clone());
    let product = multiply(&recorded, &recorded);
    assert_eq!(product.value(), &a.matmul(&a));
    assert_eq!(transpose(&recorded).value(), &a.transpose());
    assert_eq!(Adjoint::adjoint(&recorded).value(), &a.transpose());
}

#[test]
fn sparse_values_share_matvec_and_transpose_traits() {
    let sparse = SparseMatrix::from_triplets(2, 3, [(0, 1, 2), (1, 2, 4)]);
    assert_eq!(matvec(&sparse, [1, 3, 5].as_slice()), vec![6, 20]);
    assert_eq!(transpose(&sparse).shape(), (3, 2));

    let csr = CsrMatrix::from_triplets(2, 2, [(0, 0, 2.0), (1, 0, 1.0), (1, 1, 3.0)]);
    let x = Vector::new([1.0, 2.0]);
    assert_eq!(matvec(&csr, &x), Vector::new([2.0, 7.0]));
    assert_eq!(transpose(&csr).to_dense(), csr.to_dense().transpose());
}

#[test]
fn conjugation_takes_scalars_by_value_and_tensors_by_reference() {
    type C = Complex<f64>;
    let z = C::new(1.0, 2.0);
    let w = C::new(3.0, -1.0);
    let expected = C::new(1.0, -7.0);

    // Every ownership combination of a scalar gives the same product.
    assert_eq!(z.dotc(w), expected);
    assert_eq!(z.dotc(&w), expected);
    assert_eq!((&z).dotc(w), expected);
    assert_eq!((&z).dotc(&w), expected);
    assert_eq!(z.adjoint(), z.conj());
    assert_eq!((&z).adjoint(), z.conj());

    let m = Matrix::from_rows([[z, w]]);
    let conjugated = m.conj();
    assert_eq!(conjugated.to_rows(), [[z.conj(), w.conj()]]);
    assert_eq!(m.adjoint().shape(), (2, 1));
    assert_eq!(m.dotc(&m), z.dotc(z) + w.dotc(w));

    let t = Tensor::from_vec(&[1, 1, 2], vec![z, w]);
    assert_eq!(t.conj().as_slice(), conjugated.as_slice());
    assert_eq!(t.adjoint().shape(), [1, 2, 1]);

    // Real values are their own conjugates, so `dotc` agrees with `dot`.
    assert_eq!(2.0f64.dotc(3.0), 6.0);
    let v = Vector::new([1.0, 2.0]);
    assert_eq!(v.dotc(&v), Dot::dot(&v, &v));
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn resident_values_implement_linear_algebra_traits() {
    use tensorcrate::tensors::{Host, Metal};

    let a = Matrix::from_rows([[1.0f32, 2.0], [3.0, 4.0]]);
    let v = Vector::new([1.0f32, -1.0]);
    let ga = a.to_backend::<Metal>();
    let gv = v.to_backend::<Metal>();

    assert_eq!(multiply(&ga, &ga).to_backend::<Host>(), a.matmul(&a));
    assert_eq!(transpose(&ga).to_backend::<Host>(), a.transpose());
    assert_eq!(matvec(&ga, &gv).to_backend::<Host>(), a.matvec(&v));
    assert_eq!(vecmat(&gv, &ga).to_backend::<Host>(), v.vecmat(&a));
    assert_eq!(dot(&gv, &gv), 2.0);
    assert_eq!(gram(&ga).to_backend::<Host>(), gram(&a));
}
