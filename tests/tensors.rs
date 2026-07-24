//! Const-generic vector and matrix operations.

use rinterp::errors::Error;
use rinterp::numbers::{Complex, Dual};
use rinterp::tensors::{Matrix, Vector};

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

#[test]
fn vectors_are_statically_sized() {
    let u: Vector<i64, 3> = Vector::new([1, 2, 3]);
    let v: Vector<i64, 3> = Vector::new([4, 5, 6]);
    assert_eq!(u + v, Vector::new([5, 7, 9]));
    assert_eq!(u.dot(&v), 32);
    assert_eq!(u.scale(2), Vector::new([2, 4, 6]));
}

#[test]
fn matrices_are_statically_shaped() {
    let a: Matrix<i32, 2, 3> = Matrix::from_rows([[1, 2, 3], [4, 5, 6]]);
    let b: Matrix<i32, 3, 2> = Matrix::from_rows([[7, 8], [9, 10], [11, 12]]);
    let product: Matrix<i32, 2, 2> = a.matmul(&b);
    assert_eq!(product, Matrix::from_rows([[58, 64], [139, 154]]));
    assert_eq!(a.transpose(), Matrix::from_rows([[1, 4], [2, 5], [3, 6]]));

    let column: Vector<i32, 3> = Vector::new([1, 2, 3]);
    assert_eq!(a.matvec(&column), Vector::new([14, 32]));
    let row: Vector<i32, 2> = Vector::new([1, 2]);
    assert_eq!(row.vecmat(&a), Vector::new([9, 12, 15]));
}

#[test]
fn determinant_and_inverse() {
    let a: Matrix<f64, 2, 2> = Matrix::from_rows([[4.0, 7.0], [2.0, 6.0]]);
    assert!(close(a.determinant(), 10.0));
    let identity = a.matmul(&a.inverse().unwrap());
    assert!(close(*identity.get(0, 0).unwrap(), 1.0));
    assert!(close(*identity.get(0, 1).unwrap(), 0.0));
    assert!(close(*identity.get(1, 1).unwrap(), 1.0));

    let singular = Matrix::from_rows([[1.0, 2.0], [2.0, 4.0]]);
    assert_eq!(singular.inverse(), Err(Error::Singular));
}

#[test]
fn complex_and_dual_elements_work() {
    let i = Complex::new(0.0, 1.0);
    let z = Complex::new(0.0, 0.0);
    let m: Matrix<Complex<f64>, 2, 2> = Matrix::from_rows([[i, z], [z, i]]);
    assert_eq!(m.matmul(&m).get(0, 0), Some(&Complex::new(-1.0, 0.0)));

    let x = Dual::variable(2.0);
    let one = Dual::constant(1.0);
    let zero = Dual::constant(0.0);
    let a: Matrix<Dual<f64>, 2, 2> = Matrix::from_rows([[x, one], [zero, one]]);
    let squared = a.matmul(&a);
    assert_eq!(
        (
            squared.get(0, 0).unwrap().real,
            squared.get(0, 0).unwrap().dual
        ),
        (4.0, 4.0)
    );
}

#[test]
fn constructors_and_map_preserve_shape() {
    let zeros: Matrix<i32, 2, 3> = Matrix::zeros();
    assert_eq!(zeros.data(), &[[0, 0, 0], [0, 0, 0]]);
    let identity: Matrix<f64, 3, 3> = Matrix::identity();
    assert_eq!(
        identity.data(),
        &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    );

    let real: Vector<f64, 2> = Vector::new([1.0, 2.0]);
    let lifted: Vector<Complex<f64>, 2> = real.map(|&x| Complex::constant(x));
    assert_eq!(lifted.get(1), Some(&Complex::new(2.0, 0.0)));
}
