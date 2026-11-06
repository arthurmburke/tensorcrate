//! `FixedPoint` through the generic tensor algebra: vectors, matrices, small matrices and
//! persistence, all exact where a float would not be.

use tensorcrate::geometry::{SmallMatrix, SmallVector};
use tensorcrate::numbers::{FixedPoint, Rounding};
use tensorcrate::prelude::*;
use tensorcrate::tensors::{Matrix, Vector};

type Micros = FixedPoint<6>;

fn m(text: &str) -> Micros {
    text.parse().unwrap()
}

fn v(values: &[&str]) -> Vector<Micros> {
    Vector::new(values.iter().map(|s| m(s)).collect::<Vec<_>>())
}

#[test]
fn vector_reductions_are_exact() {
    // Ten dimes: a float sum gives 0.9999999999999999.
    let dimes = Vector::repeat(10, m("0.1"));
    assert_eq!(dimes.sum(), Micros::ONE);
    assert_eq!(Dot::dot(&v(&["0.5", "0.25"]), &v(&["0.4", "4"])), m("1.2"));
    assert_eq!(v(&["0.1", "0.2"]).scale(m("3")), v(&["0.3", "0.6"]));
    assert_eq!(v(&["0.3", "0.9"]).min(&v(&["0.5", "0.1"])), v(&["0.3", "0.1"]));
}

#[test]
fn matrices_multiply_and_invert() {
    let a = Matrix::from_rows([[m("2"), m("0")], [m("0"), m("4")]]);
    let x = v(&["0.5", "0.25"]);
    assert_eq!(MatVec::matvec(&a, &x), v(&["1", "1"]));
    assert_eq!(a.inverse().unwrap(), Matrix::from_rows([[m("0.5"), m("0")], [m("0"), m("0.25")]]));
    assert_eq!(a.determinant(), m("8"));
    assert_eq!(MatMul::matmul(&a, &a), Matrix::from_rows([[m("4"), m("0")], [m("0"), m("16")]]));
}

#[test]
fn small_matrices_work_on_decimals() {
    let a: SmallMatrix<Micros, 2, 2> = [[m("1.5"), m("0")], [m("0"), m("2")]].into();
    let x: SmallVector<Micros, 2> = [m("0.1"), m("0.2")].into();
    let y = a.matvec(&x);
    assert_eq!((y[0], y[1]), (m("0.15"), m("0.4")));
    assert_eq!(a.determinant(), m("3"));
}

#[test]
fn tensors_of_decimals_persist_with_their_precision() {
    let prices = v(&["0.56", "0.000001", "-12.5"]);
    let mut bytes = Vec::new();
    prices.write_to(&mut bytes).unwrap();
    assert_eq!(Vector::<Micros>::read_from(bytes.as_slice()).unwrap(), prices);
    // The precision is part of the tag: cents do not load as millionths.
    assert!(Vector::<FixedPoint<2>>::read_from(bytes.as_slice()).is_err());
}

#[test]
fn rounding_rules_are_explicit_across_precisions() {
    // 7% of 100 contracts at 50¢ each side: exactly $1.75, already on the cent.
    let fee: FixedPoint<2> = FixedPoint::<2>::from_raw(7)
        .mul_to::<2, 4>("100".parse().unwrap(), Rounding::Nearest)
        .mul_to::<6, 10>(m("0.5"), Rounding::Nearest)
        .mul_to::<6, 2>(m("0.5"), Rounding::Ceil);
    assert_eq!(fee, "1.75".parse().unwrap());
}
