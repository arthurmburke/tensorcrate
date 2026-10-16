use tensorcrate::geometry::*;
use tensorcrate::math;
#[cfg(all(feature = "metal", target_os = "macos"))]
use tensorcrate::numbers::Coefficient;
use tensorcrate::numbers::Complex;
use tensorcrate::tensors::{Adjoint, Dotc, Matrix, Tensor, Vector};

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 2e-12 * (1.0 + b.abs()), "{a} != {b}");
}

#[test]
fn cross_products_have_right_handed_orientation_and_scalar_orthogonality() {
    let a = Vec3::new([2.0, -3.0, 5.0]);
    let b = Vec3::new([7.0, 11.0, -2.0]);
    let cross = a.cross(&b);
    close(cross.dot(&a), 0.0);
    close(cross.dot(&b), 0.0);
    assert_eq!(
        Vec3::new([1, 0, 0]).cross(&Vec3::new([0, 1, 0])),
        Vec3::new([0, 0, 1])
    );
    assert_eq!(a.cross(&b), -b.cross(&a));
    let dynamic = Vector::new(a.0).cross(&Vector::new(b.0));
    assert_eq!(dynamic.as_slice(), cross.0);
    let fused: Vector<f64> = math! {cross([2,-3,5],[7,11,-2])};
    let unfused: Vector<f64> = math! {fuse=false;cross([2,-3,5],[7,11,-2])};
    assert_eq!(fused, unfused);
    assert_eq!(fused.as_slice(), cross.0);
    let a = Tensor::from_vec(&[2, 1, 3], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let b = Tensor::from_vec(&[2, 1, 3], vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    assert_eq!(a.cross(&b).as_slice(), [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
    let empty = Tensor::<f64>::from_vec(&[0, 3], Vec::new());
    assert_eq!(empty.cross(&empty).len(), 0);
    assert!(std::panic::catch_unwind(|| Vector::new([1, 2]).cross(&Vector::new([3, 4]))).is_err());
}

#[test]
fn small_products_and_complex_adjoints_match_dynamic_algebra() {
    let a = Mat3::new([[1.0, 2.0, 3.0], [0.0, -1.0, 4.0], [5.0, 6.0, 0.0]]);
    let b = a.transpose();
    let x = Vec3::new([2.0, 3.0, -1.0]);
    let dense = Matrix::from_rows(a.0);
    assert_eq!(a.matvec(&x).0, dense.matvec(&Vector::new(x.0)).as_slice());
    assert_eq!(
        a.matmul(&b).0.to_vec(),
        dense
            .matmul(&Matrix::from_rows(b.0))
            .to_rows()
            .iter()
            .map(|r| [r[0], r[1], r[2]])
            .collect::<Vec<_>>()
    );
    close(a.determinant(), 31.0);
    type C = Complex<f64>;
    let a = SmallMatrix::new([[C::new(1.0, 2.0), C::new(3.0, -4.0)]]);
    assert_eq!(a.adjoint().0, [[C::new(1.0, -2.0)], [C::new(3.0, 4.0)]]);
    let v = Vec2::new([C::new(1.0, 2.0), C::new(3.0, -4.0)]);
    assert_eq!(v.dotc(&v), C::new(30.0, 0.0));
}

#[test]
fn small_lu_pivots_reuses_factors_and_handles_row_scaling() {
    let a = Mat3::new([[0.0, 2.0, 1.0], [1.0, -2.0, 0.0], [3.0, 1.0, 2.0]]);
    let expected = Vec3::new([2.0, -3.0, 4.0]);
    let rhs = a.matvec(&expected);
    let lu = a.lu().unwrap();
    for (a, b) in lu.solve(&rhs).unwrap().0.into_iter().zip(expected.0) {
        close(a, b);
    }
    close(lu.determinant(), a.determinant());
    let product = a.matmul(&lu.inverse().unwrap());
    for i in 0..3 {
        for j in 0..3 {
            close(product[(i, j)], if i == j { 1.0 } else { 0.0 });
        }
    }
    let scaled = Mat2::new([[1e-100, 2e-100], [3e100, 4e100]]);
    let expected = Vec2::new([2.0, -1.0]);
    let rhs = scaled.matvec(&expected);
    for (a, b) in scaled.solve(&rhs).unwrap().0.into_iter().zip(expected.0) {
        close(a, b);
    }
    assert!(Mat2::new([[1.0, 1.0], [1.0, 1.0]]).lu().is_err());
    assert!(
        Mat2::new([[1.0, 1.0], [1.0, 1.0 + f64::EPSILON]])
            .lu()
            .is_err()
    );
    assert!(Mat2::new([[f64::NAN, 0.0], [0.0, 1.0]]).lu().is_err());
    assert!(SmallLu::factor(Mat2::identity(), -1.0).is_err());
    close(Vec3::new([1e200, 1e200, 0.0]).norm() / 1e200, 2.0f64.sqrt());
    close(
        Vec3::new([1e308, 1e308, 0.0]).normalized().unwrap().norm(),
        1.0,
    );
    assert!(Vec3::<f64>::zeros().normalized().is_err());
}

#[test]
fn signed_jacobians_preserve_tangential_and_flux_pairings() {
    for sign in [-1.0, 1.0] {
        let map = Jacobian::new(Mat3::new([
            [2.0, 1.0, 0.0],
            [0.0, 3.0, 1.0],
            [0.0, 0.0, sign * 4.0],
        ]))
        .unwrap();
        let u = Vec3::new([1.0, 2.0, -3.0]);
        let v = Vec3::new([2.0, -1.0, 4.0]);
        close(map.determinant(), sign * 24.0);
        close(map.measure_scale(), 24.0);
        close(map.covariant_piola(&u).dot(&map.map_vector(&v)), u.dot(&v));
        close(
            map.contravariant_piola(&u).dot(&map.map_area_normal(&v)),
            u.dot(&v),
        );
        let mapped_cross = map.covariant_piola(&u).cross(&map.covariant_piola(&v));
        for (a, b) in mapped_cross.0.into_iter().zip(map.map_curl(&u.cross(&v)).0) {
            close(a, b);
        }
    }
    let tetra = AffineTetrahedron::from_vertices([
        Vec3::new([1.0, 2.0, 3.0]),
        Vec3::new([3.0, 2.0, 3.0]),
        Vec3::new([1.0, 5.0, 3.0]),
        Vec3::new([1.0, 2.0, -1.0]),
    ])
    .unwrap();
    let r = Vec3::new([0.1, 0.2, 0.3]);
    let point = tetra.map_point(r);
    for (a, b) in tetra.reference_point(point).0.into_iter().zip(r.0) {
        close(a, b);
    }
    let bary = tetra.barycentric(point);
    close(bary.iter().sum(), 1.0);
    close(bary[0], 0.4);
    close(tetra.measure(), 4.0);
    assert!(AffineTetrahedron::from_vertices([Vec3::new([0.0; 3]); 4]).is_err());
}

fn factorial(n: usize) -> f64 {
    (1..=n).fold(1.0, |s, n| s * n as f64)
}

#[test]
fn reference_rules_integrate_all_monomials_through_their_degree() {
    for order in 1..=8 {
        let rule = QuadratureRule::<f64, 1>::gauss_legendre(order).unwrap();
        for a in 0..=rule.degree() {
            close(
                rule.integrate(|x| x[0].powi(a as i32)),
                1.0 / (a + 1) as f64,
            );
        }
    }
    for degree in 1..=7 {
        let triangle = QuadratureRule::<f64, 2>::triangle(degree).unwrap();
        for a in 0..=degree {
            for b in 0..=degree - a {
                close(
                    triangle.integrate(|x| x[0].powi(a as i32) * x[1].powi(b as i32)),
                    factorial(a) * factorial(b) / factorial(a + b + 2),
                );
            }
        }
        let tetra = QuadratureRule::<f64, 3>::tetrahedron(degree).unwrap();
        for a in 0..=degree {
            for b in 0..=degree - a {
                for c in 0..=degree - a - b {
                    close(
                        tetra.integrate(|x| {
                            x[0].powi(a as i32) * x[1].powi(b as i32) * x[2].powi(c as i32)
                        }),
                        factorial(a) * factorial(b) * factorial(c) / factorial(a + b + c + 3),
                    );
                }
            }
        }
        assert!(tetra.weights().iter().all(|&w| w > 0.0));
    }
    assert!(QuadratureRule::<f64, 1>::gauss_legendre(0).is_err());
    assert!(QuadratureRule::<f64, 2>::triangle(usize::MAX).is_err());
    assert!(QuadratureRule::<f64, 3>::tetrahedron(30).is_err());
}

#[test]
fn mapped_and_surface_integrals_have_the_correct_measure_and_orientation() {
    let map = AffineTriangle::from_vertices([
        Vec2::new([1.0, 1.0]),
        Vec2::new([3.0, 1.0]),
        Vec2::new([1.0, 4.0]),
    ])
    .unwrap();
    let rule = QuadratureRule::<f64, 2>::triangle(2).unwrap();
    close(rule.integrate_mapped(&map, |_| 1.0), 3.0);
    close(rule.integrate_mapped(&map, |x| x[0]), 5.0);
    let face = Triangle3::new([
        Vec3::new([0.0, 0.0, 2.0]),
        Vec3::new([2.0, 0.0, 2.0]),
        Vec3::new([0.0, 3.0, 2.0]),
    ])
    .unwrap();
    assert_eq!(face.normal().0, [0.0, 0.0, 1.0]);
    close(face.area(), 3.0);
    close(face.integrate(&rule, |p| p[2]), 6.0);
    let c = rule.integrate(|_| Complex::new(2.0, 3.0));
    close(c.real, 1.0);
    close(c.im, 1.5);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_cross_products_match_host_and_macro_modes() {
    use tensorcrate::tensors::Metal;
    let a = Tensor::from_vec(
        &[2, 2, 3],
        (0..12).map(|i| i as f32 - 4.0).collect::<Vec<_>>(),
    );
    let b = Tensor::from_vec(
        &[2, 2, 3],
        (0..12).map(|i| 2.0 - i as f32).collect::<Vec<_>>(),
    );
    assert_eq!(
        a.cross(&b).as_slice(),
        a.to_backend::<Metal>()
            .cross(&b.to_backend::<Metal>())
            .as_slice()
    );
    let fused: Vector<f32, Metal> = math! {backend=Metal;dtype=f32;cross([1,2,3],[4,5,6])};
    let unfused: Vector<f32, Metal> =
        math! {backend=Metal;dtype=f32;fuse=false;cross([1,2,3],[4,5,6])};
    assert_eq!(fused.as_slice(), unfused.as_slice());
    assert_eq!(fused.as_slice(), [-3.0, 6.0, -3.0]);
    let half: Vector<tensorcrate::numbers::f16, Metal> =
        math! {backend=Metal;dtype=f16;cross([1,0,0],[0,1,0])};
    assert_eq!(half.as_slice()[2].to_f32(), 1.0);
    let bf: Vector<tensorcrate::numbers::bf16, Metal> =
        math! {backend=Metal;dtype=bf16;cross([1,0,0],[0,1,0])};
    assert_eq!(bf.as_slice()[2].to_f32(), 1.0);
    let a = Vector::new([
        Complex::new(1.0f32, 2.0),
        Complex::new(3.0, -1.0),
        Complex::new(-2.0, 1.0),
    ]);
    let b = Vector::new([
        Complex::new(4.0f32, -1.0),
        Complex::new(0.0, 2.0),
        Complex::new(1.0, 1.0),
    ]);
    for (&a, &b) in a.cross(&b).as_slice().iter().zip(
        a.to_backend::<Metal>()
            .cross(&b.to_backend::<Metal>())
            .as_slice(),
    ) {
        assert!((a - b).magnitude() < 1e-5);
    }
}
