use tensorcrate::numbers::{
    Arccos, Arcsin, Arctan, Complex, Cos, Cosh, Csc, Dual, Exp, Ln, Power, Sec, Sin, Sinh, Sqrt,
    Tan, Tanh,
};
use tensorcrate::tensors::{Analytic, Kernels, Matrix, Transcendental, Vector};

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

/// Assert a dual result equals `f(x)` in the real part and `f'(x)·b` in the ε
/// part.
#[track_caller]
fn dual_eq(d: Dual<f64>, real: f64, dual: f64) {
    assert!(
        close(d.real, real) && close(d.dual, dual),
        "{d}  expected ({real}, {dual})"
    );
}

// ---- forward-mode derivatives via dual numbers ------------------------------

#[test]
fn trig_derivatives() {
    // sin' = cos, cos' = -sin, tan' = sec²
    dual_eq(Dual::variable(0.0).sin(), 0.0, 1.0); // cos 0 = 1
    dual_eq(Dual::variable(0.0).cos(), 1.0, 0.0); // -sin 0 = 0
    dual_eq(Dual::variable(0.0).tan(), 0.0, 1.0); // sec² 0 = 1

    let x = 0.7_f64;
    dual_eq(Dual::variable(x).sin(), x.sin(), x.cos());
    dual_eq(Dual::variable(x).cos(), x.cos(), -x.sin());
    dual_eq(Dual::variable(x).tan(), x.tan(), 1.0 / (x.cos() * x.cos()));
}

#[test]
fn reciprocal_trig_derivatives() {
    let x = 0.9_f64;
    // csc' = -cos/sin², sec' = sin/cos²
    dual_eq(
        Dual::variable(x).csc(),
        1.0 / x.sin(),
        -x.cos() / (x.sin() * x.sin()),
    );
    dual_eq(
        Dual::variable(x).sec(),
        1.0 / x.cos(),
        x.sin() / (x.cos() * x.cos()),
    );
}

#[test]
fn inverse_trig_derivatives() {
    let x = 0.4_f64;
    dual_eq(
        Dual::variable(x).arcsin(),
        x.asin(),
        1.0 / (1.0 - x * x).sqrt(),
    );
    dual_eq(
        Dual::variable(x).arccos(),
        x.acos(),
        -1.0 / (1.0 - x * x).sqrt(),
    );
    dual_eq(Dual::variable(x).arctan(), x.atan(), 1.0 / (1.0 + x * x));
}

#[test]
fn exp_ln_and_hyperbolic_derivatives() {
    dual_eq(Dual::variable(1.0).exp(), 1.0_f64.exp(), 1.0_f64.exp()); // exp' = exp
    dual_eq(Dual::variable(2.0).ln(), 2.0_f64.ln(), 0.5); // ln' = 1/x
    let x = 0.6_f64;
    dual_eq(Dual::variable(x).sinh(), x.sinh(), x.cosh()); // sinh' = cosh
    dual_eq(Dual::variable(x).cosh(), x.cosh(), x.sinh()); // cosh' = sinh
    dual_eq(
        Dual::variable(x).tanh(),
        x.tanh(),
        1.0 - x.tanh() * x.tanh(),
    ); // tanh' = 1-tanh²
}

#[test]
fn power_rule_exponential_rule_and_x_to_the_x() {
    // d/dx x² = 2x → at 3, value 9, slope 6.
    dual_eq(Dual::variable(3.0).power(Dual::constant(2.0)), 9.0, 6.0);
    // d/dx 2^x = 2^x·ln2 → at 3, value 8, slope 8·ln2.
    dual_eq(
        Dual::constant(2.0).power(Dual::variable(3.0)),
        8.0,
        8.0 * 2.0_f64.ln(),
    );
    // d/dx x^x = x^x(ln x + 1) → at 2, value 4, slope 4(ln2+1).
    dual_eq(
        Dual::variable(2.0).power(Dual::variable(2.0)),
        4.0,
        4.0 * (2.0_f64.ln() + 1.0),
    );
}

#[test]
fn chain_rule_composition_matches_finite_difference() {
    // f(x) = exp(sin(x²)); differentiate at x = 1.3 in one pass.
    let f = |x: Dual<f64>| (x * x).sin().exp();
    let x = 1.3;
    let got = f(Dual::variable(x));
    assert!(close(got.real, (x * x).sin().exp()));

    let h = 1e-6;
    let fd = |x: f64| ((x * x).sin()).exp();
    let numeric = (fd(x + h) - fd(x - h)) / (2.0 * h);
    assert!(
        (got.dual - numeric).abs() < 1e-5,
        "AD {} vs FD {numeric}",
        got.dual
    );
}

// ---- the ops on the plain number types --------------------------------------

/// A function generic over the trait — proves the op is implemented for every
/// number type, floats keeping their type and integers widening to `f64`.
fn apply_exp<T: Exp>(x: T) -> T::Output {
    x.exp()
}

#[test]
fn every_number_type_implements_the_ops() {
    assert!(close(apply_exp(0.0f64), 1.0));
    assert!(close(apply_exp(0.0f32) as f64, 1.0));
    assert!(close(apply_exp(0i32), 1.0)); // Output = f64
    assert!(close(apply_exp(0u8), 1.0));
    assert!(close(apply_exp(0i64), 1.0));

    // Floats keep precision; Power on f64 returns f64.
    assert_eq!(2.0f64.power(5.0), 32.0);
    assert_eq!(<i32 as Power>::power(2, 5), 32.0f64); // integer widens to f64

    // `sqrt` is one of them: floats keep their type, integers widen.
    assert_eq!(9.0f64.sqrt(), 3.0);
    assert_eq!(<i32 as Sqrt>::sqrt(9), 3.0f64);
    // The principal complex root: √(-1) = i.
    let root = <Complex<f64> as Sqrt>::sqrt(Complex::new(-1.0, 0.0));
    assert!(close(root.real, 0.0) && close(root.im, 1.0));
    // d/dx √x = 1/(2√x) → at 4, value 2, slope 1/4.
    dual_eq(Dual::variable(4.0).sqrt(), 2.0, 0.25);
}

// ---- the ops applied elementwise to tensors ---------------------------------

#[test]
fn analytic_functions_map_over_vectors_and_matrices() {
    let v = Vector::new([0.0, 1.0, 2.0]);
    let exponentials = v.exp();
    assert!(
        exponentials
            .data()
            .iter()
            .zip([0.0, 1.0, 2.0])
            .all(|(&got, x)| close(got, x.exp()))
    );

    // The shape is kept, and every element is transformed.
    let m = Matrix::from_rows([[1.0, 4.0], [9.0, 16.0]]);
    let roots = m.sqrt();
    assert_eq!(roots.shape(), (2, 2));
    assert_eq!(roots.to_rows(), [[1.0, 2.0], [3.0, 4.0]]);

    // Every function in the set is present on both shapes.
    assert!(close(Vector::new([0.0]).sin().data()[0], 0.0));
    assert!(close(Vector::new([0.0]).cos().data()[0], 1.0));
    assert!(close(Vector::new([0.0]).tan().data()[0], 0.0));
    assert!(close(Vector::new([0.0]).sec().data()[0], 1.0));
    assert!(close(
        Vector::new([0.5]).csc().data()[0],
        1.0 / 0.5f64.sin()
    ));
    assert!(close(Vector::new([0.0]).arcsin().data()[0], 0.0));
    assert!(close(
        Vector::new([0.0]).arccos().data()[0],
        std::f64::consts::FRAC_PI_2
    ));
    assert!(close(Vector::new([0.0]).arctan().data()[0], 0.0));
    assert!(close(Vector::new([1.0]).ln().data()[0], 0.0));
    assert!(close(Vector::new([0.0]).sinh().data()[0], 0.0));
    assert!(close(Vector::new([0.0]).cosh().data()[0], 1.0));
    assert!(close(Vector::new([0.0]).tanh().data()[0], 0.0));
    assert!(close(Matrix::from_rows([[0.0]]).sin().to_rows()[0][0], 0.0));
}

#[test]
fn tensor_elements_keep_their_own_definitions() {
    // Integer elements widen to `f64`, exactly as a single integer does.
    let widened = Vector::new([0i32, 1]).exp();
    assert!(close(widened.data()[0], 1.0));
    assert!(close(widened.data()[1], 1.0f64.exp()));

    // Complex elements take the complex branch: exp(iπ) = −1.
    let rotated = Vector::new([Complex::new(0.0, std::f64::consts::PI)]).exp();
    assert!(close(rotated.data()[0].real, -1.0));
    assert!(close(rotated.data()[0].im, 0.0));

    // Dual elements differentiate, elementwise and in one pass.
    let differentiated = Vector::new([Dual::variable(2.0), Dual::variable(3.0)]).ln();
    assert!(close(differentiated.data()[0].real, 2.0f64.ln()));
    assert!(close(differentiated.data()[0].dual, 0.5)); // ln' = 1/x
    assert!(close(differentiated.data()[1].dual, 1.0 / 3.0));
}

/// The same generic function as `apply_exp` above, now over a tensor — the
/// point of implementing the scalar traits for `&Vector` and `&Matrix`.
#[test]
fn the_scalar_traits_accept_tensors() {
    let v = Vector::new([0.0, 0.0]);
    assert_eq!(apply_exp(&v).data(), [1.0, 1.0]);

    let m = Matrix::from_rows([[0.0, 0.0]]);
    assert_eq!(apply_exp(&m).to_rows(), [[1.0, 1.0]]);

    // The borrow is what the impl is on, so the tensor survives the call.
    assert_eq!(v.data(), [0.0, 0.0]);
}

#[test]
fn the_function_can_be_a_value_rather_than_a_name() {
    let v = Vector::new([0.0f32, 1.0]);
    for f in Analytic::ALL {
        // Same answer whichever way it is spelled.
        let by_value = v.analytic(f);
        let expected: Vec<f32> = v.data().iter().map(|&x| f.value(x)).collect();
        assert_eq!(by_value.data(), expected);
    }

    assert_eq!(v.analytic(Analytic::Exp).data(), v.exp().data());
    assert_eq!(
        Matrix::from_rows([[0.0f32, 1.0]])
            .analytic(Analytic::Sin)
            .to_rows(),
        Matrix::from_rows([[0.0f32, 1.0]]).sin().to_rows()
    );
}

/// Backend-generic code reaches the same operations through [`Transcendental`].
fn squash<B: Kernels>(v: &Vector<f32, B>) -> Vector<f32, B> {
    v.tanh()
}

#[test]
fn the_backend_generic_surface_agrees_with_the_inherent_methods() {
    let v = Vector::new([0.0f32, 1.0]);
    assert_eq!(squash(&v).data(), v.tanh().data());

    // Reached as a trait method on a concrete tensor, it is the same kernel.
    assert_eq!(
        Transcendental::sqrt(&Vector::new([4.0f32])).data(),
        Vector::new([4.0f32]).sqrt().data()
    );
}
