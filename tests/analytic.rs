use rinterp::numbers::{
    Arccos, Arcsin, Arctan, Cos, Cosh, Csc, Dual, Exp, Ln, Power, Sec, Sin, Sinh, Tan, Tanh,
};

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
}
