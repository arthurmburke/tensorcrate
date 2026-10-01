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

// ---- elementwise powers -----------------------------------------------------

#[test]
fn a_power_applies_to_every_element() {
    let v = Vector::new([1.0, 2.0, 3.0, 4.0]);
    assert_eq!(v.pow(2.0).data(), [1.0, 4.0, 9.0, 16.0]);
    assert_eq!(v.pow(0.5).data(), [1.0, 2.0f64.sqrt(), 3.0f64.sqrt(), 2.0]);
    assert_eq!(v.pow(0.0).data(), [1.0; 4]);
    assert_eq!(v.pow(1.0).data(), v.data());

    let m = Matrix::from_rows([[1.0, 2.0], [3.0, 4.0]]);
    assert_eq!(m.pow(3.0).to_rows(), [[1.0, 8.0], [27.0, 64.0]]);

    // The exponent may vary per element.
    let bases = Vector::new([2.0, 3.0, 4.0]);
    let exponents = Vector::new([10.0, 2.0, 0.5]);
    assert_eq!(bases.pow_elementwise(&exponents).data(), [1024.0, 9.0, 2.0]);

    let base_matrix = Matrix::from_rows([[2.0, 3.0]]);
    let exponent_matrix = Matrix::from_rows([[5.0, 3.0]]);
    assert_eq!(
        base_matrix.pow_elementwise(&exponent_matrix).to_rows(),
        [[32.0, 27.0]]
    );
}

#[test]
fn the_fast_exponents_are_correctly_rounded() {
    // Specializing an exponent has to be an optimization rather than a second
    // definition. `x⁰` and `x¹` are exactly what `powf` answers; the other
    // three are single IEEE operations, so they are *correctly rounded* where
    // `powf` is merely accurate to under an ulp — they may therefore differ
    // from it in the last bit, and when they do it is `powf` that is off.
    let mut values: Vec<f64> = (-2_000..2_000).map(|i| i as f64 * 0.017).collect();
    values.extend([
        0.0,
        -0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::MIN_POSITIVE,
        f64::MAX,
        -1.0,
    ]);
    let v = Vector::new(values.clone());

    // Exact against the operation each one stands for.
    let exact: [(f64, fn(f64) -> f64); 5] = [
        (0.0, |_| 1.0),
        (1.0, |x| x),
        (2.0, |x| x * x),
        (0.5, |x| if x == f64::NEG_INFINITY { f64::INFINITY } else { x.sqrt() + 0.0 }),
        (-1.0, |x| 1.0 / x),
    ];
    for (exponent, expected) in exact {
        let fast = v.pow(exponent);
        for (&actual, &value) in fast.data().iter().zip(&values) {
            let want = expected(value);
            assert_eq!(
                actual.to_bits(),
                want.to_bits(),
                "{value}^{exponent}: {actual} against {want}"
            );
        }
    }

    // And within one ulp of `powf`, which is the accuracy `powf` itself claims.
    for exponent in [0.0f64, 1.0, 2.0, 0.5, -1.0] {
        for (&actual, &value) in v.pow(exponent).data().iter().zip(&values) {
            let reference = value.powf(exponent);
            if actual.is_nan() || reference.is_nan() {
                assert_eq!(actual.is_nan(), reference.is_nan());
                continue;
            }
            let gap = (actual.to_bits() as i128 - reference.to_bits() as i128).abs();
            assert!(
                gap <= 1,
                "{value}^{exponent}: {actual} is {gap} ulps from {reference}"
            );
        }
    }

    // The sign of zero is the one place `sqrt` and `powf` genuinely disagree,
    // and the fast path follows `powf`.
    assert_eq!(
        Vector::new([-0.0f64]).pow(0.5).data()[0].to_bits(),
        (0.0f64).to_bits()
    );

    // Single precision has its own specialization.
    let single: Vec<f32> = values.iter().map(|&x| x as f32).collect();
    let v = Vector::new(single.clone());
    for (&actual, &value) in v.pow(2.0f32).data().iter().zip(&single) {
        assert_eq!(actual.to_bits(), (value * value).to_bits());
    }

    // An exponent with no fast path goes through `powf` untouched.
    let v = Vector::new([1.7f64, -2.5, 9.0]);
    for (&actual, &value) in v.pow(3.0).data().iter().zip(v.data()) {
        assert_eq!(actual.to_bits(), value.powf(3.0).to_bits());
    }
}

#[test]
fn a_power_carries_the_element_type_with_it() {
    // Integers widen to `f64`, exactly as `Power` does for a single number.
    let counts = Vector::new([2i32, 3, 4]);
    assert_eq!(counts.pow(3).data(), [8.0f64, 27.0, 64.0]);

    // A dual tensor comes back holding the derivative: d/dx x² = 2x.
    let x = Vector::new([Dual::variable(3.0), Dual::variable(5.0)]);
    let squared = x.pow(Dual::constant(2.0));
    assert!(close(squared.data()[0].real, 9.0) && close(squared.data()[0].dual, 6.0));
    assert!(close(squared.data()[1].real, 25.0) && close(squared.data()[1].dual, 10.0));

    // And a complex tensor follows the complex definition.
    let z = Vector::new([Complex::new(0.0, 1.0)]);
    let squared = z.pow(Complex::constant(2.0));
    assert!(close(squared.data()[0].real, -1.0) && close(squared.data()[0].im, 0.0));
}

#[test]
fn a_power_is_reachable_through_the_scalar_trait() {
    // A function written once over `Power` takes a number or a tensor, since
    // the tensor references implement the same trait the numbers do.
    fn square<T: Power<f64>>(x: T) -> T::Output {
        x.power(2.0)
    }

    assert_eq!(square(3.0f64), 9.0);
    assert_eq!(square(&Vector::new([3.0f64, 4.0])).data(), [9.0, 16.0]);
    assert_eq!(
        square(&Matrix::from_rows([[3.0f64]])).to_rows(),
        [[9.0f64]]
    );

    // Both tensor operands, and a scalar base with a tensor exponent.
    let bases = Vector::new([2.0f64, 3.0]);
    let exponents = Vector::new([3.0f64, 2.0]);
    assert_eq!(Power::power(&bases, &exponents).data(), [8.0, 9.0]);
    assert_eq!(Power::power(2.0f64, &exponents).data(), [8.0, 4.0]);
    assert_eq!(
        Power::power(2.0f64, &Matrix::from_rows([[3.0f64, 4.0]])).to_rows(),
        [[8.0, 16.0]]
    );
}

/// Backend-generic code reaches powers through [`Transcendental`] too.
fn energy<B: Kernels>(v: &Vector<f32, B>) -> Vector<f32, B> {
    v.pow(2.0)
}

#[test]
fn the_backend_generic_surface_covers_powers() {
    let v = Vector::new([2.0f32, 3.0]);
    assert_eq!(energy(&v).data(), [4.0, 9.0]);
    assert_eq!(
        Transcendental::pow_elementwise(&v, &Vector::new([3.0f32, 2.0])).data(),
        [8.0, 9.0]
    );
    assert_eq!(
        Transcendental::pow(&Matrix::from_rows([[2.0f32, 3.0]]), 2.0).to_rows(),
        [[4.0, 9.0]]
    );
}

#[test]
#[should_panic(expected = "pow: vector lengths differ, 3 and 2")]
fn an_elementwise_power_checks_the_shapes() {
    let _ = Vector::new([1.0, 2.0, 3.0]).pow_elementwise(&Vector::new([1.0, 2.0]));
}

#[test]
#[should_panic(expected = "pow: matrix shapes differ")]
fn an_elementwise_matrix_power_checks_the_shapes() {
    let _ = Matrix::from_rows([[1.0, 2.0]]).pow_elementwise(&Matrix::from_rows([[1.0], [2.0]]));
}
