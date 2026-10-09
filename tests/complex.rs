use tensorcrate::numbers::{
    Arccos, Arcsin, Arctan, Complex, Conj, Cos, Cosh, Csc, Exp, Ln, Power, Sec, Sin, Sinh, Tan,
    Tanh,
};

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

#[track_caller]
fn eq(z: Complex<f64>, real: f64, im: f64) {
    assert!(
        close(z.real, real) && close(z.im, im),
        "{z} expected {real}+{im}i"
    );
}

/// Compare against the defining identity rather than a restatement of the impl.
#[track_caller]
fn approx(z: Complex<f64>, w: Complex<f64>) {
    assert!(close(z.real, w.real) && close(z.im, w.im), "{z} vs {w}");
}

#[test]
fn i_squared_is_minus_one() {
    let i = Complex::<f64>::i();
    eq(i * i, -1.0, 0.0);
}

#[test]
fn arithmetic() {
    let a = Complex::new(3.0, 4.0);
    let b = Complex::new(1.0, -2.0);
    eq(a + b, 4.0, 2.0);
    eq(a - b, 2.0, 6.0);
    // (3+4i)(1-2i) = 3 - 6i + 4i + 8 = 11 - 2i
    eq(a * b, 11.0, -2.0);
}

#[test]
fn division_uses_the_conjugate() {
    // (3+4i)/(1-2i) = (3+4i)(1+2i)/5 = (3 + 6i + 4i - 8)/5 = (-5 + 10i)/5 = -1 + 2i
    eq(Complex::new(3.0, 4.0) / Complex::new(1.0, -2.0), -1.0, 2.0);
    // Dividing by i rotates by -90°: (1+0i)/i = -i
    eq(Complex::new(1.0, 0.0) / Complex::i(), 0.0, -1.0);
    // z / z == 1 for a few values.
    for z in [
        Complex::new(3.0, 4.0),
        Complex::new(-2.0, 0.5),
        Complex::new(0.0, 7.0),
    ] {
        eq(z / z, 1.0, 0.0);
    }
    // (z/w)*w == z
    let (z, w) = (Complex::new(5.0, -3.0), Complex::new(-1.0, 4.0));
    approx((z / w) * w, z);
}

#[test]
fn magnitude_conjugate_and_sqrt() {
    let z = Complex::new(3.0, -4.0);
    assert!(close(z.abs(), 5.0));
    assert!(close(z.norm_sqr(), 25.0));
    eq(z.conj(), 3.0, 4.0);
    // z * conj(z) == |z|^2, purely real.
    eq(z * z.conj(), 25.0, 0.0);
    // sqrt(z)^2 == z, including for a negative real (the classic branch case).
    for z in [
        Complex::new(3.0, 4.0),
        Complex::new(-1.0, 0.0),
        Complex::new(0.0, -2.0),
        Complex::new(-5.0, -12.0),
    ] {
        approx(z.sqrt() * z.sqrt(), z);
    }
    eq(Complex::new(-1.0, 0.0).sqrt(), 0.0, 1.0); // principal √-1 = i
}

#[test]
fn remainder_subtracts_a_whole_multiple() {
    let (z, w) = (Complex::new(7.0, 5.0), Complex::new(2.0, 1.0));
    let r = z % w;
    // z - r must be an exact Gaussian-integer multiple of w, and |r| <= |w|/√2.
    let multiple = (z - r) / w;
    assert!(close(multiple.real, multiple.real.round()));
    assert!(close(multiple.im, multiple.im.round()));
    assert!(
        r.abs() <= w.abs() / 2.0_f64.sqrt() + 1e-9,
        "|r|={}",
        r.abs()
    );
}

#[test]
fn trig_matches_the_defining_identities() {
    let z = Complex::new(0.7, -0.4);
    // sin² + cos² = 1
    approx(
        z.sin() * z.sin() + z.cos() * z.cos(),
        Complex::constant(1.0),
    );
    // tan = sin/cos, csc = 1/sin, sec = 1/cos
    approx(z.tan(), z.sin() / z.cos());
    approx(z.csc(), Complex::constant(1.0) / z.sin());
    approx(z.sec(), Complex::constant(1.0) / z.cos());
    // Real inputs agree with the real-valued functions.
    eq(Complex::new(0.9, 0.0).cos(), 0.9_f64.cos(), 0.0);
    eq(Complex::new(0.9, 0.0).sin(), 0.9_f64.sin(), 0.0);
}

#[test]
fn hyperbolic_and_exponential() {
    let z = Complex::new(0.6, 1.1);
    approx(z.tanh(), z.sinh() / z.cosh());
    // cosh² − sinh² = 1
    approx(
        z.cosh() * z.cosh() - z.sinh() * z.sinh(),
        Complex::constant(1.0),
    );
    // Euler: e^(iπ) = −1
    eq(Complex::new(0.0, std::f64::consts::PI).exp(), -1.0, 0.0);
    // exp(ln z) == z
    approx(z.ln().exp(), z);
}

#[test]
fn inverse_trig_round_trips() {
    let z = Complex::new(0.3, 0.25);
    approx(z.arcsin().sin(), z);
    approx(z.arccos().cos(), z);
    approx(z.arctan().tan(), z);
}

#[test]
fn power() {
    // i^2 = -1 and z^2 = z*z
    let i = Complex::<f64>::i();
    approx(i.power(Complex::constant(2.0)), Complex::constant(-1.0));
    let z = Complex::new(1.5, -0.5);
    approx(z.power(Complex::constant(2.0)), z * z);
    approx(z.power(Complex::constant(0.5)), z.sqrt());
}

#[test]
fn gaussian_integers_and_display() {
    // Integer coefficients still work for +, -, * (Gaussian integers).
    let a = Complex::new(3i64, 2);
    let b = Complex::new(1i64, -1);
    assert_eq!(a * b, Complex::new(5, -1)); // 3-3i+2i+2 = 5-i
    assert_eq!(a + b, Complex::new(4, 1));

    assert_eq!(Complex::new(3, 4).to_string(), "3+4i");
    assert_eq!(Complex::new(3, -4).to_string(), "3-4i");
    assert_eq!(Complex::<i32>::i().to_string(), "0+1i");
}
