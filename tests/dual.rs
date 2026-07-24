use rinterp::numbers::Dual;

#[test]
fn epsilon_squared_is_zero_but_epsilon_is_not() {
    // ε = 0 + 1ε
    let eps = Dual::new(0.0, 1.0);
    let zero = Dual::new(0.0, 0.0);
    assert_ne!(eps, zero); // ε ≠ 0
    assert_eq!(eps * eps, zero); // ε² = 0
}

#[test]
fn add_and_sub() {
    let x = Dual::new(2.0, 3.0); // 2 + 3ε
    let y = Dual::new(5.0, 7.0); // 5 + 7ε
    assert_eq!(x + y, Dual::new(7.0, 10.0));
    assert_eq!(y - x, Dual::new(3.0, 4.0));
}

#[test]
fn mul_drops_the_epsilon_squared_term() {
    // (2 + 3ε)(5 + 7ε) = 10 + (2·7 + 3·5)ε = 10 + 29ε
    let x = Dual::new(2.0, 3.0);
    let y = Dual::new(5.0, 7.0);
    assert_eq!(x * y, Dual::new(10.0, 29.0));
}

#[test]
fn rem_matches_forward_derivative() {
    // d/dx (x % 5) = 1 where defined: 7 % 5 = 2, derivative 1.
    let r = Dual::variable(7.0) % Dual::constant(5.0);
    assert_eq!(r, Dual::new(2.0, 1.0));

    // d/dx (7 % x) at x = 3 is -trunc(7/3) = -2.
    let r = Dual::constant(7.0) % Dual::variable(3.0);
    assert_eq!(r, Dual::new(1.0, -2.0));
}

/// Evaluate `f(x) = (x*x + 3*x) % 7` and its derivative in one pass by seeding
/// `x + 1ε`.
#[test]
fn automatic_differentiation_of_a_polynomial() {
    let x = Dual::variable(4.0_f64); // f'(x) accumulates in the ε part
    let three = Dual::constant(3.0);
    let seven = Dual::constant(7.0);

    let f = (x * x + three * x) % seven;

    // f(4) = (16 + 12) % 7 = 28 % 7 = 0.
    assert_eq!(f.real, 0.0);
    // f = x² + 3x (mod 7); f'(x) = 2x + 3 = 11 at x = 4 (the remainder shift is
    // locally constant).
    assert_eq!(f.dual, 11.0);
}

#[test]
fn works_for_any_dtype() {
    // Integer coefficients.
    let a = Dual::new(3i32, 1);
    let b = Dual::new(5i32, 2);
    assert_eq!(a * b, Dual::new(15, 11)); // 3·2 + 1·5 = 11
    assert_eq!(a + b, Dual::new(8, 3));
    assert_eq!((Dual::variable(17i64) % Dual::constant(5)), Dual::new(2, 1));

    // Unsigned and 8-bit types.
    let u = Dual::new(200u8, 3) + Dual::new(50u8, 4);
    assert_eq!(u, Dual::new(250, 7));

    // f32 as well as f64.
    let s = Dual::variable(2.0f32) * Dual::variable(2.0f32);
    assert_eq!(s, Dual::new(4.0, 4.0)); // (2+ε)² = 4 + 4ε
}

#[test]
fn display_reads_as_a_plus_b_epsilon() {
    assert_eq!(Dual::new(2, 3).to_string(), "2 + 3ε");
}
