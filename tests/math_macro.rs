use tensorcrate::math;
use tensorcrate::numbers::{Complex, Dual};

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

/// The motivating example: complex arithmetic written as mathematics.
#[test]
fn complex_block_from_the_readme() {
    let z = math! {
        let x = 1 + 2i;
        let y = 1 - 2i;
        x * y
    };
    // (1+2i)(1-2i) = 1 + 4 = 5, imaginary parts cancel.
    assert_eq!(z, Complex::new(5.0, 0.0));
    assert_eq!(z.to_string(), "5+0i");
}

#[test]
fn the_result_is_a_concrete_rust_type() {
    // Not a dynamic value: this is a real `Complex<f64>`, checked by rustc.
    let z: Complex<f64> = math! { 1 + 2i };
    assert_eq!(z, Complex::new(1.0, 2.0));

    // A block with no imaginary part stays a plain f64.
    let r: f64 = math! { 2 + 3 * 4 };
    assert_eq!(r, 14.0);
}

#[test]
fn precedence_and_parentheses_follow_rust() {
    assert_eq!(math! { 2 + 3 * 4 }, 14.0);
    assert_eq!(math! { (2 + 3) * 4 }, 20.0);
    assert_eq!(math! { -2 + 3 }, 1.0);
    assert_eq!(math! { 10 / 4 }, 2.5); // real division, not integer
}

#[test]
fn reals_widen_into_complex_expressions() {
    // `3` is lifted to 3+0i so it can meet the imaginary literal.
    assert_eq!(math! { 3 + 1i }, Complex::new(3.0, 1.0));
    assert_eq!(math! { 2 * (1 + 1i) }, Complex::new(2.0, 2.0));
    // i^2 = -1
    assert_eq!(math! { 1i * 1i }, Complex::new(-1.0, 0.0));
}

#[test]
fn let_bindings_carry_their_types() {
    let z = math! {
        let a = 2 + 0i;   // Complex
        let b = 3;        // real, widened where it meets `a`
        let c = a * b;
        c + 1i
    };
    assert_eq!(z, Complex::new(6.0, 1.0));
}

#[test]
fn functions_dispatch_on_the_static_type() {
    // On a real, `exp` is the f64 one.
    assert!(close(math! { exp(0) }, 1.0));
    assert!(close(math! { sin(0) + cos(0) }, 1.0));

    // On a complex, Euler's identity: e^{iπ} = -1.
    let pi = std::f64::consts::PI;
    let z = math! { exp(pi * 1i) };
    assert!(close(z.real, -1.0) && close(z.im, 0.0), "{z}");

    // conj forces the complex algebra.
    assert_eq!(math! { conj(3 + 4i) }, Complex::new(3.0, -4.0));
}

#[test]
fn power_is_a_function_not_an_operator() {
    // `^` is deliberately rejected: Rust parses it looser than `*` and `+`,
    // which would silently misread `a * b ^ 2`.
    assert!(close(math! { pow(2, 10) }, 1024.0));
    // i^2 = -1 through the Power trait.
    let z = math! { pow(1i, 2) };
    assert!(close(z.real, -1.0) && close(z.im, 0.0), "{z}");
}

#[test]
fn dual_numbers_differentiate() {
    // Seed x = 3 + 1ε and evaluate x*x: the ε part is the derivative 2x = 6.
    // ε is spelled `d`: rustc lexes any `e` after digits as a float exponent.
    let d: Dual<f64> = math! {
        let x = 3 + 1d;
        x * x
    };
    assert_eq!(d.real, 9.0);
    assert_eq!(d.dual, 6.0);
}

#[test]
fn outer_rust_bindings_are_usable() {
    let radius = 2.0_f64;
    let pi = std::f64::consts::PI;
    let area = math! { pi * pow(radius, 2) };
    assert!(close(area, std::f64::consts::PI * 4.0));

    let z = math! { radius + 1i };
    assert_eq!(z, Complex::new(2.0, 1.0));
}

// ---- mixing the two extensions ----------------------------------------------

#[test]
fn reals_mix_freely_with_duals() {
    // A plain real widens to `x + 0ε` wherever it meets a dual.
    let d: Dual<f64> = math! { 5 + 1d };
    assert_eq!((d.real, d.dual), (5.0, 1.0));

    let d: Dual<f64> = math! {
        let x = 3 + 1d;   // seed
        2 * x + 4         // 2x + 4 -> value 10, derivative 2
    };
    assert_eq!((d.real, d.dual), (10.0, 2.0));

    // Real bindings and dual bindings in the same expression.
    let d: Dual<f64> = math! {
        let k = 7;
        let x = 1 + 1d;
        k * x
    };
    assert_eq!((d.real, d.dual), (7.0, 7.0));
}

#[test]
fn a_dual_may_have_complex_coefficients() {
    // Mixing `i` and `d` gives Dual<Complex<f64>> rather than an error.
    let z: Dual<Complex<f64>> = math! { 1 + 2i + 3d };
    assert_eq!(z.real, Complex::new(1.0, 2.0)); // 1 + 2i
    assert_eq!(z.dual, Complex::new(3.0, 0.0)); // 3ε

    // The ε coefficient itself can be imaginary: (1d) * (1i) = i·ε
    let z: Dual<Complex<f64>> = math! { 1d * 1i };
    assert_eq!(z.real, Complex::new(0.0, 0.0));
    assert_eq!(z.dual, Complex::new(0.0, 1.0));
}

#[test]
fn complex_dual_arithmetic_differentiates() {
    // f(x) = x^2 at the complex point x = (1+i), seeded with ε.
    let z: Dual<Complex<f64>> = math! {
        let x = 1 + 1i + 1d;
        x * x
    };
    // value: (1+i)^2 = 2i
    assert_eq!(z.real, Complex::new(0.0, 2.0));
    // derivative: 2x = 2 + 2i
    assert_eq!(z.dual, Complex::new(2.0, 2.0));
}

#[test]
fn a_real_dual_binding_widens_into_a_complex_one() {
    // `x` is Dual<f64>; meeting `1i` lifts its coefficients to complex.
    let z: Dual<Complex<f64>> = math! {
        let x = 2 + 1d;
        x + 1i
    };
    assert_eq!(z.real, Complex::new(2.0, 1.0));
    assert_eq!(z.dual, Complex::new(1.0, 0.0));
}

#[test]
fn complex_and_dual_unification_is_operand_order_independent() {
    let left: Dual<Complex<f64>> = math! {
        let z = 2 + 3i;
        let d = 5 + 7d;
        z + d
    };
    let right: Dual<Complex<f64>> = math! {
        let z = 2 + 3i;
        let d = 5 + 7d;
        d + z
    };

    let expected = Dual::new(Complex::new(7.0, 3.0), Complex::new(7.0, 0.0));
    assert_eq!(left, expected);
    assert_eq!(right, expected);

    // This also exercises widening through nested expressions rather than only
    // through a literal or a single binding.
    let product: Dual<Complex<f64>> = math! {
        let z = 1 + 2i;
        let d = 3 + 4d;
        (z + 2) * (d - 1)
    };
    assert_eq!(product.real, Complex::new(6.0, 4.0));
    assert_eq!(product.dual, Complex::new(12.0, 8.0));
}

#[test]
fn analytic_functions_and_power_preserve_complex_duals() {
    // exp(i + ε) = exp(i) + exp(i)ε.
    let exponential: Dual<Complex<f64>> = math! { exp(1i + 1d) };
    let exp_i = Complex::new(1.0_f64.cos(), 1.0_f64.sin());
    assert!(close(exponential.real.real, exp_i.real));
    assert!(close(exponential.real.im, exp_i.im));
    assert!(close(exponential.dual.real, exp_i.real));
    assert!(close(exponential.dual.im, exp_i.im));

    let squared: Dual<Complex<f64>> = math! { pow(1 + 1i + 1d, 2) };
    assert!(close(squared.real.real, 0.0));
    assert!(close(squared.real.im, 2.0));
    assert!(close(squared.dual.real, 2.0));
    assert!(close(squared.dual.im, 2.0));

    let conjugated: Dual<Complex<f64>> = math! { conj(1 + 2i + 3d * (1 + 1i)) };
    assert_eq!(conjugated.real, Complex::new(1.0, -2.0));
    assert_eq!(conjugated.dual, Complex::new(3.0, -3.0));
}

// ---- tensors ----------------------------------------------------------------

use tensorcrate::tensors::{Matrix, Vector};

#[test]
fn vector_and_matrix_literals() {
    let v: Vector<f64, 3> = math! { [1, 2, 3] };
    assert_eq!(v.data(), &[1.0, 2.0, 3.0]);

    let m: Matrix<f64, 2, 2> = math! { [[1, 2], [3, 4]] };
    assert_eq!(m.data(), &[[1.0, 2.0], [3.0, 4.0]]);
}

#[test]
fn elementwise_and_broadcast() {
    // Elementwise addition of two vectors.
    let v: Vector<f64, 3> = math! { [1, 2, 3] + [10, 20, 30] };
    assert_eq!(v.data(), &[11.0, 22.0, 33.0]);

    // A scalar broadcasts over a tensor, either side.
    let v: Vector<f64, 3> = math! { 2 * [1, 2, 3] };
    assert_eq!(v.data(), &[2.0, 4.0, 6.0]);
    let v: Vector<f64, 3> = math! { [1, 2, 3] + 100 };
    assert_eq!(v.data(), &[101.0, 102.0, 103.0]);
}

#[test]
fn matmul_dot_and_det() {
    let product: Matrix<f64, 2, 2> = math! { matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]) };
    assert_eq!(product.data(), &[[19.0, 22.0], [43.0, 50.0]]);

    // dot collapses two vectors to a scalar — the whole block is still fallible.
    let d: f64 = math! { dot([1, 2, 3], [4, 5, 6]) };
    assert_eq!(d, 32.0);

    let det: f64 = math! { det([[1, 2], [3, 4]]) };
    assert_eq!(det, -2.0);
}

#[test]
fn symbolic_products_are_deduced_from_operand_shapes() {
    let matrix_product: Matrix<f64, 2, 2> = math! {
        let a = [[1, 2, 3], [4, 5, 6]];
        let b = [[7, 8], [9, 10], [11, 12]];
        a @ b
    };

    let sin_t = math! {
        let a = [
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5]
        ];
        cos(a)
    };

    assert_eq!(matrix_product.data(), &[[58.0, 64.0], [139.0, 154.0]]);

    let matrix_vector: Vector<f64, 2> = math! {
        let a = [[1, 2, 3], [4, 5, 6]];
        let v = [1, 2, 3];
        a @ v
    };
    assert_eq!(matrix_vector.data(), &[14.0, 32.0]);

    let row_matrix: Vector<f64, 3> = math! {
        let v = [1, 2];
        let b = [[1, 2, 3], [4, 5, 6]];
        v @ b
    };
    assert_eq!(row_matrix.data(), &[9.0, 12.0, 15.0]);

    let dot: f64 = math! { [1, 2, 3] * [4, 5, 6] };
    assert_eq!(dot, 32.0);

    // `@` has multiplicative precedence and chains left-to-right.
    let chained: Matrix<f64, 2, 2> = math! {
        let a = [[1, 2], [3, 4]];
        let b = [[2, 0], [0, 2]];
        let c = [[1, 1], [0, 1]];
        a @ b @ c + [[1, 0], [0, 1]]
    };
    assert_eq!(chained.data(), &[[3.0, 6.0], [6.0, 15.0]]);
}

#[test]
fn matrix_vector_products_and_tensor_power() {
    let mv: Vector<f64, 2> = math! { matmul([[1, 2, 3], [4, 5, 6]], [1, 2, 3]) };
    assert_eq!(mv.data(), &[14.0, 32.0]);

    let vm: Vector<f64, 3> = math! { matmul([1, 2], [[1, 2, 3], [4, 5, 6]]) };
    assert_eq!(vm.data(), &[9.0, 12.0, 15.0]);

    let squares: Vector<f64, 3> = math! { pow([1, 2, 3], 2) };
    assert_eq!(squares.data(), &[1.0, 4.0, 9.0]);
}

#[test]
fn a_full_linear_algebra_block() {
    // A · A⁻¹ = I, written as mathematics.
    let identity: Matrix<f64, 2, 2> = math! {
        let a = [[4, 7], [2, 6]];
        matmul(a, inv(a))
    }
    .unwrap();
    assert!((identity.get(0, 0).unwrap() - 1.0).abs() < 1e-9);
    assert!(identity.get(0, 1).unwrap().abs() < 1e-9);
    assert!((identity.get(1, 1).unwrap() - 1.0).abs() < 1e-9);
}

#[test]
fn transpose_and_scaling() {
    let t: Matrix<f64, 3, 2> = math! { transpose([[1, 2, 3], [4, 5, 6]]) };
    assert_eq!(t.data(), &[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);
}

#[test]
fn tensors_of_complex_numbers() {
    // The element type is inferred from the literals inside.
    let v: Vector<Complex<f64>, 2> = math! { [1 + 1i, 2 - 1i] };
    assert_eq!(v.get(0), Some(&Complex::new(1.0, 1.0)));
    assert_eq!(v.get(1), Some(&Complex::new(2.0, -1.0)));

    // Scalars widen to the complex element type.
    let v: Vector<Complex<f64>, 2> = math! { [1, 2] + [0 + 1i, 0 - 1i] };
    assert_eq!(v.get(0), Some(&Complex::new(1.0, 1.0)));
}

#[test]
fn tensors_differentiate_through_matmul() {
    // Seed a matrix entry with ε and read the derivative out of the product.
    let squared: Matrix<Dual<f64>, 2, 2> = math! {
        let x = 2 + 1d;
        let a = [[x, 1], [0, 1]];
        matmul(a, a)
    };
    // (0,0) is x² -> value 4, derivative 2x = 4.
    let top_left = squared.get(0, 0).unwrap();
    assert_eq!((top_left.real, top_left.dual), (4.0, 4.0));
}

#[test]
fn tensors_unify_real_complex_and_dual_elements() {
    let literal: Vector<Dual<Complex<f64>>, 3> = math! { [1, 2 + 3i, 4 + 5d] };
    assert_eq!(
        literal.get(0),
        Some(&Dual::constant(Complex::new(1.0, 0.0)))
    );
    assert_eq!(
        literal.get(1),
        Some(&Dual::constant(Complex::new(2.0, 3.0)))
    );
    assert_eq!(
        literal.get(2),
        Some(&Dual::new(Complex::new(4.0, 0.0), Complex::new(5.0, 0.0)))
    );

    // Tensor/tensor and scalar/tensor widening both lift every coefficient.
    let sum: Vector<Dual<Complex<f64>>, 2> = math! { [1 + 1i, 2] + [3 + 4d, 5 + 6d] };
    assert_eq!(
        sum.get(0),
        Some(&Dual::new(Complex::new(4.0, 1.0), Complex::new(4.0, 0.0)))
    );

    let broadcast: Vector<Dual<Complex<f64>>, 2> = math! { (1 + 2i) * [3 + 1d, 4 + 2d] };
    assert_eq!(
        broadcast.get(0),
        Some(&Dual::new(Complex::new(3.0, 6.0), Complex::new(1.0, 2.0)))
    );

    let conjugated: Vector<Dual<Complex<f64>>, 1> = math! { conj([1 + 2i + 3d * (1 + 1i)]) };
    assert_eq!(
        conjugated.get(0),
        Some(&Dual::new(Complex::new(1.0, -2.0), Complex::new(3.0, -3.0)))
    );
}

#[test]
fn linear_algebra_infers_complex_dual_results() {
    let dot_product: Dual<Complex<f64>> = math! { dot([1 + 1i, 2], [3 + 1d, 4 + 2d]) };
    assert_eq!(dot_product.real, Complex::new(11.0, 3.0));
    assert_eq!(dot_product.dual, Complex::new(5.0, 1.0));

    let product: Matrix<Dual<Complex<f64>>, 2, 2> = math! {
        matmul(
            [[1 + 1i, 0], [0, 1]],
            [[2 + 1d, 0], [0, 3 + 2d]]
        )
    };
    assert_eq!(
        product.get(0, 0),
        Some(&Dual::new(Complex::new(2.0, 2.0), Complex::new(1.0, 1.0)))
    );
    assert_eq!(
        product.get(1, 1),
        Some(&Dual::new(Complex::new(3.0, 0.0), Complex::new(2.0, 0.0)))
    );
}

#[test]
fn elementwise_functions_map_over_tensors() {
    let v: Vector<f64, 3> = math! { exp([0, 0, 0]) };
    assert_eq!(v.data(), &[1.0, 1.0, 1.0]);

    let sine: Matrix<f64, 2, 2> = math! { sin([[0, 0], [0, 0]]) };
    assert_eq!(sine.data(), &[[0.0, 0.0], [0.0, 0.0]]);

    let cosine: Vector<f64, 3> = math! { cos([0, 0, 0]) };
    assert_eq!(cosine.data(), &[1.0, 1.0, 1.0]);
}
