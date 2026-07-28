//! Reverse-mode automatic differentiation.
//!
//! Forward mode is the oracle here: it is already checked against the scalar
//! `Dual` algebra and against finite differences, so a rule that disagrees
//! between the two modes is wrong in one of them. Every operation is compared
//! that way, then the pieces forward mode cannot express — `%`, and gradient
//! accumulation across a shared subexpression — are checked directly.

use rinterp::numbers::Dual;
use rinterp::tensors::dual::{DualMatrix, DualVector, gradient, gradient_wrt_matrix};
use rinterp::tensors::{
    Analytic, BinaryOp, Host, Kernels, Matrix, MatrixVar, ScalarVar, Tape, Vector, VectorVar,
};

#[cfg(all(feature = "metal", target_os = "macos"))]
use rinterp::tensors::Metal;

const TOLERANCE: f32 = 2e-3;

fn close(actual: f32, expected: f32) -> bool {
    (actual - expected).abs() <= TOLERANCE * (1.0 + expected.abs())
}

fn assert_slice_close(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(close(a, e), "{what}: at {index}, {a} vs {e}");
    }
}

/// Inside (0, 1): clear of zero, of one, and of the poles of `tan` and `csc`.
fn value(index: usize) -> f32 {
    0.15 + (index % 5) as f32 * 0.17
}

fn vector<const N: usize>() -> Vector<f32, N> {
    Vector::new(std::array::from_fn(value))
}

fn other_vector<const N: usize>() -> Vector<f32, N> {
    Vector::new(std::array::from_fn(|i| 1.3 + (i % 3) as f32 * 0.4))
}

fn matrix<const R: usize, const C: usize>() -> Matrix<f32, R, C> {
    Matrix::from_rows(std::array::from_fn(|row| {
        std::array::from_fn(|col| value(row * C + col + row))
    }))
}

fn other_matrix<const R: usize, const C: usize>() -> Matrix<f32, R, C> {
    Matrix::from_rows(std::array::from_fn(|row| {
        std::array::from_fn(|col| 1.3 + ((row + col) % 3) as f32 * 0.4)
    }))
}

/// One backward pass over `f`, returning the gradient at `at`.
fn reverse_gradient<const N: usize>(
    at: Vector<f32, N>,
    f: impl for<'t> Fn(&VectorVar<'t, N>) -> ScalarVar<'t>,
) -> Vector<f32, N> {
    let tape = Tape::new();
    let x = tape.vector(at);
    f(&x).backward();
    x.grad()
}

// ---- every analytic function ------------------------------------------------

#[test]
fn every_analytic_function_agrees_with_forward_mode() {
    let at = vector::<6>();
    for f in Analytic::ALL {
        // Σ f(x), whose gradient is f'(x) elementwise.
        let reverse = reverse_gradient(at, |x| x.analytic(f).sum());
        let forward = gradient(&at, |x| x.analytic(f).sum());
        assert_slice_close(
            reverse.as_slice(),
            forward.as_slice(),
            &format!("{f:?} gradient"),
        );

        // And against the derivative table directly.
        for (index, &derivative) in reverse.as_slice().iter().enumerate() {
            assert!(
                close(derivative, f.derivative(value(index))),
                "{f:?} at {index}: {derivative} vs {}",
                f.derivative(value(index))
            );
        }
    }
}

#[test]
fn analytic_functions_compose_through_the_chain_rule() {
    // f(x) = Σ tanh(exp(sin(x))): three nested rules in one backward pass.
    let at = vector::<5>();
    let reverse = reverse_gradient(at, |x| x.sin().exp().tanh().sum());
    let forward = gradient(&at, |x| x.sin().exp().tanh().sum());
    assert_slice_close(reverse.as_slice(), forward.as_slice(), "nested chain rule");
}

#[test]
fn scalar_analytic_functions_agree_with_the_dual_scalars() {
    use rinterp::numbers::{Exp, Sin};

    // The scalar `Dual` path is the oracle: evaluating on the seed `x + 1·ε`
    // carries the derivative in the ε part.
    let point = 0.6f32;
    let tape = Tape::<Host>::new();
    let x = tape.scalar(point);
    let y = x.sin().exp();
    y.backward();

    let oracle = Exp::exp(Sin::sin(Dual::variable(point)));
    assert!(
        close(*y.value(), oracle.real),
        "{} vs {}",
        y.value(),
        oracle.real
    );
    assert!(
        close(x.grad(), oracle.dual),
        "{} vs {}",
        x.grad(),
        oracle.dual
    );
}

// ---- every binary operation -------------------------------------------------

#[test]
fn binary_operations_on_vectors_agree_with_forward_mode() {
    let (a, b) = (vector::<7>(), other_vector::<7>());

    for (op, label) in [
        (BinaryOp::Add, "add"),
        (BinaryOp::Sub, "sub"),
        (BinaryOp::Mul, "mul"),
        (BinaryOp::Div, "div"),
    ] {
        // Differentiate with respect to the left operand, holding the right one.
        let reverse = reverse_gradient(a, |x| {
            let tape = x.tape();
            let right = tape.vector(b);
            match op {
                BinaryOp::Add => (x + &right).sum(),
                BinaryOp::Sub => (x - &right).sum(),
                BinaryOp::Mul => (x * &right).sum(),
                BinaryOp::Div => (x / &right).sum(),
                BinaryOp::Rem => unreachable!(),
            }
        });
        let forward = gradient(&a, |x| {
            let right = DualVector::constant(b);
            match op {
                BinaryOp::Add => (x + &right).sum(),
                BinaryOp::Sub => (x - &right).sum(),
                BinaryOp::Mul => (x * &right).sum(),
                BinaryOp::Div => (x / &right).sum(),
                BinaryOp::Rem => unreachable!(),
            }
        });
        assert_slice_close(
            reverse.as_slice(),
            forward.as_slice(),
            &format!("{label} lhs"),
        );

        // And with respect to the right operand.
        let reverse = reverse_gradient(b, |x| {
            let tape = x.tape();
            let left = tape.vector(a);
            match op {
                BinaryOp::Add => (&left + x).sum(),
                BinaryOp::Sub => (&left - x).sum(),
                BinaryOp::Mul => (&left * x).sum(),
                BinaryOp::Div => (&left / x).sum(),
                BinaryOp::Rem => unreachable!(),
            }
        });
        let forward = gradient(&b, |x| {
            let left = DualVector::constant(a);
            match op {
                BinaryOp::Add => (&left + x).sum(),
                BinaryOp::Sub => (&left - x).sum(),
                BinaryOp::Mul => (&left * x).sum(),
                BinaryOp::Div => (&left / x).sum(),
                BinaryOp::Rem => unreachable!(),
            }
        });
        assert_slice_close(
            reverse.as_slice(),
            forward.as_slice(),
            &format!("{label} rhs"),
        );
    }
}

#[test]
fn remainder_follows_its_piecewise_rule() {
    // Forward mode does not differentiate `%`, so this one is checked against the
    // rule itself: c = a − trunc(a/b)·b, so ∂c/∂a = 1 and ∂c/∂b = −trunc(a/b).
    let (a, b) = (other_vector::<6>(), vector::<6>());

    let tape = Tape::new();
    let left = tape.vector(a);
    let right = tape.vector(b);
    (&left % &right).sum().backward();

    assert_eq!(left.grad().to_array(), [1.0; 6]);
    for index in 0..6 {
        let quotient = (a.data()[index] / b.data()[index]).trunc();
        assert!(
            close(right.grad().as_slice()[index], -quotient),
            "∂(a%b)/∂b at {index}: {} vs {}",
            right.grad().as_slice()[index],
            -quotient
        );
    }

    // Central differences away from a jump, as a second opinion.
    let step = 1e-3f32;
    let perturbed = |delta: f32| -> f32 {
        let mut values = *b.data();
        values[0] += delta;
        let shifted = Vector::new(values);
        (0..6).map(|i| a.data()[i] % shifted.data()[i]).sum()
    };
    let numeric = (perturbed(step) - perturbed(-step)) / (2.0 * step);
    assert!(
        (right.grad().as_slice()[0] - numeric).abs() < 5e-3,
        "{} vs finite difference {numeric}",
        right.grad().as_slice()[0]
    );
}

#[test]
fn scalar_binary_operations_match_hand_derivatives() {
    let tape = Tape::<Host>::new();
    let a = tape.scalar(3.0);
    let b = tape.scalar(2.0);

    let sum = a.add(&b);
    let difference = a.sub(&b);
    let product = a.mul(&b);
    let quotient = a.div(&b);
    let remainder = a.rem(&b);

    assert_eq!(*sum.value(), 5.0);
    assert_eq!(*difference.value(), 1.0);
    assert_eq!(*product.value(), 6.0);
    assert_eq!(*quotient.value(), 1.5);
    assert_eq!(*remainder.value(), 1.0);

    // Seed each output in turn, clearing between passes.
    for (output, da, db) in [
        (&sum, 1.0, 1.0),
        (&difference, 1.0, -1.0),
        (&product, 2.0, 3.0),
        (&quotient, 0.5, -0.75),
        (&remainder, 1.0, -1.0),
    ] {
        tape.zero_grad();
        output.backward();
        assert!(close(a.grad(), da), "∂/∂a: {} vs {da}", a.grad());
        assert!(close(b.grad(), db), "∂/∂b: {} vs {db}", b.grad());
    }
}

// ---- products and reductions ------------------------------------------------

#[test]
fn products_agree_with_forward_mode() {
    let a = matrix::<3, 4>();
    let x = vector::<4>();

    // ∂/∂A of ‖A·x‖², against the forward-mode driver.
    let tape = Tape::new();
    let matrix_var = tape.matrix(a);
    let vector_var = tape.vector(x);
    let mapped = matrix_var.matvec(&vector_var);
    mapped.dot(&mapped).backward();

    let forward = gradient_wrt_matrix(&a, |m| {
        let mapped = m.matvec(&DualVector::constant(x));
        mapped.dot(&mapped)
    });
    assert_slice_close(
        matrix_var.grad().as_slice(),
        forward.as_slice(),
        "matvec/dot gradient with respect to the matrix",
    );

    // The same pass also produced ∂/∂x = 2Aᵀ(Ax), which forward mode would need
    // a separate sweep for.
    let forward_x = gradient(&x, |v| {
        let mapped = DualMatrix::constant(a).matvec(v);
        mapped.dot(&mapped)
    });
    assert_slice_close(
        vector_var.grad().as_slice(),
        forward_x.as_slice(),
        "the vector gradient from the same pass",
    );
}

#[test]
fn matmul_gradients_match_the_closed_form() {
    // f(A, B) = Σ (A·B), so ∂f/∂A = 1·Bᵀ (rows of column sums) and ∂f/∂B = Aᵀ·1.
    let a = matrix::<2, 3>();
    let b = matrix::<3, 4>();

    let tape = Tape::new();
    let left = tape.matrix(a);
    let right = tape.matrix(b);
    left.matmul(&right).sum().backward();

    for row in 0..2 {
        for col in 0..3 {
            let expected: f32 = (0..4).map(|k| b.data()[col][k]).sum();
            assert!(
                close(left.grad().data()[row][col], expected),
                "∂/∂A[{row},{col}]: {} vs {expected}",
                left.grad().data()[row][col]
            );
        }
    }
    for row in 0..3 {
        for col in 0..4 {
            let expected: f32 = (0..2).map(|k| a.data()[k][row]).sum();
            assert!(
                close(right.grad().data()[row][col], expected),
                "∂/∂B[{row},{col}]: {} vs {expected}",
                right.grad().data()[row][col]
            );
        }
    }
}

#[test]
fn fused_multiply_adds_differentiate_the_addend() {
    let a = matrix::<3, 4>();
    let x = vector::<4>();
    let bias = vector::<3>();
    let tape = Tape::new();
    let matrix_var = tape.matrix(a);
    let vector_var = tape.vector(x);
    let bias_var = tape.vector(bias);
    matrix_var
        .matvec_add(&vector_var, &bias_var)
        .sum()
        .backward();

    assert_eq!(bias_var.grad().to_array(), [1.0; 3]);
    for row in 0..3 {
        assert_eq!(matrix_var.grad().data()[row], *x.data());
    }
    for col in 0..4 {
        let expected: f32 = (0..3).map(|row| a.data()[row][col]).sum();
        assert!(close(vector_var.grad().data()[col], expected));
    }

    let b = matrix::<4, 2>();
    let addend = matrix::<3, 2>();
    let tape = Tape::new();
    let left = tape.matrix(a);
    let right = tape.matrix(b);
    let addend_var = tape.matrix(addend);
    left.matmul_add(&right, &addend_var).sum().backward();
    assert_eq!(addend_var.grad().to_rows(), [[1.0; 2]; 3]);
}

#[test]
fn vecmat_and_transpose_and_reshapes_round_trip_their_adjoints() {
    let a = matrix::<3, 4>();
    let x = vector::<3>();

    // xᵀ·A summed: ∂/∂x is the row sums of A, ∂/∂A is x broadcast across columns.
    let tape = Tape::new();
    let matrix_var = tape.matrix(a);
    let vector_var = tape.vector(x);
    vector_var.vecmat(&matrix_var).sum().backward();

    for index in 0..3 {
        let expected: f32 = (0..4).map(|col| a.data()[index][col]).sum();
        assert!(close(vector_var.grad().as_slice()[index], expected));
    }
    for row in 0..3 {
        for col in 0..4 {
            assert!(close(matrix_var.grad().data()[row][col], x.data()[row]));
        }
    }

    // Transposing twice, flattening, and the row/column views are all identities
    // as far as the gradient is concerned.
    let tape = Tape::new();
    let m = tape.matrix(a);
    m.transpose().transpose().sum().backward();
    assert_eq!(m.grad().to_rows(), [[1.0f32; 4]; 3]);

    let tape = Tape::new();
    let m = tape.matrix(a);
    m.flattened().sum().backward();
    assert_eq!(m.grad().to_rows(), [[1.0f32; 4]; 3]);

    let tape = Tape::new();
    let v = tape.vector(x);
    v.into_row().sum().backward();
    assert_eq!(v.grad().to_array(), [1.0; 3]);

    let tape = Tape::new();
    let v = tape.vector(x);
    v.into_column().transpose().sum().backward();
    assert_eq!(v.grad().to_array(), [1.0; 3]);
}

#[test]
fn a_differentiable_scalar_can_be_broadcast_across_a_tensor() {
    // f(s) = Σ (s · x) = s · Σx, so ∂f/∂s = Σx.
    let x = vector::<5>();
    let tape = Tape::new();
    let scale = tape.scalar(2.0);
    let vector_var = tape.vector(x);
    (&scale.expand::<5>() * &vector_var).sum().backward();

    let total: f32 = x.as_slice().iter().sum();
    assert!(close(scale.grad(), total), "{} vs {total}", scale.grad());
    assert_eq!(vector_var.grad().to_array(), [2.0; 5]);

    // The matrix form sums over every element.
    let m = matrix::<2, 3>();
    let tape = Tape::new();
    let offset = tape.scalar(1.0);
    let matrix_var = tape.matrix(m);
    (&offset.expand_matrix::<2, 3>() + &matrix_var)
        .sum()
        .backward();
    assert!(close(offset.grad(), 6.0));
}

// ---- accumulation semantics -------------------------------------------------

// The reference operators are the only ones a non-`Copy` `Var` has.
#[allow(clippy::op_ref)]
#[test]
fn a_value_used_twice_accumulates_both_contributions() {
    // f(x) = Σ (x ⊙ x) has gradient 2x, which only comes out right if both edges
    // into `x` contribute.
    let at = vector::<4>();
    let tape = Tape::new();
    let x = tape.vector(at);
    (&x * &x).sum().backward();
    for index in 0..4 {
        assert!(close(x.grad().as_slice()[index], 2.0 * at.data()[index]));
    }

    // A shared subexpression: f = Σ(y + y·y) with y = sin(x).
    let tape = Tape::new();
    let x = tape.vector(at);
    let y = x.sin();
    (&y + &(&y * &y)).sum().backward();
    let forward = gradient(&at, |v| {
        let y = v.sin();
        (&y + &(&y * &y)).sum()
    });
    assert_slice_close(
        x.grad().as_slice(),
        forward.as_slice(),
        "shared subexpression",
    );
}

#[test]
fn each_backward_pass_is_independent_of_the_last() {
    let at = vector::<3>();
    let tape = Tape::new();
    let x = tape.vector(at);
    let loss = x.sum();

    loss.backward();
    assert_eq!(x.grad().to_array(), [1.0; 3]);

    // Running it again recomputes rather than doubling — the adjoints a pass is
    // about to write are cleared first.
    loss.backward();
    assert_eq!(x.grad().to_array(), [1.0; 3]);

    tape.zero_grad();
    assert!(!x.has_grad());
    assert_eq!(x.grad().to_array(), [0.0; 3]);

    // Two outputs do not pile up; add them and differentiate once.
    let scaled = x.scale(3.0).sum();
    (&loss + &scaled).backward();
    assert_eq!(x.grad().to_array(), [4.0; 3]);
}

#[test]
fn a_backward_pass_clears_gradients_on_later_nodes() {
    let tape = Tape::<Host>::new();
    let x = tape.scalar(2.0);
    let earlier = x.scale(3.0);
    let later = earlier.scale(4.0);

    later.backward();
    assert!(later.has_grad());

    earlier.backward();
    assert!(!later.has_grad());
    assert_eq!(later.grad(), 0.0);
    assert_eq!(x.grad(), 3.0);
}

#[test]
#[should_panic(expected = "reverse-mode operands belong to different tapes")]
fn operands_from_different_tapes_are_rejected() {
    let left_tape = Tape::<Host>::new();
    let right_tape = Tape::<Host>::new();
    let left = left_tape.scalar(1.0);
    let right = right_tape.scalar(2.0);
    let _ = &left + &right;
}

#[test]
fn an_unused_input_has_a_zero_gradient_and_seeding_a_tensor_projects() {
    let tape = Tape::new();
    let used = tape.vector(vector::<3>());
    let unused = tape.vector(other_vector::<3>());
    used.sum().backward();
    assert!(!unused.has_grad());
    assert_eq!(unused.grad().to_array(), [0.0; 3]);

    // Seeding a non-scalar output projects along the seed: with e₁, only the
    // first component's derivatives come back.
    let tape = Tape::new();
    let x = tape.vector(vector::<3>());
    x.sin().backward_with(Vector::new([1.0, 0.0, 0.0]));
    assert!(close(x.grad().as_slice()[0], value(0).cos()));
    assert_eq!(x.grad().as_slice()[1], 0.0);
}

// ---- one backward pass versus many forward ones ------------------------------

#[test]
fn a_single_pass_reproduces_the_whole_matrix_gradient() {
    // The point of reverse mode: `gradient_wrt_matrix` runs R·C forward passes to
    // get this, and one backward pass has to agree with all of them.
    let a = matrix::<4, 5>();
    let x = vector::<5>();

    let tape = Tape::new();
    let matrix_var = tape.matrix(a);
    let mapped = matrix_var.matvec(&tape.vector(x)).tanh();
    mapped.dot(&mapped).backward();

    let forward = gradient_wrt_matrix(&a, |m| {
        let mapped = m.matvec(&DualVector::constant(x)).tanh();
        mapped.dot(&mapped)
    });
    assert_slice_close(
        matrix_var.grad().as_slice(),
        forward.as_slice(),
        "20 forward passes versus one backward pass",
    );
}

// ---- the two backends --------------------------------------------------------

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn reverse_mode_agrees_between_the_backends() {
    fn loss<'t, B: Kernels>(
        a: &MatrixVar<'t, 6, 6, B>,
        b: &MatrixVar<'t, 6, 6, B>,
        x: &VectorVar<'t, 6, B>,
    ) -> ScalarVar<'t, B> {
        let product = a.matmul(b).tanh();
        let mapped = product.matvec(x).exp();
        (&mapped * &mapped).sum().add(&product.frobenius_dot(a))
    }

    let (a, b, x) = (matrix::<6, 6>(), other_matrix::<6, 6>(), vector::<6>());

    let host_tape = Tape::<Host>::new();
    let (ha, hb, hx) = (
        host_tape.matrix(a),
        host_tape.matrix(b),
        host_tape.vector(x),
    );
    loss(&ha, &hb, &hx).backward();

    let gpu_tape = Tape::<Metal>::new();
    let (ga, gb, gx) = (
        gpu_tape.matrix(a.to_backend::<Metal>()),
        gpu_tape.matrix(b.to_backend::<Metal>()),
        gpu_tape.vector(x.to_backend::<Metal>()),
    );
    loss(&ga, &gb, &gx).backward();

    assert_slice_close(ga.grad().as_slice(), ha.grad().as_slice(), "∂loss/∂A");
    assert_slice_close(gb.grad().as_slice(), hb.grad().as_slice(), "∂loss/∂B");
    assert_slice_close(gx.grad().as_slice(), hx.grad().as_slice(), "∂loss/∂x");

    // The gradients are computed in shared memory, not shipped back and forth.
    if a.to_backend::<Metal>().is_device_resident() {
        assert!(ga.grad().is_device_resident(), "∂loss/∂A stays resident");
        assert!(gx.grad().is_device_resident(), "∂loss/∂x stays resident");
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn every_analytic_function_agrees_between_the_backends_in_reverse() {
    let at = vector::<8>();
    for f in Analytic::ALL {
        let host_tape = Tape::<Host>::new();
        let x = host_tape.vector(at);
        x.analytic(f).sum().backward();

        let gpu_tape = Tape::<Metal>::new();
        let gpu_x = gpu_tape.vector(at.to_backend::<Metal>());
        gpu_x.analytic(f).sum().backward();

        assert_slice_close(
            gpu_x.grad().as_slice(),
            x.grad().as_slice(),
            &format!("{f:?} gradient"),
        );
    }
}

#[test]
fn binary_operations_on_matrices_agree_with_forward_mode() {
    let (a, b) = (matrix::<3, 4>(), other_matrix::<3, 4>());

    for (op, label) in [
        (BinaryOp::Add, "add"),
        (BinaryOp::Sub, "sub"),
        (BinaryOp::Mul, "mul"),
        (BinaryOp::Div, "div"),
    ] {
        let tape = Tape::new();
        let left = tape.matrix(a);
        let right = tape.matrix(b);
        match op {
            BinaryOp::Add => (&left + &right).sum(),
            BinaryOp::Sub => (&left - &right).sum(),
            BinaryOp::Mul => (&left * &right).sum(),
            BinaryOp::Div => (&left / &right).sum(),
            BinaryOp::Rem => unreachable!(),
        }
        .backward();

        let forward_left = gradient_wrt_matrix(&a, |m| {
            let other = DualMatrix::constant(b);
            match op {
                BinaryOp::Add => (m + &other).sum(),
                BinaryOp::Sub => (m - &other).sum(),
                BinaryOp::Mul => (m * &other).sum(),
                BinaryOp::Div => (m / &other).sum(),
                BinaryOp::Rem => unreachable!(),
            }
        });
        let forward_right = gradient_wrt_matrix(&b, |m| {
            let other = DualMatrix::constant(a);
            match op {
                BinaryOp::Add => (&other + m).sum(),
                BinaryOp::Sub => (&other - m).sum(),
                BinaryOp::Mul => (&other * m).sum(),
                BinaryOp::Div => (&other / m).sum(),
                BinaryOp::Rem => unreachable!(),
            }
        });
        assert_slice_close(
            left.grad().as_slice(),
            forward_left.as_slice(),
            &format!("matrix {label} lhs"),
        );
        assert_slice_close(
            right.grad().as_slice(),
            forward_right.as_slice(),
            &format!("matrix {label} rhs"),
        );
    }

    // `%`, which forward mode does not differentiate, against its own rule.
    let tape = Tape::new();
    let left = tape.matrix(b);
    let right = tape.matrix(a);
    (&left % &right).sum().backward();
    assert_eq!(left.grad().to_rows(), [[1.0f32; 4]; 3]);
    for row in 0..3 {
        for col in 0..4 {
            let quotient = (b.data()[row][col] / a.data()[row][col]).trunc();
            assert!(close(right.grad().data()[row][col], -quotient));
        }
    }
}

#[test]
fn sin_of_a_matrix_product_has_the_closed_form_gradient() {
    // Y = sin(A·B), the case where the output is a matrix rather than a scalar.
    let a = matrix::<3, 4>();
    let b = matrix::<4, 2>();
    let z = a.matmul(&b);
    let cosine = z.map(|&v| v.cos());

    // (a) Reduced to a scalar loss: one pass gives ∂/∂A = cos(A·B)·Bᵀ, shaped
    // like A, and ∂/∂B = Aᵀ·cos(A·B) at the same time.
    let tape = Tape::new();
    let left = tape.matrix(a);
    let right = tape.matrix(b);
    left.matmul(&right).sin().sum().backward();

    assert_slice_close(
        left.grad().as_slice(),
        cosine.matmul(&b.transpose()).as_slice(),
        "∂Σsin(AB)/∂A",
    );
    assert_slice_close(
        right.grad().as_slice(),
        a.transpose().matmul(&cosine).as_slice(),
        "∂Σsin(AB)/∂B",
    );

    // (b) Without reducing: seed the matrix output directly. A one-hot seed picks
    // one entry of Y, so ∂Y[1][0]/∂A is cos(Z[1][0])·B[:,0] on row 1 and zero
    // elsewhere — one row of the 6×12 Jacobian per pass.
    let tape = Tape::new();
    let left = tape.matrix(a);
    let right = tape.matrix(b);
    let mapped = left.matmul(&right).sin();

    let mut seed = [[0.0f32; 2]; 3];
    seed[1][0] = 1.0;
    mapped.backward_with(Matrix::from_rows(seed));

    for row in 0..3 {
        for col in 0..4 {
            let expected = if row == 1 {
                cosine.data()[1][0] * b.data()[col][0]
            } else {
                0.0
            };
            assert!(
                close(left.grad().data()[row][col], expected),
                "∂Y[1][0]/∂A[{row},{col}]: {} vs {expected}",
                left.grad().data()[row][col]
            );
        }
    }

    // (c) An arbitrary seed projects: ⟨S, ∂Y/∂A⟩ = (S ⊙ cos(Z))·Bᵀ.
    let tape = Tape::new();
    let left = tape.matrix(a);
    let right = tape.matrix(b);
    let projection = other_matrix::<3, 2>();
    left.matmul(&right).sin().backward_with(projection);
    assert_slice_close(
        left.grad().as_slice(),
        (projection * cosine).matmul(&b.transpose()).as_slice(),
        "⟨S, ∂Y/∂A⟩",
    );
}

// ---- full Jacobians ---------------------------------------------------------

#[test]
fn the_two_modes_produce_the_same_jacobian() {
    use rinterp::tensors::dual::jacobian as forward_jacobian;
    use rinterp::tensors::tape::jacobian as reverse_jacobian;

    // f(x) = tanh(M·(x ⊙ x)): 4 inputs, 3 outputs, so reverse needs 3 passes
    // where forward needs 4 — and they must agree to the last useful digit.
    let at = vector::<4>();
    let m = matrix::<3, 4>();

    let reverse = reverse_jacobian(&at, |x| {
        let tape = x.tape();
        tape.matrix(m).matvec(&(x * x)).tanh()
    });
    let forward = forward_jacobian(&at, |x| DualMatrix::constant(m).matvec(&(x * x)).tanh());
    assert_slice_close(reverse.as_slice(), forward.as_slice(), "Jacobian");

    // Against the closed form: J = diag(1 − tanh²(z))·M·2diag(x), z = M(x⊙x).
    let squared = at * at;
    let z = m.matvec(&squared);
    for row in 0..3 {
        for col in 0..4 {
            let outer_derivative = 1.0 - z.data()[row].tanh().powi(2);
            let expected = outer_derivative * m.data()[row][col] * 2.0 * at.data()[col];
            assert!(
                close(reverse.data()[row][col], expected),
                "J[{row},{col}]: {} vs {expected}",
                reverse.data()[row][col]
            );
        }
    }
}

#[test]
fn jacobians_with_respect_to_a_matrix_agree_between_the_modes() {
    use rinterp::tensors::dual::jacobian_wrt_matrix as forward_jacobian;
    use rinterp::tensors::tape::jacobian_wrt_matrix as reverse_jacobian;

    // f(A) = A·x, whose Jacobian is 3 × 12 with ∂(Ax)ᵢ/∂Aⱼₖ = δᵢⱼxₖ.
    let a = matrix::<3, 4>();
    let x = vector::<4>();

    let reverse = reverse_jacobian(&a, |m| {
        let tape = m.tape();
        m.matvec(&tape.vector(x))
    });
    let forward = forward_jacobian(&a, |m| m.matvec(&DualVector::constant(x)));
    assert_slice_close(reverse.as_slice(), forward.as_slice(), "matrix Jacobian");

    for output in 0..3 {
        for row in 0..3 {
            for col in 0..4 {
                let expected = if output == row { x.data()[col] } else { 0.0 };
                assert!(close(reverse.data()[output][row * 4 + col], expected));
            }
        }
    }
}

#[test]
fn a_matrix_valued_function_flattens_into_a_full_jacobian() {
    use rinterp::tensors::tape::jacobian_wrt_matrix;

    // Y = sin(A·B) with A 2×3 and B 3×2: the Jacobian is 4 × 6, and
    // ∂Y[i][j]/∂A[k][l] = δᵢₖ·cos(Z[i][j])·B[l][j].
    let a = matrix::<2, 3>();
    let b = matrix::<3, 2>();
    let z = a.matmul(&b);

    let computed = jacobian_wrt_matrix(&a, |m| {
        let tape = m.tape();
        m.matmul(&tape.matrix(b)).sin().flattened()
    });

    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for l in 0..3 {
                    let expected = if i == k {
                        z.data()[i][j].cos() * b.data()[l][j]
                    } else {
                        0.0
                    };
                    let actual = computed.data()[i * 2 + j][k * 3 + l];
                    assert!(
                        close(actual, expected),
                        "∂Y[{i},{j}]/∂A[{k},{l}]: {actual} vs {expected}"
                    );
                }
            }
        }
    }
}

// ---- outer products ----------------------------------------------------------

#[test]
fn outer_products_differentiate_in_both_operands() {
    // Y = u ⊗ v, so ∂Σ(S ⊙ Y)/∂u = S·v and ∂/∂v = Sᵀ·u.
    let u = vector::<4>();
    let v = other_vector::<3>();
    let weights = matrix::<4, 3>();

    let tape = Tape::new();
    let left = tape.vector(u);
    let right = tape.vector(v);
    let product = left.outer(&right);
    (&product * &tape.matrix(weights)).sum().backward();

    assert_slice_close(
        left.grad().as_slice(),
        weights.matvec(&v).as_slice(),
        "∂/∂u of a weighted outer product",
    );
    assert_slice_close(
        right.grad().as_slice(),
        u.vecmat(&weights).as_slice(),
        "∂/∂v of a weighted outer product",
    );

    // The value is the outer product itself.
    for row in 0..4 {
        for col in 0..3 {
            assert!(close(
                product.value().data()[row][col],
                u.data()[row] * v.data()[col]
            ));
        }
    }
}

#[test]
fn the_jacobian_of_an_outer_product_is_the_rank_one_pattern() {
    use rinterp::tensors::tape::jacobian;

    // Y = u ⊗ v flattened is 12 outputs over 4 inputs, with
    // ∂Y[i][j]/∂u[k] = δᵢₖ·v[j].
    let u = vector::<4>();
    let v = other_vector::<3>();

    let computed = jacobian(&u, |x| {
        let tape = x.tape();
        x.outer(&tape.vector(v)).flattened()
    });

    for i in 0..4 {
        for j in 0..3 {
            for k in 0..4 {
                let expected = if i == k { v.data()[j] } else { 0.0 };
                let actual = computed.data()[i * 3 + j][k];
                assert!(
                    close(actual, expected),
                    "∂(u⊗v)[{i},{j}]/∂u[{k}]: {actual} vs {expected}"
                );
            }
        }
    }
}

#[test]
fn matvec_gradients_are_the_outer_product_of_adjoint_and_input() {
    // The rule behind `matvec`: y = A·x gives Ā = ȳ ⊗ x. Seeding e₁ should
    // therefore produce a matrix that is zero except for row 1, which is x.
    let a = matrix::<3, 4>();
    let x = vector::<4>();

    let tape = Tape::new();
    let matrix_var = tape.matrix(a);
    let vector_var = tape.vector(x);
    matrix_var
        .matvec(&vector_var)
        .backward_with(Vector::new([0.0, 1.0, 0.0]));

    for row in 0..3 {
        for col in 0..4 {
            let expected = if row == 1 { x.data()[col] } else { 0.0 };
            assert!(close(matrix_var.grad().data()[row][col], expected));
        }
    }
    // And x̄ = Aᵀ·ȳ is the seeded row of A.
    assert_slice_close(
        vector_var.grad().as_slice(),
        &a.data()[1],
        "the vector adjoint",
    );
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn jacobians_and_outer_products_agree_between_the_backends() {
    use rinterp::tensors::tape::jacobian_wrt_matrix;

    let a = matrix::<4, 4>();
    let b = other_matrix::<4, 4>();

    let host = jacobian_wrt_matrix(&a, |m| {
        let tape = m.tape();
        m.matmul(&tape.matrix(b)).tanh().flattened()
    });
    let resident = jacobian_wrt_matrix(&a.to_backend::<Metal>(), |m| {
        let tape = m.tape();
        m.matmul(&tape.matrix(b.to_backend::<Metal>()))
            .tanh()
            .flattened()
    });
    assert_slice_close(resident.as_slice(), host.as_slice(), "16 × 16 Jacobian");
    if a.to_backend::<Metal>().is_device_resident() {
        assert!(resident.is_device_resident(), "the Jacobian stays resident");
    }

    let u = vector::<5>();
    let v = other_vector::<4>();
    let host_tape = Tape::<Host>::new();
    let (hu, hv) = (host_tape.vector(u), host_tape.vector(v));
    hu.outer(&hv).sum().backward();

    let gpu_tape = Tape::<Metal>::new();
    let (gu, gv) = (
        gpu_tape.vector(u.to_backend::<Metal>()),
        gpu_tape.vector(v.to_backend::<Metal>()),
    );
    gu.outer(&gv).sum().backward();

    assert_slice_close(gu.grad().as_slice(), hu.grad().as_slice(), "∂(u⊗v)/∂u");
    assert_slice_close(gv.grad().as_slice(), hv.grad().as_slice(), "∂(u⊗v)/∂v");
}

#[test]
fn the_squared_error_reductions_are_all_the_same_scalar() {
    // ‖R‖²_F is the sum of squares however it is spelled: as the Frobenius inner
    // product with itself, as an elementwise square then a sum, or as the vector
    // dot product of the flattening. Same value, and same gradient 2R.
    let residual = matrix::<3, 4>();
    let expected: f32 = residual.as_slice().iter().map(|value| value * value).sum();

    let tape = Tape::new();
    let r = tape.matrix(residual);
    let frobenius = r.frobenius_dot(&r);
    let squared_then_summed = (&r * &r).sum();
    let flattened = r.flattened().dot(&r.flattened());

    assert!(close(*frobenius.value(), expected), "frobenius_dot");
    assert!(close(*squared_then_summed.value(), expected), "(R⊙R).sum()");
    assert!(close(*flattened.value(), expected), "flattened dot");

    // Each reduction differentiates to 2R — the factor of two comes from the two
    // edges into `r`, not from a special rule for squaring.
    for output in [&frobenius, &squared_then_summed, &flattened] {
        output.backward();
        assert_slice_close(
            r.grad().as_slice(),
            residual.scale(2.0).as_slice(),
            "d‖R‖²/dR",
        );
    }

    // A plain sum is *not* a loss: signed errors cancel, and its gradient is
    // constant, so descent would push every residual toward −∞.
    tape.zero_grad();
    r.sum().backward();
    assert_eq!(r.grad().to_rows(), [[1.0f32; 4]; 3]);
}

// ---- comparisons and the losses they unlock ----------------------------------

#[test]
fn comparisons_pick_the_right_operand_and_split_ties() {
    use rinterp::tensors::Compare;

    // A deliberate tie in the middle: [.., 0.5 vs 0.5, ..].
    let a = Vector::new([0.5f32, -1.0, 2.0, 0.5]);
    let b = Vector::new([1.5f32, -2.0, 2.0, 0.5]);

    let tape = Tape::new();
    let left = tape.vector(a);
    let right = tape.vector(b);
    let largest = left.maximum(&right);
    assert_eq!(largest.value().to_array(), [1.5, -1.0, 2.0, 0.5]);
    assert_eq!(
        left.minimum(&right).value().to_array(),
        [0.5, -2.0, 2.0, 0.5]
    );

    largest.sum().backward();
    // b wins the first, a wins the second, and the last two tie.
    assert_eq!(left.grad().to_array(), [0.0, 1.0, 0.5, 0.5]);
    assert_eq!(right.grad().to_array(), [1.0, 0.0, 0.5, 0.5]);

    // `Min` routes the adjoint the other way.
    let tape = Tape::new();
    let left = tape.vector(a);
    let right = tape.vector(b);
    left.minimum(&right).sum().backward();
    assert_eq!(left.grad().to_array(), [1.0, 0.0, 0.5, 0.5]);
    assert_eq!(right.grad().to_array(), [0.0, 1.0, 0.5, 0.5]);

    // The CPU table is the definition the shader has to match.
    assert_eq!(Compare::MaxShare.value(2.0, 1.0), 1.0);
    assert_eq!(Compare::MaxShare.value(1.0, 2.0), 0.0);
    assert_eq!(Compare::MaxShare.value(1.0, 1.0), 0.5);
}

#[test]
fn abs_relu_and_clamp_have_the_conventional_derivatives() {
    let at = Vector::new([-2.0f32, -0.5, 0.0, 0.5, 2.0]);

    // |x|' = sign(x), and the tie at zero splits ±1 into 0.
    let tape = Tape::new();
    let x = tape.vector(at);
    let absolute = x.abs();
    assert_eq!(absolute.value().to_array(), [2.0, 0.5, 0.0, 0.5, 2.0]);
    absolute.sum().backward();
    assert_eq!(x.grad().to_array(), [-1.0, -1.0, 0.0, 1.0, 1.0]);

    // relu' is 1 above the kink, 0 below, ½ exactly at it.
    let tape = Tape::new();
    let x = tape.vector(at);
    let rectified = x.relu();
    assert_eq!(rectified.value().to_array(), [0.0, 0.0, 0.0, 0.5, 2.0]);
    rectified.sum().backward();
    assert_eq!(x.grad().to_array(), [0.0, 0.0, 0.5, 1.0, 1.0]);

    // Clamping pins the ends, and a pinned element has no gradient.
    let tape = Tape::new();
    let x = tape.vector(at);
    let confined = x.clamp(-1.0, 1.0);
    assert_eq!(confined.value().to_array(), [-1.0, -0.5, 0.0, 0.5, 1.0]);
    confined.sum().backward();
    assert_eq!(x.grad().to_array(), [0.0, 1.0, 1.0, 1.0, 0.0]);
}

#[test]
fn comparison_gradients_agree_with_forward_mode() {
    // Away from ties both modes must agree exactly, including through a chain.
    let a = vector::<6>();
    let b = other_vector::<6>();

    let reverse = reverse_gradient(a, |x| {
        let tape = x.tape();
        let other = tape.vector(b);
        x.maximum(&other).minimum(&x.abs().scale(3.0)).sum()
    });
    let forward = gradient(&a, |x| {
        let other = DualVector::constant(b);
        x.maximum(&other).minimum(&x.abs().scale(3.0)).sum()
    });
    assert_slice_close(reverse.as_slice(), forward.as_slice(), "min/max/abs chain");

    // And relu through a matrix product, which is the shape a network uses.
    let m = matrix::<4, 6>();
    let reverse = reverse_gradient(a, |x| {
        let tape = x.tape();
        tape.matrix(m).matvec(x).relu().sum()
    });
    let forward = gradient(&a, |x| DualMatrix::constant(m).matvec(x).relu().sum());
    assert_slice_close(reverse.as_slice(), forward.as_slice(), "relu after matvec");
}

#[test]
fn absolute_error_is_now_expressible_as_a_loss() {
    // L1: Σ|Ax − y|, whose gradient is Aᵀ·sign(Ax − y) — the estimator that
    // squared error could not express.
    let a = matrix::<5, 3>();
    let x = vector::<3>();
    let targets = other_vector::<5>();

    let tape = Tape::new();
    let parameters = tape.vector(x);
    let residual = &tape.matrix(a).matvec(&parameters) - &tape.vector(targets);
    residual.abs().sum().backward();

    let signs = Vector::new(std::array::from_fn::<f32, 5, _>(|i| {
        let r = a.matvec(&x).data()[i] - targets.data()[i];
        if r > 0.0 {
            1.0
        } else if r < 0.0 {
            -1.0
        } else {
            0.0
        }
    }));
    assert_slice_close(
        parameters.grad().as_slice(),
        signs.vecmat(&a).as_slice(),
        "∂Σ|Ax−y|/∂x",
    );
}

// ---- row and column reductions ------------------------------------------------

#[test]
fn row_and_column_sums_spread_their_adjoint_back_along_the_axis() {
    let m = matrix::<3, 4>();

    let tape = Tape::new();
    let recorded = tape.matrix(m);
    let rows = recorded.row_sums();
    for row in 0..3 {
        let expected: f32 = (0..4).map(|col| m.data()[row][col]).sum();
        assert!(close(rows.value().as_slice()[row], expected));
    }

    // Seeding one row's sum credits exactly that row.
    rows.backward_with(Vector::new([0.0, 1.0, 0.0]));
    assert_eq!(recorded.grad().to_rows(), [[0.0; 4], [1.0; 4], [0.0; 4]]);

    let tape = Tape::new();
    let recorded = tape.matrix(m);
    let columns = recorded.column_sums();
    for col in 0..4 {
        let expected: f32 = (0..3).map(|row| m.data()[row][col]).sum();
        assert!(close(columns.value().as_slice()[col], expected));
    }
    columns.backward_with(Vector::new([0.0, 0.0, 1.0, 0.0]));
    for row in 0..3 {
        assert_eq!(recorded.grad().data()[row], [0.0, 0.0, 1.0, 0.0]);
    }

    // Both modes agree on a chain that reduces along an axis.
    let reverse = {
        let tape = Tape::new();
        let recorded = tape.matrix(m);
        recorded.tanh().row_sums().sum().backward();
        recorded.grad()
    };
    let forward = gradient_wrt_matrix(&m, |d| d.tanh().row_sums().sum());
    assert_slice_close(reverse.as_slice(), forward.as_slice(), "row_sums chain");
}

#[test]
fn softmax_and_cross_entropy_work_end_to_end() {
    // Row-wise softmax needs a per-row denominator, which `row_sums` supplies and
    // an outer product with ones broadcasts back across the row.
    const CLASSES: usize = 4;
    const BATCH: usize = 3;

    let logits = Matrix::<f32, BATCH, CLASSES>::from_rows([
        [1.0, 2.0, 0.5, -1.0],
        [0.2, -0.3, 1.4, 0.9],
        [-0.7, 0.1, 0.3, 2.0],
    ]);
    // One-hot targets: class 1, class 2, class 3.
    let targets = Matrix::<f32, BATCH, CLASSES>::from_rows([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]);

    let tape = Tape::new();
    let recorded = tape.matrix(logits);
    let exponentials = recorded.exp();
    let denominators = exponentials.row_sums();
    let probabilities =
        &exponentials / &denominators.outer(&tape.vector(Vector::<f32, CLASSES>::filled(1.0)));
    let loss = probabilities
        .ln()
        .frobenius_dot(&tape.matrix(targets))
        .scale(-1.0 / BATCH as f32);
    loss.backward();

    // The rows are probabilities.
    for row in 0..BATCH {
        let total: f32 = (0..CLASSES)
            .map(|col| probabilities.value().data()[row][col])
            .sum();
        assert!(close(total, 1.0), "row {row} sums to {total}");
    }

    // Cross-entropy has the textbook gradient (p − target)/batch.
    for row in 0..BATCH {
        for col in 0..CLASSES {
            let expected =
                (probabilities.value().data()[row][col] - targets.data()[row][col]) / BATCH as f32;
            assert!(
                close(recorded.grad().data()[row][col], expected),
                "∂L/∂logit[{row},{col}]: {} vs {expected}",
                recorded.grad().data()[row][col]
            );
        }
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn comparisons_and_reductions_agree_between_the_backends() {
    use rinterp::tensors::Compare;

    let a = matrix::<6, 5>();
    let b = other_matrix::<6, 5>();

    // Every comparison arm of the shader against the CPU table.
    for op in Compare::ALL {
        let host = Host::matrix_compare(&a, &b, op);
        let resident = Metal::matrix_compare(&a.to_backend(), &b.to_backend(), op);
        assert_slice_close(resident.as_slice(), host.as_slice(), &format!("{op:?}"));

        let host = Host::matrix_compare_scalar(&a, 0.5, op, false);
        let resident = Metal::matrix_compare_scalar(&a.to_backend(), 0.5, op, false);
        assert_slice_close(
            resident.as_slice(),
            host.as_slice(),
            &format!("{op:?} against a scalar"),
        );
    }

    // A gradient through relu and a row reduction, on both backends.
    let host_tape = Tape::<Host>::new();
    let host_m = host_tape.matrix(a);
    host_m.relu().row_sums().sum().backward();

    let gpu_tape = Tape::<Metal>::new();
    let gpu_m = gpu_tape.matrix(a.to_backend::<Metal>());
    gpu_m.relu().row_sums().sum().backward();

    assert_slice_close(
        gpu_m.grad().as_slice(),
        host_m.grad().as_slice(),
        "relu + row_sums gradient",
    );
    if a.to_backend::<Metal>().is_device_resident() {
        assert!(
            gpu_m.grad().is_device_resident(),
            "the gradient stays resident"
        );
    }
}
