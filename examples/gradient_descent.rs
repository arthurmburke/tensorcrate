//! Gradient descent with reverse-mode autodiff, end to end.
//!
//! ```text
//! cargo run --release --example gradient_descent
//! ```
//!
//! Two fits, both minimizing a squared error by taking one backward pass per
//! step. The first is linear, so the answer can be checked against the exact
//! normal-equation solution; the second puts a `tanh` in the way, so there is no
//! closed form and the gradient is the only way through.
//!
//! The shape of the loop is the part worth copying: build a *fresh* tape each
//! step, record the current parameters as leaves, evaluate the loss, propagate,
//! then step the parameters outside the tape. A tape is a recording of one
//! evaluation — reusing it across steps would append every step's nodes to the
//! same graph and grow without bound.

use rinterp::tensors::{Matrix, Tape, Vector};

/// Rows of a small, well-conditioned design matrix.
const SAMPLES: usize = 12;
const FEATURES: usize = 3;

fn design() -> Matrix<f32, SAMPLES, FEATURES> {
    Matrix::from_rows(std::array::from_fn(|row| {
        let t = row as f32 / SAMPLES as f32;
        [1.0, t, (t * 2.4).sin()]
    }))
}

fn main() {
    let a = design();
    let truth = Vector::new([0.4f32, -1.3, 0.9]);
    let targets = a.matvec(&truth);

    // ---- linear least squares -------------------------------------------
    //
    // L(x) = ‖A·x − y‖²/n, whose gradient is (2/n)Aᵀ(A·x − y). Reverse mode
    // never needs that formula written down. Averaging over the samples is what
    // keeps a step size of this order stable — the raw sum has n times the
    // curvature, and gradient descent diverges above 2/λmax.
    println!("least squares: fitting {FEATURES} parameters to {SAMPLES} samples");
    let mut x = Vector::<f32, FEATURES>::zeros();
    // Descent is stable below 2/λmax of the loss curvature, which is 0.567 for
    // this averaged least-squares problem; 0.6 diverges to NaN, 0.4 does not.
    let rate = 0.4;

    for step in 0..=4000 {
        let tape = Tape::new();

        // The parameters are the leaves whose gradient we want.
        let parameters = tape.vector(x);
        // Everything else is recorded too; we simply never read its gradient.
        let residual = &tape.matrix(a).matvec(&parameters) - &tape.vector(targets);
        let loss = residual.dot(&residual).scale(1.0 / SAMPLES as f32);

        loss.backward();

        if step % 1000 == 0 {
            println!(
                "  step {step:>3}  loss {:>10.3e}  nodes on tape {}",
                loss.value(),
                tape.len()
            );
        }

        // The update happens outside the tape, on ordinary tensors.
        x = x - parameters.grad().scale(rate);
    }

    // The exact solution, for comparison: x* = (AᵀA)⁻¹Aᵀy.
    let normal = a.transpose().matmul(&a);
    let exact = normal
        .inverse()
        .expect("the design matrix has full rank")
        .matvec(&a.transpose().matvec(&targets));

    println!("  fitted {x}");
    println!("  exact  {exact}");
    println!("  truth  {truth}");
    let worst = (0..FEATURES)
        .map(|i| (x.data()[i] - exact.data()[i]).abs())
        .fold(0.0f32, f32::max);
    println!("  largest deviation from the exact solution: {worst:.2e}\n");

    // ---- a fit with no closed form ---------------------------------------
    //
    // L(W) = ‖tanh(W·X) − Y‖², a matrix parameter behind a nonlinearity. One
    // backward pass gives all nine partials; forward mode would need nine
    // passes, one per entry of W.
    println!("nonlinear fit: a 3×3 weight matrix behind a tanh");
    let inputs = Matrix::<f32, FEATURES, 4>::from_rows([
        [0.9, -0.4, 0.2, 0.7],
        [0.1, 0.8, -0.6, 0.3],
        [-0.5, 0.2, 0.7, -0.9],
    ]);
    let hidden = Matrix::<f32, FEATURES, FEATURES>::from_rows([
        [0.6, -0.2, 0.3],
        [0.1, 0.5, -0.4],
        [-0.3, 0.2, 0.7],
    ]);
    let wanted = hidden.matmul(&inputs).map(|value| value.tanh());

    let mut weights = Matrix::<f32, FEATURES, FEATURES>::identity().scale(0.1);
    let rate = 0.4;

    for step in 0..=20_000 {
        let tape = Tape::new();
        let parameters = tape.matrix(weights);
        let predicted = parameters.matmul(&tape.matrix(inputs)).tanh();
        let residual = &predicted - &tape.matrix(wanted);
        let loss = residual
            .frobenius_dot(&residual)
            .scale(1.0 / (FEATURES * 4) as f32);

        loss.backward();

        if step % 5_000 == 0 {
            println!("  step {step:>3}  loss {:>10.3e}", loss.value());
        }

        weights = weights - parameters.grad().scale(rate);
    }

    let recovered = weights.matmul(&inputs).map(|value| value.tanh());
    let error = (0..FEATURES)
        .flat_map(|row| (0..4).map(move |col| (row, col)))
        .map(|(row, col)| (recovered.data()[row][col] - wanted.data()[row][col]).abs())
        .fold(0.0f32, f32::max);
    println!("  largest residual on the fitted outputs: {error:.2e}");
    println!("  recovered weights:\n{weights}");
    println!();

    multi_output_regression();
}

/// Learning a *matrix* of parameters — and a bias vector alongside it.
///
/// `Y = W·X + b·1ᵀ` over a batch: `W` is `OUT × IN`, `b` is `OUT`, and both come
/// out of the same single backward pass. Recording a matrix parameter is the only
/// change from the vector case — `tape.matrix` instead of `tape.vector`, and
/// `grad()` hands back the same shape it was given.
fn multi_output_regression() {
    const IN: usize = 4;
    const OUT: usize = 3;
    const BATCH: usize = 16;

    // A batch of inputs, one sample per column. These have to be genuinely
    // independent across rows: smooth functions of a single parameter look fine
    // but leave the design nearly collinear, and then many (W, b) pairs fit the
    // data equally well — the loss falls while the parameters wander.
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / 8_388_608.0 - 1.0 // in [-1, 1)
    };
    let inputs = Matrix::<f32, IN, BATCH>::from_rows(std::array::from_fn(|_| {
        std::array::from_fn(|_| next())
    }));
    let true_weights = Matrix::<f32, OUT, IN>::from_rows([
        [0.8, -0.5, 0.3, 1.1],
        [-0.2, 0.9, -0.7, 0.4],
        [0.5, 0.1, 1.2, -0.6],
    ]);
    let true_bias = Vector::new([0.25f32, -0.4, 0.7]);
    let ones = Vector::<f32, BATCH>::filled(1.0);
    let targets = true_weights.matmul(&inputs) + outer(&true_bias, &ones);

    println!("multi-output regression: learning a {OUT}×{IN} matrix and a {OUT}-vector bias");

    let mut weights = Matrix::<f32, OUT, IN>::zeros();
    let mut bias = Vector::<f32, OUT>::zeros();
    let rate = 0.8;

    for step in 0..=6000 {
        let tape = Tape::new();

        // Two parameter tensors of different shapes, on one tape.
        let weight_var = tape.matrix(weights);
        let bias_var = tape.vector(bias);

        // The bias is broadcast across the batch as an outer product with ones,
        // which is differentiable like anything else: its adjoint contracts back
        // into a per-output sum over the batch.
        let predicted =
            &weight_var.matmul(&tape.matrix(inputs)) + &bias_var.outer(&tape.vector(ones));
        let residual = &predicted - &tape.matrix(targets);
        let loss = residual
            .frobenius_dot(&residual)
            .scale(1.0 / (OUT * BATCH) as f32);

        loss.backward();

        // One pass, both gradients: ∂L/∂W is OUT×IN and ∂L/∂b is OUT.
        if step == 0 {
            // ∂L/∂W = (2/n)(W·X − Y)·Xᵀ, checked once against the closed form.
            let discrepancy = (weight_var.grad()
                - (predicted.value().to_owned() - targets)
                    .matmul(&inputs.transpose())
                    .scale(2.0 / (OUT * BATCH) as f32))
            .as_slice()
            .iter()
            .fold(0.0f32, |worst, value| worst.max(value.abs()));
            println!("  ∂L/∂W agrees with (2/n)(WX−Y)Xᵀ to {discrepancy:.2e}");
        }
        if step % 2000 == 0 {
            println!("  step {step:>4}  loss {:>10.3e}", loss.value());
        }

        weights = weights - weight_var.grad().scale(rate);
        bias = bias - bias_var.grad().scale(rate);
    }

    let weight_error = (weights - true_weights)
        .as_slice()
        .iter()
        .fold(0.0f32, |worst, value| worst.max(value.abs()));
    let bias_error = (bias - true_bias)
        .as_slice()
        .iter()
        .fold(0.0f32, |worst, value| worst.max(value.abs()));
    println!("  weights recovered to {weight_error:.2e}, bias to {bias_error:.2e}");
    println!("  learned W:\n{weights}");
    println!("  learned b: {bias}");
}

/// `u ⊗ v` on plain tensors, for building the reference data.
fn outer<const R: usize, const C: usize>(
    u: &Vector<f32, R>,
    v: &Vector<f32, C>,
) -> Matrix<f32, R, C> {
    Matrix::from_rows(std::array::from_fn(|row| {
        std::array::from_fn(|col| u.data()[row] * v.data()[col])
    }))
}
