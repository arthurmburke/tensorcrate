//! The update rules, side by side on problems that separate them.
//!
//! ```text
//! cargo run --release --example optimizers
//! ```
//!
//! Three comparisons, because no rule wins everywhere:
//!
//! 1. an ill-conditioned quadratic, where momentum and the adaptive rules pull
//!    away from plain descent;
//! 2. full-batch against stochastic mini-batches on the same fit;
//! 3. a two-parameter model, which is where the hand-written loop belongs.

use tensorcrate::optim::{AdaGrad, Adam, Momentum, RmsProp, Rule, Sgd, minimize};
use tensorcrate::tensors::{Matrix, Tape, Vector};

const FEATURES: usize = 2;

fn main() {
    ill_conditioned();
    println!();
    stochastic_against_full_batch();
    println!();
    two_parameters_two_rules();
}

/// `L(x) = 100(x₀−1)² + (x₁−1)²`: a narrow valley, the classic case where the
/// step size that keeps the stiff direction stable leaves the flat one crawling.
fn ill_conditioned() {
    let centre = Vector::new([1.0f32, 1.0]);
    let curvature = Vector::new([100.0f32, 1.0]);
    let steps = 300;

    println!("ill-conditioned bowl (curvature 100:1), {steps} steps from the origin");
    println!(
        "  {:<24} {:>7} {:>12} {:>12}",
        "rule", "rate", "loss", "distance"
    );

    let report = |name: &str, rate: f32, parameters: Vector<f32>| {
        let offset = parameters - centre.clone();
        let loss: f32 = (0..FEATURES)
            .map(|i| curvature[i] * offset[i].powi(2))
            .sum();
        let distance = (0..FEATURES).map(|i| offset[i].powi(2)).sum::<f32>().sqrt();
        println!("  {name:<24} {rate:>7} {loss:>12.3e} {distance:>12.4}");
    };

    // The largest step plain descent can take here: 2/λmax with λmax = 200.
    let rate = 0.009;

    let mut parameters = Vector::<f32>::zeros(FEATURES);
    minimize(&mut parameters, &mut Sgd::new(rate), steps, |x, _| {
        bowl(x, centre.clone(), curvature.clone())
    });
    report("plain descent", rate, parameters);

    let mut parameters = Vector::<f32>::zeros(FEATURES);
    minimize(
        &mut parameters,
        &mut Momentum::new(rate, 0.9),
        steps,
        |x, _| bowl(x, centre.clone(), curvature.clone()),
    );
    report("momentum (0.9)", rate, parameters);

    // Nesterov's step carries an extra `μv`, so its effective stride is `(1+μ)`
    // times heavy-ball's and it goes unstable where classical momentum is fine —
    // at 0.009 this diverges to NaN in a few dozen steps.
    let nesterov_rate = 0.005;
    let mut parameters = Vector::<f32>::zeros(FEATURES);
    minimize(
        &mut parameters,
        &mut Momentum::nesterov(nesterov_rate, 0.9),
        steps,
        |x, _| bowl(x, centre.clone(), curvature.clone()),
    );
    report("nesterov (0.9)", nesterov_rate, parameters);

    let mut parameters = Vector::<f32>::zeros(FEATURES);
    minimize(&mut parameters, &mut AdaGrad::new(0.5), steps, |x, _| {
        bowl(x, centre.clone(), curvature.clone())
    });
    report("adagrad", 0.5, parameters);

    let mut parameters = Vector::<f32>::zeros(FEATURES);
    minimize(&mut parameters, &mut RmsProp::new(0.05), steps, |x, _| {
        bowl(x, centre.clone(), curvature.clone())
    });
    report("rmsprop", 0.05, parameters);

    let mut parameters = Vector::<f32>::zeros(FEATURES);
    minimize(&mut parameters, &mut Adam::new(0.1), steps, |x, _| {
        bowl(x, centre.clone(), curvature.clone())
    });
    report("adam", 0.1, parameters);
}

fn bowl<'t>(
    x: &tensorcrate::tensors::VectorVar<'t>,
    centre: Vector<f32>,
    curvature: Vector<f32>,
) -> tensorcrate::tensors::ScalarVar<'t> {
    let tape = x.tape();
    let offset = x - &tape.vector(centre);
    (&offset * &offset).dot(&tape.vector(curvature))
}

/// The same regression fitted on all the data at once and on mini-batches.
///
/// A stochastic step is cheaper and noisier: it sees a quarter of the samples, so
/// it costs a quarter of the work and points in roughly the descent direction.
fn stochastic_against_full_batch() {
    const SAMPLES: usize = 32;
    const BATCH: usize = 8;

    let design = Matrix::<f32>::from_rows(
        (0..SAMPLES).map(|row| vec![1.0, (row as f32 / SAMPLES as f32) * 2.0 - 1.0]),
    );
    let truth = Vector::new([-0.75f32, 1.25]);
    let targets = design.matvec(&truth);

    println!("linear fit, {SAMPLES} samples, adam at 0.05");

    let mut full = Vector::<f32>::zeros(FEATURES);
    minimize(&mut full, &mut Adam::new(0.05), 600, |x, _| {
        let tape = x.tape();
        let residual = &tape.matrix(design.clone()).matvec(x) - &tape.vector(targets.clone());
        residual.dot(&residual).scale(1.0 / SAMPLES as f32)
    });
    println!("  full batch, 600 steps      {full}");

    let mut stochastic = Vector::<f32>::zeros(FEATURES);
    minimize(&mut stochastic, &mut Adam::new(0.05), 600, |x, step| {
        // The step number chooses the mini-batch; nothing else differs.
        let start = (step % (SAMPLES / BATCH)) * BATCH;
        let tape = x.tape();
        let rows: Matrix<f32> =
            Matrix::from_rows((0..BATCH).map(|row| design.row(start + row).to_vec()));
        let wanted: Vector<f32> = Vector::new(
            (0..BATCH)
                .map(|row| targets[start + row])
                .collect::<Vec<_>>(),
        );
        let residual = &tape.matrix(rows).matvec(x) - &tape.vector(wanted);
        residual.dot(&residual).scale(1.0 / BATCH as f32)
    });
    println!("  mini-batches of {BATCH}, 600 steps {stochastic}");
    println!("  truth                      {truth}");
    println!("  (each stochastic step touched {BATCH} rows instead of {SAMPLES})");
}

/// A weight matrix and a bias vector, each with its own rule.
///
/// `minimize` drives one parameter tensor, so a model with several wants the loop
/// written out which makes it obvious that the rules are the only stateful part.
fn two_parameters_two_rules() {
    const IN: usize = 3;
    const OUT: usize = 2;
    const BATCH: usize = 8;

    let inputs = Matrix::<f32>::from_rows((0..IN).map(|row| {
        (0..BATCH)
            .map(|col| ((row * 7 + col * 3) % 11) as f32 * 0.2 - 1.0)
            .collect::<Vec<_>>()
    }));
    let true_weights = Matrix::<f32>::from_rows([[0.6, -0.9, 0.3], [1.4, 0.2, -0.5]]);
    let true_bias = Vector::new([0.25f32, -0.4]);
    let targets = true_weights.matmul(&inputs)
        + Matrix::from_rows((0..OUT).map(|row| vec![true_bias[row]; BATCH]));

    let mut weights = Matrix::<f32>::zeros(OUT, IN);
    let mut bias = Vector::<f32>::zeros(OUT);
    let mut weight_rule = Adam::new(0.05);
    let mut bias_rule = Momentum::new(0.05, 0.9);

    for _ in 0..2000 {
        let tape = Tape::new();
        let weight_var = tape.matrix(weights.clone());
        let bias_var = tape.vector(bias.clone());

        let predicted = &weight_var.matmul(&tape.matrix(inputs.clone()))
            + &bias_var.outer(&tape.vector(Vector::<f32>::filled(BATCH, 1.0)));
        let residual = &predicted - &tape.matrix(targets.clone());
        let loss = residual
            .frobenius_dot(&residual)
            .scale(1.0 / (OUT * BATCH) as f32);

        loss.backward();
        weight_rule.update(&mut weights, &weight_var.grad());
        bias_rule.update(&mut bias, &bias_var.grad());
    }

    let weight_error = (weights - true_weights)
        .as_slice()
        .iter()
        .fold(0.0f32, |worst, value| worst.max(value.abs()));
    let bias_error = (bias - true_bias)
        .as_slice()
        .iter()
        .fold(0.0f32, |worst, value| worst.max(value.abs()));
    println!("two parameters, one rule each (adam for W, momentum for b)");
    println!("  weights recovered to {weight_error:.2e}, bias to {bias_error:.2e}");
}
