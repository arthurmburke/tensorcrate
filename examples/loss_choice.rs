//! Why the choice of objective matters, on one data set.
//!
//! ```text
//! cargo run --release --example loss_choice
//! ```
//!
//! The same linear model is fitted four ways to the same samples, one of which
//! has been corrupted. Squared error is the maximum-likelihood objective when the
//! noise is Gaussian with a constant variance; when that assumption is wrong, a
//! different objective is not a tweak but a different estimator.
//!
//! | objective | expression | assumption it encodes |
//! |---|---|---|
//! | sum of squares | `r.dot(&r)` | Gaussian noise, equal variance |
//! | mean squared error | the same, scaled by `1/n` | identical minimizer, gradients `n×` smaller |
//! | weighted least squares | `r.dot(&(w * r))` | Gaussian, known per-sample variance |
//! | log-cosh | `r.cosh().ln().sum()` | heavy tails: quadratic near zero, linear far out |
//! | absolute error | `r.abs().sum()` | Laplace noise: fits the median, ignores outlier magnitude |

use rinterp::tensors::{Matrix, Tape, Vector};

const SAMPLES: usize = 24;
const FEATURES: usize = 2;

type Design = Matrix<f32, SAMPLES, FEATURES>;
type Parameters = Vector<f32, FEATURES>;

fn main() {
    // A clean line, y = 0.8 + 1.5t, sampled on a grid.
    let design = Design::from_rows(std::array::from_fn(|row| {
        [1.0, row as f32 / SAMPLES as f32]
    }));
    let truth = Vector::new([0.8f32, 1.5]);
    let clean = design.matvec(&truth);

    // One sample is badly wrong — a sensor glitch, a transcription error.
    let mut corrupted = *clean.data();
    corrupted[SAMPLES / 2] += 9.0;
    let corrupted = Vector::new(corrupted);

    println!("truth                     {truth}");
    println!(
        "least squares, clean data {}",
        fit(&design, &clean, Objective::SumOfSquares)
    );
    println!();

    // With the outlier in place, squared error pays 81 for that one sample and
    // bends the whole line to reduce it.
    let squares = fit(&design, &corrupted, Objective::SumOfSquares);
    let log_cosh = fit(&design, &corrupted, Objective::LogCosh);
    let absolute = fit(&design, &corrupted, Objective::AbsoluteError);
    let weighted = fit(&design, &corrupted, Objective::Weighted);
    println!("with one corrupted sample:");
    println!("  sum of squares          {squares}");
    println!("  log-cosh                {log_cosh}");
    println!("  absolute error          {absolute}");
    println!("  weighted least squares  {weighted}");

    let error = |fitted: &Parameters| {
        (0..FEATURES)
            .map(|i| (fitted.data()[i] - truth.data()[i]).abs())
            .fold(0.0f32, f32::max)
    };
    println!();
    println!(
        "  distance from the truth — squares {:.3}, log-cosh {:.3}, absolute {:.3}, weighted {:.3}",
        error(&squares),
        error(&log_cosh),
        error(&absolute),
        error(&weighted)
    );
}

enum Objective {
    /// `Σ rᵢ²`: the Gaussian log-likelihood, up to constants.
    SumOfSquares,
    /// `Σ ln cosh rᵢ`: quadratic for small residuals, linear for large ones, and
    /// smooth everywhere — so its gradient `tanh(r)` is bounded by one and a
    /// single bad sample cannot dominate.
    LogCosh,
    /// `Σ|rᵢ|`: the Laplace log-likelihood, which fits the conditional median.
    /// Its gradient is `sign(r)`, so a residual of 9 pulls exactly as hard as one
    /// of 0.01 — bounded influence taken to its limit. The flip side is that the
    /// gradient never shrinks near the optimum, so a fixed step size dithers
    /// around it rather than settling.
    AbsoluteError,
    /// `Σ rᵢ²/σᵢ²`: squared error again, but with the known-bad sample given the
    /// variance it deserves.
    Weighted,
}

/// Plain gradient descent to convergence, differing only in the scalar it
/// minimizes.
fn fit(design: &Design, targets: &Vector<f32, SAMPLES>, objective: Objective) -> Parameters {
    // Trust every sample equally, except the one known to be unreliable.
    let mut precision = [1.0f32; SAMPLES];
    precision[SAMPLES / 2] = 0.01;
    let precision = Vector::new(precision);

    let mut parameters = Parameters::zeros();

    for step in 0..20_000 {
        // A nonsmooth objective needs a shrinking step. `|r|` keeps the full
        // magnitude of its gradient right up to the optimum, so a fixed step
        // dithers around the answer forever instead of settling into it; the
        // smooth objectives have no such trouble.
        let rate = match objective {
            Objective::AbsoluteError => 0.4 / (1.0 + step as f32 / 200.0),
            _ => 0.4,
        };
        let tape = Tape::new();
        let recorded = tape.vector(parameters);
        let residual = &tape.matrix(*design).matvec(&recorded) - &tape.vector(*targets);

        let loss = match objective {
            Objective::SumOfSquares => residual.dot(&residual),
            Objective::LogCosh => residual.cosh().ln().sum(),
            Objective::AbsoluteError => residual.abs().sum(),
            Objective::Weighted => residual.dot(&(&residual * &tape.vector(precision))),
        }
        .scale(1.0 / SAMPLES as f32);

        loss.backward();
        parameters = parameters - recorded.grad().scale(rate);
    }
    parameters
}
