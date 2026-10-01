//! What one training step costs, in kernels, bytes and allocations — with and
//! without fusion, on each backend.
//!
//! ```text
//! cargo run --release --features counters --example fusion_baseline
//! ```
//!
//! Three workloads, the ones the fusion plan baselines:
//!
//! 1. an Adam update of a 1M-element parameter tensor;
//! 2. one softmax cross-entropy step — forward, backward and an Adam update —
//!    for a linear classifier;
//! 3. one step of a two-layer MLP, every weight and bias updated by Adam.
//!
//! Each is run twice per backend: as it runs today, and with fusion switched off
//! through [`fused::with_mode`], which reproduces the old one-kernel-per-operation
//! behaviour exactly. The difference is what fusion has bought so far; what is
//! left is what the later phases (tape as IR, a lazy backend) are for.

use std::time::Instant;

use tensorcrate::counters::{self, Counts};
use tensorcrate::optim::{Adam, Rule};
use tensorcrate::tensors::fused::{self, Mode};
use tensorcrate::tensors::{Host, Kernels, Matrix, Tape, Vector};

#[cfg(all(feature = "metal", target_os = "macos"))]
use tensorcrate::tensors::Metal;

/// Steps measured after a warm-up step, so allocation pools and moment buffers
/// are in their steady state.
const STEPS: usize = 10;

fn main() {
    println!(
        "{:<26} {:<8} {:<8} {:>9} {:>10} {:>7} {:>8} {:>6} {:>10}",
        "per step", "backend", "fusion", "kernels", "MB moved", "allocs", "cmdbufs", "syncs", "time"
    );
    run::<Host>("host");
    #[cfg(all(feature = "metal", target_os = "macos"))]
    run::<Metal>("metal");
}

fn run<B: Kernels>(backend: &str) {
    for mode in [Mode::Unfused, Mode::Fused] {
        report("adam, 1M parameters", backend, mode, measure::<B>(mode, adam_step::<B>()));
    }
    for mode in [Mode::Unfused, Mode::Fused] {
        report(
            "softmax cross-entropy",
            backend,
            mode,
            measure::<B>(mode, softmax_step::<B>()),
        );
    }
    for mode in [Mode::Unfused, Mode::Fused] {
        report("two-layer MLP", backend, mode, measure::<B>(mode, mlp_step::<B>()));
    }
}

fn report(name: &str, backend: &str, mode: Mode, (counts, seconds): (Counts, f64)) {
    let per = |x: u64| x as f64 / STEPS as f64;
    println!(
        "{name:<26} {backend:<8} {:<8} {:>9.1} {:>10.2} {:>7.1} {:>8.1} {:>6.1} {:>8.3}ms",
        match mode {
            Mode::Fused => "fused",
            Mode::Unfused => "unfused",
        },
        per(counts.kernels),
        per(counts.bytes) / 1e6,
        per(counts.allocations),
        per(counts.command_buffers),
        per(counts.syncs),
        seconds / STEPS as f64 * 1e3,
    );
}

/// One warm-up call, then `STEPS` measured ones, in `mode`.
fn measure<B: Kernels>(mode: Mode, mut step: impl FnMut()) -> (Counts, f64) {
    fused::with_mode(mode, || {
        step();
        #[cfg(all(feature = "metal", target_os = "macos"))]
        tensorcrate::metal::synchronize();
        let start = Instant::now();
        let ((), counts) = counters::measure(|| {
            for _ in 0..STEPS {
                step();
            }
            #[cfg(all(feature = "metal", target_os = "macos"))]
            tensorcrate::metal::synchronize();
        });
        (counts, start.elapsed().as_secs_f64())
    })
}

fn values(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 2654435761 + seed * 40503) % 1000) as f32 / 500.0) - 1.0)
        .collect()
}

fn adam_step<B: Kernels>() -> impl FnMut() {
    const N: usize = 1 << 20;
    let gradient = Vector::new(values(N, 1)).to_backend::<B>();
    let mut parameters = Vector::new(values(N, 2)).to_backend::<B>();
    let mut rule = Adam::new(1e-3);
    move || rule.update(&mut parameters, &gradient)
}

/// A linear classifier, `softmax(W·X)` against one-hot targets.
fn softmax_step<B: Kernels>() -> impl FnMut() {
    const CLASSES: usize = 32;
    const FEATURES: usize = 256;
    const BATCH: usize = 128;
    let inputs = Matrix::from_flat(FEATURES, BATCH, values(FEATURES * BATCH, 3)).to_backend::<B>();
    let mut targets = vec![0.0f32; CLASSES * BATCH];
    for sample in 0..BATCH {
        targets[(sample * 7 % CLASSES) * BATCH + sample] = 1.0;
    }
    let targets = Matrix::from_flat(CLASSES, BATCH, targets).to_backend::<B>();
    let ones = Vector::<f32>::filled(CLASSES, 1.0).to_backend::<B>();
    let mut weights =
        Matrix::from_flat(CLASSES, FEATURES, values(CLASSES * FEATURES, 4)).to_backend::<B>();
    let mut rule = Adam::new(1e-2);

    move || {
        let tape = Tape::<B>::new();
        let w = tape.matrix(weights.to_backend::<B>());
        let logits = w.matmul(&tape.matrix(inputs.to_backend::<B>())).scale(0.05);
        // log softmax along each column: z − 1·ln(Σ exp z)ᵀ.
        let normalizer = logits.exp().column_sums().ln();
        let log_probabilities = &logits - &tape.vector(ones.to_backend::<B>()).outer(&normalizer);
        let loss = log_probabilities
            .frobenius_dot(&tape.matrix(targets.to_backend::<B>()))
            .scale(-1.0 / BATCH as f32);
        loss.backward();
        rule.update(&mut weights, &w.grad());
    }
}

/// `W₂·relu(W₁·X + b₁) + b₂` against regression targets, mean squared error.
fn mlp_step<B: Kernels>() -> impl FnMut() {
    const IN: usize = 64;
    const HIDDEN: usize = 256;
    const OUT: usize = 16;
    const BATCH: usize = 128;
    let inputs = Matrix::from_flat(IN, BATCH, values(IN * BATCH, 5)).to_backend::<B>();
    let targets = Matrix::from_flat(OUT, BATCH, values(OUT * BATCH, 6)).to_backend::<B>();
    let ones = Vector::<f32>::filled(BATCH, 1.0).to_backend::<B>();
    let mut w1 = Matrix::from_flat(HIDDEN, IN, values(HIDDEN * IN, 7))
        .to_backend::<B>();
    let mut b1 = Vector::new(values(HIDDEN, 8)).to_backend::<B>();
    let mut w2 = Matrix::from_flat(OUT, HIDDEN, values(OUT * HIDDEN, 9)).to_backend::<B>();
    let mut b2 = Vector::new(values(OUT, 10)).to_backend::<B>();
    let (mut r1, mut rb1, mut r2, mut rb2) =
        (Adam::new(1e-3), Adam::new(1e-3), Adam::new(1e-3), Adam::new(1e-3));

    move || {
        let tape = Tape::<B>::new();
        let (w1v, b1v) = (tape.matrix(w1.to_backend::<B>()), tape.vector(b1.to_backend::<B>()));
        let (w2v, b2v) = (tape.matrix(w2.to_backend::<B>()), tape.vector(b2.to_backend::<B>()));
        let ones = tape.vector(ones.to_backend::<B>());
        let hidden =
            (&w1v.matmul(&tape.matrix(inputs.to_backend::<B>())) + &b1v.outer(&ones)).relu();
        let predicted = &w2v.matmul(&hidden) + &b2v.outer(&ones);
        let residual = &predicted - &tape.matrix(targets.to_backend::<B>());
        let loss = residual
            .frobenius_dot(&residual)
            .scale(1.0 / (OUT * BATCH) as f32);
        loss.backward();
        r1.update(&mut w1, &w1v.grad());
        rb1.update(&mut b1, &b1v.grad());
        r2.update(&mut w2, &w2v.grad());
        rb2.update(&mut b2, &b2v.grad());
    }
}
