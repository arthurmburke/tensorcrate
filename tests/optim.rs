//! Optimizers: the update rules and the loop that drives them.
//!
//! The gradients themselves are already covered by the autodiff tests, so these
//! check what the optimizers add — that each rule moves parameters the way its
//! definition says, that the adaptive ones earn their keep on a badly scaled
//! problem, and that the driver is indifferent to the objective, the parameter
//! shape and the backend.

use tensorcrate::optim::{AdaGrad, Adam, Momentum, Parameter, RmsProp, Rule, Sgd, minimize};
use tensorcrate::tensors::{Matrix, Tape, Vector};

#[cfg(all(feature = "metal", target_os = "macos"))]
use tensorcrate::tensors::Metal;

fn close(actual: f32, expected: f32) -> bool {
    (actual - expected).abs() <= 2e-3 * (1.0 + expected.abs())
}

/// A quadratic bowl with a known minimum at `centre`, and curvature `scale` per
/// axis: `L(x) = Σ scaleᵢ(xᵢ − centreᵢ)²`.
fn bowl<'t>(
    x: &tensorcrate::tensors::VectorVar<'t>,
    centre: Vector<f32>,
    curvature: Vector<f32>,
) -> tensorcrate::tensors::ScalarVar<'t> {
    let tape = x.tape();
    let offset = x - &tape.vector(centre);
    (&offset * &offset).dot(&tape.vector(curvature))
}

// ---- each rule does what its definition says ---------------------------------

#[test]
fn plain_descent_takes_exactly_the_gradient_step() {
    let mut parameters = Vector::new([1.0f32, -2.0]);
    let gradient = Vector::new([0.5f32, 4.0]);
    let mut rule = Sgd::new(0.1);

    rule.update(&mut parameters, &gradient);
    assert_eq!(parameters.to_vec(), [1.0 - 0.05, -2.0 - 0.4]);
}

#[test]
fn momentum_accumulates_a_velocity_and_nesterov_looks_ahead() {
    // A constant gradient makes the velocity a geometric series, so the steps are
    // predictable by hand: v₁ = g, v₂ = μg + g.
    let gradient = Vector::new([1.0f32]);
    let mut parameters = Vector::new([0.0f32]);
    let mut rule = Momentum::new(0.1, 0.9);

    rule.update(&mut parameters, &gradient);
    assert!(close(parameters[0], -0.1));
    rule.update(&mut parameters, &gradient);
    assert!(close(parameters[0], -0.1 - 0.19)); // rate·(0.9 + 1)

    // Nesterov applies the momentum term to where the step is heading:
    // first update is rate·(g + μg).
    let mut parameters = Vector::new([0.0f32]);
    let mut rule = Momentum::nesterov(0.1, 0.9);
    rule.update(&mut parameters, &gradient);
    assert!(close(parameters[0], -0.19));

    // `reset` forgets the velocity but keeps the hyperparameters.
    rule.reset();
    let mut fresh = Vector::new([0.0f32]);
    rule.update(&mut fresh, &gradient);
    assert!(close(fresh[0], -0.19));
}

#[test]
fn the_adaptive_rules_normalize_by_the_gradient_scale() {
    // With a constant gradient, AdaGrad's first step is exactly the rate — the
    // magnitude cancels — and later steps shrink like 1/√t.
    for magnitude in [0.01f32, 1.0, 100.0] {
        let mut parameters = Vector::new([0.0f32]);
        let mut rule = AdaGrad::new(0.1);
        rule.update(&mut parameters, &Vector::new([magnitude]));
        assert!(
            close(parameters[0], -0.1),
            "AdaGrad first step with gradient {magnitude}: {}",
            parameters[0]
        );

        // RMSProp also cancels the magnitude, but its first step is inflated by
        // 1/√(1−decay): the running mean square starts at zero and only gets
        // `(1−ρ)g²` put into it, so the denominator is too small early on. That
        // is exactly the bias Adam's correction removes.
        let mut parameters = Vector::new([0.0f32]);
        let mut rule = RmsProp::new(0.1);
        rule.update(&mut parameters, &Vector::new([magnitude]));
        assert!(
            close(parameters[0], -0.1 / (1.0f32 - 0.9).sqrt()),
            "RMSProp first step with gradient {magnitude}: {}",
            parameters[0]
        );

        // Adam has no such bias: m̂/√v̂ is exactly ±1 on step one.
        let mut parameters = Vector::new([0.0f32]);
        let mut rule = Adam::new(0.1);
        rule.update(&mut parameters, &Vector::new([magnitude]));
        assert!(close(parameters[0], -0.1), "Adam first step");
    }

    // AdaGrad's denominator only grows, so equal gradients give shrinking steps.
    let mut parameters = Vector::new([0.0f32]);
    let mut rule = AdaGrad::new(0.1);
    let mut previous = 0.0;
    for step in 1..5 {
        rule.update(&mut parameters, &Vector::new([1.0]));
        let taken = previous - parameters[0];
        assert!(
            close(taken, 0.1 / (step as f32).sqrt()),
            "step {step} moved {taken}"
        );
        previous = parameters[0];
    }
}

#[test]
fn a_gradient_of_zero_leaves_every_rule_where_it_started() {
    let zero = Vector::<f32>::zeros(3);
    let start = Vector::new([1.0f32, 2.0, 3.0]);

    let mut sgd = Sgd::new(0.5);
    let mut momentum = Momentum::new(0.5, 0.9);
    let mut adagrad = AdaGrad::new(0.5);
    let mut rmsprop = RmsProp::new(0.5);
    let mut adam = Adam::new(0.5);

    for _ in 0..3 {
        let mut parameters = start.clone();
        sgd.update(&mut parameters, &zero);
        assert_eq!(parameters.to_vec(), start.to_vec(), "sgd");

        let mut parameters = start.clone();
        momentum.update(&mut parameters, &zero);
        assert_eq!(parameters.to_vec(), start.to_vec(), "momentum");

        // The adaptive denominators are guarded by epsilon, so 0/ε is 0 rather
        // than a division by zero.
        let mut parameters = start.clone();
        adagrad.update(&mut parameters, &zero);
        assert_eq!(parameters.to_vec(), start.to_vec(), "adagrad");

        let mut parameters = start.clone();
        rmsprop.update(&mut parameters, &zero);
        assert_eq!(parameters.to_vec(), start.to_vec(), "rmsprop");

        let mut parameters = start.clone();
        adam.update(&mut parameters, &zero);
        assert_eq!(parameters.to_vec(), start.to_vec(), "adam");
    }
}

// ---- every rule finds the same minimum ----------------------------------------

#[test]
fn every_rule_reaches_the_minimum_of_a_well_behaved_bowl() {
    let centre = Vector::new([1.5f32, -0.5, 2.0]);
    let curvature = Vector::new([1.0f32, 1.0, 1.0]);
    let mut parameters = Vector::<f32>::zeros(3);
    minimize(&mut parameters, &mut Sgd::new(0.1), 400, |x, _| {
        bowl(x, centre.clone(), curvature.clone())
    });
    assert_slice(&parameters, &centre, "sgd");

    let mut parameters = Vector::<f32>::zeros(3);
    minimize(
        &mut parameters,
        &mut Momentum::new(0.05, 0.9),
        400,
        |x, _| bowl(x, centre.clone(), curvature.clone()),
    );
    assert_slice(&parameters, &centre, "momentum");

    let mut parameters = Vector::<f32>::zeros(3);
    minimize(
        &mut parameters,
        &mut Momentum::nesterov(0.05, 0.9),
        400,
        |x, _| bowl(x, centre.clone(), curvature.clone()),
    );
    assert_slice(&parameters, &centre, "nesterov");

    let mut parameters = Vector::<f32>::zeros(3);
    minimize(&mut parameters, &mut AdaGrad::new(0.5), 4000, |x, _| {
        bowl(x, centre.clone(), curvature.clone())
    });
    assert_slice(&parameters, &centre, "adagrad");

    let mut parameters = Vector::<f32>::zeros(3);
    minimize(&mut parameters, &mut RmsProp::new(0.05), 2000, |x, _| {
        bowl(x, centre.clone(), curvature.clone())
    });
    assert_slice(&parameters, &centre, "rmsprop");

    let mut parameters = Vector::<f32>::zeros(3);
    minimize(&mut parameters, &mut Adam::new(0.1), 2000, |x, _| {
        bowl(x, centre.clone(), curvature.clone())
    });
    assert_slice(&parameters, &centre, "adam");
}

fn assert_slice(actual: &Vector<f32>, expected: &Vector<f32>, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: lengths differ");
    for index in 0..actual.len() {
        assert!(
            close(actual[index], expected[index]),
            "{what} at {index}: {} vs {}",
            actual[index],
            expected[index]
        );
    }
}

#[test]
fn momentum_and_adam_beat_plain_descent_on_an_ill_conditioned_bowl() {
    // Curvature spread of 100:1 — the valley plain descent zigzags across.
    let centre = Vector::new([1.0f32, 1.0]);
    let curvature = Vector::new([100.0f32, 1.0]);
    // A rate near the stability limit of the stiff direction, shared by both, so
    // the only difference is the rule.
    let rate = 0.009;
    let steps = 300;

    let mut plain = Vector::<f32>::zeros(2);
    minimize(&mut plain, &mut Sgd::new(rate), steps, |x, _| {
        bowl(x, centre.clone(), curvature.clone())
    });

    let mut accelerated = Vector::<f32>::zeros(2);
    minimize(
        &mut accelerated,
        &mut Momentum::new(rate, 0.9),
        steps,
        |x, _| bowl(x, centre.clone(), curvature.clone()),
    );

    let mut adaptive = Vector::<f32>::zeros(2);
    minimize(&mut adaptive, &mut Adam::new(0.1), steps, |x, _| {
        bowl(x, centre.clone(), curvature.clone())
    });

    let distance = |x: &Vector<f32>| ((x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2)).sqrt();
    let (plain, accelerated, adaptive) = (
        distance(&plain),
        distance(&accelerated),
        distance(&adaptive),
    );

    assert!(
        accelerated < plain / 5.0,
        "momentum should be far closer: {accelerated} vs {plain}"
    );
    assert!(
        adaptive < plain / 5.0,
        "adam should be far closer: {adaptive} vs {plain}"
    );
}

// ---- the driver is indifferent to objective, shape and backend ----------------

#[test]
fn the_same_rule_serves_different_objectives() {
    // One outlier, three objectives, one optimizer type: the estimator changes,
    // the machinery does not.
    let design = Matrix::<f32>::from_rows((0..10).map(|row| vec![1.0, row as f32 / 9.0]));
    let truth = Vector::new([0.5f32, 2.0]);
    let mut targets = design.matvec(&truth).data().to_vec();
    targets[4] += 6.0;
    let targets = Vector::new(targets);

    let mut least_squares = Vector::<f32>::zeros(2);
    minimize(&mut least_squares, &mut Adam::new(0.05), 3000, |x, _| {
        let tape = x.tape();
        let residual = &tape.matrix(design.clone()).matvec(x) - &tape.vector(targets.clone());
        residual.dot(&residual).scale(1.0 / 9.0)
    });

    let mut least_absolute = Vector::<f32>::zeros(2);
    minimize(&mut least_absolute, &mut Adam::new(0.05), 3000, |x, _| {
        let tape = x.tape();
        let residual = &tape.matrix(design.clone()).matvec(x) - &tape.vector(targets.clone());
        residual.abs().sum().scale(1.0 / 9.0)
    });

    // The robust fit stays nearer the truth; squared error chases the outlier.
    let error = |x: &Vector<f32>| (x[0] - 0.5).abs().max((x[1] - 2.0).abs());
    assert!(
        error(&least_absolute) < error(&least_squares),
        "absolute {} should beat squared {}",
        error(&least_absolute),
        error(&least_squares)
    );
}

#[test]
fn a_matrix_parameter_optimizes_the_same_way() {
    // Recover a 2×3 matrix from ‖W·X − Y‖², with the driver unchanged.
    let inputs = Matrix::<f32>::from_rows([
        [1.0, 0.2, -0.4, 0.9],
        [-0.3, 1.1, 0.5, -0.7],
        [0.6, -0.8, 1.0, 0.1],
    ]);
    let truth = Matrix::<f32>::from_rows([[0.5, -1.0, 0.25], [2.0, 0.1, -0.75]]);
    let targets = truth.matmul(&inputs);

    let mut weights = Matrix::<f32>::zeros(2, 3);
    let final_loss = minimize(&mut weights, &mut Adam::new(0.05), 4000, |w, _| {
        let tape = w.tape();
        let residual = &w.matmul(&tape.matrix(inputs.clone())) - &tape.matrix(targets.clone());
        residual.frobenius_dot(&residual)
    });

    assert!(final_loss < 1e-6, "loss {final_loss}");
    for row in 0..2 {
        for col in 0..3 {
            assert!(
                close(weights[(row, col)], truth[(row, col)]),
                "W[{row},{col}]: {} vs {}",
                weights[(row, col)],
                truth[(row, col)]
            );
        }
    }
}

#[test]
fn stochastic_steps_reach_the_same_place_as_full_batch_ones() {
    // Twelve samples in three mini-batches. Each step sees a quarter of the data
    // and a noisier gradient; the fit still converges, which is the whole premise
    // of stochastic descent.
    const SAMPLES: usize = 12;
    const BATCH: usize = 4;

    let design = Matrix::<f32>::from_rows(
        (0..SAMPLES).map(|row| vec![1.0, (row as f32 / SAMPLES as f32) * 2.0 - 1.0]),
    );
    let truth = Vector::new([-0.75f32, 1.25]);
    let targets = design.matvec(&truth);

    let mut stochastic = Vector::<f32>::zeros(2);
    minimize(&mut stochastic, &mut Adam::new(0.05), 3000, |x, step| {
        // The step number picks the batch; nothing else changes.
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
    assert_slice(&stochastic, &truth, "stochastic");

    let mut full = Vector::<f32>::zeros(2);
    minimize(&mut full, &mut Adam::new(0.05), 3000, |x, _| {
        let tape = x.tape();
        let residual = &tape.matrix(design.clone()).matvec(x) - &tape.vector(targets.clone());
        residual.dot(&residual).scale(1.0 / SAMPLES as f32)
    });
    assert_slice(&full, &truth, "full batch");
}

#[test]
fn a_scalar_parameter_works_too() {
    // Learn a log-variance: minimize r²·e^{−2s} + 2s, whose optimum is
    // s = ln|r|, the Gaussian maximum-likelihood estimate of the scale.
    let residual = 3.0f32;
    let mut log_scale = 0.0f32;
    minimize(&mut log_scale, &mut Adam::new(0.05), 3000, |s, _step| {
        let tape = s.tape();
        let precision = s.scale(-2.0).exp();
        let fit = tape.scalar(residual * residual).mul(&precision);
        fit.add(&s.scale(2.0))
    });
    assert!(
        close(log_scale, residual.ln()),
        "{log_scale} vs {}",
        residual.ln()
    );
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn optimizing_a_resident_parameter_stays_on_the_backend() {
    let inputs = Matrix::<f32>::from_rows([
        [1.0, 0.2, -0.4, 0.9],
        [-0.3, 1.1, 0.5, -0.7],
        [0.6, -0.8, 1.0, 0.1],
    ]);
    let truth = Matrix::<f32>::from_rows([[0.5, -1.0, 0.25], [2.0, 0.1, -0.75]]);
    let targets = truth.matmul(&inputs);

    let mut weights = Matrix::<f32, Metal>::filled(2, 3, 0.0);
    minimize(&mut weights, &mut Adam::new(0.05), 3000, |w, _| {
        let tape = w.tape();
        let residual = &w.matmul(&tape.matrix(inputs.to_backend::<Metal>()))
            - &tape.matrix(targets.to_backend::<Metal>());
        residual.frobenius_dot(&residual)
    });

    let host = weights.to_backend::<tensorcrate::tensors::Host>();
    for row in 0..2 {
        for col in 0..3 {
            assert!(
                close(host[(row, col)], truth[(row, col)]),
                "W[{row},{col}]: {} vs {}",
                host[(row, col)],
                truth[(row, col)]
            );
        }
    }
    // The update arithmetic — including Adam's moments and the square root —
    // never left shared memory.
    if Matrix::<f32, Metal>::filled(2, 3, 0.0).is_device_resident() {
        assert!(weights.is_device_resident(), "the parameters stay resident");
    }
}

#[test]
fn rules_are_generic_over_the_parameter_shape() {
    // `Parameter` is what lets one rule serve a scalar, a vector and a matrix.
    fn takes_any<P: Parameter>(rule: &mut impl Rule<P>, parameters: &mut P, gradient: &P) {
        rule.update(parameters, gradient.as_gradient());
    }

    let mut scalar = 1.0f32;
    takes_any(&mut Sgd::new(0.5), &mut scalar, &2.0);
    assert_eq!(scalar, 0.0);

    let mut vector = Vector::new([1.0f32, 1.0]);
    takes_any(&mut Sgd::new(0.5), &mut vector, &Vector::new([2.0, 4.0]));
    assert_eq!(vector.to_vec(), [0.0, -1.0]);

    let mut matrix = Matrix::<f32>::from_rows([[1.0, 1.0]]);
    takes_any(
        &mut Sgd::new(0.5),
        &mut matrix,
        &Matrix::from_rows([[2.0, 4.0]]),
    );
    assert_eq!(matrix.to_rows(), [[0.0, -1.0]]);
}

#[test]
fn the_driver_and_a_hand_written_loop_agree() {
    // `minimize` is a convenience, not a requirement: the same rule applied in a
    // loop has to land in the same place, which is what makes multi-parameter
    // models (one rule per tensor) straightforward.
    let centre = Vector::new([2.0f32, -1.0]);
    let curvature = Vector::new([1.0f32, 3.0]);

    let mut driven = Vector::<f32>::zeros(2);
    minimize(&mut driven, &mut Momentum::new(0.05, 0.9), 200, |x, _| {
        bowl(x, centre.clone(), curvature.clone())
    });

    let mut manual = Vector::<f32>::zeros(2);
    let mut rule = Momentum::new(0.05, 0.9);
    for _ in 0..200 {
        let tape = Tape::new();
        let recorded = tape.vector(manual.clone());
        bowl(&recorded, centre.clone(), curvature.clone()).backward();
        rule.update(&mut manual, &recorded.grad());
    }

    assert_eq!(driven.to_vec(), manual.to_vec());
}

// ---- gradients read like program inputs ---------------------------------------------

/// Every rule, freshly made, for parameters of type `P`.
fn every_rule<P: Parameter<Elem = f32>>() -> Vec<(&'static str, Box<dyn Rule<P>>)> {
    vec![
        ("sgd", Box::new(Sgd::new(0.05))),
        ("momentum", Box::new(Momentum::new(0.05, 0.9))),
        ("nesterov", Box::new(Momentum::nesterov(0.05, 0.9))),
        ("adagrad", Box::new(AdaGrad::new(0.1))),
        ("rmsprop", Box::new(RmsProp::new(0.01))),
        ("adam", Box::new(Adam::new(0.01))),
    ]
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|x| x.to_bits()).collect()
}

/// Gradient `step` for a parameter of `len`, as `f32`s that `f16` holds
/// exactly, so a gradient stored either way has the same values.
fn halvable(step: usize, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            tensorcrate::numbers::f16::from_f32(((i * 7 + step * 3) % 11) as f32 * 0.125 - 0.6)
                .to_f32()
        })
        .collect()
}

#[test]
fn a_view_of_a_larger_gradient_updates_like_a_copy_of_it() {
    use tensorcrate::tensors::Host;
    for (name, mut viewed) in every_rule::<Matrix<f32>>() {
        let (_, mut copied) = every_rule::<Matrix<f32>>()
            .into_iter()
            .find(|(other, _)| *other == name)
            .unwrap();
        let start = Matrix::from_flat(3, 4, (0..12).map(|i| i as f32 * 0.1).collect::<Vec<_>>());
        let (mut a, mut b) = (start.clone(), start);
        for step in 0..5 {
            // The parameters' gradient is the middle of a larger matrix.
            let packed = Matrix::<f32, Host>::from_flat(5, 7, halvable(step, 35));
            let view = packed.view(1..4, 2..6);
            viewed.update(&mut a, &view);
            copied.update(&mut b, &view.to_matrix());
            assert_eq!(
                bits(a.as_slice()),
                bits(b.as_slice()),
                "{name}, step {step}"
            );
        }
    }

    // A vector's gradient can be a column of a matrix.
    for (name, mut viewed) in every_rule::<Vector<f32>>() {
        let (_, mut copied) = every_rule::<Vector<f32>>()
            .into_iter()
            .find(|(other, _)| *other == name)
            .unwrap();
        let (mut a, mut b) = (Vector::new(vec![0.5f32; 6]), Vector::new(vec![0.5f32; 6]));
        for step in 0..5 {
            let packed = Matrix::<f32>::from_flat(6, 3, halvable(step, 18));
            let column = packed.column_view(1);
            viewed.update(&mut a, &column);
            copied.update(&mut b, &Vector::new(column.to_matrix().as_slice().to_vec()));
            assert_eq!(
                bits(a.as_slice()),
                bits(b.as_slice()),
                "{name}, step {step}"
            );
        }
    }
}

#[test]
#[should_panic(expected = "a 4×3 gradient for 3×4 parameters")]
fn a_matrix_gradient_must_have_the_parameters_shape() {
    let mut parameters = Matrix::<f32>::from_flat(3, 4, vec![0.0; 12]);
    let transposed = Matrix::<f32>::from_flat(4, 3, vec![1.0; 12]);
    Sgd::new(0.1).update(&mut parameters, &transposed);
}

#[test]
#[should_panic(expected = "expected 1 inputs, got 2")]
fn a_parameter_program_run_by_parts_counts_its_operands() {
    use tensorcrate::optim::run_by_parts;
    use tensorcrate::tensors::fused::{Builder, DType, Decl};
    let mut b = Builder::<f32>::new();
    let g = b.input(Decl::scalar(DType::F32));
    let p = b.update(Decl::scalar(DType::F32));
    let p = b.sub(p, g);
    b.set(0, p);
    let program = b.build().unwrap();
    let mut parameter = 1.0f32;
    let _ = run_by_parts(&program, &[&1.0f32, &2.0f32], &mut [&mut parameter]);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_rules_read_views() {
    use tensorcrate::tensors::Host;
    for (name, mut gpu_rule) in every_rule::<Matrix<f32, Metal>>() {
        let (_, mut host_rule) = every_rule::<Matrix<f32>>()
            .into_iter()
            .find(|(other, _)| *other == name)
            .unwrap();
        let start = Matrix::from_flat(3, 4, (0..12).map(|i| i as f32 * 0.1).collect::<Vec<_>>());
        let (mut gpu, mut host) = (start.to_backend::<Metal>(), start);
        for step in 0..5 {
            let packed = Matrix::<f32, Host>::from_flat(5, 7, halvable(step, 35));
            let resident = packed.to_backend::<Metal>();
            gpu_rule.update(&mut gpu, &resident.view(1..4, 2..6));
            host_rule.update(&mut host, &packed.view(1..4, 2..6));
            for (g, h) in gpu
                .to_backend::<Host>()
                .as_slice()
                .iter()
                .zip(host.as_slice())
            {
                assert!(
                    close(*g, *h),
                    "{name}, step {step}: {g} on Metal, {h} on the host"
                );
            }
        }
    }
}
