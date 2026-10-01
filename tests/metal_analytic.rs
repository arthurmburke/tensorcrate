//! The Metal analytic kernels at the edges of the `f32` range.
//!
//! The shader is compiled with fast math, whose hyperbolic functions are built
//! from `exp` and overflow to `inf / inf = NaN` long before the true result
//! stops being finite. These tests pin the GPU to the host's `f32` functions
//! at infinities and large magnitudes as well as across ordinary inputs.

#![cfg(all(feature = "metal", target_os = "macos"))]

use tensorcrate::tensors::dual::DualVector;
use tensorcrate::tensors::{Analytic, Metal, Vector};

/// The inputs the tanh comparison has to survive: infinities, values well past
/// the point where `exp(2x)` overflows, values near saturation, zero, and an
/// ordinary sweep.
fn tanh_inputs() -> Vec<f32> {
    let mut inputs = vec![
        f32::INFINITY,
        f32::NEG_INFINITY,
        100.0,
        -100.0,
        20.0,
        -20.0,
        10.0,
        -10.0,
        9.0,
        -9.0,
        0.0,
        -0.0,
        1e-3,
        -1e-3,
        1e-6,
        -1e-6,
    ];
    inputs.extend((-400..=400).map(|i| i as f32 * 0.025));
    inputs
}

/// `actual` agrees with `expected` to about `1e-6` relative, with an absolute
/// floor for results near zero; infinities and NaN must match exactly.
fn agrees(actual: f32, expected: f32) -> bool {
    if expected.is_nan() {
        return actual.is_nan();
    }
    if expected.is_infinite() {
        return actual == expected;
    }
    (actual - expected).abs() <= 1e-6 * expected.abs().max(1.0)
}

fn compare(f: Analytic, inputs: &[f32]) -> Vec<String> {
    let tangents = vec![1.0f32; inputs.len()];
    let values = Vector::<f32>::new(inputs.to_vec())
        .to_backend::<Metal>()
        .analytic(f);
    let dual = DualVector::new(
        Vector::<f32>::new(inputs.to_vec()).to_backend::<Metal>(),
        Vector::<f32>::new(tangents).to_backend::<Metal>(),
    )
    .analytic(f);

    let mut failures = Vec::new();
    for (index, &x) in inputs.iter().enumerate() {
        let want = f.value(x);
        let got = values.as_slice()[index];
        if !agrees(got, want) {
            failures.push(format!("{f:?}({x}) = {got}, host {want}"));
        }
        let got = dual.value().as_slice()[index];
        if !agrees(got, want) {
            failures.push(format!("dual {f:?}({x}) = {got}, host {want}"));
        }
        let want = f.derivative(x);
        let got = dual.tangent().as_slice()[index];
        if !agrees(got, want) {
            failures.push(format!("{f:?}'({x}) = {got}, host {want}"));
        }
    }
    failures
}

#[test]
fn metal_tanh_saturates_like_the_host() {
    let failures = compare(Analytic::Tanh, &tanh_inputs());
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn metal_sinh_and_cosh_overflow_like_the_host() {
    // e^89.4 / 2 is still below f32::MAX; e^89.5 / 2 is not.
    let mut inputs = vec![
        f32::INFINITY,
        f32::NEG_INFINITY,
        100.0,
        -100.0,
        89.5,
        -89.5,
        89.4,
        -89.4,
        89.0,
        -89.0,
        20.0,
        -20.0,
        0.0,
        -0.0,
    ];
    inputs.extend((-400..=400).map(|i| i as f32 * 0.025));
    for f in [Analytic::Sinh, Analytic::Cosh] {
        let failures = compare(f, &inputs);
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}
