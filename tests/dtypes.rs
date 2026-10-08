//! Element types other than `f32`.
//!
//! The backend-generic layers — [`Kernels`], automatic differentiation, the
//! optimizers, statistics, projections and fused programs — are written over
//! any [`Real`] element: `f32`, `f64`, `f16` and `bf16`. Each test here runs the
//! same source at a wider (or narrower) type than `f32`, and checks something
//! `f32` could not give: a gradient right to
//! `1e-12`, a bit-for-bit match between a fused and an unfused `f16` step, a
//! sort in the total order of the type it is sorting.

use half::{bf16, f16};
use tensorcrate::numbers::Real;
use tensorcrate::optim::{AdaGrad, Adam, Momentum, Parameter, RmsProp, Rule, Sgd, minimize};
use tensorcrate::projections::{project_onto_ball, project_onto_box, project_onto_capped_simplex};
use tensorcrate::statistics::{AxisStatistics, Correction, Distribution, Statistics};
use tensorcrate::tensors::fused::{self, Builder, DType, Decl, Fusable, Mode, Program};
use tensorcrate::tensors::{
    Analytic, BinaryOp, Compare, DualMatrix, DualVector, Host, Kernels, Matrix, Ordered, Reduce,
    SortOrder, Tape, Transcendental, Vector, gradient, tape,
};

#[cfg(all(feature = "metal", target_os = "macos"))]
use tensorcrate::tensors::Metal;

// ---- Kernels, written once -------------------------------------------------

/// `relu(v)` capped at `cap`, then normalized to a unit sum — three different
/// kernel families, and not a mention of the element type.
fn capped_normalized<T: Real, B: Kernels<T>>(v: &Vector<T, B>, cap: T) -> Vector<T, B> {
    let capped = v.max_scalar(T::zero()).min_scalar(cap);
    let total = B::vector_reduce(&capped, Reduce::Sum);
    B::vector_broadcast(&capped, total, BinaryOp::Div, false)
}

#[test]
fn one_generic_function_runs_at_every_element_type() {
    let values = [-1.0, 0.5, 3.0, 9.0];

    let f64s = capped_normalized::<f64, Host>(&Vector::new(values), 4.0);
    assert_eq!(f64s.data(), [0.0, 0.5 / 7.5, 3.0 / 7.5, 4.0 / 7.5]);

    let f32s = capped_normalized::<f32, Host>(&Vector::new(values.map(|x| x as f32)), 4.0);
    assert_eq!(
        f32s.data(),
        [0.0, 0.5 / 7.5, 3.0 / 7.5, 4.0 / 7.5].map(|x| x as f32)
    );

    // The compact types run the same code, rounding after every operation.
    let halves =
        capped_normalized::<f16, Host>(&Vector::new(values.map(f16::from_f64)), f16::from_f64(4.0));
    let expected = values.map(|x: f64| f16::from_f64(x.clamp(0.0, 4.0)));
    let total = expected.iter().fold(f16::ZERO, |sum, &x| sum + x);
    assert_eq!(halves.data(), expected.map(|x| x / total));
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn the_f32_instantiation_agrees_with_metal() {
    let values = Vector::new([-1.0f32, 0.5, 3.0, 9.0]);
    let host = capped_normalized::<f32, Host>(&values, 4.0);
    let resident = capped_normalized::<f32, Metal>(&values.to_backend::<Metal>(), 4.0);
    for (a, b) in host.data().iter().zip(resident.to_backend::<Host>().data()) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn comparisons_and_reductions_are_exact_in_f64() {
    let a = Vector::new([1.0f64, 2.0, f64::NAN, -0.0]);
    let b = Vector::new([2.0f64, 2.0, 1.0, 0.0]);

    assert_eq!(a.compare(&b, Compare::Less).data(), [1.0, 0.0, 0.0, 0.0]);
    assert_eq!(
        a.compare(&b, Compare::LessEqual).data(),
        [1.0, 1.0, 0.0, 1.0]
    );
    // A number beats a NaN, as in `f64::min`.
    assert_eq!(a.compare(&b, Compare::Min).data()[2], 1.0);
    assert_eq!(
        a.compare_scalar(1.5, Compare::Greater, true).data(),
        [1.0, 0.0, 0.0, 1.0]
    );

    let long = Vector::new((0..1000).map(|i| (i % 17) as f64 - 8.0).collect::<Vec<_>>());
    assert_eq!(long.reduce(Reduce::Min), -8.0);
    assert_eq!(long.reduce(Reduce::Max), 8.0);
    assert_eq!(long.reduce(Reduce::Sum), long.data().iter().sum::<f64>());
    assert_eq!(
        Vector::<f64>::new(vec![]).reduce(Reduce::Min),
        f64::INFINITY
    );
}

#[test]
fn sorting_uses_the_total_order_of_the_element_type() {
    let v = Vector::new([2.0f64, -0.0, f64::NAN, 0.0, -1.0, f64::NEG_INFINITY]);
    let ascending = v.sorted(SortOrder::Ascending);
    let bits: Vec<u64> = ascending.data().iter().map(|x| x.to_bits()).collect();
    let mut expected = v.to_vec();
    expected.sort_by(|a, b| a.total_cmp(b));
    assert_eq!(
        bits,
        expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    // −0.0 precedes +0.0, and the NaN sorts to the end.
    assert!(ascending[1].is_sign_negative() || ascending[2].is_sign_negative());
    assert!(ascending[5].is_nan());

    let descending = v.sorted(SortOrder::Descending);
    assert!(descending[0].is_nan());

    // Through the trait, on a compact type.
    fn through_the_trait<T: Real, B: Kernels<T>>(v: &Vector<T, B>) -> Vector<T, B> {
        v.sorted(SortOrder::Descending)
    }
    let h = Vector::new([1.0, 3.0, 2.0].map(bf16::from_f64));
    assert_eq!(
        through_the_trait::<bf16, Host>(&h).data(),
        [3.0, 2.0, 1.0].map(bf16::from_f64)
    );
}

#[test]
fn matrices_compare_in_f64() {
    let a = Matrix::<f64>::from_rows([[1.0, 5.0], [3.0, 2.0]]);
    let b = Matrix::<f64>::from_rows([[2.0, 4.0], [3.0, 1.0]]);
    assert_eq!(
        a.compare(&b, Compare::Greater).to_rows(),
        [[0.0, 1.0], [0.0, 1.0]]
    );
    assert_eq!(
        a.compare_scalar(2.0, Compare::GreaterEqual, false)
            .to_rows(),
        [[0.0, 1.0], [1.0, 1.0]]
    );
}

#[test]
fn ramps_are_generic_over_the_element() {
    assert_eq!(
        Vector::<f64>::ramp(4, 1.0, 0.5).data(),
        [1.0, 1.5, 2.0, 2.5]
    );
    assert_eq!(
        Vector::<f16>::ramp(3, f16::from_f64(0.0), f16::from_f64(0.25)).data(),
        [0.0, 0.25, 0.5].map(f16::from_f64)
    );
}

#[test]
fn zip_map_combines_elementwise_and_checks_lengths() {
    let a = Vector::new([1.0f64, 2.0]);
    let b = Vector::new([10i32, 20]);
    assert_eq!(a.zip_map(&b, |&x, &y| x * y as f64).data(), [10.0, 40.0]);
    let m = Matrix::<f64>::from_rows([[1.0, 2.0]]);
    assert_eq!(m.zip_map(&m, |x, y| x + y).to_rows(), [[2.0, 4.0]]);
}

#[test]
#[should_panic(expected = "lengths differ")]
fn zip_map_rejects_mismatched_lengths() {
    Vector::new([1.0f64]).zip_map(&Vector::new([1.0f64, 2.0]), |a, b| a + b);
}

// ---- the transcendental surface ---------------------------------------------

#[test]
fn analytic_functions_run_in_every_element_type() {
    let x = Vector::new([0.0f64, 1.0, 2.0]);
    let exp = x.analytic(Analytic::Exp);
    for (got, want) in exp
        .data()
        .iter()
        .zip([1.0, std::f64::consts::E, 7.38905609893065])
    {
        assert!((got - want).abs() < 1e-14);
    }
    assert_eq!(x.sqrt().data(), x.map(|v| v.sqrt()).data());

    // Powers: the fast exponents stay correctly rounded in f64 too.
    assert_eq!(Transcendental::pow(&x, 2.0).data(), [0.0, 1.0, 4.0]);

    // The compact types evaluate in f32 and round once.
    let h = Vector::new([0.0, 1.0].map(f16::from_f64));
    assert_eq!(h.exp().data(), [1.0, 2.71875].map(f16::from_f64));
    let b = Matrix::<bf16>::from_rows([[bf16::from_f64(4.0)]]);
    assert_eq!(b.sqrt().to_rows(), [[bf16::from_f64(2.0)]]);
}

// ---- statistics -----------------------------------------------------------------

fn standardized<T: Real, B: Kernels<T>>(v: &Vector<T, B>) -> Vector<T, B> {
    let fitted = Distribution::Normal {
        mean: v.mean(),
        stddev: v.stddev(Correction::Sample),
    };
    v.cdf(&fitted)
}

#[test]
fn the_statistics_traits_are_generic_over_the_element() {
    let v = Vector::new([1.0f64, 2.0, 3.0]);
    // The middle value of a symmetric sample sits at the median.
    assert!((standardized::<f64, Host>(&v).data()[1] - 0.5).abs() < 1e-15);
    assert_eq!(Statistics::mean(&v), 2.0);
    assert_eq!(Statistics::variance(&v, Correction::Sample), 1.0);

    let m = Matrix::<f64>::from_rows([[1.0, 10.0], [3.0, 30.0]]);
    let means = AxisStatistics::mean_axis(&m, tensorcrate::statistics::Axis::Rows);
    assert_eq!(means.data(), [5.5, 16.5]);
    let fits = AxisStatistics::fit_axis(
        &m,
        tensorcrate::statistics::Axis::Columns,
        tensorcrate::statistics::Family::Normal,
        Correction::Population,
    );
    assert_eq!(fits.len(), 2);
    let transformed = AxisStatistics::cdf_axis(&m, tensorcrate::statistics::Axis::Columns, &fits);
    assert!(transformed.data().iter().all(|p| (0.0..=1.0).contains(p)));
}

// ---- forward and reverse differentiation ----------------------------------

#[test]
fn forward_mode_gradients_are_exact_in_f64() {
    // ∇ Σ sin(xᵢ)² = sin(2xᵢ). In f32 this would be good to about 1e-7.
    let at = Vector::new([0.3f64, 1.1, -0.7]);
    let grad = gradient(&at, |x| x.sin().dot(&x.sin()));
    for (g, x) in grad.data().iter().zip(at.data()) {
        assert!(
            (g - (2.0 * x).sin()).abs() < 1e-14,
            "{g} vs {}",
            (2.0 * x).sin()
        );
    }
}

#[test]
fn dual_tensors_carry_any_element() {
    let a = Matrix::<f64>::from_rows([[1.0, 2.0], [3.0, 4.0]]);
    let tangent = Matrix::<f64>::identity(2);
    let squared = DualMatrix::new(a.clone(), tangent).squared();
    assert_eq!(squared.value().to_rows(), a.matmul(&a).to_rows());
    // d(A·A)/dt along I is A + A.
    assert_eq!(squared.tangent().to_rows(), [[2.0, 4.0], [6.0, 8.0]]);

    let v = DualVector::<Host, f64>::seed(Vector::new([2.0, 3.0]), 0);
    let y = (&v * &v).sum();
    assert_eq!((y.real, y.dual), (13.0, 4.0));

    let widened: DualVector<Host, f64> = DualVector::constant(Vector::new([1.5f64]));
    assert_eq!(widened.tangent().data(), [0.0]);
}

#[test]
fn reverse_mode_runs_in_f64() {
    let tape = Tape::<Host>::new();
    let a = tape.matrix(Matrix::<f64>::from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
    let x = tape.vector(Vector::new([1.0f64, 0.5, -1.0]));
    let mapped = a.matvec(&x);
    let loss = mapped.dot(&mapped);
    loss.backward();

    let projected = a.value().matvec(x.value());
    for row in 0..2 {
        for col in 0..3 {
            let expected = 2.0 * projected[row] * x.value()[col];
            assert!((a.grad()[(row, col)] - expected).abs() < 1e-14);
        }
    }

    // Scalars are tape values too, and one tape may mix element types.
    let s = tape.scalar(3.0f64);
    let t = tape.scalar(2.0f32);
    let product = &s * &s;
    product.backward();
    assert_eq!(s.grad(), 6.0);
    assert_eq!(t.grad(), 0.0f32);
}

#[test]
fn reverse_gradient_and_jacobian_are_generic() {
    let at = Vector::new([0.5f64, 2.0]);
    let grad = tape::gradient(&at, |x| x.exp().sum());
    assert!((grad[0] - 0.5f64.exp()).abs() < 1e-14);
    assert!((grad[1] - 2.0f64.exp()).abs() < 1e-14);

    let jac = tape::jacobian(&at, |x| x.sin());
    assert!((jac[(0, 0)] - 0.5f64.cos()).abs() < 1e-14);
    assert_eq!(jac[(0, 1)], 0.0);
}

#[test]
fn nonsmooth_rules_keep_their_conventions_at_f64() {
    // |x| differentiates to sign(x), with 0 at the kink.
    let tape = Tape::<Host>::new();
    let x = tape.vector(Vector::new([-2.0f64, 0.0, 3.0]));
    x.abs().sum().backward();
    assert_eq!(x.grad().data(), [-1.0, 0.0, 1.0]);

    let tape = Tape::<Host>::new();
    let s = tape.scalar(0.0f64);
    s.relu().backward();
    assert_eq!(s.grad(), 0.5);
}

// ---- optimizers -----------------------------------------------------------------

/// `Σ scaleᵢ(xᵢ − centreᵢ)²`, whose minimum is exactly `centre`.
fn bowl<'t, T: tensorcrate::tensors::fused::Element>(
    x: &tensorcrate::tensors::VectorVar<'t, Host, T>,
    centre: &Vector<T, Host>,
    curvature: &Vector<T, Host>,
) -> tensorcrate::tensors::ScalarVar<'t, Host, T> {
    let tape = x.tape();
    let offset = x - &tape.vector(centre.clone());
    (&offset * &offset).dot(&tape.vector(curvature.clone()))
}

#[test]
fn adam_converges_to_double_precision() {
    let centre = Vector::new([0.5f64, -0.25, 2.0]);
    let curvature = Vector::new([1.0f64, 4.0, 0.5]);
    let mut parameters = Vector::<f64>::zeros(3);
    let mut rule = Adam::new(0.05);
    let loss = minimize(&mut parameters, &mut rule, 2000, |x, _| {
        bowl(x, &centre, &curvature)
    });
    // Well past what f32 can represent of the distance to the optimum.
    assert!(loss < 1e-12, "loss {loss}");
    for (p, c) in parameters.data().iter().zip(centre.data()) {
        assert!((p - c).abs() < 1e-5, "{p} vs {c}");
    }
}

#[test]
fn every_rule_descends_in_f64() {
    let centre = Vector::new([1.0f64, -1.0]);
    let curvature = Vector::new([1.0f64, 2.0]);
    let start = Vector::<f64>::zeros(2);
    let initial = bowl_value(&start, &centre, &curvature);

    let mut sgd_parameters = start.clone();
    minimize(&mut sgd_parameters, &mut Sgd::new(0.1), 200, |x, _| {
        bowl(x, &centre, &curvature)
    });
    assert!(bowl_value(&sgd_parameters, &centre, &curvature) < 1e-10 * (1.0 + initial));

    let mut momentum_parameters = start.clone();
    minimize(
        &mut momentum_parameters,
        &mut Momentum::nesterov(0.05, 0.9),
        400,
        |x, _| bowl(x, &centre, &curvature),
    );
    assert!(bowl_value(&momentum_parameters, &centre, &curvature) < 1e-8);

    let mut adagrad_parameters = start.clone();
    minimize(
        &mut adagrad_parameters,
        &mut AdaGrad::new(0.5),
        2000,
        |x, _| bowl(x, &centre, &curvature),
    );
    assert!(bowl_value(&adagrad_parameters, &centre, &curvature) < 1e-4);

    let mut rms_parameters = start;
    minimize(
        &mut rms_parameters,
        &mut RmsProp::new(0.01),
        2000,
        |x, _| bowl(x, &centre, &curvature),
    );
    assert!(bowl_value(&rms_parameters, &centre, &curvature) < 1e-3);
}

fn bowl_value(x: &Vector<f64>, centre: &Vector<f64>, curvature: &Vector<f64>) -> f64 {
    x.data()
        .iter()
        .zip(centre.data())
        .zip(curvature.data())
        .map(|((x, c), k)| k * (x - c) * (x - c))
        .sum()
}

#[test]
fn scalar_parameters_are_any_element_type() {
    // A scalar `f64` is a parameter: minimize (t − 3)².
    let mut t = 0.0f64;
    let mut rule = Adam::new(0.1);
    minimize(&mut t, &mut rule, 500, |t, _| {
        let offset = t.shift(-3.0);
        &offset * &offset
    });
    assert!((t - 3.0).abs() < 1e-3, "{t}");
}

#[test]
fn matrix_parameters_and_hyperparameters_follow_the_element() {
    let target = Matrix::<f64>::from_rows([[1.0, 2.0], [3.0, 4.0]]);
    let mut weights = Matrix::<f64>::zeros(2, 2);
    let mut rule = Adam::new(0.1);
    minimize(&mut weights, &mut rule, 1500, |w, _| {
        let diff = w - &w.tape().matrix(target.clone());
        diff.frobenius_dot(&diff)
    });
    for (w, t) in weights.data().iter().zip(target.data()) {
        assert!((w - t).abs() < 1e-3);
    }
    // The hyperparameters are the element type.
    let _: f64 = rule.rate;
    let _: f64 = rule.epsilon;
}

/// The fused step and the one-op-at-a-time definition must agree bit for bit
/// at every element type: fusion changes where intermediates live, and nothing
/// else.
fn fused_equals_unfused<T>(rate: T, rounds: usize)
where
    T: fused::Element,
    Vector<T, Host>: Parameter<Elem = T, Backend = Host> + Clone,
{
    let len = 300;
    let gradient = |round: usize| {
        Vector::new(
            (0..len)
                .map(|i| T::from_f64(((i + round * 7) as f64 * 0.37).sin()))
                .collect::<Vec<_>>(),
        )
    };
    let start = Vector::new(
        (0..len)
            .map(|i| T::from_f64(i as f64 * 0.01))
            .collect::<Vec<_>>(),
    );

    let mut fused_parameters = start.clone();
    let mut fused_rule = Adam::<Vector<T, Host>>::new(rate);
    for round in 0..rounds {
        fused_rule.update(&mut fused_parameters, gradient(round).as_gradient());
    }

    let mut plain_parameters = start;
    let mut plain_rule = Adam::<Vector<T, Host>>::new(rate);
    fused::with_mode(Mode::Unfused, || {
        for round in 0..rounds {
            plain_rule.update(&mut plain_parameters, gradient(round).as_gradient());
        }
    });

    let bits = |v: &Vector<T, Host>| {
        v.data()
            .iter()
            .map(|x| x.into_f64().to_bits())
            .collect::<Vec<_>>()
    };
    assert_eq!(bits(&fused_parameters), bits(&plain_parameters));
}

#[test]
fn fused_steps_match_unfused_steps_in_every_element_type() {
    fused_equals_unfused::<f32>(0.01, 8);
    fused_equals_unfused::<f64>(0.01, 8);
    fused_equals_unfused::<f16>(f16::from_f64(0.01), 4);
    fused_equals_unfused::<bf16>(bf16::from_f64(0.01), 4);
}

// ---- fused programs ---------------------------------------------------------------

#[test]
fn a_program_can_compute_in_f64() {
    // y = sqrt(a·b + 1e-12): the constant is below f32's resolution of 1.
    let mut b = Builder::<f64>::new();
    let vector = Decl::vector(DType::F64, 3);
    let (x, w) = (b.input(vector.clone()), b.input(vector));
    let product = b.mul(x, w);
    let shifted = b.shift(product, 1e-12);
    let root = b.unary(Analytic::Sqrt, shifted);
    b.output(root, DType::F64);
    let program: Program<f64> = b.build().unwrap();

    let a = Vector::new([3.0f64, 0.0, 1e-6]);
    let w = Vector::new([1.0f64, 5.0, 1e-6]);
    let y = program.run_vectors(&[&a, &w]).remove(0);
    for ((got, a), w) in y.data().iter().zip(a.data()).zip(w.data()) {
        assert_eq!(*got, (a * w + 1e-12).sqrt());
    }
}

#[test]
fn storage_and_arithmetic_types_are_independent() {
    // f64 arithmetic over f32 and f16 storage, narrowed to f32 and bf16.
    let mut b = Builder::<f64>::new();
    let (x, h) = (
        b.input(Decl::vector(DType::F32, 3)),
        b.input(Decl::vector(DType::F16, 3)),
    );
    let sum = b.add(x, h);
    let scaled = b.scale(sum, 1e-3);
    b.output(scaled, DType::F32);
    b.output(scaled, DType::Bf16);
    b.output(scaled, DType::F64);
    let program = b.build().unwrap();

    let x = Vector::new([1.0f32, 2.0, 3.0]);
    let h = Vector::new([0.5f64, 0.25, 0.125].map(f16::from_f64));
    let inputs: [&dyn Fusable<Host>; 2] = [&x, &h];
    let mut outputs = program.run::<Host>(&inputs, &mut []);
    let exact: Vec<f64> = [1.5, 2.25, 3.125].iter().map(|v| v * 1e-3).collect();

    let wide = outputs.remove(2).into_vector::<f64>();
    assert_eq!(wide.data(), exact);
    let bf = outputs.remove(1).into_vector::<bf16>();
    assert_eq!(
        bf.data(),
        exact.iter().map(|&v| bf16::from_f64(v)).collect::<Vec<_>>()
    );
    let single = outputs.remove(0).into_vector::<f32>();
    assert_eq!(
        single.data(),
        exact.iter().map(|&v| v as f32).collect::<Vec<_>>()
    );
}

#[test]
fn f64_programs_match_the_unfused_oracle_bit_for_bit() {
    // A long vector, crossing the SIMD thresholds and the interpreter's tiles.
    let len = 3001;
    let a = Vector::new(
        (0..len)
            .map(|i| ((i as f64) * 0.013).sin() * 4.0)
            .collect::<Vec<_>>(),
    );
    let c = Vector::new(
        (0..len)
            .map(|i| ((i as f64) * 0.007).cos() + 1.5)
            .collect::<Vec<_>>(),
    );

    let mut b = Builder::<f64>::new();
    let vector = Decl::vector(DType::F64, len);
    let (x, y) = (b.input(vector.clone()), b.input(vector));
    let ratio = b.div(x, y);
    let kept = b.compare(Compare::Max, ratio, x);
    let mask = b.compare(Compare::Greater, kept, y);
    let root = b.unary(Analytic::Tanh, kept);
    let masked = b.mul(root, mask);
    b.output(masked, DType::F64);
    b.output(kept, DType::F64);
    let program = b.build().unwrap();

    let fused_out = program.run_vectors::<Host>(&[&a, &c]);
    let unfused_out = fused::with_mode(Mode::Unfused, || program.run_vectors::<Host>(&[&a, &c]));
    for (f, u) in fused_out.iter().zip(&unfused_out) {
        let bits = |v: &Vector<f64>| v.data().iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(f), bits(u));
    }
}

#[test]
fn a_program_with_f64_storage_displays_and_validates() {
    let mut b = Builder::<f64>::new();
    let x = b.input(Decl::vector(DType::F64, 10));
    let one = b.constant(0.1);
    let y = b.add(x, one);
    b.output(y, DType::F64);
    let program = b.build().unwrap();
    let listing = program.to_string();
    assert!(listing.contains("f64"), "{listing}");
    assert!(listing.contains("0.1"), "{listing}");
    assert_eq!(program.bytes(10), 10 * (8 + 8));
}

// ---- projections --------------------------------------------------------------

#[test]
fn projections_are_generic_over_the_element() {
    let weights = Vector::<f64>::new([0.7, 0.5, -0.2]);
    let projected = project_onto_capped_simplex(&weights, 1.0);
    assert!(projected.data().iter().all(|&w| w >= 0.0));
    assert!((projected.sum() - 1.0).abs() < 1e-15);

    let boxed = project_onto_box(&weights, -0.1, 0.6);
    assert_eq!(boxed.data(), [0.6, 0.5, -0.1]);

    let ball = project_onto_ball(&Vector::new([3.0f64, 4.0]), 1.0);
    assert!((ball[0] - 0.6).abs() < 1e-15 && (ball[1] - 0.8).abs() < 1e-15);

    // The compact types project too, within their resolution.
    let halves = Vector::new([0.7, 0.5, -0.2].map(f16::from_f64));
    let projected = project_onto_capped_simplex(&halves, f16::from_f64(1.0));
    assert!(projected.data().iter().all(|&w| w >= f16::ZERO));
    assert!((f64::from(projected.sum()) - 1.0).abs() < 2e-3);
}

#[test]
#[should_panic(expected = "exceed the integers")]
fn a_projection_refuses_a_vector_longer_than_the_element_can_index() {
    // bf16 counts exactly only to 2⁸.
    let long = Vector::<bf16>::filled(1000, bf16::from_f64(1.0));
    project_onto_capped_simplex(&long, bf16::from_f64(1.0));
}

// ---- backend helpers --------------------------------------------------------------

#[test]
fn stacking_works_for_any_element() {
    let rows = [Vector::new([1.0f64, 2.0]), Vector::new([3.0, 4.0])];
    assert_eq!(Vector::vstack(&rows).data(), [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(Vector::hstack(&rows).data(), [1.0, 3.0, 2.0, 4.0]);

    let (a, b) = (
        Matrix::from_flat(2, 2, [1u8, 2, 3, 4]),
        Matrix::from_flat(2, 1, [5u8, 6]),
    );
    assert_eq!(Matrix::hstack([&a, &b]).data(), [1, 2, 5, 3, 4, 6]);
    let (a, b) = (
        Matrix::from_flat(2, 2, [1u8, 2, 3, 4]),
        Matrix::from_flat(1, 2, [5u8, 6]),
    );
    assert_eq!(Matrix::vstack([&a, &b]).data(), [1, 2, 3, 4, 5, 6]);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_stacks_non_f32_elements_on_the_device() {
    let a = Vector::new([1.0f64, 2.0]).to_backend::<Metal>();
    let b = Vector::new([3.0f64, 4.0]).to_backend::<Metal>();
    let rows = Vector::vstack([&a, &b]);
    let columns = Vector::hstack([&a, &b]);
    assert!(rows.is_device_resident());
    assert!(columns.is_device_resident());
    assert_eq!(rows.as_slice(), [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(columns.as_slice(), [1.0, 3.0, 2.0, 4.0]);
}

// ---- persistence -----------------------------------------------------------------

#[test]
fn the_compact_floats_round_trip_through_a_file() {
    let v = Vector::new([1.5, -0.25, 3.0].map(f16::from_f64));
    let mut bytes = Vec::new();
    v.write_to(&mut bytes).unwrap();
    assert_eq!(Vector::<f16>::read_from(&bytes[..]).unwrap(), v);
    // …and the tag keeps them apart from the types they might be confused with.
    assert!(Vector::<bf16>::read_from(&bytes[..]).is_err());
    assert!(Vector::<f32>::read_from(&bytes[..]).is_err());

    let m = Matrix::from_rows([[bf16::from_f64(1.0), bf16::from_f64(2.0)]]);
    let mut bytes = Vec::new();
    m.write_to(&mut bytes).unwrap();
    assert_eq!(Matrix::<bf16>::read_from(&bytes[..]).unwrap(), m);
    let message = Matrix::<f16>::read_from(&bytes[..])
        .unwrap_err()
        .to_string();
    assert!(message.contains("bf16"), "{message}");
}
