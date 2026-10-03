//! Descriptive statistics and distribution functions.

use tensorcrate::statistics::special;
use tensorcrate::statistics::{Axis, Correction, Distribution, Family, Statistic};
use tensorcrate::tensors::{Matrix, Vector};

fn close(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() <= tolerance * b.abs().max(1.0)
}

// ---- independent oracles ----------------------------------------------------
//
// The production error function is a rational approximation with several dozen
// tabulated coefficients, and a single mistyped digit would be invisible in a
// spot check against a handful of textbook values. These two are slow and
// derived from the definitions rather than from a table, so they agree with the
// production code only if its coefficients are right.

/// `erf` by its non-alternating series, which has no cancellation:
/// `erf(x) = 2x·e^(−x²)/√π · Σ (2x²)ⁿ / (1·3·⋯·(2n+1))`.
fn erf_by_series(x: f64) -> f64 {
    let square = x * x;
    let mut term = 1.0;
    let mut total = 1.0;
    for n in 1..1_000 {
        term *= 2.0 * square / (2.0 * n as f64 + 1.0);
        total += term;
        if term < 1e-20 * total {
            break;
        }
    }
    2.0 * x * (-square).exp() / std::f64::consts::PI.sqrt() * total
}

/// `erfc` by its continued fraction, evaluated backwards from a deep tail. This
/// is the form that stays accurate where the series above has thrown away every
/// significant digit.
fn erfc_by_continued_fraction(x: f64) -> f64 {
    let mut tail = 0.0;
    for n in (1..=400).rev() {
        tail = (n as f64 / 2.0) / (x + tail);
    }
    (-x * x).exp() / std::f64::consts::PI.sqrt() / (x + tail)
}

#[test]
fn the_error_function_matches_its_series_and_continued_fraction() {
    let mut x: f64 = -6.0;
    while x <= 6.0 {
        if x.abs() > 1e-9 {
            assert!(
                close(special::erf(x), erf_by_series(x), 1e-13),
                "erf({x}) = {} against {}",
                special::erf(x),
                erf_by_series(x)
            );
        }
        x += 0.01;
    }

    let mut x: f64 = 1.0;
    while x <= 25.0 {
        let expected = erfc_by_continued_fraction(x);
        assert!(
            close(special::erfc(x), expected, 1e-12),
            "erfc({x}) = {} against {expected}",
            special::erfc(x)
        );
        // The logarithm has to stay usable well past the point where the
        // function itself underflows to zero.
        assert!(close(special::ln_erfc(x), expected.ln(), 1e-13));
        x += 0.05;
    }

    assert_eq!(special::erf(0.0), 0.0);
    assert_eq!(special::erfc(0.0), 1.0);
    assert!(special::erf(f64::NAN).is_nan());
    assert!(special::erfc(-30.0) == 2.0);
    assert!(special::erfc(30.0) == 0.0);
    assert!(special::ln_erfc(30.0) < -900.0 && special::ln_erfc(30.0).is_finite());
}

#[test]
fn the_normal_functions_match_published_values() {
    // Reference values for Φ and Φ⁻¹, correct to every digit shown.
    let checks = [
        (0.0, 0.5),
        (1.0, 0.841_344_746_068_542_9),
        (-1.0, 0.158_655_253_931_457_1),
        (1.96, 0.975_002_104_851_779_5),
        (-3.0, 0.001_349_898_031_630_095),
    ];
    for (z, expected) in checks {
        assert!(
            close(special::standard_normal_cdf(z), expected, 1e-14),
            "Φ({z}) = {}",
            special::standard_normal_cdf(z)
        );
    }

    let quantiles = [
        (0.95, 1.644_853_626_951_472_2),
        (0.975, 1.959_963_984_540_054),
        (0.99, 2.326_347_874_040_841),
        (0.5, 0.0),
    ];
    for (p, expected) in quantiles {
        assert!(
            close(special::standard_normal_ppf(p), expected, 1e-14),
            "Φ⁻¹({p}) = {}",
            special::standard_normal_ppf(p)
        );
    }

    assert_eq!(special::standard_normal_ppf(0.0), f64::NEG_INFINITY);
    assert_eq!(special::standard_normal_ppf(1.0), f64::INFINITY);
    assert!(special::standard_normal_ppf(1.5).is_nan());
    assert!(special::standard_normal_ppf(-0.5).is_nan());
}

#[test]
fn the_normal_quantile_inverts_the_distribution_function() {
    // The two are separate approximations with unrelated coefficients, so
    // agreeing across the whole interval — and far into the tails, where the
    // quantile switches branches — checks both at once.
    for i in 1..10_000 {
        let p = i as f64 / 10_000.0;
        let round_trip = special::standard_normal_cdf(special::standard_normal_ppf(p));
        assert!(
            close(round_trip, p, 1e-13),
            "p = {p} came back as {round_trip}"
        );
    }
    for p in [1e-300, 1e-100, 1e-20, 1e-8, 1e-3] {
        let round_trip = special::standard_normal_cdf(special::standard_normal_ppf(p));
        assert!(
            close(round_trip, p, 1e-12),
            "p = {p} came back as {round_trip}"
        );
    }

    assert!(close(special::erf_inv(special::erf(0.75)), 0.75, 1e-14));
    assert!(close(special::erf(special::erf_inv(0.25)), 0.25, 1e-14));
}

#[test]
fn the_inverse_gaussian_agrees_with_its_own_density_and_quantile() {
    for (mean, shape) in [
        (1.0, 1.0),
        (1.0, 3.0),
        (2.0, 0.5),
        (0.5, 20.0),
        (1.0, 100.0),
    ] {
        // The quantile has no closed form and is solved for, so inverting the
        // distribution function is the check that it converged.
        for i in 1..500 {
            let p = i as f64 / 500.0;
            let x = special::inverse_gaussian_ppf(p, mean, shape);
            let round_trip = special::inverse_gaussian_cdf(x, mean, shape);
            assert!(
                close(round_trip, p, 1e-12),
                "IG({mean}, {shape}) quantile at {p} came back as {round_trip}"
            );
        }

        // And the distribution function is in turn the integral of the density,
        // which nothing else here depends on.
        let median = special::inverse_gaussian_ppf(0.5, mean, shape);
        let steps = 100_000;
        let width = median / steps as f64;
        let mut area = 0.0;
        for step in 0..steps {
            area += special::inverse_gaussian_pdf((step as f64 + 0.5) * width, mean, shape) * width;
        }
        assert!(
            close(area, 0.5, 1e-6),
            "IG({mean}, {shape}) integrated to {area}"
        );
    }

    assert_eq!(special::inverse_gaussian_cdf(0.0, 1.0, 1.0), 0.0);
    assert_eq!(special::inverse_gaussian_pdf(-1.0, 1.0, 1.0), 0.0);
    assert_eq!(special::inverse_gaussian_ppf(0.0, 1.0, 1.0), 0.0);
    assert_eq!(special::inverse_gaussian_ppf(1.0, 1.0, 1.0), f64::INFINITY);
    assert!(special::inverse_gaussian_cdf(1.0, -1.0, 1.0).is_nan());

    // A shape far larger than the mean is where the naive `exp(2λ/μ)·Φ(−b)`
    // becomes `∞ · 0`; in logarithms the term is ordinary.
    let extreme = special::inverse_gaussian_cdf(1.0, 1.0, 1e6);
    assert!(extreme.is_finite() && (0.0..=1.0).contains(&extreme));
    assert!(close(extreme, 0.5, 1e-3));
}

// ---- moments ----------------------------------------------------------------

#[test]
fn moments_are_taken_over_every_element() {
    let v = Vector::new([2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
    assert_eq!(v.mean(), 5.0);
    assert_eq!(v.variance(Correction::Population), 4.0);
    assert_eq!(v.stddev(Correction::Population), 2.0);

    // n − 1 rather than n: 32/7 rather than 32/8.
    assert!(close(v.variance(Correction::Sample), 32.0 / 7.0, 1e-15));

    let moments = v.moments();
    assert_eq!(moments.count, 8);
    assert_eq!(moments.sum_squared_deviations, 32.0);

    let m = Matrix::<f64>::from_rows([[1.0, 2.0], [3.0, 4.0]]);
    assert_eq!(m.mean(), 2.5);
    assert_eq!(m.variance(Correction::Population), 1.25);
    assert!(close(
        m.stddev(Correction::Sample),
        (5.0_f64 / 3.0).sqrt(),
        1e-15
    ));
}

#[test]
fn a_sample_correction_needs_two_values_and_a_mean_needs_one() {
    assert_eq!(Correction::Population.divisor(4), Some(4));
    assert_eq!(Correction::Sample.divisor(4), Some(3));
    assert_eq!(Correction::Population.divisor(0), None);
    assert_eq!(Correction::Sample.divisor(1), None);

    let single = Vector::new([7.0_f64]);
    assert_eq!(single.mean(), 7.0);
    assert_eq!(single.variance(Correction::Population), 0.0);
    assert!(single.variance(Correction::Sample).is_nan());

    let empty = Vector::<f64>::zeros(0);
    assert!(empty.mean().is_nan());
    assert!(empty.variance(Correction::Population).is_nan());
}

#[test]
fn axis_moments_fold_the_axis_they_name() {
    let m = Matrix::<f64>::from_rows([
        [1.0, 2.0, 3.0, 4.0],
        [10.0, 20.0, 30.0, 40.0],
        [-1.0, -2.0, -3.0, -4.0],
    ]);

    // Three rows fold to three results, four columns to four.
    assert_eq!(m.mean_axis(Axis::Rows).to_vec(), [2.5, 25.0, -2.5]);
    assert_eq!(
        m.mean_axis(Axis::Columns).to_vec(),
        [10.0 / 3.0, 20.0 / 3.0, 10.0, 40.0 / 3.0]
    );
    assert_eq!(m.moments_axis(Axis::Rows).count, 4);
    assert_eq!(m.moments_axis(Axis::Columns).count, 3);

    // Against the same reduction taken one slice at a time.
    for (row, expected) in m.mean_axis(Axis::Rows).data().iter().enumerate() {
        assert_eq!(Vector::new(m.row(row).to_vec()).mean(), *expected);
    }
    for col in 0..m.cols() {
        let column: Vec<f64> = (0..m.rows()).map(|row| m[(row, col)]).collect();
        let column = Vector::new(column);
        assert!(close(
            m.mean_axis(Axis::Columns).data()[col],
            column.mean(),
            1e-15
        ));
        assert!(close(
            m.stddev_axis(Axis::Columns, Correction::Sample).data()[col],
            column.stddev(Correction::Sample),
            1e-14
        ));
        assert!(close(
            m.variance_axis(Axis::Columns, Correction::Population)
                .data()[col],
            column.variance(Correction::Population),
            1e-14
        ));
    }
}

#[test]
fn the_vectorized_moments_agree_with_a_scalar_fold() {
    // Long enough and offset far enough from zero that both the SIMD dispatch
    // and the two-pass variance matter: a one-pass `E[x²] − E[x]²` over these
    // values loses about nine of the sixteen digits.
    let values: Vec<f64> = (0..1_000)
        .map(|i| 1.0e6 + (i as f64 * 0.37).sin())
        .collect();
    let v = Vector::new(values.clone());

    let mut total = 0.0;
    for &value in &values {
        total += value;
    }
    let mean = total / values.len() as f64;
    let mut deviations = 0.0;
    for &value in &values {
        deviations += (value - mean) * (value - mean);
    }

    assert!(close(v.mean(), mean, 1e-15));
    assert!(close(v.moments().sum_squared_deviations, deviations, 1e-12));
    assert!(v.variance(Correction::Population) > 0.4);

    // The same in single precision, where the kernels have different lane
    // counts and a separate dispatch.
    let single: Vec<f32> = values.iter().map(|&value| value as f32).collect();
    let v32 = Vector::new(single.clone());
    let scalar_mean = single.iter().sum::<f32>() / single.len() as f32;
    assert!(close(v32.mean() as f64, scalar_mean as f64, 1e-6));

    // Column folds run the accumulating kernels rather than the whole-slice
    // ones, and have to give the same answers.
    let m = Matrix::from_flat(25, 40, values.clone());
    for col in 0..40 {
        let column: Vec<f64> = (0..25).map(|row| values[row * 40 + col]).collect();
        let column = Vector::new(column);
        assert!(close(
            m.mean_axis(Axis::Columns).data()[col],
            column.mean(),
            1e-14
        ));
        assert!(close(
            m.variance_axis(Axis::Columns, Correction::Sample).data()[col],
            column.variance(Correction::Sample),
            1e-10
        ));
    }
}

// ---- distributions over tensors ---------------------------------------------

#[test]
fn a_distribution_evaluates_elementwise() {
    let standard = Distribution::<f64>::standard_normal();
    let v = Vector::new([-1.0_f64, 0.0, 1.0]);

    let probabilities = v.cdf(&standard);
    assert!(close(
        probabilities.data()[0],
        0.158_655_253_931_457_1,
        1e-14
    ));
    assert_eq!(probabilities.data()[1], 0.5);
    assert!(close(
        probabilities.data()[2],
        0.841_344_746_068_542_9,
        1e-14
    ));

    // The quantile takes those probabilities back to the values they came from.
    let recovered = probabilities.ppf(&standard);
    for (actual, expected) in recovered.data().iter().zip(v.data()) {
        assert!(close(*actual, *expected, 1e-12));
    }

    let densities = v.pdf(&standard);
    assert!(close(
        densities.data()[1],
        1.0 / (2.0 * std::f64::consts::PI).sqrt(),
        1e-15
    ));
    assert!(close(densities.data()[0], densities.data()[2], 1e-15));

    // Choosing the function at runtime is the same operation.
    assert_eq!(
        v.distribution(Statistic::Cdf, &standard).to_vec(),
        probabilities.to_vec()
    );

    let shifted = Distribution::Normal {
        mean: 10.0,
        stddev: 2.0,
    };
    assert_eq!(shifted.cdf(10.0), 0.5);
    assert!(close(shifted.cdf(12.0), 0.841_344_746_068_542_9, 1e-14));
    assert!(close(
        shifted.ppf(0.975),
        10.0 + 2.0 * 1.959_963_984_540_054,
        1e-13
    ));

    // A degenerate scale is a NaN rather than a panic: it may be one row of a
    // fit over many.
    let degenerate = Distribution::Normal {
        mean: 0.0_f64,
        stddev: 0.0,
    };
    assert!(degenerate.cdf(1.0).is_nan());
}

#[test]
fn a_single_precision_tensor_keeps_full_single_precision_accuracy() {
    // The scalar path evaluates in f64 and rounds once, so a f32 tensor should
    // land within a few ulps of the f64 answer rather than within the accuracy
    // of a single-precision approximation.
    let standard32 = Distribution::<f32>::standard_normal();
    let standard64 = Distribution::<f64>::standard_normal();
    for i in -40..40 {
        let x = i as f32 / 10.0;
        let expected = standard64.cdf(x as f64);
        assert!(
            (standard32.cdf(x) as f64 - expected).abs() < 1e-7,
            "Φ({x}) = {} against {expected}",
            standard32.cdf(x)
        );
    }
}

#[test]
fn fitting_recovers_the_parameters_it_was_generated_from() {
    let v = Vector::new([2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
    let fitted = v.fit(Family::Normal, Correction::Population);
    assert_eq!(
        fitted,
        Distribution::Normal {
            mean: 5.0,
            stddev: 2.0
        }
    );
    assert_eq!(fitted.family(), Family::Normal);
    assert_eq!(fitted.parameters(), (5.0, 2.0));
    assert_eq!(
        Distribution::from_parameters(Family::Normal, (5.0, 2.0)),
        fitted
    );

    // The sample correction changes only the scale parameter.
    let sample = v.fit(Family::Normal, Correction::Sample);
    assert_eq!(sample.parameters().0, 5.0);
    assert!(sample.parameters().1 > 2.0);

    // The inverse Gaussian's shape estimate is exact for values drawn from its
    // own quantile function, up to the estimator's own bias.
    let source = Distribution::InverseGaussian {
        mean: 2.0,
        shape: 6.0,
    };
    let sample: Vec<f64> = (1..2_000).map(|i| source.ppf(i as f64 / 2_000.0)).collect();
    let recovered = Family::InverseGaussian.fit(&sample, Correction::Population);
    let (mean, shape) = recovered.parameters();
    assert!(close(mean, 2.0, 1e-2), "mean came back as {mean}");
    assert!(close(shape, 6.0, 5e-2), "shape came back as {shape}");

    // Values outside the support give NaN parameters rather than a panic.
    let invalid = Family::InverseGaussian.fit(&[1.0_f64, -1.0], Correction::Population);
    assert!(invalid.parameters().1.is_nan());
}

#[test]
fn axis_distributions_use_one_fit_per_row_or_column() {
    // Two rows on wildly different scales: the transform should put them on the
    // same one.
    let m = Matrix::<f64>::from_rows([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
    ]);

    let fits = m.fit_axis(Axis::Rows, Family::Normal, Correction::Population);
    assert_eq!(fits.len(), 2);
    assert_eq!(fits[0].parameters().0, 3.0);
    assert_eq!(fits[1].parameters().0, 3000.0);

    let probabilities = m.cdf_axis(Axis::Rows, &fits);
    // The middle of each row is its own median; the ends mirror each other.
    assert!(close(probabilities[(0, 2)], 0.5, 1e-14));
    assert!(close(probabilities[(1, 2)], 0.5, 1e-14));
    for col in 0..5 {
        assert!(close(
            probabilities[(0, col)],
            probabilities[(1, col)],
            1e-12
        ));
    }

    // And the quantile takes it back.
    let recovered = probabilities.ppf_axis(Axis::Rows, &fits);
    for row in 0..2 {
        for col in 0..5 {
            assert!(close(recovered[(row, col)], m[(row, col)], 1e-10));
        }
    }

    // Columns work the same way round.
    let column_fits = m.fit_axis(Axis::Columns, Family::Normal, Correction::Population);
    assert_eq!(column_fits.len(), 5);
    assert_eq!(column_fits[0].parameters().0, 500.5);
    let by_column = m.cdf_axis(Axis::Columns, &column_fits);
    for col in 0..5 {
        assert!(close(
            by_column[(0, col)],
            by_column[(1, col)].mul_add(-1.0, 1.0),
            1e-12
        ));
    }

    let densities = m.pdf_axis(Axis::Rows, &fits);
    assert!(densities[(0, 2)] > densities[(0, 0)]);

    // Runtime selection agrees with the named methods.
    assert_eq!(
        m.distribution_axis(Axis::Rows, Statistic::Cdf, &fits)
            .to_rows(),
        probabilities.to_rows()
    );
}

#[test]
#[should_panic(expected = "2 distributions for 5 columns")]
fn an_axis_distribution_needs_one_distribution_per_slice() {
    let m = Matrix::<f64>::from_rows([[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0]]);
    let fits = m.fit_axis(Axis::Rows, Family::Normal, Correction::Population);
    m.cdf_axis(Axis::Columns, &fits);
}

#[test]
#[should_panic(expected = "more than one family")]
fn an_axis_distribution_needs_a_single_family() {
    let m = Matrix::<f64>::from_rows([[1.0, 2.0], [3.0, 4.0]]);
    let mixed = [
        Distribution::Normal {
            mean: 1.0,
            stddev: 1.0,
        },
        Distribution::InverseGaussian {
            mean: 1.0,
            shape: 1.0,
        },
    ];
    m.cdf_axis(Axis::Rows, &mixed);
}

#[test]
fn an_axis_names_what_it_folds() {
    let shape = (3, 5);
    assert_eq!(Axis::Rows.extent(shape), 3);
    assert_eq!(Axis::Rows.depth(shape), 5);
    assert_eq!(Axis::Columns.extent(shape), 5);
    assert_eq!(Axis::Columns.depth(shape), 3);
}

// ---- the Metal tier ---------------------------------------------------------
//
// These require a Metal device. The tolerances are looser than the host tests
// because the shaders evaluate in `f32` throughout while the host evaluates the
// same mathematics in `f64` and rounds once.

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal {
    use super::close;
    use tensorcrate::statistics::{
        Axis, AxisStatistics, Correction, Distribution, Family, Statistic, Statistics,
    };
    use tensorcrate::tensors::{Host, Matrix, Metal, Vector};

    /// A mix of signs and magnitudes, deterministic and well away from zero so
    /// relative comparisons mean something.
    fn value(index: usize) -> f32 {
        ((index % 13) as f32 - 6.0) * 0.75 + 2.0
    }

    fn vector(len: usize) -> Vector<f32, Host> {
        Vector::new((0..len).map(value).collect::<Vec<_>>())
    }

    fn matrix(rows: usize, cols: usize) -> Matrix<f32, Host> {
        Matrix::from_flat(rows, cols, (0..rows * cols).map(value).collect::<Vec<_>>())
    }

    #[test]
    fn moments_agree_with_the_host() {
        for len in [1, 7, 64, 257, 1_024] {
            let host = vector(len);
            let resident = host.to_backend::<Metal>();

            assert!(
                close(Statistics::mean(&resident) as f64, host.mean() as f64, 1e-5),
                "length {len}: {} against {}",
                Statistics::mean(&resident),
                host.mean()
            );
            for correction in [Correction::Population, Correction::Sample] {
                let gpu = resident.stddev(correction);
                let cpu = host.stddev(correction);
                if cpu.is_nan() {
                    assert!(gpu.is_nan(), "length {len}: {gpu} should be NaN");
                } else {
                    assert!(
                        close(gpu as f64, cpu as f64, 1e-4),
                        "length {len}: {gpu} against {cpu}"
                    );
                }
            }
        }

        let host = matrix(37, 19);
        let resident = host.to_backend::<Metal>();
        assert!(close(
            Statistics::mean(&resident) as f64,
            host.mean() as f64,
            1e-5
        ));
        assert!(close(
            resident.variance(Correction::Sample) as f64,
            host.variance(Correction::Sample) as f64,
            1e-4
        ));
    }

    #[test]
    fn axis_moments_agree_with_the_host() {
        // Rows of every length mod four, short and long, for the row kernel's
        // four-wide reads and its tail.
        for (rows, cols) in [(23, 31), (40, 1024), (5, 4), (9, 130), (3, 2), (17, 4099)] {
            axis_moments_agree_on(rows, cols);
        }
    }

    fn axis_moments_agree_on(rows: usize, cols: usize) {
        let host = matrix(rows, cols);
        let resident = host.to_backend::<Metal>();

        for axis in [Axis::Rows, Axis::Columns] {
            let gpu = AxisStatistics::mean_axis(&resident, axis).to_backend::<Host>();
            let cpu = host.mean_axis(axis);
            assert_eq!(gpu.len(), cpu.len());
            for (index, (gpu, cpu)) in gpu.data().iter().zip(cpu.data()).enumerate() {
                assert!(
                    close(*gpu as f64, *cpu as f64, 1e-5),
                    "{axis:?} {index}: {gpu} against {cpu}"
                );
            }

            let gpu = resident
                .stddev_axis(axis, Correction::Sample)
                .to_backend::<Host>();
            let cpu = host.stddev_axis(axis, Correction::Sample);
            for (index, (gpu, cpu)) in gpu.data().iter().zip(cpu.data()).enumerate() {
                assert!(
                    close(*gpu as f64, *cpu as f64, 1e-4),
                    "{axis:?} {index}: {gpu} against {cpu}"
                );
            }

            assert_eq!(
                AxisStatistics::moments_axis(&resident, axis).count,
                host.moments_axis(axis).count
            );
        }
    }

    #[test]
    fn distribution_functions_agree_with_the_host() {
        let host = vector(512);
        let resident = host.to_backend::<Metal>();

        let normal = Distribution::Normal {
            mean: 2.0f32,
            stddev: 3.0,
        };
        for statistic in [Statistic::Pdf, Statistic::Cdf] {
            let gpu = Statistics::distribution(&resident, statistic, &normal).to_backend::<Host>();
            let cpu = host.distribution(statistic, &normal);
            for (index, (gpu, cpu)) in gpu.data().iter().zip(cpu.data()).enumerate() {
                assert!(
                    close(*gpu as f64, *cpu as f64, 1e-5),
                    "{statistic:?} {index}: {gpu} against {cpu}"
                );
            }
        }

        // The quantile takes probabilities, so it gets its own input.
        let probabilities = Vector::new((1..500).map(|i| i as f32 / 500.0).collect::<Vec<_>>());
        let resident_probabilities = probabilities.to_backend::<Metal>();
        let gpu = Statistics::ppf(&resident_probabilities, &normal).to_backend::<Host>();
        let cpu = probabilities.ppf(&normal);
        for (index, (gpu, cpu)) in gpu.data().iter().zip(cpu.data()).enumerate() {
            assert!(
                close(*gpu as f64, *cpu as f64, 1e-4),
                "quantile {index}: {gpu} against {cpu}"
            );
        }

        // The inverse Gaussian, whose quantile the shader solves for rather
        // than evaluating in closed form.
        let positive = Vector::new((1..200).map(|i| i as f32 / 40.0).collect::<Vec<_>>());
        let wald = Distribution::InverseGaussian {
            mean: 2.0f32,
            shape: 3.0,
        };
        let gpu = Statistics::cdf(&positive.to_backend::<Metal>(), &wald).to_backend::<Host>();
        let cpu = positive.cdf(&wald);
        for (index, (gpu, cpu)) in gpu.data().iter().zip(cpu.data()).enumerate() {
            assert!(
                close(*gpu as f64, *cpu as f64, 1e-4),
                "Wald distribution {index}: {gpu} against {cpu}"
            );
        }

        let gpu = Statistics::ppf(&resident_probabilities, &wald).to_backend::<Host>();
        let cpu = probabilities.ppf(&wald);
        for (index, (gpu, cpu)) in gpu.data().iter().zip(cpu.data()).enumerate() {
            assert!(
                close(*gpu as f64, *cpu as f64, 1e-3),
                "Wald quantile {index}: {gpu} against {cpu}"
            );
        }
    }

    #[test]
    fn axis_distributions_agree_with_the_host() {
        let host = matrix(16, 24);
        let resident = host.to_backend::<Metal>();

        for axis in [Axis::Rows, Axis::Columns] {
            let fits =
                AxisStatistics::fit_axis(&resident, axis, Family::Normal, Correction::Population);
            let host_fits = host.fit_axis(axis, Family::Normal, Correction::Population);
            assert_eq!(fits.len(), host_fits.len());

            let gpu = AxisStatistics::cdf_axis(&resident, axis, &fits).to_backend::<Host>();
            let cpu = host.cdf_axis(axis, &host_fits);
            for row in 0..host.rows() {
                for col in 0..host.cols() {
                    assert!(
                        close(gpu[(row, col)] as f64, cpu[(row, col)] as f64, 1e-5),
                        "{axis:?} ({row}, {col}): {} against {}",
                        gpu[(row, col)],
                        cpu[(row, col)]
                    );
                }
            }
        }
    }

    #[test]
    fn results_stay_in_shared_memory() {
        let resident = matrix(32, 32).to_backend::<Metal>();
        if !resident.is_device_resident() {
            // No Metal device on this machine; Metal operations are unavailable.
            return;
        }

        let moments = AxisStatistics::moments_axis(&resident, Axis::Rows);
        assert!(moments.means.is_device_resident());
        assert!(moments.sum_squared_deviations.is_device_resident());
        assert!(
            resident
                .stddev_axis(Axis::Rows, Correction::Sample)
                .is_device_resident()
        );

        let fits = AxisStatistics::fit_axis(
            &resident,
            Axis::Rows,
            Family::Normal,
            Correction::Population,
        );
        assert!(AxisStatistics::cdf_axis(&resident, Axis::Rows, &fits).is_device_resident());
        assert!(Statistics::cdf(&resident, &Distribution::standard_normal()).is_device_resident());
    }
}
