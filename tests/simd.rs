//! Validates the architecture-specific SIMD kernels against straightforward
//! scalar references.
//!
//! Sizes are deliberately chosen *not* to be multiples of the lane width, so the
//! ragged scalar tails in every kernel are exercised alongside the vector body.

#![cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]

use tensorcrate::simd::{f32k, f64k, fft_f32};
use tensorcrate::tensors::{BinaryOp, Compare, Matrix, Reduce, Vector};

fn approx(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
}

#[test]
fn dot_matches_scalar_across_tail_lengths() {
    for n in [1usize, 3, 4, 15, 16, 17, 37, 128, 129] {
        let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 3.0).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
        let want: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!(approx(f32k::dot(&a, &b), want, 1e-4), "n={n}");

        let a64: Vec<f64> = a.iter().map(|&x| x as f64).collect();
        let b64: Vec<f64> = b.iter().map(|&x| x as f64).collect();
        let want64: f64 = a64.iter().zip(&b64).map(|(x, y)| x * y).sum();
        assert!((f64k::dot(&a64, &b64) - want64).abs() <= 1e-9 * (1.0 + want64.abs()));
    }
}

#[test]
fn elementwise_matches_scalar_for_every_op() {
    let n = 53;
    let a: Vec<f32> = (0..n).map(|i| (i as f32) + 1.0).collect();
    let b: Vec<f32> = (0..n).map(|i| ((i % 7) as f32) + 1.0).collect();
    for op in [
        BinaryOp::Add,
        BinaryOp::Sub,
        BinaryOp::Mul,
        BinaryOp::Div,
        BinaryOp::Rem,
    ] {
        let mut got = vec![0.0f32; n];
        f32k::elementwise(&a, &b, op, &mut got);
        for i in 0..n {
            let want = match op {
                BinaryOp::Add => a[i] + b[i],
                BinaryOp::Sub => a[i] - b[i],
                BinaryOp::Mul => a[i] * b[i],
                BinaryOp::Div => a[i] / b[i],
                BinaryOp::Rem => a[i] % b[i],
            };
            assert!(approx(got[i], want, 1e-5), "op={op:?} i={i}");
        }
    }
}

#[test]
fn broadcast_respects_operand_order() {
    let n = 40;
    let x: Vec<f32> = (0..n).map(|i| (i as f32) + 2.0).collect();
    let s = 3.5f32;
    for &left in &[false, true] {
        for op in [
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Mul,
            BinaryOp::Div,
            BinaryOp::Rem,
        ] {
            let mut got = vec![0.0f32; n];
            f32k::broadcast(&x, s, op, left, &mut got);
            for i in 0..n {
                let (l, r) = if left { (s, x[i]) } else { (x[i], s) };
                let want = match op {
                    BinaryOp::Add => l + r,
                    BinaryOp::Sub => l - r,
                    BinaryOp::Mul => l * r,
                    BinaryOp::Div => l / r,
                    BinaryOp::Rem => l % r,
                };
                assert!(approx(got[i], want, 1e-5), "left={left} op={op:?} i={i}");
            }
        }
    }
}

#[test]
fn matmul_matches_triple_loop() {
    // C = 7 is not a multiple of the 4-lane f32 width, so the output-row tail runs.
    const M: usize = 5;
    const K: usize = 6;
    const N: usize = 7;
    let a: Vec<f32> = (0..M * K).map(|i| (i as f32) * 0.1).collect();
    let b: Vec<f32> = (0..K * N).map(|i| (i as f32) * 0.2 - 1.0).collect();
    let addend: Vec<f32> = (0..M * N).map(|i| (i as f32) * -0.3 + 2.0).collect();
    let mut got = vec![0.0f32; M * N];
    f32k::matmul(&a, &b, M, K, N, &mut got);
    for i in 0..M {
        for j in 0..N {
            let mut want = 0.0f32;
            for p in 0..K {
                want += a[i * K + p] * b[p * N + j];
            }
            assert!(approx(got[i * N + j], want, 1e-4), "i={i} j={j}");
        }
    }

    f32k::matmul_add(&a, &b, &addend, M, K, N, &mut got);
    for i in 0..M {
        for j in 0..N {
            let mut want = addend[i * N + j];
            for p in 0..K {
                want += a[i * K + p] * b[p * N + j];
            }
            assert!(approx(got[i * N + j], want, 1e-4), "fused i={i} j={j}");
        }
    }
}

#[test]
fn public_api_dispatch_is_correct_above_thresholds() {
    // N and the matmul op-count both clear the SIMD gates, so these calls route
    // through the NEON kernels rather than the scalar fallback.
    let a: Vector<f32> = Vector::new((0..64).map(|i| (i as f32) * 0.25 - 4.0).collect::<Vec<_>>());
    let b: Vector<f32> = Vector::new((0..64).map(|i| (i as f32).cos()).collect::<Vec<_>>());
    let want: f32 = (0..64).map(|i| a[i] * b[i]).sum();
    assert!(approx(a.dot(&b), want, 1e-3));

    let m1: Matrix<f32> = Matrix::from_rows((0..8).map(|i| {
        (0..8)
            .map(|j| ((i * 8 + j) as f32) * 0.1)
            .collect::<Vec<_>>()
    }));
    let m2: Matrix<f32> = Matrix::from_rows(
        (0..8).map(|i| (0..8).map(|j| ((i + j) as f32) - 3.0).collect::<Vec<_>>()),
    );
    let product = m1.matmul(&m2);
    for i in 0..8 {
        for j in 0..8 {
            let mut want = 0.0f32;
            for p in 0..8 {
                want += m1[(i, p)] * m2[(p, j)];
            }
            assert!(approx(product[(i, j)], want, 1e-3), "i={i} j={j}");
        }
    }
}

/// Direct-definition DFT reference: `X[k] = Σ x[n] exp(-2πi k n / N)`.
fn dft(input: &[(f64, f64)], inverse: bool) -> Vec<(f64, f64)> {
    let n = input.len();
    let sign = if inverse { 1.0 } else { -1.0 };
    (0..n)
        .map(|k| {
            let mut re = 0.0;
            let mut im = 0.0;
            for (nn, &(xr, xi)) in input.iter().enumerate() {
                let ang = sign * std::f64::consts::TAU * (k * nn) as f64 / n as f64;
                let (s, c) = ang.sin_cos();
                re += xr * c - xi * s;
                im += xr * s + xi * c;
            }
            (re, im)
        })
        .collect()
}

#[test]
fn radix2_matches_the_dft_and_round_trips() {
    for &n in &[8usize, 16, 64, 1024] {
        let complex: Vec<(f64, f64)> = (0..n)
            .map(|i| ((i as f64).cos(), (i as f64 * 0.3).sin()))
            .collect();

        // Forward transform (direction -1.0), interleaved f32 buffer.
        let mut buf: Vec<f32> = complex
            .iter()
            .flat_map(|&(r, i)| [r as f32, i as f32])
            .collect();
        fft_f32::radix2(&mut buf, n, -1.0);

        let want = dft(&complex, false);
        for k in 0..n {
            assert!(approx(buf[2 * k], want[k].0 as f32, 1e-2), "n={n} k={k} re");
            assert!(
                approx(buf[2 * k + 1], want[k].1 as f32, 1e-2),
                "n={n} k={k} im"
            );
        }

        // Inverse (direction +1.0) then 1/N normalization recovers the input.
        fft_f32::radix2(&mut buf, n, 1.0);
        for k in 0..n {
            assert!(
                approx(buf[2 * k] / n as f32, complex[k].0 as f32, 1e-2),
                "rt n={n} k={k}"
            );
            assert!(
                approx(buf[2 * k + 1] / n as f32, complex[k].1 as f32, 1e-2),
                "rt n={n} k={k}"
            );
        }
    }
}

// ---- comparisons, clamp and reductions --------------------------------------

/// The scalar definition of a comparison, written for both float widths so the
/// `f64` kernel is held to the same standard as the `f32` one.
macro_rules! scalar_compare {
    ($name:ident, $t:ty) => {
        fn $name(op: Compare, a: $t, b: $t) -> $t {
            match op {
                Compare::Min => a.min(b),
                Compare::Max => a.max(b),
                Compare::MaxShare => match a.partial_cmp(&b) {
                    Some(std::cmp::Ordering::Greater) => 1.0,
                    Some(std::cmp::Ordering::Less) => 0.0,
                    _ => 0.5,
                },
                Compare::Less => (a < b) as u8 as $t,
                Compare::LessEqual => (a <= b) as u8 as $t,
                Compare::Greater => (a > b) as u8 as $t,
                Compare::GreaterEqual => (a >= b) as u8 as $t,
            }
        }
    };
}

scalar_compare!(compare32, f32);
scalar_compare!(compare64, f64);

/// Bit equality, with the two licensed exceptions: NaN payloads are not
/// specified, and neither is the sign of a zero returned by `min`/`max` when
/// `-0.0` and `+0.0` tie — `fminnm` keeps the negative sign, `minps` keeps
/// whichever operand came second.
fn same_answer(got: f32, want: f32) -> bool {
    got.to_bits() == want.to_bits()
        || (got.is_nan() && want.is_nan())
        || (got == 0.0 && want == 0.0)
}

fn same_answer64(got: f64, want: f64) -> bool {
    got.to_bits() == want.to_bits()
        || (got.is_nan() && want.is_nan())
        || (got == 0.0 && want == 0.0)
}

/// Values with ties, both zeros and a NaN, since those are where a vector
/// comparison is free to disagree with the scalar one.
fn awkward_f32(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| match i % 11 {
            0 => 0.0,
            1 => -0.0,
            2 => f32::NAN,
            3 => f32::INFINITY,
            4 => f32::NEG_INFINITY,
            other => (other as f32) * 0.75 - 3.0,
        })
        .collect()
}

#[test]
fn compare_matches_scalar_including_ties_and_nans() {
    for n in [1usize, 3, 4, 15, 16, 17, 37, 129] {
        let a = awkward_f32(n);
        let b: Vec<f32> = awkward_f32(n).into_iter().rev().collect();
        for op in Compare::ALL {
            let mut got = vec![0.0f32; n];
            f32k::compare(&a, &b, op, &mut got);
            for i in 0..n {
                let want = compare32(op, a[i], b[i]);
                assert!(
                    same_answer(got[i], want),
                    "f32 {op:?} n={n} i={i}: {} vs {want}",
                    got[i]
                );
            }

            let a64: Vec<f64> = a.iter().map(|&x| x as f64).collect();
            let b64: Vec<f64> = b.iter().map(|&x| x as f64).collect();
            let mut got64 = vec![0.0f64; n];
            f64k::compare(&a64, &b64, op, &mut got64);
            for i in 0..n {
                let want = compare64(op, a64[i], b64[i]);
                assert!(same_answer64(got64[i], want), "f64 {op:?} n={n} i={i}");
            }
        }
    }
}

#[test]
fn compare_scalar_respects_operand_order() {
    let n = 37;
    let values = awkward_f32(n);
    for scalar in [0.0f32, -1.5, 2.0] {
        for left in [false, true] {
            for op in Compare::ALL {
                let mut got = vec![0.0f32; n];
                f32k::compare_scalar(&values, scalar, op, left, &mut got);
                for i in 0..n {
                    let (lhs, rhs) = if left {
                        (scalar, values[i])
                    } else {
                        (values[i], scalar)
                    };
                    let want = compare32(op, lhs, rhs);
                    assert!(
                        same_answer(got[i], want),
                        "{op:?} left={left} i={i}: {} vs {want}",
                        got[i]
                    );
                }
            }
        }
    }
}

// `max` then `min` rather than `f32::clamp`, which is a different function: it
// panics on a NaN bound and propagates a NaN input, where the kernels are
// specified as the pair of IEEE `minNum`/`maxNum` operations.
#[allow(clippy::manual_clamp)]
#[test]
fn clamp_matches_the_scalar_pair_of_bounds() {
    for n in [1usize, 5, 16, 33, 64] {
        let values = awkward_f32(n);
        let mut got = vec![0.0f32; n];
        f32k::clamp(&values, -1.0, 2.0, &mut got);
        for i in 0..n {
            let want = values[i].max(-1.0).min(2.0);
            assert!(
                same_answer(got[i], want),
                "n={n} i={i}: {} vs {want}",
                got[i]
            );
        }

        let values64: Vec<f64> = values.iter().map(|&x| x as f64).collect();
        let mut got64 = vec![0.0f64; n];
        f64k::clamp(&values64, -1.0, 2.0, &mut got64);
        for i in 0..n {
            assert!(same_answer64(got64[i], values64[i].max(-1.0).min(2.0)));
        }
    }
}

#[test]
fn reduce_matches_the_scalar_fold() {
    for n in [0usize, 1, 7, 16, 17, 64, 130] {
        // Integral values, so the vector fold's different association is exact
        // and `Sum` can be compared without a tolerance.
        let values: Vec<f32> = (0..n).map(|i| ((i % 13) as f32) - 6.0).collect();
        for op in Reduce::ALL {
            assert_eq!(f32k::reduce(&values, op), op.fold(&values), "{op:?} n={n}");
        }

        let values64: Vec<f64> = values.iter().map(|&x| x as f64).collect();
        let sum64: f64 = values64.iter().sum();
        assert!((f64k::reduce(&values64, Reduce::Sum) - sum64).abs() <= 1e-12);
        assert_eq!(
            f64k::reduce(&values64, Reduce::Min),
            values64.iter().copied().fold(f64::INFINITY, f64::min)
        );
        assert_eq!(
            f64k::reduce(&values64, Reduce::Max),
            values64.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        );
    }
}

#[test]
fn reduce_lets_numbers_beat_nans() {
    // `f32::min`/`f32::max` return the non-NaN operand, so a NaN anywhere in the
    // input must not poison the fold — including when it lands in a different
    // accumulator from the extreme value.
    let mut values: Vec<f32> = (0..64).map(|i| (i as f32) - 32.0).collect();
    values[5] = f32::NAN;
    values[40] = f32::NAN;
    assert_eq!(f32k::reduce(&values, Reduce::Min), -32.0);
    assert_eq!(f32k::reduce(&values, Reduce::Max), 31.0);
    assert!(f32k::reduce(&values, Reduce::Sum).is_nan());
}

// The kernels are safe functions over raw-pointer loops, so a length mismatch
// must panic in every build profile rather than read past a slice.

#[test]
#[should_panic(expected = "slice lengths differ")]
fn dot_rejects_mismatched_lengths() {
    f32k::dot(&[1.0; 64], &[1.0; 8]);
}

#[test]
#[should_panic(expected = "slice lengths differ")]
fn elementwise_rejects_a_short_output() {
    let mut out = [0.0f64; 8];
    f64k::elementwise(&[1.0; 64], &[1.0; 64], BinaryOp::Add, &mut out);
}

#[test]
#[should_panic(expected = "slice lengths differ")]
fn accumulate_rejects_a_short_input() {
    let mut totals = [0.0f32; 64];
    f32k::accumulate(&mut totals, &[1.0; 8]);
}

#[test]
#[should_panic(expected = "do not hold")]
fn matmul_rejects_extents_larger_than_the_slices() {
    let mut out = [0.0f32; 4];
    f32k::matmul_accumulate(&[1.0; 4], &[1.0; 4], 2, 16, 2, &mut out);
}

#[test]
#[should_panic(expected = "do not hold")]
fn matmul_rejects_extents_whose_product_overflows() {
    let mut out = [0.0f32; 0];
    f32k::matmul(&[], &[], usize::MAX, 2, 0, &mut out);
}

#[test]
#[should_panic(expected = "not a power of two")]
fn fft_rejects_a_length_that_is_not_a_power_of_two() {
    let mut buf = [0.0f32; 24];
    fft_f32::radix2(&mut buf, 12, -1.0);
}
