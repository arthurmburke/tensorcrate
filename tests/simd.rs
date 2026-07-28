//! Validates the architecture-specific SIMD kernels against straightforward
//! scalar references.
//!
//! Sizes are deliberately chosen *not* to be multiples of the lane width, so the
//! ragged scalar tails in every kernel are exercised alongside the vector body.

#![cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]

use rinterp::simd::{f32k, f64k, fft_f32};
use rinterp::tensors::{BinaryOp, Matrix, Vector};

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
    let a: Vector<f32, 64> = Vector::new(std::array::from_fn(|i| (i as f32) * 0.25 - 4.0));
    let b: Vector<f32, 64> = Vector::new(std::array::from_fn(|i| (i as f32).cos()));
    let want: f32 = (0..64).map(|i| a.data()[i] * b.data()[i]).sum();
    assert!(approx(a.dot(&b), want, 1e-3));

    let m1: Matrix<f32, 8, 8> = Matrix::from_rows(std::array::from_fn(|i| {
        std::array::from_fn(|j| ((i * 8 + j) as f32) * 0.1)
    }));
    let m2: Matrix<f32, 8, 8> = Matrix::from_rows(std::array::from_fn(|i| {
        std::array::from_fn(|j| ((i + j) as f32) - 3.0)
    }));
    let product = m1.matmul(&m2);
    for i in 0..8 {
        for j in 0..8 {
            let mut want = 0.0f32;
            for p in 0..8 {
                want += m1.data()[i][p] * m2.data()[p][j];
            }
            assert!(approx(product.data()[i][j], want, 1e-3), "i={i} j={j}");
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
