//! Native `f16` and `bf16` arithmetic.
//!
//! Two promises are checked here, on both backends:
//!
//! - every elementwise operation computes in the element type and rounds to
//!   it — so the result is the correctly rounded 16-bit answer, not an `f32`
//!   value that happens to be stored compactly;
//! - every accumulation (sums, dot and matrix products, moments, prefix sums)
//!   runs in `f32` and rounds once.
//!
//! The host is the oracle for the GPU. The CPU's vector kernels (`FADD v.8h`,
//! `BFCVTN`, `FMLAL`, `BFMLALB`, …) are checked bit for bit against the scalar
//! `half` operators, which are correctly rounded by construction.

use half::{bf16, f16};
use tensorcrate::numbers::Real;
use tensorcrate::tensors::{Analytic, BinaryOp, Compare, Host, Kernels, Matrix, Reduce, Vector};

/// A small deterministic generator, so a failure reproduces.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    fn uniform(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * (self.next() as f64 / (1u64 << 31) as f64)
    }

    fn vector<T: Real>(&mut self, len: usize, low: f64, high: f64) -> Vec<T> {
        (0..len)
            .map(|_| T::from_f64(self.uniform(low, high)))
            .collect()
    }
}

/// Lengths around the eight-lane vectors and the SIMD dispatch threshold.
const LENGTHS: [usize; 7] = [1, 7, 8, 9, 16, 17, 1001];

fn bits<T: Real>(values: &[T]) -> Vec<u64> {
    values.iter().map(|x| x.into_f64().to_bits()).collect()
}

// ---- accumulation runs in f32 --------------------------------------------------

#[test]
fn a_long_sum_keeps_f32_precision() {
    // A 16-bit running total stops growing at 2048 ones (f16) or 256 (bf16).
    let ones = Vector::new(vec![f16::ONE; 10_000]);
    assert_eq!(ones.sum(), f16::from_f64(10_000.0));
    assert_eq!(ones.reduce(Reduce::Sum), f16::from_f64(10_000.0));
    assert_eq!(Reduce::Sum.fold(ones.data()), f16::from_f64(10_000.0));
    assert_eq!(ones.dot(&ones), f16::from_f64(10_000.0));

    let ones = Vector::new(vec![bf16::ONE; 10_000]);
    assert_eq!(ones.sum(), bf16::from_f64(10_000.0));
    assert_eq!(ones.dot(&ones), bf16::from_f64(10_000.0));

    // A prefix sum rounds each running total once, from f32.
    let scan = Vector::new(vec![f16::ONE; 3000]).prefix_sum();
    assert_eq!(scan[2999], f16::from_f64(3000.0));
}

#[test]
fn products_and_moments_accumulate_in_f32() {
    let mut rng = Lcg(1);
    for &n in &[5usize, 64, 300] {
        let a = Matrix::from_flat(3, n, rng.vector::<f16>(3 * n, -1.0, 1.0));
        let b = Matrix::from_flat(n, 2, rng.vector::<f16>(n * 2, -1.0, 1.0));
        let product = a.matmul(&b);
        for i in 0..3 {
            for j in 0..2 {
                let exact: f64 = (0..n)
                    .map(|k| a[(i, k)].to_f64() * b[(k, j)].to_f64())
                    .sum();
                // One rounding to f16, plus the f32 accumulator's own error.
                let got = product[(i, j)].to_f64();
                assert!(
                    (got - exact).abs()
                        <= f64::from(f16::EPSILON) * exact.abs().max(1.0) / 2.0 + 1e-5,
                    "n = {n}: {got} vs {exact}"
                );
            }
        }
    }

    // The mean of values centred on a large offset: the deviations are taken
    // from the unrounded f32 mean.
    let values = Vector::new(
        (0..1000)
            .map(|i| bf16::from_f64(100.0 + f64::from(i % 3)))
            .collect::<Vec<_>>(),
    );
    let moments = values.moments();
    assert_eq!(moments.mean, bf16::from_f64(100.0 + 999.0 / 1000.0));
}

// ---- the CPU vector kernels are exactly the scalar arithmetic -------------------

fn scalar_binary<T: Real>(op: BinaryOp, a: T, b: T) -> T {
    match op {
        BinaryOp::Add => a + b,
        BinaryOp::Sub => a - b,
        BinaryOp::Mul => a * b,
        BinaryOp::Div => a / b,
        BinaryOp::Rem => a % b,
    }
}

fn host_elementwise_is_exact<T: Real>() {
    let mut rng = Lcg(2);
    for &len in &LENGTHS {
        let a = Vector::new(rng.vector::<T>(len, -8.0, 8.0));
        let b = Vector::new(rng.vector::<T>(len, 0.25, 8.0));
        for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div] {
            let got = <Host as Kernels<T>>::vector_elementwise(&a, &b, op);
            let want = a.zip_map(&b, |&x, &y| scalar_binary(op, x, y));
            assert_eq!(bits(got.data()), bits(want.data()), "{op:?}, len {len}");

            let s = b[0];
            let got = <Host as Kernels<T>>::vector_broadcast(&a, s, op, true);
            let want = a.map(|&x| scalar_binary(op, s, x));
            assert_eq!(
                bits(got.data()),
                bits(want.data()),
                "{op:?} broadcast, len {len}"
            );
        }
        for op in Compare::ALL {
            let got = <Host as Kernels<T>>::vector_compare(&a, &b, op);
            let want = a.zip_map(&b, |&x, &y| op.value(x, y));
            assert_eq!(bits(got.data()), bits(want.data()), "{op:?}, len {len}");
            let got = <Host as Kernels<T>>::vector_compare_scalar(&a, b[0], op, false);
            let want = a.map(|&x| op.value(x, b[0]));
            assert_eq!(
                bits(got.data()),
                bits(want.data()),
                "{op:?} scalar, len {len}"
            );
        }
        let (low, high) = (T::from_f64(-2.0), T::from_f64(3.0));
        let got = <Host as Kernels<T>>::vector_clamp(&a, low, high);
        let want = a.map(|&x| x.max(low).min(high));
        assert_eq!(bits(got.data()), bits(want.data()), "clamp, len {len}");

        let got = <Host as Kernels<T>>::vector_unary(&b, Analytic::Sqrt);
        let want = b.map(|&x| x.sqrt());
        assert_eq!(bits(got.data()), bits(want.data()), "sqrt, len {len}");
    }
}

#[test]
fn f16_vector_kernels_round_like_scalar_f16() {
    host_elementwise_is_exact::<f16>();
}

#[test]
fn bf16_vector_kernels_round_like_scalar_bf16() {
    host_elementwise_is_exact::<bf16>();
}

#[test]
fn half_precision_results_are_rounded_every_operation() {
    // 2049 is not an f16: 2048 + 1 must come back as 2048, where f32 arithmetic
    // stored afterwards would have given the same, but (2048 + 1) - 2048 must be
    // 0, where an f32 intermediate would give 1.
    let big = Vector::new(vec![f16::from_f64(2048.0); 32]);
    let one = Vector::new(vec![f16::ONE; 32]);
    let back = &(&big + &one) - &big;
    assert!(back.data().iter().all(|&x| x == f16::ZERO));

    // The same for bf16, which has eight significant bits: 256 + 1 = 256.
    let big = Vector::new(vec![bf16::from_f64(256.0); 32]);
    let one = Vector::new(vec![bf16::ONE; 32]);
    let back = &(&big + &one) - &big;
    assert!(back.data().iter().all(|&x| x == bf16::ZERO));
}

// ---- Metal: the f16 and bf16 kernels against the host ---------------------------

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal {
    use super::*;
    use tensorcrate::metal::MetalElement;
    use tensorcrate::statistics::{AxisStatistics, Correction, Distribution, Statistics};
    use tensorcrate::tensors::fused::{self, Builder, DType, Mode};
    use tensorcrate::tensors::{Axis, Metal, Ordered, SortOrder, Tape, Transcendental};

    fn close<T: Real>(got: &[T], want: &[T], ulps: f64, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: lengths");
        let eps = T::epsilon().into_f64();
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            let (g, w) = (g.into_f64(), w.into_f64());
            if w.is_nan() {
                assert!(g.is_nan(), "{what}[{i}]: {g} is not NaN");
                continue;
            }
            if g == w {
                continue;
            }
            assert!(
                (g - w).abs() <= ulps * eps * w.abs().max(1.0),
                "{what}[{i}]: {g} vs {w}"
            );
        }
    }

    fn resident<T: MetalElement>(values: &[T]) -> Vector<T, Metal> {
        let v = Vector::<T, Host>::new(values.to_vec()).to_backend::<Metal>();
        assert!(v.is_device_resident(), "no Metal device");
        v
    }

    fn the_gpu_matches_the_host<T: MetalElement>() {
        let mut rng = Lcg(3);
        let a_host = Vector::new(rng.vector::<T>(1001, -4.0, 4.0));
        let b_host = Vector::new(rng.vector::<T>(1001, 0.25, 4.0));
        let (a, b) = (resident(a_host.data()), resident(b_host.data()));

        // Elementwise arithmetic rounds to T every time, on both sides. Add,
        // subtract and multiply are exact in either ALU; division is within the
        // GPU's fast-math tolerance.
        for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul] {
            let gpu = <Metal as Kernels<T>>::vector_elementwise(&a, &b, op);
            let host = <Host as Kernels<T>>::vector_elementwise(&a_host, &b_host, op);
            assert!(gpu.is_device_resident());
            assert_eq!(bits(gpu.as_slice()), bits(host.data()), "{op:?}");
        }
        let gpu = <Metal as Kernels<T>>::vector_elementwise(&a, &b, BinaryOp::Div);
        let host = <Host as Kernels<T>>::vector_elementwise(&a_host, &b_host, BinaryOp::Div);
        close(gpu.as_slice(), host.data(), 1.0, "div");

        let gpu =
            <Metal as Kernels<T>>::vector_broadcast(&a, T::from_f64(1.5), BinaryOp::Mul, false);
        let host =
            <Host as Kernels<T>>::vector_broadcast(&a_host, T::from_f64(1.5), BinaryOp::Mul, false);
        assert_eq!(bits(gpu.as_slice()), bits(host.data()), "broadcast");

        // Comparisons, clamps, sorts and data movement are exact.
        for op in Compare::ALL {
            let gpu = <Metal as Kernels<T>>::vector_compare(&a, &b, op);
            let host = <Host as Kernels<T>>::vector_compare(&a_host, &b_host, op);
            assert_eq!(bits(gpu.as_slice()), bits(host.data()), "{op:?}");
        }
        let gpu = a.clamp(T::from_f64(-1.0), T::from_f64(2.0));
        let host = Ordered::clamp(&a_host, T::from_f64(-1.0), T::from_f64(2.0));
        assert_eq!(bits(gpu.as_slice()), bits(host.data()), "clamp");
        for order in SortOrder::ALL {
            let gpu = a.sorted(order);
            let host = a_host.sorted(order);
            assert_eq!(bits(gpu.as_slice()), bits(host.data()), "{order:?}");
        }

        // Folds run in f32 on both sides and differ only in summation order.
        close(&[a.sum()], &[a_host.sum()], 1.0, "sum");
        assert_eq!(a.reduce(Reduce::Min), a_host.reduce(Reduce::Min));
        assert_eq!(a.reduce(Reduce::Max), a_host.reduce(Reduce::Max));
        close(&[a.dot(&b)], &[a_host.dot(&b_host)], 1.0, "dot");
        close(
            a.prefix_sum().as_slice(),
            a_host.prefix_sum().data(),
            1.0,
            "prefix sum",
        );

        // The analytic functions: the GPU's half-precision library against the
        // host's f32-then-round.
        for f in Analytic::ALL {
            let input = match f {
                Analytic::Arcsin | Analytic::Arccos => &a_host.map(|&x| x / T::from_f64(4.5)),
                Analytic::Ln | Analytic::Sqrt => &b_host,
                _ => &a_host,
            };
            let gpu = resident(input.data()).analytic(f);
            let host = <Host as Kernels<T>>::vector_unary(input, f);
            // Sec, Csc and Tan have poles in range; compare where the host's
            // answer is moderate.
            let (g, h): (Vec<T>, Vec<T>) = gpu
                .as_slice()
                .iter()
                .zip(host.data())
                .filter(|(_, h)| h.into_f64().abs() < 50.0)
                .map(|(&g, &h)| (g, h))
                .unzip();
            close(&g, &h, 4.0, &format!("{f:?}"));
        }

        let gpu = a.pow(T::from_f64(2.0));
        let host = Transcendental::pow(&a_host, T::from_f64(2.0));
        close(gpu.as_slice(), host.data(), 1.0, "pow");
    }

    #[test]
    fn f16_kernels_on_the_gpu_match_the_host() {
        the_gpu_matches_the_host::<f16>();
    }

    #[test]
    fn bf16_kernels_on_the_gpu_match_the_host() {
        the_gpu_matches_the_host::<bf16>();
    }

    fn matrices_match<T: MetalElement>() {
        let mut rng = Lcg(4);
        for (m, k, n) in [(3, 5, 4), (17, 33, 9), (70, 130, 66)] {
            let a = Matrix::from_flat(m, k, rng.vector::<T>(m * k, -1.0, 1.0));
            let b = Matrix::from_flat(k, n, rng.vector::<T>(k * n, -1.0, 1.0));
            let c = Matrix::from_flat(m, n, rng.vector::<T>(m * n, -1.0, 1.0));
            let (ga, gb) = (a.to_backend::<Metal>(), b.to_backend::<Metal>());

            // Both sides accumulate in f32 and round once.
            let product = ga.matmul(&gb);
            close(product.as_slice(), a.matmul(&b).data(), 1.0, "matmul");
            let fused = ga.matmul_add(&gb, c.to_backend::<Metal>());
            close(
                fused.as_slice(),
                a.matmul_add(&b, c.clone()).data(),
                1.0,
                "matmul_add",
            );

            let v = Vector::new(rng.vector::<T>(k, -1.0, 1.0));
            close(
                ga.matvec(&v.to_backend::<Metal>()).as_slice(),
                a.matvec(&v).data(),
                1.0,
                "matvec",
            );
            assert_eq!(ga.transpose().as_slice(), a.transpose().data());
        }

        let input = Matrix::from_flat(9, 11, rng.vector::<T>(99, -1.0, 1.0));
        let window = Matrix::from_flat(3, 4, rng.vector::<T>(12, -1.0, 1.0));
        let gpu = <Metal as Kernels<T>>::correlate(
            &input.to_backend::<Metal>(),
            &window.to_backend::<Metal>(),
            true,
        );
        let host = <Host as Kernels<T>>::correlate(&input, &window, true);
        close(gpu.as_slice(), host.data(), 1.0, "correlate");

        // Moments and distributions.
        let gpu = input.to_backend::<Metal>();
        close(&[Statistics::mean(&gpu)], &[input.mean()], 1.0, "mean");
        close(
            &[Statistics::variance(&gpu, Correction::Sample)],
            &[input.variance(Correction::Sample)],
            2.0,
            "variance",
        );
        close(
            AxisStatistics::mean_axis(&gpu, Axis::Columns).as_slice(),
            input.mean_axis(Axis::Columns).data(),
            1.0,
            "mean_axis",
        );
        let normal = Distribution::Normal {
            mean: T::from_f64(0.0),
            stddev: T::from_f64(0.5),
        };
        close(
            Statistics::cdf(&gpu, &normal).as_slice(),
            input.cdf(&normal).data(),
            2.0,
            "cdf",
        );
    }

    #[test]
    fn f16_matrix_kernels_on_the_gpu_match_the_host() {
        matrices_match::<f16>();
    }

    #[test]
    fn bf16_matrix_kernels_on_the_gpu_match_the_host() {
        matrices_match::<bf16>();
    }

    #[test]
    fn a_long_gpu_sum_keeps_f32_precision() {
        let ones = resident(&vec![f16::ONE; 10_000]);
        assert_eq!(ones.sum(), f16::from_f64(10_000.0));
        let ones = resident(&vec![bf16::ONE; 10_000]);
        assert_eq!(ones.sum(), bf16::from_f64(10_000.0));
        let scan = resident(&vec![f16::ONE; 3000]).prefix_sum();
        assert_eq!(scan.as_slice()[2999], f16::from_f64(3000.0));
    }

    #[test]
    fn a_gpu_f16_sum_rounds_every_elementwise_operation() {
        let big = resident(&[f16::from_f64(2048.0); 64]);
        let one = resident(&[f16::ONE; 64]);
        let back = &(&big + &one) - &big;
        assert!(back.as_slice().iter().all(|&x| x == f16::ZERO));
    }

    #[test]
    fn half_precision_programs_run_resident() {
        // y = (a·b + 1)², computed in f16 registers on the GPU.
        let mut b = Builder::<f16>::new();
        let (x, w) = (b.input(DType::F16), b.input(DType::F16));
        let product = b.mul(x, w);
        let shifted = b.shift(product, f16::ONE);
        let squared = b.mul(shifted, shifted);
        b.output(squared, DType::F16);
        let program = b.build().unwrap();

        let mut rng = Lcg(5);
        let (a, w) = (
            rng.vector::<f16>(777, -2.0, 2.0),
            rng.vector::<f16>(777, -2.0, 2.0),
        );
        let (ha, hw) = (Vector::new(a.clone()), Vector::new(w.clone()));
        let host = program.run_vectors(&[&ha, &hw]).remove(0);
        let gpu = program
            .run_vectors(&[&resident(&a), &resident(&w)])
            .remove(0);
        assert!(gpu.is_device_resident());
        // Every operation rounds to f16 on both sides.
        assert_eq!(bits(gpu.as_slice()), bits(host.data()));

        // And the unfused program on the GPU agrees with the fused one.
        let unfused = fused::with_mode(Mode::Unfused, || {
            program
                .run_vectors(&[&resident(&a), &resident(&w)])
                .remove(0)
        });
        assert_eq!(bits(unfused.as_slice()), bits(gpu.as_slice()));
    }

    #[test]
    fn an_f16_adam_step_runs_on_the_gpu() {
        use tensorcrate::optim::{Adam, Rule};

        let mut rng = Lcg(6);
        let start = rng.vector::<f16>(500, -1.0, 1.0);
        let gradient = rng.vector::<f16>(500, -1.0, 1.0);

        // The default ε = 1e-8 is below f16's smallest subnormal, so it rounds
        // to zero and a small gradient's second moment underflows with it: an
        // f16 Adam needs an ε f16 can hold.
        let mut host = Vector::new(start.clone());
        let mut host_rule = Adam::new(f16::from_f64(0.01));
        host_rule.epsilon = f16::from_f64(1e-3);
        let mut gpu = resident(&start);
        let mut gpu_rule = Adam::new(f16::from_f64(0.01));
        gpu_rule.epsilon = f16::from_f64(1e-3);
        for _ in 0..3 {
            host_rule.update(&mut host, &Vector::new(gradient.clone()));
            gpu_rule.update(&mut gpu, &resident(&gradient));
        }
        assert!(gpu.is_device_resident());
        close(gpu.as_slice(), host.data(), 2.0, "adam");
    }

    #[test]
    fn reverse_mode_differentiates_resident_bf16() {
        let tape = Tape::<Metal>::new();
        let mut rng = Lcg(7);
        let a = Matrix::from_flat(4, 6, rng.vector::<bf16>(24, -1.0, 1.0));
        let x = Vector::new(rng.vector::<bf16>(6, -1.0, 1.0));
        let ra = tape.matrix(a.to_backend::<Metal>());
        let rx = tape.vector(x.to_backend::<Metal>());
        let mapped = ra.matvec(&rx).tanh();
        mapped.dot(&mapped).backward();
        assert!(rx.grad().is_device_resident());

        let host_tape = Tape::<Host>::new();
        let ha = host_tape.matrix(a);
        let hx = host_tape.vector(x);
        let mapped = ha.matvec(&hx).tanh();
        mapped.dot(&mapped).backward();
        close(rx.grad().as_slice(), hx.grad().data(), 4.0, "gradient");
        close(
            ra.grad().as_slice(),
            ha.grad().data(),
            4.0,
            "matrix gradient",
        );
    }

    #[test]
    fn half_precision_stacking_stays_on_the_device() {
        use tensorcrate::tensors::Backend;

        let value = |x: f64| f16::from_f64(x);
        let rows = [
            Metal::store_vector(&[value(1.0), value(2.0)]),
            Metal::store_vector(&[value(3.0), value(4.0)]),
        ];
        let stacked = Metal::vstack(&rows, 2);
        assert!(stacked.is_device_resident());
        assert_eq!(
            Metal::matrix_slice(&stacked),
            [1.0, 2.0, 3.0, 4.0].map(value)
        );
        let columns = Metal::hstack(&rows, 2);
        assert!(columns.is_device_resident());
        assert_eq!(
            Metal::matrix_slice(&columns),
            [1.0, 3.0, 2.0, 4.0].map(value)
        );
    }

    /// The resident results above could in principle have been computed on the
    /// host and uploaded — a fallback leaves its result resident too. The
    /// dispatch count is what shows the work ran on the GPU: one per operation,
    /// and no synchronization, which a host fallback would need to read its
    /// operands.
    #[cfg(feature = "counters")]
    #[test]
    fn half_precision_operations_dispatch_gpu_kernels() {
        use tensorcrate::counters;

        fn dispatches<T: MetalElement>() {
            let mut rng = Lcg(9);
            let a = resident(&rng.vector::<T>(4096, -1.0, 1.0));
            let b = resident(&rng.vector::<T>(4096, 0.5, 1.0));
            let m =
                Matrix::from_flat(64, 64, rng.vector::<T>(4096, -1.0, 1.0)).to_backend::<Metal>();
            tensorcrate::metal::synchronize();

            let (_, counts) = counters::measure(|| {
                let sum = &a + &b;
                let quotient = &sum / &b;
                let root = quotient.analytic(Analytic::Exp);
                let product = m.matmul(&m);
                (root, product)
            });
            assert_eq!(counts.dispatches, 4, "{counts:?}");
            assert_eq!(counts.syncs, 0, "{counts:?}");
        }
        dispatches::<f16>();
        dispatches::<bf16>();
    }

    fn epilogue_matches_host<T: MetalElement + tensorcrate::tensors::fused::Element>(
        dtype: DType,
        ulps: f64,
    ) {
        use tensorcrate::tensors::Compare;
        use tensorcrate::tensors::fused::{Fusable, Remap};

        let mut b = Builder::<T>::new();
        let product = b.input(dtype);
        let bias = b.input_remapped(dtype, Remap::Row);
        let shifted = b.add(product, bias);
        let zero = b.constant(T::zero());
        let relu = b.compare(Compare::Max, shifted, zero);
        b.output(relu, dtype);
        let program = b.build().unwrap();

        let mut rng = Lcg(17);
        for tensorops in [true, false] {
            tensorcrate::metal::set_tensorops(tensorops);
            for (m, k, n) in [(3usize, 5usize, 2usize), (17, 33, 15), (65, 130, 67)] {
                let a = Matrix::<T, Host>::from_flat(m, k, rng.vector::<T>(m * k, -1.0, 1.0));
                let w = Matrix::<T, Host>::from_flat(k, n, rng.vector::<T>(k * n, -1.0, 1.0));
                let bias = Vector::<T, Host>::new(rng.vector::<T>(n, -1.0, 1.0));
                let host = program
                    .run_matmul(&a, &w, &[&bias as &dyn Fusable<Host>])
                    .remove(0)
                    .into_vector::<T>();
                let (ga, gw, gb) = (
                    a.to_backend::<Metal>(),
                    w.to_backend::<Metal>(),
                    bias.to_backend::<Metal>(),
                );
                let gpu = program
                    .run_matmul(&ga, &gw, &[&gb as &dyn Fusable<Metal>])
                    .remove(0)
                    .into_vector::<T>();
                assert!(gpu.is_device_resident());
                close(
                    &gpu.to_vec(),
                    &host.to_vec(),
                    ulps,
                    &format!("{m}×{k}×{n}, tensorops {tensorops}"),
                );
            }
        }
        tensorcrate::metal::set_tensorops(true);
    }

    #[test]
    fn compact_matmul_epilogues_on_the_gpu_match_the_host() {
        // The f32 accumulators sum in different orders, then round to the
        // compact type once; the epilogue rounds every operation.
        epilogue_matches_host::<f16>(DType::F16, 4.0);
        epilogue_matches_host::<bf16>(DType::Bf16, 4.0);
        epilogue_matches_host::<f32>(DType::F32, 64.0);
    }
}
