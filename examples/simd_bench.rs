//! Rough throughput comparison of the architecture-specific SIMD kernels
//! against equivalent scalar loops, plus a size sweep used to tune where
//! compile-time-sized specialization pays off. Run with:
//!
//! ```text
//! cargo run --release --example simd_bench --no-default-features --features simd
//! ```
//!
//! These are wall-clock microbenchmarks, not statistically rigorous — they exist
//! to show the kernels are pulling their weight, and where the crossover lands.
//!
//! Every timed closure passes its operands through [`std::hint::black_box`]. That
//! is load-bearing rather than decorative: the sweep calls kernels whose lengths
//! are compile-time constants, so without the barrier the optimizer is entitled
//! to hoist an entire loop-invariant dot product out of the iteration loop and
//! report a time of zero.

#[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
fn main() {
    use std::hint::black_box;
    use std::time::Instant;
    use tensorcrate::simd::{f32k, fft_f32};
    use tensorcrate::tensors::{Matrix, Vector};

    /// Times `f`, reporting the per-iteration average. `iters` is scaled by the
    /// caller so that short kernels still run long enough to measure.
    fn bench(label: &str, iters: u32, mut f: impl FnMut()) -> f64 {
        f(); // warm-up
        let start = Instant::now();
        for _ in 0..iters {
            f();
        }
        let secs = start.elapsed().as_secs_f64() / iters as f64;
        println!("  {label:<28} {:>10.3} µs", secs * 1e6);
        secs
    }

    /// Nanoseconds, for the sweep tables where microseconds lose resolution.
    fn bench_ns(iters: u32, mut f: impl FnMut()) -> f64 {
        f();
        let start = Instant::now();
        for _ in 0..iters {
            f();
        }
        start.elapsed().as_secs_f64() / iters as f64 * 1e9
    }

    fn ramp<const N: usize>(phase: f32) -> [f32; N] {
        std::array::from_fn(|i| ((i as f32 + phase) * 0.017).sin())
    }

    fn square<const N: usize>() -> [[f32; N]; N] {
        std::array::from_fn(|i| std::array::from_fn(|j| ((i * N + j) as f32 % 13.0) - 6.0))
    }

    let mut sink = 0.0f32;

    // ---- scalar vs SIMD, at the sizes the kernels were written for ----------

    println!("dot (N = 65_536)");
    let a: Vec<f32> = (0..65_536).map(|i| (i as f32).sin()).collect();
    let b: Vec<f32> = (0..65_536).map(|i| (i as f32).cos()).collect();
    let scalar = bench("scalar", 2_000, || {
        let (a, b) = (black_box(&a), black_box(&b));
        let mut s = 0.0f32;
        for i in 0..a.len() {
            s += a[i] * b[i];
        }
        sink += s;
    });
    let simd = bench("simd", 2_000, || {
        sink += f32k::dot(black_box(&a), black_box(&b))
    });
    println!("  speedup: {:.2}×\n", scalar / simd);

    println!("matmul (128 × 128 × 128)");
    const BIG: usize = 128;
    let m: Vec<f32> = (0..BIG * BIG).map(|i| (i as f32 % 13.0) - 6.0).collect();
    let mut out = vec![0.0f32; BIG * BIG];
    let scalar = bench("scalar (i,j,p)", 40, || {
        let m = black_box(&m);
        for i in 0..BIG {
            for j in 0..BIG {
                let mut s = 0.0f32;
                for p in 0..BIG {
                    s += m[i * BIG + p] * m[p * BIG + j];
                }
                out[i * BIG + j] = s;
            }
        }
        sink += out[0];
    });
    let simd = bench("simd (broadcast-A)", 40, || {
        f32k::matmul(black_box(&m), black_box(&m), BIG, BIG, BIG, &mut out);
        sink += out[0];
    });
    println!("  speedup: {:.2}×\n", scalar / simd);

    // ---- size sweep --------------------------------------------------------
    //
    // The comparison that matters for compile-time sizing is not scalar-vs-SIMD
    // but runtime-length-vs-const-length, at the small and medium shapes this
    // crate actually instantiates. `runtime` passes the length as a `usize`
    // argument; `const` will call the specialized entry point once it exists.

    println!("dot: opaque length vs compile-time length");
    println!(
        "  {:>8}  {:>12}  {:>12}  {:>9}",
        "N", "opaque ns", "const ns", "speedup"
    );

    macro_rules! dot_sweep {
        ($($n:expr => $iters:expr),* $(,)?) => {$({
            const N: usize = $n;
            let a = ramp::<N>(0.0);
            let b = ramp::<N>(1.5);
            // `black_box(&a[..])` hides the length behind slice metadata;
            // `black_box(&a)` keeps it in the array type, where the optimizer
            // still sees it and specializes the `#[inline]` kernel.
            let opaque = bench_ns($iters, || {
                sink += f32k::dot(black_box(&a[..]), black_box(&b[..]));
            });
            let constant = bench_ns($iters, || {
                sink += f32k::dot(black_box(&a), black_box(&b));
            });
            println!(
                "  {:>8}  {:>12.2}  {:>12.2}  {:>8.2}×",
                N, opaque, constant, opaque / constant
            );
        })*};
    }
    dot_sweep!(
        16 => 200_000,
        32 => 200_000,
        64 => 200_000,
        128 => 100_000,
        256 => 100_000,
        1024 => 50_000,
    );
    println!();

    println!("matmul: opaque dims vs compile-time dims (square N×N×N)");
    println!(
        "  {:>8}  {:>12}  {:>12}  {:>9}",
        "N", "opaque ns", "const ns", "speedup"
    );

    macro_rules! matmul_sweep {
        ($($n:expr => $iters:expr),* $(,)?) => {$({
            const N: usize = $n;
            let a = square::<N>();
            let b = square::<N>();
            let mut c = [[0.0f32; N]; N];
            // Genuinely runtime dims: `black_box` hides the extents from the
            // optimizer, which is the only way to see the unspecialized kernel.
            let opaque = bench_ns($iters, || {
                f32k::matmul(
                    black_box(a.as_flattened()),
                    black_box(b.as_flattened()),
                    black_box(N), black_box(N), black_box(N),
                    c.as_flattened_mut(),
                );
                // The whole output must be observed, not just one element:
                // reading `c[0][0]` alone lets the optimizer delete every other
                // dot product and report a physically impossible time.
                black_box(&mut c);
            });
            // Literal dims through the same kernel, which is what a call site
            // with a size the optimizer can see gets: `matmul` is `#[inline]`,
            // so a constant extent still folds even though the tensor types
            // carry theirs at runtime.
            let constant = bench_ns($iters, || {
                f32k::matmul(
                    black_box(a.as_flattened()),
                    black_box(b.as_flattened()),
                    N, N, N,
                    c.as_flattened_mut(),
                );
                black_box(&mut c);
            });
            println!(
                "  {:>8}  {:>12.2}  {:>12.2}  {:>8.2}×",
                N, opaque, constant, opaque / constant
            );
        })*};
    }
    matmul_sweep!(
        4 => 500_000,
        8 => 200_000,
        16 => 100_000,
        32 => 20_000,
        64 => 5_000,
        128 => 1_000,
    );
    println!();

    // ---- end-to-end through the public API ---------------------------------
    //
    // What a caller actually pays. Above `MIN_MATMUL_OPS` (512 ops, i.e. N ≥ 8)
    // `Matrix::matmul` takes the SIMD path, which currently heap-allocates an
    // `R*C` output buffer and then copies it out element by element — costs the
    // kernel sweep above does not see.

    println!("Matrix::matmul end-to-end (N×N×N)");
    println!(
        "  {:>8}  {:>12}  {:>12}  {:>12}",
        "N", "total ns", "kernel ns", "overhead ns"
    );

    macro_rules! api_sweep {
        ($($n:expr => $iters:expr),* $(,)?) => {$({
            const N: usize = $n;
            let rows = square::<N>();
            let a = Matrix::<f32>::from_rows(rows);
            let b = Matrix::<f32>::from_rows(rows);
            // Kernel and end-to-end are timed in the same run so the difference
            // between them is real rather than an artifact of run-to-run drift.
            let mut scratch = [[0.0f32; N]; N];
            let kernel = bench_ns($iters, || {
                f32k::matmul(
                    black_box(rows.as_flattened()),
                    black_box(rows.as_flattened()),
                    N, N, N,
                    scratch.as_flattened_mut(),
                );
                black_box(&mut scratch);
            });
            let total = bench_ns($iters, || {
                let c = black_box(&a).matmul(black_box(&b));
                black_box(&c);
            });
            println!(
                "  {:>8}  {:>12.2}  {:>12.2}  {:>12.2}",
                N, total, kernel, total - kernel
            );
        })*};
    }
    api_sweep!(
        8 => 200_000,
        16 => 100_000,
        32 => 20_000,
        64 => 5_000,
        128 => 1_000,
    );

    println!();

    // ---- FFT ---------------------------------------------------------------
    //
    // Radix-2 over an interleaved [re, im, …] buffer. The stage loop runs
    // log2(N) times, so any per-stage allocation is paid log2(N) times per
    // transform.

    println!("fft radix-2 (interleaved complex)");
    println!("  {:>8}  {:>12}  {:>10}", "N", "ns", "stages");
    for &n in &[64usize, 256, 1024, 4096] {
        let mut buf: Vec<f32> = (0..2 * n).map(|i| (i as f32 * 0.031).sin()).collect();
        let iters = (2_000_000 / n) as u32;
        let elapsed = bench_ns(iters, || {
            fft_f32::radix2(black_box(&mut buf), n, -1.0);
            black_box(&mut buf);
        });
        println!("  {:>8}  {:>12.2}  {:>10}", n, elapsed, n.trailing_zeros());
    }

    println!();

    // ---- Vector::fft end-to-end --------------------------------------------
    //
    // Three different code paths depending on the length: power-of-two goes to
    // the vectorized radix-2 kernel, composite lengths to the recursive
    // mixed-radix decomposition, and lengths whose smallest prime factor
    // exceeds 15 to the quadratic direct DFT.

    println!("Vector::fft end-to-end");
    println!(
        "  {:>8}  {:>12}  {:>16}  {:>12}",
        "N", "ns", "path", "ns/element"
    );

    macro_rules! fft_sweep {
        ($($n:expr => ($iters:expr, $path:expr)),* $(,)?) => {$({
            const N: usize = $n;
            let v = Vector::<f32>::new(ramp::<N>(0.25));
            let elapsed = bench_ns($iters, || {
                let out = black_box(&v).fft();
                black_box(&out);
            });
            println!(
                "  {:>8}  {:>12.2}  {:>16}  {:>12.2}",
                N, elapsed, $path, elapsed / N as f64
            );
        })*};
    }
    fft_sweep!(
        64 => (100_000, "radix-2 simd"),
        256 => (50_000, "radix-2 simd"),
        1024 => (20_000, "radix-2 simd"),
        105 => (20_000, "mixed 3·5·7"),
        240 => (20_000, "mixed 2^4·3·5"),
        1000 => (5_000, "mixed 2^3·5^3"),
        17 => (100_000, "direct dft"),
        101 => (20_000, "direct dft"),
    );

    black_box(sink);
}

#[cfg(not(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64"))))]
fn main() {
    eprintln!("build with --features simd on aarch64 or x86_64 to run this benchmark");
}
