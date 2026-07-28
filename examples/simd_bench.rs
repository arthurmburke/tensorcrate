//! Rough throughput comparison of the architecture-specific SIMD kernels
//! against equivalent scalar loops. Run with:
//!
//! ```text
//! cargo run --release --example simd_bench --no-default-features --features simd
//! ```
//!
//! These are wall-clock microbenchmarks, not statistically rigorous — they exist
//! to show the kernels are pulling their weight, and where the crossover lands.

#[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
fn main() {
    use tensorcrate::simd::f32k;
    use std::time::Instant;

    fn bench(label: &str, iters: u32, mut f: impl FnMut()) -> f64 {
        // warm-up
        f();
        let start = Instant::now();
        for _ in 0..iters {
            f();
        }
        let secs = start.elapsed().as_secs_f64() / iters as f64;
        println!("  {label:<28} {:>10.3} µs", secs * 1e6);
        secs
    }

    println!("dot (N = 65_536)");
    let a: Vec<f32> = (0..65_536).map(|i| (i as f32).sin()).collect();
    let b: Vec<f32> = (0..65_536).map(|i| (i as f32).cos()).collect();
    let mut sink = 0.0f32;
    let scalar = bench("scalar", 2_000, || {
        let mut s = 0.0f32;
        for i in 0..a.len() {
            s += a[i] * b[i];
        }
        sink += s;
    });
    let simd = bench("simd", 2_000, || sink += f32k::dot(&a, &b));
    println!("  speedup: {:.2}×\n", scalar / simd);

    println!("matmul (128 × 128 × 128)");
    const N: usize = 128;
    let m: Vec<f32> = (0..N * N).map(|i| (i as f32 % 13.0) - 6.0).collect();
    let mut out = vec![0.0f32; N * N];
    let scalar = bench("scalar (i,j,p)", 40, || {
        for i in 0..N {
            for j in 0..N {
                let mut s = 0.0f32;
                for p in 0..N {
                    s += m[i * N + p] * m[p * N + j];
                }
                out[i * N + j] = s;
            }
        }
        sink += out[0];
    });
    let simd = bench("simd (broadcast-A)", 40, || {
        f32k::matmul(&m, &m, N, N, N, &mut out);
        sink += out[0];
    });
    println!("  speedup: {:.2}×", scalar / simd);

    std::hint::black_box(sink);
}

#[cfg(not(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64"))))]
fn main() {
    eprintln!("build with --features simd on aarch64 or x86_64 to run this benchmark");
}
