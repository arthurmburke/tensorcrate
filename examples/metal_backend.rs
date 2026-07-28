//! What the `Metal` tensor backend buys you: the same arithmetic, without the
//! copies. Run with:
//!
//! ```text
//! cargo run --release --example metal_backend
//! ```
//!
//! Both columns run the identical kernels on the identical GPU. The difference is
//! that the `Host` column stores its tensors in stack arrays, so every single
//! operation has to upload both operands into Metal buffers and download the
//! result, while the `Metal` column uploads once at the start and downloads once
//! at the end.
//!
//! These are wall-clock microbenchmarks, not statistically rigorous.

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() {
    use tensorcrate::tensors::{Host, Matrix, Metal, Vector};
    use std::time::Instant;

    fn bench(label: &str, iters: u32, mut f: impl FnMut()) -> f64 {
        f(); // warm-up: shader compilation, buffer pool
        let start = Instant::now();
        for _ in 0..iters {
            f();
        }
        let secs = start.elapsed().as_secs_f64() / iters as f64;
        println!("  {label:<34} {:>10.1} µs", secs * 1e6);
        secs
    }

    // A chain of matrix products: A·A·A·… Each link is a real GPU dispatch in
    // both columns, so what is being measured is the traffic around them.
    const SIZE: usize = 256;
    const LINKS: usize = 8;
    println!("matmul chain ({SIZE}×{SIZE}, {LINKS} products)");
    let a = Matrix::<f32, SIZE, SIZE>::from_rows(std::array::from_fn(|row| {
        std::array::from_fn(|col| ((row * SIZE + col) % 19) as f32 * 0.125 - 1.0)
    }));

    let mut sink = 0.0f32;
    let host = bench("host backend (copies per op)", 20, || {
        let mut product = a;
        for _ in 0..LINKS {
            product = product.matmul(&a);
        }
        sink += product.data()[0][0];
    });
    let resident = bench("metal backend (resident)", 20, || {
        let gpu = a.to_backend::<Metal>();
        let mut product = gpu.matmul(&gpu);
        for _ in 2..=LINKS {
            product = product.matmul(&gpu);
        }
        sink += product.to_backend::<Host>().data()[0][0];
    });
    println!("  speedup: {:.2}×\n", host / resident);

    // Elementwise work is pure bandwidth, so the copies dominate completely.
    const LEN: usize = 65_536;
    const STEPS: usize = 8;
    println!("elementwise chain (N = {LEN}, {STEPS} operations)");
    let v = Vector::<f32, LEN>::new(std::array::from_fn(|i| (i % 31) as f32 - 15.0));

    let host = bench("host backend (copies per op)", 50, || {
        let mut acc = v;
        for _ in 0..STEPS {
            acc = (acc * v).scale(0.5);
        }
        sink += acc.data()[0];
    });
    let resident = bench("metal backend (resident)", 50, || {
        let gpu = v.to_backend::<Metal>();
        let mut acc = &gpu * &gpu;
        acc = acc.scale(0.5);
        for _ in 2..=STEPS {
            acc = (&acc * &gpu).scale(0.5);
        }
        sink += acc.to_backend::<Host>().data()[0];
    });
    println!("  speedup: {:.2}×\n", host / resident);

    // The backend is explicit, and a switch is the only copy in sight.
    let gpu = a.to_backend::<Metal>();
    println!(
        "residency: {} (device-resident: {}), and the answers agree: {}",
        if gpu.is_device_resident() {
            "GPU-shared memory"
        } else {
            "no Metal device, CPU fallback"
        },
        gpu.matmul(&gpu).is_device_resident(),
        gpu.matmul(&gpu).to_backend::<Host>() == a.matmul(&a),
    );
    let _ = sink;
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    println!("the Metal backend needs macOS and the `metal` feature");
}
