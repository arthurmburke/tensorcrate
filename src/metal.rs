//! GPU-accelerated tensor kernels via Apple Metal.
//!
//! Compiled only with the `metal` feature on macOS. It offloads large `f32`
//! matrix products (a tiled, threadgroup-shared-memory kernel) and elementwise
//! ops to the GPU. Metal compute shaders are 32-bit, so `f64` and matrix
//! inversion stay on the (vectorized) CPU path.
//!
//! Every entry point returns `Option`: if no Metal device is available, or any
//! step fails, the caller falls back to the CPU kernel, so enabling the feature
//! never changes results — only performance.
//!
//! Input/output buffers are recycled through a small per-thread pool
//! ([`Pool`]) so repeated calls avoid re-allocating GPU memory.

use std::cell::{OnceCell, RefCell};
use std::ptr::NonNull;

use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};

/// Threadgroup tile edge; must match `TILE` in the shader. 16×16 = 256 threads.
const TILE: usize = 16;

/// The compute kernels.
///
/// `matmul_tiled` stages `TILE×TILE` blocks of A and B into threadgroup memory
/// so each loaded value is reused `TILE` times, which is far more
/// bandwidth-efficient than reading straight from device memory.
const KERNELS: &str = r#"
#include <metal_stdlib>
using namespace metal;

#define TILE 16

kernel void matmul_tiled(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& K      [[buffer(4)]],
    constant uint& N      [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]])
{
    threadgroup float Asub[TILE][TILE];
    threadgroup float Bsub[TILE][TILE];

    uint row = gid.y;
    uint col = gid.x;
    float acc = 0.0f;

    uint tiles = (K + TILE - 1) / TILE;
    for (uint t = 0; t < tiles; t++) {
        uint a_col = t * TILE + tid.x;
        uint b_row = t * TILE + tid.y;
        Asub[tid.y][tid.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bsub[tid.y][tid.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint p = 0; p < TILE; p++) {
            acc += Asub[tid.y][p] * Bsub[p][tid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

kernel void elementwise(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& op     [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    float a = A[i];
    float b = B[i];
    switch (op) {
        case 0: C[i] = a + b; break;
        case 1: C[i] = a - b; break;
        case 2: C[i] = a * b; break;
        default: C[i] = a / b; break;
    }
}
"#;

/// A pool of reusable Metal buffers (recycled across calls to avoid repeated
/// allocation). A buffer is reused when it is at least as large as requested.
#[derive(Default)]
struct Pool {
    free: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

impl Pool {
    fn acquire(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        len: usize,
    ) -> Option<Retained<ProtocolObject<dyn MTLBuffer>>> {
        if let Some(pos) = self.free.iter().position(|b| b.length() >= len) {
            return Some(self.free.swap_remove(pos));
        }
        device.newBufferWithLength_options(len.max(1), MTLResourceOptions::StorageModeShared)
    }

    fn release(&mut self, buffer: Retained<ProtocolObject<dyn MTLBuffer>>) {
        const CAP: usize = 12;
        if self.free.len() < CAP {
            self.free.push(buffer);
        }
    }
}

/// Device, queue, compiled pipelines, and buffer pool — cached per thread so the
/// shaders compile once. Metal objects are not `Send`, so a thread-local keeps
/// everything on one thread.
struct Gpu {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    matmul: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    elementwise: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pool: RefCell<Pool>,
}

thread_local! {
    static GPU: OnceCell<Option<Gpu>> = const { OnceCell::new() };
}

fn build_gpu() -> Option<Gpu> {
    let device = MTLCreateSystemDefaultDevice()?;
    let queue = device.newCommandQueue()?;
    let source = NSString::from_str(KERNELS);
    let library = device
        .newLibraryWithSource_options_error(&source, None)
        .ok()?;
    let pipeline = |name: &str| {
        let function = library.newFunctionWithName(&NSString::from_str(name))?;
        device
            .newComputePipelineStateWithFunction_error(&function)
            .ok()
    };
    Some(Gpu {
        matmul: pipeline("matmul_tiled")?,
        elementwise: pipeline("elementwise")?,
        pool: RefCell::new(Pool::default()),
        device,
        queue,
    })
}

fn with_gpu<R>(f: impl FnOnce(&Gpu) -> Option<R>) -> Option<R> {
    autoreleasepool(|_| GPU.with(|cell| cell.get_or_init(build_gpu).as_ref().and_then(f)))
}

/// Copy `src` into the front of a shared buffer's storage.
fn upload(buffer: &ProtocolObject<dyn MTLBuffer>, src: &[f32]) {
    let dst = buffer.contents().as_ptr() as *mut f32;
    unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len()) };
}

/// Read `len` floats out of the front of a shared buffer's storage.
fn download(buffer: &ProtocolObject<dyn MTLBuffer>, len: usize) -> Vec<f32> {
    let src = buffer.contents().as_ptr() as *const f32;
    let mut out = vec![0.0f32; len];
    unsafe { std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), len) };
    out
}

/// GPU `f32` matrix product `C[m×n] = A[m×k] · B[k×n]` (row-major), using the
/// tiled kernel. Returns `None` if no device is available.
pub fn matmul_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Option<Vec<f32>> {
    with_gpu(|gpu| {
        let (buf_a, buf_b, buf_c) = {
            let mut pool = gpu.pool.borrow_mut();
            (
                pool.acquire(&gpu.device, a.len() * 4)?,
                pool.acquire(&gpu.device, b.len() * 4)?,
                pool.acquire(&gpu.device, m * n * 4)?,
            )
        };
        upload(&buf_a, a);
        upload(&buf_b, b);

        let command = gpu.queue.commandBuffer()?;
        let encoder = command.computeCommandEncoder()?;
        encoder.setComputePipelineState(&gpu.matmul);
        let (mu, ku, nu) = (m as u32, k as u32, n as u32);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&buf_a), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&buf_b), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&buf_c), 0, 2);
            encoder.setBytes_length_atIndex(NonNull::from(&mu).cast(), 4, 3);
            encoder.setBytes_length_atIndex(NonNull::from(&ku).cast(), 4, 4);
            encoder.setBytes_length_atIndex(NonNull::from(&nu).cast(), 4, 5);
        }

        // Uniform threadgroups (the tiled kernel uses barriers): one group per
        // TILE×TILE output block.
        let groups = MTLSize {
            width: n.div_ceil(TILE),
            height: m.div_ceil(TILE),
            depth: 1,
        };
        let per_group = MTLSize {
            width: TILE,
            height: TILE,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, per_group);
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();

        let out = download(&buf_c, m * n);
        let mut pool = gpu.pool.borrow_mut();
        pool.release(buf_a);
        pool.release(buf_b);
        pool.release(buf_c);
        Some(out)
    })
}

/// GPU elementwise `f32` op over two equal-length buffers. `op` is 0=add,
/// 1=sub, 2=mul, 3=div. Returns `None` if no device is available.
pub fn elementwise_f32(a: &[f32], b: &[f32], op: u32) -> Option<Vec<f32>> {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    with_gpu(|gpu| {
        let (buf_a, buf_b, buf_c) = {
            let mut pool = gpu.pool.borrow_mut();
            (
                pool.acquire(&gpu.device, len * 4)?,
                pool.acquire(&gpu.device, len * 4)?,
                pool.acquire(&gpu.device, len * 4)?,
            )
        };
        upload(&buf_a, a);
        upload(&buf_b, b);

        let command = gpu.queue.commandBuffer()?;
        let encoder = command.computeCommandEncoder()?;
        encoder.setComputePipelineState(&gpu.elementwise);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&buf_a), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&buf_b), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&buf_c), 0, 2);
            encoder.setBytes_length_atIndex(NonNull::from(&op).cast(), 4, 3);
        }

        // No barriers here, so a non-uniform 1-D dispatch is fine.
        let grid = MTLSize {
            width: len,
            height: 1,
            depth: 1,
        };
        let per_group = MTLSize {
            width: 256.min(len.max(1)),
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreads_threadsPerThreadgroup(grid, per_group);
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();

        let out = download(&buf_c, len);
        let mut pool = gpu.pool.borrow_mut();
        pool.release(buf_a);
        pool.release(buf_b);
        pool.release(buf_c);
        Some(out)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = acc;
            }
        }
        c
    }

    #[test]
    fn gpu_tiled_matmul_matches_cpu() {
        // Deliberately not multiples of TILE, to exercise the ragged tiles.
        let (m, k, n) = (70usize, 45usize, 33usize);
        let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.5 - 1.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 * 0.25 + 0.1).collect();

        match matmul_f32(&a, &b, m, k, n) {
            None => eprintln!("no Metal device; skipping GPU comparison"),
            Some(gpu) => {
                let cpu = cpu_matmul(&a, &b, m, k, n);
                for (g, c) in gpu.iter().zip(&cpu) {
                    assert!((g - c).abs() < 1e-2, "gpu={g} cpu={c}");
                }
            }
        }
    }

    #[test]
    fn gpu_elementwise_matches_cpu() {
        let a: Vec<f32> = (0..1000).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..1000).map(|i| (i % 9) as f32 + 1.0).collect();
        for (op, f) in [
            (0u32, (|x: f32, y| x + y) as fn(f32, f32) -> f32),
            (1, |x, y| x - y),
            (2, |x, y| x * y),
            (3, |x, y| x / y),
        ] {
            if let Some(gpu) = elementwise_f32(&a, &b, op) {
                for (i, g) in gpu.iter().enumerate() {
                    let want = f(a[i], b[i]);
                    assert!((g - want).abs() < 1e-3, "op {op} at {i}: {g} vs {want}");
                }
            }
        }
    }
}
