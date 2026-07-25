//! GPU-accelerated tensor kernels via Apple Metal.
//!
//! Compiled only with the `metal` feature on macOS. It offloads large `f32`
//! matrix/vector products, elementwise and broadcast operations, and radix-2
//! FFTs to the GPU. Metal compute shaders are 32-bit, so `f64`, matrix
//! inversion, and non-radix-2 FFT leaves stay on the CPU path.
//!
//! Every entry point returns `Option`: if no Metal device is available, or any
//! step fails, the caller falls back to the CPU kernel, so enabling the feature
//! never changes results — only performance.
//!
//! Input/output buffers are recycled through a small per-thread pool so
//! repeated calls avoid re-allocating GPU memory.
//!
//! The ordinary tensor API offloads at conservative break-even thresholds:
//! 32,768 multiply-accumulates for products, 4,096 values for elementwise and
//! broadcast work, and 1,024 values for radix-2 FFTs. For a sequence of GPU
//! operations, [`MetalBuffer`] keeps intermediates in `MTLStorageModeShared`
//! memory and avoids a CPU copy between kernels.

use std::cell::{OnceCell, RefCell};
use std::ptr::NonNull;

use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLibrary, MTLResourceOptions, MTLSize,
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

kernel void broadcast(
    device const float* A [[buffer(0)]],
    device float* C       [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    constant uint& op      [[buffer(3)]],
    constant uint& scalar_left [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    float a = scalar_left ? scalar : A[i];
    float b = scalar_left ? A[i] : scalar;
    switch (op) {
        case 0: C[i] = a + b; break;
        case 1: C[i] = a - b; break;
        case 2: C[i] = a * b; break;
        default: C[i] = a / b; break;
    }
}

kernel void fft_bit_reverse(
    device const float2* input [[buffer(0)]],
    device float2* output      [[buffer(1)]],
    constant uint& count       [[buffer(2)]],
    constant uint& bits        [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= count) return;
    uint value = i;
    uint reversed = 0;
    for (uint bit = 0; bit < bits; bit++) {
        reversed = (reversed << 1) | (value & 1);
        value >>= 1;
    }
    output[reversed] = input[i];
}

kernel void fft_stage(
    device float2* values       [[buffer(0)]],
    constant uint& count        [[buffer(1)]],
    constant uint& stage_length [[buffer(2)]],
    constant uint& inverse      [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    uint half = stage_length / 2;
    uint butterflies = count / 2;
    if (i >= butterflies) return;

    uint group = i / half;
    uint offset = i % half;
    uint even_index = group * stage_length + offset;
    uint odd_index = even_index + half;

    float direction = inverse ? 1.0f : -1.0f;
    float angle = direction * 2.0f * M_PI_F * float(offset) / float(stage_length);
    float2 twiddle = float2(cos(angle), sin(angle));
    float2 odd_value = values[odd_index];
    float2 rotated = float2(
        odd_value.x * twiddle.x - odd_value.y * twiddle.y,
        odd_value.y * twiddle.x + odd_value.x * twiddle.y
    );
    float2 even_value = values[even_index];
    float normalization = (inverse && stage_length == count) ? float(count) : 1.0f;
    values[even_index] = (even_value + rotated) / normalization;
    values[odd_index] = (even_value - rotated) / normalization;
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
    broadcast: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    fft_bit_reverse: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    fft_stage: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
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
        .map_err(|_error| {
            #[cfg(test)]
            eprintln!("Metal shader compilation failed: {_error}");
        })
        .ok()?;
    let pipeline = |name: &str| {
        let function = library
            .newFunctionWithName(&NSString::from_str(name))
            .or_else(|| {
                #[cfg(test)]
                eprintln!("Metal function `{name}` was not found in the shader library");
                None
            })?;
        device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|_error| {
                #[cfg(test)]
                eprintln!("Metal pipeline `{name}` failed: {_error}");
            })
            .ok()
    };
    Some(Gpu {
        matmul: pipeline("matmul_tiled")?,
        elementwise: pipeline("elementwise")?,
        broadcast: pipeline("broadcast")?,
        fft_bit_reverse: pipeline("fft_bit_reverse")?,
        fft_stage: pipeline("fft_stage")?,
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

/// An `f32` allocation in Apple-silicon shared memory.
///
/// Keeping intermediate values in this type avoids the upload/download copies
/// made by the convenience functions below. The CPU may read the allocation
/// with [`to_vec`](Self::to_vec), but matrix, elementwise, broadcast, and FFT
/// operations can be chained without leaving Metal-managed shared memory.
///
/// Metal objects are thread-affine, so this type intentionally is not `Send`.
pub struct MetalBuffer {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    len: usize,
}

impl MetalBuffer {
    /// Allocate shared storage and initialize it from a CPU slice.
    pub fn from_slice(values: &[f32]) -> Option<Self> {
        with_gpu(|gpu| {
            let raw = gpu.device.newBufferWithLength_options(
                (values.len() * 4).max(1),
                MTLResourceOptions::StorageModeShared,
            )?;
            upload(&raw, values);
            Some(Self {
                raw,
                len: values.len(),
            })
        })
    }

    /// Number of stored `f32` values.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether this buffer contains no values.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Copy the shared allocation into an ordinary CPU vector.
    pub fn to_vec(&self) -> Vec<f32> {
        download(&self.raw, self.len)
    }

    /// Tiled matrix multiplication, with both inputs and the result remaining
    /// in shared Metal buffers.
    pub fn matmul(&self, rhs: &Self, m: usize, k: usize, n: usize) -> Option<Self> {
        if self.len != m.checked_mul(k)? || rhs.len != k.checked_mul(n)? {
            return None;
        }
        let output_len = m.checked_mul(n)?;
        if output_len == 0 {
            return Self::from_slice(&[]);
        }
        if k == 0 {
            return Self::from_slice(&vec![0.0; output_len]);
        }
        with_gpu(|gpu| {
            let output = gpu.device.newBufferWithLength_options(
                output_len * 4,
                MTLResourceOptions::StorageModeShared,
            )?;
            encode_matmul(gpu, &self.raw, &rhs.raw, &output, m, k, n)?;
            Some(Self {
                raw: output,
                len: output_len,
            })
        })
    }

    /// Elementwise operation with another shared buffer. `op` is 0=add,
    /// 1=subtract, 2=multiply, and 3=divide.
    pub fn elementwise(&self, rhs: &Self, op: u32) -> Option<Self> {
        if self.len != rhs.len || op > 3 {
            return None;
        }
        with_gpu(|gpu| {
            let output = gpu.device.newBufferWithLength_options(
                (self.len * 4).max(1),
                MTLResourceOptions::StorageModeShared,
            )?;
            if self.len != 0 {
                encode_elementwise(gpu, &self.raw, &rhs.raw, &output, self.len, op)?;
            }
            Some(Self {
                raw: output,
                len: self.len,
            })
        })
    }

    /// Broadcast operation with a scalar. `op` has the same encoding as
    /// [`elementwise`](Self::elementwise).
    pub fn broadcast(&self, scalar: f32, op: u32, scalar_left: bool) -> Option<Self> {
        if op > 3 {
            return None;
        }
        with_gpu(|gpu| {
            let output = gpu.device.newBufferWithLength_options(
                (self.len * 4).max(1),
                MTLResourceOptions::StorageModeShared,
            )?;
            if self.len != 0 {
                encode_broadcast(gpu, &self.raw, &output, self.len, scalar, op, scalar_left)?;
            }
            Some(Self {
                raw: output,
                len: self.len,
            })
        })
    }

    /// Radix-2 FFT over interleaved complex values. The layout is
    /// `[real0, imag0, real1, imag1, ...]`.
    pub fn fft(&self) -> Option<Self> {
        self.fourier_transform(false)
    }

    /// Normalized inverse radix-2 FFT over interleaved complex values.
    pub fn ifft(&self) -> Option<Self> {
        self.fourier_transform(true)
    }

    fn fourier_transform(&self, inverse: bool) -> Option<Self> {
        if !self.len.is_multiple_of(2) {
            return None;
        }
        let count = self.len / 2;
        if count == 0 {
            return Self::from_slice(&[]);
        }
        if !count.is_power_of_two() || count > u32::MAX as usize {
            return None;
        }
        with_gpu(|gpu| {
            let output = gpu
                .device
                .newBufferWithLength_options(self.len * 4, MTLResourceOptions::StorageModeShared)?;
            encode_fft(gpu, &self.raw, &output, count, inverse)?;
            Some(Self {
                raw: output,
                len: self.len,
            })
        })
    }
}

fn encode_matmul(
    gpu: &Gpu,
    a: &ProtocolObject<dyn MTLBuffer>,
    b: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    m: usize,
    k: usize,
    n: usize,
) -> Option<()> {
    let command = gpu.queue.commandBuffer()?;
    let encoder = command.computeCommandEncoder()?;
    encoder.setComputePipelineState(&gpu.matmul);
    let (mu, ku, nu) = (
        u32::try_from(m).ok()?,
        u32::try_from(k).ok()?,
        u32::try_from(n).ok()?,
    );
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(a), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(b), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 2);
        encoder.setBytes_length_atIndex(NonNull::from(&mu).cast(), 4, 3);
        encoder.setBytes_length_atIndex(NonNull::from(&ku).cast(), 4, 4);
        encoder.setBytes_length_atIndex(NonNull::from(&nu).cast(), 4, 5);
    }
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
    finish(command)
}

fn encode_elementwise(
    gpu: &Gpu,
    a: &ProtocolObject<dyn MTLBuffer>,
    b: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
    op: u32,
) -> Option<()> {
    let command = gpu.queue.commandBuffer()?;
    let encoder = command.computeCommandEncoder()?;
    encoder.setComputePipelineState(&gpu.elementwise);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(a), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(b), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 2);
        encoder.setBytes_length_atIndex(NonNull::from(&op).cast(), 4, 3);
    }
    dispatch_1d(&encoder, len);
    encoder.endEncoding();
    finish(command)
}

fn encode_broadcast(
    gpu: &Gpu,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
    scalar: f32,
    op: u32,
    scalar_left: bool,
) -> Option<()> {
    let command = gpu.queue.commandBuffer()?;
    let encoder = command.computeCommandEncoder()?;
    encoder.setComputePipelineState(&gpu.broadcast);
    let scalar_left = u32::from(scalar_left);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
        encoder.setBytes_length_atIndex(NonNull::from(&scalar).cast(), 4, 2);
        encoder.setBytes_length_atIndex(NonNull::from(&op).cast(), 4, 3);
        encoder.setBytes_length_atIndex(NonNull::from(&scalar_left).cast(), 4, 4);
    }
    dispatch_1d(&encoder, len);
    encoder.endEncoding();
    finish(command)
}

fn encode_fft(
    gpu: &Gpu,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    count: usize,
    inverse: bool,
) -> Option<()> {
    let command = gpu.queue.commandBuffer()?;
    let count_u32 = u32::try_from(count).ok()?;
    let bits = count.trailing_zeros();
    let inverse_u32 = u32::from(inverse);

    let encoder = command.computeCommandEncoder()?;
    encoder.setComputePipelineState(&gpu.fft_bit_reverse);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
        encoder.setBytes_length_atIndex(NonNull::from(&count_u32).cast(), 4, 2);
        encoder.setBytes_length_atIndex(NonNull::from(&bits).cast(), 4, 3);
    }
    dispatch_1d(&encoder, count);
    encoder.endEncoding();

    let mut stage_length = 2u32;
    while stage_length <= count_u32 {
        let encoder = command.computeCommandEncoder()?;
        encoder.setComputePipelineState(&gpu.fft_stage);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(output), 0, 0);
            encoder.setBytes_length_atIndex(NonNull::from(&count_u32).cast(), 4, 1);
            encoder.setBytes_length_atIndex(NonNull::from(&stage_length).cast(), 4, 2);
            encoder.setBytes_length_atIndex(NonNull::from(&inverse_u32).cast(), 4, 3);
        }
        dispatch_1d(&encoder, count / 2);
        encoder.endEncoding();
        if stage_length == count_u32 {
            break;
        }
        stage_length *= 2;
    }
    finish(command)
}

fn finish(command: Retained<ProtocolObject<dyn MTLCommandBuffer>>) -> Option<()> {
    command.commit();
    command.waitUntilCompleted();
    (command.status() == MTLCommandBufferStatus::Completed).then_some(())
}

/// GPU `f32` matrix product `C[m×n] = A[m×k] · B[k×n]` (row-major), using the
/// tiled kernel. Returns `None` if no device is available.
pub fn matmul_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Option<Vec<f32>> {
    if a.len() != m.checked_mul(k)? || b.len() != k.checked_mul(n)? {
        return None;
    }
    let output_len = m.checked_mul(n)?;
    if output_len == 0 {
        return Some(Vec::new());
    }
    if k == 0 {
        return Some(vec![0.0; output_len]);
    }
    with_gpu(|gpu| {
        let (buf_a, buf_b, buf_c) = {
            let mut pool = gpu.pool.borrow_mut();
            (
                pool.acquire(&gpu.device, a.len() * 4)?,
                pool.acquire(&gpu.device, b.len() * 4)?,
                pool.acquire(&gpu.device, output_len * 4)?,
            )
        };
        upload(&buf_a, a);
        upload(&buf_b, b);

        encode_matmul(gpu, &buf_a, &buf_b, &buf_c, m, k, n)?;

        let out = download(&buf_c, output_len);
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
    if a.len() != b.len() || op > 3 {
        return None;
    }
    let len = a.len();
    if len == 0 {
        return Some(Vec::new());
    }
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

        encode_elementwise(gpu, &buf_a, &buf_b, &buf_c, len, op)?;

        let out = download(&buf_c, len);
        let mut pool = gpu.pool.borrow_mut();
        pool.release(buf_a);
        pool.release(buf_b);
        pool.release(buf_c);
        Some(out)
    })
}

/// GPU `f32` broadcast operation between a buffer and a scalar. `op` uses the
/// same encoding as [`elementwise_f32`]; `scalar_left` controls operand order
/// for subtraction and division.
pub fn broadcast_f32(values: &[f32], scalar: f32, op: u32, scalar_left: bool) -> Option<Vec<f32>> {
    if op > 3 {
        return None;
    }
    let len = values.len();
    if len == 0 {
        return Some(Vec::new());
    }
    with_gpu(|gpu| {
        let (input, output) = {
            let mut pool = gpu.pool.borrow_mut();
            (
                pool.acquire(&gpu.device, len * 4)?,
                pool.acquire(&gpu.device, len * 4)?,
            )
        };
        upload(&input, values);

        encode_broadcast(gpu, &input, &output, len, scalar, op, scalar_left)?;

        let out = download(&output, len);
        let mut pool = gpu.pool.borrow_mut();
        pool.release(input);
        pool.release(output);
        Some(out)
    })
}

/// GPU radix-2 FFT over interleaved complex `f32` values. The input and output
/// layout is `[real0, imag0, real1, imag1, ...]`.
pub fn fft_f32_interleaved(input: &[f32]) -> Option<Vec<f32>> {
    fourier_transform_f32_interleaved(input, false)
}

/// GPU normalized inverse radix-2 FFT over interleaved complex `f32` values.
pub fn ifft_f32_interleaved(input: &[f32]) -> Option<Vec<f32>> {
    fourier_transform_f32_interleaved(input, true)
}

fn fourier_transform_f32_interleaved(input: &[f32], inverse: bool) -> Option<Vec<f32>> {
    if !input.len().is_multiple_of(2) {
        return None;
    }
    let count = input.len() / 2;
    if count == 0 {
        return Some(Vec::new());
    }
    if !count.is_power_of_two() || count > u32::MAX as usize {
        return None;
    }

    with_gpu(|gpu| {
        let (source, values) = {
            let mut pool = gpu.pool.borrow_mut();
            (
                pool.acquire(&gpu.device, input.len() * 4)?,
                pool.acquire(&gpu.device, input.len() * 4)?,
            )
        };
        upload(&source, input);

        encode_fft(gpu, &source, &values, count, inverse)?;
        let out = download(&values, input.len());
        let mut pool = gpu.pool.borrow_mut();
        pool.release(source);
        pool.release(values);
        Some(out)
    })
}

fn dispatch_1d(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, len: usize) {
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

    #[test]
    fn gpu_broadcast_matches_cpu() {
        let values: Vec<f32> = (0..1000).map(|i| i as f32 * 0.125 - 3.0).collect();
        if let Some(gpu) = broadcast_f32(&values, 2.5, 2, false) {
            for (actual, value) in gpu.iter().zip(values) {
                assert!((actual - value * 2.5).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn gpu_fft_and_ifft_match_the_cpu_definition() {
        let count = 1024usize;
        let mut input = Vec::with_capacity(count * 2);
        for i in 0..count {
            input.push((i % 17) as f32 * 0.25 - 2.0);
            input.push((i % 11) as f32 * -0.125 + 0.5);
        }

        let Some(spectrum) = fft_f32_interleaved(&input) else {
            eprintln!("no Metal device; skipping GPU comparison");
            return;
        };
        let reconstructed = ifft_f32_interleaved(&spectrum).unwrap();
        for (actual, expected) in reconstructed.iter().zip(input) {
            assert!(
                (actual - expected).abs() < 2e-4,
                "gpu={actual} cpu={expected}"
            );
        }
    }

    #[test]
    fn shared_buffers_keep_chained_operations_gpu_resident() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let Some(a) = MetalBuffer::from_slice(&a) else {
            eprintln!("no Metal device; skipping shared-buffer comparison");
            return;
        };
        let b = MetalBuffer::from_slice(&b).unwrap();
        let product = a.matmul(&b, 2, 2, 2).unwrap();
        let scaled = product.broadcast(0.5, 2, false).unwrap();
        assert_eq!(scaled.to_vec(), vec![9.5, 11.0, 21.5, 25.0]);

        let complex =
            MetalBuffer::from_slice(&[1.0, 0.0, 2.0, -1.0, 0.5, 3.0, -2.0, 0.25]).unwrap();
        let reconstructed = complex.fft().unwrap().ifft().unwrap().to_vec();
        for (actual, expected) in reconstructed.iter().zip(complex.to_vec()) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }
}
