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
//! The tensor API on its default `Host` backend offloads above fixed size
//! thresholds: 32,768 multiply-accumulates for products, 4,096 values for
//! elementwise and broadcast work, and 1,024 values for radix-2 FFTs. Because
//! those tensors live in stack arrays, each such call has to upload its operands
//! and download its result.
//!
//! To keep a *sequence* of operations on the GPU, put the tensors on the
//! [`Metal`](crate::tensors::Metal) backend, which stores their elements in
//! `MTLStorageModeShared` memory and passes the allocations from kernel to
//! kernel. [`MetalBuffer`] is that storage, usable directly when the
//! statically-shaped tensor types do not fit.

use std::cell::{OnceCell, RefCell};
use std::mem::ManuallyDrop;
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

/// Number of analytic function op codes the shader understands, taken from the
/// enum that defines them so the two cannot drift apart. The codes themselves are
/// [`Analytic::code`](crate::tensors::Analytic::code).
const ANALYTIC_OPS: u32 = crate::tensors::Analytic::ALL.len() as u32;

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
    constant uint& accumulate [[buffer(6)]],
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
        // `accumulate` adds into C instead of overwriting it, which is what the
        // forward-mode tangent `A'B + AB'` and reverse-mode gradient
        // accumulation both want — one dispatch instead of a separate add.
        C[row * N + col] = accumulate ? C[row * N + col] + acc : acc;
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

// The analytic functions, by op code. These must agree with
// `tensors::kernels::Analytic::code`, and each derivative must be written the
// same way as the matching `Dual` impl in `numbers.rs` so the GPU and CPU paths
// differ only by float precision. An unknown code yields NaN, which shows up as
// a loud test failure rather than a silently wrong number.
inline float analytic_value(uint op, float x) {
    switch (op) {
        case 0:  return sin(x);
        case 1:  return cos(x);
        case 2:  return tan(x);
        case 3:  return 1.0f / cos(x);   // sec
        case 4:  return 1.0f / sin(x);   // csc
        case 5:  return asin(x);
        case 6:  return acos(x);
        case 7:  return atan(x);
        case 8:  return exp(x);
        case 9:  return log(x);          // natural log
        case 10: return sinh(x);
        case 11: return cosh(x);
        case 12: return tanh(x);
        default: return NAN;
    }
}

inline float analytic_derivative(uint op, float x) {
    switch (op) {
        case 0:  return cos(x);
        case 1:  return -sin(x);
        case 2:  { float c = cos(x); return 1.0f / (c * c); }
        case 3:  { float c = cos(x); return sin(x) / (c * c); }
        case 4:  { float s = sin(x); return -cos(x) / (s * s); }
        case 5:  return 1.0f / sqrt(1.0f - x * x);
        case 6:  return -1.0f / sqrt(1.0f - x * x);
        case 7:  return 1.0f / (1.0f + x * x);
        case 8:  return exp(x);
        case 9:  return 1.0f / x;
        case 10: return cosh(x);
        case 11: return sinh(x);
        case 12: { float t = tanh(x); return 1.0f - t * t; }
        default: return NAN;
    }
}

kernel void unary(
    device const float* A [[buffer(0)]],
    device float* C       [[buffer(1)]],
    constant uint& op     [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    C[i] = analytic_value(op, A[i]);
}

// Forward-mode AD: `f(v + d·ε) = f(v) + f'(v)·d·ε`. Both parts come out of one
// dispatch, which also reads `v` only once.
kernel void unary_dual(
    device const float* value    [[buffer(0)]],
    device const float* tangent  [[buffer(1)]],
    device float* out_value      [[buffer(2)]],
    device float* out_tangent    [[buffer(3)]],
    constant uint& op            [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    float v = value[i];
    out_value[i] = analytic_value(op, v);
    out_tangent[i] = analytic_derivative(op, v) * tangent[i];
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
    // `half` is a type name in MSL, so the span cannot be called that.
    uint half_length = stage_length / 2;
    uint butterflies = count / 2;
    if (i >= butterflies) return;

    uint group = i / half_length;
    uint offset = i % half_length;
    uint even_index = group * stage_length + offset;
    uint odd_index = even_index + half_length;

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
    unary: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    unary_dual: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    fft_bit_reverse: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    fft_stage: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pool: RefCell<Pool>,
    /// Command buffers committed but not yet waited on — see [`commit`].
    pending: RefCell<Vec<Retained<ProtocolObject<dyn MTLCommandBuffer>>>>,
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
        unary: pipeline("unary")?,
        unary_dual: pipeline("unary_dual")?,
        fft_bit_reverse: pipeline("fft_bit_reverse")?,
        fft_stage: pipeline("fft_stage")?,
        pool: RefCell::new(Pool::default()),
        pending: RefCell::new(Vec::new()),
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
/// made by the convenience functions below. The CPU may read the allocation in
/// place with [`as_slice`](Self::as_slice) — shared storage is CPU-cached
/// memory, not a separate GPU pool — while matrix, elementwise, broadcast, and
/// FFT operations chain without ever leaving it.
///
/// This is the storage behind the [`Metal`](crate::tensors::Metal) tensor
/// backend, which wraps it in the statically-shaped `Vector`/`Matrix` API.
///
/// Metal objects are thread-affine, so this type intentionally is not `Send`.
/// Dropping one returns its allocation to the thread's buffer pool.
pub struct MetalBuffer {
    /// Returned to the pool by `Drop`, hence `ManuallyDrop`.
    raw: ManuallyDrop<Retained<ProtocolObject<dyn MTLBuffer>>>,
    len: usize,
}

impl MetalBuffer {
    /// Allocate `len` floats of shared storage, recycling a pooled allocation
    /// when one is big enough. The contents are unspecified, so every caller
    /// either uploads into it or has a kernel write every element.
    fn allocate(len: usize) -> Option<Self> {
        with_gpu(|gpu| {
            let raw = gpu
                .pool
                .borrow_mut()
                .acquire(&gpu.device, (len * 4).max(1))?;
            Some(Self {
                raw: ManuallyDrop::new(raw),
                len,
            })
        })
    }

    /// Allocate shared storage and initialize it from a CPU slice.
    pub fn from_slice(values: &[f32]) -> Option<Self> {
        let buffer = Self::allocate(values.len())?;
        upload(&buffer.raw, values);
        Some(buffer)
    }

    /// Number of stored `f32` values.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether this buffer contains no values.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Borrow the shared allocation as an ordinary slice, without copying.
    ///
    /// Operations are submitted without waiting, so this is one of the points
    /// where queued GPU work has to have actually happened; it blocks until it
    /// has. Every CPU read of a `Metal`-backed tensor comes through here.
    pub fn as_slice(&self) -> &[f32] {
        with_gpu(sync);
        // SAFETY: `MTLStorageModeShared` memory is CPU-readable at
        // `contents()`, and holds `len` initialized floats — a buffer is only
        // handed out after an upload or a kernel that writes every element. The
        // `sync` above drained every command buffer that could still be writing
        // it, and nothing on this thread can submit more while the borrow is
        // alive, so no GPU write is in flight.
        unsafe { std::slice::from_raw_parts(self.raw.contents().as_ptr().cast::<f32>(), self.len) }
    }

    /// Copy the shared allocation into an ordinary CPU vector.
    pub fn to_vec(&self) -> Vec<f32> {
        self.as_slice().to_vec()
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
        let output = Self::allocate(output_len)?;
        with_gpu(|gpu| encode_matmul(gpu, &self.raw, &rhs.raw, &output.raw, m, k, n, false))?;
        Some(output)
    }

    /// `target += A·B`, accumulated by the matmul kernel itself rather than by a
    /// second elementwise pass.
    ///
    /// The exclusive borrow of `target` is what makes the in-place GPU write
    /// sound: no [`as_slice`](Self::as_slice) borrow can be alive at the same
    /// time, and distinct buffers never share an allocation.
    pub fn matmul_accumulate(
        &self,
        rhs: &Self,
        target: &mut Self,
        m: usize,
        k: usize,
        n: usize,
    ) -> Option<()> {
        if self.len != m.checked_mul(k)? || rhs.len != k.checked_mul(n)? {
            return None;
        }
        if target.len != m.checked_mul(n)? {
            return None;
        }
        // An empty result, or an empty inner dimension, adds nothing.
        if target.len == 0 || k == 0 {
            return Some(());
        }
        with_gpu(|gpu| encode_matmul(gpu, &self.raw, &rhs.raw, &target.raw, m, k, n, true))
    }

    /// Apply an analytic function elementwise. `op` is an
    /// [`Analytic::code`](crate::tensors::Analytic::code).
    pub fn unary(&self, op: u32) -> Option<Self> {
        if op >= ANALYTIC_OPS {
            return None;
        }
        let output = Self::allocate(self.len)?;
        if self.len != 0 {
            with_gpu(|gpu| encode_unary(gpu, &self.raw, &output.raw, self.len, op))?;
        }
        Some(output)
    }

    /// Apply an analytic function to a value/tangent pair — forward-mode
    /// differentiation, `f(v) + f'(v)·d·ε` — returning `(value, tangent)`.
    ///
    /// One dispatch produces both parts. `op` is an
    /// [`Analytic::code`](crate::tensors::Analytic::code).
    pub fn unary_dual(&self, tangent: &Self, op: u32) -> Option<(Self, Self)> {
        if self.len != tangent.len || op >= ANALYTIC_OPS {
            return None;
        }
        let out_value = Self::allocate(self.len)?;
        let out_tangent = Self::allocate(self.len)?;
        if self.len != 0 {
            with_gpu(|gpu| {
                encode_unary_dual(
                    gpu,
                    &self.raw,
                    &tangent.raw,
                    &out_value.raw,
                    &out_tangent.raw,
                    self.len,
                    op,
                )
            })?;
        }
        Some((out_value, out_tangent))
    }

    /// Elementwise operation with another shared buffer. `op` is 0=add,
    /// 1=subtract, 2=multiply, and 3=divide.
    pub fn elementwise(&self, rhs: &Self, op: u32) -> Option<Self> {
        if self.len != rhs.len || op > 3 {
            return None;
        }
        let output = Self::allocate(self.len)?;
        if self.len != 0 {
            with_gpu(|gpu| {
                encode_elementwise(gpu, &self.raw, &rhs.raw, &output.raw, self.len, op)
            })?;
        }
        Some(output)
    }

    /// Broadcast operation with a scalar. `op` has the same encoding as
    /// [`elementwise`](Self::elementwise).
    pub fn broadcast(&self, scalar: f32, op: u32, scalar_left: bool) -> Option<Self> {
        if op > 3 {
            return None;
        }
        let output = Self::allocate(self.len)?;
        if self.len != 0 {
            with_gpu(|gpu| {
                encode_broadcast(
                    gpu,
                    &self.raw,
                    &output.raw,
                    self.len,
                    scalar,
                    op,
                    scalar_left,
                )
            })?;
        }
        Some(output)
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
        let output = Self::allocate(self.len)?;
        with_gpu(|gpu| encode_fft(gpu, &self.raw, &output.raw, count, inverse))?;
        Some(output)
    }
}

impl Drop for MetalBuffer {
    fn drop(&mut self) {
        // SAFETY: `raw` is live until here and this runs exactly once, so it is
        // never taken twice and nothing reads the field afterwards.
        let raw = unsafe { ManuallyDrop::take(&mut self.raw) };
        // Hand the allocation back for reuse. Metal objects are thread-affine
        // and this type is not `Send`, so this is the pool it came from.
        // `try_with`/`try_borrow_mut` cover the cases where the pool cannot take
        // it — thread-local teardown, or a drop during another allocation — and
        // then the allocation is simply released to Metal.
        let _ = GPU.try_with(|cell| {
            if let Some(Some(gpu)) = cell.get()
                && let Ok(mut pool) = gpu.pool.try_borrow_mut()
            {
                pool.release(raw);
            }
        });
    }
}

// One argument over clippy's threshold: the kernel takes three shapes and an
// accumulate flag, and naming them beats packing them into a struct here.
#[allow(clippy::too_many_arguments)]
fn encode_matmul(
    gpu: &Gpu,
    a: &ProtocolObject<dyn MTLBuffer>,
    b: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    m: usize,
    k: usize,
    n: usize,
    accumulate: bool,
) -> Option<()> {
    let command = gpu.queue.commandBuffer()?;
    let encoder = command.computeCommandEncoder()?;
    encoder.setComputePipelineState(&gpu.matmul);
    let (mu, ku, nu) = (
        u32::try_from(m).ok()?,
        u32::try_from(k).ok()?,
        u32::try_from(n).ok()?,
    );
    let accumulate = u32::from(accumulate);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(a), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(b), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 2);
        encoder.setBytes_length_atIndex(NonNull::from(&mu).cast(), 4, 3);
        encoder.setBytes_length_atIndex(NonNull::from(&ku).cast(), 4, 4);
        encoder.setBytes_length_atIndex(NonNull::from(&nu).cast(), 4, 5);
        encoder.setBytes_length_atIndex(NonNull::from(&accumulate).cast(), 4, 6);
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
    commit(gpu, command)
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
    commit(gpu, command)
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
    commit(gpu, command)
}

fn encode_unary(
    gpu: &Gpu,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
    op: u32,
) -> Option<()> {
    let command = gpu.queue.commandBuffer()?;
    let encoder = command.computeCommandEncoder()?;
    encoder.setComputePipelineState(&gpu.unary);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
        encoder.setBytes_length_atIndex(NonNull::from(&op).cast(), 4, 2);
    }
    dispatch_1d(&encoder, len);
    encoder.endEncoding();
    commit(gpu, command)
}

fn encode_unary_dual(
    gpu: &Gpu,
    value: &ProtocolObject<dyn MTLBuffer>,
    tangent: &ProtocolObject<dyn MTLBuffer>,
    out_value: &ProtocolObject<dyn MTLBuffer>,
    out_tangent: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
    op: u32,
) -> Option<()> {
    let command = gpu.queue.commandBuffer()?;
    let encoder = command.computeCommandEncoder()?;
    encoder.setComputePipelineState(&gpu.unary_dual);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(value), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(tangent), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(out_value), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(out_tangent), 0, 3);
        encoder.setBytes_length_atIndex(NonNull::from(&op).cast(), 4, 4);
    }
    dispatch_1d(&encoder, len);
    encoder.endEncoding();
    commit(gpu, command)
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
    commit(gpu, command)
}

/// Submit `command` and return without waiting for the GPU.
///
/// Blocking here is what made a chain of resident operations cost a full round
/// trip per link (~145 µs each) even though the whole point of the [`Metal`]
/// backend is that nothing is copied between them. Instead the buffer is parked
/// in [`Gpu::pending`] and the wait is deferred to [`sync`], which every path
/// that reads shared memory from the CPU calls first.
///
/// Deferring is safe because all of this module's work goes to a single
/// [`MTLCommandQueue`] on one thread: buffers execute in commit order and Metal
/// tracks the hazards between them, so a kernel still sees its predecessor's
/// output. Resources stay alive too — a command buffer retains what its encoders
/// reference, so dropping a [`MetalBuffer`] with work in flight cannot free the
/// allocation early.
///
/// The one thing given up is per-call error reporting: a command that fails on
/// the GPU is noticed at the next `sync` rather than by the call that encoded
/// it. Encoding failures are still reported immediately.
///
/// [`Metal`]: crate::tensors::Metal
fn commit(gpu: &Gpu, command: Retained<ProtocolObject<dyn MTLCommandBuffer>>) -> Option<()> {
    command.commit();
    let mut pending = gpu.pending.borrow_mut();
    pending.push(command);
    // Cap the backlog so a long run of un-read operations cannot retain command
    // buffers without bound.
    if pending.len() >= 64 {
        drop(pending);
        return sync(gpu);
    }
    Some(())
}

/// Block until every command buffer committed so far has finished.
///
/// Call this before the CPU reads any shared allocation the GPU may still be
/// writing. Returns `None` if any of them failed.
fn sync(gpu: &Gpu) -> Option<()> {
    // Taken by value so a re-entrant call cannot see a half-drained list, and so
    // the buffers are released once they have been waited on.
    let pending = std::mem::take(&mut *gpu.pending.borrow_mut());
    let mut ok = true;
    for command in &pending {
        command.waitUntilCompleted();
        if command.status() != MTLCommandBufferStatus::Completed {
            #[cfg(test)]
            eprintln!("Metal command buffer finished as {:?}", command.status());
            ok = false;
        }
    }
    ok.then_some(())
}

/// Block until all GPU work submitted on this thread has completed.
///
/// Operations on [`Metal`](crate::tensors::Metal)-backed tensors are queued and
/// return before the GPU has run them, so this is the way to make outstanding
/// work observable — reading the values back already does it implicitly. It is
/// also what a benchmark needs in order to time GPU work rather than submission.
pub fn synchronize() {
    with_gpu(sync);
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

        sync(gpu)?;
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

        sync(gpu)?;
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
        sync(gpu)?;
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
    fn accumulating_matmul_adds_into_its_target() {
        let (m, k, n) = (3usize, 4usize, 2usize);
        let a: Vec<f32> = (0..m * k).map(|i| (i % 5) as f32 - 2.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 3) as f32 * 0.5).collect();
        let Some(buf_a) = MetalBuffer::from_slice(&a) else {
            eprintln!("no Metal device; skipping accumulation comparison");
            return;
        };
        let buf_b = MetalBuffer::from_slice(&b).unwrap();

        let product = cpu_matmul(&a, &b, m, k, n);
        let seed: Vec<f32> = (0..m * n).map(|i| i as f32).collect();
        let mut target = MetalBuffer::from_slice(&seed).unwrap();
        buf_a
            .matmul_accumulate(&buf_b, &mut target, m, k, n)
            .expect("accumulating dispatch");

        for (index, (actual, base)) in target.as_slice().iter().zip(&seed).enumerate() {
            let expected = base + product[index];
            assert!((actual - expected).abs() < 1e-4, "{actual} vs {expected}");
        }
    }

    #[test]
    fn unary_dual_applies_a_function_and_its_derivative() {
        let value: Vec<f32> = (0..64).map(|i| (i % 9) as f32 * 0.1 + 0.05).collect();
        let tangent: Vec<f32> = (0..64).map(|i| (i % 4) as f32 - 1.5).collect();
        let Some(buf_value) = MetalBuffer::from_slice(&value) else {
            eprintln!("no Metal device; skipping unary comparison");
            return;
        };
        let buf_tangent = MetalBuffer::from_slice(&tangent).unwrap();

        // Op 12 is tanh: f' = 1 − tanh².
        let (values, tangents) = buf_value.unary_dual(&buf_tangent, 12).unwrap();
        for (index, (&actual, &expected)) in values.as_slice().iter().zip(&value).enumerate() {
            let want = expected.tanh();
            assert!((actual - want).abs() < 1e-4, "value at {index}");
            let derivative = 1.0 - want * want;
            let want_tangent = derivative * tangent[index];
            assert!(
                (tangents.as_slice()[index] - want_tangent).abs() < 1e-4,
                "tangent at {index}"
            );
        }

        // An op code the shader does not know is rejected before dispatch.
        assert!(buf_value.unary(ANALYTIC_OPS).is_none());
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

    /// Operations are committed without waiting, so a long dependent chain is
    /// the thing that would break if command buffers on one queue did not run in
    /// commit order, or if a kernel could start before its input was written.
    /// Each link here depends on the previous one and every link is exactly
    /// representable, so any reordering, overlap, or dropped stage is an
    /// unambiguous mismatch rather than a rounding difference.
    #[test]
    fn deferred_completion_preserves_the_order_of_a_dependent_chain() {
        const LINKS: usize = 250; // past the 64-buffer flush point, several times
        let start: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let Some(mut buffer) = MetalBuffer::from_slice(&start) else {
            eprintln!("no Metal device; skipping deferred-completion chain");
            return;
        };

        let ones = MetalBuffer::from_slice(&vec![1.0f32; 64]).unwrap();
        for _ in 0..LINKS {
            // +1 via broadcast, then +1 via elementwise: two kernels per link,
            // each reading what the one before it just wrote.
            buffer = buffer.broadcast(1.0, 0, false).unwrap();
            buffer = buffer.elementwise(&ones, 0).unwrap();
        }

        let expected: Vec<f32> = (0..64).map(|i| (i + 2 * LINKS) as f32).collect();
        assert_eq!(buffer.to_vec(), expected);
    }

    /// `synchronize` has to be enough on its own: after it returns, work queued
    /// earlier must be visible to a later read that does not itself sync.
    #[test]
    fn synchronize_makes_queued_work_observable() {
        let Some(buffer) = MetalBuffer::from_slice(&[3.0f32; 32]) else {
            eprintln!("no Metal device; skipping synchronize check");
            return;
        };
        let doubled = buffer.broadcast(2.0, 2, false).unwrap();
        synchronize();
        assert_eq!(doubled.to_vec(), vec![6.0f32; 32]);
    }
}
