//! GPU-accelerated tensor kernels via Apple Metal.
//!
//! Compiled only with the `metal` feature on macOS. It offloads large `f32`
//! matrix/vector products, elementwise and broadcast operations, and radix-2
//! FFTs to the GPU. Metal compute shaders are 32-bit, so `f64`, matrix
//! inversion, and non-radix-2 FFT leaves stay on the CPU path.
//!
//! Every entry point returns `Option`: if no Metal device is available or an
//! operation cannot be encoded, the caller falls back to the CPU kernel. A
//! failure reported asynchronously by Metal is surfaced as a panic at the next
//! synchronization point rather than exposing an incomplete output buffer.
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
use std::mem::{ManuallyDrop, size_of};
use std::ptr::NonNull;

use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};

use crate::tensors::{Analytic, BinaryOp};

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

enum class BinaryOp : ushort {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
    Rem = 4
};

enum class AnalyticOp : ushort {
    Sin = 0,
    Cos = 1,
    Tan = 2,
    Sec = 3,
    Csc = 4,
    Arcsin = 5,
    Arccos = 6,
    Arctan = 7,
    Exp = 8,
    Ln = 9,
    Sinh = 10,
    Cosh = 11,
    Tanh = 12
};

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
    constant BinaryOp& op [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    float a = A[i];
    float b = B[i];
    switch (op) {
        case BinaryOp::Add: C[i] = a + b; break;
        case BinaryOp::Sub: C[i] = a - b; break;
        case BinaryOp::Mul: C[i] = a * b; break;
        default: C[i] = a / b; break;
    }
}

kernel void broadcast(
    device const float* A [[buffer(0)]],
    device float* C       [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    constant BinaryOp& op  [[buffer(3)]],
    constant uint& scalar_left [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    float a = scalar_left ? scalar : A[i];
    float b = scalar_left ? A[i] : scalar;
    switch (op) {
        case BinaryOp::Add: C[i] = a + b; break;
        case BinaryOp::Sub: C[i] = a - b; break;
        case BinaryOp::Mul: C[i] = a * b; break;
        default: C[i] = a / b; break;
    }
}

// Copy one input vector into a row or column of a row-major output matrix.
// `output_stride == 1` writes a contiguous row; otherwise it scatters a column.
kernel void stack_vector(
    device const float* input [[buffer(0)]],
    device float* output      [[buffer(1)]],
    constant uint& count      [[buffer(2)]],
    constant uint& offset     [[buffer(3)]],
    constant uint& output_stride [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i < count) {
        output[offset + i * output_stride] = input[i];
    }
}

// Concatenate two row-major matrices with the same row count. Each thread
// writes one output element; both input reads and output writes are coalesced.
kernel void concat_horizontal(
    device const float* left  [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device float* output      [[buffer(2)]],
    constant uint& rows       [[buffer(3)]],
    constant uint& left_cols  [[buffer(4)]],
    constant uint& right_cols [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint output_cols = left_cols + right_cols;
    uint row = gid.y;
    uint col = gid.x;
    if (row >= rows || col >= output_cols) return;

    output[row * output_cols + col] = col < left_cols
        ? left[row * left_cols + col]
        : right[row * right_cols + col - left_cols];
}

// Place one matrix into a horizontal block of a wider row-major matrix. The
// encoder dispatches this once per input matrix in the same command buffer.
kernel void merge_horizontal(
    device const float* input [[buffer(0)]],
    device float* output      [[buffer(1)]],
    constant uint& rows       [[buffer(2)]],
    constant uint& input_cols [[buffer(3)]],
    constant uint& output_cols [[buffer(4)]],
    constant uint& col_offset [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    if (row < rows && col < input_cols) {
        output[row * output_cols + col_offset + col] =
            input[row * input_cols + col];
    }
}

// Transpose through a padded threadgroup tile. Adjacent threads read adjacent
// input values and write adjacent output values; the extra column avoids bank
// conflicts when the tile is read in the opposite direction.
kernel void transpose_tiled(
    device const float* input [[buffer(0)]],
    device float* output      [[buffer(1)]],
    constant uint& rows       [[buffer(2)]],
    constant uint& cols       [[buffer(3)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]])
{
    threadgroup float tile[TILE][TILE + 1];

    uint input_col = group.x * TILE + tid.x;
    uint input_row = group.y * TILE + tid.y;
    if (input_row < rows && input_col < cols) {
        tile[tid.y][tid.x] = input[input_row * cols + input_col];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint output_col = group.y * TILE + tid.x;
    uint output_row = group.x * TILE + tid.y;
    if (output_row < cols && output_col < rows) {
        output[output_row * rows + output_col] = tile[tid.x][tid.y];
    }
}

// These variants must agree with `tensors::kernels::Analytic`, and each
// derivative must match the corresponding `Dual` implementation.
inline float analytic_value(AnalyticOp op, float x) {
    switch (op) {
        case AnalyticOp::Sin:    return sin(x);
        case AnalyticOp::Cos:    return cos(x);
        case AnalyticOp::Tan:    return tan(x);
        case AnalyticOp::Sec:    return 1.0f / cos(x);
        case AnalyticOp::Csc:    return 1.0f / sin(x);
        case AnalyticOp::Arcsin: return asin(x);
        case AnalyticOp::Arccos: return acos(x);
        case AnalyticOp::Arctan: return atan(x);
        case AnalyticOp::Exp:    return exp(x);
        case AnalyticOp::Ln:     return log(x);
        case AnalyticOp::Sinh:   return sinh(x);
        case AnalyticOp::Cosh:   return cosh(x);
        case AnalyticOp::Tanh:   return tanh(x);
        default: return NAN;
    }
}

inline float analytic_derivative(AnalyticOp op, float x) {
    switch (op) {
        case AnalyticOp::Sin:    return cos(x);
        case AnalyticOp::Cos:    return -sin(x);
        case AnalyticOp::Tan:    { float c = cos(x); return 1.0f / (c * c); }
        case AnalyticOp::Sec:    { float c = cos(x); return sin(x) / (c * c); }
        case AnalyticOp::Csc:    { float s = sin(x); return -cos(x) / (s * s); }
        case AnalyticOp::Arcsin: return 1.0f / sqrt(1.0f - x * x);
        case AnalyticOp::Arccos: return -1.0f / sqrt(1.0f - x * x);
        case AnalyticOp::Arctan: return 1.0f / (1.0f + x * x);
        case AnalyticOp::Exp:    return exp(x);
        case AnalyticOp::Ln:     return 1.0f / x;
        case AnalyticOp::Sinh:   return cosh(x);
        case AnalyticOp::Cosh:   return sinh(x);
        case AnalyticOp::Tanh:   { float t = tanh(x); return 1.0f - t * t; }
        default: return NAN;
    }
}

kernel void unary(
    device const float* A [[buffer(0)]],
    device float* C       [[buffer(1)]],
    constant AnalyticOp& op [[buffer(2)]],
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
    constant AnalyticOp& op      [[buffer(4)]],
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
    /// Safe to hand out: every command buffer that could have touched these has
    /// completed.
    free: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Released while GPU work was still queued, so a pending kernel may yet
    /// read or write them. [`sync`] promotes these into `free`.
    ///
    /// Recycling one of these early is not a use-after-free — a command buffer
    /// retains the resources it references — but it would let an upload, or the
    /// next kernel, race the writes still owed to the previous owner.
    retiring: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
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

    /// Take an allocation back. `work_in_flight` says whether any command buffer
    /// has been committed but not yet waited on; if so the allocation waits for
    /// the next [`sync`] before it can be handed out again.
    fn release(&mut self, buffer: Retained<ProtocolObject<dyn MTLBuffer>>, work_in_flight: bool) {
        const CAP: usize = 12;
        let bucket = if work_in_flight {
            &mut self.retiring
        } else {
            &mut self.free
        };
        if bucket.len() < CAP {
            bucket.push(buffer);
        }
    }

    /// Every queued command buffer has completed, so anything held back is now
    /// safe to reuse.
    fn retire(&mut self) {
        self.free.append(&mut self.retiring);
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
    stack_vector: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    concat_horizontal: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    merge_horizontal: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    transpose: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
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
        stack_vector: pipeline("stack_vector")?,
        concat_horizontal: pipeline("concat_horizontal")?,
        merge_horizontal: pipeline("merge_horizontal")?,
        transpose: pipeline("transpose_tiled")?,
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
        with_gpu(|gpu| {
            sync_or_panic(gpu);
            Some(())
        })
        .expect("a Metal buffer cannot outlive its thread-local device");
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

    /// Transpose a row-major `rows × cols` matrix into a new shared buffer.
    pub(crate) fn transpose(&self, rows: usize, cols: usize) -> Option<Self> {
        if self.len != rows.checked_mul(cols)? {
            return None;
        }
        let output = Self::allocate(self.len)?;
        if self.len != 0 {
            with_gpu(|gpu| encode_transpose(gpu, &self.raw, &output.raw, rows, cols))?;
        }
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

    /// Apply an analytic function elementwise.
    pub fn unary(&self, op: Analytic) -> Option<Self> {
        let output = Self::allocate(self.len)?;
        if self.len != 0 {
            with_gpu(|gpu| encode_unary(gpu, &self.raw, &output.raw, self.len, op))?;
        }
        Some(output)
    }

    /// Apply an analytic function to a value/tangent pair — forward-mode
    /// differentiation, `f(v) + f'(v)·d·ε` — returning `(value, tangent)`.
    ///
    /// One dispatch produces both parts.
    pub fn unary_dual(&self, tangent: &Self, op: Analytic) -> Option<(Self, Self)> {
        if self.len != tangent.len {
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

    /// Elementwise operation with another shared buffer.
    pub fn elementwise(&self, rhs: &Self, op: BinaryOp) -> Option<Self> {
        if self.len != rhs.len || op == BinaryOp::Rem {
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
    pub fn broadcast(&self, scalar: f32, op: BinaryOp, scalar_left: bool) -> Option<Self> {
        if op == BinaryOp::Rem {
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

    /// Stack equal-length buffers as rows of one row-major matrix.
    pub(crate) fn vstack(inputs: &[&Self], vector_len: usize) -> Option<Self> {
        Self::stack(inputs, vector_len, 1, |index| index * vector_len)
    }

    /// Stack equal-length buffers as columns of one row-major matrix.
    pub(crate) fn hstack(inputs: &[&Self], vector_len: usize) -> Option<Self> {
        let columns = inputs.len();
        Self::stack(inputs, vector_len, columns, |index| index)
    }

    fn stack(
        inputs: &[&Self],
        vector_len: usize,
        output_stride: usize,
        offset: impl Fn(usize) -> usize,
    ) -> Option<Self> {
        if inputs.iter().any(|input| input.len != vector_len) {
            return None;
        }
        let output = Self::allocate(inputs.len().checked_mul(vector_len)?)?;
        if output.len != 0 {
            with_gpu(|gpu| {
                encode_stack(gpu, inputs, &output.raw, vector_len, output_stride, offset)
            })?;
        }
        Some(output)
    }

    /// Concatenate two row-major matrices horizontally.
    pub(crate) fn concat_matrix(
        &self,
        rhs: &Self,
        rows: usize,
        left_cols: usize,
        right_cols: usize,
    ) -> Option<Self> {
        if self.len != rows.checked_mul(left_cols)? || rhs.len != rows.checked_mul(right_cols)? {
            return None;
        }
        let output_cols = left_cols.checked_add(right_cols)?;
        let output = Self::allocate(rows.checked_mul(output_cols)?)?;
        if output.len != 0 {
            with_gpu(|gpu| {
                encode_concat(
                    gpu,
                    &self.raw,
                    &rhs.raw,
                    &output.raw,
                    rows,
                    left_cols,
                    right_cols,
                )
            })?;
        }
        Some(output)
    }

    /// Concatenate two row-major matrices vertically using contiguous blits.
    pub(crate) fn stack_matrix(
        &self,
        rhs: &Self,
        top_rows: usize,
        bottom_rows: usize,
        cols: usize,
    ) -> Option<Self> {
        if self.len != top_rows.checked_mul(cols)? || rhs.len != bottom_rows.checked_mul(cols)? {
            return None;
        }
        let output = Self::allocate(self.len.checked_add(rhs.len)?)?;
        if output.len != 0 {
            with_gpu(|gpu| {
                encode_matrix_stack(gpu, &self.raw, &rhs.raw, &output.raw, self.len, rhs.len)
            })?;
        }
        Some(output)
    }

    /// Merge equally shaped row-major matrices horizontally.
    pub(crate) fn hmerge(inputs: &[&Self], rows: usize, cols: usize) -> Option<Self> {
        let matrix_len = rows.checked_mul(cols)?;
        if inputs.iter().any(|input| input.len != matrix_len) {
            return None;
        }
        let output = Self::allocate(matrix_len.checked_mul(inputs.len())?)?;
        if output.len != 0 {
            with_gpu(|gpu| encode_hmerge(gpu, inputs, &output.raw, rows, cols))?;
        }
        Some(output)
    }

    /// Merge equally shaped row-major matrices vertically with contiguous blits.
    pub(crate) fn vmerge(inputs: &[&Self], rows: usize, cols: usize) -> Option<Self> {
        let matrix_len = rows.checked_mul(cols)?;
        if inputs.iter().any(|input| input.len != matrix_len) {
            return None;
        }
        let output = Self::allocate(matrix_len.checked_mul(inputs.len())?)?;
        if output.len != 0 {
            with_gpu(|gpu| encode_vmerge(gpu, inputs, &output.raw, matrix_len))?;
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
                // Anything committed but not yet waited on may still reference
                // this allocation, so it cannot go straight back into service.
                let work_in_flight = gpu
                    .pending
                    .try_borrow()
                    .is_ok_and(|queue| !queue.is_empty());
                pool.release(raw, work_in_flight);
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

fn encode_transpose(
    gpu: &Gpu,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    rows: usize,
    cols: usize,
) -> Option<()> {
    let rows_u32 = u32::try_from(rows).ok()?;
    let cols_u32 = u32::try_from(cols).ok()?;
    let command = gpu.queue.commandBuffer()?;
    let encoder = command.computeCommandEncoder()?;
    encoder.setComputePipelineState(&gpu.transpose);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
        encoder.setBytes_length_atIndex(NonNull::from(&rows_u32).cast(), 4, 2);
        encoder.setBytes_length_atIndex(NonNull::from(&cols_u32).cast(), 4, 3);
    }
    let groups = MTLSize {
        width: cols.div_ceil(TILE),
        height: rows.div_ceil(TILE),
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
    op: BinaryOp,
) -> Option<()> {
    let command = gpu.queue.commandBuffer()?;
    let encoder = command.computeCommandEncoder()?;
    encoder.setComputePipelineState(&gpu.elementwise);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(a), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(b), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 2);
        encoder.setBytes_length_atIndex(NonNull::from(&op).cast(), size_of::<BinaryOp>(), 3);
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
    op: BinaryOp,
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
        encoder.setBytes_length_atIndex(NonNull::from(&op).cast(), size_of::<BinaryOp>(), 3);
        encoder.setBytes_length_atIndex(NonNull::from(&scalar_left).cast(), 4, 4);
    }
    dispatch_1d(&encoder, len);
    encoder.endEncoding();
    commit(gpu, command)
}

fn encode_stack(
    gpu: &Gpu,
    inputs: &[&MetalBuffer],
    output: &ProtocolObject<dyn MTLBuffer>,
    vector_len: usize,
    output_stride: usize,
    offset: impl Fn(usize) -> usize,
) -> Option<()> {
    let count = u32::try_from(vector_len).ok()?;
    let output_stride = u32::try_from(output_stride).ok()?;
    let offsets = (0..inputs.len())
        .map(|index| u32::try_from(offset(index)).ok())
        .collect::<Option<Vec<_>>>()?;

    let command = gpu.queue.commandBuffer()?;
    let encoder = command.computeCommandEncoder()?;
    encoder.setComputePipelineState(&gpu.stack_vector);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
        encoder.setBytes_length_atIndex(NonNull::from(&count).cast(), 4, 2);
        encoder.setBytes_length_atIndex(NonNull::from(&output_stride).cast(), 4, 4);
    }
    for (input, offset) in inputs.iter().zip(&offsets) {
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&input.raw), 0, 0);
            encoder.setBytes_length_atIndex(NonNull::from(offset).cast(), 4, 3);
        }
        dispatch_1d(&encoder, vector_len);
    }
    encoder.endEncoding();
    commit(gpu, command)
}

fn encode_concat(
    gpu: &Gpu,
    left: &ProtocolObject<dyn MTLBuffer>,
    right: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    rows: usize,
    left_cols: usize,
    right_cols: usize,
) -> Option<()> {
    let rows_u32 = u32::try_from(rows).ok()?;
    let left_cols_u32 = u32::try_from(left_cols).ok()?;
    let right_cols_u32 = u32::try_from(right_cols).ok()?;
    let output_cols = left_cols.checked_add(right_cols)?;

    let command = gpu.queue.commandBuffer()?;
    let encoder = command.computeCommandEncoder()?;
    encoder.setComputePipelineState(&gpu.concat_horizontal);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(left), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(right), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 2);
        encoder.setBytes_length_atIndex(NonNull::from(&rows_u32).cast(), 4, 3);
        encoder.setBytes_length_atIndex(NonNull::from(&left_cols_u32).cast(), 4, 4);
        encoder.setBytes_length_atIndex(NonNull::from(&right_cols_u32).cast(), 4, 5);
    }
    let groups = MTLSize {
        width: output_cols.div_ceil(TILE),
        height: rows.div_ceil(TILE),
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

fn encode_matrix_stack(
    gpu: &Gpu,
    top: &ProtocolObject<dyn MTLBuffer>,
    bottom: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    top_len: usize,
    bottom_len: usize,
) -> Option<()> {
    let top_bytes = top_len.checked_mul(size_of::<f32>())?;
    let bottom_bytes = bottom_len.checked_mul(size_of::<f32>())?;
    let command = gpu.queue.commandBuffer()?;
    let encoder = command.blitCommandEncoder()?;
    unsafe {
        if top_bytes != 0 {
            encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                top, 0, output, 0, top_bytes,
            );
        }
        if bottom_bytes != 0 {
            encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                bottom,
                0,
                output,
                top_bytes,
                bottom_bytes,
            );
        }
    }
    encoder.endEncoding();
    commit(gpu, command)
}

fn encode_hmerge(
    gpu: &Gpu,
    inputs: &[&MetalBuffer],
    output: &ProtocolObject<dyn MTLBuffer>,
    rows: usize,
    cols: usize,
) -> Option<()> {
    let rows_u32 = u32::try_from(rows).ok()?;
    let cols_u32 = u32::try_from(cols).ok()?;
    let output_cols = cols.checked_mul(inputs.len())?;
    let output_cols_u32 = u32::try_from(output_cols).ok()?;
    let offsets = (0..inputs.len())
        .map(|index| u32::try_from(index.checked_mul(cols)?).ok())
        .collect::<Option<Vec<_>>>()?;

    let command = gpu.queue.commandBuffer()?;
    let encoder = command.computeCommandEncoder()?;
    encoder.setComputePipelineState(&gpu.merge_horizontal);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
        encoder.setBytes_length_atIndex(NonNull::from(&rows_u32).cast(), 4, 2);
        encoder.setBytes_length_atIndex(NonNull::from(&cols_u32).cast(), 4, 3);
        encoder.setBytes_length_atIndex(NonNull::from(&output_cols_u32).cast(), 4, 4);
    }
    let groups = MTLSize {
        width: cols.div_ceil(TILE),
        height: rows.div_ceil(TILE),
        depth: 1,
    };
    let per_group = MTLSize {
        width: TILE,
        height: TILE,
        depth: 1,
    };
    for (input, offset) in inputs.iter().zip(&offsets) {
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&input.raw), 0, 0);
            encoder.setBytes_length_atIndex(NonNull::from(offset).cast(), 4, 5);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, per_group);
    }
    encoder.endEncoding();
    commit(gpu, command)
}

fn encode_vmerge(
    gpu: &Gpu,
    inputs: &[&MetalBuffer],
    output: &ProtocolObject<dyn MTLBuffer>,
    matrix_len: usize,
) -> Option<()> {
    let matrix_bytes = matrix_len.checked_mul(size_of::<f32>())?;
    let command = gpu.queue.commandBuffer()?;
    let encoder = command.blitCommandEncoder()?;
    for (index, input) in inputs.iter().enumerate() {
        let destination_offset = index.checked_mul(matrix_bytes)?;
        unsafe {
            encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &input.raw,
                0,
                output,
                destination_offset,
                matrix_bytes,
            );
        }
    }
    encoder.endEncoding();
    commit(gpu, command)
}

fn encode_unary(
    gpu: &Gpu,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
    op: Analytic,
) -> Option<()> {
    let command = gpu.queue.commandBuffer()?;
    let encoder = command.computeCommandEncoder()?;
    encoder.setComputePipelineState(&gpu.unary);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
        encoder.setBytes_length_atIndex(NonNull::from(&op).cast(), size_of::<Analytic>(), 2);
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
    op: Analytic,
) -> Option<()> {
    let command = gpu.queue.commandBuffer()?;
    let encoder = command.computeCommandEncoder()?;
    encoder.setComputePipelineState(&gpu.unary_dual);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(value), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(tangent), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(out_value), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(out_tangent), 0, 3);
        encoder.setBytes_length_atIndex(NonNull::from(&op).cast(), size_of::<Analytic>(), 4);
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

/// Block until every command buffer committed so far has finished, and release
/// the allocations that were waiting on them.
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
    // Nothing is in flight any more, so allocations released while it was can go
    // back into service. A busy borrow means an allocation is being handed out
    // right now; the next sync will promote them instead.
    if let Ok(mut pool) = gpu.pool.try_borrow_mut() {
        pool.retire();
    }
    ok.then_some(())
}

/// Synchronize queued work and stop before an incomplete allocation can be read.
fn sync_or_panic(gpu: &Gpu) {
    assert!(
        sync(gpu).is_some(),
        "Metal command buffer failed; its output is invalid"
    );
}

/// Block until all GPU work submitted on this thread has completed.
///
/// Operations on [`Metal`](crate::tensors::Metal)-backed tensors are queued and
/// return before the GPU has run them, so this is the way to make outstanding
/// work observable — reading the values back already does it implicitly. It is
/// also what a benchmark needs in order to time GPU work rather than submission.
pub fn synchronize() {
    autoreleasepool(|_| {
        GPU.with(|cell| {
            if let Some(gpu) = cell.get_or_init(build_gpu).as_ref() {
                sync_or_panic(gpu);
            }
        });
    });
}

/// GPU elementwise `f32` op over two equal-length buffers. `op` is 0=add,
/// 1=sub, 2=mul, 3=div. Returns `None` if no device is available.
pub fn elementwise_f32(a: &[f32], b: &[f32], op: BinaryOp) -> Option<Vec<f32>> {
    if a.len() != b.len() || op == BinaryOp::Rem {
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
        pool.release(buf_a, false);
        pool.release(buf_b, false);
        pool.release(buf_c, false);
        Some(out)
    })
}

/// GPU `f32` broadcast operation between a buffer and a scalar. `op` uses the
/// same encoding as [`elementwise_f32`]; `scalar_left` controls operand order
/// for subtraction and division.
pub fn broadcast_f32(
    values: &[f32],
    scalar: f32,
    op: BinaryOp,
    scalar_left: bool,
) -> Option<Vec<f32>> {
    if op == BinaryOp::Rem {
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
        pool.release(input, false);
        pool.release(output, false);
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
        pool.release(source, false);
        pool.release(values, false);
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
            (BinaryOp::Add, (|x: f32, y| x + y) as fn(f32, f32) -> f32),
            (BinaryOp::Sub, |x, y| x - y),
            (BinaryOp::Mul, |x, y| x * y),
            (BinaryOp::Div, |x, y| x / y),
        ] {
            if let Some(gpu) = elementwise_f32(&a, &b, op) {
                for (i, g) in gpu.iter().enumerate() {
                    let want = f(a[i], b[i]);
                    assert!((g - want).abs() < 1e-3, "op {op:?} at {i}: {g} vs {want}");
                }
            }
        }
    }

    #[test]
    fn gpu_broadcast_matches_cpu() {
        let values: Vec<f32> = (0..1000).map(|i| i as f32 * 0.125 - 3.0).collect();
        if let Some(gpu) = broadcast_f32(&values, 2.5, BinaryOp::Mul, false) {
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
        let (values, tangents) = buf_value.unary_dual(&buf_tangent, Analytic::Tanh).unwrap();
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

        assert_eq!(size_of::<BinaryOp>(), 2);
        assert_eq!(size_of::<Analytic>(), 2);
    }

    #[test]
    fn a_recycled_allocation_is_never_clobbered_by_queued_work() {
        // Work is committed without waiting, so an allocation dropped while its
        // dispatch is still queued must not be handed straight back out: the
        // kernel would land on top of whatever the next owner put there. Ten
        // rounds, because the failure is a race the GPU can win by luck.
        let input: Vec<f32> = (0..256).map(|i| (i % 13) as f32 * 0.1).collect();
        let Some(source) = MetalBuffer::from_slice(&input) else {
            eprintln!("no Metal device; skipping the recycling check");
            return;
        };

        let known: Vec<f32> = (0..256).map(|i| i as f32).collect();
        for round in 0..10 {
            // Queue a dispatch and drop its output immediately.
            drop(source.unary(Analytic::Tanh).expect("unary dispatch"));
            // This may reuse that allocation; its contents must be what was
            // uploaded, not what the queued kernel owed its previous owner.
            let fresh = MetalBuffer::from_slice(&known).expect("upload");
            assert_eq!(fresh.to_vec(), known, "round {round}");
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
        let scaled = product.broadcast(0.5, BinaryOp::Mul, false).unwrap();
        assert_eq!(scaled.to_vec(), vec![9.5, 11.0, 21.5, 25.0]);

        let complex =
            MetalBuffer::from_slice(&[1.0, 0.0, 2.0, -1.0, 0.5, 3.0, -2.0, 0.25]).unwrap();
        let reconstructed = complex.fft().unwrap().ifft().unwrap().to_vec();
        for (actual, expected) in reconstructed.iter().zip(complex.to_vec()) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn stacking_reads_queued_device_results_without_host_staging() {
        let Some(first) = MetalBuffer::from_slice(&[1.0, 2.0, 3.0]) else {
            eprintln!("no Metal device; skipping device stacking comparison");
            return;
        };
        let second = MetalBuffer::from_slice(&[4.0, 5.0, 6.0]).unwrap();

        // Leave both inputs as pending GPU results. The stack dispatch must
        // consume those buffers directly, in command-queue order.
        let first = first.broadcast(10.0, BinaryOp::Add, false).unwrap();
        let second = second.broadcast(20.0, BinaryOp::Add, false).unwrap();

        let vertical = MetalBuffer::vstack(&[&first, &second], 3).unwrap();
        assert_eq!(vertical.to_vec(), vec![11.0, 12.0, 13.0, 24.0, 25.0, 26.0]);

        let horizontal = MetalBuffer::hstack(&[&first, &second], 3).unwrap();
        assert_eq!(
            horizontal.to_vec(),
            vec![11.0, 24.0, 12.0, 25.0, 13.0, 26.0]
        );
    }

    #[test]
    fn tiled_transpose_stays_queued_and_handles_partial_tiles() {
        const ROWS: usize = 19;
        const COLS: usize = 23;
        synchronize();

        let values = (0..ROWS * COLS)
            .map(|index| index as f32)
            .collect::<Vec<_>>();
        let Some(input) = MetalBuffer::from_slice(&values) else {
            eprintln!("no Metal device; skipping device transpose comparison");
            return;
        };
        let queued = input.broadcast(1.0, BinaryOp::Add, false).unwrap();
        let transposed = queued.transpose(ROWS, COLS).unwrap();

        let pending = GPU.with(|cell| {
            cell.get()
                .and_then(Option::as_ref)
                .map_or(0, |gpu| gpu.pending.borrow().len())
        });
        assert_eq!(pending, 2, "transpose unexpectedly synchronized GPU work");

        let expected = (0..COLS)
            .flat_map(|col| (0..ROWS).map(move |row| (row * COLS + col) as f32 + 1.0))
            .collect::<Vec<_>>();
        assert_eq!(transposed.to_vec(), expected);

        let empty = MetalBuffer::from_slice(&[]).unwrap();
        assert!(empty.transpose(0, COLS).unwrap().is_empty());
    }

    #[test]
    fn matrix_concat_and_stack_stay_on_the_device() {
        const ROWS: usize = 19;
        const LEFT_COLS: usize = 13;
        const RIGHT_COLS: usize = 7;
        synchronize();

        let left_values = (0..ROWS * LEFT_COLS)
            .map(|index| index as f32)
            .collect::<Vec<_>>();
        let right_values = (0..ROWS * RIGHT_COLS)
            .map(|index| 1_000.0 + index as f32)
            .collect::<Vec<_>>();
        let Some(left) = MetalBuffer::from_slice(&left_values) else {
            eprintln!("no Metal device; skipping matrix assembly comparison");
            return;
        };
        let right = MetalBuffer::from_slice(&right_values).unwrap();
        let left = left.broadcast(1.0, BinaryOp::Add, false).unwrap();
        let right = right.broadcast(2.0, BinaryOp::Add, false).unwrap();
        let concat = left
            .concat_matrix(&right, ROWS, LEFT_COLS, RIGHT_COLS)
            .unwrap();

        const TOP_ROWS: usize = 5;
        const BOTTOM_ROWS: usize = 7;
        const COLS: usize = 11;
        let top_values = (0..TOP_ROWS * COLS)
            .map(|index| index as f32)
            .collect::<Vec<_>>();
        let bottom_values = (0..BOTTOM_ROWS * COLS)
            .map(|index| 500.0 + index as f32)
            .collect::<Vec<_>>();
        let top = MetalBuffer::from_slice(&top_values)
            .unwrap()
            .broadcast(3.0, BinaryOp::Add, false)
            .unwrap();
        let bottom = MetalBuffer::from_slice(&bottom_values)
            .unwrap()
            .broadcast(4.0, BinaryOp::Add, false)
            .unwrap();
        let stack = top
            .stack_matrix(&bottom, TOP_ROWS, BOTTOM_ROWS, COLS)
            .unwrap();

        let pending = GPU.with(|cell| {
            cell.get()
                .and_then(Option::as_ref)
                .map_or(0, |gpu| gpu.pending.borrow().len())
        });
        assert_eq!(
            pending, 6,
            "matrix assembly unexpectedly synchronized GPU work"
        );

        let mut expected_concat = Vec::with_capacity(ROWS * (LEFT_COLS + RIGHT_COLS));
        for row in 0..ROWS {
            expected_concat.extend(
                left_values[row * LEFT_COLS..(row + 1) * LEFT_COLS]
                    .iter()
                    .map(|value| value + 1.0),
            );
            expected_concat.extend(
                right_values[row * RIGHT_COLS..(row + 1) * RIGHT_COLS]
                    .iter()
                    .map(|value| value + 2.0),
            );
        }
        assert_eq!(concat.to_vec(), expected_concat);

        let expected_stack = top_values
            .iter()
            .map(|value| value + 3.0)
            .chain(bottom_values.iter().map(|value| value + 4.0))
            .collect::<Vec<_>>();
        assert_eq!(stack.to_vec(), expected_stack);
    }

    #[test]
    fn matrix_merges_consume_queued_device_buffers() {
        const MATRICES: usize = 3;
        const ROWS: usize = 19;
        const COLS: usize = 7;
        synchronize();

        let host = (0..MATRICES)
            .map(|matrix| {
                (0..ROWS * COLS)
                    .map(|index| matrix as f32 * 1_000.0 + index as f32)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let Some(inputs) = host
            .iter()
            .map(|values| MetalBuffer::from_slice(values))
            .collect::<Option<Vec<_>>>()
        else {
            eprintln!("no Metal device; skipping matrix merge comparison");
            return;
        };
        let queued = inputs
            .iter()
            .enumerate()
            .map(|(index, input)| {
                input
                    .broadcast(index as f32 + 1.0, BinaryOp::Add, false)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let buffers = queued.iter().collect::<Vec<_>>();

        let horizontal = MetalBuffer::hmerge(&buffers, ROWS, COLS).unwrap();
        let vertical = MetalBuffer::vmerge(&buffers, ROWS, COLS).unwrap();

        let pending = GPU.with(|cell| {
            cell.get()
                .and_then(Option::as_ref)
                .map_or(0, |gpu| gpu.pending.borrow().len())
        });
        assert_eq!(
            pending, 5,
            "matrix merge unexpectedly synchronized GPU work"
        );

        let mut expected_horizontal = Vec::with_capacity(MATRICES * ROWS * COLS);
        for row in 0..ROWS {
            for (matrix, values) in host.iter().enumerate() {
                expected_horizontal.extend(
                    values[row * COLS..(row + 1) * COLS]
                        .iter()
                        .map(|value| value + matrix as f32 + 1.0),
                );
            }
        }
        assert_eq!(horizontal.to_vec(), expected_horizontal);

        let expected_vertical = host
            .iter()
            .enumerate()
            .flat_map(|(matrix, values)| {
                values.iter().map(move |value| value + matrix as f32 + 1.0)
            })
            .collect::<Vec<_>>();
        assert_eq!(vertical.to_vec(), expected_vertical);
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
            buffer = buffer.broadcast(1.0, BinaryOp::Add, false).unwrap();
            buffer = buffer.elementwise(&ones, BinaryOp::Add).unwrap();
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
        let doubled = buffer.broadcast(2.0, BinaryOp::Mul, false).unwrap();
        synchronize();
        assert_eq!(doubled.to_vec(), vec![6.0f32; 32]);
    }
}
