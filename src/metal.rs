//! GPU-accelerated tensor kernels via Apple Metal.
//!
//! Compiled only with the `metal` feature on macOS. It runs matrix and vector
//! products, elementwise, broadcast, reduction, sorting, convolution, statistics
//! and analytic operations, and fused elementwise programs on the GPU, for every
//! [`MetalElement`] — `f32`, `f16` and `bf16` — with each kernel compiled once
//! per type. Arithmetic runs in the element type; sums, products and moments
//! accumulate in `f32` and round once. M5 GPUs additionally use Metal 4
//! TensorOps for matrix products. Radix-2 FFTs run in `f32`; `f64`, matrix
//! inversion, and non-radix-2 FFTs stay on the CPU.
//!
//! Every entry point returns `Option`: if no Metal device is available or an
//! operation cannot be encoded, the caller falls back to the CPU kernel. A
//! failure reported asynchronously by Metal is surfaced as a panic at the next
//! synchronization point rather than exposing an incomplete output buffer.
//!
//! Input/output buffers are recycled through a small per-thread pool so
//! repeated calls avoid re-allocating GPU memory.
//!
//! The `Host` backend never comes here: its tensors stay on the CPU whatever
//! their size. To run on the GPU, put the tensors on the
//! [`Metal`](crate::tensors::Metal) backend, which stores their elements in
//! `MTLStorageModeShared` memory and passes the allocations from kernel to
//! kernel. [`MetalBuffer`] is that storage, usable directly when the
//! `Vector`/`Matrix` types do not fit.

use std::cell::{Cell, OnceCell, RefCell};
use std::marker::PhantomData;
use std::mem::{ManuallyDrop, size_of};
use std::ptr::NonNull;

use dispatch2::DispatchData;
use half::{bf16, f16};
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily, MTLLibrary, MTLResourceOptions, MTLSize,
};

use crate::numbers::Real;
use crate::tensors::{Analytic, Axis, BinaryOp, Compare, Family, Reduce, SortOrder, Statistic};

/// Threadgroup tile edge; must match `TILE` in the shader. 16×16 = 256 threads.
const TILE: usize = 16;

/// M5 TensorOps threadgroup tile. Four SIMD groups form a 2×2 arrangement of
/// 32×32 SIMD-group tiles, matching Apple's recommended 16-bit starting point.
const TENSOROPS_TILE_ROWS: usize = 64;
const TENSOROPS_TILE_COLS: usize = 64;

/// Threads per group in the tree reduction; must match `REDUCE_GROUP` in the
/// shader, which sizes its threadgroup scratch array with it.
const REDUCE_GROUP: usize = 256;

/// The compute kernels, compiled by `build.rs` and embedded in the crate.
///
/// `matmul_tiled` stages `TILE×TILE` blocks of A and B into threadgroup memory
/// so each loaded value is reused `TILE` times, which is far more
/// bandwidth-efficient than reading straight from device memory.
const KERNEL_LIBRARY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tensorcrate.metallib"));
const TENSOROPS_LIBRARY: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/tensorcrate_tensorops.metallib"));

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for half::f16 {}
    impl Sealed for half::bf16 {}
}

/// An element type the Metal kernels are compiled for: `f32`, `f16` or `bf16`.
///
/// Every typed shader is instantiated once per element type and the host picks
/// the instance by this trait, so a `MetalBuffer<f16>` runs `half` kernels and
/// never passes through `f32` storage. Accumulations — reductions, matrix
/// products, convolutions, moments — are the exception by design: they fold in
/// `f32` and round to the element type once, as the host's do.
pub trait MetalElement: Real + sealed::Sealed {
    /// The kernel-name suffix of this type's shader instances.
    #[doc(hidden)]
    const SUFFIX: &'static str;

    /// Position in [`Gpu::typed`].
    #[doc(hidden)]
    const INDEX: usize;

    /// The extreme of the IEEE total order at the end a sort trims its padding
    /// from: the largest key for an ascending sort, the smallest for a
    /// descending one. Both are NaNs, so no input NaN can sort past them.
    #[doc(hidden)]
    fn sort_padding(ascending: bool) -> Self;
}

impl MetalElement for f32 {
    const SUFFIX: &'static str = "f32";
    const INDEX: usize = 0;

    fn sort_padding(ascending: bool) -> Self {
        f32::from_bits(if ascending { 0x7FFF_FFFF } else { 0xFFFF_FFFF })
    }
}

impl MetalElement for f16 {
    const SUFFIX: &'static str = "f16";
    const INDEX: usize = 1;

    fn sort_padding(ascending: bool) -> Self {
        f16::from_bits(if ascending { 0x7FFF } else { 0xFFFF })
    }
}

impl MetalElement for bf16 {
    const SUFFIX: &'static str = "bf16";
    const INDEX: usize = 2;

    fn sort_padding(ascending: bool) -> Self {
        bf16::from_bits(if ascending { 0x7FFF } else { 0xFFFF })
    }
}

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

/// Device, queue, pipelines, and buffer pool — cached per thread. Metal objects
/// are not `Send`, so a thread-local keeps everything on one thread.
struct Gpu {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// One set of pipelines per [`MetalElement`], indexed by
    /// [`MetalElement::INDEX`].
    typed: [Typed; 3],
    /// The `f16`/`bf16` × `f32`-output TensorOps products behind `matmul_f32`.
    tensorops: Option<TensorOpsPipelines>,
    /// The scan runs over a `float` buffer whatever the tensor's type.
    scan: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    fft_bit_reverse: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    fft_stage: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pool: RefCell<Pool>,
    /// Command buffers committed but not yet waited on — see [`commit`].
    pending: RefCell<Vec<Retained<ProtocolObject<dyn MTLCommandBuffer>>>>,
}

type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// The pipelines for one element type: each is the `_f32`, `_f16` or `_bf16`
/// instance of the shader of the same name.
struct Typed {
    matmul: Pipeline,
    /// The TensorOps product for this type, on M5-class GPUs.
    tensorops: Option<Pipeline>,
    elementwise: Pipeline,
    broadcast: Pipeline,
    compare: Pipeline,
    compare_scalar: Pipeline,
    clamp: Pipeline,
    reduce: Pipeline,
    widen: Pipeline,
    narrow: Pipeline,
    sort_prepare: Pipeline,
    bitonic: Pipeline,
    stack_vector: Pipeline,
    concat_horizontal: Pipeline,
    merge_horizontal: Pipeline,
    transpose: Pipeline,
    correlate: Pipeline,
    pad_zeros: Pipeline,
    flip_both: Pipeline,
    unary: Pipeline,
    unary_dual: Pipeline,
    power: Pipeline,
    power_scalar: Pipeline,
    deviation: Pipeline,
    axis_moments: Pipeline,
    distribution: Pipeline,
    axis_distribution: Pipeline,
    fused: Pipeline,
    /// A product whose elements feed a fused program before any store.
    matmul_epilogue: Pipeline,
    /// The same on TensorOps, on M5-class GPUs.
    tensorops_epilogue: Option<Pipeline>,
}

impl Gpu {
    /// The pipelines compiled for element type `T`.
    fn kernels<T: MetalElement>(&self) -> &Typed {
        &self.typed[T::INDEX]
    }
}

struct TensorOpsPipelines {
    f16_f32: Pipeline,
    bf16_f32: Pipeline,
}

thread_local! {
    static GPU: OnceCell<Option<Gpu>> = const { OnceCell::new() };
    static TENSOROPS: Cell<bool> = const { Cell::new(true) };
}

/// Whether matrix products on this thread may use TensorOps where the GPU has
/// them (M5-class), or must use the tiled kernel. On by default; turning it
/// off is for testing the tiled kernels on hardware that would never reach
/// them, and for comparing the two.
#[doc(hidden)]
pub fn set_tensorops(enabled: bool) {
    TENSOROPS.with(|cell| cell.set(enabled));
}

fn tensorops_enabled() -> bool {
    TENSOROPS.with(Cell::get)
}

fn build_gpu() -> Option<Gpu> {
    let device = MTLCreateSystemDefaultDevice()?;
    let queue = device.newCommandQueue()?;
    let library_data = DispatchData::from_static_bytes(KERNEL_LIBRARY);
    let library = device
        .newLibraryWithData_error(&library_data)
        .map_err(|_error| {
            #[cfg(test)]
            eprintln!("Loading the compiled Metal library failed: {_error}");
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
    let tensorops_library = (|| {
        // Apple10 is the M5/A19 GPU family. Earlier families retain the tiled
        // kernel even though they can load Metal 4 code.
        if !device.supportsFamily(MTLGPUFamily::Apple10) {
            return None;
        }
        let data = DispatchData::from_static_bytes(TENSOROPS_LIBRARY);
        device.newLibraryWithData_error(&data).ok()
    })();
    let tensorops_pipeline = |name: &str| {
        let library = tensorops_library.as_ref()?;
        let function = library.newFunctionWithName(&NSString::from_str(name))?;
        device
            .newComputePipelineStateWithFunction_error(&function)
            .ok()
    };
    let tensorops = (|| {
        Some(TensorOpsPipelines {
            f16_f32: tensorops_pipeline("matmul_tensorops_f16_f32")?,
            bf16_f32: tensorops_pipeline("matmul_tensorops_bf16_f32")?,
        })
    })();
    let typed = |suffix: &str| -> Option<Typed> {
        let kernel = |name: &str| pipeline(&format!("{name}_{suffix}"));
        Some(Typed {
            matmul: kernel("matmul_tiled")?,
            tensorops: tensorops_pipeline(&format!("matmul_tensorops_{suffix}")),
            elementwise: kernel("elementwise")?,
            broadcast: kernel("broadcast")?,
            compare: kernel("compare")?,
            compare_scalar: kernel("compare_scalar")?,
            clamp: kernel("clamp_values")?,
            reduce: kernel("reduce_partial")?,
            widen: kernel("widen")?,
            narrow: kernel("narrow")?,
            sort_prepare: kernel("sort_prepare")?,
            bitonic: kernel("bitonic_stage")?,
            stack_vector: kernel("stack_vector")?,
            concat_horizontal: kernel("concat_horizontal")?,
            merge_horizontal: kernel("merge_horizontal")?,
            transpose: kernel("transpose_tiled")?,
            correlate: kernel("correlate")?,
            pad_zeros: kernel("pad_zeros")?,
            flip_both: kernel("flip_both")?,
            unary: kernel("unary")?,
            unary_dual: kernel("unary_dual")?,
            power: kernel("power")?,
            power_scalar: kernel("power_scalar")?,
            deviation: kernel("deviation_partial")?,
            axis_moments: kernel("axis_moments")?,
            distribution: kernel("distribution")?,
            axis_distribution: kernel("axis_distribution")?,
            fused: kernel("fused_elementwise")?,
            matmul_epilogue: kernel("matmul_epilogue")?,
            tensorops_epilogue: tensorops_pipeline(&format!("matmul_tensorops_epilogue_{suffix}")),
        })
    };
    Some(Gpu {
        typed: [
            typed(f32::SUFFIX)?,
            typed(f16::SUFFIX)?,
            typed(bf16::SUFFIX)?,
        ],
        tensorops,
        scan: pipeline("scan_step")?,
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
fn upload<T: Copy>(buffer: &ProtocolObject<dyn MTLBuffer>, src: &[T]) {
    let dst = buffer.contents().as_ptr() as *mut T;
    unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len()) };
}

/// Read `len` floats out of the front of a shared buffer's storage.
fn download(buffer: &ProtocolObject<dyn MTLBuffer>, len: usize) -> Vec<f32> {
    let src = buffer.contents().as_ptr() as *const f32;
    let mut out = vec![0.0f32; len];
    unsafe { std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), len) };
    out
}

/// A typed allocation in Apple-silicon shared memory.
///
/// Keeping intermediate values in this type avoids the upload/download copies
/// made by the convenience functions below. The CPU may read the allocation in
/// place with [`as_slice`](Self::as_slice) — shared storage is CPU-cached
/// memory, not a separate GPU pool — while matrix, elementwise, broadcast, and
/// FFT operations chain without ever leaving it.
///
/// This is the storage behind the [`Metal`](crate::tensors::Metal) tensor
/// backend, which pairs it with the extents to make a `Vector` or `Matrix`.
///
/// Metal objects are thread-affine, so this type intentionally is not `Send`.
/// Dropping one returns its allocation to the thread's buffer pool.
pub struct MetalBuffer<T = f32> {
    /// Returned to the pool by `Drop`, hence `ManuallyDrop`.
    raw: ManuallyDrop<Retained<ProtocolObject<dyn MTLBuffer>>>,
    len: usize,
    marker: PhantomData<T>,
}

impl<T: Copy + 'static> MetalBuffer<T> {
    /// Allocate `len` values of shared storage, recycling a pooled allocation
    /// when one is big enough. The contents are unspecified, so every caller
    /// either uploads into it or has a kernel write every element.
    pub(crate) fn allocate(len: usize) -> Option<Self> {
        let bytes = len.checked_mul(size_of::<T>())?.max(1);
        with_gpu(|gpu| {
            let raw = gpu.pool.borrow_mut().acquire(&gpu.device, bytes)?;
            Some(Self {
                raw: ManuallyDrop::new(raw),
                len,
                marker: PhantomData,
            })
        })
    }

    /// Allocate shared storage and initialize it from a CPU slice.
    pub fn from_slice(values: &[T]) -> Option<Self> {
        let buffer = Self::allocate(values.len())?;
        upload(&buffer.raw, values);
        Some(buffer)
    }

    /// Number of stored values.
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
    pub fn as_slice(&self) -> &[T] {
        with_gpu(|gpu| {
            sync_or_panic(gpu);
            Some(())
        })
        .expect("a Metal buffer cannot outlive its thread-local device");
        // SAFETY: `MTLStorageModeShared` memory is CPU-readable at
        // `contents()`, and holds `len` initialized values — a buffer is only
        // handed out after an upload or a kernel that writes every element. The
        // `sync` above drained every command buffer that could still be writing
        // it, and nothing on this thread can submit more while the borrow is
        // alive, so no GPU write is in flight.
        unsafe { std::slice::from_raw_parts(self.raw.contents().as_ptr().cast::<T>(), self.len) }
    }

    /// Copy the shared allocation into an ordinary CPU vector.
    pub fn to_vec(&self) -> Vec<T> {
        self.as_slice().to_vec()
    }

    /// The Metal allocation, for binding to a kernel.
    pub(crate) fn raw(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.raw
    }

    /// Borrow the shared allocation mutably, after waiting for every queued
    /// command buffer — any of which may still be reading or writing it.
    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        with_gpu(|gpu| {
            sync_or_panic(gpu);
            Some(())
        })
        .expect("a Metal buffer cannot outlive its thread-local device");
        // SAFETY: as for `as_slice`, and the exclusive borrow of `self` rules out
        // any other view of this allocation for as long as the slice lives.
        unsafe {
            std::slice::from_raw_parts_mut(self.raw.contents().as_ptr().cast::<T>(), self.len)
        }
    }
}

impl<T: MetalElement> MetalBuffer<T> {
    /// Matrix multiplication, with both inputs and the result remaining in
    /// shared Metal buffers. The products accumulate in `f32` and each output
    /// element rounds to `T` once.
    pub fn matmul(&self, rhs: &Self, m: usize, k: usize, n: usize) -> Option<Self> {
        if self.len != m.checked_mul(k)? || rhs.len != k.checked_mul(n)? {
            return None;
        }
        let output_len = m.checked_mul(n)?;
        if output_len == 0 {
            return Self::from_slice(&[]);
        }
        if k == 0 {
            return Self::from_slice(&vec![T::zero(); output_len]);
        }
        let output = Self::allocate(output_len)?;
        with_gpu(|gpu| {
            encode_matmul::<T>(gpu, &self.raw, &rhs.raw, &output.raw, m, k, n, false)
        })?;
        Some(output)
    }

    /// Transpose a row-major `rows × cols` matrix into a new shared buffer.
    pub(crate) fn transpose(&self, rows: usize, cols: usize) -> Option<Self> {
        if self.len != rows.checked_mul(cols)? {
            return None;
        }
        let output = Self::allocate(self.len)?;
        if self.len != 0 {
            with_gpu(|gpu| encode_transpose::<T>(gpu, &self.raw, &output.raw, rows, cols))?;
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
        with_gpu(|gpu| encode_matmul::<T>(gpu, &self.raw, &rhs.raw, &target.raw, m, k, n, true))
    }

    /// Apply an analytic function elementwise.
    pub fn unary(&self, op: Analytic) -> Option<Self> {
        let output = Self::allocate(self.len)?;
        if self.len != 0 {
            with_gpu(|gpu| encode_unary::<T>(gpu, &self.raw, &output.raw, self.len, op))?;
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
                encode_unary_dual::<T>(
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
                encode_elementwise::<T>(gpu, &self.raw, &rhs.raw, &output.raw, self.len, op)
            })?;
        }
        Some(output)
    }

    /// Elementwise `self^rhs`.
    pub fn power(&self, rhs: &Self) -> Option<Self> {
        if self.len != rhs.len {
            return None;
        }
        let output = Self::allocate(self.len)?;
        if self.len != 0 {
            with_gpu(|gpu| encode_power::<T>(gpu, &self.raw, &rhs.raw, &output.raw, self.len))?;
        }
        Some(output)
    }

    /// Elementwise power with one operand fixed. `scalar_left` selects
    /// `scalar^x` over `x^scalar`.
    ///
    /// Scalars cross to the shader as `f32`, which holds every `T` exactly.
    pub fn power_scalar(&self, scalar: T, scalar_left: bool) -> Option<Self> {
        let scalar = scalar.into_f64() as f32;
        let output = Self::allocate(self.len)?;
        if self.len != 0 {
            with_gpu(|gpu| {
                encode_power_scalar::<T>(gpu, &self.raw, &output.raw, self.len, scalar, scalar_left)
            })?;
        }
        Some(output)
    }

    /// Valid cross-correlation of an `rows × cols` input with a
    /// `window_rows × window_cols` kernel, both already resident.
    ///
    /// `flip` reverses the window, giving convolution rather than correlation.
    pub fn correlate(
        &self,
        weights: &Self,
        rows: usize,
        cols: usize,
        window_rows: usize,
        window_cols: usize,
        flip: bool,
    ) -> Option<Self> {
        if self.len != rows.checked_mul(cols)?
            || weights.len != window_rows.checked_mul(window_cols)?
        {
            return None;
        }
        let out_rows = rows.checked_sub(window_rows)? + 1;
        let out_cols = cols.checked_sub(window_cols)? + 1;
        let output = Self::allocate(out_rows.checked_mul(out_cols)?)?;
        if output.len != 0 {
            with_gpu(|gpu| {
                encode_correlate::<T>(
                    gpu,
                    &self.raw,
                    &weights.raw,
                    &output.raw,
                    rows,
                    cols,
                    window_rows,
                    window_cols,
                    flip,
                )
            })?;
        }
        Some(output)
    }

    /// Surround an `rows × cols` matrix with `pad_rows`/`pad_cols` zeros.
    pub fn pad(&self, rows: usize, cols: usize, pad_rows: usize, pad_cols: usize) -> Option<Self> {
        if self.len != rows.checked_mul(cols)? {
            return None;
        }
        let output = Self::allocate((rows + 2 * pad_rows).checked_mul(cols + 2 * pad_cols)?)?;
        if output.len != 0 {
            with_gpu(|gpu| {
                encode_pad::<T>(gpu, &self.raw, &output.raw, rows, cols, pad_rows, pad_cols)
            })?;
        }
        Some(output)
    }

    /// Reverse both axes of an `rows × cols` matrix.
    pub fn flip(&self, rows: usize, cols: usize) -> Option<Self> {
        if self.len != rows.checked_mul(cols)? {
            return None;
        }
        let output = Self::allocate(self.len)?;
        if self.len != 0 {
            with_gpu(|gpu| encode_flip::<T>(gpu, &self.raw, &output.raw, rows, cols))?;
        }
        Some(output)
    }

    /// Elementwise comparison with another shared buffer.
    pub fn compare(&self, rhs: &Self, op: Compare) -> Option<Self> {
        if self.len != rhs.len {
            return None;
        }
        let output = Self::allocate(self.len)?;
        if self.len != 0 {
            with_gpu(|gpu| encode_compare::<T>(gpu, &self.raw, &rhs.raw, &output.raw, self.len, op))?;
        }
        Some(output)
    }

    /// Elementwise comparison against a scalar; `scalar_left` puts the scalar on
    /// the left, which matters for [`Compare::MaxShare`].
    pub fn compare_scalar(&self, scalar: T, op: Compare, scalar_left: bool) -> Option<Self> {
        let scalar = scalar.into_f64() as f32;
        let output = Self::allocate(self.len)?;
        if self.len != 0 {
            with_gpu(|gpu| {
                encode_compare_scalar::<T>(
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

    /// Confine every element to `[low, high]`, in one dispatch.
    pub fn clamp(&self, low: T, high: T) -> Option<Self> {
        let (low, high) = (low.into_f64() as f32, high.into_f64() as f32);
        let output = Self::allocate(self.len)?;
        if self.len != 0 {
            with_gpu(|gpu| encode_clamp::<T>(gpu, &self.raw, &output.raw, self.len, low, high))?;
        }
        Some(output)
    }

    /// Fold the whole buffer to one value with a tree reduction.
    ///
    /// Each round folds `REDUCE_GROUP` values per threadgroup, so the length
    /// falls by that factor per dispatch — three rounds for a million elements.
    /// The answer has to come back to the CPU, so this is one of the few
    /// operations that ends in a synchronization rather than leaving work
    /// queued.
    ///
    /// The fold runs in `f32` whatever `T` is, and the `f32` total is what
    /// comes back: rounding it to `T` is the caller's one rounding, and a mean
    /// or a variance built from it should use the unrounded value.
    pub fn reduce(&self, op: Reduce) -> Option<f32> {
        if self.len == 0 {
            return Some(op.identity());
        }
        if self.len == 1 {
            return Some(self.as_slice()[0].into_f64() as f32);
        }

        // Ping-pong between two `f32` scratch buffers: a round reads one and
        // writes the (much shorter) other. Only the first round reads `T`.
        let mut groups = self.len.div_ceil(REDUCE_GROUP);
        let mut front = MetalBuffer::<f32>::allocate(groups)?;
        with_gpu(|gpu| encode_reduce::<T>(gpu, &self.raw, &front.raw, self.len, groups, op))?;
        let mut count = groups;
        if count == 1 {
            return Some(front.as_slice()[0]);
        }

        let mut back = MetalBuffer::<f32>::allocate(count.div_ceil(REDUCE_GROUP))?;
        while count > 1 {
            groups = count.div_ceil(REDUCE_GROUP);
            with_gpu(|gpu| encode_reduce::<f32>(gpu, &front.raw, &back.raw, count, groups, op))?;
            std::mem::swap(&mut front, &mut back);
            count = groups;
        }
        Some(front.as_slice()[0])
    }

    /// Inclusive prefix sum, `log2(len)` dispatches deep.
    ///
    /// Every sweep reads one buffer and writes the other, so the two allocations
    /// alternate and the result is whichever one the last sweep wrote. The
    /// additions land in a different order from the CPU's running total, which
    /// is a rounding difference rather than a disagreement.
    ///
    /// The running totals are `f32`: a 16-bit buffer is widened into an `f32`
    /// one, scanned there, and each total rounds to `T` once on the way back.
    pub fn prefix_sum(&self) -> Option<Self> {
        if self.len <= 1 {
            return Self::from_slice(self.as_slice());
        }
        let mut front = MetalBuffer::<f32>::allocate(self.len)?;
        with_gpu(|gpu| encode_convert(gpu, &gpu.kernels::<T>().widen, &self.raw, &front.raw, self.len))?;
        let mut back = MetalBuffer::<f32>::allocate(self.len)?;
        let mut offset = 1;
        while offset < self.len {
            with_gpu(|gpu| encode_scan(gpu, &front.raw, &back.raw, self.len, offset))?;
            std::mem::swap(&mut front, &mut back);
            offset *= 2;
        }
        let output = Self::allocate(self.len)?;
        with_gpu(|gpu| encode_convert(gpu, &gpu.kernels::<T>().narrow, &front.raw, &output.raw, self.len))?;
        Some(output)
    }

    /// Sort the elements in IEEE total order, on the GPU.
    ///
    /// A bitonic sort: `log²` stages of compare-exchange over a power-of-two
    /// buffer, each stage one dispatch. The input is padded up to that length
    /// with a value that sorts to the far end, so trimming the tail afterwards
    /// leaves exactly the input's elements. The shader compares monotone
    /// integer keys rather than the floats themselves, which is what makes the
    /// result identical to a CPU total-order sort (`f32::total_cmp`, or the
    /// 16-bit types' own) rather than merely similar: NaNs and `−0.0` land where
    /// the total order puts them instead of wherever an unordered
    /// compare-exchange left them.
    pub fn sort(&self, order: SortOrder) -> Option<Self> {
        if self.len <= 1 {
            return Self::from_slice(self.as_slice());
        }
        let padded = self.len.checked_next_power_of_two()?;
        let ascending = order == SortOrder::Ascending;
        let padding = T::sort_padding(ascending);

        let buffer = Self::allocate(padded)?;
        with_gpu(|gpu| {
            encode_sort_prepare::<T>(gpu, &self.raw, &buffer.raw, padded, self.len, padding)?;
            let mut block = 2;
            while block <= padded {
                let mut stride = block / 2;
                while stride > 0 {
                    encode_bitonic_stage::<T>(gpu, &buffer.raw, padded, block, stride, ascending)?;
                    stride /= 2;
                }
                block *= 2;
            }
            Some(())
        })?;

        if padded == self.len {
            return Some(buffer);
        }
        // Trim the padding. The values are in shared memory, so this reads the
        // sorted prefix in place rather than downloading it.
        Self::from_slice(&buffer.as_slice()[..self.len])
    }

    /// `Σ(xᵢ − mean)²` over the whole buffer.
    ///
    /// The first round is its own kernel, which forms and squares each
    /// deviation as it reads the value; every round after it is the ordinary
    /// summing reduction over the partials. So the buffer is read once, not
    /// once to write an elementwise result and again to fold it.
    ///
    /// Like [`reduce`](Self::reduce), this accumulates in `f32` and returns the
    /// unrounded `f32` sum; `mean` is the unrounded `f32` mean.
    pub fn sum_squared_deviations(&self, mean: f32) -> Option<f32> {
        if self.len == 0 {
            return Some(0.0);
        }
        if self.len == 1 {
            let deviation = self.as_slice()[0].into_f64() as f32 - mean;
            return Some(deviation * deviation);
        }

        let mut groups = self.len.div_ceil(REDUCE_GROUP);
        let mut front = MetalBuffer::<f32>::allocate(groups)?;
        with_gpu(|gpu| encode_deviation::<T>(gpu, &self.raw, &front.raw, self.len, groups, mean))?;
        let mut count = groups;
        if count == 1 {
            return Some(front.as_slice()[0]);
        }

        let mut back = MetalBuffer::<f32>::allocate(count.div_ceil(REDUCE_GROUP))?;
        while count > 1 {
            groups = count.div_ceil(REDUCE_GROUP);
            with_gpu(|gpu| {
                encode_reduce::<f32>(gpu, &front.raw, &back.raw, count, groups, Reduce::Sum)
            })?;
            std::mem::swap(&mut front, &mut back);
            count = groups;
        }
        Some(front.as_slice()[0])
    }

    /// One mean and one `Σ(xᵢ − mean)²` per row or per column of a
    /// `rows × cols` matrix, as `(means, deviations)`.
    ///
    /// One dispatch with a thread per result, rather than one whole-buffer
    /// reduction per slice: a `1024 × 1024` matrix reduced by rows is a
    /// thousand folds of a thousand values each, and a thousand separate
    /// dispatches would cost more in command buffers than in arithmetic.
    pub fn axis_moments(&self, rows: usize, cols: usize, axis: Axis) -> Option<(Self, Self)> {
        if self.len != rows.checked_mul(cols)? {
            return None;
        }
        let extent = axis.extent((rows, cols));
        let means = Self::allocate(extent)?;
        let deviations = Self::allocate(extent)?;
        if extent != 0 && axis.depth((rows, cols)) != 0 {
            with_gpu(|gpu| {
                encode_axis_moments::<T>(
                    gpu,
                    &self.raw,
                    &means.raw,
                    &deviations.raw,
                    rows,
                    cols,
                    extent,
                    axis,
                )
            })?;
        }
        Some((means, deviations))
    }

    /// A distribution function applied elementwise, with one parameter pair for
    /// the whole buffer.
    pub fn distribution(
        &self,
        family: Family,
        statistic: Statistic,
        parameters: (f32, f32),
    ) -> Option<Self> {
        let output = Self::allocate(self.len)?;
        if self.len != 0 {
            with_gpu(|gpu| {
                encode_distribution::<T>(
                    gpu,
                    &self.raw,
                    &output.raw,
                    self.len,
                    family,
                    statistic,
                    parameters,
                )
            })?;
        }
        Some(output)
    }

    /// The same, with a parameter pair per row or per column.
    ///
    /// `first` and `second` hold one parameter each per slice along `axis`, in
    /// the order [`axis_moments`](Self::axis_moments) produces them.
    #[allow(clippy::too_many_arguments)]
    pub fn axis_distribution(
        &self,
        first: &Self,
        second: &Self,
        rows: usize,
        cols: usize,
        axis: Axis,
        family: Family,
        statistic: Statistic,
    ) -> Option<Self> {
        let extent = axis.extent((rows, cols));
        if self.len != rows.checked_mul(cols)? || first.len != extent || second.len != extent {
            return None;
        }
        let output = Self::allocate(self.len)?;
        if self.len != 0 {
            with_gpu(|gpu| {
                encode_axis_distribution::<T>(
                    gpu,
                    &self.raw,
                    &output.raw,
                    &first.raw,
                    &second.raw,
                    self.len,
                    cols,
                    axis,
                    family,
                    statistic,
                )
            })?;
        }
        Some(output)
    }

    /// Broadcast operation with a scalar. `op` has the same encoding as
    /// [`elementwise`](Self::elementwise).
    pub fn broadcast(&self, scalar: T, op: BinaryOp, scalar_left: bool) -> Option<Self> {
        if op == BinaryOp::Rem {
            return None;
        }
        let scalar = scalar.into_f64() as f32;
        let output = Self::allocate(self.len)?;
        if self.len != 0 {
            with_gpu(|gpu| {
                encode_broadcast::<T>(
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
                encode_stack::<T>(gpu, inputs, &output.raw, vector_len, output_stride, offset)
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
                encode_concat::<T>(
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
                encode_matrix_stack::<T>(gpu, &self.raw, &rhs.raw, &output.raw, self.len, rhs.len)
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
            with_gpu(|gpu| encode_hmerge::<T>(gpu, inputs, &output.raw, rows, cols))?;
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
            with_gpu(|gpu| encode_vmerge::<T>(gpu, inputs, &output.raw, matrix_len))?;
        }
        Some(output)
    }

}

impl MetalBuffer<f32> {
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

impl MetalBuffer<f16> {
    /// Multiply FP16 inputs with TensorOps and accumulate into FP32 output.
    pub(crate) fn matmul_f32(
        &self,
        rhs: &Self,
        m: usize,
        k: usize,
        n: usize,
    ) -> Option<MetalBuffer<f32>> {
        matmul_tensorops(self, rhs, m, k, n, 0.0, |pipelines| &pipelines.f16_f32)
    }
}

impl MetalBuffer<bf16> {
    /// Multiply BF16 inputs with TensorOps and accumulate into FP32 output.
    pub(crate) fn matmul_f32(
        &self,
        rhs: &Self,
        m: usize,
        k: usize,
        n: usize,
    ) -> Option<MetalBuffer<f32>> {
        matmul_tensorops(self, rhs, m, k, n, 0.0, |pipelines| &pipelines.bf16_f32)
    }
}

fn matmul_tensorops<T: Copy + 'static, U: Copy + 'static>(
    left: &MetalBuffer<T>,
    right: &MetalBuffer<T>,
    m: usize,
    k: usize,
    n: usize,
    zero: U,
    pipeline: impl Fn(&TensorOpsPipelines) -> &ProtocolObject<dyn MTLComputePipelineState>,
) -> Option<MetalBuffer<U>> {
    if left.len != m.checked_mul(k)? || right.len != k.checked_mul(n)? {
        return None;
    }
    let output_len = m.checked_mul(n)?;
    if output_len == 0 {
        return MetalBuffer::from_slice(&[]);
    }
    if k == 0 {
        return MetalBuffer::from_slice(&vec![zero; output_len]);
    }
    let output = MetalBuffer::<U>::allocate(output_len)?;
    with_gpu(|gpu| {
        let state = pipeline(gpu.tensorops.as_ref()?);
        encode_tensorops_matmul(
            gpu,
            state,
            &left.raw,
            &right.raw,
            &output.raw,
            m,
            k,
            n,
            false,
            size_of::<U>(),
        )
    })?;
    Some(output)
}

impl<T> Drop for MetalBuffer<T> {
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
fn encode_matmul<T: MetalElement>(
    gpu: &Gpu,
    a: &ProtocolObject<dyn MTLBuffer>,
    b: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    m: usize,
    k: usize,
    n: usize,
    accumulate: bool,
) -> Option<()> {
    if !accumulate
        && tensorops_enabled()
        && let Some(pipeline) = &gpu.kernels::<T>().tensorops
    {
        return encode_tensorops_matmul(
            gpu,
            pipeline,
            a,
            b,
            output,
            m,
            k,
            n,
            accumulate,
            size_of::<T>(),
        );
    }
    let command = gpu.queue.commandBuffer()?;
    let encoder = command.computeCommandEncoder()?;
    encoder.setComputePipelineState(&gpu.kernels::<T>().matmul);
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

#[allow(clippy::too_many_arguments)]
fn encode_tensorops_matmul(
    gpu: &Gpu,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    a: &ProtocolObject<dyn MTLBuffer>,
    b: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    m: usize,
    k: usize,
    n: usize,
    accumulate: bool,
    output_element_size: usize,
) -> Option<()> {
    let command = gpu.queue.commandBuffer()?;
    if !accumulate {
        let clear = command.blitCommandEncoder()?;
        clear.fillBuffer_range_value(
            output,
            NSRange::new(0, m.checked_mul(n)?.checked_mul(output_element_size)?),
            0,
        );
        clear.endEncoding();
    }

    let encoder = command.computeCommandEncoder()?;
    encoder.setComputePipelineState(pipeline);
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
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n.div_ceil(TENSOROPS_TILE_COLS),
            height: m.div_ceil(TENSOROPS_TILE_ROWS),
            depth: 1,
        },
        MTLSize {
            width: pipeline.threadExecutionWidth() * 4,
            height: 1,
            depth: 1,
        },
    );
    encoder.endEncoding();
    commit(gpu, command)
}

fn encode_transpose<T: MetalElement>(
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
    encoder.setComputePipelineState(&gpu.kernels::<T>().transpose);
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

fn encode_elementwise<T: MetalElement>(
    gpu: &Gpu,
    a: &ProtocolObject<dyn MTLBuffer>,
    b: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
    op: BinaryOp,
) -> Option<()> {
    let command = gpu.queue.commandBuffer()?;
    let encoder = command.computeCommandEncoder()?;
    encoder.setComputePipelineState(&gpu.kernels::<T>().elementwise);
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

fn encode_compare<T: MetalElement>(
    gpu: &Gpu,
    a: &ProtocolObject<dyn MTLBuffer>,
    b: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
    op: Compare,
) -> Option<()> {
    let command = gpu.queue.commandBuffer()?;
    let encoder = command.computeCommandEncoder()?;
    encoder.setComputePipelineState(&gpu.kernels::<T>().compare);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(a), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(b), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 2);
        encoder.setBytes_length_atIndex(NonNull::from(&op).cast(), size_of::<Compare>(), 3);
    }
    dispatch_1d(&encoder, len);
    encoder.endEncoding();
    commit(gpu, command)
}

fn encode_compare_scalar<T: MetalElement>(
    gpu: &Gpu,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
    scalar: f32,
    op: Compare,
    scalar_left: bool,
) -> Option<()> {
    let command = gpu.queue.commandBuffer()?;
    let encoder = command.computeCommandEncoder()?;
    encoder.setComputePipelineState(&gpu.kernels::<T>().compare_scalar);
    let scalar_left = u32::from(scalar_left);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
        encoder.setBytes_length_atIndex(NonNull::from(&scalar).cast(), 4, 2);
        encoder.setBytes_length_atIndex(NonNull::from(&op).cast(), size_of::<Compare>(), 3);
        encoder.setBytes_length_atIndex(NonNull::from(&scalar_left).cast(), 4, 4);
    }
    dispatch_1d(&encoder, len);
    encoder.endEncoding();
    commit(gpu, command)
}

fn encode_clamp<T: MetalElement>(
    gpu: &Gpu,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
    low: f32,
    high: f32,
) -> Option<()> {
    let command = gpu.queue.commandBuffer()?;
    let encoder = command.computeCommandEncoder()?;
    encoder.setComputePipelineState(&gpu.kernels::<T>().clamp);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
        encoder.setBytes_length_atIndex(NonNull::from(&low).cast(), 4, 2);
        encoder.setBytes_length_atIndex(NonNull::from(&high).cast(), 4, 3);
    }
    dispatch_1d(&encoder, len);
    encoder.endEncoding();
    commit(gpu, command)
}

/// One round of the tree reduction, over `groups` whole threadgroups.
///
/// Dispatched by threadgroup rather than by thread: the kernel's barriers
/// require every thread of a group to arrive, which a ragged final group under
/// `dispatchThreads` would not do. The threads past `count` read the identity
/// instead.
fn encode_reduce<T: MetalElement>(
    gpu: &Gpu,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    count: usize,
    groups: usize,
    op: Reduce,
) -> Option<()> {
    let command = gpu.queue.commandBuffer()?;
    let encoder = command.computeCommandEncoder()?;
    let count_u32 = u32::try_from(count).ok()?;
    encoder.setComputePipelineState(&gpu.kernels::<T>().reduce);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
        encoder.setBytes_length_atIndex(NonNull::from(&count_u32).cast(), 4, 2);
        encoder.setBytes_length_atIndex(NonNull::from(&op).cast(), size_of::<Reduce>(), 3);
    }
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: groups,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: REDUCE_GROUP,
            height: 1,
            depth: 1,
        },
    );
    encoder.endEncoding();
    commit(gpu, command)
}

/// One elementwise conversion between a typed buffer and an `f32` one —
/// `pipeline` is a type's `widen` or `narrow`.
fn encode_convert(
    gpu: &Gpu,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
) -> Option<()> {
    let command = gpu.queue.commandBuffer()?;
    let encoder = command.computeCommandEncoder()?;
    encoder.setComputePipelineState(pipeline);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
    }
    dispatch_1d(&encoder, len);
    encoder.endEncoding();
    commit(gpu, command)
}

fn encode_scan(
    gpu: &Gpu,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
    offset: usize,
) -> Option<()> {
    let command = gpu.queue.commandBuffer()?;
    let encoder = command.computeCommandEncoder()?;
    let offset_u32 = u32::try_from(offset).ok()?;
    encoder.setComputePipelineState(&gpu.scan);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
        encoder.setBytes_length_atIndex(NonNull::from(&offset_u32).cast(), 4, 2);
    }
    dispatch_1d(&encoder, len);
    encoder.endEncoding();
    commit(gpu, command)
}

fn encode_sort_prepare<T: MetalElement>(
    gpu: &Gpu,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    padded: usize,
    count: usize,
    padding: T,
) -> Option<()> {
    let command = gpu.queue.commandBuffer()?;
    let encoder = command.computeCommandEncoder()?;
    let count_u32 = u32::try_from(count).ok()?;
    encoder.setComputePipelineState(&gpu.kernels::<T>().sort_prepare);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
        encoder.setBytes_length_atIndex(NonNull::from(&count_u32).cast(), 4, 2);
        encoder.setBytes_length_atIndex(NonNull::from(&padding).cast(), size_of::<T>(), 3);
    }
    dispatch_1d(&encoder, padded);
    encoder.endEncoding();
    commit(gpu, command)
}

fn encode_bitonic_stage<T: MetalElement>(
    gpu: &Gpu,
    values: &ProtocolObject<dyn MTLBuffer>,
    padded: usize,
    block: usize,
    stride: usize,
    ascending: bool,
) -> Option<()> {
    let command = gpu.queue.commandBuffer()?;
    let encoder = command.computeCommandEncoder()?;
    let block_u32 = u32::try_from(block).ok()?;
    let stride_u32 = u32::try_from(stride).ok()?;
    let ascending_u32 = u32::from(ascending);
    encoder.setComputePipelineState(&gpu.kernels::<T>().bitonic);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(values), 0, 0);
        encoder.setBytes_length_atIndex(NonNull::from(&block_u32).cast(), 4, 1);
        encoder.setBytes_length_atIndex(NonNull::from(&stride_u32).cast(), 4, 2);
        encoder.setBytes_length_atIndex(NonNull::from(&ascending_u32).cast(), 4, 3);
    }
    dispatch_1d(&encoder, padded);
    encoder.endEncoding();
    commit(gpu, command)
}

fn encode_broadcast<T: MetalElement>(
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
    encoder.setComputePipelineState(&gpu.kernels::<T>().broadcast);
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

fn encode_stack<T: MetalElement>(
    gpu: &Gpu,
    inputs: &[&MetalBuffer<T>],
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
    encoder.setComputePipelineState(&gpu.kernels::<T>().stack_vector);
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

fn encode_concat<T: MetalElement>(
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
    encoder.setComputePipelineState(&gpu.kernels::<T>().concat_horizontal);
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

fn encode_matrix_stack<T: MetalElement>(
    gpu: &Gpu,
    top: &ProtocolObject<dyn MTLBuffer>,
    bottom: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    top_len: usize,
    bottom_len: usize,
) -> Option<()> {
    let top_bytes = top_len.checked_mul(size_of::<T>())?;
    let bottom_bytes = bottom_len.checked_mul(size_of::<T>())?;
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

fn encode_hmerge<T: MetalElement>(
    gpu: &Gpu,
    inputs: &[&MetalBuffer<T>],
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
    encoder.setComputePipelineState(&gpu.kernels::<T>().merge_horizontal);
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

fn encode_vmerge<T: MetalElement>(
    gpu: &Gpu,
    inputs: &[&MetalBuffer<T>],
    output: &ProtocolObject<dyn MTLBuffer>,
    matrix_len: usize,
) -> Option<()> {
    let matrix_bytes = matrix_len.checked_mul(size_of::<T>())?;
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

/// The first round of a deviation fold: one partial per threadgroup.
///
/// Dispatched as whole threadgroups for the same reason [`encode_reduce`] is —
/// every thread in a group has to reach the barriers.
fn encode_deviation<T: MetalElement>(
    gpu: &Gpu,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    count: usize,
    groups: usize,
    mean: f32,
) -> Option<()> {
    let command = gpu.queue.commandBuffer()?;
    let encoder = command.computeCommandEncoder()?;
    let count_u32 = u32::try_from(count).ok()?;
    encoder.setComputePipelineState(&gpu.kernels::<T>().deviation);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
        encoder.setBytes_length_atIndex(NonNull::from(&count_u32).cast(), 4, 2);
        encoder.setBytes_length_atIndex(NonNull::from(&mean).cast(), 4, 3);
    }
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: groups,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: REDUCE_GROUP,
            height: 1,
            depth: 1,
        },
    );
    encoder.endEncoding();
    commit(gpu, command)
}

#[allow(clippy::too_many_arguments)]
fn encode_axis_moments<T: MetalElement>(
    gpu: &Gpu,
    input: &ProtocolObject<dyn MTLBuffer>,
    means: &ProtocolObject<dyn MTLBuffer>,
    deviations: &ProtocolObject<dyn MTLBuffer>,
    rows: usize,
    cols: usize,
    extent: usize,
    axis: Axis,
) -> Option<()> {
    let command = gpu.queue.commandBuffer()?;
    let encoder = command.computeCommandEncoder()?;
    let (rows_u32, cols_u32) = (u32::try_from(rows).ok()?, u32::try_from(cols).ok()?);
    encoder.setComputePipelineState(&gpu.kernels::<T>().axis_moments);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(means), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(deviations), 0, 2);
        encoder.setBytes_length_atIndex(NonNull::from(&rows_u32).cast(), 4, 3);
        encoder.setBytes_length_atIndex(NonNull::from(&cols_u32).cast(), 4, 4);
        encoder.setBytes_length_atIndex(NonNull::from(&axis).cast(), size_of::<Axis>(), 5);
    }
    dispatch_1d(&encoder, extent);
    encoder.endEncoding();
    commit(gpu, command)
}

#[allow(clippy::too_many_arguments)]
fn encode_distribution<T: MetalElement>(
    gpu: &Gpu,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
    family: Family,
    statistic: Statistic,
    parameters: (f32, f32),
) -> Option<()> {
    let command = gpu.queue.commandBuffer()?;
    let encoder = command.computeCommandEncoder()?;
    encoder.setComputePipelineState(&gpu.kernels::<T>().distribution);
    let (first, second) = parameters;
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
        encoder.setBytes_length_atIndex(NonNull::from(&family).cast(), size_of::<Family>(), 2);
        encoder.setBytes_length_atIndex(
            NonNull::from(&statistic).cast(),
            size_of::<Statistic>(),
            3,
        );
        encoder.setBytes_length_atIndex(NonNull::from(&first).cast(), 4, 4);
        encoder.setBytes_length_atIndex(NonNull::from(&second).cast(), 4, 5);
    }
    dispatch_1d(&encoder, len);
    encoder.endEncoding();
    commit(gpu, command)
}

#[allow(clippy::too_many_arguments)]
fn encode_axis_distribution<T: MetalElement>(
    gpu: &Gpu,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    first: &ProtocolObject<dyn MTLBuffer>,
    second: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
    cols: usize,
    axis: Axis,
    family: Family,
    statistic: Statistic,
) -> Option<()> {
    let command = gpu.queue.commandBuffer()?;
    let encoder = command.computeCommandEncoder()?;
    let cols_u32 = u32::try_from(cols).ok()?;
    encoder.setComputePipelineState(&gpu.kernels::<T>().axis_distribution);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(first), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(second), 0, 3);
        encoder.setBytes_length_atIndex(NonNull::from(&cols_u32).cast(), 4, 4);
        encoder.setBytes_length_atIndex(NonNull::from(&axis).cast(), size_of::<Axis>(), 5);
        encoder.setBytes_length_atIndex(NonNull::from(&family).cast(), size_of::<Family>(), 6);
        encoder.setBytes_length_atIndex(
            NonNull::from(&statistic).cast(),
            size_of::<Statistic>(),
            7,
        );
    }
    dispatch_1d(&encoder, len);
    encoder.endEncoding();
    commit(gpu, command)
}

fn encode_power<T: MetalElement>(
    gpu: &Gpu,
    a: &ProtocolObject<dyn MTLBuffer>,
    b: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
) -> Option<()> {
    let command = gpu.queue.commandBuffer()?;
    let encoder = command.computeCommandEncoder()?;
    encoder.setComputePipelineState(&gpu.kernels::<T>().power);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(a), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(b), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 2);
    }
    dispatch_1d(&encoder, len);
    encoder.endEncoding();
    commit(gpu, command)
}

fn encode_power_scalar<T: MetalElement>(
    gpu: &Gpu,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
    scalar: f32,
    scalar_left: bool,
) -> Option<()> {
    let command = gpu.queue.commandBuffer()?;
    let encoder = command.computeCommandEncoder()?;
    encoder.setComputePipelineState(&gpu.kernels::<T>().power_scalar);
    let scalar_left = u32::from(scalar_left);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
        encoder.setBytes_length_atIndex(NonNull::from(&scalar).cast(), 4, 2);
        encoder.setBytes_length_atIndex(NonNull::from(&scalar_left).cast(), 4, 3);
    }
    dispatch_1d(&encoder, len);
    encoder.endEncoding();
    commit(gpu, command)
}

fn encode_unary<T: MetalElement>(
    gpu: &Gpu,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
    op: Analytic,
) -> Option<()> {
    let command = gpu.queue.commandBuffer()?;
    let encoder = command.computeCommandEncoder()?;
    encoder.setComputePipelineState(&gpu.kernels::<T>().unary);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
        encoder.setBytes_length_atIndex(NonNull::from(&op).cast(), size_of::<Analytic>(), 2);
    }
    dispatch_1d(&encoder, len);
    encoder.endEncoding();
    commit(gpu, command)
}

fn encode_unary_dual<T: MetalElement>(
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
    encoder.setComputePipelineState(&gpu.kernels::<T>().unary_dual);
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

/// The iteration space and length of a fused program, as the shader's
/// `FusedShape` reads them.
#[repr(C)]
struct FusedShape {
    rows: u32,
    cols: u32,
    count: u32,
}

/// Input slots in the `fused_elementwise` shader.
const FUSED_INPUT_SLOTS: usize = 16;
/// Output slots in the `fused_elementwise` shader.
const FUSED_OUTPUT_SLOTS: usize = 8;

/// Encode one fused elementwise program over a `rows × cols` space.
///
/// `inputs` and `outputs` are bound in slot order. A tensor updated in place
/// appears in both lists. Unused slots are bound to a buffer that is in use —
/// the program never names them, so they are never read or written.
pub(crate) fn fused_elementwise<T: MetalElement>(
    code: &[crate::tensors::fused::Encoded],
    (rows, cols): (usize, usize),
    inputs: &[&ProtocolObject<dyn MTLBuffer>],
    outputs: &[&ProtocolObject<dyn MTLBuffer>],
) -> Option<()> {
    let len = rows.checked_mul(cols)?;
    if len == 0
        || code.is_empty()
        || outputs.is_empty()
        || inputs.len() > FUSED_INPUT_SLOTS
        || outputs.len() > FUSED_OUTPUT_SLOTS
    {
        return None;
    }
    let shape = FusedShape {
        rows: u32::try_from(rows).ok()?,
        cols: u32::try_from(cols).ok()?,
        count: u32::try_from(code.len()).ok()?,
    };
    // `setBytes` is limited to 4 KB, which bounds the program length.
    let code_bytes = std::mem::size_of_val(code);
    if code_bytes > 4096 {
        return None;
    }
    let filler = inputs.first().copied().unwrap_or(outputs[0]);
    with_gpu(|gpu| {
        let command = gpu.queue.commandBuffer()?;
        let encoder = command.computeCommandEncoder()?;
        encoder.setComputePipelineState(&gpu.kernels::<T>().fused);
        unsafe {
            encoder.setBytes_length_atIndex(NonNull::from(&code[0]).cast(), code_bytes, 0);
            encoder.setBytes_length_atIndex(
                NonNull::from(&shape).cast(),
                size_of::<FusedShape>(),
                1,
            );
            for slot in 0..FUSED_INPUT_SLOTS {
                let buffer = inputs.get(slot).copied().unwrap_or(filler);
                encoder.setBuffer_offset_atIndex(Some(buffer), 0, 2 + slot);
            }
            for slot in 0..FUSED_OUTPUT_SLOTS {
                let buffer = outputs.get(slot).copied().unwrap_or(outputs[0]);
                encoder.setBuffer_offset_atIndex(Some(buffer), 0, 2 + FUSED_INPUT_SLOTS + slot);
            }
        }
        dispatch_1d(&encoder, len);
        encoder.endEncoding();
        commit(gpu, command)
    })
}

/// Encode `program(a·b, inputs…)`: an `m × k` by `k × n` product whose every
/// element is the program's input 0, fed to it before anything is stored.
///
/// `inputs` are the program's input slots from 1 on; slot 0's buffer is never
/// read. On an M5-class GPU the product runs on TensorOps and the program on
/// the staged tile; elsewhere on the tiled kernel.
#[allow(clippy::too_many_arguments)]
pub(crate) fn matmul_epilogue<T: MetalElement>(
    code: &[crate::tensors::fused::Encoded],
    (m, k, n): (usize, usize, usize),
    a: &ProtocolObject<dyn MTLBuffer>,
    b: &ProtocolObject<dyn MTLBuffer>,
    inputs: &[&ProtocolObject<dyn MTLBuffer>],
    outputs: &[&ProtocolObject<dyn MTLBuffer>],
) -> Option<()> {
    if m == 0
        || n == 0
        || k == 0
        || code.is_empty()
        || outputs.is_empty()
        || inputs.len() >= FUSED_INPUT_SLOTS
        || outputs.len() > FUSED_OUTPUT_SLOTS
    {
        return None;
    }
    let shape = FusedShape {
        rows: u32::try_from(m).ok()?,
        cols: u32::try_from(n).ok()?,
        count: u32::try_from(code.len()).ok()?,
    };
    let inner = u32::try_from(k).ok()?;
    let code_bytes = std::mem::size_of_val(code);
    if code_bytes > 4096 {
        return None;
    }
    with_gpu(|gpu| {
        let kernels = gpu.kernels::<T>();
        let command = gpu.queue.commandBuffer()?;
        let encoder = command.computeCommandEncoder()?;
        let tensorops = kernels.tensorops_epilogue.as_ref().filter(|_| tensorops_enabled());
        encoder.setComputePipelineState(tensorops.unwrap_or(&kernels.matmul_epilogue));
        unsafe {
            encoder.setBytes_length_atIndex(NonNull::from(&code[0]).cast(), code_bytes, 0);
            encoder.setBytes_length_atIndex(NonNull::from(&shape).cast(), size_of::<FusedShape>(), 1);
            encoder.setBuffer_offset_atIndex(Some(a), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(b), 0, 3);
            encoder.setBytes_length_atIndex(NonNull::from(&inner).cast(), 4, 4);
            for slot in 1..FUSED_INPUT_SLOTS {
                let buffer = inputs.get(slot - 1).copied().unwrap_or(a);
                encoder.setBuffer_offset_atIndex(Some(buffer), 0, 4 + slot);
            }
            for slot in 0..FUSED_OUTPUT_SLOTS {
                let buffer = outputs.get(slot).copied().unwrap_or(outputs[0]);
                encoder.setBuffer_offset_atIndex(Some(buffer), 0, 4 + FUSED_INPUT_SLOTS + slot);
            }
        }
        match tensorops {
            Some(pipeline) => encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: n.div_ceil(TENSOROPS_TILE_COLS),
                    height: m.div_ceil(TENSOROPS_TILE_ROWS),
                    depth: 1,
                },
                MTLSize {
                    width: pipeline.threadExecutionWidth() * 4,
                    height: 1,
                    depth: 1,
                },
            ),
            None => encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: n.div_ceil(TILE),
                    height: m.div_ceil(TILE),
                    depth: 1,
                },
                MTLSize {
                    width: TILE,
                    height: TILE,
                    depth: 1,
                },
            ),
        }
        encoder.endEncoding();
        commit(gpu, command)
    })
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
    crate::counters::command_buffer();
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
    if !pending.is_empty() {
        crate::counters::sync();
    }
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

        encode_elementwise::<f32>(gpu, &buf_a, &buf_b, &buf_c, len, op)?;

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

        encode_broadcast::<f32>(gpu, &input, &output, len, scalar, op, scalar_left)?;

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

#[allow(clippy::too_many_arguments)]
fn encode_correlate<T: MetalElement>(
    gpu: &Gpu,
    input: &ProtocolObject<dyn MTLBuffer>,
    weights: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    rows: usize,
    cols: usize,
    window_rows: usize,
    window_cols: usize,
    flip: bool,
) -> Option<()> {
    let command = gpu.queue.commandBuffer()?;
    let encoder = command.computeCommandEncoder()?;
    encoder.setComputePipelineState(&gpu.kernels::<T>().correlate);
    let shape = [
        u32::try_from(rows).ok()?,
        u32::try_from(cols).ok()?,
        u32::try_from(window_rows).ok()?,
        u32::try_from(window_cols).ok()?,
    ];
    let flip = u32::from(flip);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(weights), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 2);
        for (index, value) in shape.iter().enumerate() {
            encoder.setBytes_length_atIndex(NonNull::from(value).cast(), 4, 3 + index);
        }
        encoder.setBytes_length_atIndex(NonNull::from(&flip).cast(), 4, 7);
    }
    dispatch_2d(&encoder, cols - window_cols + 1, rows - window_rows + 1);
    encoder.endEncoding();
    commit(gpu, command)
}

fn encode_pad<T: MetalElement>(
    gpu: &Gpu,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    rows: usize,
    cols: usize,
    pad_rows: usize,
    pad_cols: usize,
) -> Option<()> {
    let command = gpu.queue.commandBuffer()?;
    let encoder = command.computeCommandEncoder()?;
    encoder.setComputePipelineState(&gpu.kernels::<T>().pad_zeros);
    let shape = [
        u32::try_from(rows).ok()?,
        u32::try_from(cols).ok()?,
        u32::try_from(pad_rows).ok()?,
        u32::try_from(pad_cols).ok()?,
    ];
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
        for (index, value) in shape.iter().enumerate() {
            encoder.setBytes_length_atIndex(NonNull::from(value).cast(), 4, 2 + index);
        }
    }
    dispatch_2d(&encoder, cols + 2 * pad_cols, rows + 2 * pad_rows);
    encoder.endEncoding();
    commit(gpu, command)
}

fn encode_flip<T: MetalElement>(
    gpu: &Gpu,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    rows: usize,
    cols: usize,
) -> Option<()> {
    let command = gpu.queue.commandBuffer()?;
    let encoder = command.computeCommandEncoder()?;
    encoder.setComputePipelineState(&gpu.kernels::<T>().flip_both);
    let (rows_u, cols_u) = (u32::try_from(rows).ok()?, u32::try_from(cols).ok()?);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
        encoder.setBytes_length_atIndex(NonNull::from(&rows_u).cast(), 4, 2);
        encoder.setBytes_length_atIndex(NonNull::from(&cols_u).cast(), 4, 3);
    }
    dispatch_2d(&encoder, cols, rows);
    encoder.endEncoding();
    commit(gpu, command)
}

fn dispatch_2d(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    width: usize,
    height: usize,
) {
    let grid = MTLSize {
        width,
        height,
        depth: 1,
    };
    let per_group = MTLSize {
        width: 16.min(width.max(1)),
        height: 16.min(height.max(1)),
        depth: 1,
    };
    encoder.dispatchThreads_threadsPerThreadgroup(grid, per_group);
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
    fn tensorops_multiplies_half_and_bfloat_inputs() {
        // Cross both 64×64 tile boundaries and leave partial tiles on each
        // edge, so this covers the dispatch geometry as well as the data types.
        let (m, k, n) = (70usize, 33usize, 69usize);
        let left = (0..m * k)
            .map(|index| ((index % 7) as f32 - 3.0) * 0.25)
            .collect::<Vec<_>>();
        let right = (0..k * n)
            .map(|index| ((index % 5) as f32 - 2.0) * 0.125)
            .collect::<Vec<_>>();
        let expected = cpu_matmul(&left, &right, m, k, n);

        let left_f16 = left.iter().copied().map(f16::from_f32).collect::<Vec<_>>();
        let right_f16 = right.iter().copied().map(f16::from_f32).collect::<Vec<_>>();
        let Some(left_f16) = MetalBuffer::<f16>::from_slice(&left_f16) else {
            eprintln!("no Metal device; skipping TensorOps comparison");
            return;
        };
        let right_f16 = MetalBuffer::<f16>::from_slice(&right_f16).unwrap();
        let Some(compact_f16) = left_f16.matmul(&right_f16, m, k, n) else {
            eprintln!("no Metal 4 TensorOps support; skipping TensorOps comparison");
            return;
        };
        for (actual, expected) in compact_f16.as_slice().iter().zip(&expected) {
            assert!((f32::from(*actual) - expected).abs() < 0.02);
        }
        let wide_f16 = left_f16.matmul_f32(&right_f16, m, k, n).unwrap();
        for (actual, expected) in wide_f16.as_slice().iter().zip(&expected) {
            assert!((actual - expected).abs() < 1e-4);
        }

        let left_bf16 = left.iter().copied().map(bf16::from_f32).collect::<Vec<_>>();
        let right_bf16 = right
            .iter()
            .copied()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
        let left_bf16 = MetalBuffer::<bf16>::from_slice(&left_bf16).unwrap();
        let right_bf16 = MetalBuffer::<bf16>::from_slice(&right_bf16).unwrap();
        let compact_bf16 = left_bf16.matmul(&right_bf16, m, k, n).unwrap();
        for (actual, expected) in compact_bf16.as_slice().iter().zip(&expected) {
            assert!((f32::from(*actual) - expected).abs() < 0.1);
        }
        let wide_bf16 = left_bf16.matmul_f32(&right_bf16, m, k, n).unwrap();
        for (actual, expected) in wide_bf16.as_slice().iter().zip(&expected) {
            assert!((actual - expected).abs() < 1e-4);
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

        let empty = MetalBuffer::<f32>::from_slice(&[]).unwrap();
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
