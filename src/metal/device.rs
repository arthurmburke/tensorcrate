//! The per-thread Metal device: its command queue, the compiled pipelines for
//! every element type, and the buffer pool, built once on first use.

use std::cell::{Cell, OnceCell, RefCell};
use std::collections::VecDeque;

use dispatch2::DispatchData;
use half::{bf16, f16};
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLCompileOptions, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily, MTLLanguageVersion, MTLLibrary,
};

use super::MetalElement;
use super::pool::Pool;

/// The compute kernels, compiled by `build.rs` and embedded in the crate.
///
/// `matmul_tiled` stages `TILE×TILE` blocks of A and B into threadgroup memory
/// so each loaded value is reused `TILE` times, which is far more
/// bandwidth-efficient than reading straight from device memory.
pub(super) const KERNEL_LIBRARY: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/tensorcrate.metallib"));
pub(super) const TENSOROPS_LIBRARY: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/tensorcrate_tensorops.metallib"));

/// Device, queue, pipelines, and buffer pool — cached per thread. Metal objects
/// are not `Send`, so a thread-local keeps everything on one thread.
pub(super) struct Gpu {
    pub(super) device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub(super) queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// One set of pipelines per [`MetalElement`], indexed by
    /// [`MetalElement::INDEX`].
    pub(super) typed: [Typed; 3],
    /// The scan runs over a `float` buffer whatever the tensor's type.
    pub(super) scan: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// The strided copy for elements of 1, 2, 4, 8 and 16 bytes, in that
    /// order: it moves bits, so it is chosen by width rather than type.
    pub(super) strided_copy: [Pipeline; 5],
    pub(super) fft_bit_reverse: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub(super) fft_stage: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub(super) pool: RefCell<Pool>,
    /// Command buffers committed but not yet known to have finished, oldest
    /// first, each with its sequence number — see [`commit`](super::sync::commit).
    pub(super) pending: RefCell<VecDeque<(u64, CommandBuffer)>>,
    /// The sequence number of the last command buffer committed.
    pub(super) committed: Cell<u64>,
    /// Every command buffer up to this sequence number has completed.
    pub(super) completed: Cell<u64>,
    /// How many times the CPU has blocked on the GPU, for the tests.
    pub(super) waits: Cell<u64>,
    /// How many operations have been encoded, for the tests.
    pub(super) operations: Cell<u64>,
    /// The command buffer operations are being encoded into, not yet committed
    /// — see [`compute`](super::sync::compute).
    pub(super) open: RefCell<Option<super::sync::Batch>>,
    /// Specialized kernels for fused programs, by [`codegen::key`](super::codegen::key).
    pub(super) specialized: RefCell<std::collections::HashMap<Vec<u64>, Specialized>>,
    /// The TensorOps library, on M5-class GPUs.
    pub(super) tensorops_library: Option<Retained<ProtocolObject<dyn MTLLibrary>>>,
    /// Pipelines from the TensorOps library made on first use, by function
    /// name; `None` for one that could not be made.
    pub(super) tensorops_named: RefCell<std::collections::HashMap<String, Option<Pipeline>>>,
}

/// A fused program's specialized kernel, as far as it has got.
pub(super) enum Specialized {
    /// Run on the interpreter this many times so far.
    Seen(u32),
    Ready(Pipeline),
    /// The generator or the compiler turned it down; the interpreter runs it.
    Failed,
}

/// Programs whose kernels a thread keeps before the cache starts over.
const SPECIALIZED_CACHE: usize = 256;

/// Command buffers kept queued ahead of the GPU before an allocation waits for
/// an old one to finish rather than making a new one.
///
/// Waiting wakes the CPU some tens of microseconds after the command buffer
/// completes, so the GPU needs that much work still queued behind it: with
/// four, a pipelined loop of 1M-element operations ran at a third of the
/// speed it does with eight.
const QUEUE_DEPTH: u64 = 8;

pub(super) type CommandBuffer = Retained<ProtocolObject<dyn MTLCommandBuffer>>;

pub(super) type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// The pipelines for one element type: each is the `_f32`, `_f16` or `_bf16`
/// instance of the shader of the same name.
pub(super) struct Typed {
    pub(super) matmul: Pipeline,
    /// Matrix–vector products: one SIMD group per row, and the two passes of
    /// a vector–matrix product.
    pub(super) matvec_rows: Pipeline,
    pub(super) vecmat_bands: Pipeline,
    pub(super) vecmat_finish: Pipeline,
    /// The TensorOps product for a 16-bit type with a 16-bit result, on
    /// M5-class GPUs. `f32` products run on the general kernel instead — see
    /// [`Gpu::gemm`].
    pub(super) tensorops: Option<Pipeline>,
    pub(super) elementwise: Pipeline,
    pub(super) broadcast: Pipeline,
    pub(super) compare: Pipeline,
    pub(super) compare_scalar: Pipeline,
    pub(super) clamp: Pipeline,
    pub(super) reduce: Pipeline,
    pub(super) widen: Pipeline,
    pub(super) narrow: Pipeline,
    pub(super) sort_prepare: Pipeline,
    pub(super) bitonic: Pipeline,
    pub(super) transpose: Pipeline,
    pub(super) correlate: Pipeline,
    pub(super) pad_zeros: Pipeline,
    pub(super) flip_both: Pipeline,
    pub(super) unary: Pipeline,
    pub(super) unary_dual: Pipeline,
    pub(super) power: Pipeline,
    pub(super) power_scalar: Pipeline,
    pub(super) deviation: Pipeline,
    pub(super) axis_moments: Pipeline,
    /// The same along rows, one SIMD group per row.
    pub(super) row_moments: Pipeline,
    pub(super) distribution: Pipeline,
    pub(super) axis_distribution: Pipeline,
    pub(super) fused: Pipeline,
    /// A binary operation reading both operands through their strides, the
    /// broadcast kernel.
    pub(super) strided_binary: Pipeline,
    /// A fold, mean or variance over any set of axes, and the position of
    /// the extreme along one.
    pub(super) axis_reduce: Pipeline,
    pub(super) axis_arg_reduce: Pipeline,
    /// A product whose elements feed a fused program before any store.
    pub(super) matmul_epilogue: Pipeline,
    /// The same on TensorOps, on M5-class GPUs.
    pub(super) tensorops_epilogue: Option<Pipeline>,
    /// The same with relaxed precision; `f32` only.
    pub(super) tensorops_epilogue_relaxed: Option<Pipeline>,
}

impl Gpu {
    /// The TensorOps function `name` as a pipeline, made on first use.
    pub(super) fn tensorops_named(&self, name: &str) -> Option<Pipeline> {
        if let Some(pipeline) = self.tensorops_named.borrow().get(name) {
            return pipeline.clone();
        }
        let pipeline = self.tensorops_library.as_ref().and_then(|library| {
            let function = library.newFunctionWithName(&NSString::from_str(name))?;
            self.device
                .newComputePipelineStateWithFunction_error(&function)
                .ok()
        });
        self.tensorops_named
            .borrow_mut()
            .insert(name.to_owned(), pipeline.clone());
        pipeline
    }

    /// The pipelines compiled for element type `T`.
    pub(super) fn kernels<T: MetalElement>(&self) -> &Typed {
        &self.typed[T::INDEX]
    }

    /// The strided copy for elements of `T`'s width, if it is one the kernel
    /// is instantiated for. Buffers are bound from their start, which is
    /// aligned for every width, so element `i` of any of them is aligned for
    /// the kernel's access whatever `T`'s own alignment.
    pub(super) fn strided_copy<T>(&self) -> Option<&Pipeline> {
        let index = match size_of::<T>() {
            1 => 0,
            2 => 1,
            4 => 2,
            8 => 3,
            16 => 4,
            _ => return None,
        };
        Some(&self.strided_copy[index])
    }

    /// The TensorOps product for a 16-bit `T` with a 16-bit result, if the GPU
    /// has TensorOps and they are enabled.
    pub(super) fn tensorops_matmul<T: MetalElement>(&self) -> Option<&Pipeline> {
        self.kernels::<T>()
            .tensorops
            .as_ref()
            .filter(|_| tensorops_enabled())
    }

    /// The general TensorOps product — `op(A)·op(B)`, added into the output if
    /// `accumulate` — for an `m × n` result of element type `T`, with the tile
    /// it computes per threadgroup. `f32` runs at this thread's
    /// [`MatmulPrecision`]; `f16` and `bf16` accumulate in `f32` and round
    /// once. `None` without TensorOps.
    pub(super) fn gemm<T: MetalElement>(
        &self,
        operands: Operands,
        accumulate: bool,
        m: usize,
        n: usize,
    ) -> Option<(Pipeline, Tile)> {
        if !tensorops_enabled() {
            return None;
        }
        let (relaxed, (tile, shape)) = product_tile::<T>(m, n);
        let kernel = if T::SUFFIX == f32::SUFFIX {
            if relaxed { "gemm_f32r" } else { "gemm_f32" }.to_string()
        } else {
            format!("gemm_narrow_{}", T::SUFFIX)
        };
        let operands = match operands {
            Operands::Plain => "nn",
            Operands::LeftTransposed => "tn",
            Operands::RightTransposed => "nt",
        };
        let name = format!(
            "{kernel}_{tile}_{operands}{}",
            if accumulate { "_acc" } else { "" }
        );
        Some((self.tensorops_named(&name)?, shape))
    }

    /// The TensorOps product of 16-bit matrices of type `input` (`"f16"` or
    /// `"bf16"`) with an `f32` result.
    pub(super) fn gemm_widening(
        &self,
        input: &str,
        m: usize,
        n: usize,
    ) -> Option<(Pipeline, Tile)> {
        if !tensorops_enabled() {
            return None;
        }
        let (tile, shape) = tile(m, n, &[(64, SMALL)]);
        Some((
            self.tensorops_named(&format!("gemm_{input}_f32_{tile}_nn"))?,
            shape,
        ))
    }

    /// The same for a product with a fused epilogue.
    pub(super) fn tensorops_epilogue<T: MetalElement>(&self) -> Option<&Pipeline> {
        let kernels = self.kernels::<T>();
        let relaxed = match matmul_precision() {
            MatmulPrecision::Relaxed => kernels.tensorops_epilogue_relaxed.as_ref(),
            MatmulPrecision::Exact => None,
        };
        relaxed
            .or(kernels.tensorops_epilogue.as_ref())
            .filter(|_| tensorops_enabled())
    }
}

/// Which operand of a general product is read transposed.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum Operands {
    Plain,
    LeftTransposed,
    RightTransposed,
}

/// The rows, columns and SIMD groups of the tile one threadgroup of a general
/// product computes.
pub(super) type Tile = (usize, usize, usize);

/// The tiles the general product is compiled for, by their names in the
/// kernel names.
const SMALL: (&str, Tile) = ("s", (32, 32, 4));
const MEDIUM: (&str, Tile) = ("m", (64, 32, 2));
const LARGE: (&str, Tile) = ("l", (64, 64, 4));

/// Whether an `m × n` TensorOps product over `T` runs at relaxed precision on
/// this thread, and the tile it is best computed in.
pub(super) fn product_tile<T: MetalElement>(m: usize, n: usize) -> (bool, (&'static str, Tile)) {
    if T::SUFFIX != f32::SUFFIX {
        // 16-bit inputs, which the matrix units take at full rate.
        return (false, tile(m, n, &[(64, SMALL)]));
    }
    match matmul_precision() {
        MatmulPrecision::Relaxed => (true, tile(m, n, &[(64, SMALL), (256, MEDIUM)])),
        MatmulPrecision::Exact => (false, tile(m, n, &[(257, SMALL)])),
    }
}

/// The tile for an `m × n` product: the first in `smaller` whose bound the
/// product's count of 64 × 64 tiles is below, and otherwise the largest.
///
/// A product with a small output has few large tiles — a 512 × 512 result
/// has 64 — which leaves most of the GPU's cores idle, so smaller outputs get
/// smaller tiles. The bounds were measured on an M5 Max, where they double
/// the speed of products from 256³ to 1024³.
fn tile(m: usize, n: usize, smaller: &[(usize, (&'static str, Tile))]) -> (&'static str, Tile) {
    let tiles = m.div_ceil(64) * n.div_ceil(64);
    smaller
        .iter()
        .find(|(bound, _)| tiles < *bound)
        .map_or(LARGE, |&(_, tile)| tile)
}

thread_local! {
    pub(super) static GPU: OnceCell<Option<Gpu>> = const { OnceCell::new() };
    pub(super) static TENSOROPS: Cell<bool> = const { Cell::new(true) };
    static PRECISION: Cell<MatmulPrecision> = const { Cell::new(MatmulPrecision::Exact) };
}

/// How exactly the GPU must compute an `f32` matrix product.
///
/// On M5-class GPUs a product runs on the matrix units of TensorOps. Asked for
/// full `f32` accuracy they reach about 15 TFLOP/s on a large product; allowed to
/// trade accuracy for speed, about 24 — at a relative error near `1e-3`, about
/// what `f16` inputs give. That is usually plenty for training a network, and
/// rarely enough for anything that solves a system or compares results
/// exactly, so it is opt-in.
///
/// Storing the operands as `f16` or `bf16` is faster again — about 55–60
/// TFLOP/s, with `matmul_f32` keeping an `f32` result — at the cost of the
/// storage format. On GPUs without TensorOps the setting changes nothing.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum MatmulPrecision {
    /// Every product is computed to `f32` accuracy. The default.
    #[default]
    Exact,
    /// `f32` products may run in reduced precision internally, for speed.
    Relaxed,
}

/// Set how exactly `f32` matrix products on this thread's GPU are computed,
/// including the products a fused epilogue runs on.
pub fn set_matmul_precision(precision: MatmulPrecision) {
    PRECISION.with(|cell| cell.set(precision));
}

/// This thread's [`MatmulPrecision`].
pub fn matmul_precision() -> MatmulPrecision {
    PRECISION.with(Cell::get)
}

/// Whether matrix products on this thread may use TensorOps where the GPU has
/// them (M5-class), or must use the tiled kernel. On by default; turning it
/// off is for testing the tiled kernels on hardware that would never reach
/// them, and for comparing the two.
#[doc(hidden)]
pub fn set_tensorops(enabled: bool) {
    TENSOROPS.with(|cell| cell.set(enabled));
}

pub(super) fn tensorops_enabled() -> bool {
    TENSOROPS.with(Cell::get)
}

pub(super) fn build_gpu() -> Option<Gpu> {
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
    let typed = |suffix: &str| -> Option<Typed> {
        let kernel = |name: &str| pipeline(&format!("{name}_{suffix}"));
        Some(Typed {
            matmul: kernel("matmul_tiled")?,
            matvec_rows: kernel("matvec_rows")?,
            vecmat_bands: kernel("vecmat_bands")?,
            vecmat_finish: kernel("vecmat_finish")?,
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
            row_moments: kernel("row_moments")?,
            distribution: kernel("distribution")?,
            axis_distribution: kernel("axis_distribution")?,
            fused: kernel("fused_elementwise")?,
            strided_binary: kernel("strided_binary")?,
            axis_reduce: kernel("axis_reduce")?,
            axis_arg_reduce: kernel("axis_arg_reduce")?,
            matmul_epilogue: kernel("matmul_epilogue")?,
            tensorops_epilogue: tensorops_pipeline(&format!("matmul_tensorops_epilogue_{suffix}")),
            tensorops_epilogue_relaxed: tensorops_pipeline(&format!(
                "matmul_tensorops_epilogue_{suffix}_relaxed"
            )),
        })
    };
    Some(Gpu {
        typed: [
            typed(f32::SUFFIX)?,
            typed(f16::SUFFIX)?,
            typed(bf16::SUFFIX)?,
        ],
        scan: pipeline("scan_step")?,
        strided_copy: [
            pipeline("strided_copy_x8")?,
            pipeline("strided_copy_x16")?,
            pipeline("strided_copy_x32")?,
            pipeline("strided_copy_x64")?,
            pipeline("strided_copy_x128")?,
        ],
        fft_bit_reverse: pipeline("fft_bit_reverse")?,
        fft_stage: pipeline("fft_stage")?,
        pool: RefCell::new(Pool::default()),
        pending: RefCell::new(VecDeque::new()),
        committed: Cell::new(0),
        completed: Cell::new(0),
        waits: Cell::new(0),
        operations: Cell::new(0),
        open: RefCell::new(None),
        specialized: RefCell::new(std::collections::HashMap::new()),
        tensorops_library,
        tensorops_named: RefCell::new(std::collections::HashMap::new()),
        device,
        queue,
    })
}

pub(super) fn with_gpu<R>(f: impl FnOnce(&Gpu) -> Option<R>) -> Option<R> {
    autoreleasepool(|_| GPU.with(|cell| cell.get_or_init(build_gpu).as_ref().and_then(f)))
}

impl Gpu {
    /// A shared allocation of at least `len` bytes: a pooled one if a suitable
    /// one is free, after reclaiming whatever the GPU has finished with since the
    /// last look, and otherwise a new one.
    pub(super) fn acquire(&self, len: usize) -> Option<Retained<ProtocolObject<dyn MTLBuffer>>> {
        if let Some(buffer) = self.pool.borrow_mut().take(len) {
            return Some(buffer);
        }
        super::sync::reclaim(self);
        if let Some(buffer) = self.pool.borrow_mut().take(len) {
            return Some(buffer);
        }
        // When the CPU has run well ahead of the GPU, every intermediate it
        // dropped is still owed to a queued kernel. Allocating afresh then
        // costs more than a kernel — Metal zero-fills new memory: about 5 µs
        // for 256 KB and 200 µs for 16 MB — and grows without bound, so once
        // enough work is queued to keep the GPU busy, wait for the oldest
        // command holding a suitable allocation instead.
        let fence = self.pool.borrow().oldest_fitting(len);
        if let Some(fence) = fence
            && fence + QUEUE_DEPTH <= self.committed.get()
            && super::sync::wait_through(self, fence)
            && let Some(buffer) = self.pool.borrow_mut().take(len)
        {
            return Some(buffer);
        }
        Pool::allocate(&self.device, len)
    }

    /// The specialized kernel for the fused program `code` over `T`, compiling
    /// it on the program's [`COMPILE_AFTER`](super::codegen::COMPILE_AFTER)th
    /// appearance; `None` until then, or if it cannot be compiled.
    pub(super) fn specialized<T: MetalElement>(
        &self,
        code: &[crate::tensors::fused::Encoded],
        kernel: super::codegen::Kernel,
    ) -> Option<Pipeline> {
        use super::codegen;
        let key = codegen::key::<T>(code, kernel);
        let mut cache = self.specialized.borrow_mut();
        let seen = match cache.get_mut(&key) {
            Some(Specialized::Ready(pipeline)) => return Some(pipeline.clone()),
            Some(Specialized::Failed) => return None,
            Some(Specialized::Seen(count)) => {
                *count += 1;
                *count
            }
            None => {
                if cache.len() >= SPECIALIZED_CACHE {
                    cache.clear();
                }
                cache.insert(key.clone(), Specialized::Seen(1));
                1
            }
        };
        if seen < codegen::COMPILE_AFTER {
            return None;
        }
        let metal4 = matches!(kernel, codegen::Kernel::Epilogue { .. });
        let compiled =
            codegen::source::<T>(code, kernel).and_then(|source| self.compile(&source, metal4));
        let entry = match &compiled {
            Some(pipeline) => Specialized::Ready(pipeline.clone()),
            None => Specialized::Failed,
        };
        cache.insert(key, entry);
        compiled
    }

    /// Compile `source`'s `fused_program`, as Metal 4 — which TensorOps need —
    /// if `metal4`.
    fn compile(&self, source: &str, metal4: bool) -> Option<Pipeline> {
        let options = metal4.then(|| {
            let options = MTLCompileOptions::new();
            options.setLanguageVersion(MTLLanguageVersion::Version4_0);
            options
        });
        let library = self
            .device
            .newLibraryWithSource_options_error(&NSString::from_str(source), options.as_deref())
            .map_err(|_error| {
                #[cfg(test)]
                eprintln!("a fused kernel failed to compile: {_error}");
            })
            .ok()?;
        let function = library.newFunctionWithName(&NSString::from_str("fused_program"))?;
        self.device
            .newComputePipelineStateWithFunction_error(&function)
            .ok()
    }

    /// The fence for an allocation released now: the last command buffer that
    /// could still use it, or `None` if no GPU work is outstanding. When the
    /// queue cannot be inspected, the allocation is assumed to be in use.
    pub(super) fn fence(&self) -> Option<u64> {
        // Work still being encoded will be the next command buffer committed.
        match self.open.try_borrow() {
            Ok(open) if open.is_none() => {}
            _ => return Some(self.committed.get() + 1),
        }
        match self.pending.try_borrow() {
            Ok(pending) if pending.is_empty() => None,
            _ => Some(self.committed.get()),
        }
    }
}
