//! The per-thread Metal device: its command queue, the compiled pipelines for
//! every element type, and the buffer pool, built once on first use.

use std::cell::{Cell, OnceCell, RefCell};

use dispatch2::DispatchData;
use half::{bf16, f16};
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandQueue, MTLComputePipelineState, MTLCreateSystemDefaultDevice,
    MTLDevice, MTLGPUFamily, MTLLibrary,
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
    /// The `f16`/`bf16` × `f32`-output TensorOps products behind `matmul_f32`.
    pub(super) tensorops: Option<TensorOpsPipelines>,
    /// The scan runs over a `float` buffer whatever the tensor's type.
    pub(super) scan: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub(super) fft_bit_reverse: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub(super) fft_stage: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub(super) pool: RefCell<Pool>,
    /// Command buffers committed but not yet waited on — see [`commit`](super::sync::commit).
    pub(super) pending: RefCell<Vec<Retained<ProtocolObject<dyn MTLCommandBuffer>>>>,
}

pub(super) type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// The pipelines for one element type: each is the `_f32`, `_f16` or `_bf16`
/// instance of the shader of the same name.
pub(super) struct Typed {
    pub(super) matmul: Pipeline,
    /// The TensorOps product for this type, on M5-class GPUs.
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
    pub(super) stack_vector: Pipeline,
    pub(super) concat_horizontal: Pipeline,
    pub(super) merge_horizontal: Pipeline,
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
    pub(super) distribution: Pipeline,
    pub(super) axis_distribution: Pipeline,
    pub(super) fused: Pipeline,
    /// A product whose elements feed a fused program before any store.
    pub(super) matmul_epilogue: Pipeline,
    /// The same on TensorOps, on M5-class GPUs.
    pub(super) tensorops_epilogue: Option<Pipeline>,
}

impl Gpu {
    /// The pipelines compiled for element type `T`.
    pub(super) fn kernels<T: MetalElement>(&self) -> &Typed {
        &self.typed[T::INDEX]
    }
}

pub(super) struct TensorOpsPipelines {
    pub(super) f16_f32: Pipeline,
    pub(super) bf16_f32: Pipeline,
}

thread_local! {
    pub(super) static GPU: OnceCell<Option<Gpu>> = const { OnceCell::new() };
    pub(super) static TENSOROPS: Cell<bool> = const { Cell::new(true) };
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

pub(super) fn with_gpu<R>(f: impl FnOnce(&Gpu) -> Option<R>) -> Option<R> {
    autoreleasepool(|_| GPU.with(|cell| cell.get_or_init(build_gpu).as_ref().and_then(f)))
}
