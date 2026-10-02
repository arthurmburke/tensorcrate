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

mod buffer;
mod codegen;
mod device;
mod encode;
mod fused;
mod ops;
mod pool;
mod slices;
mod sync;
#[cfg(test)]
mod tests;

use half::{bf16, f16};

use crate::numbers::Real;

pub use buffer::MetalBuffer;
#[doc(hidden)]
pub use codegen::set_fused_codegen;
#[doc(hidden)]
pub use device::set_tensorops;
pub use device::{MatmulPrecision, matmul_precision, set_matmul_precision};
pub(crate) use codegen::RowStatistics;
pub(crate) use fused::{fused_elementwise, fused_rows, fused_sum, matmul_epilogue};
pub use slices::{broadcast_f32, elementwise_f32, fft_f32_interleaved, ifft_f32_interleaved};
pub use sync::synchronize;

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

