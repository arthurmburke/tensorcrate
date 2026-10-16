use super::{
    MetalBuffer,
    device::with_gpu,
    operators::device,
    sync::{compute, queued},
};
use crate::{
    numbers::{Coefficient, Complex, bf16, f16},
    tensors::MetalStorage,
};
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};
use std::{mem::size_of, ptr::NonNull};
mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for super::f16 {}
    impl Sealed for super::bf16 {}
    impl Sealed for super::Complex<f32> {}
}
/// Elements supported by the fused cross-product shader.
pub trait CrossElement: Coefficient + sealed::Sealed {
    #[doc(hidden)]
    const CROSS_INDEX: usize;
}
impl CrossElement for f32 {
    const CROSS_INDEX: usize = 0;
}
impl CrossElement for f16 {
    const CROSS_INDEX: usize = 1;
}
impl CrossElement for bf16 {
    const CROSS_INDEX: usize = 2;
}
impl CrossElement for Complex<f32> {
    const CROSS_INDEX: usize = 3;
}
pub(crate) fn cross_values<T: CrossElement>(
    a: &MetalStorage<T>,
    b: &MetalStorage<T>,
) -> MetalStorage<T> {
    assert_eq!(a.len(), b.len());
    assert!(a.len().is_multiple_of(3));
    let n = u32::try_from(a.len()).expect("Metal cross exceeds 32-bit indexing");
    let output = MetalBuffer::<T>::allocate(a.len()).expect("Metal cross allocation failed");
    if n != 0 {
        with_gpu(|gpu| {
            let encoder = compute(gpu)?;
            let pipeline = &gpu.cross[T::CROSS_INDEX];
            encoder.setComputePipelineState(pipeline);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(device(a).raw()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(device(b).raw()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(output.raw()), 0, 2);
                encoder.setBytes_length_atIndex(NonNull::from(&n).cast(), size_of::<u32>(), 3);
            }
            let group = pipeline.threadExecutionWidth() * 4;
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: a.len().div_ceil(group),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: group,
                    height: 1,
                    depth: 1,
                },
            );
            queued(gpu, a.len())
        })
        .expect("Metal cross dispatch failed");
    }
    MetalStorage::from_device(output)
}
