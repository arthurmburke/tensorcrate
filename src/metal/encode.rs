//! Encoding single kernels: binding buffers and parameters to a pipeline and
//! dispatching a grid. Nothing here allocates or waits.

use std::mem::size_of;
use std::ptr::NonNull;

use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize,
};

use crate::tensors::{Analytic, Axis, BinaryOp, Compare, Family, Reduce, Statistic};

use super::MetalElement;
use super::buffer::MetalBuffer;
use super::device::{Gpu, tensorops_enabled};
use super::sync::commit;

/// Threadgroup tile edge; must match `TILE` in the shader. 16×16 = 256 threads.
pub(super) const TILE: usize = 16;

/// M5 TensorOps threadgroup tile. Four SIMD groups form a 2×2 arrangement of
/// 32×32 SIMD-group tiles, matching Apple's recommended 16-bit starting point.
pub(super) const TENSOROPS_TILE_ROWS: usize = 64;
pub(super) const TENSOROPS_TILE_COLS: usize = 64;

/// Threads per group in the tree reduction; must match `REDUCE_GROUP` in the
/// shader, which sizes its threadgroup scratch array with it.
pub(super) const REDUCE_GROUP: usize = 256;

// One argument over clippy's threshold: the kernel takes three shapes and an
// accumulate flag, and naming them beats packing them into a struct here.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_matmul<T: MetalElement>(
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
pub(super) fn encode_tensorops_matmul(
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

pub(super) fn encode_transpose<T: MetalElement>(
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

pub(super) fn encode_elementwise<T: MetalElement>(
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

pub(super) fn encode_compare<T: MetalElement>(
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

pub(super) fn encode_compare_scalar<T: MetalElement>(
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

pub(super) fn encode_clamp<T: MetalElement>(
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
pub(super) fn encode_reduce<T: MetalElement>(
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
pub(super) fn encode_convert(
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

pub(super) fn encode_scan(
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

pub(super) fn encode_sort_prepare<T: MetalElement>(
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

pub(super) fn encode_bitonic_stage<T: MetalElement>(
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

pub(super) fn encode_broadcast<T: MetalElement>(
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

pub(super) fn encode_stack<T: MetalElement>(
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

pub(super) fn encode_concat<T: MetalElement>(
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

pub(super) fn encode_matrix_stack<T: MetalElement>(
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

pub(super) fn encode_hmerge<T: MetalElement>(
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

pub(super) fn encode_vmerge<T: MetalElement>(
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
pub(super) fn encode_deviation<T: MetalElement>(
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
pub(super) fn encode_axis_moments<T: MetalElement>(
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
pub(super) fn encode_distribution<T: MetalElement>(
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
pub(super) fn encode_axis_distribution<T: MetalElement>(
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

pub(super) fn encode_power<T: MetalElement>(
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

pub(super) fn encode_power_scalar<T: MetalElement>(
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

pub(super) fn encode_unary<T: MetalElement>(
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

pub(super) fn encode_unary_dual<T: MetalElement>(
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

pub(super) fn encode_fft(
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

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_correlate<T: MetalElement>(
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

pub(super) fn encode_pad<T: MetalElement>(
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

pub(super) fn encode_flip<T: MetalElement>(
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

pub(super) fn dispatch_2d(
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

pub(super) fn dispatch_1d(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, len: usize) {
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
