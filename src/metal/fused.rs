//! Encoding fused elementwise programs, alone or as a matrix-product epilogue.

use std::mem::size_of;
use std::ptr::NonNull;

use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLSize,
};

use super::MetalElement;
use super::device::{tensorops_enabled, with_gpu};
use super::encode::{TENSOROPS_TILE_COLS, TENSOROPS_TILE_ROWS, TILE, dispatch_1d};
use super::sync::commit;

/// The iteration space and length of a fused program, as the shader's
/// `FusedShape` reads them.
#[repr(C)]
pub(super) struct FusedShape {
    pub(super) rows: u32,
    pub(super) cols: u32,
    pub(super) count: u32,
}

/// Input slots in the `fused_elementwise` shader.
pub(super) const FUSED_INPUT_SLOTS: usize = 16;
/// Output slots in the `fused_elementwise` shader.
pub(super) const FUSED_OUTPUT_SLOTS: usize = 8;

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
        let tensorops = kernels
            .tensorops_epilogue
            .as_ref()
            .filter(|_| tensorops_enabled());
        encoder.setComputePipelineState(tensorops.unwrap_or(&kernels.matmul_epilogue));
        unsafe {
            encoder.setBytes_length_atIndex(NonNull::from(&code[0]).cast(), code_bytes, 0);
            encoder.setBytes_length_atIndex(
                NonNull::from(&shape).cast(),
                size_of::<FusedShape>(),
                1,
            );
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
