//! Encoding fused elementwise programs, alone or as a matrix-product epilogue.

use std::mem::size_of;
use std::ptr::NonNull;

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};

use super::MetalElement;
use super::codegen;
use super::device::{product_tile, with_gpu};
use super::encode::{
    TENSOROPS_TILE_COLS, TENSOROPS_TILE_ROWS, TILE, dispatch_1d, encode_vecmat_finish, vecmat_bands,
};
use super::sync::{compute, queued};

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

/// The buffer slot of the place table (see `FusedPlace` in `common.h`) in
/// the elementwise, row and sum kernels, and in the product epilogues.
const PLACES: usize = 26;
const EPILOGUE_PLACES: usize = 28;

/// Where one load of a fused program reads, as `FusedPlace` in `common.h`
/// lays it out and `fused::Place` describes it: storage element
/// `offset + col·col_step` plus, for the row split into the `lead` axes it
/// folds together, each coordinate times its axis's step.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub(crate) struct FusedPlace {
    pub(crate) offset: u32,
    pub(crate) col: u32,
    pub(crate) lead: u32,
    pub(crate) dims: [u32; crate::tensors::MAX_RANK - 1],
    pub(crate) steps: [u32; crate::tensors::MAX_RANK - 1],
}

/// Bind the table of where each of a program's loads reads.
fn bind_places(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    places: &[FusedPlace],
    index: usize,
) {
    assert!(
        !places.is_empty() && places.len() <= crate::tensors::fused::MAX_LOADS,
        "one place per load"
    );
    // SAFETY: the table is `places.len()` `FusedPlace`s, at most 4 KB.
    unsafe {
        encoder.setBytes_length_atIndex(
            NonNull::from(&places[0]).cast(),
            std::mem::size_of_val(places),
            index,
        );
    }
}

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
    places: &[FusedPlace],
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
        let encoder = compute(gpu)?;
        // A program seen before runs as a kernel of its own, reading its
        // constants from slot 0; otherwise the interpreter reads the program.
        let specialized = codegen::enabled()
            .then(|| gpu.specialized::<T>(code, codegen::Kernel::Elementwise))
            .flatten();
        match &specialized {
            Some(pipeline) => {
                let constants = codegen::constants(code);
                let bytes = std::mem::size_of_val(constants.as_slice()).max(4);
                encoder.setComputePipelineState(pipeline);
                unsafe {
                    let first = constants.first().unwrap_or(&0.0);
                    encoder.setBytes_length_atIndex(NonNull::from(first).cast(), bytes, 0);
                }
            }
            None => {
                encoder.setComputePipelineState(&gpu.kernels::<T>().fused);
                unsafe {
                    encoder.setBytes_length_atIndex(NonNull::from(&code[0]).cast(), code_bytes, 0);
                }
            }
        }
        unsafe {
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
            bind_places(&encoder, places, PLACES);
        }
        dispatch_1d(&encoder, len);
        queued(gpu, len)
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
    places: &[FusedPlace],
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
        let encoder = compute(gpu)?;
        let tensorops = gpu.tensorops_epilogue::<T>();
        // A program seen before runs compiled into a product kernel of its own,
        // in the tile that suits the product, reading its constants from slot 0.
        let specialized = tensorops.filter(|_| codegen::enabled()).and_then(|_| {
            let (relaxed, (_, tile)) = product_tile::<T>(m, n);
            let kernel = codegen::Kernel::Epilogue { tile, relaxed };
            Some((gpu.specialized::<T>(code, kernel)?, tile))
        });
        let constants;
        match &specialized {
            Some((pipeline, _)) => {
                constants = codegen::constants(code);
                let bytes = std::mem::size_of_val(constants.as_slice()).max(4);
                encoder.setComputePipelineState(pipeline);
                unsafe {
                    let first = constants.first().unwrap_or(&0.0);
                    encoder.setBytes_length_atIndex(NonNull::from(first).cast(), bytes, 0);
                }
            }
            None => {
                encoder.setComputePipelineState(tensorops.unwrap_or(&kernels.matmul_epilogue));
                unsafe {
                    encoder.setBytes_length_atIndex(NonNull::from(&code[0]).cast(), code_bytes, 0);
                }
            }
        }
        unsafe {
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
            bind_places(&encoder, places, EPILOGUE_PLACES);
        }
        match (&specialized, tensorops) {
            (Some((pipeline, (rows, cols, groups))), _) => encoder
                .dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: n.div_ceil(*cols),
                        height: m.div_ceil(*rows),
                        depth: 1,
                    },
                    MTLSize {
                        width: pipeline.threadExecutionWidth() * groups,
                        height: 1,
                        depth: 1,
                    },
                ),
            (None, Some(pipeline)) => encoder.dispatchThreadgroups_threadsPerThreadgroup(
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
            (None, None) => encoder.dispatchThreadgroups_threadsPerThreadgroup(
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
        queued(gpu, m.saturating_mul(k).saturating_mul(n) / 128)
    })
}

/// Encode the sums of a fused program's one output along each row (`rows`)
/// or down each column into `y`, without storing the output — once the
/// program has run often enough to have a kernel of its own. `None` before
/// then, leaving the caller to run the program and sum what it stores.
pub(crate) fn fused_sum<T: MetalElement>(
    code: &[crate::tensors::fused::Encoded],
    (rows, cols): (usize, usize),
    inputs: &[&ProtocolObject<dyn MTLBuffer>],
    places: &[FusedPlace],
    rows_axis: bool,
    y: &ProtocolObject<dyn MTLBuffer>,
) -> Option<()> {
    if rows == 0 || cols == 0 || code.is_empty() || inputs.len() > FUSED_INPUT_SLOTS {
        return None;
    }
    let shape = FusedShape {
        rows: u32::try_from(rows).ok()?,
        cols: u32::try_from(cols).ok()?,
        count: u32::try_from(code.len()).ok()?,
    };
    let kernel = if rows_axis {
        codegen::Kernel::RowSums
    } else {
        codegen::Kernel::ColumnSums
    };
    let filler = inputs.first().copied().unwrap_or(y);
    with_gpu(|gpu| {
        let pipeline = codegen::enabled()
            .then(|| gpu.specialized::<T>(code, kernel))
            .flatten()?;
        let constants = codegen::constants(code);
        let bind = |encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>| unsafe {
            let first = constants.first().unwrap_or(&0.0);
            let bytes = std::mem::size_of_val(constants.as_slice()).max(4);
            encoder.setBytes_length_atIndex(NonNull::from(first).cast(), bytes, 0);
            encoder.setBytes_length_atIndex(
                NonNull::from(&shape).cast(),
                size_of::<FusedShape>(),
                1,
            );
            for slot in 0..FUSED_INPUT_SLOTS {
                let buffer = inputs.get(slot).copied().unwrap_or(filler);
                encoder.setBuffer_offset_atIndex(Some(buffer), 0, 2 + slot);
            }
            bind_places(encoder, places, PLACES);
        };
        let work = rows.saturating_mul(cols);
        if rows_axis {
            const ROWS_PER_GROUP: usize = 8;
            let encoder = compute(gpu)?;
            encoder.setComputePipelineState(&pipeline);
            bind(&encoder);
            unsafe { encoder.setBuffer_offset_atIndex(Some(y), 0, 18) };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: rows.div_ceil(ROWS_PER_GROUP),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: pipeline.threadExecutionWidth() * ROWS_PER_GROUP,
                    height: 1,
                    depth: 1,
                },
            );
            return queued(gpu, work);
        }
        // The bands of rows each thread sums, as a vector–matrix product
        // splits them, and the same pass adding them up.
        let (bands, band) = vecmat_bands(rows, cols);
        let partial = super::MetalBuffer::<f32>::allocate(bands.checked_mul(cols)?)?;
        let band_u32 = u32::try_from(band).ok()?;
        let encoder = compute(gpu)?;
        encoder.setComputePipelineState(&pipeline);
        bind(&encoder);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(partial.raw()), 0, 18);
            encoder.setBytes_length_atIndex(NonNull::from(&band_u32).cast(), 4, 19);
        }
        encoder.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: cols,
                height: bands,
                depth: 1,
            },
            MTLSize {
                width: 64.min(cols),
                height: 1,
                depth: 1,
            },
        );
        queued(gpu, work)?;
        encode_vecmat_finish::<T>(gpu, partial.raw(), y, cols, bands, false)
    })
}

/// Encode a fused program that reads row statistics of its inputs as one
/// kernel over whole rows (see [`codegen::Kernel::Rows`]) — once the program
/// has run often enough to have one. `None` before then, leaving the caller to
/// compute the statistics and run the program on them.
///
/// `inputs` and `outputs` are bound as [`fused_elementwise`] binds them; the
/// slots of the statistics are bound to a filler the kernel never reads.
pub(crate) fn fused_rows<T: MetalElement>(
    code: &[crate::tensors::fused::Encoded],
    (rows, cols): (usize, usize),
    inputs: &[&ProtocolObject<dyn MTLBuffer>],
    outputs: &[&ProtocolObject<dyn MTLBuffer>],
    places: &[FusedPlace],
    statistics: codegen::RowStatistics,
) -> Option<()> {
    if rows == 0
        || cols == 0
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
    let filler = inputs.first().copied().unwrap_or(outputs[0]);
    with_gpu(|gpu| {
        let pipeline = codegen::enabled()
            .then(|| gpu.specialized::<T>(code, codegen::Kernel::Rows(statistics)))
            .flatten()?;
        let constants = codegen::constants(code);
        let encoder = compute(gpu)?;
        encoder.setComputePipelineState(&pipeline);
        unsafe {
            let first = constants.first().unwrap_or(&0.0);
            let bytes = std::mem::size_of_val(constants.as_slice()).max(4);
            encoder.setBytes_length_atIndex(NonNull::from(first).cast(), bytes, 0);
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
            bind_places(&encoder, places, PLACES);
        }
        // One threadgroup per row.
        let threads = codegen::row_threads(
            cols,
            pipeline.threadExecutionWidth(),
            pipeline.maxTotalThreadsPerThreadgroup(),
        );
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: rows,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
        queued(gpu, rows.saturating_mul(cols))
    })
}
