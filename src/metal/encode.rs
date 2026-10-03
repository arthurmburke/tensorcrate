//! Encoding single kernels: binding buffers and parameters to a pipeline and
//! dispatching a grid. Nothing here allocates or waits.

use std::mem::size_of;
use std::ptr::NonNull;

use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize,
};

use crate::tensors::backend::Strided;
use crate::tensors::kernels::{AxisReduction, Pairwise};
use crate::tensors::layout::{Split, coalesce, reach};
use crate::tensors::{Analytic, Axis, BinaryOp, Compare, Family, MAX_RANK, Reduce, Statistic};

use super::MetalElement;
use super::device::{Gpu, Operands, Tile};
use super::sync::{blit, compute, queued};

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
    // The 16-bit product has a dedicated kernel for the plain case; the general
    // one covers the rest, accumulating included.
    if !accumulate && let Some(pipeline) = gpu.tensorops_matmul::<T>() {
        return encode_tensorops_matmul(gpu, pipeline, a, b, output, m, k, n);
    }
    if let Some((pipeline, tile)) = gpu.gemm::<T>(Operands::Plain, accumulate, m, n) {
        return encode_gemm(gpu, &pipeline, a, b, output, (m, k, n), tile);
    }
    let encoder = compute(gpu)?;
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
    queued(gpu, m.saturating_mul(k).saturating_mul(n) / 128)
}

/// A TensorOps product `output = a·b`. It writes every element of `output` —
/// the cooperative store covers each in-bounds element of every tile, partial
/// edge tiles included — so the allocation needs no clearing first.
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
) -> Option<()> {
    let encoder = compute(gpu)?;
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
    queued(gpu, m.saturating_mul(k).saturating_mul(n) / 128)
}

/// `y = M·x`, or `y += M·x`, for an `rows × cols` matrix `M`: one SIMD group
/// per row.
pub(super) fn encode_matvec<T: MetalElement>(
    gpu: &Gpu,
    matrix: &ProtocolObject<dyn MTLBuffer>,
    x: &ProtocolObject<dyn MTLBuffer>,
    y: &ProtocolObject<dyn MTLBuffer>,
    (rows, cols): (usize, usize),
    accumulate: bool,
) -> Option<()> {
    const ROWS_PER_GROUP: usize = 8;
    let encoder = compute(gpu)?;
    let pipeline = &gpu.kernels::<T>().matvec_rows;
    encoder.setComputePipelineState(pipeline);
    let (rows_u32, cols_u32) = (u32::try_from(rows).ok()?, u32::try_from(cols).ok()?);
    let accumulate = u32::from(accumulate);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(matrix), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(x), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(y), 0, 2);
        encoder.setBytes_length_atIndex(NonNull::from(&rows_u32).cast(), 4, 3);
        encoder.setBytes_length_atIndex(NonNull::from(&cols_u32).cast(), 4, 4);
        encoder.setBytes_length_atIndex(NonNull::from(&accumulate).cast(), 4, 5);
    }
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
    queued(gpu, rows.saturating_mul(cols) / 64)
}

/// The bands of rows `y = v·M` sums its `rows × cols` matrix in: as many as
/// keep about this many threads busy, each band at least a few rows long.
pub(super) fn vecmat_bands(rows: usize, cols: usize) -> (usize, usize) {
    const THREADS: usize = 64 * 1024;
    const FEWEST_ROWS: usize = 16;
    let bands = (THREADS / cols.max(1)).clamp(1, rows.div_ceil(FEWEST_ROWS).max(1));
    let band = rows.div_ceil(bands).max(1);
    (rows.div_ceil(band), band)
}

/// `y = v·M`, or `y += v·M`, for an `rows × cols` matrix `M`: the sums of
/// each band of rows into `partial` (see [`vecmat_bands`]), then their totals.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_vecmat<T: MetalElement>(
    gpu: &Gpu,
    v: &ProtocolObject<dyn MTLBuffer>,
    matrix: &ProtocolObject<dyn MTLBuffer>,
    partial: &ProtocolObject<dyn MTLBuffer>,
    y: &ProtocolObject<dyn MTLBuffer>,
    (rows, cols): (usize, usize),
    accumulate: bool,
) -> Option<()> {
    let (bands, band) = vecmat_bands(rows, cols);
    let kernels = gpu.kernels::<T>();
    let (rows_u32, cols_u32) = (u32::try_from(rows).ok()?, u32::try_from(cols).ok()?);
    let band_u32 = u32::try_from(band).ok()?;
    let accumulate = u32::from(accumulate);
    let encoder = compute(gpu)?;
    encoder.setComputePipelineState(&kernels.vecmat_bands);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(v), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(matrix), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(partial), 0, 2);
        encoder.setBytes_length_atIndex(NonNull::from(&rows_u32).cast(), 4, 3);
        encoder.setBytes_length_atIndex(NonNull::from(&cols_u32).cast(), 4, 4);
        encoder.setBytes_length_atIndex(NonNull::from(&band_u32).cast(), 4, 5);
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
    queued(gpu, rows.saturating_mul(cols) / 64)?;
    encode_vecmat_finish::<T>(gpu, partial, y, cols, bands, accumulate != 0)
}

/// `y = Σ partial[b]`, or `y += …`, over the `bands` rows of `cols` partial
/// sums: the second pass of [`encode_vecmat`].
pub(super) fn encode_vecmat_finish<T: MetalElement>(
    gpu: &Gpu,
    partial: &ProtocolObject<dyn MTLBuffer>,
    y: &ProtocolObject<dyn MTLBuffer>,
    cols: usize,
    bands: usize,
    accumulate: bool,
) -> Option<()> {
    let (cols_u32, bands_u32) = (u32::try_from(cols).ok()?, u32::try_from(bands).ok()?);
    let accumulate = u32::from(accumulate);
    let encoder = compute(gpu)?;
    encoder.setComputePipelineState(&gpu.kernels::<T>().vecmat_finish);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(partial), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(y), 0, 1);
        encoder.setBytes_length_atIndex(NonNull::from(&cols_u32).cast(), 4, 2);
        encoder.setBytes_length_atIndex(NonNull::from(&bands_u32).cast(), 4, 3);
        encoder.setBytes_length_atIndex(NonNull::from(&accumulate).cast(), 4, 4);
    }
    dispatch_1d(&encoder, cols);
    queued(gpu, bands.saturating_mul(cols) / 64)
}

/// A product on the general TensorOps kernel `pipeline` (see [`Gpu::gemm`]),
/// whose threadgroups each compute one `tile.0 × tile.1` tile of the `m × n`
/// output with `tile.2` SIMD groups. It writes, or adds into, every element of
/// `output`, partial edge tiles included.
pub(super) fn encode_gemm(
    gpu: &Gpu,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    a: &ProtocolObject<dyn MTLBuffer>,
    b: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    (m, k, n): (usize, usize, usize),
    tile: Tile,
) -> Option<()> {
    let encoder = compute(gpu)?;
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
            width: n.div_ceil(tile.1),
            height: m.div_ceil(tile.0),
            depth: 1,
        },
        MTLSize {
            width: pipeline.threadExecutionWidth() * tile.2,
            height: 1,
            depth: 1,
        },
    );
    queued(gpu, m.saturating_mul(k).saturating_mul(n) / 128)
}

/// The layout argument of the `strided_copy` shader, field for field.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct StridedLayout {
    rank: u32,
    source_offset: u32,
    target_offset: u32,
    shape: [u32; MAX_RANK],
    source_strides: [u32; MAX_RANK],
    target_strides: [u32; MAX_RANK],
}

impl StridedLayout {
    /// The shader's layout for copying `shape` from `from` to `to`, with the
    /// axes [`coalesce`]d so the kernel does as little index arithmetic as the
    /// layouts allow, and the number of elements it copies. `None` if an
    /// element index, on either side, does not fit the shader's 32 bits. Both
    /// sides are assumed checked against their storage.
    pub(super) fn new(
        shape: &[usize],
        from: Strided<'_>,
        to: Strided<'_>,
    ) -> Option<(Self, usize)> {
        let count = shape
            .iter()
            .try_fold(1usize, |count, &extent| count.checked_mul(extent))?;
        u32::try_from(count).ok()?;
        u32::try_from(reach(shape, from.offset, from.strides)?).ok()?;
        u32::try_from(reach(shape, to.offset, to.strides)?).ok()?;
        let mut layout = StridedLayout {
            source_offset: u32::try_from(from.offset).ok()?,
            target_offset: u32::try_from(to.offset).ok()?,
            ..StridedLayout::default()
        };
        if count != 0 {
            let (shape, [source, target]) = coalesce(shape, [from.strides, to.strides]);
            layout.rank = shape.len() as u32;
            for axis in 0..shape.len() {
                // Every coalesced axis has two or more elements, so each
                // stride is below the reach checked above.
                layout.shape[axis] = shape[axis] as u32;
                layout.source_strides[axis] = source[axis] as u32;
                layout.target_strides[axis] = target[axis] as u32;
            }
        }
        Some((layout, count))
    }
}

/// `target[layout.to(i)] = source[layout.from(i)]` for each of `count`
/// elements of `T`'s width. `None` if no kernel is instantiated for the width.
pub(super) fn encode_strided_copy<T>(
    gpu: &Gpu,
    source: &ProtocolObject<dyn MTLBuffer>,
    target: &ProtocolObject<dyn MTLBuffer>,
    layout: &StridedLayout,
    count: usize,
) -> Option<()> {
    let pipeline = gpu.strided_copy::<T>()?;
    let encoder = compute(gpu)?;
    encoder.setComputePipelineState(pipeline);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(source), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(target), 0, 1);
        encoder.setBytes_length_atIndex(
            NonNull::from(layout).cast(),
            size_of::<StridedLayout>(),
            2,
        );
    }
    dispatch_1d(&encoder, count);
    queued(gpu, count)
}

/// The layout argument of the `strided_binary` shader, field for field.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct BinaryLayout {
    rank: u32,
    kind: u32,
    op: u32,
    a_offset: u32,
    b_offset: u32,
    shape: [u32; MAX_RANK],
    a_strides: [u32; MAX_RANK],
    b_strides: [u32; MAX_RANK],
}

impl BinaryLayout {
    /// The shader's layout for `op` over `shape`, reading the operands at `a`
    /// and `b`, with the axes [`coalesce`]d, and the number of elements it
    /// writes. `None` for an operation the shader does not have — the
    /// remainder — or if an index does not fit its 32 bits.
    pub(super) fn new(
        shape: &[usize],
        a: Strided<'_>,
        b: Strided<'_>,
        op: Pairwise,
    ) -> Option<(Self, usize)> {
        let (kind, op) = match op {
            Pairwise::Arithmetic(BinaryOp::Rem) => return None,
            Pairwise::Arithmetic(op) => (0, u16::from(op)),
            Pairwise::Compare(op) => (1, u16::from(op)),
            Pairwise::Power => (2, 0),
        };
        let count = shape
            .iter()
            .try_fold(1usize, |count, &extent| count.checked_mul(extent))?;
        u32::try_from(count).ok()?;
        u32::try_from(reach(shape, a.offset, a.strides)?).ok()?;
        u32::try_from(reach(shape, b.offset, b.strides)?).ok()?;
        let mut layout = BinaryLayout {
            kind,
            op: u32::from(op),
            a_offset: u32::try_from(a.offset).ok()?,
            b_offset: u32::try_from(b.offset).ok()?,
            ..BinaryLayout::default()
        };
        if count != 0 {
            // The output is row-major, which joins any two adjacent axes, so
            // the operands alone decide which merge.
            let (shape, [a, b]) = coalesce(shape, [a.strides, b.strides]);
            layout.rank = shape.len() as u32;
            for axis in 0..shape.len() {
                layout.shape[axis] = shape[axis] as u32;
                layout.a_strides[axis] = a[axis] as u32;
                layout.b_strides[axis] = b[axis] as u32;
            }
        }
        Some((layout, count))
    }
}

/// `c[i] = op(a[layout.a(i)], b[layout.b(i)])` for each of `count` elements.
pub(super) fn encode_strided_binary<T: MetalElement>(
    gpu: &Gpu,
    a: &ProtocolObject<dyn MTLBuffer>,
    b: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    layout: &BinaryLayout,
    count: usize,
) -> Option<()> {
    let encoder = compute(gpu)?;
    encoder.setComputePipelineState(&gpu.kernels::<T>().strided_binary);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(a), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(b), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 2);
        encoder.setBytes_length_atIndex(NonNull::from(layout).cast(), size_of::<BinaryLayout>(), 3);
    }
    dispatch_1d(&encoder, count);
    queued(gpu, count)
}

/// The layout argument of the `axis_reduce` and `axis_arg_reduce` shaders,
/// field for field.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub(super) struct ReduceLayout {
    outer_rank: u32,
    inner_rank: u32,
    offset: u32,
    inner_count: u32,
    outer_shape: [u32; MAX_RANK],
    outer_strides: [u32; MAX_RANK],
    inner_shape: [u32; MAX_RANK],
    inner_strides: [u32; MAX_RANK],
    kind: u32,
    op: u32,
    lanes: u32,
    divisor: f32,
}

/// What a reduction kernel computes for each slice.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(super) enum Folding {
    /// `axis_reduce`, with this operation.
    Values(AxisReduction),
    /// `axis_arg_reduce`, for the position of the `Max` or the `Min`.
    Position(Reduce),
}

/// The lanes of a SIMD group the reduction shaders combine across.
pub(super) const SIMD_LANES: usize = 32;

/// Fewer results than this make a SIMD group per result worthwhile even when
/// its lanes do not read neighbouring elements: one thread each would leave
/// most of the GPU idle.
const FEW_RESULTS: usize = 1024;

/// Slices shorter than this fold on one thread: a SIMD group would leave most
/// of its lanes idle.
const SHORT_SLICE: usize = 64;

impl ReduceLayout {
    /// The shader's layout for folding `split`, each side [`coalesce`]d, and
    /// the number of results. `simd_width` is the pipeline's; a SIMD group
    /// per result is used only when it is [`SIMD_LANES`]. `None` if an index
    /// does not fit the shader's 32 bits. Every slice holds an element.
    pub(super) fn new(split: &Split, folding: Folding, simd_width: usize) -> Option<(Self, usize)> {
        let (results, depth) = (split.results(), split.depth());
        u32::try_from(results.checked_mul(SIMD_LANES)?).ok()?;
        let shape = split.kept.iter().chain(split.folded.iter());
        let strides = split.kept_strides.iter().chain(split.folded_strides.iter());
        let (shape, strides): (Vec<usize>, Vec<usize>) =
            (shape.copied().collect(), strides.copied().collect());
        u32::try_from(reach(&shape, split.offset, &strides)?).ok()?;
        let (kind, op, divisor) = match folding {
            Folding::Values(AxisReduction::Fold(op)) => (0, op, 0.0),
            Folding::Values(AxisReduction::Mean) => (1, Reduce::Sum, 0.0),
            Folding::Values(AxisReduction::Variance { divisor }) => {
                (2, Reduce::Sum, divisor as f32)
            }
            Folding::Position(op) => (0, op, 0.0),
        };
        let mut layout = ReduceLayout {
            offset: u32::try_from(split.offset).ok()?,
            inner_count: u32::try_from(depth).ok()?,
            kind,
            op: u32::from(u16::from(op)),
            divisor,
            ..ReduceLayout::default()
        };
        let (outer, [outer_strides]) = coalesce(&split.kept, [&split.kept_strides]);
        layout.outer_rank = outer.len() as u32;
        for axis in 0..outer.len() {
            layout.outer_shape[axis] = outer[axis] as u32;
            layout.outer_strides[axis] = outer_strides[axis] as u32;
        }
        let (inner, [inner_strides]) = coalesce(&split.folded, [&split.folded_strides]);
        layout.inner_rank = inner.len() as u32;
        for axis in 0..inner.len() {
            layout.inner_shape[axis] = inner[axis] as u32;
            layout.inner_strides[axis] = inner_strides[axis] as u32;
        }
        // A SIMD group per result when its lanes read neighbouring elements
        // of a long slice, or when there are too few results otherwise.
        let unit_inner = inner.last().is_some() && inner_strides[inner.len() - 1] == 1;
        let grouped = simd_width == SIMD_LANES
            && depth >= SHORT_SLICE
            && (unit_inner || results < FEW_RESULTS);
        layout.lanes = if grouped { SIMD_LANES as u32 } else { 1 };
        Some((layout, results))
    }
}

/// One reduction kernel — `pipeline` is a type's `axis_reduce` or
/// `axis_arg_reduce` — over `layout`, writing `results` results to `output`.
pub(super) fn encode_axis_reduce(
    gpu: &Gpu,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    layout: &ReduceLayout,
    results: usize,
) -> Option<()> {
    let encoder = compute(gpu)?;
    encoder.setComputePipelineState(pipeline);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
        encoder.setBytes_length_atIndex(NonNull::from(layout).cast(), size_of::<ReduceLayout>(), 2);
    }
    // A whole number of SIMD groups per result, and threadgroups of 256, so
    // every group of lanes lies in one SIMD group.
    dispatch_1d(&encoder, results * layout.lanes as usize);
    queued(gpu, results * layout.inner_count as usize)
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
    let encoder = compute(gpu)?;
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
    queued(gpu, rows * cols)
}

pub(super) fn encode_elementwise<T: MetalElement>(
    gpu: &Gpu,
    a: &ProtocolObject<dyn MTLBuffer>,
    b: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
    op: BinaryOp,
) -> Option<()> {
    let encoder = compute(gpu)?;
    encoder.setComputePipelineState(&gpu.kernels::<T>().elementwise);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(a), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(b), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 2);
        encoder.setBytes_length_atIndex(NonNull::from(&op).cast(), size_of::<BinaryOp>(), 3);
    }
    dispatch_1d(&encoder, len);
    queued(gpu, len)
}

pub(super) fn encode_compare<T: MetalElement>(
    gpu: &Gpu,
    a: &ProtocolObject<dyn MTLBuffer>,
    b: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
    op: Compare,
) -> Option<()> {
    let encoder = compute(gpu)?;
    encoder.setComputePipelineState(&gpu.kernels::<T>().compare);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(a), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(b), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 2);
        encoder.setBytes_length_atIndex(NonNull::from(&op).cast(), size_of::<Compare>(), 3);
    }
    dispatch_1d(&encoder, len);
    queued(gpu, len)
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
    let encoder = compute(gpu)?;
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
    queued(gpu, len)
}

pub(super) fn encode_clamp<T: MetalElement>(
    gpu: &Gpu,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
    low: f32,
    high: f32,
) -> Option<()> {
    let encoder = compute(gpu)?;
    encoder.setComputePipelineState(&gpu.kernels::<T>().clamp);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
        encoder.setBytes_length_atIndex(NonNull::from(&low).cast(), 4, 2);
        encoder.setBytes_length_atIndex(NonNull::from(&high).cast(), 4, 3);
    }
    dispatch_1d(&encoder, len);
    queued(gpu, len)
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
    let encoder = compute(gpu)?;
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
    queued(gpu, count)
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
    let encoder = compute(gpu)?;
    encoder.setComputePipelineState(pipeline);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
    }
    dispatch_1d(&encoder, len);
    queued(gpu, len)
}

pub(super) fn encode_scan(
    gpu: &Gpu,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
    offset: usize,
) -> Option<()> {
    let encoder = compute(gpu)?;
    let offset_u32 = u32::try_from(offset).ok()?;
    encoder.setComputePipelineState(&gpu.scan);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
        encoder.setBytes_length_atIndex(NonNull::from(&offset_u32).cast(), 4, 2);
    }
    dispatch_1d(&encoder, len);
    queued(gpu, len)
}

pub(super) fn encode_sort_prepare<T: MetalElement>(
    gpu: &Gpu,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    padded: usize,
    count: usize,
    padding: T,
) -> Option<()> {
    let encoder = compute(gpu)?;
    let count_u32 = u32::try_from(count).ok()?;
    encoder.setComputePipelineState(&gpu.kernels::<T>().sort_prepare);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
        encoder.setBytes_length_atIndex(NonNull::from(&count_u32).cast(), 4, 2);
        encoder.setBytes_length_atIndex(NonNull::from(&padding).cast(), size_of::<T>(), 3);
    }
    dispatch_1d(&encoder, padded);
    queued(gpu, padded)
}

pub(super) fn encode_bitonic_stage<T: MetalElement>(
    gpu: &Gpu,
    values: &ProtocolObject<dyn MTLBuffer>,
    padded: usize,
    block: usize,
    stride: usize,
    ascending: bool,
) -> Option<()> {
    let encoder = compute(gpu)?;
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
    queued(gpu, padded)
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
    let encoder = compute(gpu)?;
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
    queued(gpu, len)
}

/// Copy `bytes` from the front of `source` to the front of `destination` on the
/// GPU timeline, so it waits for whatever is still writing `source` without the
/// CPU having to.
pub(super) fn encode_copy(
    gpu: &Gpu,
    source: &ProtocolObject<dyn MTLBuffer>,
    destination: &ProtocolObject<dyn MTLBuffer>,
    bytes: usize,
) -> Option<()> {
    let encoder = blit(gpu)?;
    unsafe {
        encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
            source,
            0,
            destination,
            0,
            bytes,
        );
    }
    queued(gpu, bytes / 4)
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
    let encoder = compute(gpu)?;
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
    queued(gpu, count)
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
    let encoder = compute(gpu)?;
    let (rows_u32, cols_u32) = (u32::try_from(rows).ok()?, u32::try_from(cols).ok()?);
    if axis == Axis::Rows {
        const ROWS_PER_GROUP: usize = 8;
        let pipeline = &gpu.kernels::<T>().row_moments;
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(means), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(deviations), 0, 2);
            encoder.setBytes_length_atIndex(NonNull::from(&rows_u32).cast(), 4, 3);
            encoder.setBytes_length_atIndex(NonNull::from(&cols_u32).cast(), 4, 4);
        }
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
        return queued(gpu, rows * cols);
    }
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
    queued(gpu, rows * cols)
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
    let encoder = compute(gpu)?;
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
    queued(gpu, len)
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
    let encoder = compute(gpu)?;
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
    queued(gpu, len)
}

pub(super) fn encode_power<T: MetalElement>(
    gpu: &Gpu,
    a: &ProtocolObject<dyn MTLBuffer>,
    b: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
) -> Option<()> {
    let encoder = compute(gpu)?;
    encoder.setComputePipelineState(&gpu.kernels::<T>().power);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(a), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(b), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 2);
    }
    dispatch_1d(&encoder, len);
    queued(gpu, len)
}

pub(super) fn encode_power_scalar<T: MetalElement>(
    gpu: &Gpu,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
    scalar: f32,
    scalar_left: bool,
) -> Option<()> {
    let encoder = compute(gpu)?;
    encoder.setComputePipelineState(&gpu.kernels::<T>().power_scalar);
    let scalar_left = u32::from(scalar_left);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
        encoder.setBytes_length_atIndex(NonNull::from(&scalar).cast(), 4, 2);
        encoder.setBytes_length_atIndex(NonNull::from(&scalar_left).cast(), 4, 3);
    }
    dispatch_1d(&encoder, len);
    queued(gpu, len)
}

pub(super) fn encode_unary<T: MetalElement>(
    gpu: &Gpu,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
    op: Analytic,
) -> Option<()> {
    let encoder = compute(gpu)?;
    encoder.setComputePipelineState(&gpu.kernels::<T>().unary);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
        encoder.setBytes_length_atIndex(NonNull::from(&op).cast(), size_of::<Analytic>(), 2);
    }
    dispatch_1d(&encoder, len);
    queued(gpu, len)
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
    let encoder = compute(gpu)?;
    encoder.setComputePipelineState(&gpu.kernels::<T>().unary_dual);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(value), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(tangent), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(out_value), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(out_tangent), 0, 3);
        encoder.setBytes_length_atIndex(NonNull::from(&op).cast(), size_of::<Analytic>(), 4);
    }
    dispatch_1d(&encoder, len);
    queued(gpu, len)
}

pub(super) fn encode_fft(
    gpu: &Gpu,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    count: usize,
    inverse: bool,
) -> Option<()> {
    let count_u32 = u32::try_from(count).ok()?;
    let bits = count.trailing_zeros();
    let inverse_u32 = u32::from(inverse);

    // Every stage goes to the same compute encoder, whose dispatches run in
    // order, each seeing the one before it.
    let encoder = compute(gpu)?;
    encoder.setComputePipelineState(&gpu.fft_bit_reverse);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
        encoder.setBytes_length_atIndex(NonNull::from(&count_u32).cast(), 4, 2);
        encoder.setBytes_length_atIndex(NonNull::from(&bits).cast(), 4, 3);
    }
    dispatch_1d(&encoder, count);

    let mut stage_length = 2u32;
    while stage_length <= count_u32 {
        encoder.setComputePipelineState(&gpu.fft_stage);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(output), 0, 0);
            encoder.setBytes_length_atIndex(NonNull::from(&count_u32).cast(), 4, 1);
            encoder.setBytes_length_atIndex(NonNull::from(&stage_length).cast(), 4, 2);
            encoder.setBytes_length_atIndex(NonNull::from(&inverse_u32).cast(), 4, 3);
        }
        dispatch_1d(&encoder, count / 2);
        if stage_length == count_u32 {
            break;
        }
        stage_length *= 2;
    }
    queued(gpu, count.saturating_mul(bits as usize + 1))
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
    let encoder = compute(gpu)?;
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
    queued(
        gpu,
        (rows * cols).saturating_mul(window_rows * window_cols) / 4,
    )
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
    let encoder = compute(gpu)?;
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
    queued(gpu, (rows + 2 * pad_rows) * (cols + 2 * pad_cols))
}

pub(super) fn encode_flip<T: MetalElement>(
    gpu: &Gpu,
    input: &ProtocolObject<dyn MTLBuffer>,
    output: &ProtocolObject<dyn MTLBuffer>,
    rows: usize,
    cols: usize,
) -> Option<()> {
    let encoder = compute(gpu)?;
    encoder.setComputePipelineState(&gpu.kernels::<T>().flip_both);
    let (rows_u, cols_u) = (u32::try_from(rows).ok()?, u32::try_from(cols).ok()?);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
        encoder.setBytes_length_atIndex(NonNull::from(&rows_u).cast(), 4, 2);
        encoder.setBytes_length_atIndex(NonNull::from(&cols_u).cast(), 4, 3);
    }
    dispatch_2d(&encoder, cols, rows);
    queued(gpu, rows * cols)
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
