//! Specialized kernels for fused programs.
//!
//! The `fused_elementwise` shader interprets a program: for every element it
//! loops over the instructions, switches on each opcode, and keeps its values
//! in a register array indexed at run time — which the GPU cannot keep in
//! registers, so every operand is a trip to thread-private memory. A program
//! that runs more than once is instead compiled into a kernel of its own:
//! straight-line code, one local per value, every opcode and storage type a
//! constant the compiler folds away.
//!
//! The kernel calls the very functions the interpreter does — `fused_load`,
//! `fused_store`, `fused_remap` and the operation helpers of `common.h`, whose
//! source it embeds — with the same arguments in the same order, so it computes
//! what the interpreter computes. Its buffers are bound exactly as the
//! interpreter's are, except that slot 0 holds the program's constants rather
//! than its code: programs that differ only in their constants — an
//! optimizer's, rebuilt every step — share one compiled kernel.
//!
//! The same straight-line code serves three other kernels (see [`Kernel`]): a
//! TensorOps product whose epilogue it is, reading the product where the
//! interpreter's epilogue reads input 0; and the sums of a program's output
//! along each row or down each column, adding where it would store.
//!
//! Kernels are compiled on a program's second appearance on a thread, so a
//! program run once never waits for the compiler, and cached by their source.

use std::cell::Cell;
use std::fmt::Write as _;

use crate::tensors::fused::Encoded;

use super::MetalElement;
use super::device::Tile;

/// The header every generated kernel embeds.
const COMMON: &str = include_str!("../../metal/common.h");

/// Appearances of a program before it is compiled.
pub(super) const COMPILE_AFTER: u32 = 2;

thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(true) };
}

/// Whether fused programs on this thread may run as specialized kernels. On by
/// default; turning it off runs every program on the interpreter, for testing
/// one against the other.
#[doc(hidden)]
pub fn set_fused_codegen(enabled: bool) {
    ENABLED.with(|cell| cell.set(enabled));
}

pub(super) fn enabled() -> bool {
    ENABLED.with(Cell::get)
}

/// The Metal type a program over `T` computes in.
fn register_type<T: MetalElement>() -> &'static str {
    match T::SUFFIX {
        "f32" => "float",
        "f16" => "half",
        _ => "bfloat",
    }
}

/// What a kernel is generated for.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(super) enum Kernel {
    /// The program alone, one thread per element.
    Elementwise,
    /// The program as the epilogue of a TensorOps product computed a `tile`
    /// at a time, its input 0 each element of the product.
    Epilogue { tile: Tile, relaxed: bool },
    /// The program's one output summed along each row, never stored: one SIMD
    /// group per row, writing that row's total.
    RowSums,
    /// The program's one output summed down each column, never stored: each
    /// thread totals one column over one band of rows, for `vecmat_finish` to
    /// add up as it adds up a vector–matrix product's bands.
    ColumnSums,
    /// The program over whole rows, one threadgroup per row, which first
    /// computes the row statistics the program reads and then its elements.
    Rows(RowStatistics),
}

/// The elements of a row each thread of a [`Kernel::Rows`] kernel keeps in
/// registers between computing a mean and the deviations from it.
const ROW_CACHE: usize = 8;

/// The threads a [`Kernel::Rows`] kernel gives each row of `cols`: enough to
/// give each a few elements, in whole SIMD groups, at most `limit`.
pub(crate) fn row_threads(cols: usize, width: usize, limit: usize) -> usize {
    let width = width.max(1);
    let wanted = cols.div_ceil(4).div_ceil(width) * width;
    wanted.clamp(width, (limit / width).max(1) * width).min(256)
}

/// The row statistics a [`Kernel::Rows`] program reads: input slots `first`
/// on, each a statistic of an input of the program.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct RowStatistics {
    /// The first input slot holding a statistic.
    pub first: u8,
    /// For each statistic, in slot order: the input it is of, that input's
    /// storage type as the shader numbers it, and whether it is the sum of
    /// squared deviations rather than the mean.
    pub statistics: [(u8, u8, bool); 16],
    pub count: usize,
}

impl RowStatistics {
    fn each(&self) -> &[(u8, u8, bool)] {
        &self.statistics[..self.count]
    }
}

/// What identifies a program's kernel: its element type, what it is generated
/// for, and its instructions without their constants' values. Cheap to compute
/// on every run.
pub(super) fn key<T: MetalElement>(code: &[Encoded], kernel: Kernel) -> Vec<u64> {
    let mut key = Vec::with_capacity(code.len() + 2);
    key.push(T::INDEX as u64);
    let discriminant = match kernel {
        Kernel::Elementwise => 0,
        Kernel::RowSums => 2,
        Kernel::ColumnSums => 3,
        Kernel::Rows(rows) => 4 | u64::from(rows.first) << 8 | (rows.count as u64) << 16,
        Kernel::Epilogue {
            tile: (rows, cols, groups),
            relaxed,
        } => {
            1 | (rows as u64) << 8
                | (cols as u64) << 24
                | (groups as u64) << 40
                | u64::from(relaxed) << 56
        }
    };
    key.push(discriminant);
    if let Kernel::Rows(rows) = kernel {
        key.extend(rows.each().iter().map(|&(of, dtype, deviations)| {
            u64::from(of) | u64::from(dtype) << 8 | u64::from(deviations) << 16
        }));
    }
    for instr in code {
        key.push(
            u64::from(instr.kind)
                | u64::from(instr.op) << 16
                | u64::from(instr.dst) << 32
                | u64::from(instr.a) << 40
                | u64::from(instr.b) << 48
                | u64::from(instr.aux) << 56,
        );
    }
    key
}

/// The program's constants, in the order its kernel reads them from slot 0.
pub(super) fn constants(code: &[Encoded]) -> Vec<f32> {
    code.iter()
        .filter(|instr| instr.kind == 1)
        .map(|instr| instr.value)
        .collect()
}

/// The constants a kernel may read from slot 0, which `setBytes` caps at 4 KB.
pub(super) const MAX_CONSTANTS: usize = 1024;

/// A program's kernel source, or `None` for a program the generator does not
/// handle, which the interpreter then runs.
pub(super) fn source<T: MetalElement>(code: &[Encoded], kernel: Kernel) -> Option<String> {
    let product = matches!(kernel, Kernel::Epilogue { .. });
    let sum = matches!(kernel, Kernel::RowSums | Kernel::ColumnSums);
    let statistics = match kernel {
        Kernel::Rows(rows) => rows.first..rows.first + rows.count as u8,
        _ => 0..0,
    };
    let body = body::<T>(code, product, sum, statistics)?;
    let mut source = String::with_capacity(COMMON.len() + body.len() + 4096);
    if product {
        // The TensorOps header comes first: `common.h` is written for every
        // language version, and this one compiles as Metal 4.
        source.push_str(
            "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace mpp::tensor_ops;\n",
        );
    }
    source.push_str(COMMON);
    match kernel {
        Kernel::Elementwise => {
            source.push_str(
                "\nkernel void fused_program(\n    constant float* k [[buffer(0)]],\n    constant FusedShape& shape [[buffer(1)]],\n",
            );
            for slot in 0..16 {
                writeln!(
                    source,
                    "    device const uchar* in{slot} [[buffer({})]],",
                    2 + slot
                )
                .ok()?;
            }
            for slot in 0..8 {
                writeln!(
                    source,
                    "    device uchar* out{slot} [[buffer({})]],",
                    18 + slot
                )
                .ok()?;
            }
            source.push_str("    uint i [[thread_position_in_grid]])\n{\n");
            source
                .push_str("    const uint rows = shape.rows;\n    const uint cols = shape.cols;\n");
            // One division for the element, shared by every remapped load.
            source.push_str("    const uint row = i / cols;\n    const uint col = i - row * cols;\n");
            source.push_str("    (void)rows; (void)row; (void)col; (void)k;\n");
            source.push_str(&body);
            source.push_str("}\n");
        }
        Kernel::Rows(rows) => {
            let t = register_type::<T>();
            // A row's total across its threadgroup: each SIMD group's sum
            // through threadgroup memory, which every SIMD group then adds up
            // itself. The second barrier keeps the next total from writing
            // `shared` before every group has read it.
            source.push_str(
                r#"
inline float row_total(float x, threadgroup float* shared, uint simd, uint simds, uint lane) {
    x = simd_sum(x);
    if (lane == 0) {
        shared[simd] = x;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = simd_sum(lane < simds ? shared[lane] : 0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

kernel void fused_program(
    constant float* k [[buffer(0)]],
    constant FusedShape& shape [[buffer(1)]],
"#,
            );
            for slot in 0..16 {
                writeln!(
                    source,
                    "    device const uchar* in{slot} [[buffer({})]],",
                    2 + slot
                )
                .ok()?;
            }
            for slot in 0..8 {
                writeln!(
                    source,
                    "    device uchar* out{slot} [[buffer({})]],",
                    18 + slot
                )
                .ok()?;
            }
            write!(
                source,
                r#"    uint row [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]],
    uint n [[threads_per_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simds [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{{
    const uint rows = shape.rows;
    const uint cols = shape.cols;
    (void)k;
    (void)rows;
    threadgroup float shared[32];
    const uint first = row * cols;
"#
            )
            .ok()?;
            // Each input's moments once, as `matrix_axis_moments` computes them:
            // the mean, then the squared deviations from it, in `float`. A
            // thread keeps its first `ROW_CACHE` elements of the row in
            // registers for the second pass.
            let mut inputs: Vec<(u8, u8, bool)> = Vec::new();
            for &(of, dtype, deviations) in rows.each() {
                match inputs.iter_mut().find(|(input, ..)| *input == of) {
                    Some(entry) => entry.2 |= deviations,
                    None => inputs.push((of, dtype, deviations)),
                }
            }
            for &(of, dtype, deviations) in &inputs {
                write!(
                    source,
                    r#"    float cache{of}[{ROW_CACHE}];
    float total{of} = 0.0f;
    for (uint j = 0; j < {ROW_CACHE}; j++) {{
        const uint col = t + j * n;
        cache{of}[j] = col < cols ? fused_load(in{of}, {dtype}u, first + col) : 0.0f;
        total{of} += cache{of}[j];
    }}
    for (uint col = t + {ROW_CACHE} * n; col < cols; col += n) {{
        total{of} += fused_load(in{of}, {dtype}u, first + col);
    }}
    const float mean{of} = row_total(total{of}, shared, simd, simds, lane) / float(cols);
"#
                )
                .ok()?;
                if deviations {
                    write!(
                        source,
                        r#"    float squares{of} = 0.0f;
    for (uint j = 0; j < {ROW_CACHE}; j++) {{
        if (t + j * n < cols) {{
            const float d = cache{of}[j] - mean{of};
            squares{of} += d * d;
        }}
    }}
    for (uint col = t + {ROW_CACHE} * n; col < cols; col += n) {{
        const float d = fused_load(in{of}, {dtype}u, first + col) - mean{of};
        squares{of} += d * d;
    }}
    const float deviations{of} = row_total(squares{of}, shared, simd, simds, lane);
"#
                    )
                    .ok()?;
                }
            }
            // Rounded to the program's type, as the statistics it would
            // otherwise read are stored.
            for (k, &(of, _, deviations)) in rows.each().iter().enumerate() {
                let which = if deviations { "deviations" } else { "mean" };
                writeln!(
                    source,
                    "    const {t} stat{} = {t}({which}{of});",
                    usize::from(rows.first) + k
                )
                .ok()?;
            }
            source.push_str(
                "    for (uint col = t; col < cols; col += n) {\n        const uint i = first + col;\n",
            );
            for line in body.lines() {
                writeln!(source, "    {line}").ok()?;
            }
            source.push_str("    }\n}\n");
        }
        Kernel::RowSums | Kernel::ColumnSums => {
            source.push_str(
                "\nkernel void fused_program(\n    constant float* k [[buffer(0)]],\n    constant FusedShape& shape [[buffer(1)]],\n",
            );
            for slot in 0..16 {
                writeln!(
                    source,
                    "    device const uchar* in{slot} [[buffer({})]],",
                    2 + slot
                )
                .ok()?;
            }
            let t = register_type::<T>();
            if kernel == Kernel::RowSums {
                write!(
                    source,
                    r#"    device {t}* y [[buffer(18)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simds [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{{
    const uint rows = shape.rows;
    const uint cols = shape.cols;
    (void)k;
    const uint row = group * simds + simd;
    if (row >= rows) {{
        return;
    }}
    float acc = 0.0f;
    for (uint col = lane; col < cols; col += 32) {{
        const uint i = row * cols + col;
"#
                )
                .ok()?;
                for line in body.lines() {
                    writeln!(source, "    {line}").ok()?;
                }
                source.push_str(
                    "    }\n    acc = simd_sum(acc);\n    if (lane == 0) {\n        y[row] = ",
                );
                writeln!(source, "{t}(acc);\n    }}\n}}").ok()?;
            } else {
                source.push_str(
                    r#"    device float* partial [[buffer(18)]],
    constant uint& band [[buffer(19)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint rows = shape.rows;
    const uint cols = shape.cols;
    (void)k;
    const uint col = gid.x;
    if (col >= cols) {
        return;
    }
    const uint last = min(gid.y * band + band, rows);
    float acc = 0.0f;
    for (uint row = gid.y * band; row < last; row++) {
        const uint i = row * cols + col;
"#,
                );
                for line in body.lines() {
                    writeln!(source, "    {line}").ok()?;
                }
                source.push_str("    }\n    partial[ulong(gid.y) * cols + col] = acc;\n}\n");
            }
        }
        Kernel::Epilogue {
            tile: (tile_rows, tile_cols, groups),
            relaxed,
        } => {
            // Bound as the interpreted epilogue binds its buffers, with the
            // constants in slot 0 in place of the program.
            let t = register_type::<T>();
            writeln!(
                source,
                "\nkernel void fused_program(\n    constant float* k [[buffer(0)]],\n    constant FusedShape& shape [[buffer(1)]],\n    device {t}* A [[buffer(2)]],\n    device {t}* B [[buffer(3)]],\n    constant uint& K [[buffer(4)]],"
            )
            .ok()?;
            for slot in 1..16 {
                writeln!(
                    source,
                    "    device const uchar* in{slot} [[buffer({})]],",
                    4 + slot
                )
                .ok()?;
            }
            for slot in 0..8 {
                writeln!(
                    source,
                    "    device uchar* out{slot} [[buffer({})]],",
                    20 + slot
                )
                .ok()?;
            }
            write!(
                source,
                r#"    uint2 group [[threadgroup_position_in_grid]],
    uint2 lane [[thread_position_in_threadgroup]],
    uint2 threads [[threads_per_threadgroup]])
{{
    const uint rows = shape.rows;
    const uint cols = shape.cols;
    (void)k;
    threadgroup float staging[{tile_rows} * {tile_cols}];
    constexpr auto descriptor = matmul2d_descriptor(
        {tile_rows}, {tile_cols}, static_cast<int>(dynamic_extent), false, false, {relaxed});
    auto tensor_a = tensor(A, dextents<int32_t, 2>{{int32_t(K), int32_t(rows)}},
                           array<int32_t, 2>{{1, int32_t(K)}});
    auto tensor_b = tensor(B, dextents<int32_t, 2>{{int32_t(cols), int32_t(K)}},
                           array<int32_t, 2>{{1, int32_t(cols)}});
    auto staged = tensor(staging, dextents<int32_t, 2>{{{tile_cols}, {tile_rows}}},
                         array<int32_t, 2>{{1, {tile_cols}}});
    matmul2d<descriptor, execution_simdgroups<{groups}>> operation;
    auto tile_a = tensor_a.slice(0, int32_t(group.y) * {tile_rows});
    auto tile_b = tensor_b.slice(int32_t(group.x) * {tile_cols}, 0);
    auto accumulator = operation.template get_destination_cooperative_tensor<
        decltype(tile_a), decltype(tile_b), float>();
    operation.run(tile_a, tile_b, accumulator);
    accumulator.store(staged);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lane.x; e < {tile_rows} * {tile_cols}; e += threads.x) {{
        const uint row = group.y * {tile_rows} + e / {tile_cols};
        const uint col = group.x * {tile_cols} + e % {tile_cols};
        if (row >= rows || col >= cols) {{
            continue;
        }}
        const uint i = row * cols + col;
        const float product = staging[e];
"#
            )
            .ok()?;
            for line in body.lines() {
                writeln!(source, "    {line}").ok()?;
            }
            source.push_str("    }\n}\n");
        }
    }
    Some(source)
}

/// The straight-line code for one element `i`: one local per value, the
/// constants read from `k`. With `product`, input 0 is the local `product`;
/// with `sum`, the one store adds to the local `acc` instead; and the input
/// slots in `statistics` are the locals `stat{slot}`.
fn body<T: MetalElement>(
    code: &[Encoded],
    product: bool,
    sum: bool,
    statistics: std::ops::Range<u8>,
) -> Option<String> {
    let t = register_type::<T>();
    let mut body = String::new();
    let mut constants = 0usize;
    // The local holding each register's current value.
    let mut local: [Option<usize>; 16] = [None; 16];
    let read = |reg: u8, local: &[Option<usize>; 16]| -> Option<String> {
        local
            .get(usize::from(reg))
            .copied()
            .flatten()
            .map(|v| format!("v{v}"))
    };
    for (at, instr) in code.iter().enumerate() {
        let value = match instr.kind {
            // Load: `op` is the remap, `a` the input slot, `b` its storage type.
            0 if product && instr.a == 0 => format!("{t}(product)"),
            // A row statistic, computed above, and read through a column
            // broadcast: the same for the whole row.
            0 if statistics.contains(&instr.a) => format!("stat{}", instr.a),
            // The index a remap reads, from the element's row and column,
            // which every kernel knows without dividing again per load.
            0 => {
                let index = match instr.op {
                    1 => "col * rows + row",
                    2 => "col",
                    3 => "row",
                    _ => "i",
                };
                format!("{t}(fused_load(in{}, {}u, {index}))", instr.a, instr.b)
            }
            1 => {
                constants += 1;
                format!("{t}(k[{}])", constants - 1)
            }
            2 => format!(
                "binary_values(BinaryOp({}), {}, {})",
                instr.op,
                read(instr.a, &local)?,
                read(instr.b, &local)?
            ),
            3 => format!(
                "analytic_value(AnalyticOp({}), {})",
                instr.op,
                read(instr.a, &local)?
            ),
            4 => format!(
                "compare_values(CompareOp({}), {}, {})",
                instr.op,
                read(instr.a, &local)?,
                read(instr.b, &local)?
            ),
            5 if sum => {
                // The output is of the program's own type, so the stored value
                // is the register's.
                writeln!(body, "    acc += float({});", read(instr.a, &local)?).ok()?;
                continue;
            }
            5 => {
                // Store: `a` is the register, `aux` the output slot, `b` its
                // storage type.
                writeln!(
                    body,
                    "    fused_store(out{}, {}u, i, float({}));",
                    instr.aux,
                    instr.b,
                    read(instr.a, &local)?
                )
                .ok()?;
                continue;
            }
            _ => return None,
        };
        writeln!(body, "    const {t} v{at} = {value};").ok()?;
        *local.get_mut(usize::from(instr.dst))? = Some(at);
    }
    if constants > MAX_CONSTANTS {
        return None;
    }
    Some(body)
}
