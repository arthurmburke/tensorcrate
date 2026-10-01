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
//! Kernels are compiled on a program's second appearance on a thread, so a
//! program run once never waits for the compiler, and cached by their source.

use std::cell::Cell;
use std::fmt::Write as _;

use crate::tensors::fused::Encoded;

use super::MetalElement;

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

/// What identifies a program's kernel: its element type and its instructions
/// without their constants' values. Cheap to compute on every run.
pub(super) fn key<T: MetalElement>(code: &[Encoded]) -> Vec<u64> {
    let mut key = Vec::with_capacity(code.len() + 1);
    key.push(T::INDEX as u64);
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
pub(super) fn source<T: MetalElement>(code: &[Encoded]) -> Option<String> {
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
            0 => format!(
                "{t}(fused_load(in{}, {}u, fused_remap({}, i, rows, cols)))",
                instr.a, instr.b, instr.op
            ),
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

    let mut source = String::with_capacity(COMMON.len() + body.len() + 2048);
    source.push_str(COMMON);
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
    source.push_str("    const uint rows = shape.rows;\n    const uint cols = shape.cols;\n");
    source.push_str("    (void)rows; (void)cols; (void)k;\n");
    source.push_str(&body);
    source.push_str("}\n");
    Some(source)
}
