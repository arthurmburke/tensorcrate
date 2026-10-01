//! Fused elementwise programs: many elementwise operations, one kernel.
//!
//! Every [`Kernels`] operation reads its operands from memory and writes a fresh
//! result back. A chain of them — the fourteen that make one [`Adam`] step, say —
//! therefore streams the whole tensor through memory once per link, and on Metal
//! pays a command buffer per link as well. A [`Program`] is the same chain as a
//! short register program that runs once per element: intermediates live in
//! registers, only the program's real inputs are read and only its real outputs
//! are written, and the whole chain costs one kernel.
//!
//! [`Adam`]: crate::optim::Adam
//!
//! # Writing a program
//!
//! [`Builder`] is the usual way in. It hands out [`Value`]s, which are
//! single-assignment and allocates registers itself:
//!
//! ```
//! use tensorcrate::tensors::fused::{Builder, DType};
//! use tensorcrate::tensors::{Analytic, Vector};
//!
//! // y = sqrt(a·b + 1), in one pass.
//! let mut b = Builder::new();
//! let (x, w) = (b.input(DType::F32), b.input(DType::F32));
//! let product = b.mul(x, w);
//! let one = b.constant(1.0);
//! let shifted = b.add(product, one);
//! let root = b.unary(Analytic::Sqrt, shifted);
//! b.output(root, DType::F32);
//! let program = b.build().unwrap();
//!
//! let a = Vector::new([3.0f32, 0.0]);
//! let w = Vector::new([1.0f32, 5.0]);
//! let y = program.run_vectors(&[&a, &w]).remove(0);
//! assert_eq!(y.as_slice(), [2.0, 1.0]);
//! ```
//!
//! [`Program::new`] takes the instructions directly, for when the register
//! assignment matters or the program comes from somewhere else.
//!
//! # Exactness
//!
//! A program performs, for every element, exactly the `f32` operations the
//! unfused chain would, in the same order. On the [`Host`] backend its result is
//! therefore identical to the unfused one bit for bit — [`Mode::Unfused`]
//! exists to check that, and the tests do. (The one gap is the sign of a zero
//! from `Min`/`Max` when `−0.0` meets `+0.0`, which the unfused kernels do not
//! pin down either; see [`Compare`].)
//!
//! Metal builds its shaders with the compiler's fast-math defaults, fused ones
//! included, and inside one kernel the compiler may contract `a·b + c` into a
//! fused multiply-add across what used to be separate kernels. Metal results
//! therefore agree with the host within the usual tolerance rather than
//! exactly — the same promise the unfused Metal kernels make.
//!
//! # Debugging
//!
//! - A program prints as a disassembly (`{program}`).
//! - [`Program::trace`] runs it unfused, one existing kernel per instruction,
//!   and returns every intermediate as a tensor.
//! - [`set_mode`] / [`with_mode`] turn fusion off for everything on this thread
//!   that runs a program — the optimizers included — so a suspected fusion bug
//!   can be confirmed or ruled out without changing the code under test.
//!
//! # Storage types
//!
//! Every input and output has its own [`DType`]. Loads widen to `f32`, stores
//! narrow from it, and all arithmetic is `f32`; `f16` and `bf16` only change
//! how many bytes cross memory.

use std::cell::Cell;
use std::fmt;

use half::{bf16, f16};

use super::{Analytic, Backend, BinaryOp, Compare, Host, Kernels, Matrix, Vector};
use crate::counters;

/// Registers a program may use. Metal keeps them in a per-thread array, so this
/// is also a bound on the shader's register pressure.
pub const REGISTERS: usize = 16;

/// Inputs a program may read, counting the tensors it updates in place.
pub const MAX_INPUTS: usize = 16;

/// Outputs a program may write, counting the tensors it updates in place.
///
/// With [`MAX_INPUTS`] and the two slots for the program itself, this stays
/// under Metal's 31 buffer arguments.
pub const MAX_OUTPUTS: usize = 8;

/// Instructions in one program. The Metal shader receives the program as
/// inline constant data, which is capped at 4 KB.
pub const MAX_INSTRUCTIONS: usize = 256;

/// A register index, below [`REGISTERS`].
pub type Reg = u8;

/// How an input or output is stored. Arithmetic is always `f32`.
///
/// The representation is part of the Metal shader ABI: keep the discriminants
/// stable and only append.
#[repr(u16)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32 = 0,
    F16 = 1,
    Bf16 = 2,
}

impl DType {
    /// Bytes per stored element.
    pub fn size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::Bf16 => 2,
        }
    }

    fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::Bf16 => "bf16",
        }
    }
}

/// Where a load reads, relative to the element being computed.
///
/// A program iterates over a `rows × cols` space in row-major order, and element
/// `(r, c)` of each input comes from:
///
/// The representation is part of the Metal shader ABI.
#[repr(u16)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Remap {
    /// `(r, c)` of a `rows × cols` input.
    Identity = 0,
    /// `(c, r)` of a `cols × rows` input — a transpose that is never
    /// materialized.
    Transpose = 1,
    /// Element `c` of a `cols`-long vector, repeated down every row.
    Row = 2,
    /// Element `r` of a `rows`-long vector, repeated across every column.
    Column = 3,
}

impl Remap {
    /// How many elements an input read this way must hold.
    pub fn input_len(self, (rows, cols): (usize, usize)) -> usize {
        match self {
            Remap::Identity | Remap::Transpose => rows * cols,
            Remap::Row => cols,
            Remap::Column => rows,
        }
    }

    /// The input element that feeds output element `index`.
    #[inline]
    fn source(self, index: usize, (rows, cols): (usize, usize)) -> usize {
        match self {
            Remap::Identity => index,
            Remap::Transpose => (index % cols) * rows + index / cols,
            Remap::Row => index % cols,
            Remap::Column => index / cols,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Remap::Identity => "",
            Remap::Transpose => ".T",
            Remap::Row => ".row",
            Remap::Column => ".col",
        }
    }
}

/// One step of a program. Every operation code is one of the existing kernel
/// enums, so an instruction means exactly what the matching kernel does.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Instr {
    /// `dst ← input[remap(i)]`, widened to `f32`.
    Load { dst: Reg, input: u8, remap: Remap },
    /// `dst ← value`, the same for every element.
    Const { dst: Reg, value: f32 },
    /// `dst ← a op b`.
    Binary {
        dst: Reg,
        op: BinaryOp,
        a: Reg,
        b: Reg,
    },
    /// `dst ← op(a)`.
    Unary { dst: Reg, op: Analytic, a: Reg },
    /// `dst ← op(a, b)`, including the `1.0`/`0.0` predicates.
    Cmp {
        dst: Reg,
        op: Compare,
        a: Reg,
        b: Reg,
    },
    /// `output[i] ← src`, narrowed to the output's storage type.
    Store { src: Reg, output: u8 },
}

impl Instr {
    /// The register written, if any.
    fn dst(&self) -> Option<Reg> {
        match *self {
            Instr::Load { dst, .. }
            | Instr::Const { dst, .. }
            | Instr::Binary { dst, .. }
            | Instr::Unary { dst, .. }
            | Instr::Cmp { dst, .. } => Some(dst),
            Instr::Store { .. } => None,
        }
    }

    /// The registers read.
    fn sources(&self) -> impl Iterator<Item = Reg> {
        let (a, b) = match *self {
            Instr::Binary { a, b, .. } | Instr::Cmp { a, b, .. } => (Some(a), Some(b)),
            Instr::Unary { a, .. } => (Some(a), None),
            Instr::Store { src, .. } => (Some(src), None),
            Instr::Load { .. } | Instr::Const { .. } => (None, None),
        };
        a.into_iter().chain(b)
    }
}

/// Why a program was rejected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProgramError {
    TooManyInstructions(usize),
    TooManyInputs(usize),
    TooManyOutputs(usize),
    /// More registers are live at once than [`REGISTERS`].
    TooManyRegisters,
    /// Instruction `at` names a register outside [`REGISTERS`].
    BadRegister {
        at: usize,
    },
    /// Instruction `at` reads a register nothing has written.
    Undefined {
        at: usize,
        reg: Reg,
    },
    /// Instruction `at` loads an input slot the program does not declare.
    BadInput {
        at: usize,
    },
    /// Instruction `at` stores to an output slot the program does not declare.
    BadOutput {
        at: usize,
    },
    /// An output is never stored, or stored twice.
    OutputNotStoredOnce {
        output: u8,
    },
    /// An in-place tensor is read through a remap, which would read elements
    /// other threads may already have overwritten.
    RemappedUpdate {
        at: usize,
    },
    /// An in-place tensor is read after its new value was stored.
    LoadAfterStore {
        at: usize,
    },
    /// An in-place tensor's input and output storage types differ.
    UpdateTypeMismatch {
        slot: usize,
    },
    /// More in-place tensors than inputs or outputs.
    BadUpdateCount,
    /// A program with no outputs does nothing.
    NoOutputs,
}

impl fmt::Display for ProgramError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProgramError::TooManyInstructions(n) => {
                write!(f, "{n} instructions, over the {MAX_INSTRUCTIONS} limit")
            }
            ProgramError::TooManyInputs(n) => write!(f, "{n} inputs, over the {MAX_INPUTS} limit"),
            ProgramError::TooManyOutputs(n) => {
                write!(f, "{n} outputs, over the {MAX_OUTPUTS} limit")
            }
            ProgramError::TooManyRegisters => {
                write!(f, "more than {REGISTERS} values are live at once")
            }
            ProgramError::BadRegister { at } => {
                write!(f, "instruction {at}: register out of range")
            }
            ProgramError::Undefined { at, reg } => {
                write!(f, "instruction {at}: r{reg} is read before it is written")
            }
            ProgramError::BadInput { at } => write!(f, "instruction {at}: no such input"),
            ProgramError::BadOutput { at } => write!(f, "instruction {at}: no such output"),
            ProgramError::OutputNotStoredOnce { output } => {
                write!(f, "output {output} must be stored exactly once")
            }
            ProgramError::RemappedUpdate { at } => {
                write!(
                    f,
                    "instruction {at}: an in-place tensor can only be loaded unremapped"
                )
            }
            ProgramError::LoadAfterStore { at } => {
                write!(
                    f,
                    "instruction {at}: an in-place tensor is loaded after it is stored"
                )
            }
            ProgramError::UpdateTypeMismatch { slot } => {
                write!(
                    f,
                    "in-place tensor {slot} is read and written as different types"
                )
            }
            ProgramError::BadUpdateCount => {
                write!(f, "more in-place tensors than inputs or outputs")
            }
            ProgramError::NoOutputs => write!(f, "the program has no outputs"),
        }
    }
}

impl std::error::Error for ProgramError {}

/// A validated elementwise program.
///
/// Its inputs are numbered `0..inputs().len()`, and the last
/// [`updated`](Self::updated) of them are tensors the program also *writes*, in
/// place: they are outputs `0..updated` as well, followed by any fresh outputs.
/// That is how an optimizer step overwrites its parameters and moments rather
/// than allocating new ones.
#[derive(Clone, Debug, PartialEq)]
pub struct Program {
    code: Vec<Instr>,
    inputs: Vec<DType>,
    outputs: Vec<DType>,
    updated: usize,
    registers: usize,
}

impl Program {
    /// Check a hand-written program.
    ///
    /// `inputs` and `outputs` give each slot's storage type; the last `updated`
    /// inputs and the first `updated` outputs are the same in-place tensors.
    pub fn new(
        code: Vec<Instr>,
        inputs: Vec<DType>,
        outputs: Vec<DType>,
        updated: usize,
    ) -> Result<Self, ProgramError> {
        if code.len() > MAX_INSTRUCTIONS {
            return Err(ProgramError::TooManyInstructions(code.len()));
        }
        if inputs.len() > MAX_INPUTS {
            return Err(ProgramError::TooManyInputs(inputs.len()));
        }
        if outputs.len() > MAX_OUTPUTS {
            return Err(ProgramError::TooManyOutputs(outputs.len()));
        }
        if outputs.is_empty() {
            return Err(ProgramError::NoOutputs);
        }
        if updated > inputs.len() || updated > outputs.len() {
            return Err(ProgramError::BadUpdateCount);
        }
        let fresh_inputs = inputs.len() - updated;
        for slot in 0..updated {
            if inputs[fresh_inputs + slot] != outputs[slot] {
                return Err(ProgramError::UpdateTypeMismatch { slot });
            }
        }

        let mut defined = [false; REGISTERS];
        let mut stored = vec![0usize; outputs.len()];
        let mut registers = 0;
        for (at, instr) in code.iter().enumerate() {
            for reg in instr.sources() {
                if usize::from(reg) >= REGISTERS {
                    return Err(ProgramError::BadRegister { at });
                }
                if !defined[usize::from(reg)] {
                    return Err(ProgramError::Undefined { at, reg });
                }
            }
            match *instr {
                Instr::Load { input, remap, .. } => {
                    let input = usize::from(input);
                    if input >= inputs.len() {
                        return Err(ProgramError::BadInput { at });
                    }
                    if input >= fresh_inputs {
                        if remap != Remap::Identity {
                            return Err(ProgramError::RemappedUpdate { at });
                        }
                        if stored[input - fresh_inputs] > 0 {
                            return Err(ProgramError::LoadAfterStore { at });
                        }
                    }
                }
                Instr::Store { output, .. } => {
                    let output = usize::from(output);
                    if output >= outputs.len() {
                        return Err(ProgramError::BadOutput { at });
                    }
                    stored[output] += 1;
                }
                _ => {}
            }
            if let Some(dst) = instr.dst() {
                if usize::from(dst) >= REGISTERS {
                    return Err(ProgramError::BadRegister { at });
                }
                defined[usize::from(dst)] = true;
                registers = registers.max(usize::from(dst) + 1);
            }
        }
        if let Some(output) = stored.iter().position(|&count| count != 1) {
            return Err(ProgramError::OutputNotStoredOnce {
                output: output as u8,
            });
        }

        Ok(Program {
            code,
            inputs,
            outputs,
            updated,
            registers,
        })
    }

    /// The instructions.
    pub fn code(&self) -> &[Instr] {
        &self.code
    }

    /// Every input slot's storage type, in-place tensors last.
    pub fn inputs(&self) -> &[DType] {
        &self.inputs
    }

    /// Every output slot's storage type, in-place tensors first.
    pub fn outputs(&self) -> &[DType] {
        &self.outputs
    }

    /// How many tensors the program updates in place.
    pub fn updated(&self) -> usize {
        self.updated
    }

    /// Inputs that are only read.
    pub fn fresh_inputs(&self) -> usize {
        self.inputs.len() - self.updated
    }

    /// Outputs allocated by each run.
    pub fn fresh_outputs(&self) -> usize {
        self.outputs.len() - self.updated
    }

    /// One more than the highest register written.
    pub fn registers(&self) -> usize {
        self.registers
    }

    /// Bytes one run over `len` elements moves: every input read once, every
    /// output written once. Remapped broadcasts are charged their full length,
    /// since that is the number of loads, even though most hit cache.
    pub fn bytes(&self, len: usize) -> usize {
        let read: usize = self.inputs.iter().map(|dtype| dtype.size()).sum();
        let written: usize = self.outputs.iter().map(|dtype| dtype.size()).sum();
        (read + written) * len
    }

    /// Run the program over a `rows × cols` iteration space.
    ///
    /// `inputs` are the read-only inputs, in slot order; `updated` are the
    /// in-place tensors, which are read as the last input slots and overwritten
    /// as the first output slots. The fresh outputs come back in slot order.
    ///
    /// # Panics
    ///
    /// If the counts, storage types or lengths disagree with the program and
    /// its remaps.
    #[track_caller]
    pub fn run<B: Kernels>(
        &self,
        shape: (usize, usize),
        inputs: &[&dyn Fusable<B>],
        updated: &mut [&mut dyn Fusable<B>],
    ) -> Vec<Output<B>> {
        assert_eq!(
            inputs.len(),
            self.fresh_inputs(),
            "fused program: expected {} inputs, got {}",
            self.fresh_inputs(),
            inputs.len()
        );
        assert_eq!(
            updated.len(),
            self.updated,
            "fused program: expected {} in-place tensors, got {}",
            self.updated,
            updated.len()
        );
        let len = shape.0 * shape.1;
        let sources: Vec<Source<'_, B>> = inputs.iter().map(|input| input.source()).collect();
        for (slot, source) in sources.iter().enumerate() {
            self.check_input(slot, source.dtype(), source.len, shape);
        }
        let mut sinks: Vec<Sink<'_, B>> = updated.iter_mut().map(|target| target.sink()).collect();
        for (slot, sink) in sinks.iter().enumerate() {
            self.check_input(self.fresh_inputs() + slot, sink.dtype(), sink.len, shape);
        }

        let fresh = match mode() {
            Mode::Fused => {
                counters::kernel(self.bytes(len), self.fresh_outputs());
                B::fused(self, shape, &sources, &mut sinks)
            }
            Mode::Unfused => unfused(self, shape, &sources, &mut sinks),
        };
        fresh
            .into_iter()
            .map(|data| Output { shape, data })
            .collect()
    }

    /// [`run`](Self::run) for the common case: `f32` vectors of one length,
    /// nothing updated in place, every output `f32`.
    #[track_caller]
    pub fn run_vectors<B: Kernels>(&self, inputs: &[&Vector<f32, B>]) -> Vec<Vector<f32, B>> {
        let len = inputs.first().map_or(0, |input| input.len());
        let inputs: Vec<&dyn Fusable<B>> = inputs.iter().map(|&v| v as &dyn Fusable<B>).collect();
        self.run((1, len), &inputs, &mut [])
            .into_iter()
            .map(Output::into_vector)
            .collect()
    }

    /// Run the program unfused — one existing kernel per instruction — and keep
    /// every intermediate.
    ///
    /// `inputs` covers every input slot, in-place tensors included; nothing is
    /// written. Step `k` of the result holds the value instruction `k` produced
    /// (as `f32`, before any narrowing), or `None` for a store.
    #[track_caller]
    pub fn trace<B: Kernels>(
        &self,
        shape: (usize, usize),
        inputs: &[&dyn Fusable<B>],
    ) -> Vec<Option<Matrix<f32, B>>> {
        assert_eq!(
            inputs.len(),
            self.inputs.len(),
            "fused trace: expected {} inputs, got {}",
            self.inputs.len(),
            inputs.len()
        );
        let sources: Vec<Source<'_, B>> = inputs.iter().map(|input| input.source()).collect();
        for (slot, source) in sources.iter().enumerate() {
            self.check_input(slot, source.dtype(), source.len, shape);
        }
        let mut registers: Vec<Option<Register<B>>> = (0..REGISTERS).map(|_| None).collect();
        self.code
            .iter()
            .map(|instr| {
                step::<B>(instr, &mut registers, |slot, remap| {
                    load_unfused(&sources[slot], shape, remap)
                })
                .map(|value| value.materialize(shape))
            })
            .collect()
    }

    #[track_caller]
    fn check_input(&self, slot: usize, dtype: DType, len: usize, shape: (usize, usize)) {
        assert_eq!(
            dtype,
            self.inputs[slot],
            "fused program: input {slot} is {} but the program reads {}",
            dtype.name(),
            self.inputs[slot].name()
        );
        for instr in &self.code {
            if let Instr::Load { input, remap, .. } = *instr
                && usize::from(input) == slot
            {
                let expected = remap.input_len(shape);
                assert_eq!(
                    len, expected,
                    "fused program: input {slot} holds {len} elements, but a {}×{} space \
                     read through {remap:?} needs {expected}",
                    shape.0, shape.1
                );
            }
        }
        if slot >= self.fresh_inputs() {
            assert_eq!(
                len,
                shape.0 * shape.1,
                "fused program: in-place tensor {} holds {len} elements, not {}",
                slot - self.fresh_inputs(),
                shape.0 * shape.1
            );
        }
    }

    /// The fixed-width encoding the Metal shader interprets.
    #[cfg_attr(not(all(feature = "metal", target_os = "macos")), allow(dead_code))]
    pub(crate) fn encode(&self) -> Vec<Encoded> {
        self.code
            .iter()
            .map(|instr| match *instr {
                Instr::Load { dst, input, remap } => Encoded {
                    kind: 0,
                    op: remap as u16,
                    dst,
                    a: input,
                    b: self.inputs[usize::from(input)] as u8,
                    aux: 0,
                    value: 0.0,
                },
                Instr::Const { dst, value } => Encoded {
                    kind: 1,
                    op: 0,
                    dst,
                    a: 0,
                    b: 0,
                    aux: 0,
                    value,
                },
                Instr::Binary { dst, op, a, b } => Encoded {
                    kind: 2,
                    op: op as u16,
                    dst,
                    a,
                    b,
                    aux: 0,
                    value: 0.0,
                },
                Instr::Unary { dst, op, a } => Encoded {
                    kind: 3,
                    op: op as u16,
                    dst,
                    a,
                    b: 0,
                    aux: 0,
                    value: 0.0,
                },
                Instr::Cmp { dst, op, a, b } => Encoded {
                    kind: 4,
                    op: op as u16,
                    dst,
                    a,
                    b,
                    aux: 0,
                    value: 0.0,
                },
                Instr::Store { src, output } => Encoded {
                    kind: 5,
                    op: 0,
                    dst: 0,
                    a: src,
                    b: self.outputs[usize::from(output)] as u8,
                    aux: output,
                    value: 0.0,
                },
            })
            .collect()
    }

    /// Whether every instruction has a Metal implementation. The shaders have
    /// no remainder, matching the unfused kernels.
    #[cfg_attr(not(all(feature = "metal", target_os = "macos")), allow(dead_code))]
    pub(crate) fn runs_on_metal(&self) -> bool {
        !self.code.iter().any(|instr| {
            matches!(
                instr,
                Instr::Binary {
                    op: BinaryOp::Rem,
                    ..
                }
            )
        })
    }
}

/// An instruction as the Metal shader reads it: twelve bytes, four-aligned.
///
/// | kind | op          | dst | a      | b      | aux    | value |
/// |------|-------------|-----|--------|--------|--------|-------|
/// | 0 load  | remap    | reg | input  | dtype  |        |       |
/// | 1 const |          | reg |        |        |        | value |
/// | 2 binary| BinaryOp | reg | reg    | reg    |        |       |
/// | 3 unary | Analytic | reg | reg    |        |        |       |
/// | 4 cmp   | Compare  | reg | reg    | reg    |        |       |
/// | 5 store |          |     | reg    | dtype  | output |       |
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(not(all(feature = "metal", target_os = "macos")), allow(dead_code))]
pub(crate) struct Encoded {
    pub kind: u16,
    pub op: u16,
    pub dst: u8,
    pub a: u8,
    pub b: u8,
    pub aux: u8,
    pub value: f32,
}

impl fmt::Display for Program {
    /// A disassembly: one instruction per line.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let types = |types: &[DType]| {
            types
                .iter()
                .map(|dtype| dtype.name())
                .collect::<Vec<_>>()
                .join(", ")
        };
        writeln!(
            f,
            "program({}) -> ({}), {} in place, {} registers",
            types(&self.inputs),
            types(&self.outputs),
            self.updated,
            self.registers
        )?;
        for (at, instr) in self.code.iter().enumerate() {
            write!(f, "  {at:>3}: ")?;
            match *instr {
                Instr::Load { dst, input, remap } => {
                    writeln!(f, "r{dst} = in{input}{}", remap.name())?
                }
                Instr::Const { dst, value } => writeln!(f, "r{dst} = {value:?}")?,
                Instr::Binary { dst, op, a, b } => {
                    writeln!(f, "r{dst} = {} r{a}, r{b}", op.name())?
                }
                Instr::Unary { dst, op, a } => writeln!(f, "r{dst} = {op:?} r{a}")?,
                Instr::Cmp { dst, op, a, b } => writeln!(f, "r{dst} = {op:?} r{a}, r{b}")?,
                Instr::Store { src, output } => writeln!(f, "out{output} = r{src}")?,
            }
        }
        Ok(())
    }
}

// ---- building programs --------------------------------------------------------

/// A value inside a [`Builder`]: written once, read any number of times.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Value(usize);

#[derive(Copy, Clone, Debug)]
enum Node {
    Load(u8, Remap),
    Const(f32),
    Binary(BinaryOp, Value, Value),
    Unary(Analytic, Value),
    Cmp(Compare, Value, Value),
}

/// Assembles a [`Program`] from single-assignment values, allocating registers.
///
/// Declare read-only inputs with [`input`](Self::input), in-place tensors with
/// [`update`](Self::update) — which returns the tensor's current value, to be
/// given its new one with [`set`](Self::set) — and fresh outputs with
/// [`output`](Self::output). The operations are emitted in the order they were
/// built; loads of the same slot and remap are shared.
#[derive(Clone, Debug, Default)]
pub struct Builder {
    nodes: Vec<Node>,
    inputs: Vec<DType>,
    updates: Vec<(DType, Option<Value>)>,
    outputs: Vec<(DType, Value)>,
    loads: Vec<((u8, Remap), Value)>,
}

impl Builder {
    pub fn new() -> Self {
        Self::default()
    }

    fn push(&mut self, node: Node) -> Value {
        self.nodes.push(node);
        Value(self.nodes.len() - 1)
    }

    /// Declare a read-only input and load it unremapped.
    ///
    /// Inputs are numbered in declaration order.
    pub fn input(&mut self, dtype: DType) -> Value {
        self.input_remapped(dtype, Remap::Identity)
    }

    /// Declare a read-only input read through `remap`.
    pub fn input_remapped(&mut self, dtype: DType, remap: Remap) -> Value {
        self.inputs.push(dtype);
        let slot = (self.inputs.len() - 1) as u8;
        self.load(slot, remap)
    }

    /// Declare a tensor updated in place and return its current value.
    pub fn update(&mut self, dtype: DType) -> Value {
        self.updates.push((dtype, None));
        // The final slot number is only known once every input is declared, so
        // update slots are numbered from the top of the range and renumbered in
        // `build`.
        let slot = (MAX_INPUTS + self.updates.len() - 1) as u8;
        self.load(slot, Remap::Identity)
    }

    /// Give in-place tensor `index` (in [`update`](Self::update) order) its
    /// new value.
    pub fn set(&mut self, index: usize, value: Value) {
        self.updates[index].1 = Some(value);
    }

    /// Store `value` to a fresh output of type `dtype`. Outputs are numbered in
    /// declaration order, after the in-place tensors.
    pub fn output(&mut self, value: Value, dtype: DType) {
        self.outputs.push((dtype, value));
    }

    fn load(&mut self, slot: u8, remap: Remap) -> Value {
        if let Some(&(_, value)) = self.loads.iter().find(|(key, _)| *key == (slot, remap)) {
            return value;
        }
        let value = self.push(Node::Load(slot, remap));
        self.loads.push(((slot, remap), value));
        value
    }

    pub fn constant(&mut self, value: f32) -> Value {
        self.push(Node::Const(value))
    }

    pub fn binary(&mut self, op: BinaryOp, a: Value, b: Value) -> Value {
        self.push(Node::Binary(op, a, b))
    }

    pub fn add(&mut self, a: Value, b: Value) -> Value {
        self.binary(BinaryOp::Add, a, b)
    }

    pub fn sub(&mut self, a: Value, b: Value) -> Value {
        self.binary(BinaryOp::Sub, a, b)
    }

    pub fn mul(&mut self, a: Value, b: Value) -> Value {
        self.binary(BinaryOp::Mul, a, b)
    }

    pub fn div(&mut self, a: Value, b: Value) -> Value {
        self.binary(BinaryOp::Div, a, b)
    }

    /// `a · factor`, the fused form of a scalar broadcast.
    pub fn scale(&mut self, a: Value, factor: f32) -> Value {
        let factor = self.constant(factor);
        self.mul(a, factor)
    }

    /// `a + offset`.
    pub fn shift(&mut self, a: Value, offset: f32) -> Value {
        let offset = self.constant(offset);
        self.add(a, offset)
    }

    pub fn unary(&mut self, op: Analytic, a: Value) -> Value {
        self.push(Node::Unary(op, a))
    }

    pub fn compare(&mut self, op: Compare, a: Value, b: Value) -> Value {
        self.push(Node::Cmp(op, a, b))
    }

    /// Allocate registers and validate.
    ///
    /// A register is freed after the last instruction that reads its value, so
    /// a long chain needs only as many registers as it has values live at once.
    pub fn build(self) -> Result<Program, ProgramError> {
        let fresh_inputs = self.inputs.len();
        let slot_of = |slot: u8| -> u8 {
            if usize::from(slot) >= MAX_INPUTS {
                (fresh_inputs + usize::from(slot) - MAX_INPUTS) as u8
            } else {
                slot
            }
        };

        // Stores, in-place tensors first.
        let mut stores: Vec<(Value, u8)> = Vec::new();
        for (index, (_, value)) in self.updates.iter().enumerate() {
            let value = value.unwrap_or(Value(usize::MAX));
            if value.0 == usize::MAX {
                return Err(ProgramError::OutputNotStoredOnce {
                    output: index as u8,
                });
            }
            stores.push((value, index as u8));
        }
        for (index, &(_, value)) in self.outputs.iter().enumerate() {
            stores.push((value, (self.updates.len() + index) as u8));
        }

        // The last node reading each value; a stored value lives to the end.
        let end = self.nodes.len();
        let mut last_use = vec![None::<usize>; self.nodes.len()];
        for (at, node) in self.nodes.iter().enumerate() {
            let operands: &[Value] = match node {
                Node::Binary(_, a, b) | Node::Cmp(_, a, b) => &[*a, *b],
                Node::Unary(_, a) => &[*a],
                Node::Load(..) | Node::Const(_) => &[],
            };
            for operand in operands {
                last_use[operand.0] = Some(at);
            }
        }
        for &(value, _) in &stores {
            last_use[value.0] = Some(end);
        }

        let mut free: Vec<Reg> = (0..REGISTERS as u8).rev().collect();
        let mut assigned: Vec<Option<Reg>> = vec![None; self.nodes.len()];
        let mut code = Vec::with_capacity(self.nodes.len() + stores.len());
        for (at, node) in self.nodes.iter().enumerate() {
            // Dead code: nothing reads it and it is not stored.
            if last_use[at].is_none() {
                continue;
            }
            let reg = |value: Value| assigned[value.0].expect("operands precede their uses");
            let instr = |dst| match *node {
                Node::Load(slot, remap) => Instr::Load {
                    dst,
                    input: slot_of(slot),
                    remap,
                },
                Node::Const(value) => Instr::Const { dst, value },
                Node::Binary(op, a, b) => Instr::Binary {
                    dst,
                    op,
                    a: reg(a),
                    b: reg(b),
                },
                Node::Unary(op, a) => Instr::Unary { dst, op, a: reg(a) },
                Node::Cmp(op, a, b) => Instr::Cmp {
                    dst,
                    op,
                    a: reg(a),
                    b: reg(b),
                },
            };
            // Operands whose last use is here can hand their register to the
            // result: every backend reads an instruction's operands before it
            // writes the destination.
            let operands: Vec<Value> = match *node {
                Node::Binary(_, a, b) | Node::Cmp(_, a, b) => vec![a, b],
                Node::Unary(_, a) => vec![a],
                Node::Load(..) | Node::Const(_) => vec![],
            };
            let mut released = Vec::new();
            for operand in operands {
                if last_use[operand.0] == Some(at) {
                    let reg = reg(operand);
                    if !released.contains(&reg) {
                        released.push(reg);
                    }
                }
            }
            let dst = match released.first() {
                Some(&reg) => reg,
                None => free.pop().ok_or(ProgramError::TooManyRegisters)?,
            };
            free.extend(released.iter().skip(1));
            code.push(instr(dst));
            assigned[at] = Some(dst);
        }
        for &(value, output) in &stores {
            code.push(Instr::Store {
                src: assigned[value.0].expect("stored values are live"),
                output,
            });
        }

        let mut inputs = self.inputs;
        inputs.extend(self.updates.iter().map(|&(dtype, _)| dtype));
        let mut outputs: Vec<DType> = self.updates.iter().map(|&(dtype, _)| dtype).collect();
        outputs.extend(self.outputs.iter().map(|&(dtype, _)| dtype));
        Program::new(code, inputs, outputs, self.updates.len())
    }
}

// ---- operands -----------------------------------------------------------------

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for half::f16 {}
    impl Sealed for half::bf16 {}
}

/// An element type a program can load and store.
pub trait Element: Copy + 'static + sealed::Sealed {
    const DTYPE: DType;

    #[doc(hidden)]
    fn source<B: Backend>(storage: &B::Vector<Self>) -> SourceData<'_, B>;

    #[doc(hidden)]
    fn sink<B: Backend>(storage: &mut B::Vector<Self>) -> SinkData<'_, B>;

    #[doc(hidden)]
    fn unwrap<B: Backend>(fresh: Fresh<B>) -> Option<B::Vector<Self>>;
}

macro_rules! element {
    ($ty:ty, $variant:ident) => {
        impl Element for $ty {
            const DTYPE: DType = DType::$variant;

            fn source<B: Backend>(storage: &B::Vector<Self>) -> SourceData<'_, B> {
                SourceData::$variant(storage)
            }

            fn sink<B: Backend>(storage: &mut B::Vector<Self>) -> SinkData<'_, B> {
                SinkData::$variant(storage)
            }

            fn unwrap<B: Backend>(fresh: Fresh<B>) -> Option<B::Vector<Self>> {
                match fresh {
                    Fresh::$variant(storage) => Some(storage),
                    _ => None,
                }
            }
        }
    };
}

element!(f32, F32);
element!(f16, F16);
element!(bf16, Bf16);

/// Borrowed storage of one of the [`DType`]s.
#[doc(hidden)]
pub enum SourceData<'a, B: Backend> {
    F32(&'a B::Vector<f32>),
    F16(&'a B::Vector<f16>),
    Bf16(&'a B::Vector<bf16>),
}

/// Mutably borrowed storage of one of the [`DType`]s.
#[doc(hidden)]
pub enum SinkData<'a, B: Backend> {
    F32(&'a mut B::Vector<f32>),
    F16(&'a mut B::Vector<f16>),
    Bf16(&'a mut B::Vector<bf16>),
}

/// Owned storage of one of the [`DType`]s, as a program allocates it.
#[doc(hidden)]
pub enum Fresh<B: Backend> {
    F32(B::Vector<f32>),
    F16(B::Vector<f16>),
    Bf16(B::Vector<bf16>),
}

/// A program input: storage and its length.
#[doc(hidden)]
pub struct Source<'a, B: Backend> {
    pub data: SourceData<'a, B>,
    pub len: usize,
}

/// An in-place program operand: storage and its length.
#[doc(hidden)]
pub struct Sink<'a, B: Backend> {
    pub data: SinkData<'a, B>,
    pub len: usize,
}

impl<B: Backend> Source<'_, B> {
    fn dtype(&self) -> DType {
        match self.data {
            SourceData::F32(_) => DType::F32,
            SourceData::F16(_) => DType::F16,
            SourceData::Bf16(_) => DType::Bf16,
        }
    }

    pub(crate) fn slice(&self) -> Slice<'_> {
        match self.data {
            SourceData::F32(storage) => Slice::F32(B::vector_slice(storage)),
            SourceData::F16(storage) => Slice::F16(B::vector_slice(storage)),
            SourceData::Bf16(storage) => Slice::Bf16(B::vector_slice(storage)),
        }
    }
}

impl<B: Backend> Sink<'_, B> {
    fn dtype(&self) -> DType {
        match self.data {
            SinkData::F32(_) => DType::F32,
            SinkData::F16(_) => DType::F16,
            SinkData::Bf16(_) => DType::Bf16,
        }
    }

    pub(crate) fn slice(&mut self) -> SliceMut<'_> {
        match &mut self.data {
            SinkData::F32(storage) => SliceMut::F32(B::vector_slice_mut(storage)),
            SinkData::F16(storage) => SliceMut::F16(B::vector_slice_mut(storage)),
            SinkData::Bf16(storage) => SliceMut::Bf16(B::vector_slice_mut(storage)),
        }
    }

    fn as_source(&self) -> Source<'_, B> {
        let data = match &self.data {
            SinkData::F32(storage) => SourceData::F32(&**storage),
            SinkData::F16(storage) => SourceData::F16(&**storage),
            SinkData::Bf16(storage) => SourceData::Bf16(&**storage),
        };
        Source {
            data,
            len: self.len,
        }
    }

    /// Replace the storage with a program's result of the same type.
    fn replace(&mut self, fresh: Fresh<B>) {
        match (&mut self.data, fresh) {
            (SinkData::F32(storage), Fresh::F32(value)) => **storage = value,
            (SinkData::F16(storage), Fresh::F16(value)) => **storage = value,
            (SinkData::Bf16(storage), Fresh::Bf16(value)) => **storage = value,
            _ => unreachable!("in-place types are checked against the program"),
        }
    }
}

/// A tensor a [`Program`] can read, or update in place: [`Vector`] and
/// [`Matrix`] of any [`Element`] type. A matrix is read in row-major order.
pub trait Fusable<B: Backend> {
    #[doc(hidden)]
    fn source(&self) -> Source<'_, B>;

    #[doc(hidden)]
    fn sink(&mut self) -> Sink<'_, B>;
}

impl<T: Element, B: Backend> Fusable<B> for Vector<T, B> {
    fn source(&self) -> Source<'_, B> {
        Source {
            data: T::source::<B>(self.storage()),
            len: self.len(),
        }
    }

    fn sink(&mut self) -> Sink<'_, B> {
        let len = self.len();
        Sink {
            data: T::sink::<B>(self.storage_mut()),
            len,
        }
    }
}

impl<T: Element, B: Backend> Fusable<B> for Matrix<T, B> {
    fn source(&self) -> Source<'_, B> {
        Source {
            data: T::source::<B>(B::matrix_as_vector(self.storage())),
            len: self.rows() * self.cols(),
        }
    }

    fn sink(&mut self) -> Sink<'_, B> {
        let len = self.rows() * self.cols();
        Sink {
            data: T::sink::<B>(B::matrix_as_vector_mut(self.storage_mut())),
            len,
        }
    }
}

/// A fresh result of a [`Program`], shaped like the iteration space.
pub struct Output<B: Backend> {
    shape: (usize, usize),
    data: Fresh<B>,
}

impl<B: Backend> Output<B> {
    pub fn dtype(&self) -> DType {
        match self.data {
            Fresh::F32(_) => DType::F32,
            Fresh::F16(_) => DType::F16,
            Fresh::Bf16(_) => DType::Bf16,
        }
    }

    /// The result as a flat vector.
    ///
    /// # Panics
    ///
    /// If the output's storage type is not `T`.
    #[track_caller]
    pub fn into_vector<T: Element>(self) -> Vector<T, B> {
        let dtype = self.dtype();
        let len = self.shape.0 * self.shape.1;
        match T::unwrap::<B>(self.data) {
            Some(storage) => Vector::from_storage(len, storage),
            None => panic!("fused output is {}, not {}", dtype.name(), T::DTYPE.name()),
        }
    }

    /// The result as a matrix of the iteration space's shape.
    #[track_caller]
    pub fn into_matrix<T: Element>(self) -> Matrix<T, B> {
        let (rows, cols) = self.shape;
        let storage = self.into_vector::<T>().into_storage();
        Matrix::from_storage(rows, cols, B::vector_into_matrix(storage))
    }
}

// ---- fusion on and off ----------------------------------------------------------

/// Whether programs run as one kernel, or as one kernel per instruction.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum Mode {
    #[default]
    Fused,
    /// Every instruction becomes the existing [`Kernels`] operation it names,
    /// with every intermediate materialized — what the code did before fusion,
    /// and on the host the exact reference fusion is tested against.
    Unfused,
}

thread_local! {
    static MODE: Cell<Mode> = const { Cell::new(Mode::Fused) };
}

/// This thread's fusion mode.
pub fn mode() -> Mode {
    MODE.with(Cell::get)
}

/// Set this thread's fusion mode.
pub fn set_mode(mode: Mode) {
    MODE.with(|cell| cell.set(mode));
}

/// Run `f` with this thread's fusion mode set to `mode`, restoring the previous
/// one afterwards — even if `f` panics.
pub fn with_mode<R>(mode: Mode, f: impl FnOnce() -> R) -> R {
    struct Restore(Mode);
    impl Drop for Restore {
        fn drop(&mut self) {
            set_mode(self.0);
        }
    }
    let _restore = Restore(self::mode());
    set_mode(mode);
    f()
}

// ---- unfused evaluation -----------------------------------------------------------

/// A register's contents in unfused evaluation. Constants stay scalars so that
/// an operation with one becomes the scalar broadcast kernel, which is what the
/// unfused code called.
enum Register<B: Backend> {
    Scalar(f32),
    Tensor(Matrix<f32, B>),
}

impl<B: Kernels> Register<B> {
    /// The value as a tensor of its own, filling a constant out to the shape.
    fn materialize(&self, (rows, cols): (usize, usize)) -> Matrix<f32, B> {
        match self {
            Register::Scalar(value) => Matrix::filled(rows, cols, *value),
            Register::Tensor(tensor) => tensor.to_backend::<B>(),
        }
    }
}

/// Evaluate one instruction with the unfused kernels.
///
/// `load` reads an input slot through a remap. A store changes no register and
/// is left to the caller; everything else writes its destination, whose new
/// value is returned.
fn step<'r, B: Kernels>(
    instr: &Instr,
    registers: &'r mut [Option<Register<B>>],
    load: impl FnOnce(usize, Remap) -> Matrix<f32, B>,
) -> Option<&'r Register<B>> {
    let value = {
        let get = |reg: Reg| {
            registers[usize::from(reg)]
                .as_ref()
                .expect("programs are validated")
        };
        match *instr {
            Instr::Load { input, remap, .. } => Register::Tensor(load(usize::from(input), remap)),
            Instr::Const { value, .. } => Register::Scalar(value),
            Instr::Binary { op, a, b, .. } => match (get(a), get(b)) {
                (Register::Scalar(a), Register::Scalar(b)) => {
                    Register::Scalar(scalar_binary(op, *a, *b))
                }
                (Register::Tensor(a), Register::Scalar(b)) => {
                    Register::Tensor(B::matrix_broadcast(a, *b, op, false))
                }
                (Register::Scalar(a), Register::Tensor(b)) => {
                    Register::Tensor(B::matrix_broadcast(b, *a, op, true))
                }
                (Register::Tensor(a), Register::Tensor(b)) => {
                    Register::Tensor(B::matrix_elementwise(a, b, op))
                }
            },
            Instr::Unary { op, a, .. } => match get(a) {
                Register::Scalar(a) => Register::Scalar(op.value(*a)),
                Register::Tensor(a) => Register::Tensor(B::matrix_unary(a, op)),
            },
            Instr::Cmp { op, a, b, .. } => match (get(a), get(b)) {
                (Register::Scalar(a), Register::Scalar(b)) => Register::Scalar(op.value(*a, *b)),
                (Register::Tensor(a), Register::Scalar(b)) => {
                    Register::Tensor(B::matrix_compare_scalar(a, *b, op, false))
                }
                (Register::Scalar(a), Register::Tensor(b)) => {
                    Register::Tensor(B::matrix_compare_scalar(b, *a, op, true))
                }
                (Register::Tensor(a), Register::Tensor(b)) => {
                    Register::Tensor(B::matrix_compare(a, b, op))
                }
            },
            Instr::Store { .. } => return None,
        }
    };
    let dst = usize::from(instr.dst().expect("every instruction but a store writes"));
    registers[dst] = Some(value);
    registers[dst].as_ref()
}

/// A load as unfused kernels: widen if needed, then materialize the remap —
/// a transpose kernel, or a stack of copies for a broadcast.
fn load_unfused<B: Kernels>(
    source: &Source<'_, B>,
    (rows, cols): (usize, usize),
    remap: Remap,
) -> Matrix<f32, B> {
    let (stored_rows, stored_cols) = match remap {
        Remap::Identity => (rows, cols),
        Remap::Transpose => (cols, rows),
        Remap::Row => (1, cols),
        Remap::Column => (rows, 1),
    };
    let tensor = match source.data {
        SourceData::F32(storage) => {
            Matrix::build(stored_rows, stored_cols, B::vector_slice(storage))
        }
        _ => Matrix::build(stored_rows, stored_cols, &source.slice().widen()),
    };
    match remap {
        Remap::Identity => tensor,
        Remap::Transpose => B::transpose(&tensor),
        Remap::Row => {
            let copies: Vec<B::Vector<f32>> = (0..rows)
                .map(|_| B::store_vector(tensor.as_slice()))
                .collect();
            Matrix::from_storage(rows, cols, B::vstack(&copies, cols))
        }
        Remap::Column => {
            let copies: Vec<B::Vector<f32>> = (0..cols)
                .map(|_| B::store_vector(tensor.as_slice()))
                .collect();
            Matrix::from_storage(rows, cols, B::hstack(&copies, rows))
        }
    }
}

fn scalar_binary(op: BinaryOp, a: f32, b: f32) -> f32 {
    match op {
        BinaryOp::Add => a + b,
        BinaryOp::Sub => a - b,
        BinaryOp::Mul => a * b,
        BinaryOp::Div => a / b,
        BinaryOp::Rem => a % b,
    }
}

/// Narrow an `f32` tensor to an output's storage type.
fn narrow<B: Kernels>(tensor: Matrix<f32, B>, dtype: DType) -> Fresh<B> {
    let storage = B::matrix_into_flattened(tensor.into_storage());
    match dtype {
        DType::F32 => Fresh::F32(storage),
        DType::F16 => Fresh::F16(B::vector_from_vec(
            B::vector_slice(&storage)
                .iter()
                .map(|&x| f16::from_f32(x))
                .collect(),
        )),
        DType::Bf16 => Fresh::Bf16(B::vector_from_vec(
            B::vector_slice(&storage)
                .iter()
                .map(|&x| bf16::from_f32(x))
                .collect(),
        )),
    }
}

/// Run a program as one existing kernel per instruction.
fn unfused<B: Kernels>(
    program: &Program,
    shape: (usize, usize),
    inputs: &[Source<'_, B>],
    updated: &mut [Sink<'_, B>],
) -> Vec<Fresh<B>> {
    let fresh_inputs = program.fresh_inputs();
    let mut registers: Vec<Option<Register<B>>> = (0..REGISTERS).map(|_| None).collect();
    let mut stored: Vec<Option<Fresh<B>>> = (0..program.outputs.len()).map(|_| None).collect();
    for instr in &program.code {
        if let Instr::Store { src, output } = *instr {
            let value = registers[usize::from(src)]
                .as_ref()
                .expect("programs are validated")
                .materialize(shape);
            let output = usize::from(output);
            stored[output] = Some(narrow(value, program.outputs[output]));
            continue;
        }
        step::<B>(instr, &mut registers, |slot, remap| {
            if slot < fresh_inputs {
                load_unfused(&inputs[slot], shape, remap)
            } else {
                load_unfused(&updated[slot - fresh_inputs].as_source(), shape, remap)
            }
        });
    }

    let mut stored = stored.into_iter().map(|value| value.expect("validated"));
    for sink in updated.iter_mut() {
        sink.replace(stored.next().expect("validated"));
    }
    stored.collect()
}

// ---- the host tile interpreter ------------------------------------------------

/// Elements per tile. Sixteen registers of this many `f32`s plus a spare fill
/// 68 KB, which stays in an Apple-silicon L1 and a typical x86 L2.
const TILE: usize = 1024;

/// A typed input slice.
#[doc(hidden)]
pub enum Slice<'a> {
    F32(&'a [f32]),
    F16(&'a [f16]),
    Bf16(&'a [bf16]),
}

/// A typed output slice.
#[doc(hidden)]
pub enum SliceMut<'a> {
    F32(&'a mut [f32]),
    F16(&'a mut [f16]),
    Bf16(&'a mut [bf16]),
}

impl Slice<'_> {
    fn widen(&self) -> Vec<f32> {
        match self {
            Slice::F32(values) => values.to_vec(),
            Slice::F16(values) => values.iter().map(|x| x.to_f32()).collect(),
            Slice::Bf16(values) => values.iter().map(|x| x.to_f32()).collect(),
        }
    }

    #[inline]
    fn get(&self, index: usize) -> f32 {
        match self {
            Slice::F32(values) => values[index],
            Slice::F16(values) => values[index].to_f32(),
            Slice::Bf16(values) => values[index].to_f32(),
        }
    }

    /// Widen `len` elements from `start`, read through `remap`, into `out`.
    fn gather(&self, start: usize, shape: (usize, usize), remap: Remap, out: &mut [f32]) {
        if remap == Remap::Identity {
            let end = start + out.len();
            match self {
                Slice::F32(values) => out.copy_from_slice(&values[start..end]),
                Slice::F16(values) => {
                    for (out, x) in out.iter_mut().zip(&values[start..end]) {
                        *out = x.to_f32();
                    }
                }
                Slice::Bf16(values) => {
                    for (out, x) in out.iter_mut().zip(&values[start..end]) {
                        *out = x.to_f32();
                    }
                }
            }
            return;
        }
        for (offset, out) in out.iter_mut().enumerate() {
            *out = self.get(remap.source(start + offset, shape));
        }
    }
}

impl SliceMut<'_> {
    fn view(&self) -> Slice<'_> {
        match self {
            SliceMut::F32(values) => Slice::F32(values),
            SliceMut::F16(values) => Slice::F16(values),
            SliceMut::Bf16(values) => Slice::Bf16(values),
        }
    }

    /// Narrow `values` into this output from `start`.
    fn scatter(&mut self, start: usize, values: &[f32]) {
        let end = start + values.len();
        match self {
            SliceMut::F32(out) => out[start..end].copy_from_slice(values),
            SliceMut::F16(out) => {
                for (out, &x) in out[start..end].iter_mut().zip(values) {
                    *out = f16::from_f32(x);
                }
            }
            SliceMut::Bf16(out) => {
                for (out, &x) in out[start..end].iter_mut().zip(values) {
                    *out = bf16::from_f32(x);
                }
            }
        }
    }

    fn fill(&mut self, start: usize, len: usize, value: f32) {
        let end = start + len;
        match self {
            SliceMut::F32(out) => out[start..end].fill(value),
            SliceMut::F16(out) => out[start..end].fill(f16::from_f32(value)),
            SliceMut::Bf16(out) => out[start..end].fill(bf16::from_f32(value)),
        }
    }
}

/// Allocate a program's fresh outputs.
fn allocate(program: &Program, len: usize) -> Vec<Owned> {
    program.outputs[program.updated..]
        .iter()
        .map(|dtype| match dtype {
            DType::F32 => Owned::F32(vec![0.0; len]),
            DType::F16 => Owned::F16(vec![f16::ZERO; len]),
            DType::Bf16 => Owned::Bf16(vec![bf16::ZERO; len]),
        })
        .collect()
}

enum Owned {
    F32(Vec<f32>),
    F16(Vec<f16>),
    Bf16(Vec<bf16>),
}

impl Owned {
    fn slice(&mut self) -> SliceMut<'_> {
        match self {
            Owned::F32(values) => SliceMut::F32(values),
            Owned::F16(values) => SliceMut::F16(values),
            Owned::Bf16(values) => SliceMut::Bf16(values),
        }
    }

    fn store<B: Backend>(self) -> Fresh<B> {
        match self {
            Owned::F32(values) => Fresh::F32(B::vector_from_vec(values)),
            Owned::F16(values) => Fresh::F16(B::vector_from_vec(values)),
            Owned::Bf16(values) => Fresh::Bf16(B::vector_from_vec(values)),
        }
    }
}

/// Run a program on the CPU, over whatever backend's storage, reading it in
/// place. This is the [`Host`] implementation and the fallback for a Metal
/// tensor that is not device-resident.
pub(crate) fn interpret_on<B: Backend>(
    program: &Program,
    shape: (usize, usize),
    inputs: &[Source<'_, B>],
    updated: &mut [Sink<'_, B>],
) -> Vec<Fresh<B>> {
    let len = shape.0 * shape.1;
    let mut fresh = allocate(program, len);
    {
        let inputs: Vec<Slice<'_>> = inputs.iter().map(Source::slice).collect();
        let mut outputs: Vec<SliceMut<'_>> = updated.iter_mut().map(Sink::slice).collect();
        outputs.extend(fresh.iter_mut().map(Owned::slice));
        interpret(program, shape, &inputs, &mut outputs);
    }
    fresh.into_iter().map(Owned::store).collect()
}

/// The tile interpreter.
///
/// The space is walked in tiles of [`TILE`] elements. Within a tile each
/// instruction runs the same vectorized kernel its unfused counterpart would,
/// over tile-sized register slices that stay in cache, so the interpretation
/// cost — one dispatch per instruction per tile — is spread over a thousand
/// elements. Constants never fill a register: an operation with one becomes the
/// scalar-broadcast kernel, exactly as in unfused code.
///
/// `outputs` holds the in-place tensors first; loads of the last
/// `program.updated()` input slots read them.
fn interpret(
    program: &Program,
    shape: (usize, usize),
    inputs: &[Slice<'_>],
    outputs: &mut [SliceMut<'_>],
) {
    let len = shape.0 * shape.1;
    let fresh_inputs = program.fresh_inputs();
    // One physical tile per register, plus a spare that receives each result
    // so an instruction may overwrite one of its own operands.
    let physical = program.registers + 1;
    let mut scratch = vec![0.0f32; physical * TILE.min(len.max(1))];
    let tile_len = TILE.min(len.max(1));
    let mut map: [usize; REGISTERS] = std::array::from_fn(|reg| reg.min(physical - 1));
    let mut spare = physical - 1;
    let mut constant: [Option<f32>; REGISTERS] = [None; REGISTERS];

    let mut start = 0;
    while start < len {
        let n = tile_len.min(len - start);
        // The tiles are disjoint `n`-long windows of `scratch`.
        let base = scratch.as_mut_ptr();
        let tile = |physical: usize| unsafe { base.add(physical * tile_len) };

        for instr in &program.code {
            match *instr {
                Instr::Load { dst, input, remap } => {
                    let slot = usize::from(input);
                    // SAFETY: a register's tile is a distinct window of scratch.
                    let out = unsafe { std::slice::from_raw_parts_mut(tile(map[dst as usize]), n) };
                    if slot < fresh_inputs {
                        inputs[slot].gather(start, shape, remap, out);
                    } else {
                        outputs[slot - fresh_inputs]
                            .view()
                            .gather(start, shape, remap, out);
                    }
                    constant[dst as usize] = None;
                }
                Instr::Const { dst, value } => constant[dst as usize] = Some(value),
                Instr::Binary { dst, op, a, b } => {
                    let result = match (constant[a as usize], constant[b as usize]) {
                        (Some(a), Some(b)) => Some(scalar_binary(op, a, b)),
                        (ca, cb) => {
                            // SAFETY: `spare` is never mapped to a register, so
                            // the output window cannot overlap an operand.
                            let out = unsafe { std::slice::from_raw_parts_mut(tile(spare), n) };
                            let view = |reg: Reg| unsafe {
                                std::slice::from_raw_parts(tile(map[reg as usize]), n)
                            };
                            match (ca, cb) {
                                (None, Some(b)) => kernel::broadcast(view(a), b, op, false, out),
                                (Some(a), None) => kernel::broadcast(view(b), a, op, true, out),
                                _ => kernel::elementwise(view(a), view(b), op, out),
                            }
                            None
                        }
                    };
                    commit(dst, result, &mut map, &mut spare, &mut constant);
                }
                Instr::Unary { dst, op, a } => {
                    let result = match constant[a as usize] {
                        Some(a) => Some(op.value(a)),
                        None => {
                            let out = unsafe { std::slice::from_raw_parts_mut(tile(spare), n) };
                            let a = unsafe { std::slice::from_raw_parts(tile(map[a as usize]), n) };
                            kernel::unary(a, op, out);
                            None
                        }
                    };
                    commit(dst, result, &mut map, &mut spare, &mut constant);
                }
                Instr::Cmp { dst, op, a, b } => {
                    let result = match (constant[a as usize], constant[b as usize]) {
                        (Some(a), Some(b)) => Some(op.value(a, b)),
                        (ca, cb) => {
                            let out = unsafe { std::slice::from_raw_parts_mut(tile(spare), n) };
                            let view = |reg: Reg| unsafe {
                                std::slice::from_raw_parts(tile(map[reg as usize]), n)
                            };
                            match (ca, cb) {
                                (None, Some(b)) => {
                                    kernel::compare_scalar(view(a), b, op, false, out)
                                }
                                (Some(a), None) => {
                                    kernel::compare_scalar(view(b), a, op, true, out)
                                }
                                _ => kernel::compare(view(a), view(b), op, out),
                            }
                            None
                        }
                    };
                    commit(dst, result, &mut map, &mut spare, &mut constant);
                }
                Instr::Store { src, output } => {
                    let output = &mut outputs[usize::from(output)];
                    match constant[src as usize] {
                        Some(value) => output.fill(start, n, value),
                        None => {
                            let values =
                                unsafe { std::slice::from_raw_parts(tile(map[src as usize]), n) };
                            output.scatter(start, values);
                        }
                    }
                }
            }
        }
        start += n;
    }
}

/// Retire an instruction's result: a constant, or the tensor just written into
/// the spare tile, which becomes `dst`'s tile while `dst`'s old one becomes the
/// spare.
fn commit(
    dst: Reg,
    result: Option<f32>,
    map: &mut [usize; REGISTERS],
    spare: &mut usize,
    constant: &mut [Option<f32>; REGISTERS],
) {
    let dst = dst as usize;
    match result {
        Some(value) => constant[dst] = Some(value),
        None => {
            std::mem::swap(&mut map[dst], spare);
            constant[dst] = None;
        }
    }
}

/// The per-tile kernels: the SIMD tier where the build has one, and otherwise
/// the scalar definitions the SIMD kernels are tested against.
mod kernel {
    use super::super::{Analytic, BinaryOp, Compare};
    use super::scalar_binary;

    #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
    use crate::simd::f32k;

    pub fn elementwise(a: &[f32], b: &[f32], op: BinaryOp, out: &mut [f32]) {
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        if op != BinaryOp::Rem {
            return f32k::elementwise(a, b, op, out);
        }
        for ((out, &a), &b) in out.iter_mut().zip(a).zip(b) {
            *out = scalar_binary(op, a, b);
        }
    }

    pub fn broadcast(
        values: &[f32],
        scalar: f32,
        op: BinaryOp,
        scalar_left: bool,
        out: &mut [f32],
    ) {
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        if op != BinaryOp::Rem {
            return f32k::broadcast(values, scalar, op, scalar_left, out);
        }
        for (out, &x) in out.iter_mut().zip(values) {
            *out = if scalar_left {
                scalar_binary(op, scalar, x)
            } else {
                scalar_binary(op, x, scalar)
            };
        }
    }

    pub fn compare(a: &[f32], b: &[f32], op: Compare, out: &mut [f32]) {
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        return f32k::compare(a, b, op, out);
        #[allow(unreachable_code)]
        for ((out, &a), &b) in out.iter_mut().zip(a).zip(b) {
            *out = op.value(a, b);
        }
    }

    pub fn compare_scalar(
        values: &[f32],
        scalar: f32,
        op: Compare,
        scalar_left: bool,
        out: &mut [f32],
    ) {
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        return f32k::compare_scalar(values, scalar, op, scalar_left, out);
        #[allow(unreachable_code)]
        for (out, &x) in out.iter_mut().zip(values) {
            *out = if scalar_left {
                op.value(scalar, x)
            } else {
                op.value(x, scalar)
            };
        }
    }

    /// `sqrt` is correctly rounded in hardware, so its vector form is exact; the
    /// other functions use the scalar definitions, which the unfused kernels use
    /// too.
    pub fn unary(values: &[f32], op: Analytic, out: &mut [f32]) {
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        if op == Analytic::Sqrt {
            return f32k::sqrt(values, out);
        }
        for (out, &x) in out.iter_mut().zip(values) {
            *out = op.value(x);
        }
    }
}

/// The [`Host`] entry point for [`Kernels::fused`].
pub(crate) fn host(
    program: &Program,
    shape: (usize, usize),
    inputs: &[Source<'_, Host>],
    updated: &mut [Sink<'_, Host>],
) -> Vec<Fresh<Host>> {
    interpret_on(program, shape, inputs, updated)
}

/// The [`Metal`](super::Metal) entry point for [`Kernels::fused`]: one
/// dispatch of the bytecode shader when every operand is device-resident and
/// every instruction has a shader implementation, and otherwise the host
/// interpreter over the shared memory — the same fallback every Metal kernel
/// has.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) fn metal(
    program: &Program,
    shape: (usize, usize),
    inputs: &[Source<'_, super::Metal>],
    updated: &mut [Sink<'_, super::Metal>],
) -> Vec<Fresh<super::Metal>> {
    resident(program, shape, inputs, updated)
        .unwrap_or_else(|| interpret_on(program, shape, inputs, updated))
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn resident(
    program: &Program,
    shape: (usize, usize),
    inputs: &[Source<'_, super::Metal>],
    updated: &[Sink<'_, super::Metal>],
) -> Option<Vec<Fresh<super::Metal>>> {
    use objc2::runtime::ProtocolObject;
    use objc2_metal::MTLBuffer;

    use super::MetalStorage;
    use crate::metal::MetalBuffer;

    if !program.runs_on_metal() {
        return None;
    }
    let len = shape.0 * shape.1;

    fn raw<T: Copy + 'static>(storage: &MetalStorage<T>) -> Option<&ProtocolObject<dyn MTLBuffer>> {
        Some(storage.device()?.raw())
    }
    let mut read = Vec::with_capacity(program.inputs.len());
    for input in inputs {
        read.push(match input.data {
            SourceData::F32(storage) => raw(storage)?,
            SourceData::F16(storage) => raw(storage)?,
            SourceData::Bf16(storage) => raw(storage)?,
        });
    }
    let mut written = Vec::with_capacity(program.outputs.len());
    for target in updated {
        let buffer = match &target.data {
            SinkData::F32(storage) => raw(storage)?,
            SinkData::F16(storage) => raw(storage)?,
            SinkData::Bf16(storage) => raw(storage)?,
        };
        read.push(buffer);
        written.push(buffer);
    }

    enum Allocation {
        F32(MetalBuffer<f32>),
        F16(MetalBuffer<f16>),
        Bf16(MetalBuffer<bf16>),
    }
    let mut fresh = Vec::with_capacity(program.fresh_outputs());
    for dtype in &program.outputs[program.updated..] {
        fresh.push(match dtype {
            DType::F32 => Allocation::F32(MetalBuffer::allocate(len)?),
            DType::F16 => Allocation::F16(MetalBuffer::allocate(len)?),
            DType::Bf16 => Allocation::Bf16(MetalBuffer::allocate(len)?),
        });
    }
    written.extend(fresh.iter().map(|allocation| match allocation {
        Allocation::F32(buffer) => buffer.raw(),
        Allocation::F16(buffer) => buffer.raw(),
        Allocation::Bf16(buffer) => buffer.raw(),
    }));

    crate::metal::fused_elementwise(&program.encode(), shape, &read, &written)?;
    Some(
        fresh
            .into_iter()
            .map(|allocation| match allocation {
                Allocation::F32(buffer) => Fresh::F32(MetalStorage::from_device(buffer)),
                Allocation::F16(buffer) => Fresh::F16(MetalStorage::from_device(buffer)),
                Allocation::Bf16(buffer) => Fresh::Bf16(MetalStorage::from_device(buffer)),
            })
            .collect(),
    )
}
