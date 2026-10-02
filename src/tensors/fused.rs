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
//! # Optimization
//!
//! [`Builder::build`] does not emit the operations as built: it hands them to
//! the optimizer, which considers equivalent programs — common subexpressions
//! merged, exact identities applied, associative chains regrouped with their
//! constants folded, constants and loads recomputed where that frees a
//! register, and many instruction orders — and keeps the one with the lowest
//! cost under a [`CostModel`]:
//!
//! ```text
//! C = α · instructions + β · peak registers + γ · critical path
//!   + δ · memory traffic + ε · special operations
//! ```
//!
//! [`Program::cost`] reports a program's terms, and [`Builder::build_with`]
//! takes the model and the [`Algebra`]. Plans are cached per thread by the
//! program's structure with its constants as placeholders, so a program
//! rebuilt every step with new constants is optimized once.
//!
//! # Exactness
//!
//! A program performs, for every element, exactly the operations of its own
//! instructions — in the program's own element type — in order. On the
//! [`Host`] backend a program fused is therefore identical to the same program
//! unfused bit for bit — [`Mode::Unfused`] exists to check that, and the tests
//! do. (The one gap is the sign of a zero from `Min`/`Max` when `−0.0` meets
//! `+0.0`, which the unfused kernels do not pin down either; see [`Compare`].)
//!
//! Against the operations as *built*, that holds under [`Algebra::Exact`],
//! whose rewrites are all exact. Under [`Algebra::Reassociate`], the default,
//! the optimizer may also regroup `(a + b) + c` as `a + (b + c)`, which can
//! round differently.
//!
//! Metal builds its shaders with the compiler's fast-math defaults, fused ones
//! included, and inside one kernel the compiler may contract `a·b + c` into a
//! fused multiply-add across what used to be separate kernels. Metal results
//! therefore agree with the host within the usual tolerance rather than
//! exactly — the same promise the unfused Metal kernels make. A program run
//! more than once on Metal is compiled into a kernel of its own rather than
//! interpreted; it calls the interpreter's own functions, so the two agree.
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
//! Every input and output has its own [`DType`], independent of the type the
//! arithmetic runs in. A [`Program<T>`] computes in `T` — `f32` by default, or
//! `f64`, `f16`, `bf16` — loads convert to `T` and stores convert from it, so
//! the compact storage types only change how many bytes cross memory, and `f64`
//! storage and arithmetic give a program the same precision as the unfused
//! `f64` kernels. The Metal shader is compiled for `f32`, `f16` and `bf16`
//! arithmetic, so a `Program<f16>` runs in `half` registers on the GPU; a
//! program in `f64`, or over `f64` storage, runs on the host.

use std::any::TypeId;
use std::cell::Cell;
use std::fmt;
use std::ops::Range;

use half::{bf16, f16};

use super::{Analytic, Axis, Backend, BinaryOp, Compare, Host, Kernels, Matrix, Vector};
use crate::counters;
use crate::numbers::Real;
use crate::parallel::Shared;

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

/// How an input or output is stored. Arithmetic is a separate matter: it runs
/// in the program's own element type, `T` in [`Program<T>`].
///
/// The representation is part of the Metal shader ABI: keep the discriminants
/// stable and only append. The shaders have no `f64`, so a program that touches
/// [`F64`](DType::F64) storage runs on the host.
#[repr(u16)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32 = 0,
    F16 = 1,
    Bf16 = 2,
    F64 = 3,
}

impl DType {
    /// Bytes per stored element.
    pub fn size(self) -> usize {
        match self {
            DType::F64 => 8,
            DType::F32 => 4,
            DType::F16 | DType::Bf16 => 2,
        }
    }

    fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::Bf16 => "bf16",
            DType::F64 => "f64",
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
///
/// `T` is the type the program computes in; it defaults to `f32`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Instr<T = f32> {
    /// `dst ← input[remap(i)]`, converted from the input's storage type to `T`.
    Load { dst: Reg, input: u8, remap: Remap },
    /// `dst ← value`, the same for every element.
    Const { dst: Reg, value: T },
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
    /// `output[i] ← src`, converted to the output's storage type.
    Store { src: Reg, output: u8 },
}

impl<T> Instr<T> {
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
///
/// `T` is the element type the arithmetic runs in. The storage types of the
/// inputs and outputs are independent of it — see [`DType`] — so a program can
/// compute in `f64` over `f32` data, or in `f32` over `f16` data. It defaults to
/// `f32`, which is also all the Metal backend computes in.
#[derive(Clone, Debug, PartialEq)]
pub struct Program<T = f32> {
    code: Vec<Instr<T>>,
    inputs: Vec<DType>,
    outputs: Vec<DType>,
    updated: usize,
    registers: usize,
    /// How many uniforms the program has: the first entries of `named`.
    uniforms: usize,
    /// The value of every named constant the optimizer saw, uniforms first.
    named: Vec<T>,
    /// The instructions whose constants are computed from uniforms, and how.
    bound: Vec<(usize, optimizer::Scalar)>,
    /// The inputs the program computes itself, one per statistic it reads
    /// (see [`Builder::row_statistic`]): input slot `given_inputs() + k` is
    /// statistic `derived[k].1` of each row of input `derived[k].0`.
    derived: Vec<(u8, RowStatistic)>,
}

/// The storage type that holds a `T` exactly.
fn dtype_of<T: 'static>() -> DType {
    if TypeId::of::<T>() == TypeId::of::<f16>() {
        DType::F16
    } else if TypeId::of::<T>() == TypeId::of::<bf16>() {
        DType::Bf16
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        DType::F64
    } else {
        DType::F32
    }
}

/// A statistic of each row of a program's input, which the program reads as
/// one value per row — see [`Builder::row_statistic`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RowStatistic {
    /// The row's mean.
    Mean,
    /// The row's sum of squared deviations from its mean: its variance times
    /// its length.
    Deviations,
}

impl<T: Real> Program<T> {
    /// Check a hand-written program.
    ///
    /// `inputs` and `outputs` give each slot's storage type; the last `updated`
    /// inputs and the first `updated` outputs are the same in-place tensors.
    pub fn new(
        code: Vec<Instr<T>>,
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
            uniforms: 0,
            named: Vec::new(),
            bound: Vec::new(),
            derived: Vec::new(),
        })
    }

    /// The current value of every uniform, in declaration order (see
    /// [`Builder::uniform`]). Empty for a hand-written program.
    pub fn uniforms(&self) -> &[T] {
        &self.named[..self.uniforms]
    }

    /// Set uniform `index` to `value`, and every constant computed from it.
    ///
    /// # Panics
    ///
    /// If the program has no uniform `index`.
    #[track_caller]
    pub fn set_uniform(&mut self, index: usize, value: T) {
        assert!(
            index < self.uniforms,
            "fused program: uniform {index} out of {}",
            self.uniforms
        );
        self.named[index] = value;
        for (at, scalar) in &self.bound {
            if let Instr::Const { value, .. } = &mut self.code[*at] {
                *value = evaluate(scalar, &self.named);
            }
        }
    }

    /// The instructions.
    pub fn code(&self) -> &[Instr<T>] {
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

    /// Inputs that are only read, counting the statistics the program
    /// computes for itself.
    pub fn fresh_inputs(&self) -> usize {
        self.inputs.len() - self.updated
    }

    /// The inputs a caller passes: those only read, less the statistics the
    /// program computes for itself.
    pub fn given_inputs(&self) -> usize {
        self.fresh_inputs() - self.derived.len()
    }

    /// The statistics the program computes for itself, in the order of the
    /// input slots after the given ones.
    pub fn row_statistics(&self) -> &[(u8, RowStatistic)] {
        &self.derived
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
    pub fn run<B: Kernels<T>>(
        &self,
        shape: (usize, usize),
        inputs: &[&dyn Fusable<B>],
        updated: &mut [&mut dyn FusableMut<B>],
    ) -> Vec<Output<B>> {
        assert_eq!(
            inputs.len(),
            self.given_inputs(),
            "fused program: expected {} inputs, got {}",
            self.given_inputs(),
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
            self.check_input(slot, source.dtype(), source.len, source.view, shape);
        }
        let mut sinks: Vec<Sink<'_, B>> = updated.iter_mut().map(|target| target.sink()).collect();
        for (slot, sink) in sinks.iter().enumerate() {
            self.check_input(
                self.fresh_inputs() + slot,
                sink.dtype(),
                sink.len,
                None,
                shape,
            );
        }

        let fresh = match mode() {
            Mode::Fused if self.derived.is_empty() => {
                counters::kernel(self.bytes(len), self.fresh_outputs());
                B::fused(self, shape, &sources, &mut sinks)
            }
            Mode::Fused => {
                // The statistics are computed from the inputs, not read.
                let statistics = self.derived.len() * shape.0 * size_of::<T>();
                counters::kernel(self.bytes(len) - statistics, self.fresh_outputs());
                B::fused_with_statistics(self, shape, &sources, &mut sinks)
            }
            Mode::Unfused => {
                let statistics = row_statistics(self, shape, &sources);
                let mut all: Vec<Source<'_, B>> = sources.iter().map(Source::reborrow).collect();
                all.extend(statistics.iter().map(source_of));
                unfused(self, shape, &all, &mut sinks)
            }
        };
        fresh
            .into_iter()
            .map(|data| Output { shape, data })
            .collect()
    }

    /// Run a program with one output of its own type, and sum that output
    /// along `axis` — [`Axis::Rows`] totals each row, [`Axis::Columns`] each
    /// column — without keeping it.
    ///
    /// The result is what running the program and summing its output with
    /// [`Kernels::matvec`] or [`Kernels::vecmat`] against ones gives: on the
    /// host exactly that. On Metal, once the program has run before, the
    /// program and the sum are one kernel, and the output is never written to
    /// memory — so a softmax is two passes over its input, the sums and then
    /// the normalized values, with nothing in between.
    ///
    /// ```
    /// use tensorcrate::tensors::fused::{Builder, DType};
    /// use tensorcrate::tensors::{Analytic, Axis, Matrix};
    ///
    /// // Σⱼ exp(xᵢⱼ), each row.
    /// let mut b = Builder::<f32>::new();
    /// let x = b.input(DType::F32);
    /// let e = b.unary(Analytic::Exp, x);
    /// b.output(e, DType::F32);
    /// let exp = b.build().unwrap();
    ///
    /// let x = Matrix::from_rows([[0.0f32, 0.0], [1.0, 0.0]]);
    /// let sums = exp.run_sum((2, 2), &[&x], Axis::Rows);
    /// assert_eq!(sums[0], 2.0);
    /// assert_eq!(sums[1], 1.0f32.exp() + 1.0);
    /// ```
    ///
    /// # Panics
    ///
    /// If the program updates tensors in place, does not have exactly one
    /// output, or that output is not of type `T`; or if `inputs` disagree with
    /// the program as in [`run`](Self::run).
    #[track_caller]
    pub fn run_sum<B: Kernels<T>>(
        &self,
        shape: (usize, usize),
        inputs: &[&dyn Fusable<B>],
        axis: Axis,
    ) -> Vector<T, B>
    where
        T: Element,
    {
        assert!(
            self.updated == 0 && self.outputs == [T::DTYPE] && self.derived.is_empty(),
            "fused sum: the program must have one {} output, update nothing in place and \
             compute no row statistics",
            T::DTYPE.name()
        );
        assert_eq!(
            inputs.len(),
            self.fresh_inputs(),
            "fused program: expected {} inputs, got {}",
            self.fresh_inputs(),
            inputs.len()
        );
        let sources: Vec<Source<'_, B>> = inputs.iter().map(|input| input.source()).collect();
        for (slot, source) in sources.iter().enumerate() {
            self.check_input(slot, source.dtype(), source.len, source.view, shape);
        }
        match mode() {
            Mode::Fused => {
                let read: usize = self.inputs.iter().map(|dtype| dtype.size()).sum();
                let total = match axis {
                    Axis::Rows => shape.0,
                    Axis::Columns => shape.1,
                };
                counters::kernel(read * shape.0 * shape.1 + total * size_of::<T>(), 1);
                B::fused_sum(self, shape, &sources, axis)
            }
            Mode::Unfused => {
                let data = unfused(self, shape, &sources, &mut []).remove(0);
                axis_sum(Output { shape, data }.into_matrix::<T>(), axis)
            }
        }
    }

    /// [`run`](Self::run) for the common case: vectors of one length, all of the
    /// program's own element type, nothing updated in place, every output of
    /// that type too.
    #[track_caller]
    pub fn run_vectors<B: Kernels<T>>(&self, inputs: &[&Vector<T, B>]) -> Vec<Vector<T, B>>
    where
        T: Element,
    {
        let len = inputs.first().map_or(0, |input| input.len());
        let inputs: Vec<&dyn Fusable<B>> = inputs.iter().map(|&v| v as &dyn Fusable<B>).collect();
        self.run((1, len), &inputs, &mut [])
            .into_iter()
            .map(Output::into_vector)
            .collect()
    }

    /// Run the program as the epilogue of the matrix product `a·b`.
    ///
    /// The product is the program's input 0 and is consumed as it is computed:
    /// on Metal the product and the program are one dispatch, and the product
    /// is never written to memory unless the program stores it. `inputs` fill
    /// slots 1 on. The iteration space is the product's shape, so a bias added
    /// to every row is an input read through [`Remap::Row`].
    ///
    /// ```
    /// use tensorcrate::tensors::fused::{Builder, DType, Remap};
    /// use tensorcrate::tensors::{Compare, Matrix, Vector};
    ///
    /// // relu(x·w + bias), one kernel.
    /// let mut b = Builder::new();
    /// let product = b.input(DType::F32);
    /// let bias = b.input_remapped(DType::F32, Remap::Row);
    /// let shifted = b.add(product, bias);
    /// let zero = b.constant(0.0);
    /// let relu = b.compare(Compare::Max, shifted, zero);
    /// b.output(relu, DType::F32);
    /// let layer = b.build().unwrap();
    ///
    /// let x = Matrix::from_rows([[1.0f32, 2.0], [3.0, 4.0]]);
    /// let w = Matrix::from_rows([[1.0f32, -1.0], [0.0, 1.0]]);
    /// let bias = Vector::new([0.5f32, -2.5]);
    /// let y = layer.run_matmul(&x, &w, &[&bias]).remove(0).into_matrix::<f32>();
    /// assert_eq!(y.to_rows(), [[1.5, 0.0], [3.5, 0.0]]);
    /// ```
    ///
    /// On the host the product is computed first and the program then runs over
    /// it, which is exactly the unfused computation, so the result is the same
    /// bit for bit. On Metal it agrees within the usual tolerance.
    ///
    /// # Panics
    ///
    /// If the inner dimensions disagree, if input 0 is not stored as `T` or is
    /// read through a remap, if the program updates tensors in place, or if
    /// `inputs` disagree with the program as in [`run`](Self::run).
    #[track_caller]
    pub fn run_matmul<B: Kernels<T>>(
        &self,
        a: &Matrix<T, B>,
        b: &Matrix<T, B>,
        inputs: &[&dyn Fusable<B>],
    ) -> Vec<Output<B>>
    where
        T: Element,
    {
        assert_eq!(
            a.cols(),
            b.rows(),
            "fused matmul: inner dimensions {} and {} disagree",
            a.cols(),
            b.rows()
        );
        assert_eq!(
            self.updated, 0,
            "fused matmul: the epilogue cannot update tensors in place"
        );
        assert!(
            self.derived.is_empty(),
            "fused matmul: the epilogue cannot compute row statistics"
        );
        assert!(
            !self.inputs.is_empty(),
            "fused matmul: the epilogue must read the product as input 0"
        );
        assert_eq!(
            inputs.len() + 1,
            self.fresh_inputs(),
            "fused matmul: expected {} inputs besides the product, got {}",
            self.fresh_inputs() - 1,
            inputs.len()
        );
        for instr in &self.code {
            if let Instr::Load {
                input: 0, remap, ..
            } = *instr
            {
                assert_eq!(
                    remap,
                    Remap::Identity,
                    "fused matmul: the product is read through {remap:?}"
                );
            }
        }
        let shape = (a.rows(), b.cols());
        let len = shape.0 * shape.1;
        self.check_input(0, T::DTYPE, len, None, shape);
        let sources: Vec<Source<'_, B>> = inputs.iter().map(|input| input.source()).collect();
        for (slot, source) in sources.iter().enumerate() {
            self.check_input(slot + 1, source.dtype(), source.len, source.view, shape);
        }

        let fresh = match mode() {
            Mode::Fused => {
                // The operands are read and the product is not.
                let operands = (a.rows() * a.cols() + b.rows() * b.cols()) * T::DTYPE.size();
                counters::kernel(
                    self.bytes(len) - len * T::DTYPE.size() + operands,
                    self.fresh_outputs(),
                );
                B::matmul_epilogue(self, a, b, &sources)
            }
            Mode::Unfused => {
                let product = B::matmul(a, b);
                let sources = with_product(&product, &sources);
                unfused(self, shape, &sources, &mut [])
            }
        };
        fresh
            .into_iter()
            .map(|data| Output { shape, data })
            .collect()
    }

    /// Run the program unfused — one existing kernel per instruction — and keep
    /// every intermediate.
    ///
    /// `inputs` covers every input slot, in-place tensors included; nothing is
    /// written. Step `k` of the result holds the value instruction `k` produced
    /// (as `T`, before any narrowing), or `None` for a store.
    #[track_caller]
    pub fn trace<B: Kernels<T>>(
        &self,
        shape: (usize, usize),
        inputs: &[&dyn Fusable<B>],
    ) -> Vec<Option<Matrix<T, B>>> {
        let expected = self.inputs.len() - self.derived.len();
        assert_eq!(
            inputs.len(),
            expected,
            "fused trace: expected {expected} inputs, got {}",
            inputs.len()
        );
        let mut sources: Vec<Source<'_, B>> = inputs.iter().map(|input| input.source()).collect();
        // Statistics come before any in-place tensors among the input slots.
        let given = self.given_inputs();
        for (index, source) in sources.iter().enumerate() {
            let slot = if index < given {
                index
            } else {
                index + self.derived.len()
            };
            self.check_input(slot, source.dtype(), source.len, source.view, shape);
        }
        let updated: Vec<Source<'_, B>> = sources.drain(given..).collect();
        let statistics = row_statistics(self, shape, &sources);
        sources.extend(statistics.iter().map(source_of));
        sources.extend(updated);
        let mut registers: Vec<Option<Register<B, T>>> = (0..REGISTERS).map(|_| None).collect();
        self.code
            .iter()
            .map(|instr| {
                step::<B, T>(instr, &mut registers, |slot, remap| {
                    load_unfused(&sources[slot], shape, remap)
                })
                .map(|value| value.materialize(shape))
            })
            .collect()
    }

    #[track_caller]
    fn check_input(
        &self,
        slot: usize,
        dtype: DType,
        len: usize,
        view: Option<View>,
        shape: (usize, usize),
    ) {
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
                // A view has a shape of its own, which the remap must read
                // whole.
                if let Some(v) = view {
                    let (rows, cols) = shape;
                    let fits = match remap {
                        Remap::Identity => (v.rows, v.cols) == (rows, cols),
                        Remap::Transpose => (v.rows, v.cols) == (cols, rows),
                        Remap::Row | Remap::Column => v.along().is_some() || len <= 1,
                    };
                    assert!(
                        fits,
                        "fused program: input {slot} is a {}×{} view, which a {rows}×{cols} \
                         space cannot read through {remap:?}",
                        v.rows, v.cols
                    );
                }
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
                    value: value.into_f64() as f32,
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
        // The shaders have no `f64` storage and no remainder.
        !self.inputs.contains(&DType::F64)
            && !self.outputs.contains(&DType::F64)
            && !self.code.iter().any(|instr| {
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

impl<T: Real + fmt::Debug> fmt::Display for Program<T> {
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
enum Node<T> {
    Load(u8, Remap),
    Const(T),
    /// Uniform number `k`, in declaration order.
    Uniform(usize),
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
///
/// `T` is the element type the program will compute in, and is normally inferred
/// from the first constant or the tensors the program runs over.
#[derive(Clone, Debug)]
pub struct Builder<T = f32> {
    nodes: Vec<Node<T>>,
    inputs: Vec<DType>,
    updates: Vec<(DType, Option<Value>)>,
    outputs: Vec<(DType, Value)>,
    loads: Vec<((u8, Remap), Value)>,
    /// Each uniform's initial value, in declaration order.
    uniforms: Vec<T>,
    /// The row statistics the program reads, in declaration order: the input
    /// slot each is of, and which.
    derived: Vec<(u8, RowStatistic)>,
}

impl<T: Real> Default for Builder<T> {
    fn default() -> Self {
        Builder {
            nodes: Vec::new(),
            inputs: Vec::new(),
            updates: Vec::new(),
            outputs: Vec::new(),
            loads: Vec::new(),
            uniforms: Vec::new(),
            derived: Vec::new(),
        }
    }
}

impl<T: Real> Builder<T> {
    pub fn new() -> Self {
        Self::default()
    }

    fn push(&mut self, node: Node<T>) -> Value {
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

    /// A statistic of each row of `input` — which must be a value an
    /// [`input`](Self::input) returned, read without a remap — as one value
    /// per row, broadcast along it.
    ///
    /// The program computes it from the input itself: running the program is
    /// running [`Kernels::matrix_axis_moments`] along [`Axis::Rows`] on the
    /// input, converted to `T`, and passing the result as an input read
    /// through [`Remap::Column`]. On the host that is what happens, so the
    /// program fused and unfused agree bit for bit. On Metal a program that has
    /// run before is one kernel, with each row's statistics and then its
    /// elements computed by one SIMD group — so a layer norm reads its input
    /// once, and writes only its result.
    ///
    /// ```
    /// use tensorcrate::tensors::fused::{Builder, DType, RowStatistic};
    /// use tensorcrate::tensors::Matrix;
    ///
    /// // (x − mean) / sqrt(deviations / n), each row.
    /// let mut b = Builder::<f32>::new();
    /// let x = b.input(DType::F32);
    /// let mean = b.row_statistic(x, RowStatistic::Mean);
    /// let deviations = b.row_statistic(x, RowStatistic::Deviations);
    /// let variance = b.scale(deviations, 0.5);
    /// let deviation = b.unary(tensorcrate::tensors::Analytic::Sqrt, variance);
    /// let centered = b.sub(x, mean);
    /// let normalized = b.div(centered, deviation);
    /// b.output(normalized, DType::F32);
    /// let program = b.build().unwrap();
    ///
    /// let x = Matrix::from_rows([[1.0f32, 3.0], [10.0, 20.0]]);
    /// let y = program.run((2, 2), &[&x], &mut []).remove(0).into_matrix::<f32>();
    /// assert_eq!(y.to_rows(), [[-1.0, 1.0], [-1.0, 1.0]]);
    /// ```
    ///
    /// # Panics
    ///
    /// If `input` is not an input read without a remap.
    #[track_caller]
    pub fn row_statistic(&mut self, input: Value, statistic: RowStatistic) -> Value {
        let slot = self
            .loads
            .iter()
            .find(|&&((slot, remap), value)| {
                value == input && remap == Remap::Identity && usize::from(slot) < MAX_INPUTS
            })
            .map(|&((slot, _), _)| slot)
            .expect("a row statistic is of a value `input` returned");
        let index = match self.derived.iter().position(|&d| d == (slot, statistic)) {
            Some(index) => index,
            None => {
                self.derived.push((slot, statistic));
                self.derived.len() - 1
            }
        };
        // Numbered after the given inputs in `build`, once their count is known.
        self.load((2 * MAX_INPUTS + index) as u8, Remap::Column)
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

    pub fn constant(&mut self, value: T) -> Value {
        self.push(Node::Const(value))
    }

    /// A constant that can be changed after the program is built: the same for
    /// every element, like [`constant`](Self::constant), but set again with
    /// [`Program::set_uniform`] at the index this is (uniforms are numbered in
    /// declaration order) without building the program again.
    ///
    /// This is what lets a program be built once and run with new scalars every
    /// time — an optimizer's step size, or a bias correction that changes with
    /// the step count. Arithmetic on uniforms and constants alone is folded
    /// into the program's constants, so it runs once per
    /// [`set_uniform`](Program::set_uniform), in the program's element type,
    /// rather than once per element.
    pub fn uniform(&mut self, value: T) -> Value {
        self.uniforms.push(value);
        let index = self.uniforms.len() - 1;
        self.push(Node::Uniform(index))
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
    pub fn scale(&mut self, a: Value, factor: T) -> Value {
        let factor = self.constant(factor);
        self.mul(a, factor)
    }

    /// `a + offset`.
    pub fn shift(&mut self, a: Value, offset: T) -> Value {
        let offset = self.constant(offset);
        self.add(a, offset)
    }

    pub fn unary(&mut self, op: Analytic, a: Value) -> Value {
        self.push(Node::Unary(op, a))
    }

    pub fn compare(&mut self, op: Compare, a: Value, b: Value) -> Value {
        self.push(Node::Cmp(op, a, b))
    }

    /// Optimize, allocate registers and validate, with this thread's
    /// [`Algebra`] and the [`CostModel::BALANCED`] cost model — a program is
    /// built before anyone knows which backend will run it.
    ///
    /// The values are compiled to the cheapest equivalent program the
    /// optimizer finds (see [`build_with`](Self::build_with)), which never
    /// costs more than the operations in the order they were built.
    pub fn build(self) -> Result<Program<T>, ProgramError> {
        self.build_with(&CostModel::BALANCED, algebra())
    }

    /// [`build`](Self::build) under a cost model and algebra of your choosing.
    ///
    /// The optimizer considers equivalent programs — common subexpressions
    /// merged, exact identities applied, associative chains regrouped when
    /// `algebra` allows, constants and loads recomputed rather than held, and
    /// many instruction orders — and keeps the one with the lowest
    ///
    /// ```text
    /// C = α · instructions + β · peak registers + γ · critical path
    ///   + δ · memory traffic + ε · special operations
    /// ```
    ///
    /// under `model`, among those that fit [`REGISTERS`] and
    /// [`MAX_INSTRUCTIONS`]. Plans are cached per thread by the program's
    /// structure, with constants as placeholders, so a program rebuilt every
    /// step with new constants — an optimizer's — is optimized once.
    pub fn build_with(
        self,
        model: &CostModel,
        algebra: Algebra,
    ) -> Result<Program<T>, ProgramError> {
        // The given inputs, then the statistics, then the in-place tensors.
        let given = self.inputs.len();
        let fresh_inputs = given + self.derived.len();
        let slot_of = |slot: u8| -> u8 {
            let slot = usize::from(slot);
            if slot >= 2 * MAX_INPUTS {
                (given + slot - 2 * MAX_INPUTS) as u8
            } else if slot >= MAX_INPUTS {
                (fresh_inputs + slot - MAX_INPUTS) as u8
            } else {
                slot as u8
            }
        };

        // Stores, in-place tensors first.
        let mut stores: Vec<(Value, u8)> = Vec::new();
        for (index, (_, value)) in self.updates.iter().enumerate() {
            let Some(value) = *value else {
                return Err(ProgramError::OutputNotStoredOnce {
                    output: index as u8,
                });
            };
            stores.push((value, index as u8));
        }
        for (index, &(_, value)) in self.outputs.iter().enumerate() {
            stores.push((value, (self.updates.len() + index) as u8));
        }

        let mut inputs = self.inputs;
        inputs.extend(self.derived.iter().map(|_| dtype_of::<T>()));
        inputs.extend(self.updates.iter().map(|&(dtype, _)| dtype));
        let mut outputs: Vec<DType> = self.updates.iter().map(|&(dtype, _)| dtype).collect();
        outputs.extend(self.outputs.iter().map(|&(dtype, _)| dtype));
        if inputs.len() > MAX_INPUTS {
            return Err(ProgramError::TooManyInputs(inputs.len()));
        }

        // Uniforms are the first named constants, each its own name whatever
        // its value; the ordinary constants are named after them.
        let uniforms = self.uniforms.len();
        let mut constants: Vec<T> = self.uniforms.clone();
        let nodes = self
            .nodes
            .iter()
            .map(|node| match *node {
                Node::Load(slot, remap) => optimizer::Node::Load {
                    slot: slot_of(slot),
                    remap: remap as u8,
                },
                Node::Const(value) => {
                    optimizer::Node::Const(placeholder(value, &mut constants, uniforms))
                }
                Node::Uniform(index) => {
                    optimizer::Node::Const(optimizer::Scalar::Named(index as u32))
                }
                Node::Binary(op, a, b) => optimizer::Node::Binary(bin(op), a.0, b.0),
                Node::Unary(op, a) => optimizer::Node::Unary(optimizer::Function(op as u16), a.0),
                Node::Cmp(op, a, b) => optimizer::Node::Cmp(cmp(op), a.0, b.0),
            })
            .collect();
        let graph = optimizer::Graph {
            nodes,
            stores: stores
                .iter()
                .map(|&(value, output)| optimizer::Store {
                    value: value.0,
                    output,
                })
                .collect(),
            input_bytes: inputs.iter().map(|dtype| dtype.size() as u8).collect(),
            output_bytes: outputs.iter().map(|dtype| dtype.size() as u8).collect(),
        };
        let optimized = optimized(graph, model, algebra)?;
        let code = optimized
            .iter()
            .map(|instr| lower(instr, &constants))
            .collect();
        let mut program = Program::new(code, inputs, outputs, self.updates.len())?;
        // The constants that depend on a uniform, to compute again when one is set.
        program.bound = optimized
            .iter()
            .enumerate()
            .filter_map(|(at, instr)| match instr {
                optimizer::Instr::Const { value, .. } if names_below(value, uniforms) => {
                    Some((at, value.clone()))
                }
                _ => None,
            })
            .collect();
        program.named = constants;
        program.uniforms = uniforms;
        program.derived = self.derived;
        Ok(program)
    }
}

// ---- optimization -----------------------------------------------------------------

use tensorcrate_fusion as optimizer;
pub use tensorcrate_fusion::{Cost, CostModel, Latency};

/// Which rewrites the program optimizer may make.
///
/// Every program built with a [`Builder`], and every kernel `math!` fuses, is
/// optimized; this says whether results may change by rounding in exchange.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum Algebra {
    /// Only rewrites that leave every result bit-identical to the operations
    /// as built: merging common subexpressions, exact identities such as
    /// `x · 1 = x`, reordering, and register choices.
    Exact,
    /// Also regroup associative chains — `(a + b) + c` as `a + (b + c)`,
    /// `a / b / c` as `a / (b · c)`, constants gathered and folded — when that
    /// is cheaper. Results may differ from the operations as built by
    /// rounding. The default.
    #[default]
    Reassociate,
}

thread_local! {
    static ALGEBRA: Cell<Algebra> = const { Cell::new(Algebra::Reassociate) };
    /// Optimized programs, by structure.
    static PLANS: std::cell::RefCell<std::collections::HashMap<PlanKey, Vec<optimizer::Instr>>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// This thread's [`Algebra`].
pub fn algebra() -> Algebra {
    ALGEBRA.with(Cell::get)
}

/// Set this thread's [`Algebra`].
pub fn set_algebra(algebra: Algebra) {
    ALGEBRA.with(|cell| cell.set(algebra));
}

/// Run `f` with this thread's [`Algebra`] set to `algebra`, restoring the
/// previous one afterwards — even if `f` panics.
pub fn with_algebra<R>(algebra: Algebra, f: impl FnOnce() -> R) -> R {
    struct Restore(Algebra);
    impl Drop for Restore {
        fn drop(&mut self) {
            set_algebra(self.0);
        }
    }
    let _restore = Restore(self::algebra());
    set_algebra(algebra);
    f()
}

/// A program's structure, constants abstracted, and how it is optimized.
#[derive(Clone, PartialEq, Eq, Hash)]
struct PlanKey {
    graph: optimizer::Graph,
    reassociate: bool,
    model: [u64; 11],
}

fn model_bits(model: &CostModel) -> [u64; 11] {
    let l = &model.latency;
    [
        model.alpha,
        model.beta,
        model.gamma,
        model.delta,
        model.epsilon,
        l.load,
        l.constant,
        l.arithmetic,
        l.divide,
        l.sqrt,
        l.transcendental,
    ]
    .map(f64::to_bits)
}

/// Programs kept per thread before the cache starts over.
const PLAN_CACHE: usize = 512;

/// The optimized code for `graph`, from the cache if its structure has been
/// seen.
fn optimized(
    graph: optimizer::Graph,
    model: &CostModel,
    algebra: Algebra,
) -> Result<Vec<optimizer::Instr>, ProgramError> {
    let key = PlanKey {
        graph,
        reassociate: algebra == Algebra::Reassociate,
        model: model_bits(model),
    };
    if let Some(code) = PLANS.with(|plans| plans.borrow().get(&key).cloned()) {
        return Ok(code);
    }
    let options = optimizer::Options {
        model: *model,
        reassociate: key.reassociate,
        max_registers: REGISTERS,
        max_instructions: MAX_INSTRUCTIONS,
        ..optimizer::Options::default()
    };
    let plan =
        optimizer::optimize(&key.graph, &options).map_err(|_| ProgramError::TooManyRegisters)?;
    PLANS.with(|plans| {
        let mut plans = plans.borrow_mut();
        if plans.len() >= PLAN_CACHE {
            plans.clear();
        }
        plans.insert(key, plan.code.clone());
    });
    Ok(plan.code)
}

/// The optimizer's view of a constant. Zeros and ones are kept as literals —
/// the identities the optimizer applies depend on them — and every other value
/// is a placeholder numbered by its distinct bits, so programs that differ only
/// in such constants share a plan.
fn placeholder<T: Real>(value: T, constants: &mut Vec<T>, uniforms: usize) -> optimizer::Scalar {
    let wide = value.into_f64();
    if wide == 0.0 || wide == 1.0 || wide == -1.0 {
        return optimizer::Scalar::literal(wide);
    }
    let bits = wide.to_bits();
    // A uniform equal to it is not the same constant: the uniform may change.
    let found = constants[uniforms..]
        .iter()
        .position(|c| c.into_f64().to_bits() == bits)
        .map(|index| uniforms + index);
    let index = match found {
        Some(index) => index,
        None => {
            constants.push(value);
            constants.len() - 1
        }
    };
    optimizer::Scalar::Named(index as u32)
}

/// Whether `scalar` reads any named constant below `limit` — a uniform.
fn names_below(scalar: &optimizer::Scalar, limit: usize) -> bool {
    match scalar {
        optimizer::Scalar::Literal(_) => false,
        optimizer::Scalar::Named(index) => (*index as usize) < limit,
        optimizer::Scalar::Binary(_, a, b) => names_below(a, limit) || names_below(b, limit),
    }
}

/// A constant's value in `T`, its folds evaluated as the unfused kernels would
/// combine them.
fn evaluate<T: Real>(scalar: &optimizer::Scalar, constants: &[T]) -> T {
    match scalar {
        optimizer::Scalar::Literal(bits) => T::from_f64(f64::from_bits(*bits)),
        optimizer::Scalar::Named(index) => constants[*index as usize],
        optimizer::Scalar::Binary(op, a, b) => scalar_binary(
            binary_op(*op),
            evaluate(a, constants),
            evaluate(b, constants),
        ),
    }
}

fn bin(op: BinaryOp) -> optimizer::Bin {
    optimizer::Bin::from_code(op as u16).expect("the operation codes agree")
}

fn binary_op(op: optimizer::Bin) -> BinaryOp {
    BinaryOp::try_from(op as u16).expect("the operation codes agree")
}

fn cmp(op: Compare) -> optimizer::Cmp {
    optimizer::Cmp::from_code(op as u16).expect("the comparison codes agree")
}

/// An optimized instruction as one of ours.
fn lower<T: Real>(instr: &optimizer::Instr, constants: &[T]) -> Instr<T> {
    match *instr {
        optimizer::Instr::Load { dst, slot, remap } => Instr::Load {
            dst,
            input: slot,
            remap: match remap {
                0 => Remap::Identity,
                1 => Remap::Transpose,
                2 => Remap::Row,
                3 => Remap::Column,
                _ => unreachable!("the remap codes agree"),
            },
        },
        optimizer::Instr::Const { dst, ref value } => Instr::Const {
            dst,
            value: evaluate(value, constants),
        },
        optimizer::Instr::Binary { dst, op, a, b } => Instr::Binary {
            dst,
            op: binary_op(op),
            a,
            b,
        },
        optimizer::Instr::Unary { dst, function, a } => Instr::Unary {
            dst,
            op: Analytic::try_from(function.0).expect("the function codes agree"),
            a,
        },
        optimizer::Instr::Cmp { dst, op, a, b } => Instr::Cmp {
            dst,
            op: Compare::try_from(op as u16).expect("the comparison codes agree"),
            a,
            b,
        },
        optimizer::Instr::Store { src, output } => Instr::Store { src, output },
    }
}

impl<T: Real> Program<T> {
    /// What this program costs under `model`, as it stands: its instructions
    /// in their order and registers.
    pub fn cost(&self, model: &CostModel) -> Cost {
        let mut nodes = Vec::with_capacity(self.code.len());
        let mut stores = Vec::new();
        let mut value_of = [usize::MAX; REGISTERS];
        for instr in &self.code {
            let read = |reg: Reg| value_of[usize::from(reg)];
            let node = match *instr {
                Instr::Load { input, remap, .. } => optimizer::Node::Load {
                    slot: input,
                    remap: remap as u8,
                },
                Instr::Const { .. } => optimizer::Node::Const(optimizer::Scalar::Named(0)),
                Instr::Binary { op, a, b, .. } => {
                    optimizer::Node::Binary(bin(op), read(a), read(b))
                }
                Instr::Unary { op, a, .. } => {
                    optimizer::Node::Unary(optimizer::Function(op as u16), read(a))
                }
                Instr::Cmp { op, a, b, .. } => optimizer::Node::Cmp(cmp(op), read(a), read(b)),
                Instr::Store { src, output } => {
                    stores.push(optimizer::Store {
                        value: read(src),
                        output,
                    });
                    continue;
                }
            };
            nodes.push(node);
            if let Some(dst) = instr.dst() {
                value_of[usize::from(dst)] = nodes.len() - 1;
            }
        }
        optimizer::cost(
            &optimizer::Graph {
                nodes,
                stores,
                input_bytes: self.inputs.iter().map(|dtype| dtype.size() as u8).collect(),
                output_bytes: self
                    .outputs
                    .iter()
                    .map(|dtype| dtype.size() as u8)
                    .collect(),
            },
            model,
        )
    }
}

// ---- operands -----------------------------------------------------------------

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for half::f16 {}
    impl Sealed for half::bf16 {}
}

/// An element type a program can load and store: the four [`Real`] types.
pub trait Element: Real + sealed::Sealed {
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
element!(f64, F64);

/// Borrowed storage of one of the [`DType`]s.
#[doc(hidden)]
pub enum SourceData<'a, B: Backend> {
    F32(&'a B::Vector<f32>),
    F16(&'a B::Vector<f16>),
    Bf16(&'a B::Vector<bf16>),
    F64(&'a B::Vector<f64>),
}

// Shared references whatever `B` is, so copyable without `B: Copy`.
impl<B: Backend> Clone for SourceData<'_, B> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<B: Backend> Copy for SourceData<'_, B> {}

/// Mutably borrowed storage of one of the [`DType`]s.
#[doc(hidden)]
pub enum SinkData<'a, B: Backend> {
    F32(&'a mut B::Vector<f32>),
    F16(&'a mut B::Vector<f16>),
    Bf16(&'a mut B::Vector<bf16>),
    F64(&'a mut B::Vector<f64>),
}

/// Owned storage of one of the [`DType`]s, as a program allocates it.
#[doc(hidden)]
pub enum Fresh<B: Backend> {
    F32(B::Vector<f32>),
    F16(B::Vector<f16>),
    Bf16(B::Vector<bf16>),
    F64(B::Vector<f64>),
}

/// A program input: storage and its length.
#[doc(hidden)]
pub struct Source<'a, B: Backend> {
    pub data: SourceData<'a, B>,
    /// The elements the source holds: its length, or a view's rows times its
    /// columns.
    pub len: usize,
    /// Where the elements lie in `data`, if not simply in order from its
    /// start.
    pub view: Option<View>,
}

/// An in-place program operand: storage and its length.
#[doc(hidden)]
pub struct Sink<'a, B: Backend> {
    pub data: SinkData<'a, B>,
    pub len: usize,
}

impl<B: Backend> Source<'_, B> {
    /// Another handle on the same storage.
    pub(crate) fn reborrow(&self) -> Source<'_, B> {
        Source {
            data: self.data,
            len: self.len,
            view: self.view,
        }
    }

    fn dtype(&self) -> DType {
        match self.data {
            SourceData::F32(_) => DType::F32,
            SourceData::F16(_) => DType::F16,
            SourceData::Bf16(_) => DType::Bf16,
            SourceData::F64(_) => DType::F64,
        }
    }

    /// The storage, if it holds `T`s.
    fn typed<T: 'static>(&self) -> Option<&B::Vector<T>> {
        fn cast<B: Backend, U: 'static, T: 'static>(
            storage: &B::Vector<U>,
        ) -> Option<&B::Vector<T>> {
            // SAFETY: `U` is `T`, so the two storage types are one type.
            (TypeId::of::<U>() == TypeId::of::<T>())
                .then(|| unsafe { &*(storage as *const B::Vector<U>).cast::<B::Vector<T>>() })
        }
        match self.data {
            SourceData::F32(storage) => cast::<B, f32, T>(storage),
            SourceData::F16(storage) => cast::<B, f16, T>(storage),
            SourceData::Bf16(storage) => cast::<B, bf16, T>(storage),
            SourceData::F64(storage) => cast::<B, f64, T>(storage),
        }
    }

    pub(crate) fn slice(&self) -> Slice<'_> {
        match self.data {
            SourceData::F32(storage) => Slice::F32(B::vector_slice(storage)),
            SourceData::F16(storage) => Slice::F16(B::vector_slice(storage)),
            SourceData::Bf16(storage) => Slice::Bf16(B::vector_slice(storage)),
            SourceData::F64(storage) => Slice::F64(B::vector_slice(storage)),
        }
    }
}

impl<B: Backend> Sink<'_, B> {
    fn dtype(&self) -> DType {
        match self.data {
            SinkData::F32(_) => DType::F32,
            SinkData::F16(_) => DType::F16,
            SinkData::Bf16(_) => DType::Bf16,
            SinkData::F64(_) => DType::F64,
        }
    }

    pub(crate) fn slice(&mut self) -> SliceMut<'_> {
        match &mut self.data {
            SinkData::F32(storage) => SliceMut::F32(B::vector_slice_mut(storage)),
            SinkData::F16(storage) => SliceMut::F16(B::vector_slice_mut(storage)),
            SinkData::Bf16(storage) => SliceMut::Bf16(B::vector_slice_mut(storage)),
            SinkData::F64(storage) => SliceMut::F64(B::vector_slice_mut(storage)),
        }
    }

    fn as_source(&self) -> Source<'_, B> {
        let data = match &self.data {
            SinkData::F32(storage) => SourceData::F32(&**storage),
            SinkData::F16(storage) => SourceData::F16(&**storage),
            SinkData::Bf16(storage) => SourceData::Bf16(&**storage),
            SinkData::F64(storage) => SourceData::F64(&**storage),
        };
        Source {
            data,
            len: self.len,
            view: None,
        }
    }

    /// Replace the storage with a program's result of the same type.
    fn replace(&mut self, fresh: Fresh<B>) {
        match (&mut self.data, fresh) {
            (SinkData::F32(storage), Fresh::F32(value)) => **storage = value,
            (SinkData::F16(storage), Fresh::F16(value)) => **storage = value,
            (SinkData::Bf16(storage), Fresh::Bf16(value)) => **storage = value,
            (SinkData::F64(storage), Fresh::F64(value)) => **storage = value,
            _ => unreachable!("in-place types are checked against the program"),
        }
    }
}

/// A tensor a [`Program`] can read: a [`Vector`], a [`Matrix`] — read in
/// row-major order — or a [`MatrixView`] of one, of any [`Element`] type.
pub trait Fusable<B: Backend> {
    #[doc(hidden)]
    fn source(&self) -> Source<'_, B>;
}

/// A tensor a [`Program`] can also update in place: a [`Vector`] or a
/// [`Matrix`], which own their elements. A view cannot be updated.
pub trait FusableMut<B: Backend>: Fusable<B> {
    #[doc(hidden)]
    fn sink(&mut self) -> Sink<'_, B>;
}

impl<T: Element, B: Backend> Fusable<B> for Vector<T, B> {
    fn source(&self) -> Source<'_, B> {
        Source {
            data: T::source::<B>(self.storage()),
            len: self.len(),
            view: None,
        }
    }
}

impl<T: Element, B: Backend> FusableMut<B> for Vector<T, B> {
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
            view: None,
        }
    }
}

impl<T: Element, B: Backend> FusableMut<B> for Matrix<T, B> {
    fn sink(&mut self) -> Sink<'_, B> {
        let len = self.rows() * self.cols();
        Sink {
            data: T::sink::<B>(B::matrix_as_vector_mut(self.storage_mut())),
            len,
        }
    }
}

// ---- views ----------------------------------------------------------------------

/// Where the elements of a strided source lie in its storage: element `(r, c)`
/// of the `rows × cols` view is element `offset + r·row_stride + c·col_stride`
/// of the storage.
#[doc(hidden)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct View {
    pub offset: usize,
    pub rows: usize,
    pub cols: usize,
    pub row_stride: usize,
    pub col_stride: usize,
}

impl View {
    /// The `(row, column)` strides of the view read as a vector: along its one
    /// row or its one column. `None` if it is neither.
    fn along(&self) -> Option<usize> {
        if self.rows == 1 {
            Some(self.col_stride)
        } else if self.cols == 1 {
            Some(self.row_stride)
        } else {
            None
        }
    }
}

/// A rectangle of a [`Matrix`], read in place — a block of rows and columns,
/// a single row or column, or the whole matrix transposed — without copying
/// it.
///
/// A view is an input to a [`Program`] like any tensor: as a matrix, a
/// transposed one, or, when it is one row or one column, a vector to broadcast.
/// It is read through its strides on both backends, so a fused program over
/// part of a matrix costs what it would over a matrix of that part's size.
///
/// ```
/// use tensorcrate::tensors::fused::{Builder, DType, Remap};
/// use tensorcrate::tensors::Matrix;
///
/// // The middle two columns of each row, plus the matrix's last column.
/// let m = Matrix::from_rows([[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]);
/// let mut b = Builder::<f32>::new();
/// let block = b.input(DType::F32);
/// let last = b.input_remapped(DType::F32, Remap::Column);
/// let sum = b.add(block, last);
/// b.output(sum, DType::F32);
/// let program = b.build().unwrap();
///
/// let y = program
///     .run((2, 2), &[&m.view(.., 1..3), &m.column_view(3)], &mut [])
///     .remove(0)
///     .into_matrix::<f32>();
/// assert_eq!(y.to_rows(), [[6.0, 7.0], [14.0, 15.0]]);
/// ```
pub struct MatrixView<'a, T, B: Backend = Host> {
    matrix: &'a Matrix<T, B>,
    view: View,
}

impl<T: Copy + 'static, B: Backend> Matrix<T, B> {
    /// The rows `rows` and columns `cols` of this matrix, read in place.
    ///
    /// # Panics
    ///
    /// If either range reaches past the matrix.
    #[track_caller]
    pub fn view(
        &self,
        rows: impl std::ops::RangeBounds<usize>,
        cols: impl std::ops::RangeBounds<usize>,
    ) -> MatrixView<'_, T, B> {
        let (rows, cols) = (bounds(rows, self.rows()), bounds(cols, self.cols()));
        MatrixView {
            matrix: self,
            view: View {
                offset: rows.start * self.cols() + cols.start,
                rows: rows.len(),
                cols: cols.len(),
                row_stride: self.cols(),
                col_stride: 1,
            },
        }
    }

    /// Row `row` of this matrix, read in place: a `1 × cols` view.
    #[track_caller]
    pub fn row_view(&self, row: usize) -> MatrixView<'_, T, B> {
        self.view(row..=row, ..)
    }

    /// Column `col` of this matrix, read in place: a `rows × 1` view.
    #[track_caller]
    pub fn column_view(&self, col: usize) -> MatrixView<'_, T, B> {
        self.view(.., col..=col)
    }

    /// The transpose of this matrix, read in place: a `cols × rows` view.
    pub fn transposed_view(&self) -> MatrixView<'_, T, B> {
        self.view(.., ..).t()
    }
}

/// `range` of `0..len`, checked.
#[track_caller]
fn bounds(range: impl std::ops::RangeBounds<usize>, len: usize) -> Range<usize> {
    use std::ops::Bound;
    let start = match range.start_bound() {
        Bound::Included(&start) => start,
        Bound::Excluded(&start) => start + 1,
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&end) => end + 1,
        Bound::Excluded(&end) => end,
        Bound::Unbounded => len,
    };
    assert!(
        start <= end && end <= len,
        "view: {start}..{end} reaches past an extent of {len}"
    );
    start..end
}

impl<'a, T: Copy + 'static, B: Backend> MatrixView<'a, T, B> {
    pub fn rows(&self) -> usize {
        self.view.rows
    }

    pub fn cols(&self) -> usize {
        self.view.cols
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.view.rows, self.view.cols)
    }

    /// The rows `rows` and columns `cols` of this view, read in place.
    ///
    /// # Panics
    ///
    /// If either range reaches past the view.
    #[track_caller]
    pub fn view(
        &self,
        rows: impl std::ops::RangeBounds<usize>,
        cols: impl std::ops::RangeBounds<usize>,
    ) -> MatrixView<'a, T, B> {
        let v = self.view;
        let (rows, cols) = (bounds(rows, v.rows), bounds(cols, v.cols));
        MatrixView {
            matrix: self.matrix,
            view: View {
                offset: v.offset + rows.start * v.row_stride + cols.start * v.col_stride,
                rows: rows.len(),
                cols: cols.len(),
                ..v
            },
        }
    }

    /// This view transposed, still in place.
    pub fn t(&self) -> MatrixView<'a, T, B> {
        let v = self.view;
        MatrixView {
            matrix: self.matrix,
            view: View {
                rows: v.cols,
                cols: v.rows,
                row_stride: v.col_stride,
                col_stride: v.row_stride,
                ..v
            },
        }
    }

    /// The view's elements, copied into a matrix of their own.
    pub fn to_matrix(&self) -> Matrix<T, B> {
        let place = Place {
            offset: self.view.offset,
            row: self.view.row_stride,
            col: self.view.col_stride,
        };
        Matrix::from_storage(
            self.view.rows,
            self.view.cols,
            B::gather(
                B::matrix_as_vector(self.matrix.storage()),
                place,
                self.shape(),
            ),
        )
    }
}

impl<T: Element, B: Backend> Fusable<B> for MatrixView<'_, T, B> {
    fn source(&self) -> Source<'_, B> {
        Source {
            data: T::source::<B>(B::matrix_as_vector(self.matrix.storage())),
            len: self.view.rows * self.view.cols,
            view: Some(self.view),
        }
    }
}

/// Where element `(row, col)` of an iteration space reads a source through a
/// remap: storage element `offset + row·self.row + col·self.col`. Every remap
/// of every source, viewed or not, is one of these.
#[doc(hidden)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Place {
    pub offset: usize,
    pub row: usize,
    pub col: usize,
}

impl Place {
    /// Read through `remap` over a `rows × cols` space, a source stored in
    /// order, or `view` of it. The source is assumed checked against the
    /// remap (see [`Program::check_input`]).
    pub(crate) fn of(view: Option<View>, remap: Remap, (rows, cols): (usize, usize)) -> Place {
        match view {
            None => match remap {
                Remap::Identity => Place {
                    offset: 0,
                    row: cols,
                    col: 1,
                },
                Remap::Transpose => Place {
                    offset: 0,
                    row: 1,
                    col: rows,
                },
                Remap::Row => Place {
                    offset: 0,
                    row: 0,
                    col: 1,
                },
                Remap::Column => Place {
                    offset: 0,
                    row: 1,
                    col: 0,
                },
            },
            Some(v) => {
                let along = v.along().unwrap_or(0);
                let (row, col) = match remap {
                    Remap::Identity => (v.row_stride, v.col_stride),
                    Remap::Transpose => (v.col_stride, v.row_stride),
                    Remap::Row => (0, along),
                    Remap::Column => (along, 0),
                };
                Place {
                    offset: v.offset,
                    row,
                    col,
                }
            }
        }
    }

    /// The storage element `(row, col)` reads.
    #[inline]
    pub(crate) fn at(&self, row: usize, col: usize) -> usize {
        self.offset + row * self.row + col * self.col
    }

    /// Whether the elements of a space `cols` wide lie in order from
    /// `offset`, as an unviewed tensor read without a remap does.
    pub(crate) fn in_order(&self, cols: usize) -> bool {
        self.col == 1 && (self.row == cols || cols == 0)
    }

    /// The last storage element a `rows × cols` space reads, plus one.
    pub(crate) fn end(&self, (rows, cols): (usize, usize)) -> usize {
        if rows == 0 || cols == 0 {
            self.offset
        } else {
            self.at(rows - 1, cols - 1) + 1
        }
    }
}

/// A fresh result of a [`Program`], shaped like the iteration space.
pub struct Output<B: Backend> {
    shape: (usize, usize),
    data: Fresh<B>,
}

impl<B: Backend> Output<B> {
    pub(crate) fn new(shape: (usize, usize), data: Fresh<B>) -> Self {
        Output { shape, data }
    }

    pub fn dtype(&self) -> DType {
        match self.data {
            Fresh::F32(_) => DType::F32,
            Fresh::F16(_) => DType::F16,
            Fresh::Bf16(_) => DType::Bf16,
            Fresh::F64(_) => DType::F64,
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
enum Register<B: Backend, T: Real> {
    Scalar(T),
    Tensor(Matrix<T, B>),
}

impl<B: Kernels<T>, T: Real> Register<B, T> {
    /// The value as a tensor of its own, filling a constant out to the shape.
    fn materialize(&self, (rows, cols): (usize, usize)) -> Matrix<T, B> {
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
fn step<'r, B: Kernels<T>, T: Real>(
    instr: &Instr<T>,
    registers: &'r mut [Option<Register<B, T>>],
    load: impl FnOnce(usize, Remap) -> Matrix<T, B>,
) -> Option<&'r Register<B, T>> {
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

/// A load as unfused kernels: convert if needed, then materialize the remap —
/// a transpose kernel, or a stack of copies for a broadcast.
fn load_unfused<B: Kernels<T>, T: Real>(
    source: &Source<'_, B>,
    shape: (usize, usize),
    remap: Remap,
) -> Matrix<T, B> {
    let (rows, cols) = shape;
    // An input already of the program's type is copied on its own backend —
    // on Metal, on the GPU — rather than read back through the CPU, which on a
    // device would wait for every queued kernel.
    let widened = match source.typed::<T>() {
        Some(_) => None,
        None => Some(B::store_vector(&source.slice().widen::<T>())),
    };
    let storage = widened
        .as_ref()
        .or(source.typed::<T>())
        .expect("one of the two");
    let place = Place::of(source.view, remap, shape);
    let stored = if source.view.is_none() && place.in_order(cols) {
        B::vector_into_matrix(B::duplicate(storage))
    } else {
        // A view, a transpose or a broadcast: every element copied into
        // place, exactly.
        B::gather(storage, place, shape)
    };
    Matrix::from_storage(rows, cols, stored)
}

/// `storage` reinterpreted as storage of `U`, when `T` is `U`.
fn retype<B: Backend, T: 'static, U: 'static>(
    storage: B::Vector<T>,
) -> Result<B::Vector<U>, B::Vector<T>> {
    if TypeId::of::<T>() != TypeId::of::<U>() {
        return Err(storage);
    }
    let storage = std::mem::ManuallyDrop::new(storage);
    // SAFETY: `T` is `U`, so the two storage types are one type; the value is
    // moved, since the original is never dropped.
    Ok(unsafe { std::mem::transmute_copy::<B::Vector<T>, B::Vector<U>>(&storage) })
}

fn scalar_binary<T: Real>(op: BinaryOp, a: T, b: T) -> T {
    match op {
        BinaryOp::Add => a + b,
        BinaryOp::Sub => a - b,
        BinaryOp::Mul => a * b,
        BinaryOp::Div => a / b,
        BinaryOp::Rem => a % b,
    }
}

/// Convert a `T` tensor to an output's storage type.
fn narrow<B: Kernels<T>, T: Real>(tensor: Matrix<T, B>, dtype: DType) -> Fresh<B> {
    let storage = B::matrix_into_flattened(tensor.into_storage());
    // Storage of the program's own type is the result as it stands.
    let storage = match dtype {
        DType::F32 => retype::<B, T, f32>(storage).map(Fresh::F32),
        DType::F16 => retype::<B, T, f16>(storage).map(Fresh::F16),
        DType::Bf16 => retype::<B, T, bf16>(storage).map(Fresh::Bf16),
        DType::F64 => retype::<B, T, f64>(storage).map(Fresh::F64),
    };
    let storage = match storage {
        Ok(fresh) => return fresh,
        Err(storage) => storage,
    };
    let values = B::vector_slice(&storage);
    match dtype {
        DType::F32 => Fresh::F32(B::vector_from_vec(convert(values))),
        DType::F16 => Fresh::F16(B::vector_from_vec(convert(values))),
        DType::Bf16 => Fresh::Bf16(B::vector_from_vec(convert(values))),
        DType::F64 => Fresh::F64(B::vector_from_vec(convert(values))),
    }
}

/// Run a program as one existing kernel per instruction.
fn unfused<B: Kernels<T>, T: Real>(
    program: &Program<T>,
    shape: (usize, usize),
    inputs: &[Source<'_, B>],
    updated: &mut [Sink<'_, B>],
) -> Vec<Fresh<B>> {
    let fresh_inputs = program.fresh_inputs();
    let mut registers: Vec<Option<Register<B, T>>> = (0..REGISTERS).map(|_| None).collect();
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
        step::<B, T>(instr, &mut registers, |slot, remap| {
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

// ---- element conversion ---------------------------------------------------------

/// `values` as `U`s: a copy when the types are the same, and otherwise one
/// rounding per element, through the exact `f64` widening.
fn convert<T: Real, U: Real>(values: &[T]) -> Vec<U> {
    match same::<T, U>(values) {
        Some(values) => values.to_vec(),
        None => values.iter().map(|&x| U::from_f64(x.into_f64())).collect(),
    }
}

/// Convert `values` into `out`, element for element.
fn convert_into<T: Real, U: Real>(values: &[T], out: &mut [U]) {
    match same::<T, U>(values) {
        Some(values) => out.copy_from_slice(values),
        None => {
            for (out, &x) in out.iter_mut().zip(values) {
                *out = U::from_f64(x.into_f64());
            }
        }
    }
}

/// `&[T]` as `&[U]` when they are the same type. A no-op conversion is the
/// common case — `f32` data under `f32` arithmetic — and this is what keeps it
/// a `memcpy`.
fn same<T: 'static, U: 'static>(values: &[T]) -> Option<&[U]> {
    // SAFETY: equal `TypeId`s mean one type, so the layouts are identical.
    (TypeId::of::<T>() == TypeId::of::<U>())
        .then(|| unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<U>(), values.len()) })
}

/// The mutable form of [`same`].
#[cfg_attr(
    not(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64"))),
    allow(dead_code)
)]
fn same_mut<T: 'static, U: 'static>(values: &mut [T]) -> Option<&mut [U]> {
    // SAFETY: as in `same`.
    (TypeId::of::<T>() == TypeId::of::<U>()).then(|| unsafe {
        std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<U>(), values.len())
    })
}

// ---- the host tile interpreter ------------------------------------------------

/// The fewest elements per tile. Sixteen registers of this many `f32`s plus a
/// spare fill 68 KB, which stays in an Apple-silicon L1 and a typical x86 L2.
const TILE: usize = 1024;

/// The most elements per tile.
const MAX_TILE: usize = 16 * 1024;

/// The scratch a tile's registers may fill. A program with few registers gets
/// longer tiles inside this budget, which spreads the per-tile cost of
/// interpreting — a dozen nanoseconds — over more elements: a one-operation
/// program over a long tensor otherwise pays it often enough to show.
const TILE_BYTES: usize = 64 * 1024;

thread_local! {
    /// The interpreter's register tiles, kept between programs so that a small
    /// one does not pay for allocating and zeroing them every time.
    static SCRATCH: std::cell::RefCell<Vec<u64>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Run `f` over `len` scratch elements of `T`. Their values are whatever the
/// last program left — every bit pattern is a valid float, and the interpreter
/// writes a tile before it reads it.
fn with_scratch<T: Real, R>(len: usize, f: impl FnOnce(&mut [T]) -> R) -> R {
    assert!(size_of::<T>() <= 8 && align_of::<T>() <= 8 && 8 % size_of::<T>() == 0);
    let words = (len * size_of::<T>()).div_ceil(8);
    let cast = |words: &mut [u64]| {
        // SAFETY: `T` is a float of at most eight bytes dividing eight, so the
        // words hold `len` of them, suitably aligned, and any bits are valid.
        unsafe { std::slice::from_raw_parts_mut(words.as_mut_ptr().cast::<T>(), len) }
    };
    SCRATCH.with(|cell| match cell.try_borrow_mut() {
        Ok(mut words_held) => {
            if words_held.len() < words {
                words_held.resize(words, 0);
            }
            f(cast(&mut words_held[..words]))
        }
        // Re-entered — a kernel that runs a program of its own — so use a
        // scratch of this call's own.
        Err(_) => f(cast(&mut vec![0u64; words])),
    })
}

/// Elements per tile for `physical` register tiles of `T`.
fn tile_len<T>(physical: usize, len: usize) -> usize {
    let budget = TILE_BYTES / (physical * size_of::<T>()).max(1);
    let tile = budget.clamp(TILE, MAX_TILE) / 64 * 64;
    tile.min(len.max(1))
}

/// A typed input slice.
#[doc(hidden)]
pub enum Slice<'a> {
    F32(&'a [f32]),
    F16(&'a [f16]),
    Bf16(&'a [bf16]),
    F64(&'a [f64]),
}

/// A typed output slice.
#[doc(hidden)]
pub enum SliceMut<'a> {
    F32(&'a mut [f32]),
    F16(&'a mut [f16]),
    Bf16(&'a mut [bf16]),
    F64(&'a mut [f64]),
}

impl Slice<'_> {
    /// The values, if they are already `T`s.
    fn typed<T: 'static>(&self) -> Option<&[T]> {
        match self {
            Slice::F32(values) => same(values),
            Slice::F16(values) => same(values),
            Slice::Bf16(values) => same(values),
            Slice::F64(values) => same(values),
        }
    }

    fn widen<T: Real>(&self) -> Vec<T> {
        match self {
            Slice::F32(values) => convert(values),
            Slice::F16(values) => convert(values),
            Slice::Bf16(values) => convert(values),
            Slice::F64(values) => convert(values),
        }
    }

    #[inline]
    fn get<T: Real>(&self, index: usize) -> T {
        match self {
            Slice::F32(values) => T::from_f64(f64::from(values[index])),
            Slice::F16(values) => T::from_f64(values[index].to_f64()),
            Slice::Bf16(values) => T::from_f64(values[index].to_f64()),
            Slice::F64(values) => T::from_f64(values[index]),
        }
    }

    /// Convert `out.len()` contiguous elements from `start` into `out`.
    fn copy_into<T: Real>(&self, start: usize, out: &mut [T]) {
        let end = start + out.len();
        match self {
            Slice::F32(values) => convert_into(&values[start..end], out),
            Slice::F16(values) => convert_into(&values[start..end], out),
            Slice::Bf16(values) => convert_into(&values[start..end], out),
            Slice::F64(values) => convert_into(&values[start..end], out),
        }
    }

    /// Convert `out.len()` elements of a `cols`-wide space from flat index
    /// `start`, each read from this storage where `place` says, into `out`.
    ///
    /// A run of the space within one row reads storage a fixed step apart: a
    /// step of one is a contiguous run, copied as one; zero, one value
    /// repeated — a column broadcast; anything else, a strided walk. So no
    /// element pays more than an addition to find.
    fn gather<T: Real>(&self, start: usize, cols: usize, place: Place, out: &mut [T]) {
        if place.in_order(cols) {
            return self.copy_into(place.offset + start, out);
        }
        let mut offset = 0;
        while offset < out.len() {
            let (row, col) = ((start + offset) / cols, (start + offset) % cols);
            let run = (cols - col).min(out.len() - offset);
            let segment = &mut out[offset..offset + run];
            let first = place.at(row, col);
            match place.col {
                1 => self.copy_into(first, segment),
                0 => segment.fill(self.get(first)),
                step => {
                    for (k, out) in segment.iter_mut().enumerate() {
                        *out = self.get(first + k * step);
                    }
                }
            }
            offset += run;
        }
    }

    /// The elements the storage holds.
    fn len(&self) -> usize {
        match self {
            Slice::F32(values) => values.len(),
            Slice::F16(values) => values.len(),
            Slice::Bf16(values) => values.len(),
            Slice::F64(values) => values.len(),
        }
    }
}

impl SliceMut<'_> {
    /// The values, if they are `T`s.
    fn typed<T: 'static>(&mut self) -> Option<&mut [T]> {
        match self {
            SliceMut::F32(values) => same_mut(values),
            SliceMut::F16(values) => same_mut(values),
            SliceMut::Bf16(values) => same_mut(values),
            SliceMut::F64(values) => same_mut(values),
        }
    }

    fn view(&self) -> Slice<'_> {
        match self {
            SliceMut::F32(values) => Slice::F32(values),
            SliceMut::F16(values) => Slice::F16(values),
            SliceMut::Bf16(values) => Slice::Bf16(values),
            SliceMut::F64(values) => Slice::F64(values),
        }
    }

    /// Convert `values` into this output from `start`.
    fn scatter<T: Real>(&mut self, start: usize, values: &[T]) {
        let end = start + values.len();
        match self {
            SliceMut::F32(out) => convert_into(values, &mut out[start..end]),
            SliceMut::F16(out) => convert_into(values, &mut out[start..end]),
            SliceMut::Bf16(out) => convert_into(values, &mut out[start..end]),
            SliceMut::F64(out) => convert_into(values, &mut out[start..end]),
        }
    }

    fn fill<T: Real>(&mut self, start: usize, len: usize, value: T) {
        let end = start + len;
        match self {
            SliceMut::F32(out) => out[start..end].fill(f32::from_f64(value.into_f64())),
            SliceMut::F16(out) => out[start..end].fill(f16::from_f64(value.into_f64())),
            SliceMut::Bf16(out) => out[start..end].fill(bf16::from_f64(value.into_f64())),
            SliceMut::F64(out) => out[start..end].fill(value.into_f64()),
        }
    }
}

/// An output's storage, for threads that each write a disjoint window of it.
#[derive(Clone, Copy)]
enum Window {
    F32(Shared<f32>),
    F16(Shared<f16>),
    Bf16(Shared<bf16>),
    F64(Shared<f64>),
}

impl SliceMut<'_> {
    fn window(&mut self) -> Window {
        match self {
            SliceMut::F32(values) => Window::F32(Shared(values.as_mut_ptr())),
            SliceMut::F16(values) => Window::F16(Shared(values.as_mut_ptr())),
            SliceMut::Bf16(values) => Window::Bf16(Shared(values.as_mut_ptr())),
            SliceMut::F64(values) => Window::F64(Shared(values.as_mut_ptr())),
        }
    }
}

impl Window {
    /// The elements `range` of the output.
    ///
    /// # Safety
    ///
    /// `range` lies within the output, which outlives the slice, and no other
    /// slice of those elements is in use.
    unsafe fn slice<'a>(self, range: Range<usize>) -> SliceMut<'a> {
        let len = range.len();
        unsafe {
            match self {
                Window::F32(base) => {
                    SliceMut::F32(std::slice::from_raw_parts_mut(base.at(range.start), len))
                }
                Window::F16(base) => {
                    SliceMut::F16(std::slice::from_raw_parts_mut(base.at(range.start), len))
                }
                Window::Bf16(base) => {
                    SliceMut::Bf16(std::slice::from_raw_parts_mut(base.at(range.start), len))
                }
                Window::F64(base) => {
                    SliceMut::F64(std::slice::from_raw_parts_mut(base.at(range.start), len))
                }
            }
        }
    }
}

/// Allocate a program's fresh outputs.
fn allocate<T>(program: &Program<T>, len: usize) -> Vec<Owned> {
    program.outputs[program.updated..]
        .iter()
        .map(|dtype| match dtype {
            DType::F32 => Owned::F32(vec![0.0; len]),
            DType::F16 => Owned::F16(vec![f16::ZERO; len]),
            DType::Bf16 => Owned::Bf16(vec![bf16::ZERO; len]),
            DType::F64 => Owned::F64(vec![0.0; len]),
        })
        .collect()
}

enum Owned {
    F32(Vec<f32>),
    F16(Vec<f16>),
    Bf16(Vec<bf16>),
    F64(Vec<f64>),
}

impl Owned {
    fn slice(&mut self) -> SliceMut<'_> {
        match self {
            Owned::F32(values) => SliceMut::F32(values),
            Owned::F16(values) => SliceMut::F16(values),
            Owned::Bf16(values) => SliceMut::Bf16(values),
            Owned::F64(values) => SliceMut::F64(values),
        }
    }

    fn store<B: Backend>(self) -> Fresh<B> {
        match self {
            Owned::F32(values) => Fresh::F32(B::vector_from_vec(values)),
            Owned::F16(values) => Fresh::F16(B::vector_from_vec(values)),
            Owned::Bf16(values) => Fresh::Bf16(B::vector_from_vec(values)),
            Owned::F64(values) => Fresh::F64(B::vector_from_vec(values)),
        }
    }
}

/// A vector of the program's own type as a program input.
pub(crate) fn source_of<B: Backend, T: Real>(vector: &Vector<T, B>) -> Source<'_, B> {
    let storage = vector.storage();
    // SAFETY (each cast): `T` is the type compared with, so `B::Vector<T>` is
    // the storage type cast to.
    let data = unsafe {
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            SourceData::F32(&*std::ptr::from_ref(storage).cast::<B::Vector<f32>>())
        } else if TypeId::of::<T>() == TypeId::of::<f16>() {
            SourceData::F16(&*std::ptr::from_ref(storage).cast::<B::Vector<f16>>())
        } else if TypeId::of::<T>() == TypeId::of::<bf16>() {
            SourceData::Bf16(&*std::ptr::from_ref(storage).cast::<B::Vector<bf16>>())
        } else {
            assert_eq!(TypeId::of::<T>(), TypeId::of::<f64>(), "a float type");
            SourceData::F64(&*std::ptr::from_ref(storage).cast::<B::Vector<f64>>())
        }
    };
    Source {
        data,
        len: vector.len(),
        view: None,
    }
}

/// The input as a `rows × cols` matrix of `T`: a copy, converted if it is
/// stored as another type.
pub(crate) fn input_matrix<B: Kernels<T>, T: Real>(
    source: &Source<'_, B>,
    shape: (usize, usize),
) -> Matrix<T, B> {
    load_unfused(source, shape, Remap::Identity)
}

/// The row statistics a program computes for itself, from its given
/// `inputs`, in the order of their input slots — each pair of moments computed
/// once, by [`Kernels::row_moments`].
pub(crate) fn row_statistics<B: Kernels<T>, T: Real>(
    program: &Program<T>,
    shape: (usize, usize),
    inputs: &[Source<'_, B>],
) -> Vec<Vector<T, B>> {
    // Each input's mean and deviations, taken as the statistics read them.
    type Moments<T, B> = (u8, Option<Vector<T, B>>, Option<Vector<T, B>>);
    let mut moments: Vec<Moments<T, B>> = Vec::new();
    for &(slot, _) in &program.derived {
        if !moments.iter().any(|&(of, ..)| of == slot) {
            let (mean, deviations) = B::row_moments(&inputs[usize::from(slot)], shape);
            moments.push((slot, Some(mean), Some(deviations)));
        }
    }
    program
        .derived
        .iter()
        .map(|&(slot, statistic)| {
            let (_, mean, deviations) = moments
                .iter_mut()
                .find(|(of, ..)| *of == slot)
                .expect("computed above");
            match statistic {
                RowStatistic::Mean => mean.take(),
                RowStatistic::Deviations => deviations.take(),
            }
            .expect("each statistic is read once")
        })
        .collect()
}

/// [`row_moments`](Kernels::row_moments) on the host, read in place when the
/// input is stored as `T`.
pub(crate) fn host_row_moments<T: Real>(
    source: &Source<'_, Host>,
    shape: (usize, usize),
) -> (Vector<T, Host>, Vector<T, Host>) {
    let moments = match source.typed::<T>().filter(|_| source.view.is_none()) {
        Some(values) => crate::statistics::axis_moments_of(&values[..], shape, Axis::Rows),
        None => input_matrix::<Host, T>(source, shape).moments_axis(Axis::Rows),
    };
    (moments.means, moments.sum_squared_deviations)
}

/// The sums of `matrix` along `axis`, as products with ones — the definition
/// of [`Program::run_sum`].
pub(crate) fn axis_sum<T: Real, B: Kernels<T>>(matrix: Matrix<T, B>, axis: Axis) -> Vector<T, B> {
    let (rows, cols) = matrix.shape();
    match axis {
        Axis::Rows => B::matvec(&matrix, &Vector::filled(cols, T::one())),
        Axis::Columns => B::vecmat(&Vector::filled(rows, T::one()), &matrix),
    }
}

/// Run a program on the CPU, over whatever backend's storage, reading it in
/// place. This is the [`Host`] implementation and the fallback for a Metal
/// tensor that is not device-resident.
pub(crate) fn interpret_on<B: Backend, T: Real>(
    program: &Program<T>,
    shape: (usize, usize),
    inputs: &[Source<'_, B>],
    updated: &mut [Sink<'_, B>],
) -> Vec<Fresh<B>> {
    let len = shape.0 * shape.1;
    let mut fresh = allocate(program, len);
    {
        let views: Vec<Option<View>> = inputs.iter().map(|input| input.view).collect();
        let inputs: Vec<Slice<'_>> = inputs.iter().map(Source::slice).collect();
        let mut outputs: Vec<SliceMut<'_>> = updated.iter_mut().map(Sink::slice).collect();
        outputs.extend(fresh.iter_mut().map(Owned::slice));
        let viewed = views.iter().any(Option::is_some);
        if viewed || !single_operation(program, &inputs, &mut outputs) {
            interpret(program, shape, &inputs, &views, &mut outputs);
        }
    }
    fresh.into_iter().map(Owned::store).collect()
}

/// Run a program of one operation on fresh operands of its own type — a lone
/// activation such as `max(x, 0)` — as the kernel that operation is unfused,
/// over the whole tensor at once, which is what the interpreter would run tile
/// by tile. Whether it did.
///
/// The interpreter's setup and its per-tile dispatch are small, but so is the
/// operation, and on a short tensor they would show.
#[cfg(not(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64"))))]
fn single_operation<T: Real>(
    _program: &Program<T>,
    _inputs: &[Slice<'_>],
    _outputs: &mut [SliceMut<'_>],
) -> bool {
    // Without the SIMD tier the interpreter's kernels are the scalar loops,
    // and there is nothing faster to hand the operation to.
    false
}

#[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
fn single_operation<T: Real>(
    program: &Program<T>,
    inputs: &[Slice<'_>],
    outputs: &mut [SliceMut<'_>],
) -> bool {
    use crate::tensors::simd_dispatch as direct;
    if program.updated != 0 || outputs.len() != 1 || inputs.len() > 2 {
        return false;
    }
    let Some(out) = outputs[0].typed::<T>() else {
        return false;
    };
    let operand = |slot: u8| {
        inputs
            .get(usize::from(slot))
            .and_then(|input| input.typed::<T>())
    };
    match program.code[..] {
        [
            Instr::Load {
                dst: x,
                input,
                remap: Remap::Identity,
            },
            Instr::Unary { dst, op, a },
            Instr::Store { src, output: 0 },
        ] if a == x && src == dst => {
            let Some(values) = operand(input) else {
                return false;
            };
            crate::vmath::unary_parallel(op, values, out);
            true
        }
        [
            Instr::Load {
                dst: x,
                input,
                remap: Remap::Identity,
            },
            Instr::Const { dst: c, value },
            ref operation,
            Instr::Store { src, output: 0 },
        ] if x != c && operation.dst() == Some(src) => {
            let Some(values) = operand(input) else {
                return false;
            };
            match *operation {
                Instr::Binary { op, a, b, .. } if (a, b) == (x, c) || (a, b) == (c, x) => {
                    direct::broadcast(values, value, op, a == c, out)
                }
                Instr::Cmp { op, a, b, .. } if (a, b) == (x, c) || (a, b) == (c, x) => {
                    direct::compare_scalar(values, value, op, a == c, out)
                }
                _ => false,
            }
        }
        [
            Instr::Load {
                dst: x,
                input: first,
                remap: Remap::Identity,
            },
            Instr::Load {
                dst: y,
                input: second,
                remap: Remap::Identity,
            },
            ref operation,
            Instr::Store { src, output: 0 },
        ] if x != y && operation.dst() == Some(src) => {
            let (Some(xs), Some(ys)) = (operand(first), operand(second)) else {
                return false;
            };
            match *operation {
                Instr::Binary { op, a, b, .. } if (a, b) == (x, y) => {
                    direct::elementwise(xs, ys, op, out)
                }
                Instr::Binary { op, a, b, .. } if (a, b) == (y, x) => {
                    direct::elementwise(ys, xs, op, out)
                }
                Instr::Cmp { op, a, b, .. } if (a, b) == (x, y) => direct::compare(xs, ys, op, out),
                Instr::Cmp { op, a, b, .. } if (a, b) == (y, x) => direct::compare(ys, xs, op, out),
                _ => false,
            }
        }
        _ => false,
    }
}

/// The tile interpreter.
///
/// The space is walked in tiles of [`TILE`] elements or more ([`tile_len`]). Within a tile each
/// instruction runs the same vectorized kernel its unfused counterpart would,
/// over tile-sized register slices that stay in cache, so the interpretation
/// cost — one dispatch per instruction per tile — is spread over a thousand
/// elements. Constants never fill a register: an operation with one becomes the
/// scalar-broadcast kernel, exactly as in unfused code.
///
/// Registers avoid copies where they can (see [`Plan`]): a load of an input
/// already in the program's type reads the input where it lies, and an
/// operation whose result is only stored writes it straight into the output.
/// A one-operation program is then one pass over memory, like the kernel it
/// replaces, rather than a copy in, the operation, and a copy out.
///
/// `outputs` holds the in-place tensors first; loads of the last
/// `program.updated()` input slots read them.
fn interpret<T: Real>(
    program: &Program<T>,
    shape: (usize, usize),
    inputs: &[Slice<'_>],
    views: &[Option<View>],
    outputs: &mut [SliceMut<'_>],
) {
    let len = shape.0 * shape.1;
    let plan = Plan::new(program, shape, inputs, views, outputs);
    // One physical tile per register, plus a spare that receives each result
    // so an instruction may overwrite one of its own operands.
    let physical = program.registers + 1;
    let tile_len = tile_len::<T>(physical, len);
    let grain = parallel_grain(program, tile_len);
    if len < 2 * grain || crate::parallel::threads() <= 1 || !crate::parallel::plain_float::<T>() {
        with_scratch::<T, _>(physical * tile_len, |scratch| {
            run_tiles(
                program,
                shape,
                inputs,
                outputs,
                &plan,
                scratch,
                tile_len,
                0..len,
            )
        });
        return;
    }
    // Every element is computed from the same element of each input — or a
    // broadcast of one, which is only read — and written to the same element
    // of each output, so ranges of tiles can run on separate threads. The
    // tensors updated in place are loaded without a remap, so a range reads
    // only its own window of them.
    let windows: Vec<Window> = outputs.iter_mut().map(SliceMut::window).collect();
    // SAFETY: `T` is a plain float; each range writes only its own window of
    // each output; the program, plan and inputs are only read.
    unsafe {
        crate::parallel::for_ranges_unchecked(len, grain, tile_len, |range| {
            let mut outputs: Vec<SliceMut<'_>> = windows
                .iter()
                .map(|window| window.slice(range.clone()))
                .collect();
            with_scratch::<T, _>(physical * tile_len, |scratch| {
                run_tiles(
                    program,
                    shape,
                    inputs,
                    &mut outputs,
                    &plan,
                    scratch,
                    tile_len,
                    range,
                )
            });
        });
    }
}

/// Element-instructions worth giving a thread of their own.
const PARALLEL_WORK: usize = 64 * 1024;

/// The fewest elements worth giving a thread however costly each is: a
/// shorter tensor is likely still in the calling core's cache, and splitting
/// it would only move it to other cores.
const PARALLEL_ELEMENTS: usize = 8 * 1024;

/// The fewest elements of `program` worth giving a thread: a whole number of
/// tiles, more of them the cheaper each element is.
fn parallel_grain<T>(program: &Program<T>, tile_len: usize) -> usize {
    let per_element: usize = program
        .code
        .iter()
        .map(|instr| match instr {
            Instr::Const { .. } => 0,
            Instr::Unary { op, .. } if *op != Analytic::Sqrt => 8,
            Instr::Binary {
                op: BinaryOp::Div | BinaryOp::Rem,
                ..
            } => 2,
            _ => 1,
        })
        .sum();
    let elements = (PARALLEL_WORK / per_element.max(1)).max(PARALLEL_ELEMENTS);
    elements.div_ceil(tile_len).max(1) * tile_len
}

/// The tile loop of [`interpret`] over the elements `range` — a whole number
/// of tiles from the start of the space — with `scratch` holding one `tile_len`
/// tile per physical register. `outputs` are the windows `range` of the
/// program's outputs.
#[allow(clippy::too_many_arguments)]
fn run_tiles<T: Real>(
    program: &Program<T>,
    shape: (usize, usize),
    inputs: &[Slice<'_>],
    outputs: &mut [SliceMut<'_>],
    plan: &Plan,
    scratch: &mut [T],
    tile_len: usize,
    range: Range<usize>,
) {
    let fresh_inputs = program.fresh_inputs();
    let physical = program.registers + 1;
    let mut map: [usize; REGISTERS] = std::array::from_fn(|reg| reg.min(physical - 1));
    let mut spare = physical - 1;
    let mut constant: [Option<T>; REGISTERS] = [None; REGISTERS];
    // A register whose value lies outside scratch — in an input it was loaded
    // from, or an output it was written to — for the current tile. Its own
    // tile in `map` is then unused, and comes back into play when it is next
    // written through the spare.
    let mut outside: [Option<*const T>; REGISTERS] = [None; REGISTERS];

    let mut start = range.start;
    while start < range.end {
        let n = tile_len.min(range.end - start);
        // Where the tile lies in the windows of the outputs.
        let local = start - range.start;
        // The tiles are disjoint `n`-long windows of `scratch`.
        let base = scratch.as_mut_ptr();
        let tile = |physical: usize| unsafe { base.add(physical * tile_len) };
        // SAFETY (for every `view`): a register's value is `n` elements, in its
        // own scratch tile or in the window `[start, start + n)` of an input or
        // output, and no instruction writes the memory it reads.
        let place = |reg: Reg, outside: &[Option<*const T>; REGISTERS], map: &[usize]| {
            outside[reg as usize].unwrap_or_else(|| tile(map[reg as usize]).cast_const())
        };

        for (index, instr) in program.code.iter().enumerate() {
            // Where a computed result goes: its output, if the plan sends it
            // there, and otherwise the spare tile.
            let target = |outputs: &mut [SliceMut<'_>]| -> (*mut T, bool) {
                match plan.direct[index] {
                    Some(output) => {
                        let values = outputs[output].typed::<T>().expect("planned");
                        (unsafe { values.as_mut_ptr().add(local) }, true)
                    }
                    None => (tile(spare), false),
                }
            };
            match *instr {
                Instr::Load { dst, input, .. } => {
                    let slot = usize::from(input);
                    let place = plan.places[index];
                    constant[dst as usize] = None;
                    if plan.alias[index] {
                        let values = inputs[slot].typed::<T>().expect("planned");
                        // SAFETY: the plan checked that the space's elements
                        // lie within the input, in order from `place.offset`.
                        outside[dst as usize] =
                            Some(unsafe { values.as_ptr().add(place.offset + start) });
                        continue;
                    }
                    outside[dst as usize] = None;
                    // SAFETY: a register's tile is a distinct window of scratch.
                    let out = unsafe { std::slice::from_raw_parts_mut(tile(map[dst as usize]), n) };
                    if slot < fresh_inputs {
                        inputs[slot].gather(start, shape.1, place, out);
                    } else {
                        // An updated tensor, read in order from its window.
                        outputs[slot - fresh_inputs].view().copy_into(local, out);
                    }
                }
                Instr::Const { dst, value } => {
                    constant[dst as usize] = Some(value);
                    outside[dst as usize] = None;
                }
                Instr::Binary { dst, op, a, b } => {
                    let result = match (constant[a as usize], constant[b as usize]) {
                        (Some(a), Some(b)) => Ok(scalar_binary(op, a, b)),
                        (ca, cb) => {
                            let (out, direct) = target(outputs);
                            // SAFETY: the target is the spare tile, which no
                            // register maps, or an output no register reads
                            // yet, so it cannot overlap an operand.
                            let out = unsafe { std::slice::from_raw_parts_mut(out, n) };
                            let view = |reg: Reg| unsafe {
                                std::slice::from_raw_parts(place(reg, &outside, &map), n)
                            };
                            match (ca, cb) {
                                (None, Some(b)) => kernel::broadcast(view(a), b, op, false, out),
                                (Some(a), None) => kernel::broadcast(view(b), a, op, true, out),
                                _ => kernel::elementwise(view(a), view(b), op, out),
                            }
                            Err(direct.then_some(out.as_ptr()))
                        }
                    };
                    commit(
                        dst,
                        result,
                        &mut map,
                        &mut spare,
                        &mut constant,
                        &mut outside,
                    );
                }
                Instr::Unary { dst, op, a } => {
                    let result = match constant[a as usize] {
                        Some(a) => Ok(op.value(a)),
                        None => {
                            let (out, direct) = target(outputs);
                            let out = unsafe { std::slice::from_raw_parts_mut(out, n) };
                            let a =
                                unsafe { std::slice::from_raw_parts(place(a, &outside, &map), n) };
                            kernel::unary(a, op, out);
                            Err(direct.then_some(out.as_ptr()))
                        }
                    };
                    commit(
                        dst,
                        result,
                        &mut map,
                        &mut spare,
                        &mut constant,
                        &mut outside,
                    );
                }
                Instr::Cmp { dst, op, a, b } => {
                    let result = match (constant[a as usize], constant[b as usize]) {
                        (Some(a), Some(b)) => Ok(op.value(a, b)),
                        (ca, cb) => {
                            let (out, direct) = target(outputs);
                            let out = unsafe { std::slice::from_raw_parts_mut(out, n) };
                            let view = |reg: Reg| unsafe {
                                std::slice::from_raw_parts(place(reg, &outside, &map), n)
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
                            Err(direct.then_some(out.as_ptr()))
                        }
                    };
                    commit(
                        dst,
                        result,
                        &mut map,
                        &mut spare,
                        &mut constant,
                        &mut outside,
                    );
                }
                Instr::Store { src, output } => {
                    let output = &mut outputs[usize::from(output)];
                    match constant[src as usize] {
                        Some(value) => output.fill(local, n, value),
                        None => {
                            let from = place(src, &outside, &map);
                            // Already there if the plan wrote it in place.
                            let there = output.typed::<T>().is_some_and(|values| {
                                std::ptr::eq(from, unsafe { values.as_ptr().add(local) })
                            });
                            if !there {
                                let values = unsafe { std::slice::from_raw_parts(from, n) };
                                output.scatter(local, values);
                            }
                        }
                    }
                }
            }
        }
        start += n;
    }
}

/// Which copies the tile interpreter can skip for one program over one set of
/// operands. Both are decided per instruction, once, before any tile runs.
struct Plan {
    /// Where each load reads its input (see [`Place`]); the default for every
    /// other instruction.
    places: Vec<Place>,
    /// A load reads its input in place: the input is already of the program's
    /// type, the space's elements lie in it in order, and it is not one
    /// updated in place (whose memory an earlier tile's stores may have been
    /// written to).
    alias: Vec<bool>,
    /// A computed result is written straight into this output: the output is
    /// of the program's type, the result is what a later store sends there, and
    /// — for a tensor updated in place — nothing loads that tensor afterwards.
    direct: Vec<Option<usize>>,
}

impl Plan {
    fn new<T: Real>(
        program: &Program<T>,
        shape: (usize, usize),
        inputs: &[Slice<'_>],
        views: &[Option<View>],
        outputs: &mut [SliceMut<'_>],
    ) -> Self {
        let code = &program.code;
        let fresh_inputs = program.fresh_inputs();
        let places: Vec<Place> = code
            .iter()
            .map(|instr| match *instr {
                Instr::Load { input, remap, .. } if usize::from(input) < fresh_inputs => {
                    let slot = usize::from(input);
                    let place = Place::of(views.get(slot).copied().flatten(), remap, shape);
                    assert!(
                        shape.0 * shape.1 == 0 || place.end(shape) <= inputs[slot].len(),
                        "fused program: input {slot} is read past its storage"
                    );
                    place
                }
                _ => Place::default(),
            })
            .collect();
        let alias = code
            .iter()
            .zip(&places)
            .map(|(instr, place)| match *instr {
                Instr::Load { input, .. } => {
                    let slot = usize::from(input);
                    slot < fresh_inputs
                        && place.in_order(shape.1)
                        && inputs[slot].typed::<T>().is_some()
                }
                _ => false,
            })
            .collect();

        let mut direct = vec![None; code.len()];
        let mut taken = vec![false; outputs.len()];
        for (store, instr) in code.iter().enumerate() {
            let Instr::Store { src, output } = *instr else {
                continue;
            };
            let output = usize::from(output);
            // The instruction whose value the store reads.
            let Some(source) = (0..store).rev().find(|&k| code[k].dst() == Some(src)) else {
                continue;
            };
            let computed = matches!(
                code[source],
                Instr::Binary { .. } | Instr::Unary { .. } | Instr::Cmp { .. }
            );
            let reloaded = output < program.updated
                && code[source..].iter().any(|instr| {
                    matches!(*instr, Instr::Load { input, .. }
                        if usize::from(input) == fresh_inputs + output)
                });
            if computed
                && !reloaded
                && !taken[output]
                && direct[source].is_none()
                && outputs[output].typed::<T>().is_some()
            {
                direct[source] = Some(output);
                taken[output] = true;
            }
        }
        Plan {
            places,
            alias,
            direct,
        }
    }
}

/// Retire an instruction's result: a constant (`Ok`), or a tensor (`Err`) —
/// written either into the spare tile, which becomes `dst`'s tile while `dst`'s
/// old one becomes the spare, or, when the plan sent it there, into an output,
/// where `dst` then reads it.
fn commit<T: Real>(
    dst: Reg,
    result: Result<T, Option<*const T>>,
    map: &mut [usize; REGISTERS],
    spare: &mut usize,
    constant: &mut [Option<T>; REGISTERS],
    outside: &mut [Option<*const T>; REGISTERS],
) {
    let dst = dst as usize;
    match result {
        Ok(value) => {
            constant[dst] = Some(value);
            outside[dst] = None;
        }
        Err(written) => {
            if written.is_none() {
                std::mem::swap(&mut map[dst], spare);
            }
            outside[dst] = written;
            constant[dst] = None;
        }
    }
}

/// The per-tile kernels: the SIMD tier where the build has one for the element
/// type, and otherwise the scalar definitions the SIMD kernels are tested
/// against.
mod kernel {
    use super::super::{Analytic, BinaryOp, Compare};
    use super::{Real, scalar_binary};
    #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
    use super::{same, same_mut};

    pub fn elementwise<T: Real>(a: &[T], b: &[T], op: BinaryOp, out: &mut [T]) {
        if crate::compact::elementwise(a, b, op, out) {
            return;
        }
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        if op != BinaryOp::Rem {
            if let (Some(a), Some(b), Some(out)) = (
                same::<T, f32>(a),
                same::<T, f32>(b),
                same_mut::<T, f32>(out),
            ) {
                return crate::simd::f32k::elementwise(a, b, op, out);
            }
            if let (Some(a), Some(b), Some(out)) = (
                same::<T, f64>(a),
                same::<T, f64>(b),
                same_mut::<T, f64>(out),
            ) {
                return crate::simd::f64k::elementwise(a, b, op, out);
            }
        }
        for ((out, &a), &b) in out.iter_mut().zip(a).zip(b) {
            *out = scalar_binary(op, a, b);
        }
    }

    pub fn broadcast<T: Real>(
        values: &[T],
        scalar: T,
        op: BinaryOp,
        scalar_left: bool,
        out: &mut [T],
    ) {
        if crate::compact::broadcast(values, scalar, op, scalar_left, out) {
            return;
        }
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        if op != BinaryOp::Rem {
            if let (Some(values), Some(scalar), Some(out)) = (
                same::<T, f32>(values),
                same::<T, f32>(std::slice::from_ref(&scalar)),
                same_mut::<T, f32>(out),
            ) {
                return crate::simd::f32k::broadcast(values, scalar[0], op, scalar_left, out);
            }
            if let (Some(values), Some(scalar), Some(out)) = (
                same::<T, f64>(values),
                same::<T, f64>(std::slice::from_ref(&scalar)),
                same_mut::<T, f64>(out),
            ) {
                return crate::simd::f64k::broadcast(values, scalar[0], op, scalar_left, out);
            }
        }
        for (out, &x) in out.iter_mut().zip(values) {
            *out = if scalar_left {
                scalar_binary(op, scalar, x)
            } else {
                scalar_binary(op, x, scalar)
            };
        }
    }

    pub fn compare<T: Real>(a: &[T], b: &[T], op: Compare, out: &mut [T]) {
        if crate::compact::compare(a, b, op, out) {
            return;
        }
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            if let (Some(a), Some(b), Some(out)) = (
                same::<T, f32>(a),
                same::<T, f32>(b),
                same_mut::<T, f32>(out),
            ) {
                return crate::simd::f32k::compare(a, b, op, out);
            }
            if let (Some(a), Some(b), Some(out)) = (
                same::<T, f64>(a),
                same::<T, f64>(b),
                same_mut::<T, f64>(out),
            ) {
                return crate::simd::f64k::compare(a, b, op, out);
            }
        }
        for ((out, &a), &b) in out.iter_mut().zip(a).zip(b) {
            *out = op.value(a, b);
        }
    }

    pub fn compare_scalar<T: Real>(
        values: &[T],
        scalar: T,
        op: Compare,
        scalar_left: bool,
        out: &mut [T],
    ) {
        if crate::compact::compare_scalar(values, scalar, op, scalar_left, out) {
            return;
        }
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            if let (Some(values), Some(scalar), Some(out)) = (
                same::<T, f32>(values),
                same::<T, f32>(std::slice::from_ref(&scalar)),
                same_mut::<T, f32>(out),
            ) {
                return crate::simd::f32k::compare_scalar(values, scalar[0], op, scalar_left, out);
            }
            if let (Some(values), Some(scalar), Some(out)) = (
                same::<T, f64>(values),
                same::<T, f64>(std::slice::from_ref(&scalar)),
                same_mut::<T, f64>(out),
            ) {
                return crate::simd::f64k::compare_scalar(values, scalar[0], op, scalar_left, out);
            }
        }
        for (out, &x) in out.iter_mut().zip(values) {
            *out = if scalar_left {
                op.value(scalar, x)
            } else {
                op.value(x, scalar)
            };
        }
    }

    /// `sqrt` is correctly rounded in hardware, so its vector form is exact; the
    /// other functions are the vectorized `vmath` ones, which the unfused
    /// kernels use too.
    pub fn unary<T: Real>(values: &[T], op: Analytic, out: &mut [T]) {
        if op == Analytic::Sqrt && crate::compact::sqrt(values, out) {
            return;
        }
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        if op == Analytic::Sqrt {
            if let (Some(values), Some(out)) = (same::<T, f32>(values), same_mut::<T, f32>(out)) {
                return crate::simd::f32k::sqrt(values, out);
            }
            if let (Some(values), Some(out)) = (same::<T, f64>(values), same_mut::<T, f64>(out)) {
                return crate::simd::f64k::sqrt(values, out);
            }
        }
        crate::vmath::unary_slice(op, values, out);
    }
}

/// `product` as input 0, then `inputs`.
fn with_product<'a, T: Element, B: Backend>(
    product: &'a Matrix<T, B>,
    inputs: &'a [Source<'_, B>],
) -> Vec<Source<'a, B>> {
    let mut sources = Vec::with_capacity(inputs.len() + 1);
    sources.push(product.source());
    sources.extend(inputs.iter().map(Source::reborrow));
    sources
}

/// The [`Host`] entry point for [`Kernels::matmul_epilogue`]: the product,
/// then the interpreter over it — the unfused computation, without counting
/// two kernels.
pub(crate) fn host_matmul<T: Element>(
    program: &Program<T>,
    a: &Matrix<T, Host>,
    b: &Matrix<T, Host>,
    inputs: &[Source<'_, Host>],
) -> Vec<Fresh<Host>> {
    let product = a.matmul(b);
    let sources = with_product(&product, inputs);
    interpret_on(program, product.shape(), &sources, &mut [])
}

/// The [`Host`] entry point for [`Kernels::fused`].
pub(crate) fn host<T: Real>(
    program: &Program<T>,
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
///
/// The shader is compiled once per [`MetalElement`](crate::metal::MetalElement)
/// and the program's own `T` picks the instance, so a `Program<f16>` runs in
/// `half` registers on the GPU just as it runs in `f16` tiles on the host.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) fn metal<T: crate::metal::MetalElement>(
    program: &Program<T>,
    shape: (usize, usize),
    inputs: &[Source<'_, super::Metal>],
    updated: &mut [Sink<'_, super::Metal>],
) -> Vec<Fresh<super::Metal>> {
    resident(program, shape, inputs, updated)
        .unwrap_or_else(|| interpret_on(program, shape, inputs, updated))
}

/// The [`Metal`](super::Metal) entry point for
/// [`Kernels::fused_with_statistics`]: one kernel over whole rows when every
/// operand is device-resident and the program has a kernel of its own, and
/// otherwise the statistics and then [`metal`].
#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) fn metal_with_statistics<T: crate::metal::MetalElement>(
    program: &Program<T>,
    shape: (usize, usize),
    inputs: &[Source<'_, super::Metal>],
    updated: &mut [Sink<'_, super::Metal>],
) -> Vec<Fresh<super::Metal>> {
    if let Some(fresh) = resident_rows(program, shape, inputs, updated) {
        return fresh;
    }
    let statistics = row_statistics(program, shape, inputs);
    let mut all: Vec<Source<'_, super::Metal>> = inputs.iter().map(Source::reborrow).collect();
    all.extend(statistics.iter().map(source_of));
    metal(program, shape, &all, updated)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn resident_rows<T: crate::metal::MetalElement>(
    program: &Program<T>,
    shape: (usize, usize),
    inputs: &[Source<'_, super::Metal>],
    updated: &[Sink<'_, super::Metal>],
) -> Option<Vec<Fresh<super::Metal>>> {
    if !program.runs_on_metal() || program.derived.len() > 16 {
        return None;
    }
    let mut read = device::sources(inputs)?;
    // The statistics' slots, which the kernel computes rather than reads.
    let filler = *read.first()?;
    read.extend(program.derived.iter().map(|_| filler));
    let mut written = Vec::with_capacity(program.outputs.len());
    for target in updated {
        let buffer = device::sink(&target.data)?;
        read.push(buffer);
        written.push(buffer);
    }
    let fresh = device::allocate(program, shape.0 * shape.1)?;
    written.extend(fresh.iter().map(device::Allocation::raw));

    let mut statistics = [(0u8, 0u8, false); 16];
    for (entry, &(of, statistic)) in statistics.iter_mut().zip(&program.derived) {
        *entry = (
            of,
            program.inputs[usize::from(of)] as u8,
            statistic == RowStatistic::Deviations,
        );
    }
    let statistics = crate::metal::RowStatistics {
        first: program.given_inputs() as u8,
        statistics,
        count: program.derived.len(),
    };
    let places = shader_places(program, shape, inputs, 0)?;
    crate::metal::fused_rows::<T>(
        &program.encode(),
        shape,
        &read,
        &written,
        &places,
        statistics,
    )?;
    Some(
        fresh
            .into_iter()
            .map(device::Allocation::into_fresh)
            .collect(),
    )
}

/// [`row_moments`](Kernels::row_moments) on Metal: the moments kernel on the
/// input where it lies, when it is resident and of type `T`.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) fn metal_row_moments<T: crate::metal::MetalElement>(
    source: &Source<'_, super::Metal>,
    (rows, cols): (usize, usize),
) -> (Vector<T, super::Metal>, Vector<T, super::Metal>) {
    let resident = (rows != 0 && cols != 0 && source.view.is_none())
        .then(|| source.typed::<T>()?.axis_moments(rows, cols, Axis::Rows))
        .flatten();
    match resident {
        Some((means, deviations)) => (
            Vector::from_storage(rows, means),
            Vector::from_storage(rows, deviations),
        ),
        None => super::Metal::matrix_axis_moments(&input_matrix(source, (rows, cols)), Axis::Rows),
    }
}

/// The [`Metal`](super::Metal) entry point for [`Kernels::fused_sum`]: one
/// kernel when the operands are device-resident and the program has a kernel
/// of its own, and otherwise the program and then the sums.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) fn metal_sum<T: crate::metal::MetalElement + Element>(
    program: &Program<T>,
    shape: (usize, usize),
    inputs: &[Source<'_, super::Metal>],
    axis: Axis,
) -> Vector<T, super::Metal> {
    resident_sum(program, shape, inputs, axis).unwrap_or_else(|| {
        let data = metal(program, shape, inputs, &mut []).remove(0);
        axis_sum(Output { shape, data }.into_matrix::<T>(), axis)
    })
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn resident_sum<T: crate::metal::MetalElement + Element>(
    program: &Program<T>,
    shape: (usize, usize),
    inputs: &[Source<'_, super::Metal>],
    axis: Axis,
) -> Option<Vector<T, super::Metal>> {
    use crate::metal::MetalBuffer;
    use crate::tensors::MetalStorage;

    if !program.runs_on_metal() {
        return None;
    }
    let read = device::sources(inputs)?;
    let len = match axis {
        Axis::Rows => shape.0,
        Axis::Columns => shape.1,
    };
    let places = shader_places(program, shape, inputs, 0)?;
    let sums = MetalBuffer::<T>::allocate(len)?;
    crate::metal::fused_sum::<T>(
        &program.encode(),
        shape,
        &read,
        &places,
        axis == Axis::Rows,
        sums.raw(),
    )?;
    Some(Vector::from_storage(len, MetalStorage::from_device(sums)))
}

/// The [`Metal`](super::Metal) entry point for [`Kernels::matmul_epilogue`]:
/// one dispatch when the operands are device-resident and the program runs on
/// the shader, and otherwise the product followed by [`metal`].
#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) fn metal_matmul<T: crate::metal::MetalElement + Element>(
    program: &Program<T>,
    a: &Matrix<T, super::Metal>,
    b: &Matrix<T, super::Metal>,
    inputs: &[Source<'_, super::Metal>],
) -> Vec<Fresh<super::Metal>> {
    resident_matmul(program, a, b, inputs).unwrap_or_else(|| {
        let product = a.matmul(b);
        let sources = with_product(&product, inputs);
        metal(program, product.shape(), &sources, &mut [])
    })
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod device {
    use half::{bf16, f16};
    use objc2::runtime::ProtocolObject;
    use objc2_metal::MTLBuffer;

    use super::{DType, Fresh, Program, SinkData, Source, SourceData};
    use crate::metal::MetalBuffer;
    use crate::tensors::{Metal, MetalStorage};

    pub(super) type Raw = ProtocolObject<dyn MTLBuffer>;

    pub(super) fn raw<T: Copy + 'static>(storage: &MetalStorage<T>) -> Option<&Raw> {
        Some(storage.device()?.raw())
    }

    /// The device buffer behind each source, or `None` if any is not resident
    /// or is `f64`, which the shaders do not have.
    pub(super) fn sources<'a>(inputs: &'a [Source<'_, Metal>]) -> Option<Vec<&'a Raw>> {
        inputs
            .iter()
            .map(|input| match input.data {
                SourceData::F32(storage) => raw(storage),
                SourceData::F16(storage) => raw(storage),
                SourceData::Bf16(storage) => raw(storage),
                SourceData::F64(_) => None,
            })
            .collect()
    }

    /// The elements in the device buffer behind a source: its storage, whole,
    /// whatever part of it a view reads.
    pub(super) fn stored_len(input: &Source<'_, Metal>) -> Option<usize> {
        Some(match input.data {
            SourceData::F32(storage) => storage.device()?.len(),
            SourceData::F16(storage) => storage.device()?.len(),
            SourceData::Bf16(storage) => storage.device()?.len(),
            SourceData::F64(_) => return None,
        })
    }

    /// The device buffer behind an in-place operand.
    pub(super) fn sink<'a>(data: &'a SinkData<'_, Metal>) -> Option<&'a Raw> {
        match data {
            SinkData::F32(storage) => raw(storage),
            SinkData::F16(storage) => raw(storage),
            SinkData::Bf16(storage) => raw(storage),
            SinkData::F64(_) => None,
        }
    }

    /// A fresh output buffer of one of the shader's storage types.
    pub(super) enum Allocation {
        F32(MetalBuffer<f32>),
        F16(MetalBuffer<f16>),
        Bf16(MetalBuffer<bf16>),
    }

    impl Allocation {
        pub(super) fn raw(&self) -> &Raw {
            match self {
                Allocation::F32(buffer) => buffer.raw(),
                Allocation::F16(buffer) => buffer.raw(),
                Allocation::Bf16(buffer) => buffer.raw(),
            }
        }

        pub(super) fn into_fresh(self) -> Fresh<Metal> {
            match self {
                Allocation::F32(buffer) => Fresh::F32(MetalStorage::from_device(buffer)),
                Allocation::F16(buffer) => Fresh::F16(MetalStorage::from_device(buffer)),
                Allocation::Bf16(buffer) => Fresh::Bf16(MetalStorage::from_device(buffer)),
            }
        }
    }

    /// Buffers for the program's fresh outputs, `len` elements each.
    pub(super) fn allocate<T>(program: &Program<T>, len: usize) -> Option<Vec<Allocation>> {
        program.outputs[program.updated..]
            .iter()
            .map(|dtype| {
                Some(match dtype {
                    DType::F32 => Allocation::F32(MetalBuffer::allocate(len)?),
                    DType::F16 => Allocation::F16(MetalBuffer::allocate(len)?),
                    DType::Bf16 => Allocation::Bf16(MetalBuffer::allocate(len)?),
                    DType::F64 => return None,
                })
            })
            .collect()
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn resident<T: crate::metal::MetalElement>(
    program: &Program<T>,
    shape: (usize, usize),
    inputs: &[Source<'_, super::Metal>],
    updated: &[Sink<'_, super::Metal>],
) -> Option<Vec<Fresh<super::Metal>>> {
    if !program.runs_on_metal() {
        return None;
    }
    let len = shape.0 * shape.1;
    let mut read = device::sources(inputs)?;
    let mut written = Vec::with_capacity(program.outputs.len());
    for target in updated {
        let buffer = device::sink(&target.data)?;
        read.push(buffer);
        written.push(buffer);
    }
    let fresh = device::allocate(program, len)?;
    written.extend(fresh.iter().map(device::Allocation::raw));

    let places = shader_places(program, shape, inputs, 0)?;
    crate::metal::fused_elementwise::<T>(&program.encode(), shape, &read, &written, &places)?;
    Some(
        fresh
            .into_iter()
            .map(device::Allocation::into_fresh)
            .collect(),
    )
}

/// Where each input slot of `program` is read through each remap, for the
/// shaders: entry `slot·4 + remap` is `[offset, row step, column step]` (see
/// [`Place`]). `inputs` are the given inputs from slot `first` on; every other
/// slot — a product, a statistic, a tensor updated in place — is read in
/// order. `None` if a read would reach past its buffer, or an index does not
/// fit the shaders' 32 bits.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn shader_places<T>(
    program: &Program<T>,
    shape: (usize, usize),
    inputs: &[Source<'_, super::Metal>],
    first: usize,
) -> Option<Vec<[u32; 3]>> {
    let mut table = vec![[0u32; 3]; MAX_INPUTS * 4];
    // Every load, and every input a row statistic is of — which the kernel
    // reads in order, whether or not the program loads it otherwise.
    let loads = program.code.iter().filter_map(|instr| match *instr {
        Instr::Load { input, remap, .. } => Some((input, remap)),
        _ => None,
    });
    let statistics = program.derived.iter().map(|&(of, _)| (of, Remap::Identity));
    for (input, remap) in loads.chain(statistics) {
        let slot = usize::from(input);
        let source = slot.checked_sub(first).and_then(|index| inputs.get(index));
        let place = Place::of(source.and_then(|source| source.view), remap, shape);
        if let Some(source) = source
            && shape.0 * shape.1 != 0
            && place.end(shape) > device::stored_len(source)?
        {
            return None;
        }
        table[slot * 4 + remap as usize] = [
            u32::try_from(place.offset).ok()?,
            u32::try_from(place.row).ok()?,
            u32::try_from(place.col).ok()?,
        ];
    }
    Some(table)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn resident_matmul<T: crate::metal::MetalElement>(
    program: &Program<T>,
    a: &Matrix<T, super::Metal>,
    b: &Matrix<T, super::Metal>,
    inputs: &[Source<'_, super::Metal>],
) -> Option<Vec<Fresh<super::Metal>>> {
    use super::Backend;

    if !program.runs_on_metal() {
        return None;
    }
    let (m, k, n) = (a.rows(), a.cols(), b.cols());
    let left = device::raw(super::Metal::matrix_as_vector(a.storage()))?;
    let right = device::raw(super::Metal::matrix_as_vector(b.storage()))?;
    let read = device::sources(inputs)?;
    let fresh = device::allocate(program, m * n)?;
    let written: Vec<&device::Raw> = fresh.iter().map(device::Allocation::raw).collect();

    // The product is input 0, and the given inputs follow it.
    let places = shader_places(program, (m, n), inputs, 1)?;
    crate::metal::matmul_epilogue::<T>(
        &program.encode(),
        (m, k, n),
        left,
        right,
        &read,
        &written,
        &places,
    )?;
    Some(
        fresh
            .into_iter()
            .map(device::Allocation::into_fresh)
            .collect(),
    )
}
