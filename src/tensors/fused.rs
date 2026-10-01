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

use half::{bf16, f16};

use super::{Analytic, Backend, BinaryOp, Compare, Host, Kernels, Matrix, Vector};
use crate::counters;
use crate::numbers::Real;

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
    pub fn run<B: Kernels<T>>(
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
        assert_eq!(self.updated, 0, "fused matmul: the epilogue cannot update tensors in place");
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
            if let Instr::Load { input: 0, remap, .. } = *instr {
                assert_eq!(remap, Remap::Identity, "fused matmul: the product is read through {remap:?}");
            }
        }
        let shape = (a.rows(), b.cols());
        let len = shape.0 * shape.1;
        self.check_input(0, T::DTYPE, len, shape);
        let sources: Vec<Source<'_, B>> = inputs.iter().map(|input| input.source()).collect();
        for (slot, source) in sources.iter().enumerate() {
            self.check_input(slot + 1, source.dtype(), source.len, shape);
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
    pub fn build_with(self, model: &CostModel, algebra: Algebra) -> Result<Program<T>, ProgramError> {
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
    let plan = optimizer::optimize(&key.graph, &options).map_err(|_| ProgramError::TooManyRegisters)?;
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
                Instr::Binary { op, a, b, .. } => optimizer::Node::Binary(bin(op), read(a), read(b)),
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
                output_bytes: self.outputs.iter().map(|dtype| dtype.size() as u8).collect(),
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
    pub len: usize,
}

/// An in-place program operand: storage and its length.
#[doc(hidden)]
pub struct Sink<'a, B: Backend> {
    pub data: SinkData<'a, B>,
    pub len: usize,
}

impl<B: Backend> Source<'_, B> {
    /// Another handle on the same storage.
    fn reborrow(&self) -> Source<'_, B> {
        Source {
            data: self.data,
            len: self.len,
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
    (rows, cols): (usize, usize),
    remap: Remap,
) -> Matrix<T, B> {
    let (stored_rows, stored_cols) = match remap {
        Remap::Identity => (rows, cols),
        Remap::Transpose => (cols, rows),
        Remap::Row => (1, cols),
        Remap::Column => (rows, 1),
    };
    // An input already of the program's type is copied on its own backend —
    // on Metal, on the GPU — rather than read back through the CPU, which on a
    // device would wait for every queued kernel.
    let stored = match source.typed::<T>() {
        Some(storage) => B::duplicate(storage),
        None => B::store_vector(&source.slice().widen::<T>()),
    };
    let tensor = Matrix::from_storage(stored_rows, stored_cols, B::vector_into_matrix(stored));
    match remap {
        Remap::Identity => tensor,
        Remap::Transpose => B::transpose(&tensor),
        Remap::Row | Remap::Column => {
            let vector = B::matrix_as_vector(tensor.storage());
            let count = if remap == Remap::Row { rows } else { cols };
            let copies: Vec<B::Vector<T>> = (0..count).map(|_| B::duplicate(vector)).collect();
            let storage = if remap == Remap::Row {
                B::vstack(&copies, cols)
            } else {
                B::hstack(&copies, rows)
            };
            Matrix::from_storage(rows, cols, storage)
        }
    }
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

    /// Convert `out.len()` elements from `start`, read through `remap`, into `out`.
    ///
    /// The broadcasts walk the space a row segment at a time: a [`Row`]
    /// segment is a contiguous run of the vector, a [`Column`] segment one
    /// value repeated, so neither pays an index computation per element.
    ///
    /// [`Row`]: Remap::Row
    /// [`Column`]: Remap::Column
    fn gather<T: Real>(&self, start: usize, shape: (usize, usize), remap: Remap, out: &mut [T]) {
        let (rows, cols) = shape;
        match remap {
            Remap::Identity => self.copy_into(start, out),
            Remap::Row | Remap::Column => {
                let mut offset = 0;
                while offset < out.len() {
                    let (row, col) = ((start + offset) / cols, (start + offset) % cols);
                    let run = (cols - col).min(out.len() - offset);
                    let segment = &mut out[offset..offset + run];
                    if remap == Remap::Row {
                        self.copy_into(col, segment);
                    } else {
                        segment.fill(self.get(row));
                    }
                    offset += run;
                }
            }
            Remap::Transpose => {
                let (mut row, mut col) = (start / cols, start % cols);
                for out in out.iter_mut() {
                    *out = self.get(col * rows + row);
                    col += 1;
                    if col == cols {
                        col = 0;
                        row += 1;
                    }
                }
            }
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
        let inputs: Vec<Slice<'_>> = inputs.iter().map(Source::slice).collect();
        let mut outputs: Vec<SliceMut<'_>> = updated.iter_mut().map(Sink::slice).collect();
        outputs.extend(fresh.iter_mut().map(Owned::slice));
        interpret(program, shape, &inputs, &mut outputs);
    }
    fresh.into_iter().map(Owned::store).collect()
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
    outputs: &mut [SliceMut<'_>],
) {
    let len = shape.0 * shape.1;
    let plan = Plan::new(program, inputs, outputs);
    // One physical tile per register, plus a spare that receives each result
    // so an instruction may overwrite one of its own operands.
    let physical = program.registers + 1;
    let tile_len = tile_len::<T>(physical, len);
    with_scratch::<T, _>(physical * tile_len, |scratch| {
        run_tiles(program, shape, inputs, outputs, &plan, scratch, tile_len)
    });
}

/// The tile loop of [`interpret`], over `scratch` holding one `tile_len` tile
/// per physical register.
fn run_tiles<T: Real>(
    program: &Program<T>,
    shape: (usize, usize),
    inputs: &[Slice<'_>],
    outputs: &mut [SliceMut<'_>],
    plan: &Plan,
    scratch: &mut [T],
    tile_len: usize,
) {
    let len = shape.0 * shape.1;
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

    let mut start = 0;
    while start < len {
        let n = tile_len.min(len - start);
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
                        (unsafe { values.as_mut_ptr().add(start) }, true)
                    }
                    None => (tile(spare), false),
                }
            };
            match *instr {
                Instr::Load { dst, input, remap } => {
                    let slot = usize::from(input);
                    constant[dst as usize] = None;
                    if plan.alias[index] {
                        let values = inputs[slot].typed::<T>().expect("planned");
                        outside[dst as usize] = Some(unsafe { values.as_ptr().add(start) });
                        continue;
                    }
                    outside[dst as usize] = None;
                    // SAFETY: a register's tile is a distinct window of scratch.
                    let out = unsafe { std::slice::from_raw_parts_mut(tile(map[dst as usize]), n) };
                    if slot < fresh_inputs {
                        inputs[slot].gather(start, shape, remap, out);
                    } else {
                        outputs[slot - fresh_inputs]
                            .view()
                            .gather(start, shape, remap, out);
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
                        Some(value) => output.fill(start, n, value),
                        None => {
                            let from = place(src, &outside, &map);
                            // Already there if the plan wrote it in place.
                            let there = output.typed::<T>().is_some_and(|values| {
                                std::ptr::eq(from, unsafe { values.as_ptr().add(start) })
                            });
                            if !there {
                                let values = unsafe { std::slice::from_raw_parts(from, n) };
                                output.scatter(start, values);
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
    /// A load reads its input in place: the input is already of the program's
    /// type, is read without a remap, and is not one updated in place (whose
    /// memory an earlier tile's stores may have been written to).
    alias: Vec<bool>,
    /// A computed result is written straight into this output: the output is
    /// of the program's type, the result is what a later store sends there, and
    /// — for a tensor updated in place — nothing loads that tensor afterwards.
    direct: Vec<Option<usize>>,
}

impl Plan {
    fn new<T: Real>(
        program: &Program<T>,
        inputs: &[Slice<'_>],
        outputs: &mut [SliceMut<'_>],
    ) -> Self {
        let code = &program.code;
        let fresh_inputs = program.fresh_inputs();
        let alias = code
            .iter()
            .map(|instr| match *instr {
                Instr::Load { input, remap, .. } => {
                    let slot = usize::from(input);
                    remap == Remap::Identity
                        && slot < fresh_inputs
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
        Plan { alias, direct }
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

    crate::metal::fused_elementwise::<T>(&program.encode(), shape, &read, &written)?;
    Some(fresh.into_iter().map(device::Allocation::into_fresh).collect())
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

    crate::metal::matmul_epilogue::<T>(&program.encode(), (m, k, n), left, right, &read, &written)?;
    Some(fresh.into_iter().map(device::Allocation::into_fresh).collect())
}
