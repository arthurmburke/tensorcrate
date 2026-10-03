//! Gradient-descent optimizers.
//!
//! Two layers, so the objective, the update rule and the batching strategy stay
//! independent of one another:
//!
//! - a [`Rule`] turns a gradient into a parameter update and owns whatever state
//!   that takes — [`Sgd`] has none, [`Momentum`] keeps a velocity, [`Adam`] keeps
//!   two moments and a step count;
//! - [`minimize`] drives the loop: record the parameters on a fresh tape, build
//!   the loss with *your* objective, propagate, apply the rule.
//!
//! Nothing here knows what the loss means. The objective is a closure, so
//! squared error, log-cosh, absolute error, cross-entropy and a regularized
//! variant of any of them are the same call with a different body.
//!
//! ```
//! use tensorcrate::optim::{Adam, minimize};
//! use tensorcrate::tensors::{Matrix, Vector};
//!
//! // Fit x to A·x = y by minimizing the residual, whatever the rule.
//! let a = Matrix::<f32>::from_rows([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]]);
//! let targets = a.matvec(&Vector::new([0.5f32, -0.25]));
//!
//! let mut parameters = Vector::<f32>::zeros(2);
//! let mut rule = Adam::new(0.1);
//! minimize(&mut parameters, &mut rule, 400, |x, _step| {
//!     let tape = x.tape();
//!     let residual = &tape.matrix(a.clone()).matvec(x) - &tape.vector(targets.clone());
//!     residual.dot(&residual)
//! });
//!
//! assert!((parameters[0] - 0.5).abs() < 1e-3);
//! assert!((parameters[1] + 0.25).abs() < 1e-3);
//! ```
//!
//! # Stochastic descent
//!
//! Which *samples* a step sees is a property of the objective, not of the rule,
//! so [`minimize`] passes the step number to the closure and lets it choose. A
//! mini-batch objective indexes its data by `step % batches`; a full-batch one
//! ignores the argument. Both use the same rules.
//!
//! # Several parameters
//!
//! [`minimize`] drives one parameter tensor. A model with a weight matrix *and* a
//! bias wants one rule per tensor and a hand-written loop — see
//! `examples/optimizers.rs`, which does exactly that. The rules are the reusable
//! part; the loop is four lines.

use std::sync::Arc;

use num_traits::Float;

use crate::numbers::Real;
use crate::tensors::fused::{
    Builder, Element, Fusable, FusableMut, FusableOf, Instr, Output, Program, Remap,
};
use crate::tensors::tape::Adjoint;
use crate::tensors::{
    Analytic, BinaryOp, Host, Kernels, Matrix, ScalarVar, Tape, Tensor, Var, Vector,
};

/// A tensor an optimizer can carry: the elementwise algebra the update rules
/// need, plus the ability to put itself on a tape.
///
/// Implemented for scalars, [`Vector`], [`Matrix`] and N-dimensional
/// [`Tensor`] of any element type —
/// `f32`, `f64`, [`f16`](crate::numbers::f16) or [`bf16`](crate::numbers::bf16) —
/// so a rule is written once and applies to any of them, on either backend,
/// without leaving it. (Metal computes in `f32`, `f16` and `bf16`, so those are
/// the element types of a resident parameter.)
///
/// A rule's step is a fused [`Program`], run the way
/// [`Program::run`](crate::tensors::fused::Program::run) runs one: the gradient
/// is an input, only read, and the parameters and the rule's state are updated
/// in place. So a tensor's gradient is anything a program reads — see
/// [`Gradient`](Parameter::Gradient).
pub trait Parameter: Sized + 'static {
    /// The element type: what the hyperparameters, the loss and the arithmetic
    /// of a step are made of.
    type Elem: Element;

    /// Where this parameter's elements live.
    type Backend: Kernels<Self::Elem>;

    /// What a rule reads this parameter's gradient from. For a [`Vector`], a
    /// [`Matrix`] or a [`Tensor`], any tensor of
    /// [`Self::Elem`](Parameter::Elem) a fused program reads in the
    /// parameter's shape ([`FusableOf`]): the same type, or a view of a
    /// larger one — a [`MatrixView`](crate::tensors::fused::MatrixView) or a
    /// [`TensorView`](crate::tensors::TensorView). For a scalar, the scalar.
    /// For a parameter type of your own, usually the type itself.
    type Gradient<'a>: ?Sized + 'a;

    /// This parameter as a gradient for one of its own type — what generic
    /// code passes to [`Rule::update`].
    fn as_gradient(&self) -> &Self::Gradient<'_>;

    /// A tensor of *this* parameter's shape, all zeros — the starting point for
    /// the moment buffers the adaptive rules keep. With runtime dimensions the
    /// shape comes from an existing tensor rather than from the type.
    fn zeros_like(&self) -> Self;

    /// A second copy on the same backend.
    fn duplicate(&self) -> Self;

    fn add(&self, other: &Self) -> Self;
    fn subtract(&self, other: &Self) -> Self;
    fn multiply(&self, other: &Self) -> Self;
    fn divide(&self, other: &Self) -> Self;

    /// Multiply every element by a constant.
    fn scale(&self, factor: Self::Elem) -> Self;

    /// Add a constant to every element.
    fn shift(&self, offset: Self::Elem) -> Self;

    /// Elementwise square root, for the adaptive rules' denominators.
    fn sqrt(&self) -> Self;

    /// Record this parameter as a tape leaf, so a backward pass can produce its
    /// gradient.
    fn record<'t>(&self, tape: &'t Tape<Self::Backend>) -> Var<'t, Self, Self::Backend>
    where
        Self: Adjoint<Self::Backend>;

    /// Run a fused elementwise [`Program`] over this parameter's shape, as
    /// [`Program::run`](crate::tensors::fused::Program::run) would: `inputs`
    /// fill the program's given inputs and are only read, `updated` are its
    /// tensors updated in place, overwritten with their new values, and the
    /// program's fresh outputs are returned. The parameters come first among
    /// `updated`, and fix the shape.
    ///
    /// This is how the rules take a whole update in one kernel. A parameter
    /// type of your own can implement it with [`run_by_parts`], which runs the
    /// program one operation at a time through the methods above.
    fn fused(
        program: &Program<Self::Elem>,
        inputs: &[&Self::Gradient<'_>],
        updated: &mut [&mut Self],
    ) -> Vec<Self>;
}

/// [`Parameter::fused`] through a parameter type's own elementwise methods,
/// for a type with no program runner of its own: it supports exactly the
/// operations they spell — `+ − × ÷`, `sqrt`, and constants — and panics on
/// anything else.
///
/// It reads its operands as [`Program::run`](crate::tensors::fused::Program::run)
/// does: as many inputs as the program has given inputs and as many in-place
/// tensors as it updates, every tensor updated in place read before any is
/// overwritten. A program that remaps a load or computes row statistics needs
/// a shape, which a parameter type of your own does not have, so it panics.
///
/// ```
/// use tensorcrate::optim::{Parameter, run_by_parts};
/// # use tensorcrate::tensors::fused::Program;
/// # use tensorcrate::tensors::{Host, Tape, Var};
/// # #[derive(Clone, Copy, Debug, PartialEq)]
/// # struct Pair(f32, f32);
/// impl Parameter for Pair {
///     type Elem = f32;
///     type Backend = Host;
///     type Gradient<'a> = Pair;
///
///     fn as_gradient(&self) -> &Pair {
///         self
///     }
///
///     fn fused(program: &Program, inputs: &[&Pair], updated: &mut [&mut Pair]) -> Vec<Pair> {
///         run_by_parts(program, inputs, updated)
///     }
///     // ...
/// #   fn zeros_like(&self) -> Self { Pair(0.0, 0.0) }
/// #   fn duplicate(&self) -> Self { *self }
/// #   fn add(&self, o: &Self) -> Self { Pair(self.0 + o.0, self.1 + o.1) }
/// #   fn subtract(&self, o: &Self) -> Self { Pair(self.0 - o.0, self.1 - o.1) }
/// #   fn multiply(&self, o: &Self) -> Self { Pair(self.0 * o.0, self.1 * o.1) }
/// #   fn divide(&self, o: &Self) -> Self { Pair(self.0 / o.0, self.1 / o.1) }
/// #   fn scale(&self, f: f32) -> Self { Pair(self.0 * f, self.1 * f) }
/// #   fn shift(&self, f: f32) -> Self { Pair(self.0 + f, self.1 + f) }
/// #   fn sqrt(&self) -> Self { Pair(self.0.sqrt(), self.1.sqrt()) }
/// #   fn record<'t>(&self, _: &'t Tape<Host>) -> Var<'t, Self, Host> { unreachable!() }
/// }
///
/// use tensorcrate::optim::{Rule, Sgd};
/// let mut p = Pair(1.0, 2.0);
/// Sgd::new(0.5).update(&mut p, &Pair(2.0, -2.0));
/// assert_eq!(p, Pair(0.0, 3.0));
/// ```
#[track_caller]
pub fn run_by_parts<P: Parameter>(
    program: &Program<P::Elem>,
    inputs: &[&P],
    updated: &mut [&mut P],
) -> Vec<P> {
    assert_eq!(
        inputs.len(),
        program.given_inputs(),
        "fused program: expected {} inputs, got {}",
        program.given_inputs(),
        inputs.len()
    );
    assert_eq!(
        updated.len(),
        program.updated(),
        "fused program: expected {} in-place tensors, got {}",
        program.updated(),
        updated.len()
    );
    assert!(
        program.row_statistics().is_empty(),
        "a parameter program computes no row statistics: a parameter type has no rows"
    );
    enum Value<P, T> {
        Scalar(T),
        Tensor(P),
    }
    // Any operand fixes the shape a constant has to be filled out to.
    let like = inputs
        .first()
        .map(|p| p.zeros_like())
        .or_else(|| updated.first().map(|p| p.zeros_like()))
        .expect("a parameter program has at least one tensor operand");
    let fill = |value: P::Elem| like.shift(value);
    let tensor = |value: &Value<P, P::Elem>| match value {
        Value::Scalar(value) => fill(*value),
        Value::Tensor(tensor) => tensor.duplicate(),
    };

    let given = program.given_inputs();
    let mut registers: Vec<Option<Value<P, P::Elem>>> = (0..16).map(|_| None).collect();
    let mut outputs: Vec<Option<P>> = (0..program.outputs().len()).map(|_| None).collect();
    for instr in program.code() {
        let get = |reg: u8| registers[usize::from(reg)].as_ref().expect("validated");
        let (dst, value) = match *instr {
            Instr::Load { dst, input, remap } => {
                assert_eq!(remap, Remap::Identity, "parameter programs read unremapped");
                let slot = usize::from(input);
                let source: &P = if slot < given {
                    inputs[slot]
                } else {
                    &*updated[slot - given]
                };
                (dst, Value::Tensor(source.duplicate()))
            }
            Instr::Const { dst, value } => (dst, Value::Scalar(value)),
            Instr::Binary { dst, op, a, b } => {
                let value = match (get(a), get(b), op) {
                    (Value::Scalar(a), Value::Scalar(b), _) => Value::Scalar(match op {
                        BinaryOp::Add => *a + *b,
                        BinaryOp::Sub => *a - *b,
                        BinaryOp::Mul => *a * *b,
                        BinaryOp::Div => *a / *b,
                        BinaryOp::Rem => *a % *b,
                    }),
                    // The two forms `scale` and `shift` are exactly.
                    (Value::Tensor(a), Value::Scalar(b), BinaryOp::Mul) => {
                        Value::Tensor(a.scale(*b))
                    }
                    (Value::Scalar(a), Value::Tensor(b), BinaryOp::Mul) => {
                        Value::Tensor(b.scale(*a))
                    }
                    (Value::Tensor(a), Value::Scalar(b), BinaryOp::Add) => {
                        Value::Tensor(a.shift(*b))
                    }
                    (Value::Scalar(a), Value::Tensor(b), BinaryOp::Add) => {
                        Value::Tensor(b.shift(*a))
                    }
                    (a, b, op) => {
                        let (a, b) = (tensor(a), tensor(b));
                        Value::Tensor(match op {
                            BinaryOp::Add => a.add(&b),
                            BinaryOp::Sub => a.subtract(&b),
                            BinaryOp::Mul => a.multiply(&b),
                            BinaryOp::Div => a.divide(&b),
                            BinaryOp::Rem => panic!("Parameter has no remainder"),
                        })
                    }
                };
                (dst, value)
            }
            Instr::Unary {
                dst,
                op: Analytic::Sqrt,
                a,
            } => (
                dst,
                match get(a) {
                    Value::Scalar(a) => Value::Scalar(a.sqrt()),
                    Value::Tensor(a) => Value::Tensor(a.sqrt()),
                },
            ),
            Instr::Unary { op, .. } => panic!("Parameter has no {op:?}"),
            Instr::Cmp { op, .. } => panic!("Parameter has no {op:?} comparison"),
            Instr::Store { src, output } => {
                outputs[usize::from(output)] = Some(tensor(get(src)));
                continue;
            }
        };
        registers[usize::from(dst)] = Some(value);
    }

    let mut outputs = outputs.into_iter().map(|output| output.expect("validated"));
    for target in updated.iter_mut() {
        **target = outputs.next().expect("validated");
    }
    outputs.collect()
}

/// Run `program` over a `shape` space, which is every parameter program's.
fn run_fused<E: Element, B: Kernels<E>, T: FusableMut<B>>(
    program: &Program<E>,
    shape: (usize, usize),
    inputs: &[&(dyn FusableOf<E, B> + '_)],
    updated: &mut [&mut T],
) -> Vec<Output<B>> {
    let inputs: Vec<&dyn Fusable<B>> = inputs
        .iter()
        .map(|&input| input as &dyn Fusable<B>)
        .collect();
    let mut updated: Vec<&mut dyn FusableMut<B>> = updated
        .iter_mut()
        .map(|target| &mut **target as &mut dyn FusableMut<B>)
        .collect();
    program.run(shape, &inputs, &mut updated)
}

/// A scalar parameter — a learned temperature, or a log-variance. It carries no
/// storage of its own, so its tape is the host one; a scalar living inside a
/// resident graph is a job for [`Tape`] directly rather than for [`minimize`].
impl<T: Element> Parameter for T {
    type Elem = T;
    type Backend = Host;
    type Gradient<'a> = T;

    fn as_gradient(&self) -> &T {
        self
    }

    fn zeros_like(&self) -> Self {
        T::zero()
    }

    fn duplicate(&self) -> Self {
        *self
    }

    fn add(&self, other: &Self) -> Self {
        *self + *other
    }

    fn subtract(&self, other: &Self) -> Self {
        *self - *other
    }

    fn multiply(&self, other: &Self) -> Self {
        *self * *other
    }

    fn divide(&self, other: &Self) -> Self {
        *self / *other
    }

    fn scale(&self, factor: T) -> Self {
        *self * factor
    }

    fn shift(&self, offset: T) -> Self {
        *self + offset
    }

    fn sqrt(&self) -> Self {
        Float::sqrt(*self)
    }

    fn record<'t>(&self, tape: &'t Tape<Host>) -> Var<'t, Self, Host> {
        tape.scalar(*self)
    }

    fn fused(program: &Program<T>, inputs: &[&Self], updated: &mut [&mut Self]) -> Vec<Self> {
        let inputs: Vec<Vector<T>> = inputs.iter().map(|&&x| Vector::new([x])).collect();
        let mut targets: Vec<Vector<T>> = updated.iter().map(|x| Vector::new([**x])).collect();
        let outputs = {
            let inputs: Vec<&dyn FusableOf<T, Host>> = inputs
                .iter()
                .map(|x| x as &dyn FusableOf<T, Host>)
                .collect();
            let mut targets: Vec<&mut Vector<T>> = targets.iter_mut().collect();
            run_fused(program, (1, 1), &inputs, &mut targets)
        };
        for (target, value) in updated.iter_mut().zip(&targets) {
            **target = value[0];
        }
        outputs
            .into_iter()
            .map(|output| output.into_vector::<T>().as_slice()[0])
            .collect()
    }
}

impl<T: Element, B: Kernels<T>> Parameter for Vector<T, B> {
    type Elem = T;
    type Backend = B;
    type Gradient<'a> = dyn FusableOf<T, B> + 'a;

    fn as_gradient(&self) -> &(dyn FusableOf<T, B> + '_) {
        self
    }

    fn zeros_like(&self) -> Self {
        Vector::filled(self.len(), T::zero())
    }

    fn duplicate(&self) -> Self {
        self.to_backend::<B>()
    }

    fn add(&self, other: &Self) -> Self {
        B::vector_elementwise(self, other, BinaryOp::Add)
    }

    fn subtract(&self, other: &Self) -> Self {
        B::vector_elementwise(self, other, BinaryOp::Sub)
    }

    fn multiply(&self, other: &Self) -> Self {
        B::vector_elementwise(self, other, BinaryOp::Mul)
    }

    fn divide(&self, other: &Self) -> Self {
        B::vector_elementwise(self, other, BinaryOp::Div)
    }

    fn scale(&self, factor: T) -> Self {
        B::vector_broadcast(self, factor, BinaryOp::Mul, false)
    }

    fn shift(&self, offset: T) -> Self {
        B::vector_broadcast(self, offset, BinaryOp::Add, false)
    }

    fn sqrt(&self) -> Self {
        B::vector_unary(self, Analytic::Sqrt)
    }

    fn record<'t>(&self, tape: &'t Tape<B>) -> Var<'t, Self, B> {
        tape.vector(self.to_backend::<B>())
    }

    /// Over a `1 × len` space. A gradient that is a matrix or a view must be
    /// one row or one column of `len`.
    #[track_caller]
    fn fused(
        program: &Program<T>,
        inputs: &[&(dyn FusableOf<T, B> + '_)],
        updated: &mut [&mut Self],
    ) -> Vec<Self> {
        let len = updated
            .first()
            .map(|v| v.len())
            .or_else(|| inputs.first().map(|input| input.source().len))
            .unwrap_or(0);
        for input in inputs {
            if let Some((rows, cols)) = input.shape() {
                assert!(
                    rows * cols == len && (rows == 1 || cols == 1),
                    "a {rows}×{cols} gradient for a vector of {len}"
                );
            }
        }
        run_fused(program, (1, len), inputs, updated)
            .into_iter()
            .map(Output::into_vector)
            .collect()
    }
}

impl<T: Element, B: Kernels<T>> Parameter for Matrix<T, B> {
    type Elem = T;
    type Backend = B;
    type Gradient<'a> = dyn FusableOf<T, B> + 'a;

    fn as_gradient(&self) -> &(dyn FusableOf<T, B> + '_) {
        self
    }

    fn zeros_like(&self) -> Self {
        let (rows, cols) = self.shape();
        Matrix::filled(rows, cols, T::zero())
    }

    fn duplicate(&self) -> Self {
        self.to_backend::<B>()
    }

    fn add(&self, other: &Self) -> Self {
        B::matrix_elementwise(self, other, BinaryOp::Add)
    }

    fn subtract(&self, other: &Self) -> Self {
        B::matrix_elementwise(self, other, BinaryOp::Sub)
    }

    fn multiply(&self, other: &Self) -> Self {
        B::matrix_elementwise(self, other, BinaryOp::Mul)
    }

    fn divide(&self, other: &Self) -> Self {
        B::matrix_elementwise(self, other, BinaryOp::Div)
    }

    fn scale(&self, factor: T) -> Self {
        B::matrix_broadcast(self, factor, BinaryOp::Mul, false)
    }

    fn shift(&self, offset: T) -> Self {
        B::matrix_broadcast(self, offset, BinaryOp::Add, false)
    }

    fn sqrt(&self) -> Self {
        B::matrix_unary(self, Analytic::Sqrt)
    }

    fn record<'t>(&self, tape: &'t Tape<B>) -> Var<'t, Self, B> {
        tape.matrix(self.to_backend::<B>())
    }

    /// Over the parameters' shape. A gradient that is a matrix or a view must
    /// have that shape too: one with as many elements in another shape — the
    /// transpose, say — is a mistake rather than something to read flat.
    #[track_caller]
    fn fused(
        program: &Program<T>,
        inputs: &[&(dyn FusableOf<T, B> + '_)],
        updated: &mut [&mut Self],
    ) -> Vec<Self> {
        let shape = updated
            .first()
            .map(|m| m.shape())
            .or_else(|| inputs.first().and_then(|input| input.shape()))
            .unwrap_or((0, 0));
        for input in inputs {
            if let Some(other) = input.shape() {
                assert_eq!(
                    other, shape,
                    "a {}×{} gradient for {}×{} parameters",
                    other.0, other.1, shape.0, shape.1
                );
            }
        }
        run_fused(program, shape, inputs, updated)
            .into_iter()
            .map(Output::into_matrix)
            .collect()
    }
}

/// A tensor of any rank, run as a fused program reads one: over the space of
/// its rows, every axis but the last folded into them, or over `1 × len`
/// below two axes. The update is the same elementwise program either way,
/// so a `[H, D, D]` weight takes the steps a `(H·D) × D` matrix would.
impl<T: Element, B: Kernels<T>> Parameter for Tensor<T, B> {
    type Elem = T;
    type Backend = B;
    type Gradient<'a> = dyn FusableOf<T, B> + 'a;

    fn as_gradient(&self) -> &(dyn FusableOf<T, B> + '_) {
        self
    }

    fn zeros_like(&self) -> Self {
        Tensor::filled(self.shape(), T::zero())
    }

    fn duplicate(&self) -> Self {
        self.contiguous()
    }

    fn add(&self, other: &Self) -> Self {
        self.elementwise(other, BinaryOp::Add)
    }

    fn subtract(&self, other: &Self) -> Self {
        self.elementwise(other, BinaryOp::Sub)
    }

    fn multiply(&self, other: &Self) -> Self {
        self.elementwise(other, BinaryOp::Mul)
    }

    fn divide(&self, other: &Self) -> Self {
        self.elementwise(other, BinaryOp::Div)
    }

    fn scale(&self, factor: T) -> Self {
        Tensor::scale(self, factor)
    }

    fn shift(&self, offset: T) -> Self {
        self.with_scalar(offset, BinaryOp::Add, false)
    }

    fn sqrt(&self) -> Self {
        self.unary(Analytic::Sqrt)
    }

    fn record<'t>(&self, tape: &'t Tape<B>) -> Var<'t, Self, B> {
        tape.tensor(self.contiguous())
    }

    /// Over the parameters' rows. A gradient with a shape — a tensor of two
    /// or more axes, a matrix or a view — must have the parameters' number of
    /// elements and, for parameters of two or more axes, their row length.
    #[track_caller]
    fn fused(
        program: &Program<T>,
        inputs: &[&(dyn FusableOf<T, B> + '_)],
        updated: &mut [&mut Self],
    ) -> Vec<Self> {
        let dims = updated.first().map(|t| t.shape().to_vec());
        let len = dims
            .as_ref()
            .map(|dims| dims.iter().product())
            .or_else(|| inputs.first().map(|input| input.source().len))
            .unwrap_or(0);
        let space = match dims.as_deref() {
            Some([leading @ .., last]) if !leading.is_empty() => {
                (leading.iter().product::<usize>(), *last)
            }
            _ => (1, len),
        };
        for input in inputs {
            if let Some((rows, cols)) = input.shape() {
                let fits = rows * cols == len && (space.0 == 1 || cols == space.1);
                assert!(
                    fits,
                    "a {rows}×{cols} gradient for parameters of shape {:?}",
                    dims.as_deref().unwrap_or(&[len])
                );
            }
        }
        let dims = dims.unwrap_or_else(|| vec![len]);
        run_fused(program, space, inputs, updated)
            .into_iter()
            .map(|output| Tensor::from_vector(&dims, output.into_vector()))
            .collect()
    }
}

/// How a gradient becomes a parameter update.
///
/// A rule instance belongs to one parameter tensor, because that is where its
/// state lives: reusing an [`Adam`] across two different weights would mix their
/// moment estimates.
pub trait Rule<P: Parameter> {
    /// Apply one update in place.
    ///
    /// The gradient is read the way a fused program reads an input: for a
    /// tensor, the same tensor type or a view of a larger one, of the
    /// parameters' element type. Generic code over `P` passes
    /// [`gradient.as_gradient()`](Parameter::as_gradient).
    ///
    /// ```
    /// use tensorcrate::optim::{Adam, Rule};
    /// use tensorcrate::tensors::Matrix;
    ///
    /// // The weights' gradient is the left block of a packed one.
    /// let mut weights = Matrix::<f32>::from_flat(2, 2, vec![1.0; 4]);
    /// let packed = Matrix::<f32>::from_flat(2, 3, vec![0.5; 6]);
    /// Adam::new(0.1).update(&mut weights, &packed.view(.., 0..2));
    /// ```
    ///
    /// The element type is part of the gradient's type, so a gradient of
    /// another one does not compile:
    ///
    /// ```compile_fail
    /// use tensorcrate::numbers::f16;
    /// use tensorcrate::optim::{Rule, Sgd};
    /// use tensorcrate::tensors::Vector;
    ///
    /// let mut parameters = Vector::new(vec![0.0f32; 4]);
    /// let gradient: Vector<f16> = Vector::new(vec![f16::ONE; 4]);
    /// Sgd::new(0.1).update(&mut parameters, &gradient);
    /// ```
    fn update(&mut self, parameters: &mut P, gradient: &P::Gradient<'_>);

    /// Forget accumulated state, keeping the hyperparameters — for restarting a
    /// fit without rebuilding the rule.
    fn reset(&mut self);
}

/// Plain gradient descent: `p ← p − rate·g`.
///
/// The step size is the whole algorithm, and it has to respect the curvature:
/// descent diverges above `2/λmax` of the loss's Hessian.
#[derive(Clone, Debug)]
pub struct Sgd<T = f32> {
    program: Program<T>,
}

impl<T> Sgd<T>
where
    T: Element + Real,
{
    pub fn new(rate: T) -> Self {
        let dtype = T::DTYPE;
        let mut b = Builder::new();
        let g = b.input(dtype);
        let p = b.update(dtype);
        let step = b.scale(g, rate);
        let p = b.sub(p, step);
        b.set(0, p);
        let program = b.build().expect("the rule's program is valid");
        Sgd { program }
    }
}

impl<P: Parameter> Rule<P> for Sgd<P::Elem> {
    fn update(&mut self, parameters: &mut P, gradient: &P::Gradient<'_>) {
        P::fused(&self.program, &[gradient], &mut [parameters]);
    }

    fn reset(&mut self) {}
}

/// Descent with a velocity: `v ← μv + g`, then `p ← p − rate·v`.
///
/// The velocity averages successive gradients, which cancels the oscillation
/// across a narrow valley and accumulates along it — the reason momentum beats
/// plain descent on badly conditioned problems.
///
/// With [`nesterov`](Self::nesterov) set, the step looks ahead: `p ← p −
/// rate·(g + μv)`, applying the momentum term where the parameters are about to
/// be rather than where they are.
#[derive(Clone, Debug)]
pub struct Momentum<P: Parameter> {
    program: Program<P::Elem>,
    velocity: Option<P>,
}

impl<P: Parameter> Momentum<P> {
    /// Classical momentum; `0.9` is the usual coefficient.
    pub fn new(rate: P::Elem, momentum: P::Elem) -> Self {
        let dtype = <P::Elem as Element>::DTYPE;
        let mut b = Builder::new();
        let g = b.input(dtype);
        let p = b.update(dtype);
        let previous = b.update(dtype);
        let decayed = b.scale(previous, momentum);
        let velocity = b.add(decayed, g);
        let step = b.scale(velocity, rate);
        let p = b.sub(p, step);
        b.set(0, p);
        b.set(1, velocity);
        let program = b.build().expect("the rule's program is valid");

        Momentum {
            program,
            velocity: None,
        }
    }

    /// The look-ahead variant.
    pub fn nesterov(rate: P::Elem, momentum: P::Elem) -> Self {
        let dtype = <P::Elem as Element>::DTYPE;
        let mut b = Builder::new();
        let g = b.input(dtype);
        let p = b.update(dtype);
        let previous = b.update(dtype);
        let decayed = b.scale(previous, momentum);
        let velocity = b.add(decayed, g);
        let ahead = b.scale(velocity, momentum);
        let step = b.add(g, ahead);
        let step = b.scale(step, rate);
        let p = b.sub(p, step);
        b.set(0, p);
        b.set(1, velocity);
        let program = b.build().expect("the rule's program is valid");

        Momentum {
            program,
            velocity: None,
        }
    }
}

impl<P: Parameter> Rule<P> for Momentum<P> {
    /// One fused kernel: the velocity and parameters are updated in place.
    fn update(&mut self, parameters: &mut P, gradient: &P::Gradient<'_>) {
        let velocity = if let Some(velocity) = &mut self.velocity {
            velocity
        } else {
            self.velocity.insert(parameters.zeros_like())
        };

        P::fused(&self.program, &[gradient], &mut [parameters, velocity]);
    }

    fn reset(&mut self) {
        self.velocity = None;
    }
}

/// Per-parameter rates from the running sum of squared gradients:
/// `G ← G + g⊙g`, then `p ← p − rate·g/(√G + ε)`.
///
/// Rarely-moved parameters keep a large effective step. The denominator only
/// grows, so the steps only shrink — which is why [`RmsProp`] exists.
#[derive(Clone, Debug)]
pub struct AdaGrad<P: Parameter> {
    program: Program<P::Elem>,
    total: Option<P>,
}

impl<P: Parameter> AdaGrad<P> {
    pub fn new(rate: P::Elem) -> Self {
        let dtype = <P::Elem as Element>::DTYPE;
        let mut b = Builder::new();
        let g = b.input(dtype);
        let p = b.update(dtype);
        let squared = b.mul(g, g);
        let previous = b.update(dtype);
        let total = b.add(previous, squared);
        let epsilon = <P::Elem as Real>::from_f64(1e-8);
        let p = descend(&mut b, p, g, total, epsilon, rate);
        b.set(0, p);
        b.set(1, total);
        let program = b.build().expect("the rule's program is valid");

        AdaGrad {
            program,
            total: None,
        }
    }
}

impl<P: Parameter> Rule<P> for AdaGrad<P> {
    /// One fused kernel, updating the parameters and the running total in place.
    fn update(&mut self, parameters: &mut P, gradient: &P::Gradient<'_>) {
        let total = if let Some(total) = &mut self.total {
            total
        } else {
            self.total.insert(parameters.zeros_like())
        };
        P::fused(&self.program, &[gradient], &mut [parameters, total]);
    }

    fn reset(&mut self) {
        self.total = None;
    }
}

/// [`AdaGrad`] with a forgetting factor: `S ← ρS + (1−ρ)g⊙g`, then
/// `p ← p − rate·g/(√S + ε)`.
///
/// The exponential average keeps the denominator from growing without bound, so
/// the effective step size adapts instead of decaying to nothing.
#[derive(Clone, Debug)]
pub struct RmsProp<P: Parameter> {
    program: Program<P::Elem>,
    mean_square: Option<P>,
}

impl<P: Parameter> RmsProp<P> {
    pub fn new(rate: P::Elem) -> Self {
        let dtype = <P::Elem as Element>::DTYPE;
        let mut b = Builder::new();
        let g = b.input(dtype);
        let p = b.update(dtype);
        let squared = b.mul(g, g);
        let decay = <P::Elem as Real>::from_f64(0.9);
        let epsilon = <P::Elem as Real>::from_f64(1e-8);
        let squared = b.scale(squared, <P::Elem as num_traits::One>::one() - decay);
        let previous = b.update(dtype);
        let decayed = b.scale(previous, decay);
        let mean_square = b.add(decayed, squared);
        let p = descend(&mut b, p, g, mean_square, epsilon, rate);
        b.set(0, p);
        b.set(1, mean_square);
        let program = b.build().expect("the rule's program is valid");

        RmsProp {
            program,
            mean_square: None,
        }
    }
}

impl<P: Parameter> Rule<P> for RmsProp<P> {
    /// One fused kernel, updating the parameters and the mean square in place.
    fn update(&mut self, parameters: &mut P, gradient: &P::Gradient<'_>) {
        let mean_square = if let Some(mean_square) = &mut self.mean_square {
            mean_square
        } else {
            self.mean_square.insert(parameters.zeros_like())
        };
        P::fused(&self.program, &[gradient], &mut [parameters, mean_square]);
    }

    fn reset(&mut self) {
        self.mean_square = None;
    }
}

/// Momentum and per-parameter scaling together, with the bias correction that
/// makes the first few steps behave.
///
/// `m ← β₁m + (1−β₁)g`, `v ← β₂v + (1−β₂)g⊙g`, and after correcting both for
/// their zero initialization, `p ← p − rate·m̂/(√v̂ + ε)`.
///
/// Its two fused programs — the first step's, which creates the moments, and
/// every later step's, which updates them in place — are built once, by
/// [`new`](Self::new). The coefficients and the bias corrections are their
/// [uniforms](crate::tensors::fused::Builder::uniform), set before each run, so
/// an update builds nothing; changing a coefficient field between updates
/// takes effect at the next one. The programs are optimized under the thread's
/// [`Algebra`](crate::tensors::fused::Algebra) when the rule is created.
#[derive(Clone, Debug)]
pub struct Adam<P: Parameter> {
    pub rate: P::Elem,
    pub first_decay: P::Elem,
    pub second_decay: P::Elem,
    pub epsilon: P::Elem,
    first: Option<P>,
    second: Option<P>,
    steps: u32,
    /// The first step: the moments are fresh outputs.
    start: Program<P::Elem>,
    /// Every later step: the moments are updated in place.
    resume: Program<P::Elem>,
}

/// The uniforms of Adam's programs, by index.
mod adam {
    pub const RATE: usize = 0;
    pub const FIRST_DECAY: usize = 1;
    pub const SECOND_DECAY: usize = 2;
    pub const EPSILON: usize = 3;
    /// `1 / (1 − β₁ᵗ)`.
    pub const FIRST_CORRECTION: usize = 4;
    /// `1 / (1 − β₂ᵗ)`.
    pub const SECOND_CORRECTION: usize = 5;
}

impl<P: Parameter> Adam<P> {
    /// The usual coefficients: `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`.
    pub fn new(rate: P::Elem) -> Self {
        Adam {
            rate,
            first_decay: <P::Elem as Real>::from_f64(0.9),
            second_decay: <P::Elem as Real>::from_f64(0.999),
            epsilon: <P::Elem as Real>::from_f64(1e-8),
            first: None,
            second: None,
            steps: 0,
            start: Self::program(false),
            resume: Self::program(true),
        }
    }

    /// One step as a program over the gradient (input 0) and the parameters
    /// (updated 0), with the moments updated in place when `resumed` and
    /// returned as fresh outputs otherwise. Every coefficient is a uniform, its
    /// placeholder value one, so the program's structure is the same whatever
    /// the coefficients turn out to be.
    fn program(resumed: bool) -> Program<P::Elem> {
        let dtype = <P::Elem as Element>::DTYPE;
        let one = <P::Elem as num_traits::One>::one();
        let mut b = Builder::new();
        // In the order of the indices in `adam`.
        let rate = b.uniform(one);
        let first_decay = b.uniform(one);
        let second_decay = b.uniform(one);
        let epsilon = b.uniform(one);
        let first_correction = b.uniform(one);
        let second_correction = b.uniform(one);
        let g = b.input(dtype);
        let p = b.update(dtype);
        // `1 − β` involves only a constant and a uniform, so it is folded into
        // the program's constants and computed once per run, in the element
        // type, exactly as `one - decay` was.
        let unit = b.constant(one);
        let first_share = b.sub(unit, first_decay);
        let second_share = b.sub(unit, second_decay);

        let fresh_first = b.mul(g, first_share);
        let squared = b.mul(g, g);
        let fresh_second = b.mul(squared, second_share);
        let (first, second) = if resumed {
            let (m, v) = (b.update(dtype), b.update(dtype));
            let m = b.mul(m, first_decay);
            let v = b.mul(v, second_decay);
            (b.add(m, fresh_first), b.add(v, fresh_second))
        } else {
            (fresh_first, fresh_second)
        };

        // Both moments start at zero, so early estimates are biased toward it;
        // dividing by `1 − βᵗ` undoes exactly that.
        let corrected_first = b.mul(first, first_correction);
        let corrected_second = b.mul(second, second_correction);

        let root = b.unary(Analytic::Sqrt, corrected_second);
        let denominator = b.add(root, epsilon);
        let step = b.div(corrected_first, denominator);
        let step = b.mul(step, rate);
        let p = b.sub(p, step);
        b.set(0, p);
        if resumed {
            b.set(1, first);
            b.set(2, second);
        } else {
            b.output(first, dtype);
            b.output(second, dtype);
        }
        b.build().expect("the rule's program is valid")
    }
}

impl<P: Parameter> Rule<P> for Adam<P> {
    /// One fused kernel: parameters and both moments are read once and
    /// overwritten in place. Unfused, this was fourteen kernels and eleven
    /// intermediate tensors.
    fn update(&mut self, parameters: &mut P, gradient: &P::Gradient<'_>) {
        self.steps += 1;

        let one = <P::Elem as num_traits::One>::one();
        let first_correction = one - self.first_decay.powi(self.steps as i32);
        let second_correction = one - self.second_decay.powi(self.steps as i32);
        let resumed = self.first.is_some() && self.second.is_some();
        let program = if resumed {
            &mut self.resume
        } else {
            &mut self.start
        };
        program.set_uniform(adam::RATE, self.rate);
        program.set_uniform(adam::FIRST_DECAY, self.first_decay);
        program.set_uniform(adam::SECOND_DECAY, self.second_decay);
        program.set_uniform(adam::EPSILON, self.epsilon);
        program.set_uniform(adam::FIRST_CORRECTION, first_correction.recip());
        program.set_uniform(adam::SECOND_CORRECTION, second_correction.recip());

        match (&mut self.first, &mut self.second) {
            (Some(first), Some(second)) => {
                P::fused(program, &[gradient], &mut [parameters, first, second]);
            }
            _ => {
                let mut moments = P::fused(program, &[gradient], &mut [parameters]).into_iter();
                self.first = moments.next();
                self.second = moments.next();
            }
        }
    }

    fn reset(&mut self) {
        self.first = None;
        self.second = None;
        self.steps = 0;
    }
}

/// `p − rate · g / (√s + ε)`, the step the adaptive rules share.
fn descend<T: Real>(
    b: &mut Builder<T>,
    p: crate::tensors::fused::FusedValue,
    g: crate::tensors::fused::FusedValue,
    scale: crate::tensors::fused::FusedValue,
    epsilon: T,
    rate: T,
) -> crate::tensors::fused::FusedValue {
    let root = b.unary(Analytic::Sqrt, scale);
    let denominator = b.shift(root, epsilon);
    let step = b.div(g, denominator);
    let step = b.scale(step, rate);
    b.sub(p, step)
}

/// The named projection a [`Constrained`] rule applies after each update.
///
/// A closure has no `Debug`, so the name stands in for it there.
struct Projection<'a, P> {
    name: String,
    apply: Arc<dyn Fn(&mut P) + 'a>,
}

impl<P> Clone for Projection<'_, P> {
    fn clone(&self) -> Self {
        Projection {
            name: self.name.clone(),
            apply: Arc::clone(&self.apply),
        }
    }
}

impl<P> std::fmt::Debug for Projection<'_, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Projection({})", self.name)
    }
}

/// Constrained rule for a custom parameter type. Applies a
/// projection after the update, so the parameter stays in a feasible set. The
/// projection is a closure, so it can be anything: clipping, normalization,
/// or a more complicated operation.
#[derive(Clone, Debug)]
pub struct Constrained<'a, P, R> {
    rule: R,
    projection: Projection<'a, P>,
}

impl<'a, P, R> Constrained<'a, P, R>
where
    P: Parameter,
    R: Rule<P>,
{
    /// Wrap `rule` so that `projection` runs after every update. `name`
    /// identifies the projection in `Debug` output.
    pub fn new<S: Into<String>, F: Fn(&mut P) + 'a>(name: S, rule: R, projection: F) -> Self {
        Constrained {
            rule,
            projection: Projection {
                name: name.into(),
                apply: Arc::new(projection),
            },
        }
    }
}

impl<'a, P: Parameter, R: Rule<P>> Rule<P> for Constrained<'a, P, R> {
    fn update(&mut self, parameters: &mut P, gradient: &P::Gradient<'_>) {
        self.rule.update(parameters, gradient);
        (self.projection.apply)(parameters);
    }

    fn reset(&mut self) {
        self.rule.reset();
    }
}

/// Run `steps` updates of `rule` on `parameters`, minimizing whatever scalar
/// `objective` builds.
///
/// Each step records the parameters on a fresh tape, evaluates the objective,
/// propagates once, and hands the gradient to the rule. The step number is passed
/// through so a stochastic objective can pick its mini-batch; a full-batch one
/// ignores it.
///
/// Returns the loss at the last step, which is the cheapest useful progress
/// signal. For a loss curve, keep one in the closure.
pub fn minimize<P, R>(
    parameters: &mut P,
    rule: &mut R,
    steps: usize,
    objective: impl for<'t> Fn(&Var<'t, P, P::Backend>, usize) -> ScalarVar<'t, P::Backend, P::Elem>,
) -> P::Elem
where
    P: Parameter + Adjoint<P::Backend>,
    R: Rule<P>,
{
    let mut last = <P::Elem as Float>::nan();
    for step in 0..steps {
        let tape = Tape::<P::Backend>::new();
        let recorded = parameters.record(&tape);
        let loss = objective(&recorded, step);
        loss.backward();
        last = *loss.value();
        rule.update(parameters, recorded.grad().as_gradient());
    }
    last
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensors::fused::DType;

    /// [`run_by_parts`], for parameter types without a program runner, has to
    /// give the same answers as the real one.
    #[test]
    fn the_provided_parameter_runner_matches_the_fused_one() {
        let mut b = Builder::new();
        let g = b.input(DType::F32);
        let p = b.update(DType::F32);
        let m = b.update(DType::F32);
        let decayed = b.scale(m, 0.9f32);
        let m = b.add(decayed, g);
        let root = b.unary(Analytic::Sqrt, m);
        let lifted = b.shift(root, 1e-3);
        let three = b.constant(3.0);
        let thirds = b.div(lifted, three);
        let step = b.div(g, thirds);
        let p = b.sub(p, step);
        b.set(0, p);
        b.set(1, m);
        b.output(thirds, DType::F32);
        let program = b.build().unwrap();

        let g = Vector::new(
            (0..100)
                .map(|i| (i as f32 * 0.37).sin())
                .collect::<Vec<_>>(),
        );
        let start = Vector::new((0..100).map(|i| i as f32 * 0.01).collect::<Vec<_>>());
        let moment = Vector::new(vec![0.5f32; 100]);

        let (mut p1, mut m1) = (start.clone(), moment.clone());
        let fresh1 = Parameter::fused(&program, &[g.as_gradient()], &mut [&mut p1, &mut m1]);
        let (mut p2, mut m2) = (start, moment);
        let fresh2 = run_by_parts(&program, &[&g], &mut [&mut p2, &mut m2]);

        let bits = |v: &Vector<f32>| v.as_slice().iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&p1), bits(&p2));
        assert_eq!(bits(&m1), bits(&m2));
        assert_eq!(bits(&fresh1[0]), bits(&fresh2[0]));
    }
}
