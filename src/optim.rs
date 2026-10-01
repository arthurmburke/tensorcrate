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

use crate::tensors::fused::{Builder, DType, Fusable, Instr, Output, Program, Remap};
use crate::tensors::tape::Adjoint;
use crate::tensors::{Analytic, BinaryOp, Host, Kernels, Matrix, ScalarVar, Tape, Var, Vector};

/// A tensor an optimizer can carry: the elementwise algebra the update rules
/// need, plus the ability to put itself on a tape.
///
/// Implemented for `f32`, [`Vector`] and [`Matrix`], so a rule is written once
/// and applies to any of them — on either backend, without leaving it.
pub trait Parameter: Sized + 'static {
    /// Where this parameter's elements live.
    type Backend: Kernels;

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
    fn scale(&self, factor: f32) -> Self;

    /// Add a constant to every element.
    fn shift(&self, offset: f32) -> Self;

    /// Elementwise square root, for the adaptive rules' denominators.
    fn sqrt(&self) -> Self;

    /// Record this parameter as a tape leaf, so a backward pass can produce its
    /// gradient.
    fn record<'t>(&self, tape: &'t Tape<Self::Backend>) -> Var<'t, Self, Self::Backend>
    where
        Self: Adjoint<Self::Backend>;

    /// Run a fused elementwise [`Program`] over tensors of this parameter's
    /// shape: `inputs` are read, `updated` are overwritten in place, and the
    /// program's fresh outputs are returned. Every slot is `f32` and read
    /// unremapped.
    ///
    /// This is how the rules below take a whole update in one kernel. The
    /// provided implementation is for parameter types of your own: it runs the
    /// program one operation at a time through the methods above, so it
    /// supports exactly the operations they spell — `+ − × ÷`, `sqrt`, and
    /// constants — and panics on anything else.
    fn fused(program: &Program, inputs: &[&Self], updated: &mut [&mut Self]) -> Vec<Self> {
        unfused_parameter(program, inputs, updated)
    }
}

/// [`Parameter::fused`] through the trait's own elementwise methods.
fn unfused_parameter<P: Parameter>(
    program: &Program,
    inputs: &[&P],
    updated: &mut [&mut P],
) -> Vec<P> {
    enum Value<P> {
        Scalar(f32),
        Tensor(P),
    }
    // Any operand fixes the shape a constant has to be filled out to.
    let like = inputs
        .first()
        .map(|p| p.zeros_like())
        .or_else(|| updated.first().map(|p| p.zeros_like()))
        .expect("a parameter program has at least one tensor operand");
    let fill = |value: f32| like.shift(value);
    let tensor = |value: &Value<P>| match value {
        Value::Scalar(value) => fill(*value),
        Value::Tensor(tensor) => tensor.duplicate(),
    };

    let fresh = program.fresh_inputs();
    let mut registers: Vec<Option<Value<P>>> = (0..16).map(|_| None).collect();
    let mut outputs: Vec<Option<P>> = (0..program.outputs().len()).map(|_| None).collect();
    for instr in program.code() {
        let get = |reg: u8| registers[usize::from(reg)].as_ref().expect("validated");
        let (dst, value) = match *instr {
            Instr::Load { dst, input, remap } => {
                assert_eq!(remap, Remap::Identity, "parameter programs read unremapped");
                let slot = usize::from(input);
                let source: &P = if slot < fresh {
                    inputs[slot]
                } else {
                    &*updated[slot - fresh]
                };
                (dst, Value::Tensor(source.duplicate()))
            }
            Instr::Const { dst, value } => (dst, Value::Scalar(value)),
            Instr::Binary { dst, op, a, b } => {
                let value = match (get(a), get(b), op) {
                    (Value::Scalar(a), Value::Scalar(b), _) => Value::Scalar(match op {
                        BinaryOp::Add => a + b,
                        BinaryOp::Sub => a - b,
                        BinaryOp::Mul => a * b,
                        BinaryOp::Div => a / b,
                        BinaryOp::Rem => a % b,
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

/// Run `program` over same-shaped tensors, which is every parameter program.
fn run_fused<B: Kernels, T: Fusable<B>>(
    program: &Program,
    shape: (usize, usize),
    inputs: &[&T],
    updated: &mut [&mut T],
) -> Vec<Output<B>> {
    let inputs: Vec<&dyn Fusable<B>> = inputs
        .iter()
        .map(|&input| input as &dyn Fusable<B>)
        .collect();
    let mut updated: Vec<&mut dyn Fusable<B>> = updated
        .iter_mut()
        .map(|target| &mut **target as &mut dyn Fusable<B>)
        .collect();
    program.run(shape, &inputs, &mut updated)
}

/// A scalar parameter — a learned temperature, or a log-variance. It carries no
/// storage of its own, so its tape is the host one; a scalar living inside a
/// resident graph is a job for [`Tape`] directly rather than for [`minimize`].
impl Parameter for f32 {
    type Backend = Host;

    fn zeros_like(&self) -> Self {
        0.0
    }

    fn duplicate(&self) -> Self {
        *self
    }

    fn add(&self, other: &Self) -> Self {
        self + other
    }

    fn subtract(&self, other: &Self) -> Self {
        self - other
    }

    fn multiply(&self, other: &Self) -> Self {
        self * other
    }

    fn divide(&self, other: &Self) -> Self {
        self / other
    }

    fn scale(&self, factor: f32) -> Self {
        self * factor
    }

    fn shift(&self, offset: f32) -> Self {
        self + offset
    }

    fn sqrt(&self) -> Self {
        f32::sqrt(*self)
    }

    fn record<'t>(&self, tape: &'t Tape<Host>) -> Var<'t, Self, Host> {
        tape.scalar(*self)
    }

    fn fused(program: &Program, inputs: &[&Self], updated: &mut [&mut Self]) -> Vec<Self> {
        let inputs: Vec<Vector<f32>> = inputs.iter().map(|&&x| Vector::new([x])).collect();
        let mut targets: Vec<Vector<f32>> = updated.iter().map(|x| Vector::new([**x])).collect();
        let outputs = {
            let inputs: Vec<&Vector<f32>> = inputs.iter().collect();
            let mut targets: Vec<&mut Vector<f32>> = targets.iter_mut().collect();
            run_fused(program, (1, 1), &inputs, &mut targets)
        };
        for (target, value) in updated.iter_mut().zip(&targets) {
            **target = value[0];
        }
        outputs
            .into_iter()
            .map(|output| output.into_vector::<f32>().as_slice()[0])
            .collect()
    }
}

impl<B: Kernels> Parameter for Vector<f32, B> {
    type Backend = B;

    fn zeros_like(&self) -> Self {
        Vector::filled(self.len(), 0.0)
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

    fn scale(&self, factor: f32) -> Self {
        B::vector_broadcast(self, factor, BinaryOp::Mul, false)
    }

    fn shift(&self, offset: f32) -> Self {
        B::vector_broadcast(self, offset, BinaryOp::Add, false)
    }

    fn sqrt(&self) -> Self {
        B::vector_unary(self, Analytic::Sqrt)
    }

    fn record<'t>(&self, tape: &'t Tape<B>) -> Var<'t, Self, B> {
        tape.vector(self.to_backend::<B>())
    }

    fn fused(program: &Program, inputs: &[&Self], updated: &mut [&mut Self]) -> Vec<Self> {
        let len = inputs
            .first()
            .map(|v| v.len())
            .or_else(|| updated.first().map(|v| v.len()))
            .unwrap_or(0);
        run_fused(program, (1, len), inputs, updated)
            .into_iter()
            .map(Output::into_vector)
            .collect()
    }
}

impl<B: Kernels> Parameter for Matrix<f32, B> {
    type Backend = B;

    fn zeros_like(&self) -> Self {
        let (rows, cols) = self.shape();
        Matrix::filled(rows, cols, 0.0)
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

    fn scale(&self, factor: f32) -> Self {
        B::matrix_broadcast(self, factor, BinaryOp::Mul, false)
    }

    fn shift(&self, offset: f32) -> Self {
        B::matrix_broadcast(self, offset, BinaryOp::Add, false)
    }

    fn sqrt(&self) -> Self {
        B::matrix_unary(self, Analytic::Sqrt)
    }

    fn record<'t>(&self, tape: &'t Tape<B>) -> Var<'t, Self, B> {
        tape.matrix(self.to_backend::<B>())
    }

    fn fused(program: &Program, inputs: &[&Self], updated: &mut [&mut Self]) -> Vec<Self> {
        let shape = inputs
            .first()
            .map(|m| m.shape())
            .or_else(|| updated.first().map(|m| m.shape()))
            .unwrap_or((0, 0));
        run_fused(program, shape, inputs, updated)
            .into_iter()
            .map(Output::into_matrix)
            .collect()
    }
}

/// How a gradient becomes a parameter update.
///
/// A rule instance belongs to one parameter tensor, because that is where its
/// state lives: reusing an [`Adam`] across two different weights would mix their
/// moment estimates.
pub trait Rule<P> {
    /// Apply one update in place.
    fn update(&mut self, parameters: &mut P, gradient: &P);

    /// Forget accumulated state, keeping the hyperparameters — for restarting a
    /// fit without rebuilding the rule.
    fn reset(&mut self);
}

/// Plain gradient descent: `p ← p − rate·g`.
///
/// The step size is the whole algorithm, and it has to respect the curvature:
/// descent diverges above `2/λmax` of the loss's Hessian.
#[derive(Copy, Clone, Debug)]
pub struct Sgd {
    pub rate: f32,
}

impl Sgd {
    pub fn new(rate: f32) -> Self {
        Sgd { rate }
    }
}

impl<P: Parameter> Rule<P> for Sgd {
    fn update(&mut self, parameters: &mut P, gradient: &P) {
        let mut b = Builder::new();
        let g = b.input(DType::F32);
        let p = b.update(DType::F32);
        let step = b.scale(g, self.rate);
        let p = b.sub(p, step);
        b.set(0, p);
        let program = b.build().expect("the rule's program is valid");
        P::fused(&program, &[gradient], &mut [parameters]);
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
pub struct Momentum<P> {
    pub rate: f32,
    pub momentum: f32,
    pub nesterov: bool,
    velocity: Option<P>,
}

impl<P> Momentum<P> {
    /// Classical momentum; `0.9` is the usual coefficient.
    pub fn new(rate: f32, momentum: f32) -> Self {
        Momentum {
            rate,
            momentum,
            nesterov: false,
            velocity: None,
        }
    }

    /// The look-ahead variant.
    pub fn nesterov(rate: f32, momentum: f32) -> Self {
        Momentum {
            rate,
            momentum,
            nesterov: true,
            velocity: None,
        }
    }
}

impl<P: Parameter> Rule<P> for Momentum<P> {
    /// One fused kernel: the velocity and parameters are updated in place.
    fn update(&mut self, parameters: &mut P, gradient: &P) {
        let mut b = Builder::new();
        let g = b.input(DType::F32);
        let p = b.update(DType::F32);
        let resumed = self.velocity.is_some();
        let velocity = if resumed {
            let previous = b.update(DType::F32);
            let decayed = b.scale(previous, self.momentum);
            b.add(decayed, g)
        } else {
            g
        };
        let step = if self.nesterov {
            let ahead = b.scale(velocity, self.momentum);
            b.add(g, ahead)
        } else {
            velocity
        };
        let step = b.scale(step, self.rate);
        let p = b.sub(p, step);
        b.set(0, p);
        if resumed {
            b.set(1, velocity);
        } else {
            b.output(velocity, DType::F32);
        }
        let program = b.build().expect("the rule's program is valid");

        match &mut self.velocity {
            Some(velocity) => {
                P::fused(&program, &[gradient], &mut [parameters, velocity]);
            }
            None => {
                self.velocity = P::fused(&program, &[gradient], &mut [parameters]).pop();
            }
        }
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
pub struct AdaGrad<P> {
    pub rate: f32,
    pub epsilon: f32,
    total: Option<P>,
}

impl<P> AdaGrad<P> {
    pub fn new(rate: f32) -> Self {
        AdaGrad {
            rate,
            epsilon: 1e-8,
            total: None,
        }
    }
}

impl<P: Parameter> Rule<P> for AdaGrad<P> {
    /// One fused kernel, updating the parameters and the running total in place.
    fn update(&mut self, parameters: &mut P, gradient: &P) {
        let mut b = Builder::new();
        let g = b.input(DType::F32);
        let p = b.update(DType::F32);
        let resumed = self.total.is_some();
        let squared = b.mul(g, g);
        let total = if resumed {
            let previous = b.update(DType::F32);
            b.add(previous, squared)
        } else {
            squared
        };
        let p = descend(&mut b, p, g, total, self.epsilon, self.rate);
        b.set(0, p);
        if resumed {
            b.set(1, total);
        } else {
            b.output(total, DType::F32);
        }
        let program = b.build().expect("the rule's program is valid");

        match &mut self.total {
            Some(total) => {
                P::fused(&program, &[gradient], &mut [parameters, total]);
            }
            None => self.total = P::fused(&program, &[gradient], &mut [parameters]).pop(),
        }
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
pub struct RmsProp<P> {
    pub rate: f32,
    pub decay: f32,
    pub epsilon: f32,
    mean_square: Option<P>,
}

impl<P> RmsProp<P> {
    pub fn new(rate: f32) -> Self {
        RmsProp {
            rate,
            decay: 0.9,
            epsilon: 1e-8,
            mean_square: None,
        }
    }
}

impl<P: Parameter> Rule<P> for RmsProp<P> {
    /// One fused kernel, updating the parameters and the mean square in place.
    fn update(&mut self, parameters: &mut P, gradient: &P) {
        let mut b = Builder::new();
        let g = b.input(DType::F32);
        let p = b.update(DType::F32);
        let resumed = self.mean_square.is_some();
        let squared = b.mul(g, g);
        let squared = b.scale(squared, 1.0 - self.decay);
        let mean_square = if resumed {
            let previous = b.update(DType::F32);
            let decayed = b.scale(previous, self.decay);
            b.add(decayed, squared)
        } else {
            squared
        };
        let p = descend(&mut b, p, g, mean_square, self.epsilon, self.rate);
        b.set(0, p);
        if resumed {
            b.set(1, mean_square);
        } else {
            b.output(mean_square, DType::F32);
        }
        let program = b.build().expect("the rule's program is valid");

        match &mut self.mean_square {
            Some(mean_square) => {
                P::fused(&program, &[gradient], &mut [parameters, mean_square]);
            }
            None => self.mean_square = P::fused(&program, &[gradient], &mut [parameters]).pop(),
        }
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
#[derive(Clone, Debug)]
pub struct Adam<P> {
    pub rate: f32,
    pub first_decay: f32,
    pub second_decay: f32,
    pub epsilon: f32,
    first: Option<P>,
    second: Option<P>,
    steps: u32,
}

impl<P> Adam<P> {
    /// The usual coefficients: `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`.
    pub fn new(rate: f32) -> Self {
        Adam {
            rate,
            first_decay: 0.9,
            second_decay: 0.999,
            epsilon: 1e-8,
            first: None,
            second: None,
            steps: 0,
        }
    }
}

impl<P: Parameter> Rule<P> for Adam<P> {
    /// One fused kernel: parameters and both moments are read once and
    /// overwritten in place. Unfused, this was fourteen kernels and eleven
    /// intermediate tensors.
    fn update(&mut self, parameters: &mut P, gradient: &P) {
        self.steps += 1;

        let mut b = Builder::new();
        let g = b.input(DType::F32);
        let p = b.update(DType::F32);
        let resumed = self.first.is_some() && self.second.is_some();

        let fresh_first = b.scale(g, 1.0 - self.first_decay);
        let squared = b.mul(g, g);
        let fresh_second = b.scale(squared, 1.0 - self.second_decay);
        let (first, second) = if resumed {
            let (m, v) = (b.update(DType::F32), b.update(DType::F32));
            let m = b.scale(m, self.first_decay);
            let v = b.scale(v, self.second_decay);
            (b.add(m, fresh_first), b.add(v, fresh_second))
        } else {
            (fresh_first, fresh_second)
        };

        // Both moments start at zero, so early estimates are biased toward it;
        // dividing by `1 − βᵗ` undoes exactly that.
        let first_correction = 1.0 - self.first_decay.powi(self.steps as i32);
        let second_correction = 1.0 - self.second_decay.powi(self.steps as i32);
        let corrected_first = b.scale(first, first_correction.recip());
        let corrected_second = b.scale(second, second_correction.recip());

        let root = b.unary(Analytic::Sqrt, corrected_second);
        let denominator = b.shift(root, self.epsilon);
        let step = b.div(corrected_first, denominator);
        let step = b.scale(step, self.rate);
        let p = b.sub(p, step);
        b.set(0, p);
        if resumed {
            b.set(1, first);
            b.set(2, second);
        } else {
            b.output(first, DType::F32);
            b.output(second, DType::F32);
        }
        let program = b.build().expect("the rule's program is valid");

        match (&mut self.first, &mut self.second) {
            (Some(first), Some(second)) => {
                P::fused(&program, &[gradient], &mut [parameters, first, second]);
            }
            _ => {
                let mut moments = P::fused(&program, &[gradient], &mut [parameters]).into_iter();
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
fn descend(
    b: &mut Builder,
    p: crate::tensors::fused::Value,
    g: crate::tensors::fused::Value,
    scale: crate::tensors::fused::Value,
    epsilon: f32,
    rate: f32,
) -> crate::tensors::fused::Value {
    let root = b.unary(Analytic::Sqrt, scale);
    let denominator = b.shift(root, epsilon);
    let step = b.div(g, denominator);
    let step = b.scale(step, rate);
    b.sub(p, step)
}

/// A simple wrapper around a function that allows us
/// to use to in debug structs.
#[derive(Clone)]
struct DebugFn<F> {
    name: String,
    f: F,
}

impl<F> DebugFn<F> {
    pub fn new<S: Into<String>>(name: S, f: F) -> Self {
        DebugFn {
            name: name.into(),
            f,
        }
    }
}

impl<F> std::fmt::Debug for DebugFn<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DebugFn({})", self.name)
    }
}

type Proj<'a, P> = DebugFn<Arc<dyn Fn(&mut P) + 'a>>;

/// Constrained rule for a custom parameter type. Applies a
/// projection after the update, so the parameter stays in a feasible set. The
/// projection is a closure, so it can be anything: clipping, normalization,
/// or a more complicated operation.
#[derive(Clone, Debug)]
pub struct Constrained<'a, P, R> {
    rule: R,
    projection: Proj<'a, P>,
}

impl<'a, P, R> Constrained<'a, P, R>
where
    P: Parameter,
    R: Rule<P>,
{
    pub fn new<S: Into<String>, F: Fn(&mut P) + 'a>(name: S, rule: R, projection: F) -> Self {
        Constrained {
            rule,
            projection: DebugFn::new(name.into(), Arc::new(projection)),
        }
    }
}

impl<'a, P: Parameter, R: Rule<P>> Rule<P> for Constrained<'a, P, R> {
    fn update(&mut self, parameters: &mut P, gradient: &P) {
        self.rule.update(parameters, gradient);
        (self.projection.f)(parameters);
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
    objective: impl for<'t> Fn(&Var<'t, P, P::Backend>, usize) -> ScalarVar<'t, P::Backend>,
) -> f32
where
    P: Parameter + Adjoint<P::Backend>,
    R: Rule<P>,
{
    let mut last = f32::NAN;
    for step in 0..steps {
        let tape = Tape::<P::Backend>::new();
        let recorded = parameters.record(&tape);
        let loss = objective(&recorded, step);
        loss.backward();
        last = *loss.value();
        rule.update(parameters, &recorded.grad());
    }
    last
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The provided [`Parameter::fused`], for parameter types without a
    /// program runner, has to give the same answers as the real one.
    #[test]
    fn the_provided_parameter_runner_matches_the_fused_one() {
        let mut b = Builder::new();
        let g = b.input(DType::F32);
        let p = b.update(DType::F32);
        let m = b.update(DType::F32);
        let decayed = b.scale(m, 0.9);
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
        let fresh1 = Parameter::fused(&program, &[&g], &mut [&mut p1, &mut m1]);
        let (mut p2, mut m2) = (start, moment);
        let fresh2 = unfused_parameter(&program, &[&g], &mut [&mut p2, &mut m2]);

        let bits = |v: &Vector<f32>| v.as_slice().iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&p1), bits(&p2));
        assert_eq!(bits(&m1), bits(&m2));
        assert_eq!(bits(&fresh1[0]), bits(&fresh2[0]));
    }
}
