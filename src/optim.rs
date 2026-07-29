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
        *parameters = parameters.subtract(&gradient.scale(self.rate));
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
    fn update(&mut self, parameters: &mut P, gradient: &P) {
        let velocity = match self.velocity.take() {
            Some(previous) => previous.scale(self.momentum).add(gradient),
            None => gradient.duplicate(),
        };
        let step = if self.nesterov {
            gradient.add(&velocity.scale(self.momentum))
        } else {
            velocity.duplicate()
        };
        *parameters = parameters.subtract(&step.scale(self.rate));
        self.velocity = Some(velocity);
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
    fn update(&mut self, parameters: &mut P, gradient: &P) {
        let squared = gradient.multiply(gradient);
        let total = match self.total.take() {
            Some(previous) => previous.add(&squared),
            None => squared,
        };
        let step = gradient.divide(&total.sqrt().shift(self.epsilon));
        *parameters = parameters.subtract(&step.scale(self.rate));
        self.total = Some(total);
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
    fn update(&mut self, parameters: &mut P, gradient: &P) {
        let squared = gradient.multiply(gradient).scale(1.0 - self.decay);
        let mean_square = match self.mean_square.take() {
            Some(previous) => previous.scale(self.decay).add(&squared),
            None => squared,
        };
        let step = gradient.divide(&mean_square.sqrt().shift(self.epsilon));
        *parameters = parameters.subtract(&step.scale(self.rate));
        self.mean_square = Some(mean_square);
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
    fn update(&mut self, parameters: &mut P, gradient: &P) {
        self.steps += 1;

        let first = match self.first.take() {
            Some(previous) => previous
                .scale(self.first_decay)
                .add(&gradient.scale(1.0 - self.first_decay)),
            None => gradient.scale(1.0 - self.first_decay),
        };
        let squared = gradient.multiply(gradient);
        let second = match self.second.take() {
            Some(previous) => previous
                .scale(self.second_decay)
                .add(&squared.scale(1.0 - self.second_decay)),
            None => squared.scale(1.0 - self.second_decay),
        };

        // Both moments start at zero, so early estimates are biased toward it;
        // dividing by `1 − βᵗ` undoes exactly that.
        let first_correction = 1.0 - self.first_decay.powi(self.steps as i32);
        let second_correction = 1.0 - self.second_decay.powi(self.steps as i32);
        let corrected_first = first.scale(first_correction.recip());
        let corrected_second = second.scale(second_correction.recip());

        let step = corrected_first.divide(&corrected_second.sqrt().shift(self.epsilon));
        *parameters = parameters.subtract(&step.scale(self.rate));

        self.first = Some(first);
        self.second = Some(second);
    }

    fn reset(&mut self) {
        self.first = None;
        self.second = None;
        self.steps = 0;
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
